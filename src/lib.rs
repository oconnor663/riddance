//! Riddance provides the [`Registry`] container, which stores objects and issues unique IDs for
//! them, also known as a "slot map" or an "arena". Features include:
//!
//! - New IDs can be "reserved" atomically, without locking the [`Registry`]. See [`reserve_id`]
//!   and [`reserve_ids`].
//! - When the generation of a slot reaches its maximum, the slot is "retired" instead of allowing
//!   the generation to roll over to zero. This prevents logic errors from colliding IDs.
//! - The default [`Id`] type is 64 bits, but callers that need smallers IDs can use [`Id32`],
//!   which has a configurable number of generation bits.
//! - The [`recycle`] method makes it possible to reuse previously retired slots, though it can
//!   introduce logic errors if you violate its contract. It's mainly intended for callers who use
//!   [`Id32`].
//! - By default ID types incorporate the `T` type parameter of the `Registry` that created them,
//!   to avoid confusing IDs from different registries.
//!
//! # Example
//!
//! ```
//! # fn main() {
//! use riddance::{Id, Registry};
//!
//! struct Person {
//!     name: String,
//!     friends: Vec<Id<Person>>,
//! }
//!
//! let mut people = Registry::new();
//! let alice_id = people.insert(Person { name: "Alice".into(), friends: vec![] });
//! let bob_id = people.insert(Person { name: "Bob".into(), friends: vec![] });
//! people[alice_id].friends.push(bob_id);
//! people[bob_id].friends.push(alice_id);
//!
//! people.remove(bob_id);
//! assert!(people.get(alice_id).is_some());
//! assert!(people.get(bob_id).is_none());
//! # }
//! ```
//!
//! [`reserve_id`]: Registry::reserve_id
//! [`reserve_ids`]: Registry::reserve_ids
//! [`Id32`]: id::Id32
//! [`recycle`]: Registry::recycle

use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use typenum::Unsigned;

pub mod id;
pub mod id_map;
pub mod iter;

#[cfg(test)]
mod test;

use id::IdTrait;

// These "static" asserts will get compiled out in any case we care about. Someday we'll be able to
// actually enforce these bounds statically.
fn static_assert_index_bits<IndexBits: Unsigned>() {
    assert!(IndexBits::U32 >= 1);
    assert!(IndexBits::U32 <= 32);
}

fn static_assert_generation_bits<GenerationBits: Unsigned>() {
    assert!(GenerationBits::U32 <= 31);
}

// We do u32 -> usize casts all over the place.
fn static_assert_u32_fits_in_usize() {
    assert!(mem::size_of::<usize>() >= mem::size_of::<u32>());
}

fn debug_assert_high_state_bits_clear<GenerationBits: Unsigned>(state: u32) {
    static_assert_generation_bits::<GenerationBits>();
    if GenerationBits::U32 < 31 {
        debug_assert_eq!(
            0,
            state >> (GenerationBits::U32 + 1),
            "illegal high bits set in state",
        );
    }
}

fn generation_from_state<GenerationBits: Unsigned>(state: u32) -> u32 {
    debug_assert_high_state_bits_clear::<GenerationBits>(state);
    state & ((u32::MAX >> 1) >> (31 - GenerationBits::U32))
}

fn flag_bit_from_state<GenerationBits: Unsigned>(state: u32) -> bool {
    debug_assert_high_state_bits_clear::<GenerationBits>(state);
    (state >> GenerationBits::U32) > 0
}

fn state_is_occupied<GenerationBits: Unsigned>(state: u32) -> bool {
    !flag_bit_from_state::<GenerationBits>(state)
}

fn state_is_empty<GenerationBits: Unsigned>(state: u32) -> bool {
    flag_bit_from_state::<GenerationBits>(state)
}

fn retired_state<GenerationBits: Unsigned>() -> u32 {
    static_assert_generation_bits::<GenerationBits>();
    u32::MAX >> (31 - GenerationBits::U32)
}

fn state_is_retired<GenerationBits: Unsigned>(state: u32) -> bool {
    debug_assert_high_state_bits_clear::<GenerationBits>(state);
    state == retired_state::<GenerationBits>()
}

fn empty_state_from_occupied<GenerationBits: Unsigned>(occupied_state: u32) -> u32 {
    debug_assert!(state_is_occupied::<GenerationBits>(occupied_state));
    occupied_state | (1 << GenerationBits::U32)
}

fn occupied_state_from_empty<GenerationBits: Unsigned>(empty_state: u32) -> u32 {
    debug_assert!(state_is_empty::<GenerationBits>(empty_state));
    // Note that the retired state can roll over to generation zero after recycling. This means we
    // need to add 1 *before* we clear the flag bit.
    (empty_state + 1) & !(u32::MAX << GenerationBits::U32)
}

fn word_count_from_state_count<GenerationBits: Unsigned>(state_count: u32) -> u32 {
    // NOTE: The number of state bits is GenerationBits + 1.
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => state_count,
        8..=15 => state_count.div_ceil(2),
        4..=7 => state_count.div_ceil(4),
        2..=3 => state_count.div_ceil(8),
        1 => state_count.div_ceil(16),
        0 => state_count.div_ceil(32),
    }
}

fn state_count_from_word_count<GenerationBits: Unsigned>(word_count: u32) -> u32 {
    // NOTE: The number of state bits is GenerationBits + 1.
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => word_count,
        8..=15 => word_count.saturating_mul(2),
        4..=7 => word_count.saturating_mul(4),
        2..=3 => word_count.saturating_mul(8),
        1 => word_count.saturating_mul(16),
        0 => word_count.saturating_mul(32),
    }
}

const fn unused_states_in_last_word<GenerationBits: Unsigned>(state_count: u32) -> u32 {
    // NOTE: The number of state bits is GenerationBits + 1.
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => 0,
        8..=15 => 1 - (state_count.wrapping_sub(1) % 2),
        4..=7 => 3 - (state_count.wrapping_sub(1) % 4),
        2..=3 => 7 - (state_count.wrapping_sub(1) % 8),
        1 => 15 - (state_count.wrapping_sub(1) % 16),
        0 => 31 - (state_count.wrapping_sub(1) % 32),
    }
}

unsafe fn read_state<GenerationBits: Unsigned>(state_words: *const u32, index: u32) -> u32 {
    let i = index as usize;
    // NOTE: The number of state bits is GenerationBits + 1.
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => *state_words.add(i),
        8..=15 => *(state_words as *const u16).add(i) as u32,
        4..=7 => *(state_words as *const u8).add(i) as u32,
        2..=3 => (*(state_words as *const u8).add(i / 2) as u32 >> (4 * (i % 2))) & 0b1111,
        1 => (*(state_words as *const u8).add(i / 4) as u32 >> (2 * (i % 4))) & 0b11,
        0 => (*(state_words as *const u8).add(i / 8) as u32 >> (i % 8)) & 0b1,
    }
}

unsafe fn write_state<GenerationBits: Unsigned>(state_words: *mut u32, index: u32, state: u32) {
    debug_assert_high_state_bits_clear::<GenerationBits>(state);
    let i = index as usize;
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => *state_words.add(i as usize) = state,
        8..=15 => *(state_words as *mut u16).add(i) = state as u16,
        4..=7 => *(state_words as *mut u8).add(i) = state as u8,
        2..=3 => {
            let entry = &mut *(state_words as *mut u8).add(i / 2);
            *entry &= !(0b1111 << (4 * (i % 2)));
            *entry |= (state as u8 & 0b1111) << (4 * (i % 2));
        }
        1 => {
            let entry = &mut *(state_words as *mut u8).add(i / 4);
            *entry &= !(0b11 << (2 * (i % 4)));
            *entry |= (state as u8 & 0b11) << (2 * (i % 4));
        }
        0 => {
            let entry = &mut *(state_words as *mut u8).add(i / 8);
            *entry &= !(0b1 << (i % 8));
            *entry |= (state as u8 & 0b1) << (i % 8);
        }
    }
}

// This is similar to the private Unique<T> pointer wrapper from the standard library, which is
// used in the implementation of Vec. See https://doc.rust-lang.org/nomicon/phantom-data.html and
// https://github.com/rust-lang/rust/blob/1.74.0/library/core/src/ptr/unique.rs#L37-L45.
#[repr(transparent)]
struct Unique<T> {
    pointer: NonNull<T>,
    _phantom: PhantomData<T>,
}

impl<T> Unique<T> {
    fn from_vec(vec: &ManuallyDrop<Vec<T>>) -> Self {
        unsafe {
            Self {
                pointer: NonNull::new_unchecked(vec.as_ptr() as *mut T),
                _phantom: PhantomData,
            }
        }
    }

    fn as_ptr(&self) -> *mut T {
        self.pointer.as_ptr()
    }
}

unsafe impl<T: Send> Send for Unique<T> {}

unsafe impl<T: Sync> Sync for Unique<T> {}

struct Slots<T, GenerationBits: Unsigned> {
    // NOTES:
    // - If a state is 16 bits or less, multiple states get packed into each state word. See
    //   read_state and write_state.
    // - The number of bits in each state is GenerationBits + 1. The flag bit (highest order) is 1
    //   if the slot is free and 0 if it's occupied, *unless* the slot index is greater than or
    //   equal to self.len.
    //
    // INVARIANTS:
    // 1. The entire *capacity* of the state_words Vec is always zero-initialized. That means that
    //    when we increment self.len to allocate a new slot, the slot state is already "occupied,
    //    generation 0".
    // 2. Values are logically initialized if the flag bit in their state is 0 *and* their index is
    //    less than self.len.
    values_ptr: Unique<MaybeUninit<T>>,
    values_cap: usize,
    state_words_ptr: Unique<u32>,
    state_words_cap: usize,
    len: u32,
    _phantom: PhantomData<GenerationBits>,
}

impl<T, GenerationBits: Unsigned> Slots<T, GenerationBits> {
    fn with_capacity(capacity: u32) -> Self {
        // Don't allocate these Vecs directly into ManuallyDrop, because the second allocation
        // might panic, and we don't want to leak the first one in that case. Instead, move them
        // into ManuallyDrop only after both allocations have succeeded.
        let values = Vec::with_capacity(capacity as usize);
        let mut state_words =
            Vec::with_capacity(word_count_from_state_count::<GenerationBits>(capacity) as usize);
        // Zero-initialize all the capacity in state_words.
        unsafe {
            ptr::write_bytes(state_words.as_mut_ptr(), 0, state_words.capacity());
        }
        let values = ManuallyDrop::new(values);
        let state_words = ManuallyDrop::new(state_words);
        Self {
            values_cap: values.capacity(),
            values_ptr: Unique::from_vec(&values),
            state_words_cap: state_words.capacity(),
            state_words_ptr: Unique::from_vec(&state_words),
            len: 0,
            _phantom: PhantomData,
        }
    }

    pub fn capacity(&self) -> u32 {
        let state_words: u32 = self.state_words_cap.try_into().unwrap_or(u32::MAX);
        let states: u32 = state_count_from_word_count::<GenerationBits>(state_words);
        let values: u32 = self.values_cap.try_into().unwrap_or(u32::MAX);
        cmp::min(states, values)
    }

    unsafe fn state_unchecked(&self, index: u32) -> u32 {
        read_state::<GenerationBits>(self.state_words_ptr.as_ptr(), index)
    }

    fn state(&self, index: u32) -> Option<u32> {
        if index < self.len {
            Some(unsafe { self.state_unchecked(index) })
        } else {
            None
        }
    }

    unsafe fn set_state_unchecked(&mut self, index: u32, state: u32) {
        write_state::<GenerationBits>(self.state_words_ptr.as_ptr(), index, state)
    }

    unsafe fn value_unchecked(&self, index: u32) -> &MaybeUninit<T> {
        &*self.values_ptr.as_ptr().add(index as usize)
    }

    unsafe fn value_unchecked_mut(&mut self, index: u32) -> &mut MaybeUninit<T> {
        &mut *self.values_ptr.as_ptr().add(index as usize)
    }

    unsafe fn reconstitute_values_vec(&self) -> ManuallyDrop<Vec<MaybeUninit<T>>> {
        ManuallyDrop::new(Vec::from_raw_parts(
            self.values_ptr.as_ptr(),
            // These values aren't guaranteed to be initialized, but that's ok because their type
            // is MaybeUninit<T>.
            self.len as usize,
            self.values_cap,
        ))
    }

    // The length of this Vec is always equal to its capacity. All state words are zero-initialized
    // when they're allocated. See with_capacity() and reserve().
    unsafe fn reconstitute_state_words_vec(&self) -> ManuallyDrop<Vec<u32>> {
        ManuallyDrop::new(Vec::from_raw_parts(
            self.state_words_ptr.as_ptr(),
            word_count_from_state_count::<GenerationBits>(self.len) as usize,
            self.state_words_cap,
        ))
    }

    fn reserve(&mut self, additional: u32) {
        if self.len.checked_add(additional).is_none() {
            panic!("requested capacity exceeds u32::MAX");
        }
        // Account for unused state bits in the rightmost u32 in use.
        let last_word_cap = unused_states_in_last_word::<GenerationBits>(self.len);
        let additional_words =
            word_count_from_state_count::<GenerationBits>(additional.saturating_sub(last_word_cap));
        unsafe {
            let mut values = self.reconstitute_values_vec();
            let mut state_words = self.reconstitute_state_words_vec();
            // Either of these reserve calls could panic. We need to record any changes made by the
            // first call before we make the second call.
            values.reserve(additional as usize);
            self.values_cap = values.capacity();
            self.values_ptr = Unique::from_vec(&values);
            state_words.reserve(additional_words as usize);
            // Zero-initialize all the *new* capacity in state_words.
            ptr::write_bytes(
                state_words.as_mut_ptr().add(self.state_words_cap),
                0,
                state_words.capacity() - self.state_words_cap,
            );
            self.state_words_cap = state_words.capacity();
            self.state_words_ptr = Unique::from_vec(&state_words);
        }
    }
}

impl<T, GenerationBits: Unsigned> Drop for Slots<T, GenerationBits> {
    fn drop(&mut self) {
        unsafe {
            // These Vecs will drop at end-of-scope.
            let _states = ManuallyDrop::into_inner(self.reconstitute_state_words_vec());
            let _values = ManuallyDrop::into_inner(self.reconstitute_values_vec());
            // If dropping an element panics, we'll unwind out of this loop and skip dropping
            // subsequent elements. The two Vecs above will still drop during unwinding, but any
            // resources owned by individual elements (i.e. a String or a File) will be leaked.
            if mem::needs_drop::<T>() {
                for i in 0..self.len {
                    if state_is_occupied::<GenerationBits>(self.state_unchecked(i)) {
                        self.value_unchecked_mut(i).assume_init_drop();
                    }
                }
            }
        }
    }
}

impl<T, GenerationBits: Unsigned> Clone for Slots<T, GenerationBits>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        // Don't allocate these Vecs directly into ManuallyDrop, because the second allocation
        // might panic, and we don't want to leak the first one in that case. Instead, move them
        // into ManuallyDrop only after both allocations have succeeded.
        let new_values_vec = Vec::with_capacity(self.len as usize);
        let state_words_cap = word_count_from_state_count::<GenerationBits>(self.len) as usize;
        let mut new_state_words_vec = Vec::with_capacity(state_words_cap);
        // Memcpy all possibly-non-zero state words from self, and then zero-initialize any
        // additional capacity we received.
        unsafe {
            ptr::copy_nonoverlapping(
                self.state_words_ptr.as_ptr(),
                new_state_words_vec.as_mut_ptr(),
                state_words_cap,
            );
            ptr::write_bytes(
                new_state_words_vec.as_mut_ptr().add(state_words_cap),
                0,
                new_state_words_vec.capacity() - state_words_cap,
            );
        }
        // Assemble the new Slots. Some of the state flag bits will indicate occupied slots, but as
        // long as we only increment len when we actually insert a cloned element, the new Slots
        // will be safe to drop. This arrangement means we won't leak previously cloned elements if
        // one of the clones panics.
        // XXX: These partially-cloned Slots aren't safe to return, because they violates the
        // assumption that not-yet-used slot words are zero-initialized. However, Slots::drop
        // doesn't rely on that assumption.
        let new_values_vec = ManuallyDrop::new(new_values_vec);
        let new_state_words_vec = ManuallyDrop::new(new_state_words_vec);
        let mut cloned_slots = Slots {
            values_cap: new_values_vec.capacity(),
            values_ptr: Unique::from_vec(&new_values_vec),
            state_words_cap: new_state_words_vec.capacity(),
            state_words_ptr: Unique::from_vec(&new_state_words_vec),
            len: 0,
            _phantom: PhantomData,
        };
        // Clone and insert individual elements, keeping cloned_slots.len consistent with the
        // number of elements inserted.
        for i in 0..self.len {
            unsafe {
                if state_is_occupied::<GenerationBits>(self.state_unchecked(i)) {
                    // These clones could panic.
                    let cloned_element = self.value_unchecked(i).assume_init_ref().clone();
                    cloned_slots.value_unchecked_mut(i).write(cloned_element);
                    cloned_slots.len = i + 1;
                }
            }
        }
        // Finally, bump cloned_slots.len to equal the original (if it's not already) and return.
        // This last step restores the invariant we've so far been violating, that all slot words
        // at indexes greater than or equal to slots.len are zero-initialized.
        cloned_slots.len = self.len;
        cloned_slots
    }
}

impl<T, GenerationBits: Unsigned> fmt::Debug for Slots<T, GenerationBits>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        struct EmptySlot;

        impl fmt::Debug for EmptySlot {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
                write!(f, "<empty>")
            }
        }

        let mut list = f.debug_list();
        for i in 0..self.len {
            unsafe {
                if state_is_occupied::<GenerationBits>(self.state_unchecked(i)) {
                    list.entry(self.value_unchecked(i).assume_init_ref());
                } else {
                    list.entry(&EmptySlot);
                }
            }
        }
        list.finish()
    }
}

/// The default 64-bit ID type.
///
/// This ID type has 32 index bits and 31 generation bits (not 32, because the [`Registry`] needs an
/// extra bit to mark free slots). That's enough for 4 billion elements and a retirement rate of
/// one slot for every 2 billion removals. Most callers using this default ID type don't need to
/// worry about these limits and don't need to call [`recycle`].
///
/// Callers with very high performance requirements, for whom the difference between 64-bit IDs and
/// 32-bit IDs matters, can consider using [`Id32`] instead. That type uses a configurable number
/// of generation bits, and you have to think carefully about how many index bits you'll need and
/// whether you'll need to [`recycle`].
///
/// [`recycle`]: Registry::recycle
/// [`Id32`]: id::Id32
pub type Id<T> = id::Id64<T>;

/// A container that issues IDs and maps them to stored values, also called a "slot map" or an
/// "arena".
pub struct Registry<T, ID: IdTrait = Id<T>> {
    slots: Slots<T, ID::GenerationBits>,
    free_indexes: Vec<u32>,
    retired_indexes: Vec<u32>,
    reservation_cursor: AtomicU32,
}

impl<T> Registry<T, Id<T>> {
    /// Construct a new, empty `Registry<T>` with the default [`Id`] type.
    ///
    /// The registry will not allocate until elements are inserted into it.
    pub fn new() -> Self {
        Self::with_id_type()
    }

    /// Construct a new, empty `Registry<T>` with the default `Id` type and with at least the
    /// specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_id_type_and_capacity(capacity)
    }
}

impl<T, ID: IdTrait> Registry<T, ID> {
    /// Construct a new, empty `Registry<T>` with a custom ID type.
    ///
    /// The registry will not allocate until elements are inserted into it.
    ///
    /// # Example
    ///
    /// ```
    /// use riddance::{Id, Registry};
    ///
    /// type TypeErasedId = Id<()>;
    ///
    /// # fn main() {
    /// let mut registry: Registry::<String, TypeErasedId> = Registry::with_id_type();
    /// let id: TypeErasedId = registry.insert(String::from("foo"));
    /// # }
    /// ```
    pub fn with_id_type() -> Self {
        Self::with_id_type_and_capacity(0)
    }

    /// Construct a new, empty `Registry<T>` with a custom ID type and with at least the specified
    /// capacity.
    pub fn with_id_type_and_capacity(capacity: usize) -> Self {
        static_assert_index_bits::<ID::IndexBits>();
        static_assert_generation_bits::<ID::GenerationBits>();
        static_assert_u32_fits_in_usize();
        let capacity: u32 = capacity.try_into().expect("capacity overflow");
        Self {
            slots: Slots::with_capacity(capacity),
            free_indexes: Vec::new(),
            retired_indexes: Vec::new(),
            reservation_cursor: AtomicU32::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len as usize - self.free_indexes.len() - self.retired_indexes.len()
    }

    pub fn capacity(&self) -> usize {
        self.slots.capacity() as usize
    }

    // We currently check for two possible violations:
    // 1. The index of an ID should never be larger than the number of slots. The only ways to
    //    violate this rule are to assemble a garbage ID by hand or to use an ID from one Registry
    //    with another of the same time.
    // 2. The generation of an ID should never be newer than its slot. In addition to the cases
    //    above, you can also violate this rule by retaining a dangling ID across a call to
    //    recycle().
    fn debug_best_effort_checks_for_contract_violations(&self, id: ID) {
        if !cfg!(debug_assertions) {
            return;
        }
        if id.is_null() {
            return;
        }
        if id.index() >= self.slots.len {
            // This ID must be part of a pending reservation.
            let new_slots_reserved =
                self.reservation_cursor.load(Relaxed) - self.free_indexes.len() as u32;
            debug_assert!(id.index() - self.slots.len < new_slots_reserved);
            return;
        }
        let state = self.slots.state(id.index()).unwrap();
        let max_generation = if state_is_occupied::<ID::GenerationBits>(state) {
            state
        } else {
            // A reservation has a generation that's one higher than its slot.
            generation_from_state::<ID::GenerationBits>(state).saturating_add(1)
        };
        debug_assert!(
            id.generation() <= max_generation,
            "ID generation is newer than its slot; did it dangle across a recycle()?",
        );
    }

    pub fn contains_id(&self, id: ID) -> bool {
        self.debug_best_effort_checks_for_contract_violations(id);
        if let Some(state) = self.slots.state(id.index()) {
            // This comparison can only succeed if the generation matches and the flag bit is 0.
            state == id.generation()
        } else {
            false
        }
    }

    // Get a reference to an element. If [`remove`](Registry::remove) has been called on `id`,
    // `get` will return `None`.
    pub fn get(&self, id: ID) -> Option<&T> {
        if self.contains_id(id) {
            Some(unsafe { self.get_unchecked(id) })
        } else {
            None
        }
    }

    // Get a mutable reference to an element. If [`remove`](Registry::remove) has been called on
    // `id`, `get_mut` will return `None`.
    pub fn get_mut(&mut self, id: ID) -> Option<&mut T> {
        if self.contains_id(id) {
            Some(unsafe { self.get_unchecked_mut(id) })
        } else {
            None
        }
    }

    /// Get a reference to an element without checking the size of the Registry or the generation
    /// of the ID.
    ///
    /// This function is safe if and only if `self.contains_id(id)` is `true`.
    pub unsafe fn get_unchecked(&self, id: ID) -> &T {
        self.slots.value_unchecked(id.index()).assume_init_ref()
    }

    /// Get a mutable reference to an element without checking the size of the Registry or the
    /// generation of the ID.
    ///
    /// This function is safe if and only if `self.contains_id(id)` is `true`.
    pub unsafe fn get_unchecked_mut(&mut self, id: ID) -> &mut T {
        self.slots.value_unchecked_mut(id.index()).assume_init_mut()
    }

    // INVARIANT: The caller must check that self.free_indexes is not empty.
    unsafe fn insert_into_reused_slot(&mut self, new_value_fn: impl FnOnce(ID) -> T) -> ID {
        debug_assert!(self.free_indexes.len() > 0);
        // This pop decrements self.free_indexes.len().
        let index = self.free_indexes.pop().unwrap_unchecked();
        let empty_state = self.slots.state_unchecked(index);
        debug_assert!(state_is_empty::<ID::GenerationBits>(empty_state));
        let occupied_state = occupied_state_from_empty::<ID::GenerationBits>(empty_state);
        let new_id = ID::new_unchecked(index, occupied_state);
        // This call could panic, so do it before modifying any state.
        let value = new_value_fn(new_id);
        self.slots.set_state_unchecked(index, occupied_state);
        self.slots.value_unchecked_mut(index).write(value);
        ID::new_unchecked(index, occupied_state)
    }

    // INVARIANT: The caller must reserve space.
    unsafe fn insert_into_new_slot(&mut self, new_value_fn: impl FnOnce(ID) -> T) -> ID {
        debug_assert!((self.slots.len as usize) < self.capacity());
        debug_assert!(self.slots.len < ID::max_len());
        let index = self.slots.len;
        let new_id = ID::new_unchecked(index, 0);
        // This call could panic, so do it before modifying any state.
        let value = new_value_fn(new_id);
        // New state capacity is zero-initialized, so we only need to write the value here.
        debug_assert_eq!(self.slots.state_unchecked(index), 0);
        self.slots.value_unchecked_mut(index).write(value);
        self.slots.len += 1;
        ID::new_unchecked(index, 0)
    }

    #[must_use]
    pub fn insert(&mut self, value: T) -> ID {
        assert_eq!(*self.reservation_cursor.get_mut(), 0, "pending reservation");
        // Reuse a free slot if there are any.
        if !self.free_indexes.is_empty() {
            return unsafe { self.insert_into_reused_slot(|_| value) };
        }
        // Panic if the index space is full.
        assert!(self.slots.len < ID::max_len(), "all slots occupied");
        self.slots.reserve(1);
        unsafe { self.insert_into_new_slot(|_| value) }
    }

    /// If `id` refers to an element in the registry, remove the element and return it.
    ///
    /// This method returns `Some` if any only if [`contains_id`] would have returned `true`. After
    /// calling `remove` on an ID, that ID and any copies of it become "dangling". [`contains_id`]
    /// will return `false`, and [`get`], [`get_mut`], and any further calls to `remove` will
    /// return `None`.
    ///
    /// See also [`recycle`].
    ///
    /// [`contains_id`]: Registry::contains_id
    /// [`get`]: Registry::get
    /// [`get_mut`]: Registry::get_mut
    /// [`recycle`]: Registry::recycle
    pub fn remove(&mut self, id: ID) -> Option<T> {
        assert_eq!(*self.reservation_cursor.get_mut(), 0, "pending reservation");
        self.debug_best_effort_checks_for_contract_violations(id);
        let Some(state) = self.slots.state(id.index()) else {
            return None;
        };
        if state != id.generation() {
            return None;
        }
        // The ID generation matches the slot state, which means the slot is occupied. Add this
        // slot to the free list or the retired list (which could panic), set the free bit in its
        // state, and move out its value.
        let empty_state = empty_state_from_occupied::<ID::GenerationBits>(state);
        if state_is_retired::<ID::GenerationBits>(empty_state) {
            self.retired_indexes.push(id.index());
        } else {
            self.free_indexes.push(id.index());
        }
        unsafe {
            self.slots.set_state_unchecked(id.index(), empty_state);
            Some(self.slots.value_unchecked(id.index()).assume_init_read())
        }
    }

    /// Reserve an ID that doesn't exist yet. You **must** allocate pending reservations before
    /// most other operations, see below.
    ///
    /// Note that this method doesn't require mutable access to the `Registry`. It uses atomics
    /// internally, and for example you can reserve an ID while other threads are reading existing
    /// elements.
    ///
    /// The new reservation is "pending", and [`contains_id`] will report `false` for the reserved
    /// ID. Similarly, [`get`] and [`get_mut`] will return `None`. After making any number of
    /// pending reservations, you **must** fill them using one of the following methods, which
    /// require mutable access to the registry:
    ///
    /// - [`fill_pending_reservations`]
    /// - [`fill_pending_reservations_with`]
    /// - [`fill_pending_reservations_with_id`]
    ///
    /// The following methods will panic if there are any pending reservations:
    ///
    /// - [`insert`]
    /// - [`remove`]
    /// - [`recycle`]
    /// - [`clone`]
    ///
    /// See also [`reserve_ids`].
    ///
    /// [`contains_id`]: Registry::contains_id
    /// [`get`]: Registry::get
    /// [`get_mut`]: Registry::get_mut
    /// [`fill_pending_reservations`]: Registry::fill_pending_reservations
    /// [`fill_pending_reservations_with`]: Registry::fill_pending_reservations_with
    /// [`fill_pending_reservations_with_id`]: Registry::fill_pending_reservations_with_id
    /// [`allocate_empty_reservations`]: Registry::allocate_empty_reservations
    /// [`insert`]: Registry::insert
    /// [`remove`]: Registry::remove
    /// [`recycle`]: Registry::recycle
    /// [`clone`]: Registry::clone
    /// [`reserve_ids`]: Registry::reserve_ids
    #[must_use]
    pub fn reserve_id(&self) -> ID {
        self.reserve_ids(1).next().unwrap()
    }

    /// Reserve a range of IDs that don't exist yet. You **must** allocate reservations before
    /// doing any other mutations, see [`reserve_id`].
    ///
    /// Note that unconsumed IDs in this iterator are _not_ returned to the `Registry` when you
    /// drop it, and the slots they refer to are effectively leaked. If you reserved more IDs than
    /// you need, you can save them for later, or you can [`remove`] them when you have mutable
    /// access to the `Registry`.
    ///
    /// [`reserve_id`]: Registry::reserve_id
    /// [`remove`]: Registry::remove
    #[must_use]
    pub fn reserve_ids(&self, count: usize) -> iter::ReservationIter<'_, T, ID> {
        let count: u32 = count.try_into().expect("capacity overflow");
        // Take the reservation with compare-exchange instead of a fetch-add, so that we can check
        // for overflow.
        let mut start = self.reservation_cursor.load(Relaxed);
        let mut end;
        loop {
            // Make sure this reservation wouldn't overflow the reservation cursor.
            end = start.checked_add(count).expect("capacity overflow");
            // Make sure this reservation wouldn't overflow self.len.
            let new_slots = end.saturating_sub(self.free_indexes.len() as u32);
            self.slots
                .len
                .checked_add(new_slots)
                .expect("capacity overflow");
            // Make sure this reservation wouldn't exceed the available ID bits.
            assert!(
                self.slots.len + new_slots <= ID::max_len(),
                "not enough index bits",
            );
            // Try to commit the reservation. Since we need to loop here anyway, we use the weak
            // version of compare-exchange.
            let result = self
                .reservation_cursor
                .compare_exchange_weak(start, end, Relaxed, Relaxed);
            match result {
                // success
                Ok(_) => {
                    return iter::ReservationIter {
                        registry: self,
                        start,
                        end,
                    };
                }
                // failure, continue the loop
                Err(new_start) => start = new_start,
            }
        }
    }

    pub fn fill_pending_reservations(&mut self, value: T)
    where
        T: Copy,
    {
        self.fill_pending_reservations_with(|| value);
    }

    pub fn fill_pending_reservations_with(&mut self, mut new_value_fn: impl FnMut() -> T) {
        self.fill_pending_reservations_with_id(|_| new_value_fn());
    }

    pub fn fill_pending_reservations_with_id(&mut self, mut new_value_fn: impl FnMut(ID) -> T) {
        let reservations = *self.reservation_cursor.get_mut();
        let reused_slots = cmp::min(reservations as usize, self.free_indexes.len());
        let new_slots = reservations - reused_slots as u32;
        // We check for overflow in reserve_ids().
        let new_len = self.slots.len + new_slots;

        // Pre-allocate any new slots.
        self.slots.reserve(new_slots);

        // Reuse free slots.
        for _ in 0..reused_slots {
            unsafe {
                self.insert_into_reused_slot(&mut new_value_fn);
            }
            *self.reservation_cursor.get_mut() -= 1;
        }
        debug_assert_eq!(*self.reservation_cursor.get_mut(), new_slots);

        // Populate any new slots we allocated above. Their states are already zero.
        for _ in 0..new_slots {
            unsafe {
                self.insert_into_new_slot(&mut new_value_fn);
            }
            *self.reservation_cursor.get_mut() -= 1;
        }
        debug_assert_eq!(*self.reservation_cursor.get_mut(), 0);
        debug_assert_eq!(new_len, self.slots.len);
    }

    /// Mark all retired slots as free, making them available for future insertions and
    /// reservations. Dangling IDs (that is, IDs passed to [`remove`]) from before the call to
    /// `recycle` **must not** be used again with any method on this `Registry`. Generally callers
    /// should delete all dangling IDs (or replace them with [`null`]) before calling `recycle`.
    ///
    /// If you retain dangling IDs across a call to `recycle`, they can collide with newly issued
    /// IDs, and calls to [`get`], [`get_mut`], and [`contains_id`] can return confusing results.
    /// This behavior is memory-safe, but these are logic bugs, similar to the logic bugs that can
    /// arise if you modify a key after it's been inserted into a [`HashMap`].
    ///
    /// # Panics
    ///
    /// `Registry` makes a best effort to detect violations of this rule. **Any** method on
    /// `Registry` may panic if it sees an ID generation that's newer than the corresponding slot.
    /// These checks are currently only done in debug mode, but this is not guaranteed.
    ///
    /// [`null`]: IdTrait::null
    /// [`remove`]: Registry::remove
    /// [`get`]: Registry::get
    /// [`get_mut`]: Registry::get_mut
    /// [`contains_id`]: Registry::contains_id
    /// [`HashMap`]: https://doc.rust-lang.org/std/collections/struct.HashMap.html
    pub fn recycle(&mut self) {
        assert_eq!(*self.reservation_cursor.get_mut(), 0, "pending reservation");
        // This clears retired_indexes.
        self.free_indexes.append(&mut self.retired_indexes);
    }

    /// Iterate over `(ID, &T)`. Equivalent to iterating over `&Registry`.
    pub fn iter(&self) -> iter::Iter<'_, T, ID> {
        iter::Iter {
            registry: self,
            index: 0,
        }
    }

    /// Iterate over `(ID, &mut T)`. Equivalent to iterating over `&mut Registry`.
    pub fn iter_mut(&mut self) -> iter::IterMut<'_, T, ID> {
        iter::IterMut {
            registry: self,
            index: 0,
        }
    }

    /// Iterate over `(ID, T)`. Equivalent to iterating over `Registry`.
    pub fn into_iter(self) -> iter::IntoIter<T, ID> {
        iter::IntoIter {
            registry: self,
            index: 0,
        }
    }

    /// Iterate over `ID`.
    pub fn ids(&self) -> iter::Ids<'_, T, ID> {
        iter::Ids { inner: self.iter() }
    }

    /// Iterate over `&T`.
    pub fn values(&self) -> iter::Values<'_, T, ID> {
        iter::Values { inner: self.iter() }
    }

    /// Iterate over `&mut T`.
    pub fn values_mut(&mut self) -> iter::ValuesMut<'_, T, ID> {
        iter::ValuesMut {
            inner: self.iter_mut(),
        }
    }

    /// Iterate over `T`.
    pub fn into_values(self) -> iter::IntoValues<T, ID> {
        iter::IntoValues {
            inner: self.into_iter(),
        }
    }
}

impl<T, ID: IdTrait> Clone for Registry<T, ID>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let cloned = Self {
            slots: self.slots.clone(),
            free_indexes: self.free_indexes.clone(),
            retired_indexes: self.retired_indexes.clone(),
            reservation_cursor: AtomicU32::new(0),
        };
        // Reservations are atomic, so one thread taking a reservation can race against another
        // thread cloning. This isn't a data race, and it isn't UB, but it's almost certainly a
        // bug, so we panic if we detect it.
        if self.reservation_cursor.load(Relaxed) != 0 {
            panic!("can't clone a Registry with pending reservations");
        }
        cloned
    }
}

impl<T, ID: IdTrait> std::ops::Index<ID> for Registry<T, ID> {
    type Output = T;

    fn index(&self, id: ID) -> &T {
        self.get(id).unwrap()
    }
}

impl<T, ID: IdTrait> std::ops::IndexMut<ID> for Registry<T, ID> {
    fn index_mut(&mut self, id: ID) -> &mut T {
        self.get_mut(id).unwrap()
    }
}

// TODO: Figure out how to derive this. Currently typenum bounds get in the way.
impl<T, ID: IdTrait> fmt::Debug for Registry<T, ID>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("Registry")
            .field("slots", &self.slots)
            .field("free_indexes", &self.free_indexes)
            .field("retired_indexes", &self.retired_indexes)
            .field("reservation_cursor", &self.reservation_cursor)
            .finish()
    }
}
