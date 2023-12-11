use std::cmp;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use typenum::Unsigned;

use id::IdTrait;

pub mod id;

#[cfg(test)]
mod test;

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

/// the default 64-bit ID type
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

#[derive(Debug)]
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

    pub fn insert(&mut self, value: T) -> ID {
        assert_eq!(*self.reservation_cursor.get_mut(), 0, "pending reservation");
        // Reuse a free slot if there are any.
        if let Some(index) = self.free_indexes.pop() {
            unsafe {
                let empty_state = self.slots.state_unchecked(index);
                // Note that if this slot was previously retired and has been recycled, the new
                // generation will wrap back to 0.
                let occupied_state = occupied_state_from_empty::<ID::GenerationBits>(empty_state);
                self.slots.set_state_unchecked(index, occupied_state);
                self.slots.value_unchecked_mut(index).write(value);
                // The flag bit is zero for the occupied state, so its value is equal to the
                // generation of the new ID.
                return ID::new_unchecked(index, occupied_state);
            }
        }
        // Panic if the index space is full.
        assert!(self.slots.len < ID::max_len(), "all slots occupied");
        // Reserve a slot. New state capacity is zero-initialized, so we only need to initialize
        // the value here.
        self.slots.reserve(1);
        unsafe {
            self.slots.value_unchecked_mut(self.slots.len).write(value);
            let new_id = ID::new_unchecked(self.slots.len, 0);
            self.slots.len += 1;
            new_id
        }
    }

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
    /// pending reservations, you **must** allocate them. There are two ways to allocate
    /// reservations, both of which require mutable access to the registry. First, you can fill
    /// them with values using one of the following methods, after which [`contains_id`] will
    /// report `true`:
    ///
    /// - [`fill_pending_reservations`]
    /// - [`fill_pending_reservations_with`]
    /// - [`fill_pending_reservations_with_id`]
    ///
    /// Alternatively, you can allocate (or potentially reuse) empty reserved slots by calling
    /// [`allocate_empty_reservations`]. In that case [`contains_id`] still reports `false` for
    /// each reserved ID until you call [`fill_empty_reservation`] on it, which you can do at any
    /// time. You can also [`remove`] an empty reservation without filling it, in which case
    /// further attempts to fill it will return an error. (But note that you can't [`remove`] a
    /// _pending_ reservation. See immediately below.)
    ///
    /// The following methods will panic if there are any pending reservations:
    ///
    /// - [`insert`]
    /// - [`remove`]
    /// - [`recycle`]
    /// - [`clone`]
    /// - [`fill_empty_reservation`]
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
    /// [`fill_empty_reservation`]: Registry::fill_empty_reservation
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
    pub fn reserve_ids(&self, count: usize) -> ReservationIter<'_, T, ID> {
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
                    return ReservationIter {
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
        let cursor = *self.reservation_cursor.get_mut() as usize;
        let reused_slots = cmp::min(cursor, self.free_indexes.len());
        let reused_slots_start = self.free_indexes.len() - reused_slots as usize;
        let new_slots = (cursor - reused_slots) as u32;
        // We check for overflow in reserve_ids().
        let new_len = self.slots.len + new_slots;

        // Pre-allocate any new slots. This is the only step that can fail.
        self.slots.reserve(new_slots);

        // Reuse free slots.
        for i in (reused_slots_start..self.free_indexes.len()).rev() {
            unsafe {
                let free_index = *self.free_indexes.get_unchecked(i);
                let state = self.slots.state_unchecked(free_index);
                let new_state = occupied_state_from_empty::<ID::GenerationBits>(state);
                self.slots.value_unchecked_mut(free_index).write(value);
                self.slots.set_state_unchecked(free_index, new_state);
            }
        }
        self.free_indexes.truncate(reused_slots_start);

        // Populate any new slots we allocated above. Their states are already zero.
        for i in 0..new_slots {
            unsafe {
                self.slots
                    .value_unchecked_mut(self.slots.len + i)
                    .write(value);
                debug_assert_eq!(self.slots.state_unchecked(self.slots.len), 0);
            }
        }

        self.slots.len = new_len;
        *self.reservation_cursor.get_mut() = 0;
    }

    pub fn fill_pending_reservations_with<F>(&mut self, mut new_value_fn: F)
    where
        F: FnMut() -> T,
    {
        self.fill_pending_reservations_with_id(|_| new_value_fn());
    }

    pub fn fill_pending_reservations_with_id<F>(&mut self, mut new_value_fn: F)
    where
        F: FnMut(ID) -> T,
    {
        let cursor: &mut u32 = self.reservation_cursor.get_mut();
        let reused_slots = cmp::min(*cursor as usize, self.free_indexes.len());
        let reused_slots_start = self.free_indexes.len() - reused_slots;
        let new_slots = *cursor - reused_slots as u32;
        // We check for overflow in reserve_ids().
        let new_len = self.slots.len + new_slots;

        // Pre-allocate any new slots.
        self.slots.reserve(new_slots);

        // Reuse free slots.
        for i in (reused_slots_start..self.free_indexes.len()).rev() {
            unsafe {
                let free_index = *self.free_indexes.get_unchecked(i);
                let state = self.slots.state_unchecked(free_index);
                let new_state = occupied_state_from_empty::<ID::GenerationBits>(state);
                let new_id = ID::new_unchecked(free_index, new_state);
                // This could panic. Do it before other writes.
                self.slots
                    .value_unchecked_mut(free_index)
                    .write(new_value_fn(new_id));
                self.slots.set_state_unchecked(free_index, new_state);
            }
            self.free_indexes.pop();
            *cursor -= 1;
        }
        debug_assert_eq!(*cursor, new_slots);

        // Populate any new slots we allocated above. Their states are already zero.
        for _ in 0..new_slots {
            unsafe {
                let new_id = ID::new_unchecked(self.slots.len, 0);
                // This could panic. Do it before other writes.
                self.slots
                    .value_unchecked_mut(self.slots.len)
                    .write(new_value_fn(new_id));
                debug_assert_eq!(self.slots.state_unchecked(self.slots.len), 0);
            }
            self.slots.len += 1;
            *cursor -= 1;
        }
        debug_assert_eq!(*cursor, 0);
        debug_assert_eq!(new_len, self.slots.len);
    }

    pub fn allocate_empty_reservations(&mut self) {
        let cursor: &mut u32 = self.reservation_cursor.get_mut();
        let reused_slots = cmp::min(*cursor as usize, self.free_indexes.len());
        let reused_slots_start = self.free_indexes.len() - reused_slots;
        let new_slots = *cursor - reused_slots as u32;

        // Pre-allocate any new slots.
        self.slots.reserve(new_slots);

        // Reuse free slots. We don't need to modify their states at all, just remove them from the
        // free indexes list.
        self.free_indexes.truncate(reused_slots_start);

        // Newly allocated slots have state zero, which isn't correct for an empty reservation.
        // They need to have the retired state, so that filling them wraps the generation to zero.
        let retired_state = empty_state_from_occupied::<ID::GenerationBits>(ID::max_generation());
        debug_assert!(state_is_retired::<ID::GenerationBits>(retired_state));
        for i in 0..new_slots {
            unsafe {
                // TODO: Optimize this to memset whole u32 words.
                self.slots
                    .set_state_unchecked(self.slots.len + i, retired_state);
            }
        }
        self.slots.len += new_slots;
        *cursor = 0;
    }

    /// Provide a value for one empty reservation. This is used together with
    /// [`allocate_empty_reservations`].
    ///
    /// Empty reservations can be filled in any order. You can also [`remove`] a reserved ID
    /// without ever filling it. But note that you can't [`remove`] a _pending_ reservation,
    /// because you can't call [`remove`] at all while there are pending reservations. See
    /// [`reserve_id`]. If you try to fill a reservation multiple times, or if you call this method
    /// with IDs that aren't reserved, it will return an error.
    ///
    /// [`allocate_empty_reservations`]: Registry::allocate_empty_reservations
    /// [`remove`]: Registry::remove
    /// [`reserve_id`]: Registry::reserve_id
    pub fn fill_empty_reservation(
        &mut self,
        id: ID,
        value: T,
    ) -> Result<(), FillEmptyReservationError<T>> {
        assert_eq!(*self.reservation_cursor.get_mut(), 0, "pending reservation");
        self.debug_best_effort_checks_for_contract_violations(id);
        let error_kind;
        if let Some(state) = self.slots.state(id.index()) {
            let expected_generation = if id.generation() == 0 {
                ID::max_generation()
            } else {
                id.generation() - 1
            };
            let reserved_state =
                empty_state_from_occupied::<ID::GenerationBits>(expected_generation);
            if state == reserved_state {
                // Happy path: the reservation is valid.
                unsafe {
                    self.slots.set_state_unchecked(id.index(), id.generation());
                    self.slots.value_unchecked_mut(id.index()).write(value);
                }
                return Ok(());
            }
            let state_generation = generation_from_state::<ID::GenerationBits>(state);
            if state == id.generation() {
                error_kind = FillEmptyReservationErrorKind::Exists;
            } else if state_generation >= id.generation() {
                error_kind = FillEmptyReservationErrorKind::Dangling;
            } else {
                error_kind = FillEmptyReservationErrorKind::GenerationTooNew;
            }
        } else {
            error_kind = FillEmptyReservationErrorKind::IndexOutOfBounds;
        }
        Err(FillEmptyReservationError {
            inner: value,
            kind: error_kind,
        })
    }

    /// Mark all retired slots as free. You **must** delete all dangling IDs (or replace them with
    /// [`null`]) before calling this function.
    ///
    /// When you call [`remove`] Registry::remove on an ID, that ID and any copies of it become
    /// "dangling". Calling [`get`] or [`get_mut`] on a dangling ID is normally guaranteed to
    /// return `None`, and [`contains_id`] is normally guaranteed to return `false`. To provide
    /// these guarantees, `Registry` "retires" a slot when its generation reaches the maximum.
    ///
    /// When you call `recycle`, all of these retired slots are made available for new insertions,
    /// and their generation starts back over at 0. If you retain any dangling IDs across the call
    /// to `retire`, they could collide with newly issued IDs, and calls to [`get`], [`get_mut`],
    /// and [`contains_id`] can return confusing results. This behavior is memory-safe, but it's a
    /// logic error, similar to the logic errors that can arise if you modify a key after it's been
    /// inserted into a [`HashMap`].
    ///
    /// # Panics
    ///
    /// `Registry` makes a best effort to detect violations of this rule. _Any_ method on
    /// `Registry` may panic if it sees an ID generation that's newer than the corresponding slot.
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

pub struct ReservationIter<'registry, T, ID: IdTrait = Id<T>> {
    registry: &'registry Registry<T, ID>,
    // Note that these bounds are positions in (or beyond) the free list, not slot indexes.
    start: u32,
    end: u32,
}

impl<'registry, T, ID: IdTrait> Iterator for ReservationIter<'registry, T, ID> {
    type Item = ID;

    fn next(&mut self) -> Option<ID> {
        if self.start < self.end {
            let id = unsafe {
                if (self.start as usize) < self.registry.free_indexes.len() {
                    let index = *self
                        .registry
                        .free_indexes
                        .get_unchecked(self.start as usize);
                    let state = self.registry.slots.state_unchecked(self.start);
                    let generation = occupied_state_from_empty::<ID::GenerationBits>(state);
                    ID::new_unchecked(index, generation)
                } else {
                    let index = self.registry.slots.len
                        + (self.start - self.registry.free_indexes.len() as u32);
                    // reserve_ids checked that these indexes won't exceed ID::max_len.
                    ID::new_unchecked(index, 0)
                }
            };
            self.start += 1;
            Some(id)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FillEmptyReservationErrorKind {
    Exists,
    Dangling,
    GenerationTooNew,
    IndexOutOfBounds,
}

#[derive(Copy, Clone)]
pub struct FillEmptyReservationError<T> {
    kind: FillEmptyReservationErrorKind,
    inner: T,
}

impl<T> FillEmptyReservationError<T> {
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> fmt::Debug for FillEmptyReservationError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FillEmptyReservationError")
            .field("kind", &self.kind)
            .finish()
    }
}

impl<T> fmt::Display for FillEmptyReservationError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let message = match self.kind {
            FillEmptyReservationErrorKind::Exists => "entry with this ID already exists",
            FillEmptyReservationErrorKind::Dangling => "this ID has been removed",
            FillEmptyReservationErrorKind::GenerationTooNew => {
                "ID generation too new (dangling ID retained across recycle?)"
            }
            FillEmptyReservationErrorKind::IndexOutOfBounds => "ID index out of bounds",
        };
        write!(f, "{}", message)
    }
}

impl<T> error::Error for FillEmptyReservationError<T> {}
