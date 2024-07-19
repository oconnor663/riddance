//! Riddance provides the [`Registry`] container, which stores objects and issues unique IDs for
//! them, also known as a "slot map" or an "arena". Features include:
//!
//! - New IDs can be "reserved" atomically, without locking the [`Registry`]. See [`reserve_id`]
//!   and [`reserve_ids`].
//! - When the generation of a slot reaches its maximum, the slot is "retired" instead of allowing
//!   the generation to roll over to zero. This prevents logic errors from colliding IDs.
//! - The default [`Id`] type is 64 bits, but callers that need smallers IDs can use [`Id32`],
//!   which has a configurable number of generation bits.
//! - The [`recycle_retired`] method makes it possible to reuse previously retired slots, though it
//!   can introduce logic errors if you violate its contract. It's mainly intended for callers who
//!   use [`Id32`].
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
//! [`recycle_retired`]: Registry::recycle_retired

use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use typenum::Unsigned;

pub mod error;
pub mod id;
pub mod iter;
pub mod map;
pub mod state;

#[cfg(test)]
mod test;

use id::IdTrait;
use state::State;

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

struct Slots<T, GenerationBits: Unsigned> {
    // NOTES:
    // - If a state is 16 bits or less, multiple states get packed into each state word. See
    //   read_state and write_state.
    // - The number of bits in each state is GenerationBits + 1.
    // - Values are logically initialized if the their state is odd.
    // - INVARIANT: state_words.len() is exactly the minimum needed to hold values.len() *states*.
    //   (However it might contain fewer *words*, if GenerationBits < 16.) To maintain this
    //   invariant, we always allocate capacity in both vectors before pushing into either, in case
    //   one of the allocations panics.
    values: Vec<MaybeUninit<T>>,
    state_words: Vec<u32>,
    _phantom: PhantomData<GenerationBits>,
}

impl<T, GenerationBits: Unsigned> Slots<T, GenerationBits> {
    /// In general there's never any reason to allocate more than 2^32 elements of capacity,
    /// because we never store more than 32 index bits, and insertions beyond that will panic.
    /// However, we need to store capacities internally as usize, because the allocator is allowed
    /// to give us more capacity than request, and we need to preserve that capacity when we
    /// reconstitute Vecs. Since the external capacity APIs (with_capacity and reserve) use usize,
    /// and the internal representation is usize, it's cleaner to use usize capacity throughout.
    fn with_capacity(capacity: usize) -> Self {
        let state_words_capacity = state::word_count_from_state_count::<GenerationBits>(capacity);
        Self {
            values: Vec::with_capacity(capacity),
            state_words: Vec::with_capacity(state_words_capacity),
            _phantom: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn debug_assert_len_invariant(&self) {
        // state_words.len() should be exactly the length needed to represent values.len() states.
        // But we don't impose any requirements on the capacity, both because Vec is allowed to do
        // whatever it wants with capacity over-allocation, and also because allocating more
        // capacity in both Vecs isn't atomic.
        debug_assert_eq!(
            state::word_count_from_state_count::<GenerationBits>(self.values.len()),
            self.state_words.len(),
        );
    }

    unsafe fn state_unchecked(&self, index: usize) -> State<GenerationBits> {
        state::read_state::<GenerationBits>(self.state_words.as_ptr(), index)
    }

    fn state(&self, index: usize) -> Option<State<GenerationBits>> {
        if index < self.len() {
            Some(unsafe { self.state_unchecked(index) })
        } else {
            None
        }
    }

    unsafe fn set_state_unchecked(&mut self, index: usize, state: State<GenerationBits>) {
        state::write_state::<GenerationBits>(self.state_words.as_mut_ptr(), index, state)
    }

    unsafe fn value_unchecked(&self, index: usize) -> &MaybeUninit<T> {
        self.values.get_unchecked(index)
    }

    unsafe fn value_unchecked_mut(&mut self, index: usize) -> &mut MaybeUninit<T> {
        self.values.get_unchecked_mut(index)
    }

    fn insert(&mut self, value: T) {
        self.debug_assert_len_invariant();
        self.values.reserve(1);
        if self.values.len() % state::states_per_word::<GenerationBits>() == 0 {
            self.state_words.reserve(1);
            // At this point all reservations have succeeded. The rest of this function can't fail.
            self.state_words.push(0);
        }
        unsafe {
            self.set_state_unchecked(
                self.values.len(),
                State::new(1), // generation 0, occupied flag set
            );
        }
        self.values.push(MaybeUninit::new(value));
        self.debug_assert_len_invariant();
    }

    fn allocate_empty_slots(&mut self, additional: usize) {
        self.debug_assert_len_invariant();
        self.values.reserve(additional);
        let unused_initialized_states =
            state::unused_states_in_last_word::<GenerationBits>(self.values.len());
        let new_states = additional.saturating_sub(unused_initialized_states);
        let new_state_words = state::word_count_from_state_count::<GenerationBits>(new_states);
        self.state_words.reserve(new_state_words);
        // At this point all reservations have succeeded. The rest of this function can't fail.
        unsafe {
            // SAFETY: MaybeUninit<T> does not need to be initialized, and setting the new state
            // words to zero marks the slots empty. The call to reserve() would have panicked if
            // the capacity overflowed (in fact, if it exceeded isize::MAX).
            debug_assert!(self.values.len() <= self.values.capacity() - additional);
            self.values.set_len(self.values.len() + additional);
        }
        self.state_words.resize(
            state::word_count_from_state_count::<GenerationBits>(self.values.len()),
            0,
        );
        self.debug_assert_len_invariant();
    }
}

impl<T, GenerationBits: Unsigned> Drop for Slots<T, GenerationBits> {
    fn drop(&mut self) {
        // If a new Slots is dropped partway through a clone(), the normal len invariant might not
        // hold. (self.state_words might be longer than necessary.) That's fine.
        if mem::needs_drop::<T>() {
            for i in 0..self.values.len() {
                unsafe {
                    if self.state_unchecked(i).is_occupied() {
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
        // If either of these allocations fails, cloned_slots won't be constructed or dropped.
        let mut cloned_slots = Self {
            values: Vec::with_capacity(self.values.len()),
            state_words: self.state_words.clone(),
            _phantom: PhantomData,
        };
        // Clone elements from self one-by-one into cloned_slots. Increment the length each time
        // through the loop, so that if any of these clones panics, we won't try to drop elements
        // we haven't initialized yet. This would technically violate the normal len invariant
        // (self.state_words would be longer than necessary), but the Drop impl above is ok with
        // that.
        for i in 0..self.values.len() {
            unsafe {
                if self.state_unchecked(i).is_occupied() {
                    let cloned_element = self.value_unchecked(i).assume_init_ref().clone();
                    // NB: Updating the length before writing the value is required. There are
                    // debug asserts in libstd that fire if we do this backwards.
                    cloned_slots.values.set_len(i + 1);
                    cloned_slots.value_unchecked_mut(i).write(cloned_element);
                }
            }
        }
        unsafe {
            cloned_slots.values.set_len(self.values.len());
        }
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
        for i in 0..self.values.len() {
            unsafe {
                if self.state_unchecked(i).is_occupied() {
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
/// This ID type has 32 index bits and 31 generation bits (not 32, because the [`Registry`] needs
/// an extra bit to mark free slots). That's enough for 4 billion elements and a retirement rate of
/// one slot for every 2 billion removals. Most callers using this default ID type don't need to
/// worry about these limits and don't need to call [`recycle_retired`].
///
/// Callers with very high performance requirements, for whom the difference between 64-bit IDs and
/// 32-bit IDs matters, can consider using [`Id32`] instead. That type uses a configurable number
/// of generation bits, and you have to think carefully about how many index bits you'll need and
/// whether you'll need to [`recycle_retired`].
///
/// [`recycle_retired`]: Registry::recycle_retired
/// [`Id32`]: id::Id32
pub type Id<T> = id::Id64<T>;

/// A container that issues IDs and maps them to stored values, also called a "slot map" or an
/// "arena".
pub struct Registry<T, ID: IdTrait = Id<T>> {
    slots: Slots<T, ID::GenerationBits>,
    free_indexes: Vec<usize>,
    retired_indexes: Vec<usize>,
    reservation_cursor: AtomicU64,
}

impl<T> Registry<T, Id<T>> {
    /// Construct a new, empty `Registry<T>` with the default [`Id`] type.
    ///
    /// The `Registry` will not allocate until elements are inserted into it.
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
    /// The `Registry` will not allocate until elements are inserted into it.
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
        Self {
            slots: Slots::with_capacity(capacity),
            free_indexes: Vec::new(),
            retired_indexes: Vec::new(),
            reservation_cursor: AtomicU64::new(0),
        }
    }

    /// Returns the number of elements in the `Registry`.
    pub fn len(&self) -> usize {
        self.slots.len() - self.free_indexes.len() - self.retired_indexes.len()
    }

    // We currently check for two possible violations:
    // 1. The index of an ID should never be larger than the number of slots plus the number of
    //    unallocated reservations. The only ways to violate this rule are to assemble a garbage ID
    //    by hand or to use an ID from one Registry with another of the same time.
    // 2. The generation of an ID should never be newer than its slot. In addition to the cases
    //    above, you can also violate this rule by retaining a dangling ID across a call to
    //    recycle_retired().
    fn debug_best_effort_checks_for_contract_violations(&self, id: ID) {
        if !cfg!(debug_assertions) {
            return;
        }
        if id.is_null() {
            return;
        }
        let new_slots_reserved = self
            .reservation_cursor
            .load(Relaxed)
            .saturating_sub(self.free_indexes.len() as u64);
        let valid_slots = self.slots.len() as u64 + new_slots_reserved;
        debug_assert!((id.index() as u64) < valid_slots);
        let Some(state) = self.slots.state(id.index()) else {
            // an unallocated reservation
            debug_assert_eq!(id.generation(), 0);
            return;
        };
        if state.is_retired() {
            // The retired state is numerically zero (wrapped around), but it's effectively greater
            // than all other states. There's nothing left to check.
            return;
        }
        debug_assert!(
            // Note that the generation of a reserved slot equals the generation of the reserving
            // ID, so we don't need a special case for reservations here.
            id.generation() <= state.generation(),
            "ID generation ({}) is newer than its slot ({}); did it dangle across a recycle_retired()?",
            id.generation(),
            state.generation(),
        );
    }

    /// Returns `true` if the `Registry` contains the given `id`. If [`remove`](Registry::remove)
    /// has been called on `id`, `contains_id` will return `false`.
    pub fn contains_id(&self, id: ID) -> bool {
        self.debug_best_effort_checks_for_contract_violations(id);
        if let Some(state) = self.slots.state(id.index()) {
            // This comparison can only succeed if the generation matches and the occupied bit is
            // one. Note that Id64 is laid out to save an instruction here (the occupied bit is in
            // the ID).
            state == id.matching_state()
        } else {
            false
        }
    }

    /// Get a reference to an element. If [`remove`](Registry::remove) has been called on `id`,
    /// `get` will return `None`.
    pub fn get(&self, id: ID) -> Option<&T> {
        if self.contains_id(id) {
            Some(unsafe { self.get_unchecked(id) })
        } else {
            None
        }
    }

    /// Get a mutable reference to an element. If [`remove`](Registry::remove) has been called on
    /// `id`, `get_mut` will return `None`.
    pub fn get_mut(&mut self, id: ID) -> Option<&mut T> {
        if self.contains_id(id) {
            Some(unsafe { self.get_unchecked_mut(id) })
        } else {
            None
        }
    }

    /// Get a reference to an element without checking the size of the `Registry` or the generation
    /// of the ID.
    ///
    /// This function is safe if and only if `self.contains_id(id)` is `true`.
    pub unsafe fn get_unchecked(&self, id: ID) -> &T {
        self.slots.value_unchecked(id.index()).assume_init_ref()
    }

    /// Get a mutable reference to an element without checking the size of the `Registry` or the
    /// generation of the ID.
    ///
    /// This function is safe if and only if `self.contains_id(id)` is `true`.
    pub unsafe fn get_unchecked_mut(&mut self, id: ID) -> &mut T {
        self.slots.value_unchecked_mut(id.index()).assume_init_mut()
    }

    /// Adds a value to the `Registry` and returns an ID that can be used to access that element.
    #[must_use]
    pub fn insert(&mut self, value: T) -> ID {
        self.allocate_reservations();
        if let Some(free_index) = self.free_indexes.pop() {
            // Reuse a free slot if there are any.
            unsafe {
                self.slots.value_unchecked_mut(free_index).write(value);
                let empty_state = self.slots.state_unchecked(free_index);
                debug_assert!(empty_state.is_empty());
                let occupied_state = empty_state.next_occupied_state();
                self.slots.set_state_unchecked(free_index, occupied_state);
                ID::new(free_index, occupied_state.generation())
            }
        } else {
            // Otherwise allocate a new slot. Panic if the index space is full.
            assert!(self.slots.len() < ID::max_len(), "all slots occupied");
            let new_index = self.slots.len();
            self.slots.insert(value);
            ID::new(new_index, 0)
        }
    }

    /// If `id` refers to an element in the `Registry`, remove the element and return it.
    ///
    /// This method returns `Some` if any only if [`contains_id`] would have returned `true`. After
    /// calling `remove` on an ID, that ID and any copies of it become "dangling". [`contains_id`]
    /// will return `false`, and [`get`], [`get_mut`], and any further calls to `remove` will
    /// return `None`.
    ///
    /// Calling `remove` on an ID that has been reserved with [`reserve_id`] or [`reserve_ids`] but
    /// not yet given a value with [`insert_reserved`], will free the reserved slot and return
    /// `None`.
    ///
    /// See also [`recycle_retired`].
    ///
    /// [`contains_id`]: Registry::contains_id
    /// [`get`]: Registry::get
    /// [`get_mut`]: Registry::get_mut
    /// [`reserve_id`]: Registry::reserve_id
    /// [`reserve_ids`]: Registry::reserve_ids
    /// [`insert_reserved`]: Registry::insert_reserved
    /// [`recycle_retired`]: Registry::recycle_retired
    pub fn remove(&mut self, id: ID) -> Option<T> {
        self.debug_best_effort_checks_for_contract_violations(id);
        self.allocate_reservations();
        let Some(state) = self.slots.state(id.index()) else {
            return None;
        };
        if state != id.matching_state() {
            return None;
        }
        // The ID matches the slot state. Add this slot to the free list or the retired list (which
        // could panic), set the free bit in its state, and move out its value.
        let empty_state = state.next_empty_state();
        if empty_state.is_retired() {
            // The state wraps back to zero after the max generation.
            self.retired_indexes.push(id.index());
        } else {
            self.free_indexes.push(id.index());
        }
        unsafe {
            self.slots.set_state_unchecked(id.index(), empty_state);
            Some(self.slots.value_unchecked(id.index()).assume_init_read())
        }
    }

    /// Reserve an ID that you can insert a value for later.
    ///
    /// Note that this method doesn't require mutable access to the `Registry`. It uses an atomic
    /// cursor internally, and you can reserve an ID while other threads are e.g. reading existing
    /// elements.
    ///
    /// Until you call [`insert_reserved`], the reserved slot is empty, and [`contains_id`] will
    /// report `false` for the reserved ID. Similarly, [`get`] and [`get_mut`] will return `None`.
    /// If a reservation is no longer needed, you can [`remove`] it without inserting a value.
    ///
    /// Whenever you call a `&mut self` method that might change the size of the `Registry` or its
    /// free list ([`insert`], [`insert_reserved`], [`remove`], or [`recycle_retired`]), the
    /// `Registry` will automatically allocate space for any pending reservations. You can
    /// optionally call [`allocate_reservations`] to do this work in advance.
    ///
    /// To reserve many IDs at once, see [`reserve_ids`].
    ///
    /// [`contains_id`]: Registry::contains_id
    /// [`get`]: Registry::get
    /// [`get_mut`]: Registry::get_mut
    /// [`allocate_reservations`]: Registry::allocate_reservations
    /// [`insert`]: Registry::insert
    /// [`remove`]: Registry::remove
    /// [`recycle_retired`]: Registry::recycle_retired
    /// [`insert_reserved`]: Registry::insert_reserved
    /// [`reserve_ids`]: Registry::reserve_ids
    #[must_use]
    pub fn reserve_id(&self) -> ID {
        self.reserve_ids(1).next().unwrap()
    }

    /// Reserve a range of IDs that you can insert values for later.
    ///
    /// See [`reserve_id`].
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
        // We only support indexes up to u32::MAX, but we use an AtomicU64 for the reservation
        // cursor, so that we don't have to worry about the atomic fetch-add overflowing it.
        // Overflow is still theoretically possible, but you'd need 4 billion threads.
        assert!(count <= u32::MAX as usize, "reservation exceeds u32::MAX");
        let old_cursor = self.reservation_cursor.fetch_add(count as u64, Relaxed);
        let new_cursor = old_cursor + count as u64;
        let new_total_slots =
            new_cursor.saturating_sub(self.free_indexes.len() as u64) + self.slots.len() as u64;
        if new_total_slots > ID::max_len() as u64 {
            // It's tempting to try to "repair" the reservation_cursor here, but not having any
            // ordering guarantees makes this impossible. Consider this scenario:
            // - thread A: reserve a gajillion
            // - thread B: reserve 1
            // - thread A: repair a gajillion.
            // - thread C: reserve 1
            // - thread B: repair 1
            // Now thread C is proceeding but is 1 off from where it should be.
            panic!("new length exceeds max for the ID type ({})", ID::max_len());
        }
        return iter::ReservationIter {
            registry: self,
            start: old_cursor as usize,
            end: new_cursor as usize,
        };
    }

    /// Allocate space for reserved slots.
    ///
    /// See [`reserve_id`] and [`reserve_ids`]. This method is called internally by any method that
    /// might change the size of the `Registry` or its free list ([`insert`], [`insert_reserved`],
    /// [`remove`], or [`recycle_retired`]), so you don't need to call it explicitly unless you
    /// want to force the allocation to happen sooner.
    ///
    /// [`reserve_id`]: Registry::reserve_id
    /// [`reserve_ids`]: Registry::reserve_ids
    /// [`insert`]: Registry::insert
    /// [`insert_reserved`]: Registry::insert_reserved
    /// [`remove`]: Registry::remove
    /// [`recycle_retired`]: Registry::recycle_retired
    pub fn allocate_reservations(&mut self) {
        let cursor: &mut u64 = self.reservation_cursor.get_mut();
        if *cursor == 0 {
            return; // There are no reservations.
        }
        let cursor_usize = (*cursor).try_into().expect("reservations overflow usize");

        let reused_slots = cmp::min(cursor_usize, self.free_indexes.len());
        let reused_slots_start = self.free_indexes.len() - reused_slots;
        let new_slots = cursor_usize - reused_slots;

        // Pre-allocate any new slots. Their initial state (0) is already valid for a reserved
        // slot.
        if new_slots > 0 {
            self.slots.allocate_empty_slots(new_slots);
        }

        // Reuse free slots. We don't need to modify their states at all, just remove them from the
        // free indexes list.
        self.free_indexes.truncate(reused_slots_start);

        // Reset the reservation cursor.
        *cursor = 0;
    }

    /// Insert a value for a reserved ID.
    ///
    /// See [`reserve_id`] and [`reserve_ids`]. Empty reservations can be filled in any order. If
    /// you try to fill a reservation multiple times, or if you call this method with IDs that
    /// aren't reserved, it will return an error.
    ///
    /// [`remove`]: Registry::remove
    /// [`reserve_id`]: Registry::reserve_id
    /// [`reserve_ids`]: Registry::reserve_ids
    pub fn insert_reserved(
        &mut self,
        id: ID,
        value: T,
    ) -> Result<(), error::InsertReservedError<T>> {
        self.debug_best_effort_checks_for_contract_violations(id);
        self.allocate_reservations();
        let error_kind;
        if let Some(state) = self.slots.state(id.index()) {
            if state == id.reserved_state() {
                // Happy path: the reservation is valid.
                unsafe {
                    self.slots
                        .set_state_unchecked(id.index(), id.matching_state());
                    self.slots.value_unchecked_mut(id.index()).write(value);
                }
                return Ok(());
            }
            if state == id.matching_state() {
                error_kind = error::InsertReservedErrorKind::Exists;
            } else if state.generation() >= id.generation() {
                error_kind = error::InsertReservedErrorKind::Dangling;
            } else {
                error_kind = error::InsertReservedErrorKind::GenerationTooNew;
            }
        } else {
            error_kind = error::InsertReservedErrorKind::IndexOutOfBounds;
        }
        Err(error::InsertReservedError {
            inner: value,
            kind: error_kind,
        })
    }

    /// Mark all retired slots as free, making them available for future insertions and
    /// reservations.
    ///
    /// Callers using `Registry` with the default [`Id`] type shouldn't need this method. [`Id`]
    /// has 31 generation bits, so the retirement rate is one slot per ~2 billion removals, too
    /// slow to matter in almost any practical case. This method is intended for callers using
    /// [`Id32`](id::Id32).
    ///
    /// Dangling IDs (that is, IDs passed to [`remove`]) from before the call to `recycle_retired`
    /// must not be used again with any method on this `Registry`. Generally callers should delete
    /// all dangling IDs (or replace them with [`null`]) before calling `recycle_retired`. If you
    /// retain dangling IDs across a call to `recycle_retired`, they can collide with newly issued
    /// IDs, and calls to [`get`], [`get_mut`], and [`contains_id`] can return confusing results.
    /// Calling [`insert_reserved`] on these IDs can also lead to memory leaks. This behavior is
    /// memory-safe, but these are logic bugs, similar to the logic bugs that can arise if you
    /// modify a key after it's been inserted into a [`HashMap`].
    ///
    /// # Panics
    ///
    /// `Registry` makes a best effort to detect violations of this rule. Any method on `Registry`
    /// may panic if it sees an ID generation that's newer than the corresponding slot. These
    /// checks are currently only done in debug mode, but this is not guaranteed.
    ///
    /// [`null`]: IdTrait::null
    /// [`remove`]: Registry::remove
    /// [`get`]: Registry::get
    /// [`get_mut`]: Registry::get_mut
    /// [`contains_id`]: Registry::contains_id
    /// [`insert_reserved`]: Registry::insert_reserved
    /// [`HashMap`]: https://doc.rust-lang.org/std/collections/struct.HashMap.html
    pub fn recycle_retired(&mut self) {
        self.allocate_reservations();
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
        Self {
            slots: self.slots.clone(),
            free_indexes: self.free_indexes.clone(),
            retired_indexes: self.retired_indexes.clone(),
            // A clone that races against another thread taking a reservation is probably a bug,
            // but even if reservation_cursor is non-zero, it could be that the cloning thread and
            // the reserving thread are synchronized. (The caller might be cloning prior to calling
            // some &mut self method for example.)
            reservation_cursor: AtomicU64::new(self.reservation_cursor.load(Relaxed)),
        }
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
