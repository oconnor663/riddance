use std::marker::PhantomData;
use std::mem::{size_of, MaybeUninit};
use std::num::{NonZeroU16, NonZeroU32};
use std::sync::atomic::{AtomicI64, Ordering::Relaxed};

const DEFAULT_GBITS: u32 = 10;

fn max_slots<const GBITS: u32>() -> u32 {
    assert_gbits::<GBITS>();
    1 << (32 - GBITS)
}

fn max_index<const GBITS: u32>() -> u32 {
    assert_gbits::<GBITS>();
    max_slots::<GBITS>() - 1
}

fn max_generation<const GBITS: u32>() -> u16 {
    assert_gbits::<GBITS>();
    (1 << GBITS) - 1
}

const FREE_TAG_BITS: u16 = 0b11 << 14;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SlotTag {
    Occupied,
    Reserved,
    Free,
}

unsafe fn slot_tag(slot_state: u16) -> SlotTag {
    match slot_state >> 14 {
        0b00 => SlotTag::Occupied,
        0b10 => SlotTag::Reserved,
        0b11 => SlotTag::Free,
        _ => {
            debug_assert!(false, "corrupt slot tag");
            std::hint::unreachable_unchecked();
        }
    }
}

fn slot_generation<const GBITS: u32>(slot_state: u16) -> u16 {
    let all_but_tag_mask = u16::MAX >> 2;
    debug_assert!((slot_state & all_but_tag_mask) <= max_generation::<GBITS>());
    let generation_mask = u16::MAX >> (16 - GBITS);
    slot_state & generation_mask
}

struct Slot<'a, T> {
    state: &'a u16,
    value: &'a MaybeUninit<T>,
}

struct SlotMut<'a, T> {
    state: &'a mut u16,
    value: &'a mut MaybeUninit<T>,
}

fn assert_gbits<const GBITS: u32>() {
    // GBITS must be non-zero to keep the generation compatible with NonZeroU16 and the Id
    // representation compatible with NonZeroU32. GBITS must be <=14 so that the generations lits
    // sign bit.
    assert!(GBITS > 0);
    assert!(GBITS <= 14);
}

// Note that we can't use #[derive(...)] for common traits here, because for example Id should be
// Copy and Ord and Eq even when T isn't. See https://github.com/rust-lang/rust/issues/108894.
#[repr(transparent)]
pub struct Id<T, const GBITS: u32 = DEFAULT_GBITS>(
    NonZeroU32,
    // https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
    PhantomData<fn() -> T>,
);

impl<T, const GBITS: u32> Id<T, GBITS> {
    fn new(index: u32, generation: NonZeroU16) -> Self {
        debug_assert!(index <= (u32::MAX >> GBITS));
        debug_assert!(generation.get() < (1 << GBITS));
        Self((index << GBITS) | NonZeroU32::from(generation), PhantomData)
    }

    pub fn index(&self) -> u32 {
        let index = self.0.get() >> GBITS;
        debug_assert!(index <= max_index::<GBITS>());
        index
    }

    pub fn generation(&self) -> u16 {
        assert_gbits::<GBITS>();
        let mask = u16::MAX >> (16 - GBITS);
        let generation = self.0.get() as u16 & mask;
        debug_assert!(generation <= max_generation::<GBITS>() as u16);
        generation
    }

    // The null id has a generation of 0 (which is never issued to any non-null id) and an index of
    // 1 (for compatibility with NonZeroU32).
    pub fn null() -> Self {
        assert_gbits::<GBITS>();
        Self(NonZeroU32::new(1 << GBITS).unwrap(), PhantomData)
    }

    pub fn is_null(&self) -> bool {
        *self == Self::null()
    }

    /// `id.exists(&registry)` is shorthand for `registry.contains_id(id)`.
    pub fn exists(&self, registry: &Registry<T, GBITS>) -> bool {
        registry.contains_id(*self)
    }

    /// `id.is_dangling(&registry)` is shorthand for `!registry.contains_id(id)`.
    pub fn is_dangling(&self, registry: &Registry<T, GBITS>) -> bool {
        !registry.contains_id(*self)
    }
}

impl<T, const GBITS: u32> Copy for Id<T, GBITS> {}

impl<T, const GBITS: u32> Clone for Id<T, GBITS> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<T, const GBITS: u32> std::fmt::Debug for Id<T, GBITS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("Id")
            .field("index", &self.index())
            .field("generation", &self.generation())
            .finish()
    }
}

impl<T, const GBITS: u32> std::hash::Hash for Id<T, GBITS> {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.0.hash(state);
    }
}

impl<T, const GBITS: u32> PartialEq for Id<T, GBITS> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T, const GBITS: u32> Eq for Id<T, GBITS> {}

impl<T, const GBITS: u32> PartialOrd for Id<T, GBITS> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T, const GBITS: u32> Ord for Id<T, GBITS> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

#[derive(Debug)]
pub struct Registry<T, const GBITS: u32 = DEFAULT_GBITS> {
    // Invariant: self.slot_states.len() and self.slot_values.len() are always equal. They could be
    // one Vec of tuples, but we allocate them separately for performance. (Many operations only
    // need to look up the slot generation and don't need the value, and the value type might have
    // an alignment greater than 2.)
    slot_states: Vec<u16>,
    slot_values: Vec<MaybeUninit<T>>,
    free_indexes: Vec<u32>,
    retired_indexes: Vec<u32>,
    // We can make this an AtomicI32 on platforms that don't support AtomicI64. In that case we'll
    // need to use load+compare_exchange instead of fetch_sub, to prevent overflow.
    reservation_cursor: AtomicI64,
}

impl<T> Registry<T, DEFAULT_GBITS> {
    pub fn new() -> Self {
        Self::with_gbits()
    }
}

impl<T, const GBITS: u32> Registry<T, GBITS> {
    pub fn with_gbits() -> Self {
        assert_gbits::<GBITS>();
        assert!(size_of::<usize>() >= size_of::<u32>());
        Self {
            slot_states: Vec::new(),
            slot_values: Vec::new(),
            free_indexes: Vec::new(),
            retired_indexes: Vec::new(),
            reservation_cursor: AtomicI64::new(0),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slot_states: Vec::with_capacity(capacity),
            slot_values: Vec::with_capacity(capacity),
            free_indexes: Vec::new(),
            retired_indexes: Vec::new(),
            reservation_cursor: AtomicI64::new(0),
        }
    }

    fn num_slots(&self) -> u32 {
        debug_assert_eq!(self.slot_states.len(), self.slot_values.len());
        debug_assert!((self.slot_states.len() as u64) <= (max_slots::<GBITS>() as u64));
        self.slot_states.len() as u32
    }

    unsafe fn slot_from_known_good_index(&self, index: u32) -> Slot<T> {
        debug_assert!(index < self.num_slots());
        let state = self.slot_states.get_unchecked(index as usize);
        let value = self.slot_values.get_unchecked(index as usize);
        Slot { state, value }
    }

    unsafe fn slot_mut_from_known_good_index(&mut self, index: u32) -> SlotMut<T> {
        debug_assert!(index < self.num_slots());
        let state = self.slot_states.get_unchecked_mut(index as usize);
        let value = self.slot_values.get_unchecked_mut(index as usize);
        SlotMut { state, value }
    }

    fn debug_assert_in_bounds(&self, id: Id<T, GBITS>) {
        if id.is_null() {
            return;
        }
        let mut upper_bound = self.num_slots() as i64;
        let reservation_cursor = self.reservation_cursor.load(Relaxed);
        if reservation_cursor < 0 {
            upper_bound = upper_bound.saturating_add(reservation_cursor.saturating_abs());
        }
        debug_assert!(
            (id.index() as i64) < upper_bound,
            "index out of bounds (from another Registry?)",
        );
    }

    fn slot(&self, id: Id<T, GBITS>) -> Option<Slot<T>> {
        self.debug_assert_in_bounds(id);
        if id.index() < self.num_slots() {
            let slot = unsafe { self.slot_from_known_good_index(id.index()) };
            debug_assert!(
                id.generation() <= slot_generation::<GBITS>(*slot.state),
                "ID newer than slot (dangling ID retained across recycle?)",
            );
            Some(slot)
        } else {
            None
        }
    }

    fn slot_mut(&mut self, id: Id<T, GBITS>) -> Option<SlotMut<T>> {
        debug_assert_eq!(self.slot_states.len(), self.slot_values.len());
        self.debug_assert_in_bounds(id);
        if id.index() < self.num_slots() {
            let slot = unsafe { self.slot_mut_from_known_good_index(id.index()) };
            debug_assert!(
                id.generation() <= slot_generation::<GBITS>(*slot.state),
                "ID newer than slot (dangling ID retained across recycle?)",
            );
            Some(slot)
        } else {
            None
        }
    }

    pub fn contains_id(&self, id: Id<T, GBITS>) -> bool {
        self.get(id).is_some()
    }

    pub fn get(&self, id: Id<T, GBITS>) -> Option<&T> {
        if let Some(slot) = self.slot(id) {
            // Upper state bits are zero if the slot is occupied.
            if id.generation() == *slot.state {
                unsafe {
                    return Some(slot.value.assume_init_ref());
                }
            }
        }
        None
    }

    pub fn get_mut(&mut self, id: Id<T, GBITS>) -> Option<&mut T> {
        if let Some(slot) = self.slot_mut(id) {
            // Upper state bits are zero if the slot is occupied.
            if id.generation() == *slot.state {
                unsafe {
                    return Some(slot.value.assume_init_mut());
                }
            }
        }
        None
    }

    pub fn insert(&mut self, value: T) -> Id<T, GBITS> {
        // TODO: check for a pending reservation here
        // Reuse a free slot if there are any.
        if let Some(index) = self.free_indexes.pop() {
            unsafe {
                let slot = self.slot_mut_from_known_good_index(index);
                debug_assert_eq!(slot_tag(*slot.state), SlotTag::Free);
                debug_assert!(slot_generation::<GBITS>(*slot.state) < max_generation::<GBITS>());
                let new_generation = slot_generation::<GBITS>(*slot.state) + 1;
                // The occupied state is the generation with no higher bits set.
                *slot.state = new_generation;
                slot.value.write(value);
                *self.reservation_cursor.get_mut() = self.free_indexes.len() as i64;
                return Id::new(index, NonZeroU16::new_unchecked(new_generation));
            }
        }
        // Panic if the index space is full.
        assert!(
            (self.slot_states.len() as u32) < max_slots::<GBITS>(),
            "all slots occupied",
        );
        // Reserve Vec space.
        self.slot_states.reserve(1);
        self.slot_values.reserve(1);
        // Allocate a new slot with generation 1. Skipping generation 0 is what lets use NonZeroU32
        // inside of Id, which saves space in types like Option<Id>.
        let new_index = self.num_slots();
        let new_generation = NonZeroU16::new(1).unwrap();
        // The occupied state is the generation with no higher bits set.
        self.slot_states.push(new_generation.get());
        self.slot_values.push(MaybeUninit::new(value));
        Id::<T, GBITS>::new(new_index, new_generation)
    }

    pub fn remove(&mut self, id: Id<T, GBITS>) -> Option<T> {
        // TODO: check for a pending reservation here
        let Some(slot) = self.slot_mut(id) else {
            return None;
        };
        // Removing an occupied slot returns the value if the generation matches. Removing a
        // free slot has no effect and returns nothing. Removing a reserved slot also returns
        // nothing, but it frees the slot if the generation matches. (Note that the generation
        // gets bumped in the transition from Free to Occupied or Free to Reserved, but not in
        // the transition from Reserved to Occupied.)
        let tag = unsafe { slot_tag(*slot.state) };
        match tag {
            SlotTag::Occupied | SlotTag::Reserved => {
                if id.generation() != slot_generation::<GBITS>(*slot.state) {
                    return None;
                }
            }
            SlotTag::Free => return None,
        }
        // The slot matches the ID. Free it or retire it, depending on the generation. The Vec push
        // could panic, so do this before modifying the tag.
        if slot_generation::<GBITS>(*slot.state) < max_generation::<GBITS>() {
            self.free_indexes.push(id.index());
            *self.reservation_cursor.get_mut() = self.free_indexes.len() as i64;
        } else {
            self.retired_indexes.push(id.index());
        }
        // Grab the slot again, since we had to let the first borrow die to access
        // self.free_indexes and self.retired_indexes above.
        let slot = unsafe { self.slot_mut_from_known_good_index(id.index()) };
        // Set the tag bits in the slot state.
        *slot.state |= FREE_TAG_BITS;
        debug_assert_eq!(unsafe { slot_tag(*slot.state) }, SlotTag::Free);
        // If the previous tag was Occupied, read out the removed value. Either way the slot is now
        // logically uninitialized.
        if tag == SlotTag::Occupied {
            unsafe { Some(slot.value.assume_init_read()) }
        } else {
            None
        }
    }

    /// Move all retired slots to the free list, and set every slot in the free list to generation
    /// zero. The caller must guarantee that there are no dangling IDs before calling `recycle`, or
    /// else their dangling IDs might get reused for newly inserted objects.
    pub fn recycle(&mut self) {
        // In debug mode, sanity check the generations of all the free and retired slots.
        for &index in &self.free_indexes {
            let slot = unsafe { self.slot_from_known_good_index(index) };
            debug_assert!(unsafe { slot_tag(*slot.state) } == SlotTag::Free);
            debug_assert!(slot_generation::<GBITS>(*slot.state) < max_generation::<GBITS>());
        }
        for &index in &self.retired_indexes {
            let slot = unsafe { self.slot_from_known_good_index(index) };
            debug_assert!(unsafe { slot_tag(*slot.state) } == SlotTag::Free);
            debug_assert_eq!(
                slot_generation::<GBITS>(*slot.state),
                max_generation::<GBITS>(),
            );
        }
        // If append succeeds, it clears the retired indexes Vec.
        self.free_indexes.append(&mut self.retired_indexes);
        for i in 0..self.free_indexes.len() {
            let slot_index = self.free_indexes[i];
            let slot = unsafe { self.slot_mut_from_known_good_index(slot_index) };
            // The Free slot tag is the two high bits set, and the generation bits in a recycled
            // slot are all zero.
            *slot.state = FREE_TAG_BITS;
        }
    }

    /// Create an iterator that will yield a count of new IDs.
    /// Note that this takes `&self`, not `&mut self`. Internally it bumps an atomic cursor, and
    /// multiple threads can call it at once.
    pub fn reserve_ids(&self, count: usize) -> ReservationIterator<T, GBITS> {
        todo!()
    }

    fn fill_reservation(&mut self, id: Id<T, GBITS>, value: T) {
        todo!()
    }
}

impl<T, const GBITS: u32> Drop for Registry<T, GBITS> {
    fn drop(&mut self) {
        for (state, value) in self.slot_states.iter().zip(&mut self.slot_values) {
            let tag = unsafe { slot_tag(*state) };
            if tag == SlotTag::Occupied {
                unsafe {
                    value.assume_init_drop();
                }
            }
        }
    }
}

impl<T, const GBITS: u32> Clone for Registry<T, GBITS>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        // Make all our Vec allocations before we clone any values into the new values list, to
        // avoid leaking if there's a panic.
        let mut clone = Self {
            slot_states: Vec::with_capacity(self.num_slots() as usize),
            slot_values: Vec::with_capacity(self.num_slots() as usize),
            free_indexes: self.free_indexes.clone(),
            retired_indexes: self.retired_indexes.clone(),
            // Note that cloning a Registry can race with other threads making reservations.
            reservation_cursor: AtomicI64::new(self.reservation_cursor.load(Relaxed)),
        };
        // Now clone values and move (state, value) pairs into the new Vecs one-by-one. The state
        // will be safe to drop at each point.
        for (state, value) in self.slot_states.iter().zip(&self.slot_values) {
            let tag = unsafe { slot_tag(*state) };
            if tag == SlotTag::Occupied {
                // This clone could panic, so we only push the state entry after it succeeds.
                let new_val = unsafe { value.assume_init_ref().clone() };
                clone.slot_values.push(MaybeUninit::new(new_val));
            } else {
                clone.slot_values.push(MaybeUninit::uninit());
            }
            clone.slot_states.push(*state);
        }
        clone
    }
}

impl<T, const GBITS: u32> std::ops::Index<Id<T, GBITS>> for Registry<T, GBITS> {
    type Output = T;

    fn index(&self, id: Id<T, GBITS>) -> &T {
        self.get(id).unwrap()
    }
}

impl<T, const GBITS: u32> std::ops::IndexMut<Id<T, GBITS>> for Registry<T, GBITS> {
    fn index_mut(&mut self, id: Id<T, GBITS>) -> &mut T {
        self.get_mut(id).unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct ReservationIterator<'registry, T, const GBITS: u32> {
    registry: &'registry Registry<T, GBITS>,
    current_position: i64,
    end_position: i64,
}

impl<'registry, T, const GBITS: u32> Iterator for ReservationIterator<'registry, T, GBITS> {
    type Item = Id<T, GBITS>;

    fn next(&mut self) -> Option<Id<T, GBITS>> {
        // The position moves down, through the free_indexes list while the position is positive,
        // and then into will-be-newly-allocated slots if it goes negative. I stole this design
        // from Bevy :)
        if self.current_position == self.end_position {
            return None;
        }
        self.current_position -= 1;
        debug_assert!(self.current_position >= self.end_position);
        debug_assert!(self.current_position < self.registry.free_indexes.len() as i64);
        if self.current_position >= 0 {
            // Consume the free list from right to left, like a stack. This avoids a memmove later
            // if we didn't reserve the whole thing.
            unsafe {
                let slot_index = *self
                    .registry
                    .free_indexes
                    .get_unchecked(self.current_position as usize);
                let slot = self.registry.slot_from_known_good_index(slot_index);
                let tag = slot_tag(*slot.state);
                debug_assert_eq!(tag, SlotTag::Free);
                let slot_generation = slot_generation::<GBITS>(*slot.state);
                debug_assert!(slot_generation < max_generation::<GBITS>());
                let new_generation = slot_generation + 1;
                Some(Id::new(
                    slot_index,
                    NonZeroU16::new_unchecked(new_generation),
                ))
            }
        } else {
            let index = self.registry.num_slots() as i64 - 1 - self.current_position;
            debug_assert!(index <= max_index::<GBITS>() as i64);
            let generation = NonZeroU16::new(1).unwrap();
            Some(Id::new(index as u32, generation))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_remove() {
        let mut registry = Registry::new();
        let e1 = registry.insert("foo".to_string());
        let e2 = registry.insert("bar".to_string());
        assert!(e1.exists(&registry));
        assert!(e2.exists(&registry));
        assert_eq!(registry.get(e1), Some(&"foo".to_string()));
        assert_eq!(registry.get(e2), Some(&"bar".to_string()));
        assert_eq!(registry.get_mut(e1), Some(&mut "foo".to_string()));
        assert_eq!(registry.get_mut(e2), Some(&mut "bar".to_string()));
        assert_eq!(&registry[e1], "foo");
        assert_eq!(&registry[e2], "bar");
        assert_eq!(&mut registry[e1], "foo");
        assert_eq!(&mut registry[e2], "bar");

        assert_eq!(registry.remove(e1), Some("foo".into()));
        assert!(e1.is_dangling(&registry));
        assert!(e2.exists(&registry));
        assert_eq!(registry.get(e1), None);
        assert_eq!(registry.get(e2), Some(&"bar".to_string()));
        assert_eq!(registry.get_mut(e1), None);
        assert_eq!(registry.get_mut(e2), Some(&mut "bar".to_string()));
        assert_eq!(&registry[e2], "bar");
        assert_eq!(&mut registry[e2], "bar");

        assert_eq!(registry.remove(e2), Some("bar".into()));
        assert!(e1.is_dangling(&registry));
        assert!(e2.is_dangling(&registry));
        assert_eq!(registry.get(e1), None);
        assert_eq!(registry.get(e2), None);
        assert_eq!(registry.get_mut(e1), None);
        assert_eq!(registry.get_mut(e2), None);
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn test_dangling_id_after_recycle_panics() {
        let mut registry = Registry::new();
        let id = registry.insert(());
        registry.remove(id);
        registry.recycle();
        registry.contains_id(id);
    }

    fn out_of_bounds_index_test_case() {
        let registry1 = Registry::new();
        let mut registry2 = Registry::new();
        let id = registry2.insert(());
        assert!(!registry1.contains_id(id));
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn test_out_of_bounds_index() {
        out_of_bounds_index_test_case();
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_out_of_bounds_index() {
        out_of_bounds_index_test_case();
    }

    #[test]
    #[should_panic]
    fn test_gbits_0_panics() {
        Registry::<(), 0>::with_gbits();
    }

    #[test]
    #[should_panic]
    fn test_gbits_15_panics() {
        Registry::<(), 15>::with_gbits();
    }

    fn full_registry() -> Registry<(), 14> {
        let mut registry = Registry::<(), 14>::with_gbits();
        let len = max_index::<14>() as usize + 1;
        for _ in 0..len {
            registry.insert(());
        }
        registry
    }

    #[test]
    fn test_fill_slots() {
        full_registry();
    }

    #[test]
    #[should_panic]
    fn test_overfill_slots() {
        let mut registry = full_registry();
        registry.insert(());
    }

    #[test]
    fn test_gbits_1() {
        // With GBITS=1, there's only 1 possible generation (remember that generation 0 is never
        // allocated), so every freed Id is immediately retired.
        let mut registry = Registry::<(), 1>::with_gbits();
        let e0 = registry.insert(());
        let e1 = registry.insert(());
        assert_eq!(registry.slot_states, [1, 1]);
        assert_eq!(e0.index(), 0);
        assert_eq!(e0.generation(), 1);
        assert_eq!(e1.index(), 1);
        assert_eq!(e1.generation(), 1);
        registry.remove(e0);
        assert_eq!(registry.slot_states, [FREE_TAG_BITS | 1, 1]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0]);
        registry.remove(e1);
        assert_eq!(registry.slot_states, [FREE_TAG_BITS | 1, FREE_TAG_BITS | 1]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0, 1]);
        registry.recycle();
        assert_eq!(registry.slot_states, [FREE_TAG_BITS | 0, FREE_TAG_BITS | 0]);
        assert_eq!(registry.free_indexes, [0, 1]);
        assert_eq!(registry.retired_indexes, []);
    }

    #[test]
    fn test_gbits_2() {
        // With GBITS=2, there are 3 possible generations (remember that generation 0 is never
        // allocated). Confirm that we get a new slot on the 4th allocate/free cycle.
        let mut registry = Registry::<(), 2>::with_gbits();
        let mut id = registry.insert(());
        assert_eq!(registry.slot_states, [1]);
        registry.remove(id);
        assert_eq!(registry.slot_states, [FREE_TAG_BITS | 1]);
        assert_eq!(registry.free_indexes, [0]);
        assert_eq!(registry.retired_indexes, []);
        id = registry.insert(());
        assert_eq!(registry.slot_states, [2]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, []);
        registry.remove(id);
        assert_eq!(registry.slot_states, [FREE_TAG_BITS | 2]);
        assert_eq!(registry.free_indexes, [0]);
        assert_eq!(registry.retired_indexes, []);
        id = registry.insert(());
        assert_eq!(registry.slot_states, [3]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, []);
        registry.remove(id);
        assert_eq!(registry.slot_states, [FREE_TAG_BITS | 3]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0]);
        id = registry.insert(());
        assert_eq!(registry.slot_states, [FREE_TAG_BITS | 3, 1]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0]);
        registry.remove(id);
        assert_eq!(registry.slot_states, [FREE_TAG_BITS | 3, FREE_TAG_BITS | 1]);
        assert_eq!(registry.free_indexes, [1]);
        assert_eq!(registry.retired_indexes, [0]);
    }
}
