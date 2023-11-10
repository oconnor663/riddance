use std::marker::PhantomData;
use std::mem::{size_of, MaybeUninit};
use std::num::{NonZeroU16, NonZeroU32};

const DEFAULT_GBITS: u32 = 10;

const fn max_index<const GBITS: u32>() -> u32 {
    u32::MAX >> GBITS
}

const fn max_generation<const GBITS: u32>() -> i16 {
    i16::MAX >> (15 - GBITS)
}

fn assert_gbits<const GBITS: u32>() {
    // GBITS must be non-zero to keep the generation compatible with NonZeroU16 and the Id
    // representation compatible with NonZeroU32. GBITS must be less than 16 so that the slots list
    // has a sign bit.
    assert!(GBITS > 0);
    assert!(GBITS < 16);
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

    pub fn generation(&self) -> NonZeroU16 {
        assert_gbits::<GBITS>();
        let mask = u16::MAX >> (16 - GBITS);
        let generation = self.0.get() as u16 & mask;
        debug_assert!(generation > 0);
        debug_assert!(generation <= max_generation::<GBITS>() as u16);
        unsafe { NonZeroU16::new_unchecked(generation) }
    }

    // The null id has a generation of 0 (which is never issued to any non-null id) and an index of
    // 1 (for compatibility with NonZeroU32).
    pub fn null() -> Self {
        assert_gbits::<GBITS>();
        Self(
            unsafe { NonZeroU32::new_unchecked(1 << GBITS) },
            PhantomData,
        )
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
    // A positive generation means the slot is occupied. A zero or negative generation means the
    // slot is free or retired. (The zero generation only occurs after recycling.)
    generations: Vec<i16>,
    values: Vec<MaybeUninit<T>>,
    free_indexes: Vec<u32>,
    retired_indexes: Vec<u32>,
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
            generations: Vec::new(),
            values: Vec::new(),
            free_indexes: Vec::new(),
            retired_indexes: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            generations: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            free_indexes: Vec::new(),
            retired_indexes: Vec::new(),
        }
    }

    fn debug_detect_logic_errors(&self, id: Id<T, GBITS>) {
        if let Some(slot) = self.generations.get(id.index() as usize) {
            // The only way to have an Id with a generation larger than the absolute value of the
            // corresponding slot is to hold a dangling Id across a call to recycle(). We can't
            // always detect this condition, and if you bump the generation of the recycled slot to
            // exactly match the generation of your dangling Id, you'll get a false positive.
            debug_assert!(
                id.generation().get() <= slot.abs() as u16,
                "dangling slot detected after recycle",
            );
        } else {
            // The only way to have an index that's larger than any allocated slot is to get it
            // from a different registry.
            debug_assert!(false, "Id came from a different registry");
        }
    }

    pub fn contains_id(&self, id: Id<T, GBITS>) -> bool {
        self.debug_detect_logic_errors(id);
        if let Some(&slot) = self.generations.get(id.index() as usize) {
            id.generation().get() as i16 == slot
        } else {
            false
        }
    }

    pub fn get(&self, id: Id<T, GBITS>) -> Option<&T> {
        if let Some(&slot) = self.generations.get(id.index() as usize) {
            if id.generation().get() as i16 == slot {
                unsafe {
                    return Some(self.values[id.index() as usize].assume_init_ref());
                }
            }
        }
        None
    }

    pub fn get_mut(&mut self, id: Id<T, GBITS>) -> Option<&mut T> {
        if let Some(&slot) = self.generations.get(id.index() as usize) {
            if id.generation().get() as i16 == slot {
                unsafe {
                    return Some(self.values[id.index() as usize].assume_init_mut());
                }
            }
        }
        None
    }

    pub fn try_insert(&mut self, value: T) -> Option<Id<T, GBITS>> {
        debug_assert_eq!(self.generations.len(), self.values.len());
        if let Some(index) = self.free_indexes.pop() {
            // Reuse a free slot.
            let slot = &mut self.generations[index as usize];
            debug_assert!(*slot <= 0, "occupied");
            debug_assert!(*slot > -max_generation::<GBITS>(), "retired");
            // The free generation is the negative of the occupied generation, so the new occupied
            // generation is the negative of the free generation plus one.
            let new_generation = -(*slot) + 1;
            debug_assert!(new_generation <= max_generation::<GBITS>());
            *slot = new_generation;
            self.values[index as usize].write(value);
            return Some(Id::<T, GBITS>::new(index, unsafe {
                NonZeroU16::new_unchecked(new_generation as u16)
            }));
        }
        // Check whether the whole slot space is already allocated.
        if self.generations.len() as u64 == max_index::<GBITS>() as u64 + 1 {
            return None;
        }
        // Allocate a new slot with generation 1. Skipping generation 0 is what lets use NonZeroU32
        // inside of Id, which saves space in types like Option<Id>.
        let new_index = self.generations.len() as u32;
        let new_generation = NonZeroU16::new(1).unwrap();
        self.generations.push(new_generation.get() as i16);
        self.values.push(MaybeUninit::new(value));
        Some(Id::<T, GBITS>::new(new_index, new_generation))
    }

    pub fn insert(&mut self, value: T) -> Id<T, GBITS> {
        self.try_insert(value).expect("all slots are occupied")
    }

    pub fn remove(&mut self, id: Id<T, GBITS>) -> Option<T> {
        self.debug_detect_logic_errors(id);
        if let Some(slot) = self.generations.get_mut(id.index() as usize) {
            if *slot == id.generation().get() as i16 {
                // Push before freeing the slot, since push could panic.
                if *slot >= max_generation::<GBITS>() {
                    self.retired_indexes.push(id.index());
                } else {
                    self.free_indexes.push(id.index());
                }
                // A negative generation in the slots list means a free or retired slot. But note
                // that the generation in an Id is never negative.
                *slot = -(*slot);
                // Read out the old value. The slot is now logically uninitialized.
                unsafe {
                    return Some(self.values[id.index() as usize].assume_init_read());
                }
            }
        }
        None
    }

    /// Move all retired slots to the free list, and set every slot in the free list to generation
    /// zero. The called must guarantee that there are no dangling IDs before calling `recycle`, or
    /// else their dangling IDs might eventually appear to be valid.
    pub fn recycle(&mut self) {
        // In debug mode, sanity check all the free and retired slots.
        for &index in &self.free_indexes {
            debug_assert!((index as usize) < self.generations.len());
            let slot = unsafe { self.generations.get_unchecked_mut(index as usize) };
            debug_assert!(*slot <= 0);
            debug_assert!(*slot > -max_generation::<GBITS>());
        }
        for &index in &self.retired_indexes {
            debug_assert!((index as usize) < self.generations.len());
            let slot = unsafe { self.generations.get_unchecked_mut(index as usize) };
            debug_assert!(*slot == -max_generation::<GBITS>());
        }
        // Reserve space before modifying any slots, since reserve might panic.
        self.free_indexes.reserve(self.retired_indexes.len());
        self.free_indexes.extend_from_slice(&self.retired_indexes);
        self.retired_indexes.clear();
        for &index in &self.free_indexes {
            let slot = unsafe { self.generations.get_unchecked_mut(index as usize) };
            *slot = 0;
        }
    }
}

impl<T, const GBITS: u32> Drop for Registry<T, GBITS> {
    fn drop(&mut self) {
        for (&index, value) in self.generations.iter().zip(&mut self.values) {
            if index > 0 {
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
        debug_assert_eq!(self.generations.len(), self.values.len());
        let mut new_values = Vec::with_capacity(self.values.len());
        for (&gen, val) in self.generations.iter().zip(self.values.iter()) {
            if gen > 0 {
                let new_val = unsafe { val.assume_init_ref().clone() };
                new_values.push(MaybeUninit::new(new_val));
            } else {
                new_values.push(MaybeUninit::uninit());
            }
        }
        Self {
            generations: self.generations.clone(),
            values: new_values,
            free_indexes: self.free_indexes.clone(),
            retired_indexes: self.retired_indexes.clone(),
        }
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
    fn test_gbits_16_panics() {
        Registry::<(), 16>::with_gbits();
    }

    fn full_registry() -> Registry<(), 15> {
        let mut registry = Registry::<(), 15>::with_gbits();
        let len = max_index::<15>() as usize + 1;
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
        assert_eq!(registry.generations, [1, 1]);
        assert_eq!(e0.index(), 0);
        assert_eq!(e0.generation().get(), 1);
        assert_eq!(e1.index(), 1);
        assert_eq!(e1.generation().get(), 1);
        registry.remove(e0);
        assert_eq!(registry.generations, [-1, 1]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0]);
        registry.remove(e1);
        assert_eq!(registry.generations, [-1, -1]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0, 1]);
        registry.recycle();
        assert_eq!(registry.generations, [0, 0]);
        assert_eq!(registry.free_indexes, [0, 1]);
        assert_eq!(registry.retired_indexes, []);
    }

    #[test]
    fn test_gbits_2() {
        // With GBITS=2, there are 3 possible generations (remember that generation 0 is never
        // allocated). Confirm that we get a new slot on the 4th allocate/free cycle.
        let mut registry = Registry::<(), 2>::with_gbits();
        let mut id = registry.insert(());
        assert_eq!(registry.generations, [1]);
        registry.remove(id);
        assert_eq!(registry.generations, [-1]);
        assert_eq!(registry.free_indexes, [0]);
        assert_eq!(registry.retired_indexes, []);
        id = registry.insert(());
        assert_eq!(registry.generations, [2]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, []);
        registry.remove(id);
        assert_eq!(registry.generations, [-2]);
        assert_eq!(registry.free_indexes, [0]);
        assert_eq!(registry.retired_indexes, []);
        id = registry.insert(());
        assert_eq!(registry.generations, [3]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, []);
        registry.remove(id);
        assert_eq!(registry.generations, [-3]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0]);
        id = registry.insert(());
        assert_eq!(registry.generations, [-3, 1]);
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0]);
        registry.remove(id);
        assert_eq!(registry.generations, [-3, -1]);
        assert_eq!(registry.free_indexes, [1]);
        assert_eq!(registry.retired_indexes, [0]);
    }
}
