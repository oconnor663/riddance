use std::mem::size_of;
use std::num::{NonZeroU16, NonZeroU32};

const DEFAULT_GBITS: u32 = 10;

const fn max_index<const GBITS: u32>() -> u32 {
    u32::MAX >> GBITS
}

const fn max_generation<const GBITS: u32>() -> i16 {
    i16::MAX >> (15 - GBITS)
}

fn assert_gbits<const GBITS: u32>() {
    // GBITS must be non-zero to keep the generation compatible with NonZeroU16 and the Entity
    // representation compatible with NonZeroU32. GBITS must be less than 16 so that the slots list
    // has a sign bit.
    assert!(GBITS > 0);
    assert!(GBITS < 16);
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Entity<const GBITS: u32 = DEFAULT_GBITS>(NonZeroU32);

impl<const GBITS: u32> Entity<GBITS> {
    fn new(index: u32, generation: NonZeroU16) -> Self {
        debug_assert!(index <= (u32::MAX >> GBITS));
        debug_assert!(generation.get() < (1 << GBITS));
        Self((index << GBITS) | NonZeroU32::from(generation))
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

    // The null entity has a generation of 0 (which is never issued to any non-null entity) and an
    // index of 1 (for compatibility with NonZeroU32).
    pub fn null() -> Self {
        assert_gbits::<GBITS>();
        Self(unsafe { NonZeroU32::new_unchecked(1 << GBITS) })
    }

    /// `entity.exists(&allocator)` is shorthand for `allocator.contains(entity)`.
    pub fn exists(&self, allocator: &EntityAllocator<GBITS>) -> bool {
        allocator.contains(*self)
    }

    /// `entity.is_dangling(&allocator)` is shorthand for `!allocator.contains(entity)`.
    pub fn is_dangling(&self, allocator: &EntityAllocator<GBITS>) -> bool {
        !allocator.contains(*self)
    }
}

impl<const GBITS: u32> std::fmt::Debug for Entity<GBITS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("Entity")
            .field("index", &self.index())
            .field("generation", &self.generation())
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct EntityAllocator<const GBITS: u32 = DEFAULT_GBITS> {
    // The value in each slot is its generation. A positive generation means the slot is occupied.
    // A zero or negative generation means the slot is free or retired. (The zero generation only
    // occurs after recycling.)
    slots: Vec<i16>,
    free_slots: Vec<u32>,
    retired_slots: Vec<u32>,
}

impl EntityAllocator<DEFAULT_GBITS> {
    pub fn new() -> Self {
        Self::new_custom()
    }
}

impl<const GBITS: u32> EntityAllocator<GBITS> {
    pub fn new_custom() -> Self {
        assert_gbits::<GBITS>();
        assert!(size_of::<usize>() >= size_of::<u32>());
        Self {
            slots: Vec::new(),
            free_slots: Vec::new(),
            retired_slots: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_slots: Vec::new(),
            retired_slots: Vec::new(),
        }
    }

    fn debug_detect_logic_errors(&self, entity: Entity<GBITS>) {
        if let Some(slot) = self.slots.get(entity.index() as usize) {
            // The only way to have an entity with a generation larger than the absolute value of
            // the corresponding slot is to hold a dangling entity across a call to recycle(). We
            // can't always detect this condition, and if you bump the generation of the recycled
            // slot to exactly match the generation of your dangling entity, you'll get a false
            // positive.
            debug_assert!(
                entity.generation().get() <= slot.abs() as u16,
                "dangling slot detected after recycle",
            );
        } else {
            // The only way to have an index that's larger than any allocated slot is to get it
            // from a different allocator.
            debug_assert!(false, "entity came from a different allocator");
        }
    }

    pub fn contains(&self, entity: Entity<GBITS>) -> bool {
        self.debug_detect_logic_errors(entity);
        if let Some(&slot) = self.slots.get(entity.index() as usize) {
            entity.generation().get() as i16 == slot
        } else {
            false
        }
    }

    pub fn try_allocate(&mut self) -> Option<Entity<GBITS>> {
        if let Some(index) = self.free_slots.pop() {
            // Reuse a free slot.
            let slot = &mut self.slots[index as usize];
            debug_assert!(*slot <= 0, "occupied");
            debug_assert!(*slot > -max_generation::<GBITS>(), "retired");
            // The free generation is the negative of the occupied generation, so the new occupied
            // generation is the negative of the free generation plus one.
            let new_generation = -(*slot) + 1;
            debug_assert!(new_generation <= max_generation::<GBITS>());
            *slot = new_generation;
            return Some(Entity::<GBITS>::new(index, unsafe {
                NonZeroU16::new_unchecked(new_generation as u16)
            }));
        }
        // Check whether the whole slot space is already allocated.
        if self.slots.len() as u64 == max_index::<GBITS>() as u64 + 1 {
            return None;
        }
        // Allocate a new slot with generation 1. Skipping generation 0 is what lets use NonZeroU32
        // inside of Entity, which saves space in types like Option<Entity>.
        let new_index = self.slots.len() as u32;
        let new_generation = NonZeroU16::new(1).unwrap();
        self.slots.push(new_generation.get() as i16);
        Some(Entity::<GBITS>::new(new_index, new_generation))
    }

    pub fn allocate(&mut self) -> Entity<GBITS> {
        self.try_allocate().expect("all slots are occupied")
    }

    pub fn free(&mut self, entity: Entity<GBITS>) {
        self.debug_detect_logic_errors(entity);
        if let Some(slot) = self.slots.get_mut(entity.index() as usize) {
            if *slot == entity.generation().get() as i16 {
                // Push before freeing the slot, since push could panic.
                if *slot == max_generation::<GBITS>() {
                    self.retired_slots.push(entity.index());
                } else {
                    self.free_slots.push(entity.index());
                }
                // A negative generation in the slots list means a free or retired slot. But note
                // that the generation in an entity is never negative.
                *slot = -(*slot);
            }
        }
    }

    /// Move all retired slots to the free list, and set every slot in the free list to generation
    /// zero. The called must guarantee that there are no dangling entities before calling
    /// `recycle`, or else their dangling entities might eventually appear to be valid.
    pub fn recycle(&mut self) {
        // In debug mode, sanity check all the free and retired slots.
        for &index in &self.free_slots {
            debug_assert!((index as usize) < self.slots.len());
            let slot = unsafe { self.slots.get_unchecked_mut(index as usize) };
            debug_assert!(*slot <= 0);
            debug_assert!(*slot > -max_generation::<GBITS>());
        }
        for &index in &self.retired_slots {
            debug_assert!((index as usize) < self.slots.len());
            let slot = unsafe { self.slots.get_unchecked_mut(index as usize) };
            debug_assert!(*slot == -max_generation::<GBITS>());
        }
        // Reserve space before modifying any slots, since reserve might panic.
        self.free_slots.reserve(self.retired_slots.len());
        self.free_slots.extend_from_slice(&self.retired_slots);
        self.retired_slots.clear();
        for &index in &self.free_slots {
            let slot = unsafe { self.slots.get_unchecked_mut(index as usize) };
            *slot = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let mut allocator = EntityAllocator::new();
        let e1 = allocator.allocate();
        let e2 = allocator.allocate();
        assert!(e1.exists(&allocator));
        assert!(e2.exists(&allocator));
        allocator.free(e1);
        assert!(e1.is_dangling(&allocator));
        assert!(e2.exists(&allocator));
        allocator.free(e2);
        assert!(e1.is_dangling(&allocator));
        assert!(e2.is_dangling(&allocator));
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn test_dangling_entity_after_recycle_panics() {
        let mut allocator = EntityAllocator::new();
        let entity = allocator.allocate();
        allocator.free(entity);
        allocator.recycle();
        allocator.contains(entity);
    }

    fn out_of_bounds_index_test_case() {
        let allocator1 = EntityAllocator::new();
        let mut allocator2 = EntityAllocator::new();
        let entity = allocator2.allocate();
        assert!(!allocator1.contains(entity));
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
        EntityAllocator::<0>::new_custom();
    }

    #[test]
    #[should_panic]
    fn test_gbits_16_panics() {
        EntityAllocator::<16>::new_custom();
    }

    fn full_allocator() -> EntityAllocator<15> {
        let mut allocator = EntityAllocator::<15>::new_custom();
        let len = max_index::<15>() as usize + 1;
        for _ in 0..len {
            allocator.allocate();
        }
        allocator
    }

    #[test]
    fn test_fill_slots() {
        full_allocator();
    }

    #[test]
    #[should_panic]
    fn test_overfill_slots() {
        let mut allocator = full_allocator();
        allocator.allocate();
    }

    #[test]
    fn test_gbits_1() {
        // With GBITS=1, there's only 1 possible generation (remember that generation 0 is never
        // allocated), so every freed entity is immediately retired.
        let mut allocator = EntityAllocator::<1>::new_custom();
        let e0 = allocator.allocate();
        let e1 = allocator.allocate();
        assert_eq!(allocator.slots, [1, 1]);
        assert_eq!(e0.index(), 0);
        assert_eq!(e0.generation().get(), 1);
        assert_eq!(e1.index(), 1);
        assert_eq!(e1.generation().get(), 1);
        allocator.free(e0);
        assert_eq!(allocator.slots, [-1, 1]);
        assert_eq!(allocator.free_slots, []);
        assert_eq!(allocator.retired_slots, [0]);
        allocator.free(e1);
        assert_eq!(allocator.slots, [-1, -1]);
        assert_eq!(allocator.free_slots, []);
        assert_eq!(allocator.retired_slots, [0, 1]);
        allocator.recycle();
        assert_eq!(allocator.slots, [0, 0]);
        assert_eq!(allocator.free_slots, [0, 1]);
        assert_eq!(allocator.retired_slots, []);
    }

    #[test]
    fn test_gbits_2() {
        // With GBITS=2, there are 3 possible generations (remember that generation 0 is never
        // allocated). Confirm that we get a new slot on the 4th allocate/free cycle.
        let mut allocator = EntityAllocator::<2>::new_custom();
        let mut entity = allocator.allocate();
        assert_eq!(allocator.slots, [1]);
        allocator.free(entity);
        assert_eq!(allocator.slots, [-1]);
        assert_eq!(allocator.free_slots, [0]);
        assert_eq!(allocator.retired_slots, []);
        entity = allocator.allocate();
        assert_eq!(allocator.slots, [2]);
        assert_eq!(allocator.free_slots, []);
        assert_eq!(allocator.retired_slots, []);
        allocator.free(entity);
        assert_eq!(allocator.slots, [-2]);
        assert_eq!(allocator.free_slots, [0]);
        assert_eq!(allocator.retired_slots, []);
        entity = allocator.allocate();
        assert_eq!(allocator.slots, [3]);
        assert_eq!(allocator.free_slots, []);
        assert_eq!(allocator.retired_slots, []);
        allocator.free(entity);
        assert_eq!(allocator.slots, [-3]);
        assert_eq!(allocator.free_slots, []);
        assert_eq!(allocator.retired_slots, [0]);
        entity = allocator.allocate();
        assert_eq!(allocator.slots, [-3, 1]);
        assert_eq!(allocator.free_slots, []);
        assert_eq!(allocator.retired_slots, [0]);
        allocator.free(entity);
        assert_eq!(allocator.slots, [-3, -1]);
        assert_eq!(allocator.free_slots, [1]);
        assert_eq!(allocator.retired_slots, [0]);
    }
}
