use num_traits::{bounds::UpperBounded, AsPrimitive};
use std::num::{NonZeroU32, NonZeroU64};

pub trait Entity: Sized + Copy
where
    usize: AsPrimitive<Self::Index>,
    i64: AsPrimitive<Self::Generation>,
{
    type Index: Copy + UpperBounded + AsPrimitive<usize>;
    type Generation: Copy + UpperBounded + AsPrimitive<i64>;

    const GBITS: u32;

    fn index(&self) -> Self::Index;

    fn generation(&self) -> Self::Generation;

    fn from_index_and_generation(index: Self::Index, generation: Self::Generation) -> Self;

    fn null() -> Self {
        Self::from_index_and_generation(1usize.as_(), 0i64.as_())
    }

    fn is_null(&self) -> bool {
        self.index().as_() == 1usize && self.generation().as_() == 0i64
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Entity32<const GBITS: u32 = 10>(pub NonZeroU32);

impl<const GBITS: u32> Entity for Entity32<GBITS> {
    type Index = u32;
    type Generation = i16;

    const GBITS: u32 = GBITS;

    #[inline]
    fn index(&self) -> u32 {
        self.0.get() >> GBITS
    }

    #[inline]
    fn generation(&self) -> i16 {
        assert!(GBITS <= 15);
        let mask = (1u32 << GBITS) - 1;
        (self.0.get() & mask) as i16
    }

    #[inline]
    fn from_index_and_generation(index: u32, generation: i16) -> Entity32<GBITS> {
        Entity32(NonZeroU32::new((index << GBITS) + generation as u32).unwrap())
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Entity64<const GBITS: u32 = 24>(pub NonZeroU64);

impl<const GBITS: u32> Entity for Entity64<GBITS> {
    type Index = u64;
    type Generation = i32;

    const GBITS: u32 = GBITS;

    #[inline]
    fn index(&self) -> u64 {
        self.0.get() >> GBITS
    }

    #[inline]
    fn generation(&self) -> i32 {
        debug_assert!(GBITS <= 31);
        let mask = (1u64 << GBITS) - 1;
        (self.0.get() & mask) as i32
    }

    #[inline]
    fn from_index_and_generation(index: u64, generation: i32) -> Entity64<GBITS> {
        Entity64(NonZeroU64::new((index << GBITS) + generation as u64).unwrap())
    }
}

#[derive(Clone, Debug)]
pub struct RrEntityAllocator<E: Entity>
where
    usize: AsPrimitive<<E as Entity>::Index>,
    i64: AsPrimitive<<E as Entity>::Generation>,
{
    slots: Vec<E::Generation>,
    free_slots: Vec<E::Index>,
    retired_slots: Vec<E::Index>,
}

impl<E: Entity> RrEntityAllocator<E>
where
    usize: AsPrimitive<<E as Entity>::Index>,
    i64: AsPrimitive<<E as Entity>::Generation>,
{
    pub fn new() -> Self {
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

    fn slot_generation(&self, entity: E) -> E::Generation {
        self.slots[entity.index().as_()]
    }

    pub fn allocate(&mut self) -> E {
        if let Some(index) = self.free_slots.pop() {
            let generation = self.slots[index.as_()];
            debug_assert!(generation.as_() <= 0i64);
            let new_generation = -generation.as_() + 1i64;
            self.slots[index.as_()] = new_generation.as_();
            return E::from_index_and_generation(index, new_generation.as_());
        }
        let max_index = E::Index::max_value().as_() >> E::GBITS;
        assert!(self.slots.len() < max_index, "indexes exhausted");
        let new_index = self.slots.len().as_();
        let new_generation = 1i64.as_();
        self.slots.push(new_generation);
        E::from_index_and_generation(new_index, new_generation)
    }

    pub fn free(&mut self, entity: E) {
        todo!()
    }

    pub fn contains(&self, entity: E) -> bool {
        debug_assert!(
            entity.generation().as_() <= self.slot_generation(entity).as_(),
            "dangling entity held across recycle",
        );
        entity.generation().as_() == self.slot_generation(entity).as_()
    }

    pub fn recycle(&mut self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {}
}
