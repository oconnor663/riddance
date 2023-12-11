use crate::{id::IdTrait, Id, Registry};

pub struct ReservationIter<'registry, T, ID: IdTrait = Id<T>> {
    pub(crate) registry: &'registry Registry<T, ID>,
    // Note that these bounds are positions in (or beyond) the free list, not slot indexes.
    pub(crate) start: u32,
    pub(crate) end: u32,
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
                    let generation = crate::occupied_state_from_empty::<ID::GenerationBits>(state);
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
