//! iterator types

use crate::{id::IdTrait, Id, Registry};

/// An iterator over `(ID, &T)`. Returned by [`iter`](crate::Registry::iter) or automatically
/// constructed by iterating over `&Registry`.
#[derive(Clone, Debug)]
pub struct Iter<'registry, T, ID: IdTrait = Id<T>> {
    pub(crate) registry: &'registry Registry<T, ID>,
    pub(crate) index: u32,
}

impl<'registry, T, ID: IdTrait> Iterator for Iter<'registry, T, ID> {
    type Item = (ID, &'registry T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.registry.slots.len {
            unsafe {
                let state = self.registry.slots.state_unchecked(self.index);
                let slot = self.registry.slots.value_unchecked(self.index).as_ptr();
                self.index += 1;
                if crate::state_is_occupied::<ID::GenerationBits>(state) {
                    return Some((ID::new_unchecked(self.index - 1, state), &*slot));
                }
            }
        }
        None
    }
}

impl<'registry, T, ID: IdTrait> IntoIterator for &'registry Registry<T, ID> {
    type Item = (ID, &'registry T);
    type IntoIter = Iter<'registry, T, ID>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator over `ID`. Returned by [`ids`](crate::Registry::ids).
#[derive(Clone, Debug)]
pub struct Ids<'registry, T, ID: IdTrait = Id<T>> {
    pub(crate) inner: Iter<'registry, T, ID>,
}

impl<'registry, T, ID: IdTrait> Iterator for Ids<'registry, T, ID> {
    type Item = ID;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(id, _val)| id)
    }
}

/// An iterator over `T`. Returned by [`values`](crate::Registry::values).
#[derive(Clone, Debug)]
pub struct Values<'registry, T, ID: IdTrait = Id<T>> {
    pub(crate) inner: Iter<'registry, T, ID>,
}

impl<'registry, T, ID: IdTrait> Iterator for Values<'registry, T, ID> {
    type Item = &'registry T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_id, val)| val)
    }
}

/// An iterator over `(ID, &mut T)`. Returned by [`iter_mut`](crate::Registry::iter_mut) or
/// automatically constructed by iterating over `&mut Registry`.
#[derive(Debug)]
pub struct IterMut<'registry, T, ID: IdTrait = Id<T>> {
    pub(crate) registry: &'registry mut Registry<T, ID>,
    pub(crate) index: u32,
}

impl<'registry, T, ID: IdTrait> Iterator for IterMut<'registry, T, ID> {
    type Item = (ID, &'registry mut T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.registry.slots.len {
            unsafe {
                let state = self.registry.slots.state_unchecked(self.index);
                let slot = self
                    .registry
                    .slots
                    .value_unchecked_mut(self.index)
                    .as_mut_ptr();
                self.index += 1;
                if crate::state_is_occupied::<ID::GenerationBits>(state) {
                    return Some((ID::new_unchecked(self.index - 1, state), &mut *slot));
                }
            }
        }
        None
    }
}

impl<'registry, T, ID: IdTrait> IntoIterator for &'registry mut Registry<T, ID> {
    type Item = (ID, &'registry mut T);
    type IntoIter = IterMut<'registry, T, ID>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// An iterator over `&mut T`. Returned by [`values_mut`](crate::Registry::values_mut).
#[derive(Debug)]
pub struct ValuesMut<'registry, T, ID: IdTrait = Id<T>> {
    pub(crate) inner: IterMut<'registry, T, ID>,
}

impl<'registry, T, ID: IdTrait> Iterator for ValuesMut<'registry, T, ID> {
    type Item = &'registry mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_id, val)| val)
    }
}

/// An iterator over `(ID, T)`. Returned by [`into_iter`](crate::Registry::into_iter) or
/// automatically constructed by iterating over `Registry`.
#[derive(Clone, Debug)]
pub struct IntoIter<T, ID: IdTrait = Id<T>> {
    pub(crate) registry: Registry<T, ID>,
    pub(crate) index: u32,
}

impl<T, ID: IdTrait> Iterator for IntoIter<T, ID> {
    type Item = (ID, T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.registry.slots.len {
            unsafe {
                let state = self.registry.slots.state_unchecked(self.index);
                let slot = self.registry.slots.value_unchecked(self.index).as_ptr();
                self.index += 1;
                if crate::state_is_occupied::<ID::GenerationBits>(state) {
                    // Mark the slot as free so we don't drop its value when IntoIter drops.
                    let empty_state = crate::empty_state_from_occupied::<ID::GenerationBits>(state);
                    self.registry
                        .slots
                        .set_state_unchecked(self.index - 1, empty_state);
                    return Some((ID::new_unchecked(self.index - 1, state), slot.read()));
                }
            }
        }
        None
    }
}

impl<T, ID: IdTrait> IntoIterator for Registry<T, ID> {
    type Item = (ID, T);
    type IntoIter = IntoIter<T, ID>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}

/// An iterator over `T`. Returned by [`into_values`](crate::Registry::into_values).
#[derive(Debug)]
pub struct IntoValues<T, ID: IdTrait = Id<T>> {
    pub(crate) inner: IntoIter<T, ID>,
}

impl<T, ID: IdTrait> Iterator for IntoValues<T, ID> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_id, val)| val)
    }
}

/// Returned by [`reserve_ids`](crate::Registry::reserve_ids).
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
