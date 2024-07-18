//! a secondary map from IDs to other values
//!
//! This module isn't very fleshed out, but it's meant as an example of how to build these
//! collections. It doesn't depend on any private Riddance APIs, and it doesn't require any unsafe
//! code.

use super::{IdTrait, Registry};
use std::hash::{Hash, Hasher};
use std::mem;

struct IndexOnlyWrapper<ID: IdTrait>(ID);

impl<ID: IdTrait> PartialEq for IndexOnlyWrapper<ID> {
    fn eq(&self, other: &Self) -> bool {
        self.0.index() == other.0.index()
    }
}

impl<ID: IdTrait> Eq for IndexOnlyWrapper<ID> {}

impl<ID: IdTrait> Hash for IndexOnlyWrapper<ID> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.index().hash(state);
    }
}

/// a `hashbrown::HashMap` from IDs to other values, which aggressively removes dangling keys
///
/// # Example
///
/// ```
/// use riddance::{Registry, id::IdTrait, id_map::IdMap};
///
/// let mut registry = Registry::new();
/// let id = registry.insert("foo");
///
/// let mut map = IdMap::new();
/// map.insert(id, 42);
/// assert_eq!(map.get(id), Some(&42));
///
/// // If another ID uses the same index in the registry,
/// // it will also use the same entry in the IdMap.
/// registry.remove(id);
/// let new_id = registry.insert("bar");
/// assert_eq!(id.index(), new_id.index());
/// map.insert(new_id, 99);
/// assert_eq!(map.get(id), None);
/// ```
pub struct IdMap<ID: IdTrait, V> {
    map: hashbrown::HashMap<IndexOnlyWrapper<ID>, V>,
}

impl<ID: IdTrait, V> IdMap<ID, V> {
    pub fn new() -> Self {
        Self {
            map: hashbrown::HashMap::new(),
        }
    }

    pub fn get(&self, id: ID) -> Option<&V> {
        let (k, v) = self.map.get_key_value(&IndexOnlyWrapper(id))?;
        if k.0 == id {
            Some(v)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, id: ID) -> Option<&mut V> {
        match self.map.entry(IndexOnlyWrapper(id)) {
            hashbrown::hash_map::Entry::Occupied(view) => {
                if view.key().0 == id {
                    return Some(view.into_mut());
                }
                if view.key().0.generation() < id.generation() {
                    view.remove();
                }
            }
            hashbrown::hash_map::Entry::Vacant(_) => {}
        }
        None
    }

    pub fn insert(&mut self, id: ID, value: V) -> Option<V> {
        match self.map.raw_entry_mut().from_key(&IndexOnlyWrapper(id)) {
            hashbrown::hash_map::RawEntryMut::Occupied(mut view) => {
                if view.key().0 == id {
                    return Some(mem::replace(view.into_mut(), value));
                }
                if view.key().0.generation() < id.generation() {
                    *view.key_mut() = IndexOnlyWrapper(id);
                    *view.get_mut() = value;
                }
                // If the map key is newer than id, we don't insert value into the map at all.
            }
            hashbrown::hash_map::RawEntryMut::Vacant(view) => {
                view.insert(IndexOnlyWrapper(id), value);
            }
        }
        None
    }

    pub fn remove(&mut self, id: ID) -> Option<V> {
        match self.map.entry(IndexOnlyWrapper(id)) {
            hashbrown::hash_map::Entry::Occupied(view) => {
                if view.key().0 == id {
                    return Some(view.remove());
                }
                if view.key().0.generation() < id.generation() {
                    view.remove();
                }
            }
            hashbrown::hash_map::Entry::Vacant(_) => {}
        }
        None
    }

    pub fn vacuum<T>(&mut self, registry: &Registry<T, ID>) {
        self.map.retain(|k, _v| registry.contains_id(k.0));
    }
}

impl<ID: IdTrait, V> std::ops::Index<ID> for IdMap<ID, V> {
    type Output = V;

    fn index(&self, id: ID) -> &V {
        self.get(id).unwrap()
    }
}

impl<ID: IdTrait, V> std::ops::IndexMut<ID> for IdMap<ID, V> {
    fn index_mut(&mut self, id: ID) -> &mut V {
        self.get_mut(id).unwrap()
    }
}

pub struct IdFlatMap<ID: IdTrait, V> {
    vec: Vec<V>,
    map: hashbrown::HashMap<IndexOnlyWrapper<ID>, usize>,
    reverse_map: hashbrown::HashMap<usize, ID>,
}

impl<ID: IdTrait, V> IdFlatMap<ID, V> {
    pub fn new() -> Self {
        Self {
            vec: Vec::new(),
            map: hashbrown::HashMap::new(),
            reverse_map: hashbrown::HashMap::new(),
        }
    }

    pub fn as_slice(&self) -> &[V] {
        self.vec.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [V] {
        self.vec.as_mut_slice()
    }

    pub fn get(&self, id: ID) -> Option<&V> {
        let (&IndexOnlyWrapper(previous_id), &vec_index) =
            self.map.get_key_value(&IndexOnlyWrapper(id))?;
        if previous_id == id {
            debug_assert!(vec_index < self.vec.len());
            self.vec.get(vec_index)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, id: ID) -> Option<&mut V> {
        match self.map.entry(IndexOnlyWrapper(id)) {
            hashbrown::hash_map::Entry::Occupied(view) => {
                let previous_id = view.key().0;
                let vec_index = *view.get();
                if previous_id == id {
                    debug_assert!(vec_index < self.vec.len());
                    return self.vec.get_mut(vec_index);
                }
                if previous_id.generation() < id.generation() {
                    // The caller's generation is newer than ours. Remove the previous value.
                    view.remove();
                    self.finish_remove(vec_index);
                }
            }
            hashbrown::hash_map::Entry::Vacant(_) => {}
        }
        None
    }

    pub fn insert(&mut self, id: ID, value: V) -> Option<V> {
        match self.map.raw_entry_mut().from_key(&IndexOnlyWrapper(id)) {
            hashbrown::hash_map::RawEntryMut::Occupied(mut view) => {
                let previous_id = view.key().0;
                let vec_index = *view.get();
                if previous_id == id {
                    // An exact match: replace it and and return the old value.
                    return Some(mem::replace(&mut self.vec[vec_index], value));
                }
                if previous_id.generation() < id.generation() {
                    // A match but the caller's generation is newer: replace the old value, update
                    // the generation, and return nothing.
                    self.vec[vec_index] = value;
                    *view.key_mut() = IndexOnlyWrapper(id);
                    self.reverse_map.insert(vec_index, id);
                }
                // If there's a match but the caller's generation is older, do nothing.
            }
            hashbrown::hash_map::RawEntryMut::Vacant(view) => {
                // No match. Append the new value.
                let vec_index = self.vec.len();
                self.vec.push(value);
                view.insert(IndexOnlyWrapper(id), vec_index);
                self.reverse_map.insert(vec_index, id);
            }
        }
        None
    }

    // Finish removing an index that's already been removed from self.map. Clearing the self.map
    // entry needs to be done by the caller before this, because otherwise the borrows clash.
    fn finish_remove(&mut self, vec_index: usize) -> V {
        // Remove the previous value with a swap_remove and update the map and the reverse map.
        // Being able to use swap_remove here is the main reason we have the reverse map.
        let end_index = self.vec.len() - 1;
        let end_id = self.reverse_map[&end_index];
        let value = self.vec.swap_remove(vec_index);
        if vec_index != end_index {
            debug_assert!(self.map.contains_key(&IndexOnlyWrapper(end_id)));
            self.map.insert(IndexOnlyWrapper(end_id), vec_index);
            self.reverse_map.insert(vec_index, end_id);
        }
        self.reverse_map.remove(&end_index);
        value
    }

    pub fn remove(&mut self, id: ID) -> Option<V> {
        match self.map.entry(IndexOnlyWrapper(id)) {
            hashbrown::hash_map::Entry::Occupied(view) => {
                let previous_id = view.key().0;
                let vec_index = *view.get();
                if id.generation() < previous_id.generation() {
                    // Short-circuit if the caller's generation is older.
                    return None;
                }
                // The caller's generation is equal or newer. Remove the previous value.
                view.remove();
                let value = self.finish_remove(vec_index);
                // Only return the previous value if the generations were equal.
                if previous_id == id {
                    Some(value)
                } else {
                    None
                }
            }
            hashbrown::hash_map::Entry::Vacant(_) => {
                // This is also the empty case.
                None
            }
        }
    }

    pub fn vacuum<T>(&mut self, registry: &Registry<T, ID>) {
        // Allocating a Vec for the IDs we're going to remove is kind of a shame, but each removal
        // requires mutable access to all the internal collections, and we can't easily do that
        // while we're iterating over any of them.
        let to_remove: Vec<ID> = self
            .map
            .keys()
            .filter_map(|id| (!registry.contains_id(id.0)).then_some(id.0))
            .collect();
        for id in to_remove {
            self.remove(id);
        }
    }
}

impl<ID: IdTrait, V> std::ops::Index<ID> for IdFlatMap<ID, V> {
    type Output = V;

    fn index(&self, id: ID) -> &V {
        self.get(id).unwrap()
    }
}

impl<ID: IdTrait, V> std::ops::IndexMut<ID> for IdFlatMap<ID, V> {
    fn index_mut(&mut self, id: ID) -> &mut V {
        self.get_mut(id).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_id_map() {
        let mut reg = Registry::new();
        let id0 = reg.insert(());
        let id1 = reg.insert(());
        let id2 = reg.insert(());
        // Remove indexes 1 and 2 and reallocate index 1.
        reg.remove(id2);
        reg.remove(id1);
        let id1_new = reg.insert(());
        assert_eq!(id1_new.index(), 1);

        let mut map = IdMap::new();
        map.insert(id0, "foo");
        map.insert(id1, "bar");
        map.insert(id2, "baz");
        assert_eq!(map.get(id0), Some(&"foo"));
        assert_eq!(map.get(id1), Some(&"bar"));
        assert_eq!(map.get(id2), Some(&"baz"));

        // overwrite "foo" with "FOO"
        assert_eq!(map.insert(id0, "FOO"), Some("foo"));
        assert_eq!(map.get(id0), Some(&"FOO"));

        // `get_mut` (but not `get`) with id1_new drops the id1 key/value pair
        assert_eq!(map.get(id1_new), None);
        assert_eq!(map.get(id1), Some(&"bar"));
        assert_eq!(map.get_mut(id1_new), None);
        assert_eq!(map.get(id1), None);

        // `vacuum` drops id2
        assert_eq!(map.get(id2), Some(&"baz"));
        map.vacuum(&reg);
        assert_eq!(map.get(id2), None);
    }

    #[test]
    fn test_id_flat_map() {
        let mut reg = Registry::new();
        let id0 = reg.insert(());
        let id1 = reg.insert(());
        let id2 = reg.insert(());
        // Remove indexes 1 and 2 and reallocate index 1.
        reg.remove(id2);
        reg.remove(id1);
        let id1_new = reg.insert(());
        assert_eq!(id1_new.index(), 1);

        let mut map = IdFlatMap::new();
        map.insert(id0, "foo");
        map.insert(id1, "bar");
        map.insert(id2, "baz");
        assert_eq!(map.get(id0), Some(&"foo"));
        assert_eq!(map.get(id1), Some(&"bar"));
        assert_eq!(map.get(id2), Some(&"baz"));
        assert_eq!(map.reverse_map.get(&0), Some(&id0));
        assert_eq!(map.reverse_map.get(&1), Some(&id1));
        assert_eq!(map.reverse_map.get(&2), Some(&id2));

        // overwrite "foo" with "FOO"
        assert_eq!(map.insert(id0, "FOO"), Some("foo"));
        assert_eq!(map.get(id0), Some(&"FOO"));

        // `get_mut` (but not `get`) with id1_new drops the id1 key/value pair
        assert_eq!(map.get(id1_new), None);
        assert_eq!(map.get(id1), Some(&"bar"));
        assert_eq!(map.get_mut(id1_new), None);
        assert_eq!(map.get(id1), None);
        assert_eq!(map.reverse_map.get(&0), Some(&id0));
        assert_eq!(map.reverse_map.get(&1), Some(&id2));
        assert_eq!(map.reverse_map.get(&2), None);

        // `vacuum` drops id2
        assert_eq!(map.get(id2), Some(&"baz"));
        map.vacuum(&reg);
        assert_eq!(map.get(id2), None);
        assert_eq!(map.reverse_map.get(&0), Some(&id0));
        assert_eq!(map.reverse_map.get(&1), None);
        assert_eq!(map.reverse_map.get(&2), None);
    }
}
