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

        // overwrite "foo" with "fooo"
        assert_eq!(map.insert(id0, "foo"), Some("foo"));

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
}
