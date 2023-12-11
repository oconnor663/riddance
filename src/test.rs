use super::*;
use crate::id::{Id32, Id64, Id8};
use std::panic;

#[track_caller]
fn should_panic<T>(f: impl FnOnce() -> T) {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(f));
    assert!(result.is_err(), "panic expected but missing");
    println!("↑↑↑↑↑ expected panic ↑↑↑↑↑\n");
}

#[test]
fn test_state_helpers() {
    for state in 0..8 {
        if state < 4 {
            assert!(state_is_occupied::<typenum::U2>(state));
        } else {
            assert!(state_is_empty::<typenum::U2>(state));
        }
        assert_eq!(state_is_retired::<typenum::U2>(state), state == 0b111);
    }

    #[cfg(debug_assertions)]
    should_panic(|| debug_assert_high_state_bits_clear::<typenum::U2>(8));
}

#[test]
fn test_insert_and_remove() {
    let mut registry = Registry::new();
    let e1 = registry.insert("foo".to_string());
    let e2 = registry.insert("bar".to_string());
    assert!(registry.contains_id(e1));
    assert!(registry.contains_id(e2));
    assert_eq!(registry.get(e1), Some(&"foo".to_string()));
    assert_eq!(registry.get(e2), Some(&"bar".to_string()));
    assert_eq!(registry.get_mut(e1), Some(&mut "foo".to_string()));
    assert_eq!(registry.get_mut(e2), Some(&mut "bar".to_string()));
    assert_eq!(&registry[e1], "foo");
    assert_eq!(&registry[e2], "bar");
    assert_eq!(&mut registry[e1], "foo");
    assert_eq!(&mut registry[e2], "bar");

    assert_eq!(registry.remove(e1), Some("foo".into()));
    assert!(!registry.contains_id(e1));
    assert!(registry.contains_id(e2));
    assert_eq!(registry.get(e1), None);
    assert_eq!(registry.get(e2), Some(&"bar".to_string()));
    assert_eq!(registry.get_mut(e1), None);
    assert_eq!(registry.get_mut(e2), Some(&mut "bar".to_string()));
    assert_eq!(&registry[e2], "bar");
    assert_eq!(&mut registry[e2], "bar");

    assert_eq!(registry.remove(e2), Some("bar".into()));
    assert!(!registry.contains_id(e1));
    assert!(!registry.contains_id(e2));
    assert_eq!(registry.get(e1), None);
    assert_eq!(registry.get(e2), None);
    assert_eq!(registry.get_mut(e1), None);
    assert_eq!(registry.get_mut(e2), None);
}

#[test]
fn test_out_of_bounds_index_panics() {
    let registry1 = Registry::new();
    let mut registry2 = Registry::new();
    let id = registry2.insert(());
    if cfg!(debug_assertions) {
        // In debug mode, we detect the contract violation and panic.
        should_panic(|| registry1.contains_id(id));
    } else {
        // In release mode, we don't check for contract violations.
        assert!(!registry1.contains_id(id));
    }
}

#[test]
fn test_dangling_id_after_recycle_panics() {
    let mut registry = Registry::<(), Id8<(), 1>>::with_id_type();

    let id1 = registry.insert(());
    assert_eq!(id1.index(), 0);
    assert_eq!(id1.generation(), 0);
    registry.remove(id1);

    let id2 = registry.insert(());
    assert_eq!(id2.index(), 0);
    assert_eq!(id2.generation(), 1);
    registry.remove(id2);

    assert_eq!(registry.free_indexes.len(), 0);
    assert_eq!(registry.retired_indexes.len(), 1);
    registry.recycle();
    assert_eq!(registry.free_indexes.len(), 1);
    assert_eq!(registry.retired_indexes.len(), 0);

    let id3 = registry.insert(());
    assert_eq!(id3.index(), 0);
    assert_eq!(id3.generation(), 0);
    // Looking up id1 gives a false positive.
    assert!(registry.contains_id(id1));
    // Looking up id2 panics in debug mode.
    if cfg!(debug_assertions) {
        // In debug mode, we detect the contract violation and panic.
        should_panic(|| registry.contains_id(id2));
    } else {
        // In release mode, we don't check for contract violations.
        assert!(!registry.contains_id(id2));
    }
}

#[test]
#[should_panic]
fn test_gbits_8_panics() {
    Registry::<(), Id8<(), 8>>::with_id_type();
}

fn full_registry<const GENERATION_BITS: usize>() -> Registry<(), Id8<(), GENERATION_BITS>>
where
    Id8<(), GENERATION_BITS>: IdTrait,
{
    let mut registry = Registry::<(), Id8<(), GENERATION_BITS>>::with_id_type();
    // One index is unrepresentable, and another is reserved for the null ID.
    let max_len = (1 << (8 - GENERATION_BITS)) - 2;
    assert_eq!(max_len, Id8::<(), GENERATION_BITS>::max_len());
    for _ in 0..max_len {
        _ = registry.insert(());
    }
    registry
}

#[test]
fn test_fill_slots() {
    full_registry::<0>();
    full_registry::<1>();
    full_registry::<2>();
    full_registry::<3>();
    full_registry::<4>();
    full_registry::<5>();
    full_registry::<6>();
    full_registry::<7>();
}

#[test]
#[should_panic]
fn test_overfill_slots_0() {
    _ = full_registry::<0>().insert(());
}

#[test]
#[should_panic]
fn test_overfill_slots_4() {
    _ = full_registry::<4>().insert(());
}

#[test]
#[should_panic]
fn test_overfill_slots_7() {
    _ = full_registry::<7>().insert(());
}

#[test]
#[should_panic]
fn test_gbits_too_large() {
    full_registry::<8>();
}

#[test]
fn test_gbits_0() {
    // With GBITS=0, there's only 1 possible generation, so every freed Id is immediately
    // retired.
    let mut registry = Registry::<(), Id8<(), 0>>::with_id_type();
    let e0 = registry.insert(());
    let e1 = registry.insert(());
    assert_eq!(registry.len(), 2);
    assert_eq!(registry.slots.state(0), Some(0));
    assert_eq!(registry.slots.state(1), Some(0));
    assert_eq!(e0.index(), 0);
    assert_eq!(e0.generation(), 0);
    assert_eq!(e1.index(), 1);
    assert_eq!(e1.generation(), 0);
    registry.remove(e0);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.slots.state(0), Some(1));
    assert_eq!(registry.slots.state(1), Some(0));
    assert_eq!(registry.free_indexes, []);
    assert_eq!(registry.retired_indexes, [0]);
    registry.remove(e1);
    assert_eq!(registry.len(), 0);
    assert_eq!(registry.slots.state(0), Some(1));
    assert_eq!(registry.slots.state(1), Some(1));
    assert_eq!(registry.free_indexes, []);
    assert_eq!(registry.retired_indexes, [0, 1]);
    registry.recycle();
    assert_eq!(registry.len(), 0);
    assert_eq!(registry.slots.state(0), Some(1));
    assert_eq!(registry.slots.state(1), Some(1));
    assert_eq!(registry.free_indexes, [0, 1]);
    assert_eq!(registry.retired_indexes, []);
}

#[test]
fn test_gbits_1() {
    // With GBITS=1, there are 2 possible generations. Confirm that we get a new slot on the
    // 3rd allocate/free cycle.
    let mut registry = Registry::<(), Id8<(), 1>>::with_id_type();
    assert_eq!(registry.slots.len, 0);

    let mut id = registry.insert(());
    assert_eq!(id.index(), 0);
    assert_eq!(id.generation(), 0);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.slots.len, 1);
    assert_eq!(registry.slots.state(0), Some(0b00));
    assert_eq!(registry.free_indexes, []);
    assert_eq!(registry.retired_indexes, []);

    registry.remove(id);
    assert_eq!(registry.len(), 0);
    assert_eq!(registry.slots.len, 1);
    assert_eq!(registry.slots.state(0), Some(0b10));
    assert_eq!(registry.free_indexes, [0]);
    assert_eq!(registry.retired_indexes, []);

    id = registry.insert(());
    assert_eq!(id.index(), 0);
    assert_eq!(id.generation(), 1);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.slots.len, 1);
    assert_eq!(registry.slots.state(0), Some(0b01));
    assert_eq!(registry.free_indexes, []);
    assert_eq!(registry.retired_indexes, []);

    registry.remove(id);
    assert_eq!(registry.len(), 0);
    assert_eq!(registry.slots.len, 1);
    assert_eq!(registry.slots.state(0), Some(0b11));
    assert_eq!(registry.free_indexes, []);
    assert_eq!(registry.retired_indexes, [0]);

    id = registry.insert(());
    assert_eq!(id.index(), 1);
    assert_eq!(id.generation(), 0);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.slots.len, 2);
    assert_eq!(registry.slots.state(0), Some(0b11));
    assert_eq!(registry.slots.state(1), Some(0b00));
    assert_eq!(registry.free_indexes, []);
    assert_eq!(registry.retired_indexes, [0]);
}

// This test does a few asserts, but its real purpose is to run under Miri and make sure we
// don't leak memory or touch any freed memory.
fn do_cloning_and_dropping<ID: IdTrait>() {
    const NUM_INSERTIONS: usize = 100;
    // We're going to do 100 insertions but also 50 removals, so we need 6 index bits.
    let mut registry = Registry::<String, ID>::with_id_type();
    let mut ids = Vec::new();
    for i in 0..NUM_INSERTIONS {
        dbg!(i);
        let new_id = registry.insert(i.to_string());
        ids.push(new_id);
        // Remove half of the strings we insert. Removing the same ID twice is fine.
        let remove_result = registry.remove(ids[i / 2]);
        assert_eq!(remove_result.is_some(), i % 2 == 0);
    }

    // Clone the registry.
    let cloned = registry.clone();

    // Assert that slots.len is equal in the original and the clone, and also double check that
    // any slot words past slots.len are zero in both. We rely on this invariant when
    // allocating new slots.
    assert_eq!(registry.slots.len, cloned.slots.len);
    let first_unused_slot_word =
        word_count_from_state_count::<ID::GenerationBits>(registry.slots.len) as usize;
    unsafe {
        // Note that the state_words_cap of the original might not equal that of the clone,
        // because it's up to the global allocator whether we get more capacity than we ask
        // for, and it might not be deterministic.
        for i in first_unused_slot_word..registry.slots.state_words_cap {
            assert_eq!(0, *registry.slots.state_words_ptr.as_ptr().add(i));
        }
        for i in first_unused_slot_word..cloned.slots.state_words_cap {
            assert_eq!(0, *cloned.slots.state_words_ptr.as_ptr().add(i));
        }
    }

    // Read all the elements of both the original and the clone, to try to turn any bugs into
    // Miri errors.
    for &id in &ids {
        if let Some(s) = registry.get(id) {
            assert_eq!(
                s.parse::<usize>().unwrap(),
                cloned[id].parse::<usize>().unwrap()
            );
        } else {
            assert!(cloned.get(id).is_none());
        }
    }
}

#[test]
fn test_cloning_and_dropping() {
    // We're going to do 100 insertions but also 50 removals, so we need at least 6 index bits.
    do_cloning_and_dropping::<Id8<String, 2>>();
    do_cloning_and_dropping::<Id32<String, 10>>();
    do_cloning_and_dropping::<Id64<String>>();
}

#[test]
fn test_empty_clone() {
    // Allocate an ID but then remove it before cloning. slots.len should be 1 in both the
    // original and the clone.
    let mut registry = Registry::new();
    let id = registry.insert(());
    _ = registry.remove(id);
    let clone = registry.clone();
    assert_eq!(registry.slots.len, 1);
    assert_eq!(clone.slots.len, 1);
}

#[test]
fn test_id_sizes() {
    assert_eq!(1, mem::size_of::<Id8<(), 4>>());
    assert_eq!(4, mem::size_of::<Id32<(), 10>>());
    assert_eq!(8, mem::size_of::<Id64<()>>());
}

#[test]
fn test_with_capacity() {
    for cap in 0..100 {
        let registry = Registry::<String>::with_capacity(cap);
        assert!(registry.capacity() >= cap);
    }
}

#[test]
fn test_pending_reservation_panics() {
    let mut registry = Registry::<String>::new();
    _ = registry.reserve_ids(1);

    // These methods don't panic.
    let null = Id::null();
    registry.len();
    registry.capacity();
    registry.contains_id(null);
    registry.get(null);
    registry.get_mut(null);

    // These methods do panic.
    should_panic(|| registry.insert(String::new()));
    should_panic(|| registry.remove(null));
    should_panic(|| registry.recycle());
    should_panic(|| registry.clone());
    should_panic(|| registry.fill_empty_reservation(null, "foo".into()));
}

#[test]
fn test_reserve_ids() {
    let mut registry = Registry::<String>::new();
    let id0_old = registry.insert("old".into());
    let id1 = registry.insert("old".into());
    registry.remove(id0_old);

    let mut reservation = registry.reserve_ids(3);

    let id0 = reservation.next().unwrap();
    assert_eq!(id0.index(), 0);
    assert_eq!(id0.generation(), 1);
    assert!(registry.get(id0).is_none());

    let id2 = reservation.next().unwrap();
    assert_eq!(id2.index(), 2);
    assert_eq!(id2.generation(), 0);
    assert!(registry.get(id2).is_none());

    let id3 = reservation.next().unwrap();
    assert_eq!(id3.index(), 3);
    assert_eq!(id3.generation(), 0);
    assert!(registry.get(id3).is_none());

    assert!(reservation.next().is_none());

    assert!(registry.get(id0).is_none());
    assert_eq!(registry.get(id1).unwrap(), "old");
    assert!(registry.get(id2).is_none());
    assert!(registry.get(id3).is_none());
    registry.fill_pending_reservations_with(|| "new".into());
    assert_eq!(registry.get(id0).unwrap(), "new");
    assert_eq!(registry.get(id1).unwrap(), "old");
    assert_eq!(registry.get(id2).unwrap(), "new");
    assert_eq!(registry.get(id3).unwrap(), "new");
}

#[test]
fn test_empty_reservations() {
    let mut registry = Registry::<String>::new();
    let id0_old = registry.insert("old".into());
    let id1 = registry.insert("old".into());
    registry.remove(id0_old);
    assert!(!registry.contains_id(id0_old));
    assert!(registry.contains_id(id1));

    // Reserve a couple of IDs individually.
    let id0 = registry.reserve_id();
    assert_eq!(id0.index(), 0);
    assert_eq!(id0.generation(), 1);
    assert!(!registry.contains_id(id0));
    let id2 = registry.reserve_id();
    assert_eq!(id2.index(), 2);
    assert_eq!(id2.generation(), 0);
    assert!(!registry.contains_id(id2));

    // Now there are pending reservations, and methods like insert will panic. See also
    // test_pending_reservation_panics.
    should_panic(|| registry.insert(String::new()));

    // Allocate empty slots for those IDs.
    registry.allocate_empty_reservations();
    assert!(!registry.contains_id(id0));
    assert!(!registry.contains_id(id2));

    // Now there are no pending reservations, and a regular insert can succeed.
    let id3 = registry.insert("new".into());
    assert!(registry.contains_id(id3));

    // Fill the empty slots.
    registry.fill_empty_reservation(id0, "new".into()).unwrap();
    assert!(registry.contains_id(id0));
    assert!(!registry.contains_id(id2));
    registry.fill_empty_reservation(id2, "new".into()).unwrap();
    assert!(registry.contains_id(id0));
    assert!(registry.contains_id(id2));

    // Trying to fill IDs that aren't reserved should fail.
    let error = registry
        .fill_empty_reservation(id0_old, "blarg".into())
        .unwrap_err();
    assert_eq!(error.kind, FillEmptyReservationErrorKind::Dangling);
    for id in [id0, id1, id2, id3] {
        let error = registry
            .fill_empty_reservation(id, "blarg".into())
            .unwrap_err();
        assert_eq!(error.kind, FillEmptyReservationErrorKind::Exists);
        assert_eq!(error.into_inner(), "blarg");
    }
    registry.remove(id0);
    dbg!(id0.generation());
    let error = registry
        .fill_empty_reservation(id0, "blarg".into())
        .unwrap_err();
    assert_eq!(error.kind, FillEmptyReservationErrorKind::Dangling);

    // The following cases trip debug assertions before returning an error, so we only run them
    // in release mode.
    #[cfg(not(debug_assertions))]
    {
        // An ID with a generation that's newer than its slot (and not a reservation for that
        // slot) can only be produced by retaining a dangling ID across a call to recycle, or
        // by handcrafting a bad ID. Here we do it by hancrafting.

        // For an empty slot, generation + 1 is a valid reservation. Test +2 here.
        let too_new_id = Id::new(id0.index(), id0.generation() + 2).unwrap();
        let error = registry
            .fill_empty_reservation(too_new_id, "blarg".into())
            .unwrap_err();
        assert_eq!(error.kind, FillEmptyReservationErrorKind::GenerationTooNew);

        // For an occupied slot, generation + 1 should be impossible.
        let too_new_id = Id::new(id1.index(), id1.generation() + 1).unwrap();
        let error = registry
            .fill_empty_reservation(too_new_id, "blarg".into())
            .unwrap_err();
        assert_eq!(error.kind, FillEmptyReservationErrorKind::GenerationTooNew);

        // An ID with an out-of-bounds index should never be possible other than by
        // handcrafting it.
        let out_of_bounds_id = Id::new(id3.index() + 1, 0).unwrap();
        let error = registry
            .fill_empty_reservation(out_of_bounds_id, "blarg".into())
            .unwrap_err();
        assert_eq!(error.kind, FillEmptyReservationErrorKind::IndexOutOfBounds);
    }
}

#[test]
fn test_fill_pending_reservations() {
    let mut registry = Registry::<i32>::with_id_type();
    let id0_old = registry.insert(99);
    let id1_old = registry.insert(99);
    registry.remove(id1_old);
    registry.remove(id0_old);
    let mut iterator = registry.reserve_ids(3);
    let id0 = iterator.next().unwrap();
    let id1 = iterator.next().unwrap();
    let mut iterator = registry.reserve_ids(3);
    let id2 = iterator.next().unwrap();
    let id3 = iterator.next().unwrap();
    registry.fill_pending_reservations(42);
    for id in [id0, id1, id2, id3] {
        assert_eq!(registry[id], 42);
    }
}

#[test]
fn test_fill_pending_reservations_with_id() {
    type ID = Id<()>;
    let mut registry = Registry::<ID, ID>::with_id_type();
    let id0 = registry.reserve_id();
    let mut iterator = registry.reserve_ids(3);
    let id1 = iterator.next().unwrap();
    let id2 = iterator.next().unwrap();
    let id3 = iterator.next().unwrap();
    registry.fill_pending_reservations_with_id(|id| id);
    for id in [id0, id1, id2, id3] {
        assert_eq!(id, registry[id]);
    }
}
