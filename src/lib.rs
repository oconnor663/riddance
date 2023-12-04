use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::num::{NonZeroU32, NonZeroU64};
use std::ptr::{self, NonNull};
use typenum::Unsigned;

fn static_assert_index_bits<IndexBits: Unsigned>() {
    // These asserts will get compiled out in any case we care about. Someday we'll be able to
    // actually enforce these bounds statically.
    assert!(IndexBits::U32 >= 1);
    assert!(IndexBits::U32 <= 32);
}

fn static_assert_generation_bits<GenerationBits: Unsigned>() {
    // This assert will get compiled out in any case we care about. Someday we'll be able to
    // actually enforce this bound statically.
    assert!(GenerationBits::U32 <= 31);
}

fn debug_assert_high_state_bits_clear<GenerationBits: Unsigned>(state: u32) {
    static_assert_generation_bits::<GenerationBits>();
    if GenerationBits::U32 < 31 {
        debug_assert_eq!(
            0,
            state >> (GenerationBits::U32 + 1),
            "illegal high bits set in state",
        );
    }
}

fn generation_from_state<GenerationBits: Unsigned>(state: u32) -> u32 {
    debug_assert_high_state_bits_clear::<GenerationBits>(state);
    state & ((u32::MAX >> 1) >> (31 - GenerationBits::U32))
}

fn flag_bit_from_state<GenerationBits: Unsigned>(state: u32) -> bool {
    debug_assert_high_state_bits_clear::<GenerationBits>(state);
    (state >> GenerationBits::U32) > 0
}

fn state_is_occupied<GenerationBits: Unsigned>(state: u32) -> bool {
    !flag_bit_from_state::<GenerationBits>(state)
}

fn state_is_empty<GenerationBits: Unsigned>(state: u32) -> bool {
    flag_bit_from_state::<GenerationBits>(state)
}

fn state_is_retired<GenerationBits: Unsigned>(state: u32) -> bool {
    debug_assert_high_state_bits_clear::<GenerationBits>(state);
    state == (u32::MAX >> (31 - GenerationBits::U32))
}

fn empty_state_from_occupied<GenerationBits: Unsigned>(occupied_state: u32) -> u32 {
    debug_assert!(state_is_occupied::<GenerationBits>(occupied_state));
    occupied_state | (1 << GenerationBits::U32)
}

fn occupied_state_from_empty<GenerationBits: Unsigned>(empty_state: u32) -> u32 {
    debug_assert!(state_is_empty::<GenerationBits>(empty_state));
    // Note that the retired state can roll over to generation zero after recycling. This means we
    // need to add 1 *before* we clear the flag bit.
    (empty_state + 1) & !(u32::MAX << GenerationBits::U32)
}

fn word_count_from_state_count<GenerationBits: Unsigned>(state_count: usize) -> usize {
    // NOTE: The number of state bits is GenerationBits + 1.
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => state_count,
        8..=15 => state_count.div_ceil(2),
        4..=7 => state_count.div_ceil(4),
        2..=3 => state_count.div_ceil(8),
        1 => state_count.div_ceil(16),
        0 => state_count.div_ceil(32),
    }
}

fn state_count_from_word_count<GenerationBits: Unsigned>(word_count: usize) -> usize {
    // NOTE: The number of state bits is GenerationBits + 1.
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => word_count,
        8..=15 => word_count.saturating_mul(2),
        4..=7 => word_count.saturating_mul(4),
        2..=3 => word_count.saturating_mul(8),
        1 => word_count.saturating_mul(16),
        0 => word_count.saturating_mul(32),
    }
}

const fn unused_states_in_last_word<GenerationBits: Unsigned>(state_count: usize) -> usize {
    // NOTE: The number of state bits is GenerationBits + 1.
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => 0,
        8..=15 => 1 - (state_count.wrapping_sub(1) % 2),
        4..=7 => 3 - (state_count.wrapping_sub(1) % 4),
        2..=3 => 7 - (state_count.wrapping_sub(1) % 8),
        1 => 15 - (state_count.wrapping_sub(1) % 16),
        0 => 31 - (state_count.wrapping_sub(1) % 32),
    }
}

unsafe fn read_state<GenerationBits: Unsigned>(state_words: *const u32, index: u32) -> u32 {
    let i = index as usize;
    // NOTE: The number of state bits is GenerationBits + 1.
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => *state_words.add(i),
        8..=15 => *(state_words as *const u16).add(i) as u32,
        4..=7 => *(state_words as *const u8).add(i) as u32,
        2..=3 => (*(state_words as *const u8).add(i / 2) as u32 >> (4 * (i % 2))) & 0b1111,
        1 => (*(state_words as *const u8).add(i / 4) as u32 >> (2 * (i % 4))) & 0b11,
        0 => (*(state_words as *const u8).add(i / 8) as u32 >> (i % 8)) & 0b1,
    }
}

unsafe fn write_state<GenerationBits: Unsigned>(state_words: *mut u32, index: u32, state: u32) {
    debug_assert_high_state_bits_clear::<GenerationBits>(state);
    let index = index as usize;
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => *state_words.add(index as usize) = state,
        8..=15 => *(state_words as *mut u16).add(index) = state as u16,
        4..=7 => *(state_words as *mut u8).add(index) = state as u8,
        2..=3 => {
            let entry = &mut *(state_words as *mut u8).add(index / 2);
            *entry &= !(0b1111 << (4 * (index % 2)));
            *entry |= (state as u8 & 0b1111) << (4 * (index % 2));
        }
        1 => {
            let entry = &mut *(state_words as *mut u8).add(index / 4);
            *entry &= !(0b11 << (2 * (index % 4)));
            *entry |= (state as u8 & 0b11) << (2 * (index % 4));
        }
        0 => {
            let entry = &mut *(state_words as *mut u8).add(index / 8);
            *entry &= !(0b1 << (index % 8));
            *entry |= (state as u8 & 0b1) << (index % 8);
        }
    }
}

// This is similar to the private Unique<T> pointer wrapper from the standard library, which is
// used in the implementation of Vec. See https://doc.rust-lang.org/nomicon/phantom-data.html and
// https://github.com/rust-lang/rust/blob/1.74.0/library/core/src/ptr/unique.rs#L37-L45.
#[repr(transparent)]
pub struct Unique<T> {
    pointer: NonNull<T>,
    _phantom: PhantomData<T>,
}

impl<T> Unique<T> {
    fn from_vec(vec: &ManuallyDrop<Vec<T>>) -> Self {
        unsafe {
            Self {
                pointer: NonNull::new_unchecked(vec.as_ptr() as *mut T),
                _phantom: PhantomData,
            }
        }
    }

    fn as_ptr(&self) -> *mut T {
        self.pointer.as_ptr()
    }
}

unsafe impl<T: Send> Send for Unique<T> {}

unsafe impl<T: Sync> Sync for Unique<T> {}

struct Slots<T, GenerationBits: Unsigned> {
    // NOTES:
    // - If a state is 16 bits or less, multiple states get packed into each state word. See
    //   read_state and write_state.
    // - The number of bits in each state is GenerationBits + 1. The flag bit (highest order) is 1
    //   if the slot is free and 0 if it's occupied, *unless* the slot index is greater than or
    //   equal to self.len.
    //
    // INVARIANTS:
    // 1. The entire *capacity* of the state_words Vec is always zero-initialized. That means that
    //    when we increment self.len to allocate a new slot, the slot state is already "occupied,
    //    generation 0".
    // 2. Values are logically initialized if the flag bit in their state is 0 *and* their index is
    //    less than self.len.
    values_ptr: Unique<MaybeUninit<T>>,
    values_cap: usize,
    state_words_ptr: Unique<u32>,
    state_words_cap: usize,
    len: u32,
    _phantom: PhantomData<GenerationBits>,
}

impl<T, GenerationBits: Unsigned> Slots<T, GenerationBits> {
    fn with_capacity(capacity: usize) -> Self {
        if capacity > u32::MAX as usize {
            panic!("requested capacity exceeds u32::MAX");
        }
        // Don't allocate these Vecs directly into ManuallyDrop, because the second allocation
        // might panic, and we don't want to leak the first one in that case. Instead, move them
        // into ManuallyDrop only after both allocations have succeeded.
        let values = Vec::with_capacity(capacity);
        let mut state_words =
            Vec::with_capacity(word_count_from_state_count::<GenerationBits>(capacity));
        // Zero-initialize all the capacity in state_words.
        unsafe {
            ptr::write_bytes(state_words.as_mut_ptr(), 0, state_words.capacity());
        }
        let values = ManuallyDrop::new(values);
        let state_words = ManuallyDrop::new(state_words);
        Self {
            values_cap: values.capacity(),
            values_ptr: Unique::from_vec(&values),
            state_words_cap: state_words.capacity(),
            state_words_ptr: Unique::from_vec(&state_words),
            len: 0,
            _phantom: PhantomData,
        }
    }

    pub fn capacity(&self) -> usize {
        cmp::min(
            self.values_cap,
            state_count_from_word_count::<GenerationBits>(self.state_words_cap),
        )
    }

    unsafe fn state_unchecked(&self, index: u32) -> u32 {
        read_state::<GenerationBits>(self.state_words_ptr.as_ptr(), index)
    }

    fn state(&self, index: u32) -> Option<u32> {
        if index < self.len {
            Some(unsafe { self.state_unchecked(index) })
        } else {
            None
        }
    }

    unsafe fn set_state_unchecked(&mut self, index: u32, state: u32) {
        write_state::<GenerationBits>(self.state_words_ptr.as_ptr(), index, state)
    }

    unsafe fn value_unchecked(&self, index: u32) -> &MaybeUninit<T> {
        &*self.values_ptr.as_ptr().add(index as usize)
    }

    unsafe fn value_mut_unchecked(&mut self, index: u32) -> &mut MaybeUninit<T> {
        &mut *self.values_ptr.as_ptr().add(index as usize)
    }

    unsafe fn reconstitute_values_vec(&self) -> ManuallyDrop<Vec<MaybeUninit<T>>> {
        ManuallyDrop::new(Vec::from_raw_parts(
            self.values_ptr.as_ptr(),
            // These values aren't guaranteed to be initialized, but that's ok because their type
            // is MaybeUninit<T>.
            self.len as usize,
            self.values_cap,
        ))
    }

    // The length of this Vec is always equal to its capacity. All state words are zero-initialized
    // when they're allocated. See with_capacity() and reserve().
    unsafe fn reconstitute_state_words_vec(&self) -> ManuallyDrop<Vec<u32>> {
        ManuallyDrop::new(Vec::from_raw_parts(
            self.state_words_ptr.as_ptr(),
            word_count_from_state_count::<GenerationBits>(self.len as usize),
            self.state_words_cap,
        ))
    }

    fn reserve(&mut self, additional: usize) {
        if (additional > u32::MAX as usize) || self.len.checked_add(additional as u32).is_none() {
            panic!("requested capacity exceeds u32::MAX");
        }
        // Account for unused state bits in the rightmost u32 in use.
        let last_word_cap = unused_states_in_last_word::<GenerationBits>(self.len as usize);
        let additional_words =
            word_count_from_state_count::<GenerationBits>(additional.saturating_sub(last_word_cap));
        unsafe {
            let mut values = self.reconstitute_values_vec();
            let mut state_words = self.reconstitute_state_words_vec();
            // Either of these reserve calls could panic. We need to record any changes made by the
            // first call before we make the second call.
            values.reserve(additional);
            self.values_cap = values.capacity();
            self.values_ptr = Unique::from_vec(&values);
            state_words.reserve(additional_words);
            // Zero-initialize all the *new* capacity in state_words.
            ptr::write_bytes(
                state_words.as_mut_ptr().add(self.state_words_cap),
                0,
                state_words.capacity() - self.state_words_cap,
            );
            self.state_words_cap = state_words.capacity();
            self.state_words_ptr = Unique::from_vec(&state_words);
        }
    }
}

impl<T, GenerationBits: Unsigned> Drop for Slots<T, GenerationBits> {
    fn drop(&mut self) {
        unsafe {
            // These Vecs will drop at end-of-scope.
            let _states = ManuallyDrop::into_inner(self.reconstitute_state_words_vec());
            let _values = ManuallyDrop::into_inner(self.reconstitute_values_vec());
            // If dropping an element panics, we'll unwind out of this loop and skip dropping
            // subsequent elements. The two Vecs above will still drop during unwinding, but any
            // resources owned by individual elements (i.e. a String or a File) will be leaked.
            if mem::needs_drop::<T>() {
                for i in 0..self.len {
                    if state_is_occupied::<GenerationBits>(self.state_unchecked(i)) {
                        self.value_mut_unchecked(i).assume_init_drop();
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
        // Don't allocate these Vecs directly into ManuallyDrop, because the second allocation
        // might panic, and we don't want to leak the first one in that case. Instead, move them
        // into ManuallyDrop only after both allocations have succeeded.
        let needed_capacity: usize = self.len.try_into().unwrap();
        let new_values_vec = Vec::with_capacity(needed_capacity);
        let needed_capacity_state_words =
            word_count_from_state_count::<GenerationBits>(needed_capacity);
        let mut new_state_words_vec = Vec::with_capacity(needed_capacity_state_words);
        // Memcpy all possibly-non-zero state words from self, and then zero-initialize any
        // additional capacity we received.
        unsafe {
            ptr::copy_nonoverlapping(
                self.state_words_ptr.as_ptr(),
                new_state_words_vec.as_mut_ptr(),
                needed_capacity_state_words,
            );
            ptr::write_bytes(
                new_state_words_vec
                    .as_mut_ptr()
                    .add(needed_capacity_state_words),
                0,
                new_state_words_vec.capacity() - needed_capacity_state_words,
            );
        }
        // Assemble the new Slots. Some of the state flag bits will indicate occupied slots, but as
        // long as we only increment len when we actually insert a cloned element, the new Slots
        // will be safe to drop. This arrangement means we won't leak previously cloned elements if
        // one of the clones panics.
        // XXX: These partially-cloned Slots aren't safe to return, because they violates the
        // assumption that not-yet-used slot words are zero-initialized. However, Slots::drop
        // doesn't rely on that assumption.
        let new_values_vec = ManuallyDrop::new(new_values_vec);
        let new_state_words_vec = ManuallyDrop::new(new_state_words_vec);
        let mut cloned_slots = Slots {
            values_cap: new_values_vec.capacity(),
            values_ptr: Unique::from_vec(&new_values_vec),
            state_words_cap: new_state_words_vec.capacity(),
            state_words_ptr: Unique::from_vec(&new_state_words_vec),
            len: 0,
            _phantom: PhantomData,
        };
        // Clone and insert individual elements, keeping cloned_slots.len consistent with the
        // number of elements inserted.
        for i in 0..self.len {
            unsafe {
                if state_is_occupied::<GenerationBits>(self.state_unchecked(i)) {
                    // These clones could panic.
                    let cloned_element = self.value_unchecked(i).assume_init_ref().clone();
                    cloned_slots.value_mut_unchecked(i).write(cloned_element);
                    cloned_slots.len = i + 1;
                }
            }
        }
        // Finally, bump cloned_slots.len to equal the original (if it's not already) and return.
        // This last step restores the invariant we've so far been violating, that all slot words
        // at indexes greater than or equal to slots.len are zero-initialized.
        cloned_slots.len = self.len;
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
        for i in 0..self.len {
            unsafe {
                if state_is_occupied::<GenerationBits>(self.state_unchecked(i)) {
                    list.entry(self.value_unchecked(i).assume_init_ref());
                } else {
                    list.entry(&EmptySlot);
                }
            }
        }
        list.finish()
    }
}

pub trait IdType: Sized + Copy {
    type IndexBits: Unsigned;
    type GenerationBits: Unsigned;

    // The index must be less or equal to max_len (it's equal for the null ID), and the generation
    // must not exceed the range of IndexBits and GenerationBits respectively.
    unsafe fn new_unchecked(index: u32, generation: u32) -> Self;

    fn index(&self) -> u32;

    fn generation(&self) -> u32;

    fn max_len() -> u32 {
        static_assert_index_bits::<Self::IndexBits>();
        // The all-1-bits index is unrepresentable, and the index one below that is reserved for
        // the null ID. For example, if IndexBits=1 then index 1 is unrepresentable, index 0 is
        // null, and max_len is 0. So you can create a Registry with IndexBits=1 if you really want
        // to, but you can't insert anything into it. We avoid assuming anything about
        // GenerationBits, because we support GenerationBits=0.
        (u32::MAX >> (32 - Self::IndexBits::U32)) - 1
    }

    fn max_generation() -> u32 {
        static_assert_generation_bits::<Self::GenerationBits>();
        (u32::MAX >> 1) >> (31 - Self::GenerationBits::U32)
    }

    fn new(index: u32, generation: u32) -> Option<Self> {
        // For the null ID, index == max_len.
        if index > Self::max_len() || generation > Self::max_generation() {
            None
        } else {
            Some(unsafe { Self::new_unchecked(index, generation) })
        }
    }

    fn null() -> Self {
        Self::new(Self::max_len(), 0).unwrap()
    }

    fn is_null(&self) -> bool {
        self.index() == Self::max_len()
    }

    fn debug_format(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "Id {{ index: {}, generation {} }}",
            self.index(),
            self.generation(),
        )
    }
}

// Note that we can't use #[derive(...)] for common traits here, because for example Id should be
// Copy and Ord and Eq even when T isn't. See https://github.com/rust-lang/rust/issues/108894.
#[repr(transparent)]
pub struct Id64<T>(
    NonZeroU64,
    // https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
    PhantomData<fn() -> T>,
);

impl<T> IdType for Id64<T> {
    type IndexBits = typenum::U32;
    type GenerationBits = typenum::U31;

    unsafe fn new_unchecked(index: u32, generation: u32) -> Self {
        // There's 1 unused bit in Id64, and we could shift left by either 31 or 32 bits here. It
        // seems better to shift 32, since that makes the index an aligned 4-byte word, which could
        // be useful to the optimizer somehow.
        let data = ((index as u64) << 32) | generation as u64;
        unsafe {
            Self(
                // Note that adding 1 here makes data=u64::MAX unrepresentable, rather than data=0.
                NonZeroU64::new_unchecked(data + 1),
                PhantomData,
            )
        }
    }

    fn index(&self) -> u32 {
        // Note that subtracting 1 here makes data=u64::MAX unrepresentable, rather than data=0.
        let data = self.0.get() - 1;
        (data >> 32) as u32
    }

    fn generation(&self) -> u32 {
        // Note that subtracting 1 here makes data=u64::MAX unrepresentable, rather than data=0.
        let data = self.0.get() - 1;
        debug_assert_eq!(data & (1 << 31), 0, "this bit should never be set");
        data as u32
    }
}

impl<T> Copy for Id64<T> {}

impl<T> Clone for Id64<T> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<T> std::fmt::Debug for Id64<T>
where
    Self: IdType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.debug_format(f)
    }
}

impl<T> std::hash::Hash for Id64<T> {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.0.hash(state);
    }
}

impl<T> PartialEq for Id64<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id64<T> {}

impl<T> PartialOrd for Id64<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T> Ord for Id64<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

// Note that we can't use #[derive(...)] for common traits here, because for example Id should be
// Copy and Ord and Eq even when T isn't. See https://github.com/rust-lang/rust/issues/108894.
#[repr(transparent)]
pub struct Id32<T, const GENERATION_BITS: usize>(
    NonZeroU32,
    // https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
    PhantomData<fn() -> T>,
);

impl<T, const GENERATION_BITS: usize> IdType for Id32<T, GENERATION_BITS>
where
    typenum::Const<GENERATION_BITS>: typenum::ToUInt,
    typenum::U<GENERATION_BITS>: Unsigned,
    typenum::U32: std::ops::Sub<typenum::U<GENERATION_BITS>>,
    <typenum::U32 as std::ops::Sub<typenum::U<GENERATION_BITS>>>::Output: Unsigned,
{
    type IndexBits = typenum::Diff<typenum::U32, typenum::U<GENERATION_BITS>>;
    type GenerationBits = typenum::U<GENERATION_BITS>;

    unsafe fn new_unchecked(index: u32, generation: u32) -> Self {
        let data = (index << GENERATION_BITS) | generation;
        unsafe {
            Self(
                // Note that adding 1 here makes data=u32::MAX unrepresentable, rather than data=0.
                NonZeroU32::new_unchecked(data + 1),
                PhantomData,
            )
        }
    }

    fn index(&self) -> u32 {
        // Note that subtracting 1 here makes data=u32::MAX unrepresentable, rather than data=0.
        let data = self.0.get() - 1;
        data >> GENERATION_BITS
    }

    fn generation(&self) -> u32 {
        // Note that subtracting 1 here makes data=u32::MAX unrepresentable, rather than data=0.
        let data = self.0.get() - 1;
        data & !(u32::MAX << GENERATION_BITS)
    }
}

impl<T, const GENERATION_BITS: usize> Copy for Id32<T, GENERATION_BITS> {}

impl<T, const GENERATION_BITS: usize> Clone for Id32<T, GENERATION_BITS> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<T, const GENERATION_BITS: usize> std::fmt::Debug for Id32<T, GENERATION_BITS>
where
    Self: IdType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.debug_format(f)
    }
}

impl<T, const GENERATION_BITS: usize> std::hash::Hash for Id32<T, GENERATION_BITS> {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.0.hash(state);
    }
}

impl<T, const GENERATION_BITS: usize> PartialEq for Id32<T, GENERATION_BITS> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T, const GENERATION_BITS: usize> Eq for Id32<T, GENERATION_BITS> {}

impl<T, const GENERATION_BITS: usize> PartialOrd for Id32<T, GENERATION_BITS> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T, const GENERATION_BITS: usize> Ord for Id32<T, GENERATION_BITS> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

pub type Id<T> = Id64<T>;

#[derive(Clone, Debug)]
pub struct Registry<T, ID: IdType = Id<T>> {
    slots: Slots<T, ID::GenerationBits>,
    free_indexes: Vec<u32>,
    retired_indexes: Vec<u32>,
}

impl<T> Registry<T, Id<T>> {
    pub fn new() -> Self {
        Self::with_id_type()
    }
}

impl<T, ID: IdType> Registry<T, ID> {
    pub fn with_id_type() -> Self {
        Self::with_id_type_and_capacity(0)
    }

    pub fn with_id_type_and_capacity(capacity: usize) -> Self {
        static_assert_index_bits::<ID::IndexBits>();
        static_assert_generation_bits::<ID::GenerationBits>();
        Self {
            slots: Slots::with_capacity(capacity),
            free_indexes: Vec::new(),
            retired_indexes: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len as usize - self.free_indexes.len() - self.retired_indexes.len()
    }

    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    // We currently check for two possible violations:
    // 1. The index of an ID should never be larger than the number of slots. The only ways to
    //    violate this rule are to assemble a garbage ID by hand or to use an ID from one Registry
    //    with another of the same time.
    // 2. The generation of an ID should never be newer than its slot. In addition to the cases
    //    above, you can also violate this rule by retaining a dangling ID across a call to
    //    recycle().
    fn debug_best_effort_checks_for_contract_violations(&self, id: ID) {
        debug_assert!(
            id.index() < self.slots.len,
            "ID index is past the highest slot; is it from another Registry?",
        );
        let Some(state) = self.slots.state(id.index()) else {
            return;
        };
        debug_assert!(
            id.generation() <= generation_from_state::<ID::GenerationBits>(state),
            "ID generation is newer than its slot; did it dangle across a recycle()?",
        );
    }

    pub fn get(&self, id: ID) -> Option<&T> {
        self.debug_best_effort_checks_for_contract_violations(id);
        let Some(state) = self.slots.state(id.index()) else {
            return None;
        };
        // This comparison can only succeed if the generation matches and the flag bit is 0.
        if state != id.generation() {
            return None;
        }
        unsafe { Some(self.slots.value_unchecked(id.index()).assume_init_ref()) }
    }

    pub fn get_mut(&mut self, id: ID) -> Option<&mut T> {
        self.debug_best_effort_checks_for_contract_violations(id);
        let Some(state) = self.slots.state(id.index()) else {
            return None;
        };
        // This comparison can only succeed if the generation matches and the flag bit is 0.
        if state != id.generation() {
            return None;
        }
        unsafe { Some(self.slots.value_mut_unchecked(id.index()).assume_init_mut()) }
    }

    pub fn contains_id(&self, id: ID) -> bool {
        self.get(id).is_some()
    }

    #[must_use]
    pub fn insert(&mut self, value: T) -> ID {
        // Reuse a free slot if there are any.
        if let Some(index) = self.free_indexes.pop() {
            unsafe {
                let empty_state = self.slots.state_unchecked(index);
                // Note that if this slot was previously retired and has been recycled, the new
                // generation will wrap back to 0.
                let occupied_state = occupied_state_from_empty::<ID::GenerationBits>(empty_state);
                self.slots.set_state_unchecked(index, occupied_state);
                self.slots.value_mut_unchecked(index).write(value);
                // The flag bit is zero for the occupied state, so its value is equal to the
                // generation of the new ID.
                return ID::new_unchecked(index, occupied_state);
            }
        }
        // Panic if the index space is full.
        assert!(self.slots.len < ID::max_len(), "all slots occupied");
        // Reserve a slot. New state capacity is zero-initialized, so we only need to initialize
        // the value here.
        self.slots.reserve(1);
        unsafe {
            self.slots.value_mut_unchecked(self.slots.len).write(value);
            let new_id = ID::new_unchecked(self.slots.len, 0);
            self.slots.len += 1;
            new_id
        }
    }

    pub fn remove(&mut self, id: ID) -> Option<T> {
        self.debug_best_effort_checks_for_contract_violations(id);
        let Some(state) = self.slots.state(id.index()) else {
            return None;
        };
        if state != id.generation() {
            return None;
        }
        // The ID generation matches the slot state, which means the slot is occupied. Add this
        // slot to the free list or the retired list (which could panic), set the free bit in its
        // state, and move out its value.
        let empty_state = empty_state_from_occupied::<ID::GenerationBits>(state);
        if state_is_retired::<ID::GenerationBits>(empty_state) {
            self.retired_indexes.push(id.index());
        } else {
            self.free_indexes.push(id.index());
        }
        unsafe {
            self.slots.set_state_unchecked(id.index(), empty_state);
            Some(self.slots.value_unchecked(id.index()).assume_init_read())
        }
    }

    /// Mark all retired slots as free. You **must** delete all dangling IDs (or replace them with
    /// [`null`]) before calling this function.
    ///
    /// When you call [`remove`] Registry::remove on an ID, that ID and any copies of it become
    /// "dangling". Calling [`get`] or [`get_mut`] on a dangling ID is normally guaranteed to
    /// return `None`, and [`contains_id`] is normally guaranteed to return `false`. To provide
    /// these guarantees, `Registry` "retires" a slot when its generation reaches the maximum.
    ///
    /// When you call `recycle`, all of these retired slots are made available for new insertions,
    /// and their generation starts back over at 0. If you retain any dangling IDs across the call
    /// to `retire`, they could collide with newly issued IDs, and calls to [`get`], [`get_mut`],
    /// and [`contains_id`] can return confusing results. This behavior is memory-safe, but it's a
    /// logic error, similar to the logic errors that can arise if you modify a key after it's been
    /// inserted into a [`HashMap`].
    ///
    /// # Panics
    ///
    /// `Registry` makes a best effort to detect violations of this rule. _Any_ method on
    /// `Registry` may panic if it sees an ID generation that's newer than the corresponding slot.
    ///
    /// [`null`]: Id::null
    /// [`remove`]: Registry::remove
    /// [`get`]: Registry::get
    /// [`get_mut`]: Registry::get_mut
    /// [`contains_id`]: Registry::contains_id
    /// [`HashMap`]: https://doc.rust-lang.org/std/collections/struct.HashMap.html
    pub fn recycle(&mut self) {
        // This clears retired_indexes.
        self.free_indexes.append(&mut self.retired_indexes);
    }
}

impl<T, ID: IdType> std::ops::Index<ID> for Registry<T, ID> {
    type Output = T;

    fn index(&self, id: ID) -> &T {
        self.get(id).unwrap()
    }
}

impl<T, ID: IdType> std::ops::IndexMut<ID> for Registry<T, ID> {
    fn index_mut(&mut self, id: ID) -> &mut T {
        self.get_mut(id).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU8;

    #[repr(transparent)]
    pub struct Id8<T, const GENERATION_BITS: usize>(
        NonZeroU8,
        // https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
        PhantomData<fn() -> T>,
    );

    impl<T, const GENERATION_BITS: usize> IdType for Id8<T, GENERATION_BITS>
    where
        typenum::Const<GENERATION_BITS>: typenum::ToUInt,
        typenum::U<GENERATION_BITS>: Unsigned,
        typenum::U8: std::ops::Sub<typenum::U<GENERATION_BITS>>,
        <typenum::U8 as std::ops::Sub<typenum::U<GENERATION_BITS>>>::Output: Unsigned,
    {
        type IndexBits = typenum::Diff<typenum::U8, typenum::U<GENERATION_BITS>>;
        type GenerationBits = typenum::U<GENERATION_BITS>;

        unsafe fn new_unchecked(index: u32, generation: u32) -> Self {
            let data = ((index as u8) << GENERATION_BITS) | generation as u8;
            unsafe {
                Self(
                    // Note that adding 1 here makes data=u8::MAX unrepresentable, rather than data=0.
                    NonZeroU8::new_unchecked(data + 1),
                    PhantomData,
                )
            }
        }

        fn index(&self) -> u32 {
            // Note that subtracting 1 here makes data=u8::MAX unrepresentable, rather than data=0.
            let data = (self.0.get() - 1) as u32;
            data >> GENERATION_BITS
        }

        fn generation(&self) -> u32 {
            // Note that subtracting 1 here makes data=u8::MAX unrepresentable, rather than data=0.
            let data = (self.0.get() - 1) as u32;
            data & !(u32::MAX << GENERATION_BITS)
        }
    }

    impl<T, const GENERATION_BITS: usize> Copy for Id8<T, GENERATION_BITS> {}

    impl<T, const GENERATION_BITS: usize> Clone for Id8<T, GENERATION_BITS> {
        fn clone(&self) -> Self {
            Self(self.0, PhantomData)
        }
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
        let result = std::panic::catch_unwind(|| registry1.contains_id(id));
        if cfg!(debug_assertions) {
            // In debug mode, we detect the contract violation and panic.
            result.unwrap_err();
        } else {
            // In release mode, we don't check for contract violations.
            assert_eq!(result.unwrap(), false);
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
        let id2_result = std::panic::catch_unwind(|| registry.contains_id(id2));
        if cfg!(debug_assertions) {
            // In debug mode, we detect the contract violation and panic.
            id2_result.unwrap_err();
        } else {
            // In release mode, we don't check for contract violations.
            assert_eq!(id2_result.unwrap(), false);
        }
    }

    #[test]
    #[should_panic]
    fn test_gbits_8_panics() {
        Registry::<(), Id8<(), 8>>::with_id_type();
    }

    fn full_registry<const GENERATION_BITS: usize>() -> Registry<(), Id8<(), GENERATION_BITS>>
    where
        Id8<(), GENERATION_BITS>: IdType,
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
        assert_eq!(registry.slots.state(0), Some(0));
        assert_eq!(registry.slots.state(1), Some(0));
        assert_eq!(e0.index(), 0);
        assert_eq!(e0.generation(), 0);
        assert_eq!(e1.index(), 1);
        assert_eq!(e1.generation(), 0);
        registry.remove(e0);
        assert_eq!(registry.slots.state(0), Some(1));
        assert_eq!(registry.slots.state(1), Some(0));
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0]);
        registry.remove(e1);
        assert_eq!(registry.slots.state(0), Some(1));
        assert_eq!(registry.slots.state(1), Some(1));
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0, 1]);
        registry.recycle();
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
        assert_eq!(registry.slots.len, 1);
        assert_eq!(registry.slots.state(0), Some(0b00));
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, []);

        registry.remove(id);
        assert_eq!(registry.slots.len, 1);
        assert_eq!(registry.slots.state(0), Some(0b10));
        assert_eq!(registry.free_indexes, [0]);
        assert_eq!(registry.retired_indexes, []);

        id = registry.insert(());
        assert_eq!(id.index(), 0);
        assert_eq!(id.generation(), 1);
        assert_eq!(registry.slots.len, 1);
        assert_eq!(registry.slots.state(0), Some(0b01));
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, []);

        registry.remove(id);
        assert_eq!(registry.slots.len, 1);
        assert_eq!(registry.slots.state(0), Some(0b11));
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0]);

        id = registry.insert(());
        assert_eq!(id.index(), 1);
        assert_eq!(id.generation(), 0);
        assert_eq!(registry.slots.len, 2);
        assert_eq!(registry.slots.state(0), Some(0b11));
        assert_eq!(registry.slots.state(1), Some(0b00));
        assert_eq!(registry.free_indexes, []);
        assert_eq!(registry.retired_indexes, [0]);
    }

    // This test does a few asserts, but its real purpose is to run under Miri and make sure we
    // don't leak memory or touch any freed memory.
    fn do_cloning_and_dropping<ID: IdType>() {
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
            word_count_from_state_count::<ID::GenerationBits>(registry.slots.len as usize);
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
}
