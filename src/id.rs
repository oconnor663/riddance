//! additional ID types

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::{NonZeroU32, NonZeroU64};
use typenum::Unsigned;

#[cfg(test)]
use std::num::NonZeroU8;

pub trait IdTrait: Sized + Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Hash {
    type IndexBits: Unsigned;
    type GenerationBits: Unsigned;

    // The index must be less or equal to max_len (it's equal for the null ID), and the generation
    // must not exceed the range of IndexBits and GenerationBits respectively.
    unsafe fn new_unchecked(index: u32, generation: u32) -> Self;

    fn index(&self) -> u32;

    fn generation(&self) -> u32;

    fn max_len() -> u32 {
        crate::static_assert_index_bits::<Self::IndexBits>();
        // The all-1-bits index is unrepresentable, and the index one below that is reserved for
        // the null ID. For example, if IndexBits=1 then index 1 is unrepresentable, index 0 is
        // null, and max_len is 0. So you can create a Registry with IndexBits=1 if you really want
        // to, but you can't insert anything into it. We avoid assuming anything about
        // GenerationBits, because we support GenerationBits=0.
        (u32::MAX >> (32 - Self::IndexBits::U32)) - 1
    }

    fn max_generation() -> u32 {
        crate::static_assert_generation_bits::<Self::GenerationBits>();
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

/// This is what the [`Id`](crate::Id) type alias at the crate root points to.
// Note that we can't use #[derive(...)] for common traits here, because for example Id should be
// Copy and Ord and Eq even when T isn't. See https://github.com/rust-lang/rust/issues/108894.
#[repr(transparent)]
pub struct Id64<T>(
    NonZeroU64,
    // https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
    PhantomData<fn() -> T>,
);

impl<T> IdTrait for Id64<T> {
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
    Self: IdTrait,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.debug_format(f)
    }
}

impl<T> Hash for Id64<T> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
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

/// A smaller ID type for caller who want to save space.
///
/// Using this ID type requires picking a value for the `GENERATION_BITS` const parameter, which
/// must be between 0 and 31 inclusive. The number of index bits is 32 minus `GENERATION_BITS`.
/// Setting `GENERATION_BITS` to 0 means that any removed IDs are immediately retired (see
/// [`recycle`](crate::Registry::recycle)). Setting it to 31 means that the only possible ID is the
/// [`null`](IdTrait::null) ID, and any call to [`insert`](crate::Registry::insert) will panic.
/// Most callers will probably want a value somewhere in the middle, like 10 or 12. But in general
/// you're using this ID type because you know exactly what your application needs, so who am I to
/// tell you what to do? :)
///
/// # Example
///
/// ```
/// # fn main() {
/// use riddance::{id::Id32, Registry};
///
/// struct Person {
///     name: String,
///     friends: Vec<PersonId>,
/// }
///
/// // GENERATION_BITS = 12, which allows up to 2^20 ≈ 1 million possible elements.
/// type PersonId = Id32<Person, 12>;
///
/// let mut people = Registry::<Person, PersonId>::with_id_type();
/// let alice_id = people.insert(Person { name: "Alice".into(), friends: vec![] });
/// let bob_id = people.insert(Person { name: "Bob".into(), friends: vec![] });
/// people[alice_id].friends.push(bob_id);
/// people[bob_id].friends.push(alice_id);
///
/// people.remove(bob_id);
/// assert!(people.get(alice_id).is_some());
/// assert!(people.get(bob_id).is_none());
/// # }
/// ```
// Note that we can't use #[derive(...)] for common traits here, because for example Id should be
// Copy and Ord and Eq even when T isn't. See https://github.com/rust-lang/rust/issues/108894.
#[repr(transparent)]
pub struct Id32<T, const GENERATION_BITS: usize>(
    NonZeroU32,
    // https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
    PhantomData<fn() -> T>,
);

impl<T, const GENERATION_BITS: usize> IdTrait for Id32<T, GENERATION_BITS>
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
    Self: IdTrait,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.debug_format(f)
    }
}

impl<T, const GENERATION_BITS: usize> Hash for Id32<T, GENERATION_BITS> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
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

#[cfg(test)]
pub use id8::Id8;

#[cfg(test)]
mod id8 {
    use super::*;

    /// Id8 is only for testing.
    #[repr(transparent)]
    pub struct Id8<T, const GENERATION_BITS: usize>(
        NonZeroU8,
        // https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
        PhantomData<fn() -> T>,
    );

    impl<T, const GENERATION_BITS: usize> IdTrait for Id8<T, GENERATION_BITS>
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

    impl<T, const GENERATION_BITS: usize> Hash for Id8<T, GENERATION_BITS> {
        fn hash<H>(&self, state: &mut H)
        where
            H: Hasher,
        {
            self.0.hash(state);
        }
    }

    impl<T, const GENERATION_BITS: usize> PartialEq for Id8<T, GENERATION_BITS> {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl<T, const GENERATION_BITS: usize> Eq for Id8<T, GENERATION_BITS> {}

    impl<T, const GENERATION_BITS: usize> PartialOrd for Id8<T, GENERATION_BITS> {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    impl<T, const GENERATION_BITS: usize> Ord for Id8<T, GENERATION_BITS> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.cmp(&other.0)
        }
    }
}
