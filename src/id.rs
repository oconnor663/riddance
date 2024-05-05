//! additional ID types

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroU64;
use typenum::Unsigned;

use crate::state::State;

pub trait IdTrait: Sized + Copy + Clone + PartialEq + Eq + Hash {
    type IndexBits: Unsigned;
    type GenerationBits: Unsigned;

    fn new(index: usize, generation: u32) -> Self;
    fn index(&self) -> usize;
    fn generation(&self) -> u32;
    fn matching_state(&self) -> State<Self::GenerationBits>;
    fn null() -> Self;
    fn is_null(&self) -> bool;

    fn max_len() -> usize {
        crate::static_assert_index_bits::<Self::IndexBits>();
        // The max index (equal to max_len) is reserved for the null ID. You can create a Registry
        // with IndexBits=1 if you really want to, but you can't insert anything into it. We avoid
        // assuming anything about GenerationBits, because we support GenerationBits=0.
        (u32::MAX >> (32 - Self::IndexBits::U32)) as usize
    }

    fn max_generation() -> u32 {
        crate::static_assert_generation_bits::<Self::GenerationBits>();
        (u32::MAX >> 1) >> (31 - Self::GenerationBits::U32)
    }

    fn reserved_state(&self) -> State<Self::GenerationBits> {
        // the same generation, with the occupied bit unset
        State::new(self.generation() << 1)
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
// Copy and Eq even when T isn't. See https://github.com/rust-lang/rust/issues/108894.
#[repr(transparent)]
pub struct Id64<T>(
    NonZeroU64,
    // https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
    PhantomData<fn() -> T>,
);

impl<T> IdTrait for Id64<T> {
    type IndexBits = typenum::U32;
    type GenerationBits = typenum::U31;

    fn new(index: usize, generation: u32) -> Self {
        // We store the entire state word in the top 32 bits of the ID. Since occupied states are
        // odd, this means that bit 32 is always 1.
        // - Putting the index in the low 32-bits is nice, because extracting those doesn't usually
        //   require an instruction. (And e.g. Registry::get_unchecked ignores the generation.)
        // - This lets us use the NonZeroU64 representation. (Option<Id> is probably rare, since
        //   IDs can be null anyway, but it's nice to have.)
        // - This saves an instruction when checking an ID against a slot state.
        // - This makes the null ID all 1's, which is also nice.
        debug_assert_eq!(index as u64 >> 32, 0, "high bits should not be set");
        debug_assert_eq!(generation >> 31, 0, "the high bit should not be set");
        Self(
            unsafe {
                NonZeroU64::new_unchecked(
                    ((((generation << 1) + 1) as u64) << 32) | index as u32 as u64,
                )
            },
            PhantomData,
        )
    }

    fn index(&self) -> usize {
        self.0.get() as u32 as usize
    }

    fn generation(&self) -> u32 {
        (self.0.get() >> 33) as u32
    }

    fn matching_state(&self) -> State<Self::GenerationBits> {
        // Bit 32 is always 1.
        debug_assert_eq!(1, (self.0.get() >> 32) & 1);
        // This operation might not require an instruction at all, if the caller can just load the
        // upper 32 bits directly from memory into a register. That's why matching_state is
        // designed the way it is.
        State::new((self.0.get() >> 32) as u32)
    }

    fn null() -> Self {
        Self(NonZeroU64::new(u64::MAX).unwrap(), PhantomData)
    }

    fn is_null(&self) -> bool {
        self.0.get() == u64::MAX
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

/// A smaller ID type for caller who want to save space.
///
/// Using this ID type requires picking a value for the `GENERATION_BITS` const parameter, which
/// must be between 0 and 31 inclusive. The number of index bits is 32 minus `GENERATION_BITS`.
/// Setting `GENERATION_BITS` to 0 means that any removed IDs are immediately retired (see
/// [`recycle_retired`](crate::Registry::recycle_retired)). Setting it to 31 means that the only
/// possible ID is the [`null`](IdTrait::null) ID, and any call to
/// [`insert`](crate::Registry::insert) will panic. Most callers will probably want a value
/// somewhere in the middle, like 10 or 12. But in general you're using this ID type because you
/// know exactly what your application needs, so who am I to tell you what to do? :)
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
// Copy and Eq even when T isn't. See https://github.com/rust-lang/rust/issues/108894.
#[repr(transparent)]
pub struct Id32<T, const GENERATION_BITS: usize>(
    u32,
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

    fn new(index: usize, generation: u32) -> Self {
        // Unlike Id64 above, where bit 32 is always 1, this type has no extra bits, and it doesn't
        // get a NonZero representation. This time we put the index in the high bits, because
        // extracting the high bits is always a single instruction (right shift with immediate).
        // Extracting the low bits might require two instructions, for example on RISC-V when the
        // `andi` bitmask is larger than 11 bits.
        // TODO: Fail for excessively large generations?
        Self((index << GENERATION_BITS) as u32 | generation, PhantomData)
    }

    fn index(&self) -> usize {
        (self.0 >> GENERATION_BITS) as usize
    }

    fn generation(&self) -> u32 {
        self.0 & !(u32::MAX << GENERATION_BITS)
    }

    fn matching_state(&self) -> State<Self::GenerationBits> {
        State::new((self.generation() << 1) + 1)
    }

    fn null() -> Self {
        Self(u32::MAX, PhantomData)
    }

    fn is_null(&self) -> bool {
        self.0 == u32::MAX
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

#[cfg(test)]
pub use id8::Id8;

#[cfg(test)]
mod id8 {
    use super::*;

    /// Id8 is only for testing. It's mostly a copy-paste of Id32 with fewer bits.
    #[repr(transparent)]
    pub struct Id8<T, const GENERATION_BITS: usize>(
        u8,
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

        fn new(index: usize, generation: u32) -> Self {
            Self(
                ((index << GENERATION_BITS) as u32 | generation) as u8,
                PhantomData,
            )
        }

        fn index(&self) -> usize {
            (self.0 >> GENERATION_BITS) as usize
        }

        fn generation(&self) -> u32 {
            (self.0 & !(u8::MAX << GENERATION_BITS)) as u32
        }

        fn matching_state(&self) -> State<Self::GenerationBits> {
            State::new((self.generation() << 1) + 1)
        }

        fn null() -> Self {
            Self(u8::MAX, PhantomData)
        }

        fn is_null(&self) -> bool {
            self.0 == u8::MAX
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

    #[test]
    fn test_id_basics() {
        let id64 = Id64::<()>::new(42, 99);
        assert_eq!(id64.index(), 42);
        assert_eq!(id64.generation(), 99);

        let id32 = Id32::<(), 10>::new(42, 99);
        assert_eq!(id32.index(), 42);
        assert_eq!(id32.generation(), 99);

        let id8 = Id8::<(), 4>::new(7, 15);
        assert_eq!(id8.index(), 7);
        assert_eq!(id8.generation(), 15);
    }
}
