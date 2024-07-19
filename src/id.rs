//! additional ID types

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use typenum::Unsigned;

use crate::state::State;

pub trait IdTrait: Sized + Copy + Clone + PartialEq + Eq + Hash + fmt::Debug {
    type IndexBits: Unsigned;
    type GenerationBits: Unsigned;

    fn new(index: usize, generation: u32) -> Self;
    fn index(&self) -> usize;
    fn generation(&self) -> u32;
    fn matching_state(&self) -> State<Self::GenerationBits>;
    /// Produce an ID that's guaranteed never to be present in any `Registry`.
    fn null() -> Self;
    /// Returns `true` only for the ID returned by `null` above.
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

/// A smaller ID type for caller who want to save space.
///
/// Using this ID type requires picking a value for the `GENERATION_BITS` const parameter, which
/// must be between 0 and 31 inclusive. The number of index bits is 32 minus `GENERATION_BITS`.
/// Setting `GENERATION_BITS` to 0 means that any removed IDs are immediately retired (see
/// [`recycle_retired`](crate::Registry::recycle_retired)). Setting it to 31 means there are only
/// two possible slot indexes, and since one slot reserved for the [`null`](IdTrait::null) ID,
/// trying to [`insert`](crate::Registry::insert) a second element will panic. Most callers will
/// probably want a value somewhere in the middle, like 10 or 12. But in general you're using this
/// ID type because you know exactly what your application needs, so who am I to tell you what to
/// do? :)
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
        // Unlike the default 64-bit `Id`, where bit 32 is always 1, this type has no extra bits,
        // and it doesn't get a NonZero representation. This time we put the index in the high
        // bits, because extracting the high bits is always a single instruction (right shift with
        // immediate). Extracting the low bits might require two instructions, for example on
        // RISC-V when the `andi` bitmask is larger than 11 bits.
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

impl<T, const GENERATION_BITS: usize> fmt::Debug for Id32<T, GENERATION_BITS>
where
    Self: IdTrait,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "Id {{ index: {}, generation {} }}",
            self.index(),
            self.generation(),
        )
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

    impl<T, const GENERATION_BITS: usize> fmt::Debug for Id8<T, GENERATION_BITS>
    where
        Self: IdTrait,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(
                f,
                "Id {{ index: {}, generation {} }}",
                self.index(),
                self.generation(),
            )
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
        let id64 = crate::Id::<()>::new(42, 99);
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
