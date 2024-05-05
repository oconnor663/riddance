use std::marker::PhantomData;
use typenum::Unsigned;

#[derive(Copy, Clone, Debug)]
pub struct State<GenerationBits: Unsigned>(pub u32, PhantomData<GenerationBits>);

impl<GenerationBits: Unsigned> State<GenerationBits> {
    pub fn new(bits: u32) -> Self {
        if GenerationBits::U32 < 31 {
            debug_assert_eq!(
                0,
                bits >> (GenerationBits::U32 + 1),
                "illegal high bits set in state",
            );
        }
        Self(bits, PhantomData)
    }

    pub fn generation(self) -> u32 {
        // The low bit is the occupied/empty bit.
        self.0 >> 1
    }

    pub fn is_occupied(self) -> bool {
        // Odd states are occupied.
        self.0 & 1 == 1
    }

    pub fn is_empty(self) -> bool {
        // Even states are empty.
        self.0 & 1 == 0
    }

    pub fn is_retired(self) -> bool {
        self.0 == 0
    }

    pub fn next_empty_state(self) -> Self {
        debug_assert!(self.is_occupied());
        // Occupied states are odd, so this might overflow, and we need to mask it.
        let mask = u32::MAX >> (31 - GenerationBits::U32);
        Self::new(self.0.wrapping_add(1) & mask)
    }

    pub fn next_occupied_state(self) -> Self {
        debug_assert!(self.is_empty());
        // Empty states are even, so this can't overflow.
        Self::new(self.0 + 1)
    }
}

impl<GenerationBits: Unsigned> PartialEq for State<GenerationBits> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<GenerationBits: Unsigned> Eq for State<GenerationBits> {}

pub const fn states_per_word<GenerationBits: Unsigned>() -> usize {
    // NOTE: The number of state bits is GenerationBits + 1.
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => 1,
        8..=15 => 2,
        4..=7 => 4,
        2..=3 => 8,
        1 => 16,
        0 => 32,
    }
}

pub const fn word_count_from_state_count<GenerationBits: Unsigned>(state_count: usize) -> usize {
    state_count.div_ceil(states_per_word::<GenerationBits>())
}

pub const fn unused_states_in_last_word<GenerationBits: Unsigned>(state_count: usize) -> usize {
    let spw = states_per_word::<GenerationBits>();
    spw - 1 - (state_count.wrapping_sub(1) % spw)
}

pub unsafe fn read_state<GenerationBits: Unsigned>(
    state_words: *const u32,
    index: usize,
) -> State<GenerationBits> {
    // NOTE: The number of state bits is GenerationBits + 1.
    let bits = match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => *state_words.add(index),
        8..=15 => *(state_words as *const u16).add(index) as u32,
        4..=7 => *(state_words as *const u8).add(index) as u32,
        2..=3 => (*(state_words as *const u8).add(index / 2) as u32 >> (4 * (index % 2))) & 0b1111,
        1 => (*(state_words as *const u8).add(index / 4) as u32 >> (2 * (index % 4))) & 0b11,
        0 => (*(state_words as *const u8).add(index / 8) as u32 >> (index % 8)) & 0b1,
    };
    State::new(bits)
}

pub unsafe fn write_state<GenerationBits: Unsigned>(
    state_words: *mut u32,
    index: usize,
    state: State<GenerationBits>,
) {
    match GenerationBits::U32 {
        32.. => panic!("generation bits must be 31 or less"),
        16..=31 => *state_words.add(index) = state.0,
        8..=15 => *(state_words as *mut u16).add(index) = state.0 as u16,
        4..=7 => *(state_words as *mut u8).add(index) = state.0 as u8,
        2..=3 => {
            let entry = &mut *(state_words as *mut u8).add(index / 2);
            *entry &= !(0b1111 << (4 * (index % 2)));
            *entry |= (state.0 as u8 & 0b1111) << (4 * (index % 2));
        }
        1 => {
            let entry = &mut *(state_words as *mut u8).add(index / 4);
            *entry &= !(0b11 << (2 * (index % 4)));
            *entry |= (state.0 as u8 & 0b11) << (2 * (index % 4));
        }
        0 => {
            let entry = &mut *(state_words as *mut u8).add(index / 8);
            *entry &= !(0b1 << (index % 8));
            *entry |= (state.0 as u8 & 0b1) << (index % 8);
        }
    }
}
