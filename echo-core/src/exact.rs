//! Exact, order-independent f32 accumulator: the CPU oracle for ECHO's replacement of float
//! `atomicAdd` (plan Task 5). The device code in `echo-kernels/echo_atomic.cu` mirrors
//! [`ExactAcc::add`] and [`ExactAcc::to_f32`] operation for operation; the two must agree bit
//! for bit (checked by `echo-gpu-tests/tests/gpu_exact.rs`).
//!
//! # Representation
//! A finite normal input is `v = ±m·2^(E−150)` with `m` the 24-bit significand and `E` the
//! biased exponent (1..=254). It is added as the integer `±m << ((E−1) % 32)` into the cells
//! starting at `(E−1) / 32`: the low 32 bits go to that cell, the remaining (at most 23) bits
//! to the next one. Cell `i` weighs `2^(32i − 149)`, so the cells hold the sum exactly as a
//! fixed-point integer in units of `2^−149`. Cells are signed and never normalized while
//! accumulating, so the state after any multiset of additions is independent of their order.
//!
//! # Special values
//! - Zero and subnormal inputs count as ±0 (ignored), matching `atom.add.f32`'s flush to zero.
//! - NaN sets `nan`; +inf / −inf set `pos_inf` / `neg_inf`.
//! - [`ExactAcc::to_f32`]: `nan`, or both infinities → NaN (`0x7fc0_0000`); one infinity →
//!   that infinity; otherwise the exact sum rounded to nearest-even. A magnitude below
//!   `2^−126` (including exact zero) becomes `+0.0`; a rounded magnitude above `f32::MAX`
//!   becomes ±inf.
//!
//! # Limit
//! Every addition puts less than `2^32` into a cell, and `(2^31 − 1)·(2^32 − 1) < 2^63`, so a
//! cell absorbs `2^31 − 1` contributions without overflowing `i64`. On the GPU this is a limit
//! of **2^31 − 1 atomic adds per target address per launch**: no more may hit one target
//! before `__echo_atomic_finalize` folds its cells into the target. Beyond that a cell can wrap
//! (two's complement, same on CPU and GPU) and the result is meaningless; nothing detects it.
//! c006 launches are many orders of magnitude below this.

/// Number of 32-bit-weighted cells. Inputs touch cells 0..=8; cell 9 absorbs carries.
pub const CELLS: usize = 10;

/// Canonical NaN produced by [`ExactAcc::to_f32`].
pub const NAN_BITS: u32 = 0x7fc0_0000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExactAcc {
    cells: [i64; CELLS],
    nan: bool,
    pos_inf: bool,
    neg_inf: bool,
}

impl ExactAcc {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `v` exactly. Order-independent: integer additions commute.
    pub fn add(&mut self, v: f32) {
        let bits = v.to_bits();
        let neg = (bits >> 31) != 0;
        let e = (bits >> 23) & 0xff;
        let f = bits & 0x7f_ffff;
        if e == 0 {
            return; // zero or subnormal: flushed to zero
        }
        if e == 255 {
            if f != 0 {
                self.nan = true;
            } else if neg {
                self.neg_inf = true;
            } else {
                self.pos_inf = true;
            }
            return;
        }
        let m = (f | 0x80_0000) as u64;
        let k = e - 1;
        let cell = (k / 32) as usize;
        let x = m << (k % 32);
        let lo = (x & 0xffff_ffff) as i64;
        let hi = (x >> 32) as i64;
        if neg {
            self.cells[cell] = self.cells[cell].wrapping_sub(lo);
            self.cells[cell + 1] = self.cells[cell + 1].wrapping_sub(hi);
        } else {
            self.cells[cell] = self.cells[cell].wrapping_add(lo);
            self.cells[cell + 1] = self.cells[cell + 1].wrapping_add(hi);
        }
    }

    /// The exact sum rounded to f32 (see the module docs for special values).
    pub fn to_f32(&self) -> f32 {
        f32::from_bits(self.to_bits())
    }

    fn to_bits(&self) -> u32 {
        const INF: u32 = 0x7f80_0000;
        if self.nan || (self.pos_inf && self.neg_inf) {
            return NAN_BITS;
        }
        if self.pos_inf {
            return INF;
        }
        if self.neg_inf {
            return 0x8000_0000 | INF;
        }

        // Carry-normalize into a two's-complement integer: 10 u32 limbs plus a signed top word.
        // |carry| <= 2^31 + 1 throughout, so no step can overflow i64.
        let mut limbs = [0u32; CELLS];
        let mut carry: i64 = 0;
        for i in 0..CELLS {
            let c = self.cells[i];
            let t = (c & 0xffff_ffff) + carry;
            limbs[i] = t as u32;
            carry = (c >> 32) + (t >> 32);
        }
        let neg = carry < 0;
        let sign: u32 = if neg { 0x8000_0000 } else { 0 };
        if carry != 0 && carry != -1 {
            return sign | INF; // |sum| >= 2^320 units
        }
        if neg {
            // Magnitude = two's-complement negation of the 320-bit value.
            let mut c: u64 = 1;
            for limb in limbs.iter_mut() {
                let t = (!*limb) as u64 + c;
                *limb = t as u32;
                c = t >> 32;
            }
            if c != 0 {
                return sign | INF; // magnitude exactly 2^320 units
            }
        }

        // Position p of the leading one bit.
        let mut top: i32 = CELLS as i32 - 1;
        while top >= 0 && limbs[top as usize] == 0 {
            top -= 1;
        }
        if top < 0 {
            return 0; // exact zero -> +0.0
        }
        let p = 32 * top + 31 - limbs[top as usize].leading_zeros() as i32;
        if p < 23 {
            return 0; // magnitude < 2^-126 -> +0.0
        }
        let mut exp = p - 22; // biased exponent: value = m * 2^(p - 23 - 149)
        let shift = p - 23;
        let w = (shift / 32) as usize;
        let b = shift % 32;
        let next = if w + 1 < CELLS { limbs[w + 1] as u64 } else { 0 };
        let mut m = (((limbs[w] as u64) | (next << 32)) >> b) as u32 & 0xff_ffff;
        if shift > 0 {
            let r = shift - 1;
            let rw = (r / 32) as usize;
            let rb = r % 32;
            let round = (limbs[rw] >> rb) & 1 != 0;
            let mut sticky = (limbs[rw] & ((1u32 << rb) - 1)) != 0;
            for &l in &limbs[..rw] {
                sticky |= l != 0;
            }
            if round && (sticky || (m & 1) != 0) {
                m += 1;
                if m == 1 << 24 {
                    m >>= 1;
                    exp += 1;
                }
            }
        }
        if exp >= 255 {
            return sign | INF;
        }
        sign | ((exp as u32) << 23) | (m & 0x7f_ffff)
    }
}
