//! Deterministic `ex2`, `lg2`, `sin`, `cos` (Task 4).
//!
//! These are the CPU oracle for the device functions in `echo-kernels/echo_math.cu`
//! (`__echo_ex2f`, `__echo_lg2f`, `__echo_sinf`, `__echo_cosf`). Both sides perform the same
//! sequence of correctly rounded IEEE-754 binary32 operations (round to nearest even, subnormals
//! kept), so a GPU result equals the value returned here bit for bit, on every GPU.
//!
//! Operation correspondence (Rust here -> CUDA in `echo_math.cu`):
//! `a * b` -> `__fmul_rn`, `a + b` -> `__fadd_rn`, `a - b` -> `__fsub_rn`, `a / b` -> `__fdiv_rn`,
//! [`fma`] -> `__fmaf_rn`, [`rint`] -> `rintf`, `to_bits`/`from_bits` -> `__float_as_int`/`__int_as_float`,
//! `k as i32` (only for integral `|k| < 2^31`) -> `(int)k`, `e as f32` (small integer) -> `(float)e`.
//!
//! Nothing here calls the platform libm: [`fma`] and [`rint`] are implemented with exact
//! f64 arithmetic and bit operations instead of `f32::mul_add` / `f32::round_ties_even`, which
//! lower to libm calls on x86-64 targets without FMA/SSE4.1.
//!
//! Special values: every NaN result is the canonical NaN `0x7fffffff` (what NVIDIA GPUs produce).
//! `ex2f`: NaN -> NaN, `x >= 128` -> +inf, `x < -150` -> +0.
//! `lg2f`: NaN or `x < 0` -> NaN, ±0 -> -inf, +inf -> +inf.
//! `sinf`/`cosf`: NaN, ±inf -> NaN; `sinf(±0) = ±0`. ≤ 2 ulp over the whole finite range:
//! Cody-Waite reduction for `|x| <= 105615`, exact Payne-Hanek reduction above (Ruling R9).
//! Integer ops used by Payne-Hanek: 32x32->64-bit multiply, shifts, `leading_zeros` (`__clzll`),
//! and exact u32 -> f32 conversions of 24-bit values.

/// Canonical NaN bit pattern, as produced by NVIDIA GPU arithmetic.
pub const NAN_BITS: u32 = 0x7fff_ffff;

// ex2: c_k = (ln 2)^k / k! rounded to nearest f32.
const EX2_C: [u32; 8] = [
    0x3f80_0000, // 1
    0x3f31_7218, // 0.6931471824645996
    0x3e75_fdf0, // 0.24022650718688965
    0x3d63_5847, // 0.05550410971045494
    0x3c1d_955b, // 0.009618128649890423
    0x3aae_c3ff, // 0.0013333557872101665
    0x3921_8489, // 0.0001540352968731895
    0x377f_e5fe, // 1.5252733646775596e-05
];

// lg2: K = 2/ln 2 split as K_HI + K_LO (each rounded to nearest f32), and K/3, K/5, K/7, K/9 rounded to f32.
const LG2_KHI: u32 = 0x4038_aa3b; // 2.885390043258667
const LG2_KLO: u32 = 0x3325_7060; // 3.851926067000022e-08
const LG2_K3: u32 = 0x3f76_384f; // 0.9617967009544373
const LG2_K5: u32 = 0x3f13_bb63; // 0.5770780444145203
const LG2_K7: u32 = 0x3ed3_0bb1; // 0.41219857335090637
const LG2_K9: u32 = 0x3ea4_258a; // 0.32059890031814575
/// Largest f32 below sqrt(2) (1.4142135381698608); mantissas above it are halved.
const SQRT2_BELOW: u32 = 0x3fb5_04f3;

// sin/cos: 2/pi and the Cody-Waite split of pi/2: C1 = RN(pi/2), C2 = RN(pi/2 - C1), C3 = RN(pi/2 - C1 - C2).
const TWO_OVER_PI: u32 = 0x3f22_f983; // 0.6366197466850281
const PIO2_1: u32 = 0x3fc9_0fdb; // 1.5707963705062866
const PIO2_2: u32 = 0xb33b_bd2e; // -4.371138828673793e-08
const PIO2_3: u32 = 0xa6f7_2ced; // -1.7151245100058819e-15
// Taylor coefficients rounded to nearest f32.
const SIN_S3: u32 = 0xbe2a_aaab; // -1/6
const SIN_S5: u32 = 0x3c08_8889; // 1/120
const SIN_S7: u32 = 0xb950_0d01; // -1/5040
const SIN_S9: u32 = 0x3638_ef1d; // 1/362880
const COS_C2: u32 = 0xbf00_0000; // -1/2
const COS_C4: u32 = 0x3d2a_aaab; // 1/24
const COS_C6: u32 = 0xbab6_0b61; // -1/720
const COS_C8: u32 = 0x37d0_0d01; // 1/40320

#[inline(always)]
fn k(bits: u32) -> f32 {
    f32::from_bits(bits)
}

#[inline(always)]
fn nan() -> f32 {
    f32::from_bits(NAN_BITS)
}

/// Replaces any NaN by the canonical NaN (a no-op on the GPU, whose NaNs are already canonical).
#[inline(always)]
fn canon(y: f32) -> f32 {
    if y.is_nan() {
        nan()
    } else {
        y
    }
}

/// Correctly rounded fused multiply-add `a * b + c` (one rounding), without libm.
///
/// `a * b` is exact in f64 (48 significant bits). The f64 sum is then rounded to odd
/// (via TwoSum's exact error term), and rounding a 53-bit round-to-odd value to f32 equals
/// rounding the exact value to f32, including subnormal and overflowing results.
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    let p = a as f64 * b as f64;
    let cd = c as f64;
    let s = p + cd;
    if !s.is_finite() {
        return s as f32;
    }
    // TwoSum: err = (p + cd) - s exactly. Neither p nor cd can overflow or underflow in f64.
    let bb = s - p;
    let err = (p - (s - bb)) + (cd - bb);
    if err == 0.0 {
        return s as f32;
    }
    let sb = s.to_bits();
    let odd = if sb & 1 == 1 {
        s
    } else if (err > 0.0) == (s > 0.0) {
        f64::from_bits(sb + 1) // one ulp away from zero, towards the exact value
    } else {
        f64::from_bits(sb - 1) // one ulp towards zero, towards the exact value
    };
    odd as f32
}

/// Round to the nearest integer, ties to even (IEEE roundToIntegralTiesToEven, CUDA `rintf`), without libm.
pub fn rint(x: f32) -> f32 {
    let a = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    if !(a < 8_388_608.0) {
        return x; // |x| >= 2^23 is already an integer; also ±inf and NaN
    }
    let r = (a + 8_388_608.0) - 8_388_608.0;
    f32::from_bits(r.to_bits() | (x.to_bits() & 0x8000_0000))
}

/// 2^x.
pub fn ex2f(x: f32) -> f32 {
    if x.is_nan() {
        return nan();
    }
    if x >= 128.0 {
        return f32::INFINITY;
    }
    if x < -150.0 {
        return 0.0;
    }
    let n = rint(x);
    let f = x - n; // exact, |f| <= 0.5
    let mut p = k(EX2_C[7]);
    p = fma(p, f, k(EX2_C[6]));
    p = fma(p, f, k(EX2_C[5]));
    p = fma(p, f, k(EX2_C[4]));
    p = fma(p, f, k(EX2_C[3]));
    p = fma(p, f, k(EX2_C[2]));
    p = fma(p, f, k(EX2_C[1]));
    p = fma(p, f, k(EX2_C[0]));
    // n in [-150, 128]; both halves lie in [-75, 64], so both scale factors are normal floats and
    // only the final multiply can round (once, into the subnormal range when the result is tiny).
    let ni = n as i32;
    let h = ni / 2; // truncates towards zero, like C
    let s1 = f32::from_bits(((h + 127) as u32) << 23);
    let s2 = f32::from_bits(((ni - h + 127) as u32) << 23);
    (p * s1) * s2
}

/// log2(x).
pub fn lg2f(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return nan();
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    let mut xs = x;
    let mut e: i32 = 0;
    if x < f32::MIN_POSITIVE {
        xs = x * 8_388_608.0; // 2^23, exact
        e = -23;
    }
    let b = xs.to_bits();
    e += (b >> 23) as i32 - 127;
    let mut mb = (b & 0x007f_ffff) | 0x3f80_0000; // m in [1, 2)
    if mb > SQRT2_BELOW {
        mb -= 0x0080_0000; // m / 2, now in [sqrt(1/2), 1)
        e += 1;
    }
    let m = f32::from_bits(mb);
    // log2 m = K·atanh-series(s), s = (m-1)/(m+1) = s + s_lo (s_lo recovers the division error).
    let u = m - 1.0; // exact (Sterbenz)
    let v = m + 1.0; // rounded
    let v_lo = m - (v - 1.0); // exact: m + 1 = v + v_lo
    let s = u / v;
    let rem = fma(-s, v, u); // exact remainder u - s·v
    let s_lo = fma(-s, v_lo, rem) / v; // ≈ u/(v + v_lo) - s, ~2^-24 relative to s
    let z = s * s;
    let mut p = k(LG2_K9);
    p = fma(p, z, k(LG2_K7));
    p = fma(p, z, k(LG2_K5));
    p = fma(p, z, k(LG2_K3)); // p ≈ K(1/3 + z/5 + z²/7 + z³/9)
    // Everything except K_HI·s is at most ~1% of the result, so its roundings barely matter;
    // the leading product K_HI·s is carried exactly into the single rounding of the last fma.
    let tail = fma(s * z, p, fma(k(LG2_KLO), s, k(LG2_KHI) * s_lo));
    let l = fma(k(LG2_KHI), s, tail);
    (e as f32) + l
}

/// Cody-Waite reduction: returns r ≈ x - k·pi/2 and the quadrant k mod 4.
#[inline(always)]
fn reduce(x: f32) -> (f32, i32) {
    let kf = rint(x * k(TWO_OVER_PI));
    let nk = -kf;
    let mut r = fma(nk, k(PIO2_1), x);
    r = fma(nk, k(PIO2_2), r);
    r = fma(nk, k(PIO2_3), r);
    // |k| >= 2^31 is a multiple of 2^8 (so k mod 4 = 0); below that `as i32` is exact.
    let q = if kf < 2_147_483_648.0 && kf > -2_147_483_648.0 { kf as i32 } else { 0 };
    (r, q & 3)
}

/// Bits of 2/pi after the binary point: word j holds bit positions 32j+1 ..= 32j+32 (weights
/// 2^-(32j+1) .. 2^-(32j+32)). Seven words cover every f32 exponent: the largest window starts at word 3.
const TWO_OVER_PI_BITS: [u32; 7] =
    [0xa2f9_836e, 0x4e44_1529, 0xfc27_57d1, 0xf534_ddc0, 0xdb62_9599, 0x3c43_9041, 0xfe51_63ab];

/// Above this |x| the Cody-Waite reduction is not accurate enough and Payne-Hanek is used.
const REDUCE_FAST_MAX: f32 = 105_615.0;

/// Bit `k` of the little-endian 160-bit integer `p`.
#[inline(always)]
fn bit160(p: &[u32; 5], k: u32) -> u32 {
    (p[(k >> 5) as usize] >> (k & 31)) & 1
}

/// Bits `pos .. pos+64` of the little-endian 160-bit integer `p` (needs pos/32 + 2 <= 4).
#[inline(always)]
fn bits64(p: &[u32; 5], pos: u32) -> u64 {
    let i = (pos >> 5) as usize;
    let off = pos & 31;
    let lo = ((p[i + 1] as u64) << 32) | p[i] as u64;
    if off == 0 {
        lo
    } else {
        (lo >> off) | ((p[i + 2] as u64) << (64 - off))
    }
}

/// Exact Payne-Hanek reduction for finite |x| > 105615: returns r ≈ x - k·pi/2 (|r| <= pi/4) and k mod 4.
///
/// |x| = m·2^sh with m the 24-bit significand (x is normal here, sh in [-7, 104]). Only the bits of
/// 2/pi from word j0 on matter mod 4: m·2^sh·(bits before word j0) is a multiple of 4. The product
/// P = m × (four words from j0) is formed exactly with 32x32->64-bit integer multiplies; its binary
/// point sits at bit 128 - s, where s = sh - 32·j0. The quadrant is the two integer bits above it,
/// the reduced argument comes from the 64 fraction bits below it (rounded to the nearest quadrant),
/// normalised with a leading-zero count, split exactly into two f32 parts and multiplied by pi/2.
fn reduce_big(x: f32) -> (f32, i32) {
    let xb = x.to_bits();
    let ebits = ((xb >> 23) & 0xff) as i32;
    let m = (xb & 0x007f_ffff) | 0x0080_0000;
    let sh = ebits - 150;
    let j0 = if sh >= 2 { (sh - 2) / 32 } else { 0 };
    let s = sh - 32 * j0; // in [-7, 33]
    let j = j0 as usize;
    let w = [
        TWO_OVER_PI_BITS[j],
        TWO_OVER_PI_BITS[j + 1],
        TWO_OVER_PI_BITS[j + 2],
        TWO_OVER_PI_BITS[j + 3],
    ];
    let mut p = [0u32; 5];
    let mut carry = 0u64;
    for i in 0..4 {
        let t = (m as u64) * (w[3 - i] as u64) + carry;
        p[i] = t as u32;
        carry = t >> 32;
    }
    p[4] = carry as u32;
    let q0 = (128 - s) as u32; // binary point, in [95, 135]
    let mut q = (bit160(&p, q0) | (bit160(&p, q0 + 1) << 1)) as i32;
    let f = bits64(&p, q0 - 64); // fraction in units of 2^-64
    q += (f >> 63) as i32; // fraction >= 1/2: round up to the next quadrant, fraction becomes f - 1
    let neg = (f >> 63) != 0;
    let u = if neg { f.wrapping_neg() } else { f }; // |fraction| in units of 2^-64, <= 2^63
    let mut r = 0.0f32;
    if u != 0 {
        let lz = u.leading_zeros() as i32;
        let n = u << lz;
        let h = (n >> 40) as u32; // top 24 bits, exactly representable
        let l = ((n >> 16) & 0x00ff_ffff) as u32; // next 24 bits
        // |fraction| = h·2^(-24-lz) + l·2^(-48-lz) + (dropped < 2^(-48-lz)); both scales are normal.
        let sh_h = f32::from_bits(((127 - 24 - lz) as u32) << 23);
        let sh_l = f32::from_bits(((127 - 48 - lz) as u32) << 23);
        let mut fh = (h as f32) * sh_h;
        let mut fl = (l as f32) * sh_l;
        if neg {
            fh = -fh;
            fl = -fl;
        }
        r = fma(fh, k(PIO2_1), fma(fh, k(PIO2_2), fl * k(PIO2_1)));
    }
    if (xb >> 31) != 0 {
        (-r, (-q) & 3) // x = -(q·pi/2 + r)
    } else {
        (r, q & 3)
    }
}

/// Fast Cody-Waite reduction for |x| <= 105615, Payne-Hanek above.
#[inline(always)]
fn reduce_any(x: f32) -> (f32, i32) {
    if f32::from_bits(x.to_bits() & 0x7fff_ffff) > REDUCE_FAST_MAX {
        reduce_big(x)
    } else {
        reduce(x)
    }
}

#[inline(always)]
fn sin_poly(r: f32) -> f32 {
    let z = r * r;
    let mut p = k(SIN_S9);
    p = fma(p, z, k(SIN_S7));
    p = fma(p, z, k(SIN_S5));
    p = fma(p, z, k(SIN_S3));
    let t = z * r;
    fma(t, p, r)
}

#[inline(always)]
fn cos_poly(r: f32) -> f32 {
    let z = r * r;
    let mut p = k(COS_C8);
    p = fma(p, z, k(COS_C6));
    p = fma(p, z, k(COS_C4));
    p = fma(p, z, k(COS_C2));
    fma(p, z, 1.0)
}

#[inline(always)]
fn not_finite(x: f32) -> bool {
    x.to_bits() & 0x7f80_0000 == 0x7f80_0000
}

/// sin(x).
pub fn sinf(x: f32) -> f32 {
    if not_finite(x) {
        return nan();
    }
    if x == 0.0 {
        return x; // keeps the sign of zero
    }
    let (r, q) = reduce_any(x);
    let y = match q {
        0 => sin_poly(r),
        1 => cos_poly(r),
        2 => -sin_poly(r),
        _ => -cos_poly(r),
    };
    canon(y)
}

/// cos(x).
pub fn cosf(x: f32) -> f32 {
    if not_finite(x) {
        return nan();
    }
    let (r, q) = reduce_any(x);
    let y = match q {
        0 => cos_poly(r),
        1 => -sin_poly(r),
        2 => -cos_poly(r),
        _ => sin_poly(r),
    };
    canon(y)
}
