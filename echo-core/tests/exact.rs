//! CPU tests of the exact, order-independent f32 accumulator (plan Task 5).
use echo_core::exact::ExactAcc;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

fn sum(vals: &[f32]) -> f32 {
    let mut acc = ExactAcc::new();
    for &v in vals {
        acc.add(v);
    }
    acc.to_f32()
}

fn p2(e: i32) -> f32 {
    2f32.powi(e)
}

/// Exact reference for inputs whose f64 sum is exact (see callers): f64 `as f32` rounds to
/// nearest-even and overflows to inf; ECHO additionally flushes |x| < 2^-126 (and -0) to +0.
fn reference(vals: &[f32]) -> f32 {
    let s: f64 = vals.iter().map(|&v| v as f64).sum();
    if s.abs() < 2f64.powi(-126) {
        0.0
    } else {
        s as f32
    }
}

/// Random normal f32 with a uniformly chosen binary exponent in [lo, hi] and random sign.
fn rand_f32(rng: &mut StdRng, lo: i32, hi: i32) -> f32 {
    let e = rng.gen_range(lo..=hi);
    let mant: u32 = rng.gen_range(0..1u32 << 23);
    let sign: u32 = if rng.gen::<bool>() { 1 << 31 } else { 0 };
    f32::from_bits(sign | (((e + 127) as u32) << 23) | mant)
}

#[test]
fn order_independent() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut vals: Vec<f32> = (0..1000)
        .map(|_| {
            let mag = 10f64.powf(rng.gen_range(-30.0..30.0)) as f32;
            if rng.gen::<bool>() {
                -mag
            } else {
                mag
            }
        })
        .collect();
    let first = sum(&vals).to_bits();
    for _ in 0..50 {
        vals.shuffle(&mut rng);
        assert_eq!(sum(&vals).to_bits(), first);
    }
    // Non-tautology guard: the shared answer is the correctly rounded sum. The f64 sum's error
    // (<= 1000 * 1e30 * 2^-53 ~ 1e17) is far below half an f32 ulp of the result.
    let f64_sum: f64 = vals.iter().map(|&v| v as f64).sum();
    let got = f32::from_bits(first) as f64;
    assert!(f64_sum.abs() > 1e29, "test data too cancelled to be meaningful: {f64_sum}");
    assert!(((got - f64_sum) / f64_sum).abs() < 1e-7, "got {got}, f64 sum {f64_sum}");
}

#[test]
fn cancels_exactly() {
    assert_eq!(sum(&[1e30, 1.0, -1e30]), 1.0);
}

#[test]
fn exact_not_sequential() {
    let vals = [1.0, p2(-24), p2(-24)];
    assert_eq!(vals.iter().fold(0f32, |a, &b| a + b), 1.0, "premise: sequential f32 loses both");
    assert_eq!(sum(&vals), 1.0 + p2(-23));
}

#[test]
fn tie_rounds_to_even() {
    assert_eq!(sum(&[1.0, p2(-24)]).to_bits(), 1f32.to_bits());
    // A tie whose lower neighbour is odd rounds up: (1 + 2^-23) + 2^-24 -> 1 + 2^-22.
    assert_eq!(sum(&[1.0 + p2(-23), p2(-24)]), 1.0 + p2(-22));
    // Just above the tie rounds up even when the neighbour is even.
    assert_eq!(sum(&[1.0, p2(-24), p2(-60)]), 1.0 + p2(-23));
    // Negative ties are symmetric.
    assert_eq!(sum(&[-1.0, -p2(-24)]).to_bits(), (-1f32).to_bits());
    assert_eq!(sum(&[-1.0, -p2(-24), -p2(-24)]), -(1.0 + p2(-23)));
}

#[test]
fn specials() {
    assert!(sum(&[f32::NAN, 1.0]).is_nan());
    assert!(sum(&[f32::INFINITY, f32::NEG_INFINITY]).is_nan());
    assert_eq!(sum(&[f32::INFINITY, 1e38, 1e38]), f32::INFINITY);
    assert_eq!(sum(&[f32::NEG_INFINITY, 1e38]), f32::NEG_INFINITY);
    assert_eq!(sum(&[f32::MAX, f32::MAX]), f32::INFINITY);
    assert_eq!(sum(&[-f32::MAX, -f32::MAX]), f32::NEG_INFINITY);
    assert_eq!(sum(&[1e-40]).to_bits(), 0);
    // Subnormal inputs are ignored entirely (FTZ), not just when alone.
    assert_eq!(sum(&[1.0, 1e-40, 1e-40]), 1.0);
    assert_eq!(sum(&[-1e-40]).to_bits(), 0);
    assert_eq!(sum(&[-0.0]).to_bits(), 0);
    assert_eq!(sum(&[]).to_bits(), 0);
    assert_eq!(sum(&[1.5, -1.5]).to_bits(), 0, "exact zero is +0.0");
    // MAX's ulp is 2^104. Below half an ulp stays MAX; the tie at half an ulp rounds to the
    // even neighbour 2^128, which overflows to inf (IEEE rounds with unbounded exponent).
    assert_eq!(sum(&[f32::MAX, p2(102)]), f32::MAX);
    assert_eq!(sum(&[f32::MAX, p2(103)]), f32::INFINITY);
}

#[test]
fn tiny_result_flushes_to_zero() {
    // The brief's [2^-126, -2^-127] cannot express this: -2^-127 is a subnormal input and is
    // flushed to zero on input, giving 2^-126. Normal inputs with exact sum 2^-127 instead:
    assert_eq!(sum(&[p2(-125), -1.5 * p2(-126)]).to_bits(), 0);
    assert_eq!(sum(&[-p2(-125), 1.5 * p2(-126)]).to_bits(), 0);
    // ...and the brief's literal inputs, under FTZ inputs:
    assert_eq!(sum(&[p2(-126), -p2(-127)]), p2(-126));
    // The smallest normal survives.
    assert_eq!(sum(&[p2(-125), -p2(-126)]), p2(-126));
}

/// Inputs drawn from a 21-binade window [lo, hi] are integers below 2^44 in units of
/// 2^(lo-23); 400 of them sum below 2^53, so every f64 partial sum is exact and f64 is an
/// independent exact reference. Covers mid, subnormal-boundary and overflow ranges.
#[test]
fn matches_exact_f64_reference() {
    let mut rng = StdRng::seed_from_u64(11);
    for &(lo, hi) in &[(-10, 10), (-126, -106), (107, 127), (-20, 0)] {
        for _ in 0..500 {
            let n = rng.gen_range(1..=400);
            let vals: Vec<f32> = (0..n).map(|_| rand_f32(&mut rng, lo, hi)).collect();
            assert_eq!(sum(&vals).to_bits(), reference(&vals).to_bits(), "range {lo}..{hi} vals {vals:?}");
        }
    }
}

/// Many additions into the same cells exercise carry normalization: 2^20 copies of a value
/// whose significand straddles two cells.
#[test]
fn many_terms_carry() {
    let v = f32::from_bits(0x3fff_ffff); // 1.9999999, E = 127, shift 30: spans cells 3 and 4
    let mut acc = ExactAcc::new();
    for _ in 0..(1u32 << 20) {
        acc.add(v);
        acc.add(-1.0);
    }
    assert_eq!(acc.to_f32(), ((v as f64 - 1.0) * (1u64 << 20) as f64) as f32);
}
