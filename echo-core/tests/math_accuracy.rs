//! Accuracy and special-value tests for `echo_core::math` (Task 4).
//! "Within 2 ulp" is measured against the f64 `std` function (platform libm, test-only) rounded to f32.

use echo_core::math::{cosf, ex2f, fma, lg2f, rint, sinf};
use rand::{Rng, SeedableRng};

/// Maps f32 bit patterns onto a line where adjacent floats differ by 1; +0 and -0 both map to 0.
fn ordinal(x: f32) -> i64 {
    let b = x.to_bits() as i32;
    if b < 0 {
        -((b & 0x7fff_ffff) as i64)
    } else {
        b as i64
    }
}

fn ulp_diff(got: f32, want: f32) -> u64 {
    if got.is_nan() || want.is_nan() {
        return if got.is_nan() && want.is_nan() { 0 } else { u64::MAX };
    }
    (ordinal(got) - ordinal(want)).unsigned_abs()
}

/// Returns (max ulp, argument where it occurred), and panics if the max exceeds 2.
fn check(name: &str, xs: impl Iterator<Item = f32>, f: fn(f32) -> f32, reference: fn(f64) -> f64) -> u64 {
    let mut worst = (0u64, 0f32, 0f32, 0f32);
    let mut n = 0u64;
    for x in xs {
        let got = f(x);
        let want = reference(x as f64) as f32;
        let d = ulp_diff(got, want);
        if d > worst.0 {
            worst = (d, x, got, want);
        }
        n += 1;
    }
    println!(
        "{name}: {n} points, max ulp {} at x={:e} ({:#010x}): got {:e}, want {:e}",
        worst.0,
        worst.1,
        worst.1.to_bits(),
        worst.2,
        worst.3
    );
    assert!(worst.0 <= 2, "{name}: max ulp {} > 2 at x={:e}", worst.0, worst.1);
    worst.0
}

fn linspace(lo: f64, hi: f64, n: usize) -> impl Iterator<Item = f32> {
    (0..n).map(move |i| (lo + (hi - lo) * (i as f64) / ((n - 1) as f64)) as f32)
}

#[test]
fn ex2f_within_2ulp() {
    check("ex2f", linspace(-149.0, 127.9, 2_000_001), ex2f, f64::exp2);
}

/// Deterministic 32-bit mix (splitmix-style finaliser), used to fill low mantissa bits.
fn mix(a: u32, b: u32) -> u32 {
    let mut h = a.wrapping_mul(0x9e37_79b9) ^ b.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

/// Every normal exponent x 4,096 mantissas `(j << 11) | hashed low 11 bits`, plus 440 subnormals in
/// each of the 23 subnormal binades (bit patterns [2^b, 2^(b+1))) with hashed low bits.
fn lg2f_samples() -> impl Iterator<Item = f32> {
    let normals = (1u32..=254)
        .flat_map(|e| (0u32..4096).map(move |j| f32::from_bits((e << 23) | (j << 11) | (mix(e, j) & 0x7ff))));
    let subnormals =
        (0u32..23).flat_map(|b| (0u32..440).map(move |i| f32::from_bits((1 << b) | (mix(1000 + b, i) & ((1 << b) - 1)))));
    normals.chain(subnormals)
}

#[test]
fn lg2f_within_2ulp() {
    check("lg2f", lg2f_samples(), lg2f, f64::log2);
}

/// Every positive finite f32 (2,139,095,039 inputs). Run with `--ignored`; ~1 minute in release on 16 threads.
#[test]
#[ignore = "exhaustive; run with --release -- --ignored"]
fn lg2f_within_2ulp_exhaustive() {
    let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as u32;
    let hs: Vec<_> = (0..nt)
        .map(|t| {
            std::thread::spawn(move || {
                let (mut n, mut over, mut worst, mut at) = (0u64, 0u64, 0u64, 0u32);
                let mut b = 1 + t;
                while b < 0x7f80_0000 {
                    let x = f32::from_bits(b);
                    let d = ulp_diff(lg2f(x), (x as f64).log2() as f32);
                    if d > 2 {
                        over += 1;
                    }
                    if d > worst {
                        worst = d;
                        at = b;
                    }
                    n += 1;
                    b += nt;
                }
                (n, over, worst, at)
            })
        })
        .collect();
    let (mut n, mut over, mut worst, mut at) = (0u64, 0u64, 0u64, 0u32);
    for h in hs {
        let r = h.join().unwrap();
        n += r.0;
        over += r.1;
        if r.2 > worst {
            worst = r.2;
            at = r.3;
        }
    }
    println!("lg2f exhaustive: {n} inputs, max ulp {worst} at {at:#010x} ({:e}), {over} inputs > 2 ulp", f32::from_bits(at));
    assert_eq!(n, 0x7f7f_ffff);
    assert_eq!(over, 0, "{over} inputs > 2 ulp; max {worst} at {at:#010x}");
}

#[test]
fn sinf_cosf_within_2ulp_reduced_range() {
    check("sinf", linspace(-105_615.0, 105_615.0, 2_000_001), sinf, f64::sin);
    check("cosf", linspace(-105_615.0, 105_615.0, 2_000_001), cosf, f64::cos);
}

/// Every 2048th f32 bit pattern with finite |x| > 105615 (both signs): the Payne-Hanek range.
fn large_arguments() -> impl Iterator<Item = f32> {
    (0u32..1 << 21)
        .map(|i| f32::from_bits(i << 11))
        .filter(|x| x.is_finite() && x.abs() > 105_615.0)
}

/// The reference is glibc's f64 sin/cos, which does its own exact (Payne-Hanek) reduction for huge
/// arguments, so it is accurate across the whole finite range and a valid reference here.
#[test]
fn sinf_cosf_within_2ulp_full_range() {
    let n_large = large_arguments().count();
    assert!(n_large > 900_000, "{n_large} large arguments");
    let xs = || linspace(-105_615.0, 105_615.0, 2_000_001).chain(large_arguments());
    check("sinf full range", xs(), sinf, f64::sin);
    check("cosf full range", xs(), cosf, f64::cos);
    check("sinf |x|>105615", large_arguments(), sinf, f64::sin);
    check("cosf |x|>105615", large_arguments(), cosf, f64::cos);
}

#[test]
fn sinf_cosf_bounded() {
    let mut n = 0u64;
    for x in (0u32..1 << 24).map(|i| f32::from_bits(i << 8)).filter(|x| x.is_finite()) {
        for y in [sinf(x), cosf(x)] {
            assert!(y.abs() <= 1.0, "x={x:e} ({:#010x}) gave {y:e}", x.to_bits());
        }
        n += 1;
    }
    println!("sinf/cosf bounded: {n} finite inputs (every 256th bit pattern)");
}

#[test]
fn special_values() {
    assert!(ex2f(f32::NAN).is_nan());
    assert_eq!(ex2f(f32::INFINITY), f32::INFINITY);
    assert_eq!(ex2f(f32::NEG_INFINITY).to_bits(), 0.0f32.to_bits());
    assert_eq!(ex2f(0.0), 1.0);
    assert_eq!(ex2f(128.0), f32::INFINITY);
    assert_eq!(ex2f(-150.5).to_bits(), 0.0f32.to_bits());
    assert_eq!(ex2f(-149.0).to_bits(), 1, "2^-149 is the smallest subnormal");
    assert_eq!(ex2f(-126.0), f32::MIN_POSITIVE);

    assert_eq!(lg2f(0.0), f32::NEG_INFINITY);
    assert_eq!(lg2f(-0.0), f32::NEG_INFINITY);
    assert!(lg2f(-1.0).is_nan());
    assert!(lg2f(f32::NAN).is_nan());
    assert_eq!(lg2f(1.0).to_bits(), 0.0f32.to_bits());
    assert_eq!(lg2f(f32::INFINITY), f32::INFINITY);
    assert_eq!(lg2f(f32::from_bits(1)), -149.0, "smallest subnormal");
    assert_eq!(lg2f(f32::MIN_POSITIVE), -126.0);

    assert!(sinf(f32::INFINITY).is_nan());
    assert!(sinf(f32::NEG_INFINITY).is_nan());
    assert!(cosf(f32::INFINITY).is_nan());
    assert!(cosf(f32::NAN).is_nan());
    assert_eq!(cosf(0.0), 1.0);
    assert_eq!(sinf(0.0).to_bits(), 0.0f32.to_bits());
    assert_eq!(sinf(-0.0).to_bits(), (-0.0f32).to_bits(), "sin(-0) keeps its sign");
    assert_eq!(sinf(f32::from_bits(0x8000_0001)).to_bits(), 0x8000_0001, "sin(subnormal) = subnormal");

    // Every NaN result is the GPU's canonical NaN, so GPU and oracle agree bit for bit.
    for v in [ex2f(f32::NAN), lg2f(-1.0), lg2f(f32::from_bits(0xffc0_0000)), sinf(f32::INFINITY), cosf(f32::NAN)] {
        assert_eq!(v.to_bits(), 0x7fff_ffff);
    }
}

/// The oracle's software fma must equal a correctly rounded fma. `f32::mul_add` (libm fmaf here)
/// is the reference; the inputs are chosen so products and sums land in the subnormal range,
/// near ties, and with heavy cancellation.
#[test]
fn fma_is_correctly_rounded() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x4543_484f);
    let mut checked = 0u64;
    for i in 0..2_000_000u32 {
        let (a, b, c) = match i % 5 {
            4 => {
                // Double-rounding family: a*b = ±2^(E-24)(1 - 2^-2j) with j >= 15, c in [2^E, 2^(E+1)).
                // The exact sum lies within 2^(E-54) of an f32 midpoint, so a plain f64 sum rounds
                // onto the midpoint and then to the wrong (even) neighbour about half the time.
                let j = rng.gen_range(15..=23);
                let c = f32::from_bits(rng.gen_range(0x0080_0000u32..0x7e00_0000) | (rng.gen::<u32>() & 0x8000_0000));
                let e = ((c.to_bits() >> 23) & 0xff) as i32 - 127;
                let sign = if rng.gen::<bool>() { 1.0f32 } else { -1.0 };
                let scale = 2f32.powi(e - 24);
                let u = 2f32.powi(-j);
                (sign * scale * (1.0 + u), 1.0 - u, c)
            }
            0 => (f32::from_bits(rng.gen()), f32::from_bits(rng.gen()), f32::from_bits(rng.gen())),
            1 => {
                // product near 2^-126..2^-149, addend subnormal
                let a = f32::from_bits(rng.gen_range(0x1000_0000..0x2000_0000));
                let b = f32::from_bits(rng.gen_range(0x1000_0000..0x2000_0000) | (rng.gen::<u32>() & 0x8000_0000));
                (a, b, f32::from_bits(rng.gen::<u32>() & 0x807f_ffff))
            }
            2 => {
                // c = -round(a*b) + small perturbation: catastrophic cancellation
                let a = f32::from_bits(rng.gen_range(0x3000_0000..0x5000_0000));
                let b = f32::from_bits(rng.gen_range(0x3000_0000..0x5000_0000));
                let p = a * b;
                (a, b, f32::from_bits((-p).to_bits().wrapping_add(rng.gen_range(0..4))))
            }
            _ => {
                // operands with few mantissa bits, so exact results often sit at a tie
                let a = f32::from_bits(rng.gen_range(0x3f80_0000u32..0x4000_0000) & 0xffff_f000);
                let b = f32::from_bits(rng.gen_range(0x3f80_0000u32..0x4000_0000) & 0xffff_f000);
                let c = f32::from_bits(rng.gen_range(0x3000_0000u32..0x4100_0000) & 0xffff_ff00);
                (a, b, c)
            }
        };
        let want = a.mul_add(b, c);
        let got = fma(a, b, c);
        if want.is_nan() {
            assert!(got.is_nan(), "fma({a:e}, {b:e}, {c:e})");
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "fma({a:e}, {b:e}, {c:e}) = {got:e}, want {want:e}");
        }
        checked += 1;
    }
    // Hand-picked double-rounding traps: the exact result is just below an f32 midpoint, but the
    // f64 sum rounds onto the midpoint, and ties-to-even then picks the wrong neighbour.
    let u15 = 2f32.powi(-15);
    let u16 = 2f32.powi(-16);
    let traps = [
        // 1 + 2^-23 + 2^-24 - 2^-54 -> correct 1 + 2^-23; naive f64 path gives 1 + 2^-22
        (2f32.powi(-24) * (1.0 + u15), 1.0 - u15, 1.0 + f32::EPSILON, 0x3f80_0001u32),
        // largest subnormal + 2^-150 - 2^-182 -> correct 0x007fffff; naive gives 0x00800000
        (2f32.powi(-75) * (1.0 + u16), 2f32.powi(-75) * (1.0 - u16), f32::from_bits(0x007f_ffff), 0x007f_ffff),
    ];
    for (a, b, c, want) in traps {
        assert_eq!(a.mul_add(b, c).to_bits(), want, "libm reference for fma({a:e}, {b:e}, {c:e})");
        assert_eq!(fma(a, b, c).to_bits(), want, "fma({a:e}, {b:e}, {c:e})");
    }
    println!("fma: {checked} random triples equal libm fmaf");
}

#[test]
fn rint_rounds_half_to_even() {
    for (x, want) in [(0.5f32, 0.0f32), (1.5, 2.0), (2.5, 2.0), (-2.5, -2.0), (-0.3, -0.0), (8_388_607.5, 8_388_608.0)] {
        assert_eq!(rint(x).to_bits(), want.to_bits(), "rint({x})");
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    for _ in 0..1_000_000 {
        let x = f32::from_bits(rng.gen());
        let want = x.round_ties_even();
        if x.is_nan() {
            assert!(rint(x).is_nan());
        } else {
            assert_eq!(rint(x).to_bits(), want.to_bits(), "rint({x:e})");
        }
    }
}
