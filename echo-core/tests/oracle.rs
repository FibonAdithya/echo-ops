//! CPU tests of echo_core::oracle against f64 formulas (Task 7, Step 1).
//! The oracles define ECHO's results bit for bit; these tests check that they compute the
//! right quantities (to f32 accuracy), and pin the parts of the operation order that can be
//! pinned with hand-built inputs.

use echo_core::oracle::{
    bn_backward, bn_forward_inference, bn_forward_training, sgemm, Op,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Challenge shapes `(out, in, batch)` (Review Focus 4: k = 1, m = 2, batches 72/104/128/250).
const SHAPES: [(usize, usize, usize); 5] = [
    (2, 256, 104),
    (256, 1, 128),
    (256, 256, 128),
    (256, 256, 250),
    (256, 256, 72),
];

fn uniform(rng: &mut StdRng, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(lo..hi)).collect()
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

/// Largest `|got - want| / max(|want|, rms(want))` over the buffer. A non-finite `got`
/// counts as infinite error (`f64::max` would otherwise drop a NaN silently).
fn max_scaled_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let r = rms(want);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| if g.is_finite() { (g as f64 - w).abs() / w.abs().max(r) } else { f64::INFINITY })
        .fold(0.0, f64::max)
}

fn op_at(op: Op, buf: &[f32], ld: usize, row: usize, col: usize) -> f32 {
    match op {
        Op::N => buf[row + col * ld],
        Op::T => buf[col + row * ld],
    }
}

/// One sgemm case against f64. Error per element is scaled by the dot product's own
/// magnitude `sum_p |opA[i,p]*opB[p,j]| + |beta*C_old|` (the standard dot-product error
/// measure; a plain relative error is meaningless where the sum cancels to near zero).
#[allow(clippy::too_many_arguments)]
fn check_sgemm(
    rng: &mut StdRng,
    ta: Op,
    tb: Op,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
) -> f64 {
    let a_len = lda * if ta == Op::N { k } else { m };
    let b_len = ldb * if tb == Op::N { n } else { k };
    let a = uniform(rng, a_len, -1.0, 1.0);
    let b = uniform(rng, b_len, -1.0, 1.0);
    // With beta == 0 the old C must not be read: fill it with NaN so reading it would show.
    let c_old: Vec<f32> = if beta == 0.0 {
        vec![f32::NAN; ldc * n]
    } else {
        uniform(rng, ldc * n, -1.0, 1.0)
    };
    let mut c = c_old.clone();
    sgemm(ta, tb, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);
    let mut worst = 0.0f64;
    for j in 0..n {
        for i in 0..m {
            let mut dot = 0.0f64;
            let mut mag = 0.0f64;
            for p in 0..k {
                let t = op_at(ta, &a, lda, i, p) as f64 * op_at(tb, &b, ldb, p, j) as f64;
                dot += t;
                mag += t.abs();
            }
            let (want, scale) = if beta == 0.0 {
                (alpha as f64 * dot, (alpha as f64).abs() * mag)
            } else {
                let old = beta as f64 * c_old[i + j * ldc] as f64;
                (alpha as f64 * dot + old, (alpha as f64).abs() * mag + old.abs())
            };
            let got = c[i + j * ldc] as f64;
            assert!(got.is_finite(), "C[{i},{j}] = {got} (beta = {beta})");
            let err = (got - want).abs() / scale.max(f64::MIN_POSITIVE);
            worst = worst.max(err);
        }
        // Padding rows i in m..ldc must be left untouched.
        for i in m..ldc {
            assert_eq!(c[i + j * ldc].to_bits(), c_old[i + j * ldc].to_bits(), "padding C[{i},{j}] written");
        }
    }
    worst
}

#[test]
fn sgemm_matches_f64_reference() {
    let mut rng = StdRng::seed_from_u64(0x7a01);
    let mut worst = 0.0f64;
    for &(out, inp, batch) in &SHAPES {
        // nn/linear.rs:88-99 forward: NN, m=out, n=batch, k=in, lda=out, ldb=in, ldc=out, beta=0.
        worst = worst.max(check_sgemm(&mut rng, Op::N, Op::N, out, batch, inp, out, inp, out, 1.0, 0.0));
        // nn/linear.rs:144-160 weight grad: NT, m=out, n=in, k=batch, lda=out, ldb=in, ldc=out, beta in {0,1}.
        for beta in [0.0f32, 1.0] {
            worst = worst.max(check_sgemm(&mut rng, Op::N, Op::T, out, inp, batch, out, inp, out, 1.0, beta));
        }
        // nn/linear.rs:189-199 input grad: TN, m=in, n=batch, k=out, lda=out, ldb=out, ldc=in, beta=0.
        worst = worst.max(check_sgemm(&mut rng, Op::T, Op::N, inp, batch, out, out, out, inp, 1.0, 0.0));
    }
    // General alpha/beta, all four transposes, padded leading dimensions.
    for ta in [Op::N, Op::T] {
        for tb in [Op::N, Op::T] {
            let (m, n, k) = (37, 19, 23);
            let lda = if ta == Op::N { m + 3 } else { k + 2 };
            let ldb = if tb == Op::N { k + 1 } else { n + 4 };
            worst = worst.max(check_sgemm(&mut rng, ta, tb, m, n, k, lda, ldb, m + 5, 0.5, -0.75));
            worst = worst.max(check_sgemm(&mut rng, ta, tb, m, n, k, lda, ldb, m + 5, -1.5, 0.0));
        }
    }
    println!("sgemm max scaled error vs f64: {worst:e}");
    assert!(worst <= 1e-5, "sgemm max scaled error {worst:e} > 1e-5");
}

/// Pins the accumulation order (p ascending) and the use of fma, with inputs where the
/// alternatives give different results.
#[test]
fn sgemm_accumulates_p_ascending_with_fma() {
    // 1x1 result, k = 3: opA = [1, 1, 1], opB = [1, 1e8, -1e8].
    // Ascending: 1 -> 1e8 (the 1 is lost, ulp(1e8) = 8) -> 0.  Descending: -1e8 -> 0 -> 1.
    let a = [1.0f32, 1.0, 1.0];
    let b = [1.0f32, 1e8, -1e8];
    let mut c = [f32::NAN];
    sgemm(Op::T, Op::N, 1, 1, 3, 1.0, &a, 3, &b, 3, 0.0, &mut c, 1);
    assert_eq!(c[0].to_bits(), 0.0f32.to_bits(), "p must ascend: got {}", c[0]);

    // k = 2: opA = [1, x], opB = [-(1 + 2^-11), x], x = 1 + 2^-12, so x*x = 1 + 2^-11 + 2^-24.
    // fma keeps the 2^-24; a separate multiply rounds x*x to 1 + 2^-11 and the sum is 0.
    let x = 1.0f32 + 2f32.powi(-12);
    let a = [1.0f32, x];
    let b = [-(1.0f32 + 2f32.powi(-11)), x];
    let mut c = [0.0f32];
    sgemm(Op::T, Op::N, 1, 1, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 1);
    assert_eq!(c[0], 2f32.powi(-24), "accumulation must use fma");

    // blend: beta != 0 uses fma(alpha, acc, beta*old): alpha = x, acc = x, beta*old = -(1+2^-11).
    let a = [x];
    let b = [1.0f32];
    let mut c = [1.0f32 + 2f32.powi(-11)];
    sgemm(Op::N, Op::N, 1, 1, 1, x, &a, 1, &b, 1, -1.0, &mut c, 1);
    assert_eq!(c[0], 2f32.powi(-24), "blend must be fma(alpha, acc, beta*old)");
}

struct BnCase {
    n: usize,
    c: usize,
    x: Vec<f32>,
    gamma: Vec<f32>,
    beta_c: Vec<f32>,
}

fn bn_case(rng: &mut StdRng, n: usize, c: usize) -> BnCase {
    // Per-channel offset and spread, so means and variances differ across channels.
    let off = uniform(rng, c, -2.0, 2.0);
    let spread = uniform(rng, c, 0.1, 3.0);
    let mut x = vec![0.0f32; n * c];
    for i in 0..n {
        for ch in 0..c {
            x[i * c + ch] = off[ch] + spread[ch] * rng.gen_range(-1.0f32..1.0);
        }
    }
    BnCase { n, c, gamma: uniform(rng, c, 0.5, 1.5), beta_c: uniform(rng, c, -0.5, 0.5), x }
}

/// f64 per-channel mean and biased variance of the f32 input.
fn stats64(bc: &BnCase) -> (Vec<f64>, Vec<f64>) {
    let (n, c) = (bc.n, bc.c);
    let mut mean = vec![0.0f64; c];
    let mut var = vec![0.0f64; c];
    for ch in 0..c {
        let m = (0..n).map(|i| bc.x[i * c + ch] as f64).sum::<f64>() / n as f64;
        let v = (0..n).map(|i| (bc.x[i * c + ch] as f64 - m).powi(2)).sum::<f64>() / n as f64;
        mean[ch] = m;
        var[ch] = v;
    }
    (mean, var)
}

#[test]
fn bn_training_matches_f64_formulas() {
    let mut rng = StdRng::seed_from_u64(0x7a02);
    let eps = 1e-5f64;
    for &n in &[72usize, 104, 128, 250] {
        let c = 256;
        for &(factor, alpha, beta) in &[(0.1f64, 1.0f32, 0.0f32), (1.0, 1.0, 0.0), (0.1, 0.5, 1.0)] {
            let bc = bn_case(&mut rng, n, c);
            let rmean0 = uniform(&mut rng, c, -1.0, 1.0);
            let rvar0 = uniform(&mut rng, c, 0.5, 2.0);
            let y0: Vec<f32> = if beta == 0.0 { vec![f32::NAN; n * c] } else { uniform(&mut rng, n * c, -1.0, 1.0) };
            let (mut y, mut rmean, mut rvar) = (y0.clone(), rmean0.clone(), rvar0.clone());
            let (mut smean, mut sinv) = (vec![0.0f32; c], vec![0.0f32; c]);
            bn_forward_training(
                n, c, alpha, beta, &bc.x, &mut y, &bc.gamma, &bc.beta_c, factor, &mut rmean, &mut rvar, eps,
                &mut smean, &mut sinv,
            );
            let (mean, var) = stats64(&bc);
            let inv: Vec<f64> = var.iter().map(|v| 1.0 / (v + eps).sqrt()).collect();
            let mut y64 = vec![0.0f64; n * c];
            for i in 0..n {
                for ch in 0..c {
                    let k = i * c + ch;
                    let v = bc.gamma[ch] as f64 * (bc.x[k] as f64 - mean[ch]) * inv[ch] + bc.beta_c[ch] as f64;
                    y64[k] = alpha as f64 * v + if beta == 0.0 { 0.0 } else { beta as f64 * y0[k] as f64 };
                }
            }
            let f = factor as f32 as f64;
            let rmean64: Vec<f64> = (0..c).map(|ch| f * mean[ch] + (1.0 - f) * rmean0[ch] as f64).collect();
            // Running variance uses the unbiased estimate ss/(N-1).
            let rvar64: Vec<f64> = (0..c)
                .map(|ch| f * var[ch] * n as f64 / (n - 1) as f64 + (1.0 - f) * rvar0[ch] as f64)
                .collect();
            let errs = [
                ("y", max_scaled_err(&y, &y64)),
                ("saved_mean", max_scaled_err(&smean, &mean)),
                ("saved_invstd", max_scaled_err(&sinv, &inv)),
                ("running_mean", max_scaled_err(&rmean, &rmean64)),
                ("running_var", max_scaled_err(&rvar, &rvar64)),
            ];
            for (name, e) in errs {
                assert!(e <= 1e-5, "N={n} factor={factor} beta={beta}: {name} scaled error {e:e} > 1e-5");
            }
        }
    }
}

#[test]
fn bn_training_n1_uses_biased_variance_for_running_var() {
    // N = 1: ss/(N-1) is undefined, so the running variance uses var (= 0 here).
    let x = [3.0f32, -2.0];
    let mut y = [0.0f32; 2];
    let (mut rmean, mut rvar) = ([0.0f32; 2], [1.0f32; 2]);
    let (mut sm, mut si) = ([0.0f32; 2], [0.0f32; 2]);
    bn_forward_training(1, 2, 1.0, 0.0, &x, &mut y, &[1.0; 2], &[0.0; 2], 0.5, &mut rmean, &mut rvar, 1e-5, &mut sm, &mut si);
    assert_eq!(sm, x);
    assert_eq!(rvar, [0.5f32, 0.5]);
    assert_eq!(rmean, [1.5f32, -1.0]);
    assert_eq!(y, [0.0f32, 0.0]);
}

#[test]
fn bn_inference_matches_f64_formulas() {
    let mut rng = StdRng::seed_from_u64(0x7a03);
    let eps = 1e-5f64;
    for &n in &[72usize, 104, 128, 250] {
        for &(alpha, beta) in &[(1.0f32, 0.0f32), (0.5, 1.0)] {
            let c = 256;
            let bc = bn_case(&mut rng, n, c);
            let rmean = uniform(&mut rng, c, -1.0, 1.0);
            let rvar = uniform(&mut rng, c, 0.01, 4.0);
            let y0: Vec<f32> = if beta == 0.0 { vec![f32::NAN; n * c] } else { uniform(&mut rng, n * c, -1.0, 1.0) };
            let mut y = y0.clone();
            bn_forward_inference(n, c, alpha, beta, &bc.x, &mut y, &bc.gamma, &bc.beta_c, &rmean, &rvar, eps);
            let mut y64 = vec![0.0f64; n * c];
            for i in 0..n {
                for ch in 0..c {
                    let k = i * c + ch;
                    let inv = 1.0 / (rvar[ch] as f64 + eps).sqrt();
                    let v = bc.gamma[ch] as f64 * (bc.x[k] as f64 - rmean[ch] as f64) * inv + bc.beta_c[ch] as f64;
                    y64[k] = alpha as f64 * v + if beta == 0.0 { 0.0 } else { beta as f64 * y0[k] as f64 };
                }
            }
            let e = max_scaled_err(&y, &y64);
            assert!(e <= 1e-5, "N={n} beta={beta}: y scaled error {e:e} > 1e-5");
        }
    }
}

#[test]
fn bn_backward_matches_f64_formulas() {
    let mut rng = StdRng::seed_from_u64(0x7a04);
    let eps = 1e-5f64;
    for &n in &[72usize, 104, 128, 250] {
        for &(beta_data, beta_param, alpha_param) in &[(0.0f32, 0.0f32, 1.0f32), (0.0, 1.0, 1.0), (1.0, 1.0, 0.5)] {
            let c = 256;
            let bc = bn_case(&mut rng, n, c);
            // Saved statistics from a forward-training pass, as in the challenge.
            let mut y = vec![0.0f32; n * c];
            let (mut rm, mut rv) = (vec![0.0f32; c], vec![1.0f32; c]);
            let (mut sm, mut si) = (vec![0.0f32; c], vec![0.0f32; c]);
            bn_forward_training(
                n, c, 1.0, 0.0, &bc.x, &mut y, &bc.gamma, &bc.beta_c, 0.1, &mut rm, &mut rv, eps, &mut sm, &mut si,
            );
            let dy = uniform(&mut rng, n * c, -1.0, 1.0);
            let nan_or = |rng: &mut StdRng, b: f32, len: usize| -> Vec<f32> {
                if b == 0.0 { vec![f32::NAN; len] } else { uniform(rng, len, -1.0, 1.0) }
            };
            let dx0 = nan_or(&mut rng, beta_data, n * c);
            let dg0 = nan_or(&mut rng, beta_param, c);
            let db0 = nan_or(&mut rng, beta_param, c);
            let (mut dx, mut dg, mut db) = (dx0.clone(), dg0.clone(), db0.clone());
            bn_backward(
                n, c, 1.0, beta_data, alpha_param, beta_param, &bc.x, &dy, &mut dx, &bc.gamma, &mut dg, &mut db,
                eps, &sm, &si,
            );
            let mut dx64 = vec![0.0f64; n * c];
            let mut dg64 = vec![0.0f64; c];
            let mut db64 = vec![0.0f64; c];
            for ch in 0..c {
                let (m, s, g) = (sm[ch] as f64, si[ch] as f64, bc.gamma[ch] as f64);
                let xhat: Vec<f64> = (0..n).map(|i| (bc.x[i * c + ch] as f64 - m) * s).collect();
                let sdy: f64 = (0..n).map(|i| dy[i * c + ch] as f64).sum();
                let sdyx: f64 = (0..n).map(|i| dy[i * c + ch] as f64 * xhat[i]).sum();
                for i in 0..n {
                    let k = i * c + ch;
                    let v = g * s / n as f64 * (n as f64 * dy[k] as f64 - sdy - xhat[i] * sdyx);
                    dx64[k] = v + if beta_data == 0.0 { 0.0 } else { beta_data as f64 * dx0[k] as f64 };
                }
                let acc = |old: f32| if beta_param == 0.0 { 0.0 } else { beta_param as f64 * old as f64 };
                dg64[ch] = alpha_param as f64 * sdyx + acc(dg0[ch]);
                db64[ch] = alpha_param as f64 * sdy + acc(db0[ch]);
            }
            for (name, e) in [
                ("dx", max_scaled_err(&dx, &dx64)),
                ("dscale", max_scaled_err(&dg, &dg64)),
                ("dbias", max_scaled_err(&db, &db64)),
            ] {
                assert!(e <= 1e-5, "N={n} betaData={beta_data} betaParam={beta_param}: {name} scaled error {e:e} > 1e-5");
            }
        }
    }
}
