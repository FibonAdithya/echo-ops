//! CPU oracles for ECHO's BLAS and batch-norm kernels (Task 7).
//!
//! These functions define every result bit of `echo-kernels/echo_blas.cu` and
//! `echo-kernels/echo_bn.cu`: the GPU kernels perform the same f32 operations in the same
//! order, each rounded to nearest even (`.rn`), and the GPU tests compare them bit for bit.
//! Rust never contracts `a * b + c` into an fma, and `f32::mul_add` is a single fused
//! multiply-add, so each line below is one IEEE operation.
//!
//! Layouts follow the libraries ECHO replaces:
//! - `sgemm` is column-major like cuBLAS: element `(i, j)` of a matrix with leading
//!   dimension `ld` is at `i + j * ld`.
//! - Batch norm uses cuDNN's NCHW float tensors of shape `N x C x 1 x 1`, so element
//!   `(n, c)` is at `n * C + c`; per-channel parameters have length `C`. SPATIAL mode only.
//!
//! `blend(v, old, alpha, beta)` = `beta == 0 ? alpha*v : fma(alpha, v, beta*old)`; with
//! `beta == 0` the old value is never read (cuBLAS/cuDNN semantics: it may be NaN).

/// Transpose flag, numbered like `cublasOperation_t` (N = 0, T = 1).
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    N = 0,
    T = 1,
}

/// `beta == 0 ? alpha*v : fma(alpha, v, beta*old)`. `old` is only evaluated when `beta != 0`.
#[inline]
fn blend(v: f32, old: impl FnOnce() -> f32, alpha: f32, beta: f32) -> f32 {
    if beta == 0.0 {
        alpha * v
    } else {
        alpha.mul_add(v, beta * old())
    }
}

#[inline]
fn at(op: Op, buf: &[f32], ld: usize, row: usize, col: usize) -> f32 {
    match op {
        Op::N => buf[row + col * ld],
        Op::T => buf[col + row * ld],
    }
}

/// `C = alpha * op(A) * op(B) + beta * C`, column-major, `op(A)` is `m x k`, `op(B)` is `k x n`.
///
/// Per element: `acc = 0; for p in 0..k ascending: acc = fma(opA[i,p], opB[p,j], acc)`, then
/// `C[i,j] = blend(acc, C[i,j], alpha, beta)`. Rows `m..ldc` of `C` are not touched.
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    transa: Op,
    transb: Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc = at(transa, a, lda, i, p).mul_add(at(transb, b, ldb, p, j), acc);
            }
            let idx = i + j * ldc;
            c[idx] = blend(acc, || c[idx], alpha, beta);
        }
    }
}

/// `cudnnBatchNormalizationForwardTraining`, SPATIAL, `N x C x 1 x 1`.
///
/// `exponential_average_factor` and `epsilon` are the cuDNN `double` arguments; both are
/// rounded to f32 (`as f32`) before use. Per channel, `n` ascending:
/// `sum += x`; `mean = sum / N`; `d = x - mean`, `ss = fma(d, d, ss)`; `var = ss / N`;
/// `invstd = rcp(sqrt(var + eps))`; `y = blend(fma(gamma, (x - mean) * invstd, beta_c), y, alpha, beta)`;
/// `rmean = fma(f, mean, (1 - f) * rmean)`; `rvar = fma(f, ss / (N - 1), (1 - f) * rvar)`
/// (`var` instead of `ss / (N - 1)` when `N == 1`); `saved_mean = mean`, `saved_invstd = invstd`.
/// Requires `n >= 1`.
#[allow(clippy::too_many_arguments)]
pub fn bn_forward_training(
    n: usize,
    c: usize,
    alpha: f32,
    beta: f32,
    x: &[f32],
    y: &mut [f32],
    bn_scale: &[f32],
    bn_bias: &[f32],
    exponential_average_factor: f64,
    running_mean: &mut [f32],
    running_var: &mut [f32],
    epsilon: f64,
    saved_mean: &mut [f32],
    saved_inv_variance: &mut [f32],
) {
    let nf = n as f32;
    let eps = epsilon as f32;
    let factor = exponential_average_factor as f32;
    let one_minus = 1.0f32 - factor;
    for ch in 0..c {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i * c + ch];
        }
        let mean = sum / nf;
        let mut ss = 0.0f32;
        for i in 0..n {
            let d = x[i * c + ch] - mean;
            ss = d.mul_add(d, ss);
        }
        let var = ss / nf;
        let invstd = 1.0f32 / (var + eps).sqrt();
        let (g, bc) = (bn_scale[ch], bn_bias[ch]);
        for i in 0..n {
            let k = i * c + ch;
            let v = g.mul_add((x[k] - mean) * invstd, bc);
            y[k] = blend(v, || y[k], alpha, beta);
        }
        running_mean[ch] = factor.mul_add(mean, one_minus * running_mean[ch]);
        let unbiased = if n > 1 { ss / ((n - 1) as f32) } else { var };
        running_var[ch] = factor.mul_add(unbiased, one_minus * running_var[ch]);
        saved_mean[ch] = mean;
        saved_inv_variance[ch] = invstd;
    }
}

/// `cudnnBatchNormalizationForwardInference`, SPATIAL, `N x C x 1 x 1`.
///
/// `y = blend(fma(gamma, (x - mean) * rcp(sqrt(var + eps)), beta_c), y, alpha, beta)` with
/// `eps = epsilon as f32`.
#[allow(clippy::too_many_arguments)]
pub fn bn_forward_inference(
    n: usize,
    c: usize,
    alpha: f32,
    beta: f32,
    x: &[f32],
    y: &mut [f32],
    bn_scale: &[f32],
    bn_bias: &[f32],
    estimated_mean: &[f32],
    estimated_variance: &[f32],
    epsilon: f64,
) {
    let eps = epsilon as f32;
    for i in 0..n {
        for ch in 0..c {
            let k = i * c + ch;
            let invstd = 1.0f32 / (estimated_variance[ch] + eps).sqrt();
            let v = bn_scale[ch].mul_add((x[k] - estimated_mean[ch]) * invstd, bn_bias[ch]);
            y[k] = blend(v, || y[k], alpha, beta);
        }
    }
}

/// `cudnnBatchNormalizationBackward`, SPATIAL, `N x C x 1 x 1`, using the saved mean and
/// inverse std from forward training (`epsilon` is accepted for signature parity but unused,
/// as in cuDNN when the saved statistics are given).
///
/// Per channel, `n` ascending: `xhat = (x - saved_mean) * saved_invstd`; `db += dy`;
/// `dg = fma(dy, xhat, dg)`; then `kk = (gamma * saved_invstd) / N` and
/// `dx = blend(kk * ((N * dy - db) - xhat * dg), dx, alpha_data, beta_data)`,
/// `dscale = blend(dg, dscale, alpha_param, beta_param)`, `dbias = blend(db, dbias, alpha_param, beta_param)`.
#[allow(clippy::too_many_arguments)]
pub fn bn_backward(
    n: usize,
    c: usize,
    alpha_data: f32,
    beta_data: f32,
    alpha_param: f32,
    beta_param: f32,
    x: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    bn_scale: &[f32],
    d_bn_scale: &mut [f32],
    d_bn_bias: &mut [f32],
    _epsilon: f64,
    saved_mean: &[f32],
    saved_inv_variance: &[f32],
) {
    let nf = n as f32;
    for ch in 0..c {
        let (m, s) = (saved_mean[ch], saved_inv_variance[ch]);
        let mut db = 0.0f32;
        let mut dg = 0.0f32;
        for i in 0..n {
            let k = i * c + ch;
            let xhat = (x[k] - m) * s;
            db += dy[k];
            dg = dy[k].mul_add(xhat, dg);
        }
        let kk = (bn_scale[ch] * s) / nf;
        for i in 0..n {
            let k = i * c + ch;
            let xhat = (x[k] - m) * s;
            let v = kk * ((nf * dy[k] - db) - xhat * dg);
            dx[k] = blend(v, || dx[k], alpha_data, beta_data);
        }
        d_bn_scale[ch] = blend(dg, || d_bn_scale[ch], alpha_param, beta_param);
        d_bn_bias[ch] = blend(db, || d_bn_bias[ch], alpha_param, beta_param);
    }
}
