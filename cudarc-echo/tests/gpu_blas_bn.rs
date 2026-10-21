//! Task 7b (GPU): the slim `cudarc::cublas` / `cudarc::cudnn` modules, called the way the
//! challenge calls them, equal `echo_core::oracle` bit for bit. Every test is `#[ignore]`:
//!   cargo test --release --features cuda-version-from-build-system,cudnn --test gpu_blas_bn -- --ignored

use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig};
use cudarc::cudnn::{result::CudnnError, sys::*, Cudnn};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use echo_core::oracle::{bn_backward, bn_forward_inference, bn_forward_training, sgemm, Op};
use std::ffi::c_void;
use std::sync::Arc;

/// Challenge shapes `(out, in, batch)` (k = 1, m = 2, batches 72/104/128/250).
const SHAPES: [(usize, usize, usize); 5] = [(2, 256, 104), (256, 1, 128), (256, 256, 128), (256, 256, 250), (256, 256, 72)];
const BN_BATCHES: [usize; 4] = [72, 104, 128, 250];
const BN_C: usize = 256;
/// BatchNorm1d momentum and eps of the challenge (nn/mlp.rs).
const MOMENTUM: f64 = 0.1;
const EPS: f64 = 1e-5;

/// splitmix64: seeded, no dependencies.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)).collect()
    }
}

fn gpu() -> Arc<CudaStream> {
    CudaContext::new(0).expect("CudaContext::new(0)").default_stream()
}

/// Number of elements whose bits differ (prints the first few).
fn diff_bits(what: &str, got: &[f32], want: &[f32]) -> usize {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut bad = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            if bad < 3 {
                eprintln!("  {what}[{i}]: gpu {g:e} oracle {w:e}");
            }
            bad += 1;
        }
    }
    println!("{what}: {} elements, {bad} differ", got.len());
    bad
}

#[allow(clippy::too_many_arguments)]
fn gemm_case(
    s: &Arc<CudaStream>,
    blas: &CudaBlas,
    rng: &mut Rng,
    ta: Op,
    tb: Op,
    (m, n, k): (usize, usize, usize),
    (lda, ldb, ldc): (usize, usize, usize),
    alpha: f32,
    beta: f32,
) -> usize {
    let a = rng.uniform(lda * if ta == Op::N { k } else { m }, -1.0, 1.0);
    let b = rng.uniform(ldb * if tb == Op::N { n } else { k }, -1.0, 1.0);
    let c0 = if beta == 0.0 { vec![f32::NAN; ldc * n] } else { rng.uniform(ldc * n, -1.0, 1.0) };
    let mut want = c0.clone();
    sgemm(ta, tb, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut want, ldc);
    let op = |o: Op| if o == Op::N { cublasOperation_t::CUBLAS_OP_N } else { cublasOperation_t::CUBLAS_OP_T };
    let cfg = GemmConfig {
        transa: op(ta),
        transb: op(tb),
        m: m as i32,
        n: n as i32,
        k: k as i32,
        alpha,
        lda: lda as i32,
        ldb: ldb as i32,
        beta,
        ldc: ldc as i32,
    };
    let (da, db) = (s.memcpy_stod(&a).unwrap(), s.memcpy_stod(&b).unwrap());
    let mut dc = s.memcpy_stod(&c0).unwrap();
    unsafe { blas.gemm(cfg, &da, &db, &mut dc) }.unwrap();
    let got = s.memcpy_dtov(&dc).unwrap();
    diff_bits(&format!("gemm {ta:?}{tb:?} m={m} n={n} k={k} alpha={alpha} beta={beta}"), &got, &want)
}

#[test]
#[ignore]
fn sgemm_matches_oracle() {
    let s = gpu();
    let blas = CudaBlas::new(s.clone()).unwrap();
    let mut rng = Rng(0x7b01);
    let mut bad = 0;
    for &(out, inp, batch) in &SHAPES {
        // nn/linear.rs:88-99 forward
        bad += gemm_case(&s, &blas, &mut rng, Op::N, Op::N, (out, batch, inp), (out, inp, out), 1.0, 0.0);
        // nn/linear.rs:144-160 weight gradient, overwrite and accumulate
        for beta in [0.0f32, 1.0] {
            bad += gemm_case(&s, &blas, &mut rng, Op::N, Op::T, (out, inp, batch), (out, inp, out), 1.0, beta);
        }
        // nn/linear.rs:189-199 input gradient
        bad += gemm_case(&s, &blas, &mut rng, Op::T, Op::N, (inp, batch, out), (out, out, inp), 1.0, 0.0);
    }
    // alpha != 1, beta != 0 / 1, all transposes, padded leading dimensions, partial tiles.
    for ta in [Op::N, Op::T] {
        for tb in [Op::N, Op::T] {
            let (m, n, k) = (37, 19, 23);
            let lda = if ta == Op::N { m + 3 } else { k + 2 };
            let ldb = if tb == Op::N { k + 1 } else { n + 4 };
            bad += gemm_case(&s, &blas, &mut rng, ta, tb, (m, n, k), (lda, ldb, m + 5), 0.5, -0.75);
            bad += gemm_case(&s, &blas, &mut rng, ta, tb, (m, n, k), (lda, ldb, m + 5), -1.5, 0.0);
        }
    }
    s.synchronize().unwrap();
    assert_eq!(bad, 0, "elements differ from the oracle");
}

/// The challenge's CudnnTensorDescriptor (nn/tensor.rs), reduced to what the tests need.
struct Desc(cudnnTensorDescriptor_t);
impl Desc {
    fn new(n: usize, c: usize) -> Result<Self, CudnnError> {
        let mut d = std::ptr::null_mut();
        unsafe {
            match cudnnCreateTensorDescriptor(&mut d) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {}
                e => return Err(CudnnError(e)),
            }
            let fmt = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
            let dt = cudnnDataType_t::CUDNN_DATA_FLOAT;
            match cudnnSetTensor4dDescriptor(d, fmt, dt, n as i32, c as i32, 1, 1) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(Self(d)),
                e => Err(CudnnError(e)),
            }
        }
    }
}
impl Drop for Desc {
    fn drop(&mut self) {
        unsafe { cudnnDestroyTensorDescriptor(self.0) };
    }
}

fn vp(x: &f32) -> *const c_void {
    x as *const f32 as *const c_void
}

/// Device copy of `v`.
fn up(s: &Arc<CudaStream>, v: &[f32]) -> CudaSlice<f32> {
    s.memcpy_stod(v).unwrap()
}

/// Forward training through the cuDNN API vs the oracle; returns differing elements and the
/// oracle's saved mean / inverse std (for the backward tests).
fn train_case(s: &Arc<CudaStream>, dnn: &Cudnn, rng: &mut Rng, n: usize, c: usize, alpha: f32, beta: f32) -> (usize, Vec<f32>, Vec<f32>) {
    let x = rng.uniform(n * c, -2.0, 3.0);
    let scale = rng.uniform(c, 0.5, 1.5);
    let bias = rng.uniform(c, -0.5, 0.5);
    let rm0 = rng.uniform(c, -0.2, 0.2);
    let rv0 = rng.uniform(c, 0.5, 1.5);
    let y0 = if beta == 0.0 { vec![f32::NAN; n * c] } else { rng.uniform(n * c, -1.0, 1.0) };
    let (mut y, mut rm, mut rv) = (y0.clone(), rm0.clone(), rv0.clone());
    let (mut sm, mut si) = (vec![0.0; c], vec![0.0; c]);
    bn_forward_training(n, c, alpha, beta, &x, &mut y, &scale, &bias, MOMENTUM, &mut rm, &mut rv, EPS, &mut sm, &mut si);

    let (dx, dsc, dbi) = (up(s, &x), up(s, &scale), up(s, &bias));
    let (mut dy, mut drm, mut drv) = (up(s, &y0), up(s, &rm0), up(s, &rv0));
    let (mut dsm, mut dsi) = (s.alloc_zeros::<f32>(c).unwrap(), s.alloc_zeros::<f32>(c).unwrap());
    let (xd, yd, bd) = (Desc::new(n, c).unwrap(), Desc::new(n, c).unwrap(), Desc::new(1, c).unwrap());
    let st = unsafe {
        cudnnBatchNormalizationForwardTraining(
            dnn.handle,
            cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
            vp(&alpha),
            vp(&beta),
            xd.0,
            dx.device_ptr(s).0 as *const _,
            yd.0,
            dy.device_ptr_mut(s).0 as *mut _,
            bd.0,
            dsc.device_ptr(s).0 as *const _,
            dbi.device_ptr(s).0 as *const _,
            MOMENTUM,
            drm.device_ptr_mut(s).0 as *mut _,
            drv.device_ptr_mut(s).0 as *mut _,
            EPS,
            dsm.device_ptr_mut(s).0 as *mut _,
            dsi.device_ptr_mut(s).0 as *mut _,
        )
    };
    assert_eq!(st, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    let tag = format!("bn_train N={n} C={c} alpha={alpha} beta={beta}");
    let mut bad = diff_bits(&format!("{tag} y"), &s.memcpy_dtov(&dy).unwrap(), &y);
    bad += diff_bits(&format!("{tag} running_mean"), &s.memcpy_dtov(&drm).unwrap(), &rm);
    bad += diff_bits(&format!("{tag} running_var"), &s.memcpy_dtov(&drv).unwrap(), &rv);
    bad += diff_bits(&format!("{tag} saved_mean"), &s.memcpy_dtov(&dsm).unwrap(), &sm);
    bad += diff_bits(&format!("{tag} saved_inv_std"), &s.memcpy_dtov(&dsi).unwrap(), &si);
    (bad, sm, si)
}

#[test]
#[ignore]
fn bn_forward_training_matches_oracle() {
    let s = gpu();
    let dnn = Cudnn::new(s.clone()).unwrap();
    let mut rng = Rng(0x7b02);
    let mut bad = 0;
    for &n in &BN_BATCHES {
        bad += train_case(&s, &dnn, &mut rng, n, BN_C, 1.0, 0.0).0;
    }
    // alpha != 1 with beta != 0, and the N == 1 running-variance branch (var instead of ss/(N-1)).
    bad += train_case(&s, &dnn, &mut rng, 128, BN_C, 0.75, -0.5).0;
    bad += train_case(&s, &dnn, &mut rng, 1, BN_C, 1.0, 0.0).0;
    bad += train_case(&s, &dnn, &mut rng, 1, 37, 1.25, 0.5).0;
    assert_eq!(bad, 0, "elements differ from the oracle");
}

/// Guards the N == 1 test above: its running variance must take the `var` branch, so it has
/// to differ from what the N > 1 formula (ss / (N - 1) = 0 / 0) would give.
#[test]
fn n1_running_var_is_not_nan() {
    let (x, scale, bias) = (vec![0.3f32; 4], vec![1.0f32; 4], vec![0.0f32; 4]);
    let (mut y, mut rm, mut rv) = (vec![0.0; 4], vec![0.0; 4], vec![1.0; 4]);
    let (mut sm, mut si) = (vec![0.0; 4], vec![0.0; 4]);
    bn_forward_training(1, 4, 1.0, 0.0, &x, &mut y, &scale, &bias, MOMENTUM, &mut rm, &mut rv, EPS, &mut sm, &mut si);
    assert!(rv.iter().all(|v| v.is_finite()), "{rv:?}");
}

#[test]
#[ignore]
fn bn_forward_inference_matches_oracle() {
    let s = gpu();
    let dnn = Cudnn::new(s.clone()).unwrap();
    let mut rng = Rng(0x7b03);
    let mut bad = 0;
    let mut cases: Vec<(usize, f32, f32)> = BN_BATCHES.iter().map(|&n| (n, 1.0, 0.0)).collect();
    cases.push((128, 0.75, -0.5));
    for (n, alpha, beta) in cases {
        let c = BN_C;
        let x = rng.uniform(n * c, -2.0, 3.0);
        let scale = rng.uniform(c, 0.5, 1.5);
        let bias = rng.uniform(c, -0.5, 0.5);
        let rm = rng.uniform(c, -0.2, 0.2);
        let rv = rng.uniform(c, 0.5, 1.5);
        let y0 = if beta == 0.0 { vec![f32::NAN; n * c] } else { rng.uniform(n * c, -1.0, 1.0) };
        let mut y = y0.clone();
        bn_forward_inference(n, c, alpha, beta, &x, &mut y, &scale, &bias, &rm, &rv, EPS);
        let (dx, dsc, dbi, drm, drv) = (up(&s, &x), up(&s, &scale), up(&s, &bias), up(&s, &rm), up(&s, &rv));
        let mut dy = up(&s, &y0);
        let (xd, yd, bd) = (Desc::new(n, c).unwrap(), Desc::new(n, c).unwrap(), Desc::new(1, c).unwrap());
        let st = unsafe {
            cudnnBatchNormalizationForwardInference(
                dnn.handle,
                cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
                vp(&alpha),
                vp(&beta),
                xd.0,
                dx.device_ptr(&s).0 as *const _,
                yd.0,
                dy.device_ptr_mut(&s).0 as *mut _,
                bd.0,
                dsc.device_ptr(&s).0 as *const _,
                dbi.device_ptr(&s).0 as *const _,
                drm.device_ptr(&s).0 as *const _,
                drv.device_ptr(&s).0 as *const _,
                EPS,
            )
        };
        assert_eq!(st, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        bad += diff_bits(&format!("bn_infer N={n} alpha={alpha} beta={beta} y"), &s.memcpy_dtov(&dy).unwrap(), &y);
    }
    assert_eq!(bad, 0, "elements differ from the oracle");
}

#[test]
#[ignore]
fn bn_backward_matches_oracle() {
    let s = gpu();
    let dnn = Cudnn::new(s.clone()).unwrap();
    let mut rng = Rng(0x7b04);
    let mut bad = 0;
    // (N, alphaData, betaData, alphaParam, betaParam)
    let mut cases = vec![];
    for &n in &BN_BATCHES {
        for bp in [0.0f32, 1.0] {
            cases.push((n, 1.0f32, 0.0f32, 1.0f32, bp));
        }
    }
    cases.push((128, 0.5, 0.25, 1.5, -0.75)); // alphaData != 1, betaData != 0
    cases.push((104, 1.0, 1.0, 1.0, 1.0));
    for (n, ad, bd, ap, bpar) in cases {
        let c = BN_C;
        // Saved statistics as the forward pass leaves them.
        let (_, sm, si) = train_case(&s, &dnn, &mut rng, n, c, 1.0, 0.0);
        let x = rng.uniform(n * c, -2.0, 3.0);
        let dy = rng.uniform(n * c, -1.0, 1.0);
        let scale = rng.uniform(c, 0.5, 1.5);
        let dx0 = if bd == 0.0 { vec![f32::NAN; n * c] } else { rng.uniform(n * c, -1.0, 1.0) };
        let (dg0, db0) = if bpar == 0.0 {
            (vec![f32::NAN; c], vec![f32::NAN; c])
        } else {
            (rng.uniform(c, -1.0, 1.0), rng.uniform(c, -1.0, 1.0))
        };
        let (mut dx, mut dg, mut db) = (dx0.clone(), dg0.clone(), db0.clone());
        bn_backward(n, c, ad, bd, ap, bpar, &x, &dy, &mut dx, &scale, &mut dg, &mut db, EPS, &sm, &si);
        let (gx, gdy, gsc, gsm, gsi) = (up(&s, &x), up(&s, &dy), up(&s, &scale), up(&s, &sm), up(&s, &si));
        let (mut gdx, mut gdg, mut gdb) = (up(&s, &dx0), up(&s, &dg0), up(&s, &db0));
        let descs: Vec<Desc> = (0..3).map(|_| Desc::new(n, c).unwrap()).collect();
        let bdesc = Desc::new(1, c).unwrap();
        let st = unsafe {
            cudnnBatchNormalizationBackward(
                dnn.handle,
                cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
                vp(&ad),
                vp(&bd),
                vp(&ap),
                vp(&bpar),
                descs[0].0,
                gx.device_ptr(&s).0 as *const _,
                descs[1].0,
                gdy.device_ptr(&s).0 as *const _,
                descs[2].0,
                gdx.device_ptr_mut(&s).0 as *mut _,
                bdesc.0,
                gsc.device_ptr(&s).0 as *const _,
                gdg.device_ptr_mut(&s).0 as *mut _,
                gdb.device_ptr_mut(&s).0 as *mut _,
                EPS,
                gsm.device_ptr(&s).0 as *const _,
                gsi.device_ptr(&s).0 as *const _,
            )
        };
        assert_eq!(st, cudnnStatus_t::CUDNN_STATUS_SUCCESS);
        let tag = format!("bn_bwd N={n} aD={ad} bD={bd} aP={ap} bP={bpar}");
        bad += diff_bits(&format!("{tag} dx"), &s.memcpy_dtov(&gdx).unwrap(), &dx);
        bad += diff_bits(&format!("{tag} dscale"), &s.memcpy_dtov(&gdg).unwrap(), &dg);
        bad += diff_bits(&format!("{tag} dbias"), &s.memcpy_dtov(&gdb).unwrap(), &db);
    }
    assert_eq!(bad, 0, "elements differ from the oracle");
}
