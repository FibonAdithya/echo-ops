//! Unit tests of the ECHO runtime glue. The GPU test is `#[ignore]`; run it alone:
//!   cargo test --features cuda-version-from-build-system --lib echo:: -- --ignored --test-threads=1

use super::*;
use crate::driver::{CudaContext, LaunchConfig, PushKernelArg};
use crate::nvrtc::Ptx;
use crate::RTSigFuel;
use std::sync::atomic::Ordering;

const FASTMATH: &str = include_str!("../../../echo-kernels/fixtures/fastmath.ptx");
const ATOMICS: &str = include_str!("../../../echo-kernels/fixtures/atomics.ptx");
const ATOMIC_USED: &str = include_str!("../../../echo-kernels/fixtures/atomic_used.ptx");

#[test]
fn lib_body_drops_only_the_header() {
    let body = lib_body();
    assert!(!body.contains(".version") && !body.contains(".target") && !body.contains(".address_size"));
    assert!(LIB_PTX.ends_with(body));
    assert!(body.contains(".visible .entry __echo_atomic_finalize()"));
    assert!(body.contains(".visible .global .align 4 .u32 __echo_acc_error;"));
}

#[test]
fn fixtures_normalize_with_embedded_lib() {
    let n = normalize(FASTMATH).unwrap();
    assert!(n.ptx.contains(lib_body()), "library not spliced");
    assert!(echo_core::ptx::hazards(&n.ptx).is_empty());
    assert!(n.finalize_kernels.is_empty());
    // loading already-normalized text again is a no-op (ruling R16)
    assert_eq!(normalize(&n.ptx).unwrap().ptx, n.ptx);

    let a = normalize(ATOMICS).unwrap();
    assert_eq!(a.finalize_kernels.iter().collect::<Vec<_>>(), vec!["atomics"]);

    let e = echo_core::ptx::normalize(ATOMIC_USED, lib_body()).unwrap_err();
    assert!(e.to_string().starts_with("ECHO: atomic_used:"), "{e}");
    assert!(e.reason.contains("return value"), "{e}");
    let d = normalize(ATOMIC_USED).unwrap_err();
    assert_eq!(d.0, sys::cudaError_enum::CUDA_ERROR_INVALID_PTX);
}

#[test]
fn capacity_knob_never_panics() {
    use state::{capacity_from, DEFAULT_CAPACITY};
    assert_eq!(capacity_from(None), DEFAULT_CAPACITY);
    assert_eq!(capacity_from(Some("64")), 64);
    assert_eq!(capacity_from(Some(" 4096 ")), 4096);
    for bad in ["", "0", "-1", "abc", "1e6", "4294967296"] {
        assert_eq!(capacity_from(Some(bad)), DEFAULT_CAPACITY, "{bad:?}");
    }
}

#[test]
fn non_ptx_input_is_refused() {
    for bad in ["\x7fELF\x02\x01\x01", "", "not ptx at all"] {
        let e = normalize(bad).unwrap_err();
        assert_eq!(e.0, sys::cudaError_enum::CUDA_ERROR_INVALID_PTX, "{bad:?}");
    }
    let img: Vec<std::ffi::c_char> = [0xffu8, 0xfe, 0x00].iter().map(|&b| b as std::ffi::c_char).collect();
    assert!(image_text(&img).is_err());
}

#[cfg(feature = "runtime")]
#[test]
fn canonical_prop_fields() {
    let p = canonical_prop();
    assert_eq!(p.name[..5].iter().map(|&c| c as u8).collect::<Vec<_>>(), b"ECHO\0");
    assert_eq!(p.multiProcessorCount, 128);
    assert_eq!(p.totalGlobalMem, 4294967296);
    assert_eq!(p.maxGridSize, [2147483647, 65535, 65535]);
}

#[test]
#[ignore]
fn internal_memory_not_counted() {
    let ctx = CudaContext::new(0).unwrap();
    assert!(
        state::existing(ctx.cu_ctx).is_none(),
        "accumulator table already allocated in this process: run this test alone"
    );
    // From here on CudaStream::alloc* counts device memory, as in TIG's runtime.
    ctx.initial_memory_lock.store(true, Ordering::Relaxed);
    let stream = ctx.default_stream();
    let (n, k) = (4096usize, 16usize);
    let v = stream.memcpy_stod(&vec![1.0f32; n]).unwrap();
    let mut out = stream.alloc_zeros::<f32>(k).unwrap();

    // Control: an allocation through CudaStream is counted, so the check below can fail.
    let before = RTSigFuel::get_device_memory_used();
    let probe = stream.alloc_zeros::<u8>(1 << 20).unwrap();
    assert_eq!(RTSigFuel::get_device_memory_used(), before + (1 << 20));
    drop(probe);
    assert_eq!(RTSigFuel::get_device_memory_used(), before);

    let usage = *ctx.memory_usage.read().unwrap();
    let module = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
    let table = state::existing(ctx.cu_ctx).expect("table allocated at load_module");
    assert_eq!(table.capacity, state::DEFAULT_CAPACITY);
    let f = module.load_function("atomics").unwrap();
    assert!(f.echo_finalize);
    let (n_i, k_i) = (n as i32, k as i32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut out)
            .arg(&v)
            .arg(&n_i)
            .arg(&k_i)
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(RTSigFuel::get_device_memory_used(), before, "ECHO memory was counted");
    assert_eq!(*ctx.memory_usage.read().unwrap(), usage, "ECHO memory was counted");
    let host = stream.memcpy_dtov(&out).unwrap();
    assert_eq!(host, vec![(n / k) as f32; k]);
}

#[test]
#[ignore]
fn unflagged_kernel_launches_no_finalize() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let fm = ctx.load_module(Ptx::from_src(FASTMATH)).unwrap();
    let at = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
    let f = fm.load_function("fastmath").unwrap();
    let g = at.load_function("atomics").unwrap();
    assert!(!f.echo_finalize && g.echo_finalize);
    let n = 256usize;
    let x = stream.memcpy_stod(&vec![1.5f32; n]).unwrap();
    let mut out = stream.alloc_zeros::<f32>(8 * n).unwrap();
    let n_i = n as i32;
    let count = || state::FINALIZE_LAUNCHES.load(Ordering::Relaxed);
    let c0 = count();
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&x)
            .arg(&x)
            .arg(&x)
            .arg(&mut out)
            .arg(&n_i)
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(count(), c0, "an unflagged kernel enqueued a finalize");
    let mut acc = stream.alloc_zeros::<f32>(4).unwrap();
    let k_i = 4i32;
    unsafe {
        stream
            .launch_builder(&g)
            .arg(&mut acc)
            .arg(&x)
            .arg(&n_i)
            .arg(&k_i)
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(count(), c0 + 1, "a flagged kernel must enqueue exactly one finalize");
    assert_eq!(stream.memcpy_dtov(&acc).unwrap(), vec![96.0f32; 4]);
}

/// Task 10 (Ruling R35). In a submodule so libtest (which runs tests in name order) runs
/// them after `internal_memory_not_counted`, which needs a process with no table yet.
mod readback {
    use super::*;

    /// After an overflow-free flagged launch, data-returning copies check the accumulator
    /// error flag without any device-to-host copy of their own.
    #[test]
    #[ignore]
    fn flag_check_issues_no_dtoh_copy() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let at = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
        assert!(state::existing(ctx.cu_ctx).is_some(), "no table: the copies below would check nothing");
        let g = at.load_function("atomics").unwrap();
        assert!(g.echo_finalize);
        let n = 256usize;
        let x = stream.memcpy_stod(&vec![1.5f32; n]).unwrap();
        let mut acc = stream.alloc_zeros::<f32>(4).unwrap();
        let (n_i, k_i) = (n as i32, 4i32);
        let f0 = state::FINALIZE_LAUNCHES.load(Ordering::Relaxed);
        unsafe {
            stream
                .launch_builder(&g)
                .arg(&mut acc)
                .arg(&x)
                .arg(&n_i)
                .arg(&k_i)
                .launch(LaunchConfig::for_num_elems(n as u32))
        }
        .unwrap();
        assert_eq!(state::FINALIZE_LAUNCHES.load(Ordering::Relaxed), f0 + 1);
        let one = acc.slice(0..1);
        // Counted at the transport (driver::result::memcpy_dtoh_*), so any flag read through a
        // device copy shows up whether or not ECHO's own code counts it.
        let copies = || crate::driver::result::DTOH_COPIES.load(Ordering::Relaxed);
        let c0 = copies();
        for _ in 0..100 {
            assert_eq!(stream.memcpy_dtov(&one).unwrap(), vec![96.0f32]);
        }
        let n = copies() - c0;
        assert_eq!(n, 100, "100 memcpy_dtov of 1 element issued {n} device-to-host copies, not 100");
        let c1 = copies();
        stream.synchronize().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(copies() - c1, 0, "synchronize issued a device-to-host copy");
    }

    /// Allocations that are fully overwritten at once are not zeroed first; `alloc` is.
    #[test]
    #[ignore]
    fn stod_and_clone_skip_memset() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let count = || state::ALLOC_MEMSETS.load(Ordering::Relaxed);
        let src: Vec<u32> = (0..1000u32).map(|i| i.wrapping_mul(2654435761)).collect();

        let c0 = count();
        let a = stream.memcpy_stod(&src).unwrap();
        assert_eq!(count() - c0, 0, "memcpy_stod zeroed its allocation");

        let c1 = count();
        let b = a.clone();
        let c = stream.clone_dtod(&a.slice(10..500)).unwrap();
        assert_eq!(count() - c1, 0, "clone / clone_dtod zeroed their allocations");

        let c2 = count();
        let z = unsafe { stream.alloc::<u32>(1000) }.unwrap();
        assert_eq!(count() - c2, 1, "alloc must still zero its memory");

        assert_eq!(stream.memcpy_dtov(&a).unwrap(), src);
        assert_eq!(stream.memcpy_dtov(&b).unwrap(), src);
        assert_eq!(stream.memcpy_dtov(&c).unwrap(), src[10..500].to_vec());
        assert!(stream.memcpy_dtov(&z).unwrap().iter().all(|&w| w == 0));
    }
}

/// Task 7b: the slim cuBLAS / cuDNN modules charge fuel and mix the runtime signature exactly
/// as tig-foundation/cudarc b3fccf5 does. `add_fuel_used(x)` stores `5 * x`.
///
/// The fork's batch-norm wrappers read the x descriptor with `get_tensor_nd_info`, which calls
/// the charged `cudnnGetTensorNdDescriptor` twice: +2 fuel units per batch-norm call, and its
/// signature constant XORs in twice (net zero). The parity tool measures the same totals on
/// the real fork (harness/src/bin/parity.rs).
/// GPU tests: `cargo test --features cuda-version-from-build-system,cudnn --lib blas_dnn -- --ignored`.
#[cfg(all(feature = "cublas", feature = "cudnn"))]
mod blas_dnn {
    use super::*;
    use crate::cublas::{sys::cublasOperation_t as Op, sys::cublasStatus_t as BS, CudaBlas, Gemm, GemmConfig};
    use crate::cudnn::sys::{self as dnn, cudnnStatus_t as DS};
    use crate::cudnn::Cudnn;
    use core::ffi::c_void;
    use std::sync::Mutex;

    /// The fuel counter and signature are process globals: these tests take turns.
    static GLOBALS: Mutex<()> = Mutex::new(());

    const SIG_BLAS_CREATE: u64 = 0xE2BDD547F7392179;
    const SIG_BLAS_SET_STREAM: u64 = 0x1EAC85E3BF6C374E;
    const SIG_BLAS_DESTROY: u64 = 0x8A7BC086D0AADB9E;
    const SIG_SGEMM: u64 = 0x9BA28F54D3299160;
    const SIG_DNN_CREATE: u64 = 0x25D07AF2CE19A961;
    const SIG_DNN_SET_STREAM: u64 = 0x493654E1F00FB353;
    const SIG_DNN_DESTROY: u64 = 0x823C44E529F6FB35;
    const SIG_DESC_CREATE: u64 = 0x4A32D50DF1F25D24;
    const SIG_DESC_SET4D: u64 = 0x6C257A487802E52F;
    const SIG_DESC_DESTROY: u64 = 0x8806919860ADEB7E;
    const SIG_BN_TRAIN: u64 = 0x92A6234EDD0FA265;
    const SIG_BN_INFER: u64 = 0x081AAEB75F8682B1;
    const SIG_BN_BWD: u64 = 0x34C16E93BE2C8A33;
    /// Fuel units of the two `cudnnGetTensorNdDescriptor` calls inside each batch-norm call.
    const ND_INFO: u64 = 2;

    fn clear() {
        RTSigFuel::take_fuel_used();
        RTSigFuel::take_runtime_signature();
    }

    fn take() -> (u64, u64) {
        (RTSigFuel::take_fuel_used(), RTSigFuel::take_runtime_signature())
    }

    unsafe fn desc(
        format: dnn::cudnnTensorFormat_t,
        dtype: dnn::cudnnDataType_t,
        d: [i32; 4],
    ) -> dnn::cudnnTensorDescriptor_t {
        let mut t = std::ptr::null_mut();
        assert_eq!(dnn::cudnnCreateTensorDescriptor(&mut t), DS::CUDNN_STATUS_SUCCESS);
        let st = dnn::cudnnSetTensor4dDescriptor(t, format, dtype, d[0], d[1], d[2], d[3]);
        assert_eq!(st, DS::CUDNN_STATUS_SUCCESS);
        t
    }

    unsafe fn nchw(n: i32, c: i32) -> dnn::cudnnTensorDescriptor_t {
        use dnn::{cudnnDataType_t as T, cudnnTensorFormat_t as F};
        desc(F::CUDNN_TENSOR_NCHW, T::CUDNN_DATA_FLOAT, [n, c, 1, 1])
    }

    /// Forward training on (x = y, bn) descriptors. ptr: x, y, scale, bias, rmean, rvar, smean, sinv.
    unsafe fn bn_train(
        h: dnn::cudnnHandle_t,
        mode: dnn::cudnnBatchNormMode_t,
        x: dnn::cudnnTensorDescriptor_t,
        bn: dnn::cudnnTensorDescriptor_t,
        ptr: [*mut c_void; 8],
    ) -> DS {
        let (a, b) = (1.0f32, 0.0f32);
        let (ap, bp) = (&a as *const f32 as *const c_void, &b as *const f32 as *const c_void);
        dnn::cudnnBatchNormalizationForwardTraining(
            h, mode, ap, bp, x, ptr[0], x, ptr[1], bn, ptr[2], ptr[3], 0.1, ptr[4], ptr[5], 1e-5,
            ptr[6], ptr[7],
        )
    }

    /// Backward on (x = dy = dx, bn). ptr: x, dy, dx, scale, dscale, dbias, smean, sinv.
    unsafe fn bn_bwd(
        h: dnn::cudnnHandle_t,
        x: dnn::cudnnTensorDescriptor_t,
        bn: dnn::cudnnTensorDescriptor_t,
        ptr: [*mut c_void; 8],
        beta_param: f32,
    ) -> DS {
        let (one, zero) = (1.0f32, 0.0f32);
        let o = &one as *const f32 as *const c_void;
        let z = &zero as *const f32 as *const c_void;
        let bpar = &beta_param as *const f32 as *const c_void;
        dnn::cudnnBatchNormalizationBackward(
            h,
            dnn::cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
            o, z, o, bpar, x, ptr[0], x, ptr[1], x, ptr[2], bn, ptr[3], ptr[4], ptr[5], 1e-5,
            ptr[6], ptr[7],
        )
    }

    /// No GPU needed: a handle without kernels still charges, then reports NOT_INITIALIZED.
    #[test]
    fn sys_charges_without_gpu() {
        let _g = GLOBALS.lock().unwrap_or_else(|e| e.into_inner());
        let fake = 0x1000 as *mut c_void; // never dereferenced: the call stops before launching
        let (n, c) = (104i32, 256i32);
        unsafe {
            clear();
            let mut h = std::ptr::null_mut();
            assert_eq!(dnn::cudnnCreate(&mut h), DS::CUDNN_STATUS_SUCCESS);
            assert_eq!(dnn::cudnnSetStream(h, std::ptr::null_mut()), DS::CUDNN_STATUS_SUCCESS);
            let x = nchw(n, c);
            let y = nchw(n, c);
            let bn = nchw(1, c);
            let spatial = dnn::cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL;
            let st = bn_train(h, spatial, x, bn, [fake; 8]);
            assert_eq!(st, DS::CUDNN_STATUS_NOT_INITIALIZED);
            let nc3 = (n * c) as u64 * 3;
            let sig = SIG_DNN_CREATE ^ SIG_DNN_SET_STREAM ^ SIG_DESC_CREATE ^ SIG_DESC_SET4D ^ SIG_BN_TRAIN;
            assert_eq!(take(), (5 * (1 + 1 + 6 + ND_INFO + nc3), sig));
            assert_eq!(bn_bwd(h, x, bn, [fake; 8], 1.0), DS::CUDNN_STATUS_NOT_INITIALIZED);
            assert_eq!(take(), (5 * (ND_INFO + nc3), SIG_BN_BWD));
            let (a, b) = (1.0f32, 0.0f32);
            let (ap, bp) = (&a as *const f32 as *const c_void, &b as *const f32 as *const c_void);
            let st = dnn::cudnnBatchNormalizationForwardInference(
                h, spatial, ap, bp, x, fake, y, fake, bn, fake, fake, fake, fake, 1e-5,
            );
            assert_eq!(st, DS::CUDNN_STATUS_NOT_INITIALIZED);
            assert_eq!(take(), (5 * (ND_INFO + nc3), SIG_BN_INFER));
            for d in [x, y, bn] {
                assert_eq!(dnn::cudnnDestroyTensorDescriptor(d), DS::CUDNN_STATUS_SUCCESS);
            }
            assert_eq!(dnn::cudnnDestroy(h), DS::CUDNN_STATUS_SUCCESS);
            assert_eq!(take(), (5 * 4, SIG_DESC_DESTROY ^ SIG_DNN_DESTROY));

            use crate::cublas::sys as bsys;
            let mut bh = std::ptr::null_mut();
            assert_eq!(bsys::cublasCreate_v2(&mut bh), BS::CUBLAS_STATUS_SUCCESS);
            let (one, zero) = (1.0f32, 0.0f32);
            let st = bsys::cublasSgemm_v2(
                bh, Op::CUBLAS_OP_N, Op::CUBLAS_OP_T, 2, 256, 104, &one, fake as _, 2, fake as _,
                256, &zero, fake as _, 2,
            );
            assert_eq!(st, BS::CUBLAS_STATUS_NOT_INITIALIZED);
            assert_eq!(bsys::cublasDestroy_v2(bh), BS::CUBLAS_STATUS_SUCCESS);
            let sig = SIG_BLAS_CREATE ^ SIG_SGEMM ^ SIG_BLAS_DESTROY;
            assert_eq!(take(), (5 * (1 + 2 * 256 * 104 + 1), sig));
        }
    }

    /// Unsupported layouts, types, modes and shapes are refused before anything is launched.
    #[test]
    fn unsupported_inputs_are_refused() {
        let _g = GLOBALS.lock().unwrap_or_else(|e| e.into_inner());
        use dnn::cudnnBatchNormMode_t as M;
        use dnn::cudnnDataType_t as T;
        use dnn::cudnnTensorFormat_t as F;
        let fake = 0x1000 as *mut c_void;
        unsafe {
            let mut h = std::ptr::null_mut();
            assert_eq!(dnn::cudnnCreate(&mut h), DS::CUDNN_STATUS_SUCCESS);
            let ok = nchw(8, 4);
            let bn = nchw(1, 4);
            let sp = M::CUDNN_BATCHNORM_SPATIAL;
            let not_sup = DS::CUDNN_STATUS_NOT_SUPPORTED;
            assert_eq!(bn_train(h, M::CUDNN_BATCHNORM_PER_ACTIVATION, ok, bn, [fake; 8]), not_sup);
            assert_eq!(bn_train(h, M::CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ok, bn, [fake; 8]), not_sup);
            for (f, t, d) in [
                (F::CUDNN_TENSOR_NHWC, T::CUDNN_DATA_FLOAT, [8, 4, 1, 1]),
                (F::CUDNN_TENSOR_NCHW, T::CUDNN_DATA_DOUBLE, [8, 4, 1, 1]),
                (F::CUDNN_TENSOR_NCHW, T::CUDNN_DATA_HALF, [8, 4, 1, 1]),
                (F::CUDNN_TENSOR_NCHW, T::CUDNN_DATA_FLOAT, [8, 4, 2, 1]),
                (F::CUDNN_TENSOR_NCHW, T::CUDNN_DATA_FLOAT, [8, 4, 1, 3]),
            ] {
                let x = desc(f, t, d);
                assert_eq!(bn_train(h, sp, x, bn, [fake; 8]), not_sup, "{f:?} {t:?} {d:?}");
                assert_eq!(bn_bwd(h, x, bn, [fake; 8], 0.0), not_sup, "{f:?} {t:?} {d:?}");
                dnn::cudnnDestroyTensorDescriptor(x);
            }
            // Shape mismatch between x and the parameter descriptor.
            let bn5 = nchw(1, 5);
            assert_eq!(bn_train(h, sp, ok, bn5, [fake; 8]), DS::CUDNN_STATUS_BAD_PARAM);
            // NULL running / saved statistics: ECHO always computes them.
            for i in 4..8 {
                let mut ptr = [fake; 8];
                ptr[i] = std::ptr::null_mut();
                assert_eq!(bn_train(h, sp, ok, bn, ptr), not_sup, "null arg {i}");
            }
            for i in 6..8 {
                let mut ptr = [fake; 8];
                ptr[i] = std::ptr::null_mut();
                assert_eq!(bn_bwd(h, ok, bn, ptr, 0.0), not_sup, "null arg {i}");
            }
            // An unset descriptor fails in the charged nd-info query.
            let mut unset = std::ptr::null_mut();
            dnn::cudnnCreateTensorDescriptor(&mut unset);
            assert_ne!(bn_train(h, sp, unset, bn, [fake; 8]), DS::CUDNN_STATUS_SUCCESS);
            let st = dnn::cudnnSetTensor4dDescriptor(unset, F::CUDNN_TENSOR_NCHW, T::CUDNN_DATA_FLOAT, 0, 4, 1, 1);
            assert_eq!(st, DS::CUDNN_STATUS_BAD_PARAM);
            for d in [ok, bn, bn5, unset] {
                dnn::cudnnDestroyTensorDescriptor(d);
            }
            dnn::cudnnDestroy(h);

            use crate::cublas::sys as bsys;
            let mut bh = std::ptr::null_mut();
            bsys::cublasCreate_v2(&mut bh);
            let (one, zero) = (1.0f32, 0.0f32);
            let sg = |ta, m: i32, k: i32, lda: i32| {
                bsys::cublasSgemm_v2(
                    bh, ta, Op::CUBLAS_OP_N, m, 4, k, &one, fake as _, lda, fake as _, k.max(1),
                    &zero, fake as _, m.max(1),
                )
            };
            assert_eq!(sg(Op::CUBLAS_OP_CONJG, 4, 4, 4), BS::CUBLAS_STATUS_INVALID_VALUE);
            assert_eq!(sg(Op::CUBLAS_OP_N, 4, 4, 3), BS::CUBLAS_STATUS_INVALID_VALUE); // lda < m
            assert_eq!(sg(Op::CUBLAS_OP_T, 8, 4, 3), BS::CUBLAS_STATUS_INVALID_VALUE); // lda < k
            assert_eq!(sg(Op::CUBLAS_OP_N, 0, 4, 1), BS::CUBLAS_STATUS_SUCCESS); // m == 0: no-op
            bsys::cublasDestroy_v2(bh);
            clear();
        }
    }

    #[test]
    fn host_module_links_the_kernels() {
        let n = normalize(kernels::LIB_HOST_PTX).unwrap();
        for k in ["__echo_sgemm(", "__echo_bn_fwd_train(", "__echo_bn_fwd_infer(", "__echo_bn_bwd("] {
            assert!(n.ptx.contains(&format!(".visible .entry {k}")), "{k} missing");
        }
        assert!(n.finalize_kernels.is_empty());
    }

    /// GPU: the challenge's call sequences (nn/linear.rs, nn/batch_norm.rs) through the public
    /// API charge `5 * (fork formula)`.
    #[test]
    #[ignore]
    fn fuel_matches_fork_formula() {
        let (f, _) = run_sequences();
        let (m, n, k) = (256u64, 128u64, 256u64);
        let (bn_n, bn_c) = (128u64, 256u64);
        assert_eq!(f[0], 5 * (1 + 1 + m * n * k), "CudaBlas::new + sgemm");
        assert_eq!(
            f[1],
            5 * (1 + 1 + 6 + ND_INFO + bn_n * bn_c * 3),
            "Cudnn::new + 3 descriptors + BN training"
        );
        assert_eq!(f[2], 5 * (ND_INFO + bn_n * bn_c * 3), "BN backward");
    }

    #[test]
    #[ignore]
    fn signature_matches_fork_constants() {
        let (_, s) = run_sequences();
        assert_eq!(s[0], SIG_BLAS_CREATE ^ SIG_BLAS_SET_STREAM ^ SIG_SGEMM);
        // The three descriptors' create/set constants XOR in three times each, i.e. once.
        let dnn = SIG_DNN_CREATE ^ SIG_DNN_SET_STREAM ^ SIG_DESC_CREATE ^ SIG_DESC_SET4D ^ SIG_BN_TRAIN;
        assert_eq!(s[1], dnn);
        assert_eq!(s[2], SIG_BN_BWD);
    }

    /// Runs (1) `CudaBlas::new` + one sgemm, (2) `Cudnn::new` + 3 descriptors + BN forward
    /// training, (3) one BN backward; returns the fuel and signature taken after each.
    fn run_sequences() -> ([u64; 3], [u64; 3]) {
        let _g = GLOBALS.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let (m, n, k) = (256usize, 128usize, 256usize);
        let a = stream.memcpy_stod(&vec![0.5f32; m * k]).unwrap();
        let b = stream.memcpy_stod(&vec![0.25f32; k * n]).unwrap();
        let mut c = stream.alloc_zeros::<f32>(m * n).unwrap();
        let (bn_n, bn_c) = (128i32, 256i32);
        let nc = (bn_n * bn_c) as usize;
        let x = stream.memcpy_stod(&(0..nc).map(|i| (i % 7) as f32).collect::<Vec<_>>()).unwrap();
        let bufs: Vec<_> = (0..9).map(|_| stream.memcpy_stod(&vec![1.0f32; nc]).unwrap()).collect();
        let dp = |s: &crate::driver::CudaSlice<f32>| s.cu_device_ptr as *mut c_void;
        let (mut fuel, mut sig) = ([0u64; 3], [0u64; 3]);
        clear();
        let blas = CudaBlas::new(stream.clone()).unwrap();
        let cfg = GemmConfig {
            transa: Op::CUBLAS_OP_N,
            transb: Op::CUBLAS_OP_N,
            m: m as i32,
            n: n as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: m as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: m as i32,
        };
        unsafe { blas.gemm(cfg, &a, &b, &mut c) }.unwrap();
        (fuel[0], sig[0]) = take();
        let cudnn = Cudnn::new(stream.clone()).unwrap();
        unsafe {
            let xd = nchw(bn_n, bn_c);
            let yd = nchw(bn_n, bn_c);
            let bnd = nchw(1, bn_c);
            let (one, zero) = (1.0f32, 0.0f32);
            let o = &one as *const f32 as *const c_void;
            let z = &zero as *const f32 as *const c_void;
            let st = dnn::cudnnBatchNormalizationForwardTraining(
                cudnn.handle,
                dnn::cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
                o, z, xd, dp(&x), yd, dp(&bufs[0]), bnd, dp(&bufs[1]), dp(&bufs[2]), 0.1,
                dp(&bufs[3]), dp(&bufs[4]), 1e-5, dp(&bufs[5]), dp(&bufs[6]),
            );
            assert_eq!(st, DS::CUDNN_STATUS_SUCCESS);
            (fuel[1], sig[1]) = take();
            let ptr = [
                dp(&x), dp(&bufs[0]), dp(&bufs[7]), dp(&bufs[1]), dp(&bufs[8]), dp(&bufs[2]),
                dp(&bufs[5]), dp(&bufs[6]),
            ];
            assert_eq!(bn_bwd(cudnn.handle, xd, bnd, ptr, 1.0), DS::CUDNN_STATUS_SUCCESS);
            (fuel[2], sig[2]) = take();
            stream.synchronize().unwrap();
            for d in [xd, yd, bnd] {
                dnn::cudnnDestroyTensorDescriptor(d);
            }
        }
        drop((blas, cudnn));
        clear();
        (fuel, sig)
    }
}
