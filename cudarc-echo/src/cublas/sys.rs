//! ECHO replacements for the cuBLAS `sys` functions the challenge reaches. Types and
//! signatures are copied from tig-foundation/cudarc b3fccf5 `src/cublas/sys/mod.rs`; each
//! function starts with that fork's `RTSigFuel` lines, verbatim.
//!
//! A handle is a boxed `BlasHandle` (stream + ECHO kernels) behind the opaque
//! `cublasHandle_t`. The kernels are attached by `CudaBlas::new` (`attach`); a handle made with
//! `cublasCreate_v2` alone has none, and `cublasSgemm_v2` on it returns NOT_INITIALIZED.
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::echo::kernels::{blocks, p, Kernels};
use crate::RTSigFuel;

pub type cublasHandle_t = *mut cublasContext;
pub type cudaStream_t = *mut CUstream_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_CONJG = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cublasContext {
    _unused: [u8; 0],
}

/// What a `cublasHandle_t` points at.
pub(crate) struct BlasHandle {
    stream: cudaStream_t,
    kernels: Option<Kernels>,
}

unsafe fn state<'a>(handle: cublasHandle_t) -> Option<&'a mut BlasHandle> {
    (handle as *mut BlasHandle).as_mut()
}

/// Gives the handle its ECHO kernels (one library module per handle).
pub(crate) unsafe fn attach(handle: cublasHandle_t, kernels: Kernels) -> cublasStatus_t {
    match state(handle) {
        Some(h) => {
            h.kernels = Some(kernels);
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        None => cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED,
    }
}

pub unsafe fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0xE2BDD547F7392179);
    if handle.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    let h = Box::new(BlasHandle { stream: std::ptr::null_mut(), kernels: None });
    *handle = Box::into_raw(h) as cublasHandle_t;
    cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

pub unsafe fn cublasSetStream_v2(
    handle: cublasHandle_t,
    streamId: cudaStream_t,
) -> cublasStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0x1EAC85E3BF6C374E);
    match state(handle) {
        Some(h) => {
            h.stream = streamId;
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        None => cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED,
    }
}

pub unsafe fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0x8A7BC086D0AADB9E);
    if handle.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED;
    }
    let h = Box::from_raw(handle as *mut BlasHandle);
    // Like the real cublasDestroy, wait for the handle's work before its kernel module goes.
    let synced = match &h.kernels {
        Some(k) => k.synchronize(h.stream as crate::driver::sys::CUstream),
        None => Ok(()),
    };
    drop(h);
    match synced {
        Ok(()) => cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        Err(_) => cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED,
    }
}

/// `C = alpha * op(A) * op(B) + beta * C`, column-major, by `__echo_sgemm`. `alpha` and `beta`
/// are host pointers (CUBLAS_POINTER_MODE_HOST, the default). `CUBLAS_OP_C` is `CUBLAS_OP_T`
/// for real matrices; `CUBLAS_OP_CONJG` is not supported.
pub unsafe fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::core::ffi::c_int,
    B: *const f32,
    ldb: ::core::ffi::c_int,
    beta: *const f32,
    C: *mut f32,
    ldc: ::core::ffi::c_int,
) -> cublasStatus_t {
    RTSigFuel::add_fuel_used(m as u64 * n as u64 * k as u64);
    RTSigFuel::mix_runtime_signature(0x9BA28F54D3299160);
    echo_sgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
}

fn op_flag(op: cublasOperation_t) -> Option<i32> {
    match op {
        cublasOperation_t::CUBLAS_OP_N => Some(0),
        cublasOperation_t::CUBLAS_OP_T | cublasOperation_t::CUBLAS_OP_C => Some(1),
        cublasOperation_t::CUBLAS_OP_CONJG => None,
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn echo_sgemm(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: *const f32,
    c: *mut f32,
    ldc: i32,
) -> cublasStatus_t {
    use cublasStatus_t::*;
    let Some(h) = state(handle) else { return CUBLAS_STATUS_NOT_INITIALIZED };
    let (Some(ta), Some(tb)) = (op_flag(transa), op_flag(transb)) else {
        return CUBLAS_STATUS_INVALID_VALUE;
    };
    if m < 0 || n < 0 || k < 0 {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let rows_a = if ta == 0 { m } else { k };
    let rows_b = if tb == 0 { k } else { n };
    if lda < rows_a.max(1) || ldb < rows_b.max(1) || ldc < m.max(1) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if alpha.is_null() || beta.is_null() {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if m == 0 || n == 0 {
        return CUBLAS_STATUS_SUCCESS;
    }
    if c.is_null() || (k > 0 && (a.is_null() || b.is_null())) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    let Some(kern) = h.kernels.as_ref() else { return CUBLAS_STATUS_NOT_INITIALIZED };
    let (alpha, beta) = (*alpha, *beta);
    let mut params = [
        p(&ta), p(&tb), p(&m), p(&n), p(&k), p(&alpha),
        p(&a), p(&lda), p(&b), p(&ldb), p(&beta), p(&c), p(&ldc),
    ];
    let grid = (blocks(m as usize, 16), blocks(n as usize, 16), 1);
    match kern.launch(&kern.sgemm, grid, (16, 16, 1), h.stream as _, &mut params) {
        Ok(()) => CUBLAS_STATUS_SUCCESS,
        Err(_) => CUBLAS_STATUS_EXECUTION_FAILED,
    }
}
