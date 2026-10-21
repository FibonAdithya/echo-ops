//! `CudaBlas` and `Gemm<f32>`, with the signatures of tig-foundation/cudarc b3fccf5
//! `src/cublas/safe.rs`, on top of the ECHO `sys` functions.
#![allow(clippy::too_many_arguments)]

use super::sys;
use crate::driver::{CudaStream, DevicePtr, DevicePtrMut};
use crate::echo::kernels::Kernels;
use core::ffi::c_int;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CublasError(pub sys::cublasStatus_t);

impl sys::cublasStatus_t {
    pub fn result(self) -> Result<(), CublasError> {
        match self {
            sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            _ => Err(CublasError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CublasError {}

/// Wrapper around [sys::cublasHandle_t]. Holds one ECHO library module for its kernels.
#[derive(Debug)]
pub struct CudaBlas {
    pub(crate) handle: sys::cublasHandle_t,
    pub(crate) stream: Arc<CudaStream>,
}

unsafe impl Send for CudaBlas {}
unsafe impl Sync for CudaBlas {}

impl CudaBlas {
    /// Creates a new cublas handle and sets the stream to the `device`'s stream.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CublasError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let mut handle = std::ptr::null_mut();
        unsafe { sys::cublasCreate_v2(&mut handle) }.result()?;
        // Dropping `blas` destroys the handle if a later step fails.
        let blas = Self { handle, stream };
        unsafe { sys::cublasSetStream_v2(blas.handle, blas.stream.cu_stream() as _) }.result()?;
        let kernels = Kernels::load(blas.stream.context())
            .map_err(|_| CublasError(sys::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED))?;
        unsafe { sys::attach(blas.handle, kernels) }.result()?;
        Ok(blas)
    }

    /// Returns a reference to the underlying cublas handle.
    pub fn handle(&self) -> &sys::cublasHandle_t {
        &self.handle
    }

    /// Sets the handle's stream.
    ///
    /// # Safety
    /// This is unsafe because you can end up scheduling multiple concurrent kernels that all
    /// write to the same memory address.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CublasError> {
        self.stream = stream;
        unsafe { sys::cublasSetStream_v2(self.handle, self.stream.cu_stream() as _) }.result()
    }
}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { sys::cublasDestroy_v2(handle) }.result().unwrap();
        }
    }
}

/// Configuration for [Gemm]
#[derive(Debug, Copy, Clone)]
pub struct GemmConfig<T> {
    pub transa: sys::cublasOperation_t,
    pub transb: sys::cublasOperation_t,
    pub m: c_int,
    pub n: c_int,
    pub k: c_int,
    pub alpha: T,
    pub lda: c_int,
    pub ldb: c_int,
    pub beta: T,
    pub ldc: c_int,
}

/// Matrix matrix multiplication with elements of type `T` (ECHO: `f32` only).
pub trait Gemm<T> {
    /// Matrix matrix multiplication. See
    /// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)
    ///
    /// # Safety
    /// This is unsafe because improper arguments may lead to invalid
    /// memory accesses.
    unsafe fn gemm<A: DevicePtr<T>, B: DevicePtr<T>, C: DevicePtrMut<T>>(
        &self,
        cfg: GemmConfig<T>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError>;
}

impl Gemm<f32> for CudaBlas {
    unsafe fn gemm<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        cfg: GemmConfig<f32>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError> {
        let (a, _record_a) = a.device_ptr(&self.stream);
        let (b, _record_b) = b.device_ptr(&self.stream);
        let (c, _record_c) = c.device_ptr_mut(&self.stream);
        sys::cublasSgemm_v2(
            self.handle,
            cfg.transa,
            cfg.transb,
            cfg.m,
            cfg.n,
            cfg.k,
            (&cfg.alpha) as *const _,
            a as *const _,
            cfg.lda,
            b as *const _,
            cfg.ldb,
            (&cfg.beta) as *const _,
            c as *mut _,
            cfg.ldc,
        )
        .result()
    }
}
