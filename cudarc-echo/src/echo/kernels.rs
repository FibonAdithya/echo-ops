//! ECHO's BLAS and batch-norm kernels (echo-kernels/echo_blas.cu, echo_bn.cu) as used by the
//! slim `cublas` and `cudnn` modules (plan Task 7b).
//!
//! The kernels are obtained only through the public module API (Ruling R27): a PTX module
//! with a header and no code is loaded with `CudaContext::load_module`, which splices the ECHO
//! library into it, and the kernels are looked up by name. Each cuBLAS / cuDNN handle owns one
//! such module.
//!
//! Launches go straight to `cuLaunchKernel` on the handle's raw stream, as the real libraries
//! do: no fuel check, no event bookkeeping (the safe wrappers record their slices' events).

use crate::driver::{result, sys, CudaContext, CudaFunction, CudaModule, DriverError};
use crate::nvrtc::Ptx;
use core::ffi::c_void;
use std::sync::Arc;

/// A module with no code of its own; `load_module` links the ECHO library into it.
///
/// Header as in echo-kernels/ptx/echo_lib.ptx except `.version` (ruling R41): 7.0, the lowest
/// PTX ISA tried (7.0, 7.5, 7.8, 8.0, 8.5, 8.7) for which this module with the library spliced
/// in assembles with CUDA 12.8 `ptxas -arch=sm_70` and `-arch=sm_89` (MEASURED on tig-gpu-alt,
/// final fix wave). The library body needs nothing newer, so this module does not raise the
/// driver floor above the algorithm's own PTX (see the crate docs in `echo/mod.rs`).
pub(crate) const LIB_HOST_PTX: &str = "//\n// ECHO: host module for the library kernels\n//\n\n.version 7.0\n.target sm_70\n.address_size 64\n";

pub(crate) struct Kernels {
    ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    pub(crate) sgemm: CudaFunction,
    pub(crate) bn_fwd_train: CudaFunction,
    pub(crate) bn_fwd_infer: CudaFunction,
    pub(crate) bn_bwd: CudaFunction,
}

impl Kernels {
    /// Loads the library module into `ctx` and looks up the four kernels.
    pub(crate) fn load(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        let module = ctx.load_module(Ptx::from_src(LIB_HOST_PTX))?;
        Ok(Self {
            ctx: ctx.clone(),
            sgemm: module.load_function("__echo_sgemm")?,
            bn_fwd_train: module.load_function("__echo_bn_fwd_train")?,
            bn_fwd_infer: module.load_function("__echo_bn_fwd_infer")?,
            bn_bwd: module.load_function("__echo_bn_bwd")?,
            _module: module,
        })
    }

    /// Waits for all work on `stream` (a handle's stream). Called by `cublasDestroy_v2` and
    /// `cudnnDestroy` before the handle's kernel module is dropped, so the module is never
    /// unloaded under a running kernel; the real cuBLAS / cuDNN destroy calls synchronize too.
    ///
    /// # Safety
    /// `stream` must be a live stream of this context, or null (the legacy default stream).
    pub(crate) unsafe fn synchronize(&self, stream: sys::CUstream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        result::stream::synchronize(stream)
    }

    /// Launches `f` on `stream` with dynamic shared memory 0.
    ///
    /// # Safety
    /// `params` must point at values matching the kernel's parameter list, and every device
    /// pointer among them must be valid for the accesses the kernel makes.
    pub(crate) unsafe fn launch(
        &self,
        f: &CudaFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        stream: sys::CUstream,
        params: &mut [*mut c_void],
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        result::launch_kernel(f.cu_function, grid, block, 0, stream, params)
    }
}

/// `ceil(a / b)` for the launch grids.
pub(crate) fn blocks(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// Kernel parameter pointer for `cuLaunchKernel`.
pub(crate) fn p<T>(v: &T) -> *mut c_void {
    v as *const T as *mut c_void
}
