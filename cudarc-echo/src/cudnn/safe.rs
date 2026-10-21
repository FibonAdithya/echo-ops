//! `Cudnn`, with the fields and signatures of tig-foundation/cudarc b3fccf5
//! `src/cudnn/safe/core.rs`, on top of the ECHO `sys` functions.

use super::{result, result::CudnnError, sys};
use crate::driver::CudaStream;
use crate::echo::kernels::Kernels;
use std::sync::Arc;

/// A handle to cuDNN. Holds one ECHO library module for its kernels.
#[derive(Debug)]
pub struct Cudnn {
    pub handle: sys::cudnnHandle_t,
    pub stream: Arc<CudaStream>,
}

impl Cudnn {
    /// Creates a new cudnn handle and sets the stream to the `device`'s stream.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(stream: Arc<CudaStream>) -> Result<Arc<Self>, CudnnError> {
        stream.context().bind_to_thread().unwrap();
        let handle = result::create_handle()?;
        // Dropping `cudnn` destroys the handle if a later step fails.
        let cudnn = Self { handle, stream };
        unsafe { result::set_stream(cudnn.handle, cudnn.stream.cu_stream() as *mut _) }?;
        let kernels = Kernels::load(cudnn.stream.context())
            .map_err(|_| CudnnError(sys::cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR))?;
        unsafe { sys::attach(cudnn.handle, kernels) }.result()?;
        Ok(Arc::new(cudnn))
    }

    /// Sets the handle's stream.
    ///
    /// # Safety
    /// This is unsafe because you can end up scheduling multiple concurrent kernels that all
    /// write to the same memory address.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CudnnError> {
        self.stream = stream;
        unsafe { result::set_stream(self.handle, self.stream.cu_stream() as *mut _) }
    }
}

impl Drop for Cudnn {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}
