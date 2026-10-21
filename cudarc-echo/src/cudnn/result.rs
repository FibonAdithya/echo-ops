//! A thin wrapper around [sys] providing [Result]s with [CudnnError].

use super::sys;

pub type CudnnResult<T> = Result<T, CudnnError>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CudnnError(pub sys::cudnnStatus_t);

impl sys::cudnnStatus_t {
    pub fn result(self) -> Result<(), CudnnError> {
        match self {
            sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(CudnnError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CudnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CudnnError {}

/// Creates a handle. See [sys::cudnnCreate].
pub fn create_handle() -> Result<sys::cudnnHandle_t, CudnnError> {
    let mut handle = std::ptr::null_mut();
    unsafe { sys::cudnnCreate(&mut handle) }.result()?;
    Ok(handle)
}

/// Destroys a handle previously created with [create_handle()].
///
/// # Safety
/// `handle` must not have been freed already.
pub unsafe fn destroy_handle(handle: sys::cudnnHandle_t) -> Result<(), CudnnError> {
    sys::cudnnDestroy(handle).result()
}

/// Sets the stream the handle's calls are enqueued on.
///
/// # Safety
/// `handle` and `stream` must be valid.
pub unsafe fn set_stream(
    handle: sys::cudnnHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CudnnError> {
    sys::cudnnSetStream(handle, stream).result()
}
