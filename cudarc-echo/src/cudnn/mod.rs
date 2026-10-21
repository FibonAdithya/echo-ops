//! ECHO cuDNN: the part of the cuDNN API that TIG's c006 challenge uses (`Cudnn`, 4-D tensor
//! descriptors and the three SPATIAL batch-norm calls on `N x C x 1 x 1` float tensors),
//! computed by ECHO's fixed-order kernels instead of libcudnn (plan Task 7b). Same paths,
//! names and signatures as tig-foundation/cudarc at b3fccf5; the `sys` functions keep that
//! fork's fuel and runtime-signature charges verbatim.
//!
//! Anything other than NCHW, FLOAT, SPATIAL and `h*w == 1` returns
//! `CUDNN_STATUS_NOT_SUPPORTED`. Every other cuDNN function is absent.

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
