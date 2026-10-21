//! ECHO cuBLAS: the part of the cuBLAS API that TIG's c006 challenge uses (`CudaBlas`,
//! `Gemm<f32>::gemm`), computed by ECHO's fixed-order `__echo_sgemm` kernel instead of
//! libcublas (plan Task 7b). Same paths, names and signatures as tig-foundation/cudarc at
//! b3fccf5; the `sys` functions keep that fork's fuel and runtime-signature charges verbatim.
//!
//! Not provided (using them is a compile error): f64/f16 `Gemm`, `Gemv`, strided-batched
//! gemm, pointer modes, and every other cuBLAS function.

pub mod safe;
#[allow(warnings)]
pub mod sys;

/// The fork's `cublas::result::CublasError` path.
pub mod result {
    pub use super::safe::CublasError;
}

pub use safe::*;
