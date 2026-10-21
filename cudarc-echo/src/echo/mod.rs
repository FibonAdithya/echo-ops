//! ECHO runtime integration (plan Task 6).
//!
//! - `load_module` normalizes the algorithm's PTX with `echo_core::ptx::normalize` and links
//!   the ECHO device library (`echo-kernels/ptx/echo_lib.ptx`, embedded below) into it.
//! - Every loaded module gets its own copy of the library's `__echo_acc_*` globals; they are
//!   bound to the per-context accumulator table (`state.rs`).
//! - After a launch of a kernel in `finalize_kernels`, `__echo_atomic_finalize` and
//!   `__echo_atomic_reset` are enqueued on the same stream (`launch.rs`).
//! - `synchronize` and data-returning copies read the context's mapped host error flag,
//!   which `__echo_atomic_reset` sets from `__echo_acc_error` (Ruling R35).
//! - ECHO's own device memory goes through the raw driver allocator, so it is not counted
//!   against the algorithm's memory limit (`RTSigFuel::add_device_memory_used`).
//!
//! Scope (controller ruling R26): ECHO hooks the safe API only — `CudaContext::load_module`,
//! `LaunchArgs::launch`/`launch_cooperative`, `CudaStream`/`CudaContext::synchronize`,
//! `CudaStream::memcpy_dtoh`/`memcpy_dtov`, `CudaStream::alloc*` and
//! `runtime::result::device::get_device_prop`. The raw public `driver::result::module::load`/
//! `load_data`, `driver::result::launch_kernel` and everything in `driver::sys` bypass ECHO by
//! design; no existing algorithm uses them to load modules or launch kernels.
//!
//! # Minimum driver (ruling R41)
//!
//! The driver JIT-compiles every module ECHO loads, so it must support each module's PTX ISA
//! `.version`.
//! - ECHO's own module for the cuBLAS / cuDNN replacements (`kernels::LIB_HOST_PTX` with the
//!   library spliced in) declares `.version 7.0`. Of 7.0, 7.5, 7.8, 8.0, 8.5 and 8.7, every
//!   version assembles with CUDA 12.8 `ptxas -arch=sm_70` and `-arch=sm_89`; 7.0 is the lowest
//!   tried (MEASURED, tig-gpu-alt, final fix wave).
//! - An algorithm module keeps its own `.version`; the library body is spliced into it. TIG's
//!   pipeline (nvcc 12.6) emits `.version 8.5`. Two such modules (nn_iza, neural_advanced_v4
//!   from Task 9b) normalize with the library spliced and assemble for sm_70 and sm_89
//!   (MEASURED, final fix wave); Task 9b ran them through tig-runtime on the RTX 3080 Ti
//!   (MEASURED by Task 9b). The c006 harness PTX (nvcc 12.8) is `.version 8.7`.
//! - So the floor is set by the algorithm's PTX, not by ECHO: ISA 8.5 needs a CUDA 12.5 driver
//!   (Linux >= 555.42) and ISA 8.7 a CUDA 12.8 driver (Linux >= 570.26); ECHO's own module
//!   alone (ISA 7.0) would need a CUDA 11.0 driver (Linux >= 450.36). All three are ESTIMATES
//!   from NVIDIA's PTX ISA and CUDA release-note tables, not measured. Drivers actually run:
//!   570.211.01 (L40) and 580.105.08 (RTX 3080 Ti), MEASURED with nvidia-smi.
//!
//! # Known limitations (rulings R38, R43)
//!
//! - **Deferred float atomics change same-launch semantics.** `atomicAdd(float)` contributions
//!   go to the exact accumulator and become visible only when `__echo_atomic_finalize` runs
//!   after the kernel. A read of the target in the same launch sees the pre-launch value, and
//!   a plain store by the same thread to the same address after the atomic yields
//!   store + contributions instead of the store. Affected existing algorithms (from the
//!   whole-branch review; atomic / later store): `horizon` (kernel.cu:58/63), `nova_prime_opt`
//!   (kernel.cu:168/173), `nova_prime_fgd` and `nova_prime_opt2` (kernel.cu:162/167). In each,
//!   lane 0 of a warp does `atomicAdd(&updates[idx], …)` and then stores `updates[idx] = update`.
//!   Their results are identical across GPUs (MEASURED,
//!   Task 8) but differ in meaning from the source; the report's "Known limitations and open
//!   items" section gives their quality under ECHO against the baseline.
//! - **Integer atomics whose returned value is used** (ruling R19) are allowed, and the value
//!   returned depends on thread order. No static check catches an algorithm whose result
//!   depends on it; the end-to-end cross-GPU comparison is the only guard.
//! - **Untested GPU generations.** Only sm_86 (RTX 3080 Ti, RTX 3090) and sm_89 (L40) ran.
//!   Volta, Turing, Ampere GA100, Hopper and Blackwell are untested.
//! - **Host libm** on x86-64 (a spec risk for host-side maths in algorithms) is not measured.
//! - **Test knob and counters.** `ECHO_ACC_CAPACITY` (see `state::DEFAULT_CAPACITY`) is a test
//!   knob; an invalid value is ignored with a message. `state::FINALIZE_LAUNCHES`,
//!   `state::ALLOC_MEMSETS` (all builds) and `driver::result::DTOH_COPIES` (unit-test builds)
//!   are debug counters read only by tests; they do not affect results.

#[cfg(any(feature = "cublas", feature = "cudnn"))]
pub(crate) mod kernels;
pub(crate) mod state;
#[cfg(test)]
mod tests;

use crate::driver::{result::DriverError, sys};
use std::sync::OnceLock;

/// The ECHO device library, built by echo-kernels/build.sh from every echo-kernels/*.cu.
pub(crate) const LIB_PTX: &str = include_str!("../../../echo-kernels/ptx/echo_lib.ptx");

/// The library text that `normalize` splices after the module's `.address_size` line:
/// `LIB_PTX` without its header (leading comments, `.version`, `.target`, `.address_size`).
pub(crate) fn lib_body() -> &'static str {
    static BODY: OnceLock<&'static str> = OnceLock::new();
    BODY.get_or_init(|| {
        let mut off = 0;
        for line in LIB_PTX.split_inclusive('\n') {
            off += line.len();
            if line.trim_start().starts_with(".address_size") {
                return &LIB_PTX[off..];
            }
        }
        panic!("echo_lib.ptx has no .address_size line");
    })
}

/// `load_module` failure: the reason goes to stderr, the caller gets CUDA_ERROR_INVALID_PTX.
pub(crate) fn refuse(msg: &str) -> DriverError {
    eprintln!("{msg}");
    DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_PTX)
}

/// Bytes of a `PtxKind::Image` as PTX text, or a refusal for anything that is not PTX text
/// (a cubin or fatbin image).
pub(crate) fn image_text(image: &[std::ffi::c_char]) -> Result<String, DriverError> {
    let bytes: Vec<u8> = image.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    match String::from_utf8(bytes) {
        Ok(s) => Ok(s),
        Err(_) => Err(refuse("ECHO: load_module: image is not PTX text (cubin and fatbin images are refused)")),
    }
}

/// Normalizes `src` and links the library. Errors are printed as `ECHO: <kernel>:<line>: ...`.
pub(crate) fn normalize(src: &str) -> Result<echo_core::ptx::Normalized, DriverError> {
    let is_ptx = src
        .lines()
        .any(|l| l.trim_start().starts_with(".version"));
    if !is_ptx || src.starts_with('\x7f') {
        return Err(refuse("ECHO: load_module: input is not PTX text (cubin and fatbin images are refused)"));
    }
    echo_core::ptx::normalize(src, lib_body()).map_err(|e| refuse(&e.to_string()))
}

/// The canonical `cudaDeviceProp` returned by `get_device_prop` (values in `echo_core::prop`).
#[cfg(feature = "runtime")]
pub(crate) fn canonical_prop() -> crate::runtime::sys::cudaDeviceProp {
    use echo_core::prop as p;
    // SAFETY: cudaDeviceProp is a plain C struct of integers and arrays; all-zero is valid.
    let mut prop: crate::runtime::sys::cudaDeviceProp = unsafe { std::mem::zeroed() };
    for (dst, &src) in prop.name.iter_mut().zip(p::NAME.as_bytes()) {
        *dst = src as std::ffi::c_char;
    }
    prop.major = p::MAJOR;
    prop.minor = p::MINOR;
    prop.multiProcessorCount = p::MULTI_PROCESSOR_COUNT;
    prop.warpSize = p::WARP_SIZE;
    prop.maxThreadsPerBlock = p::MAX_THREADS_PER_BLOCK;
    prop.maxThreadsDim = p::MAX_THREADS_DIM;
    prop.maxGridSize = p::MAX_GRID_SIZE;
    prop.sharedMemPerBlock = p::SHARED_MEM_PER_BLOCK as usize;
    prop.sharedMemPerBlockOptin = p::SHARED_MEM_PER_BLOCK_OPTIN as usize;
    prop.regsPerBlock = p::REGS_PER_BLOCK;
    prop.totalConstMem = p::TOTAL_CONST_MEM as usize;
    prop.totalGlobalMem = p::TOTAL_GLOBAL_MEM as usize;
    prop
}
