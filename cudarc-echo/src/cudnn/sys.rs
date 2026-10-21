//! ECHO replacements for the cuDNN `sys` functions the challenge reaches. Types and signatures
//! are copied from tig-foundation/cudarc b3fccf5 `src/cudnn/sys/mod.rs`; each function starts
//! with that fork's `RTSigFuel` lines, verbatim. The three batch-norm calls charge through
//! `get_tensor_nd_info`, copied verbatim too: it calls `cudnnGetTensorNdDescriptor` twice, so
//! each batch-norm call also charges 2 fuel units and XORs `0xC6DF9D5EFF31B834` twice (no net
//! signature change), exactly as in the fork.
//!
//! A handle is a boxed `DnnHandle` (stream + ECHO kernels) behind the opaque `cudnnHandle_t`;
//! a tensor descriptor is a boxed `TensorDesc`. The kernels are attached by `Cudnn::new`
//! (`attach`); a handle made with `cudnnCreate` alone has none, and the batch-norm calls on it
//! return CUDNN_STATUS_NOT_INITIALIZED.
//!
//! Supported: SPATIAL mode on NCHW FLOAT tensors with `h == w == 1` (x/y/dy/dx `N x C x 1 x 1`,
//! scale/bias/mean/var `1 x C x 1 x 1`); anything else is CUDNN_STATUS_NOT_SUPPORTED. The
//! running and saved statistics buffers are required (cuDNN allows NULL to skip them; ECHO
//! returns CUDNN_STATUS_NOT_SUPPORTED). Arithmetic: see echo_core::oracle.
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use super::result::CudnnError;
use crate::echo::kernels::{blocks, p, Kernels};
use crate::RTSigFuel;

pub type cudaStream_t = *mut CUstream_st;
pub type cudnnHandle_t = *mut cudnnContext;
pub type cudnnTensorDescriptor_t = *mut cudnnTensorStruct;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudnnBatchNormMode_t {
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL = 1,
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudnnDataType_t {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
    CUDNN_DATA_BFLOAT16 = 9,
    CUDNN_DATA_INT64 = 10,
    CUDNN_DATA_BOOLEAN = 11,
    CUDNN_DATA_FP8_E4M3 = 12,
    CUDNN_DATA_FP8_E5M2 = 13,
    CUDNN_DATA_FAST_FLOAT_FOR_FP8 = 14,
    CUDNN_DATA_FP8_E8M0 = 15,
    CUDNN_DATA_FP4_E2M1 = 16,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudnnStatus_t {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1001,
    CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH = 1002,
    CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH = 1003,
    CUDNN_STATUS_DEPRECATED = 1004,
    CUDNN_STATUS_LICENSE_ERROR = 1005,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 1006,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 1007,
    CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED = 1008,
    CUDNN_STATUS_BAD_PARAM = 2000,
    CUDNN_STATUS_BAD_PARAM_NULL_POINTER = 2002,
    CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER = 2003,
    CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED = 2004,
    CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND = 2005,
    CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT = 2006,
    CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH = 2007,
    CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH = 2008,
    CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES = 2009,
    CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE = 2010,
    CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH = 2011,
    CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE = 2012,
    CUDNN_STATUS_NOT_SUPPORTED = 3000,
    CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN = 3001,
    CUDNN_STATUS_NOT_SUPPORTED_SHAPE = 3002,
    CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE = 3003,
    CUDNN_STATUS_NOT_SUPPORTED_LAYOUT = 3004,
    CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER = 3005,
    CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART = 3006,
    CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH = 3007,
    CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING = 3008,
    CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE = 3009,
    CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT = 3010,
    CUDNN_STATUS_NOT_SUPPORTED_PADDING = 3011,
    CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM = 3012,
    CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API = 3013,
    CUDNN_STATUS_INTERNAL_ERROR = 4000,
    CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED = 4001,
    CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE = 4002,
    CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED = 4003,
    CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED = 4004,
    CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM = 4005,
    CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED = 4006,
    CUDNN_STATUS_EXECUTION_FAILED = 5000,
    CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER = 5001,
    CUDNN_STATUS_EXECUTION_FAILED_CUBLAS = 5002,
    CUDNN_STATUS_EXECUTION_FAILED_CUDART = 5003,
    CUDNN_STATUS_EXECUTION_FAILED_CURAND = 5004,
    CUDNN_STATUS_INVALID_VALUE = 2001,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudnnContext {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudnnTensorStruct {
    _unused: [u8; 0],
}

/// What a `cudnnHandle_t` points at.
pub(crate) struct DnnHandle {
    stream: cudaStream_t,
    kernels: Option<Kernels>,
}

/// What a `cudnnTensorDescriptor_t` points at. `dims` is `[n, c, h, w]`; `None` until set.
pub(crate) struct TensorDesc {
    set: Option<(cudnnTensorFormat_t, cudnnDataType_t, [i32; 4])>,
}

unsafe fn handle_state<'a>(handle: cudnnHandle_t) -> Option<&'a mut DnnHandle> {
    (handle as *mut DnnHandle).as_mut()
}

unsafe fn desc_state<'a>(desc: cudnnTensorDescriptor_t) -> Option<&'a mut TensorDesc> {
    (desc as *mut TensorDesc).as_mut()
}

/// Gives the handle its ECHO kernels (one library module per handle).
pub(crate) unsafe fn attach(handle: cudnnHandle_t, kernels: Kernels) -> cudnnStatus_t {
    match handle_state(handle) {
        Some(h) => {
            h.kernels = Some(kernels);
            cudnnStatus_t::CUDNN_STATUS_SUCCESS
        }
        None => cudnnStatus_t::CUDNN_STATUS_BAD_PARAM,
    }
}

pub unsafe fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0x25D07AF2CE19A961);
    if handle.is_null() {
        return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM;
    }
    let h = Box::new(DnnHandle { stream: std::ptr::null_mut(), kernels: None });
    *handle = Box::into_raw(h) as cudnnHandle_t;
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

pub unsafe fn cudnnSetStream(handle: cudnnHandle_t, streamId: cudaStream_t) -> cudnnStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0x493654E1F00FB353);
    match handle_state(handle) {
        Some(h) => {
            h.stream = streamId;
            cudnnStatus_t::CUDNN_STATUS_SUCCESS
        }
        None => cudnnStatus_t::CUDNN_STATUS_BAD_PARAM,
    }
}

pub unsafe fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0x823C44E529F6FB35);
    let mut synced = Ok(());
    if !handle.is_null() {
        let h = Box::from_raw(handle as *mut DnnHandle);
        // Like the real cudnnDestroy, wait for the handle's work before its kernel module goes.
        if let Some(k) = &h.kernels {
            synced = k.synchronize(h.stream as crate::driver::sys::CUstream);
        }
        drop(h);
    }
    match synced {
        Ok(()) => cudnnStatus_t::CUDNN_STATUS_SUCCESS,
        Err(_) => cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED,
    }
}

pub unsafe fn cudnnCreateTensorDescriptor(
    tensorDesc: *mut cudnnTensorDescriptor_t,
) -> cudnnStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0x4A32D50DF1F25D24);
    if tensorDesc.is_null() {
        return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM;
    }
    *tensorDesc = Box::into_raw(Box::new(TensorDesc { set: None })) as cudnnTensorDescriptor_t;
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

pub unsafe fn cudnnSetTensor4dDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    n: ::core::ffi::c_int,
    c: ::core::ffi::c_int,
    h: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
) -> cudnnStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0x6C257A487802E52F);
    let Some(d) = desc_state(tensorDesc) else { return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM };
    if n <= 0 || c <= 0 || h <= 0 || w <= 0 {
        return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM;
    }
    d.set = Some((format, dataType, [n, c, h, w]));
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

pub unsafe fn cudnnDestroyTensorDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
) -> cudnnStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0x8806919860ADEB7E);
    if !tensorDesc.is_null() {
        drop(Box::from_raw(tensorDesc as *mut TensorDesc));
    }
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

/// Kept because the fork's batch-norm wrappers call it (through `get_tensor_nd_info`) and
/// charge for it. A set 4-D descriptor reports 4 dimensions `[n, c, h, w]` with packed strides.
pub unsafe fn cudnnGetTensorNdDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    nbDimsRequested: ::core::ffi::c_int,
    dataType: *mut cudnnDataType_t,
    nbDims: *mut ::core::ffi::c_int,
    dimA: *mut ::core::ffi::c_int,
    strideA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    RTSigFuel::add_fuel_used(1);
    RTSigFuel::mix_runtime_signature(0xC6DF9D5EFF31B834);
    let Some(d) = desc_state(tensorDesc) else { return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM };
    let Some((format, dtype, dims)) = d.set else {
        return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED;
    };
    if nbDimsRequested < 0 || dataType.is_null() || nbDims.is_null() {
        return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM;
    }
    let req = (nbDimsRequested as usize).min(4);
    if req > 0 && (dimA.is_null() || strideA.is_null()) {
        return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM;
    }
    let [n, c, h, w] = dims;
    let strides = match format {
        cudnnTensorFormat_t::CUDNN_TENSOR_NHWC => [h * w * c, 1, w * c, c],
        _ => [c * h * w, h * w, w, 1],
    };
    *dataType = dtype;
    *nbDims = 4;
    for i in 0..req {
        *dimA.add(i) = dims[i];
        *strideA.add(i) = strides[i];
    }
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

pub unsafe fn get_tensor_nd_info(
    desc: cudnnTensorDescriptor_t,
) -> Result<(cudnnDataType_t, Vec<i32>, Vec<i32>), CudnnError> {
    // First, get the number of dimensions
    let mut nb_dims: i32 = 0;
    let mut data_type: cudnnDataType_t = cudnnDataType_t::CUDNN_DATA_FLOAT;

    // Call with null pointers to get the number of dimensions
    match cudnnGetTensorNdDescriptor(
        desc,
        0, // nbDimsRequested = 0 to query the actual number of dimensions
        &mut data_type,
        &mut nb_dims,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ) {
        cudnnStatus_t::CUDNN_STATUS_SUCCESS => {}
        e => return Err(CudnnError(e)),
    }

    // Now allocate vectors for dimensions and strides
    let mut dims: Vec<i32> = vec![0; nb_dims as usize];
    let mut strides: Vec<i32> = vec![0; nb_dims as usize];

    // Get the actual data
    match cudnnGetTensorNdDescriptor(
        desc,
        nb_dims,
        &mut data_type,
        &mut nb_dims,
        dims.as_mut_ptr(),
        strides.as_mut_ptr(),
    ) {
        cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok((data_type, dims, strides)),
        e => Err(CudnnError(e)),
    }
}

/// `(N, C)` of an `N x C x 1 x 1` NCHW FLOAT descriptor, or NOT_SUPPORTED / BAD_PARAM.
unsafe fn nc_of(desc: cudnnTensorDescriptor_t) -> Result<(i32, i32), cudnnStatus_t> {
    let Some(d) = desc_state(desc) else { return Err(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM) };
    let Some((format, dtype, [n, c, h, w])) = d.set else {
        return Err(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
    };
    if format != cudnnTensorFormat_t::CUDNN_TENSOR_NCHW
        || dtype != cudnnDataType_t::CUDNN_DATA_FLOAT
        || h != 1
        || w != 1
    {
        return Err(cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED);
    }
    Ok((n, c))
}

/// Checks mode and descriptors: `data` are the `N x C x 1 x 1` tensors (all the same shape),
/// `param` the `1 x C x 1 x 1` one. Returns `(N, C)`.
unsafe fn bn_shape(
    mode: cudnnBatchNormMode_t,
    data: &[cudnnTensorDescriptor_t],
    param: cudnnTensorDescriptor_t,
) -> Result<(i32, i32), cudnnStatus_t> {
    if mode != cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL {
        return Err(cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED);
    }
    let (n, c) = nc_of(data[0])?;
    for &d in &data[1..] {
        if nc_of(d)? != (n, c) {
            return Err(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM);
        }
    }
    if nc_of(param)? != (1, c) {
        return Err(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM);
    }
    if n as i64 * c as i64 > i32::MAX as i64 {
        return Err(cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED);
    }
    Ok((n, c))
}

/// The handle's kernels and stream, or NOT_INITIALIZED.
unsafe fn ready<'a>(handle: cudnnHandle_t) -> Result<(&'a Kernels, cudaStream_t), cudnnStatus_t> {
    let Some(h) = handle_state(handle) else { return Err(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM) };
    match h.kernels.as_ref() {
        Some(k) => Ok((k, h.stream)),
        None => Err(cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED),
    }
}

fn any_null(ptrs: &[*const ::core::ffi::c_void]) -> bool {
    ptrs.iter().any(|q| q.is_null())
}

fn status(r: Result<(), cudnnStatus_t>) -> cudnnStatus_t {
    match r {
        Ok(()) => cudnnStatus_t::CUDNN_STATUS_SUCCESS,
        Err(e) => e,
    }
}

fn launched(r: Result<(), crate::driver::DriverError>) -> Result<(), cudnnStatus_t> {
    r.map_err(|_| cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED)
}

pub unsafe fn cudnnBatchNormalizationForwardTraining(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    bnBias: *const ::core::ffi::c_void,
    exponentialAverageFactor: f64,
    resultRunningMean: *mut ::core::ffi::c_void,
    resultRunningVariance: *mut ::core::ffi::c_void,
    epsilon: f64,
    resultSaveMean: *mut ::core::ffi::c_void,
    resultSaveInvVariance: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    match get_tensor_nd_info(xDesc) {
        Ok((_, dims, _)) => {
            RTSigFuel::add_fuel_used(dims.into_iter().map(|x| x as u64).product::<u64>() * 3);
        }
        Err(e) => return e.0,
    };
    RTSigFuel::mix_runtime_signature(0x92A6234EDD0FA265);
    status((|| {
        let (n, c) = bn_shape(mode, &[xDesc, yDesc], bnScaleBiasMeanVarDesc)?;
        if any_null(&[alpha, beta, x, y as _, bnScale, bnBias]) {
            return Err(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM);
        }
        if any_null(&[
            resultRunningMean as _,
            resultRunningVariance as _,
            resultSaveMean as _,
            resultSaveInvVariance as _,
        ]) {
            return Err(cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED);
        }
        let (kern, stream) = ready(handle)?;
        let (alpha, beta) = (*(alpha as *const f32), *(beta as *const f32));
        let factor = exponentialAverageFactor as f32;
        let eps = epsilon as f32;
        let mut params = [
            p(&n), p(&c), p(&alpha), p(&beta), p(&x), p(&y), p(&bnScale), p(&bnBias),
            p(&factor), p(&resultRunningMean), p(&resultRunningVariance), p(&eps),
            p(&resultSaveMean), p(&resultSaveInvVariance),
        ];
        let grid = (blocks(c as usize, 128), 1, 1);
        launched(kern.launch(&kern.bn_fwd_train, grid, (128, 1, 1), stream as _, &mut params))
    })())
}

pub unsafe fn cudnnBatchNormalizationForwardInference(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    bnBias: *const ::core::ffi::c_void,
    estimatedMean: *const ::core::ffi::c_void,
    estimatedVariance: *const ::core::ffi::c_void,
    epsilon: f64,
) -> cudnnStatus_t {
    match get_tensor_nd_info(xDesc) {
        Ok((_, dims, _)) => {
            RTSigFuel::add_fuel_used(dims.into_iter().map(|x| x as u64).product::<u64>() * 3);
        }
        Err(e) => return e.0,
    };
    RTSigFuel::mix_runtime_signature(0x081AAEB75F8682B1);
    status((|| {
        let (n, c) = bn_shape(mode, &[xDesc, yDesc], bnScaleBiasMeanVarDesc)?;
        if any_null(&[alpha, beta, x, y as _, bnScale, bnBias, estimatedMean, estimatedVariance]) {
            return Err(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM);
        }
        let (kern, stream) = ready(handle)?;
        let (alpha, beta) = (*(alpha as *const f32), *(beta as *const f32));
        let eps = epsilon as f32;
        let mut params = [
            p(&n), p(&c), p(&alpha), p(&beta), p(&x), p(&y), p(&bnScale), p(&bnBias),
            p(&estimatedMean), p(&estimatedVariance), p(&eps),
        ];
        let grid = (blocks(n as usize * c as usize, 256), 1, 1);
        launched(kern.launch(&kern.bn_fwd_infer, grid, (256, 1, 1), stream as _, &mut params))
    })())
}

pub unsafe fn cudnnBatchNormalizationBackward(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alphaDataDiff: *const ::core::ffi::c_void,
    betaDataDiff: *const ::core::ffi::c_void,
    alphaParamDiff: *const ::core::ffi::c_void,
    betaParamDiff: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    dBnScaleResult: *mut ::core::ffi::c_void,
    dBnBiasResult: *mut ::core::ffi::c_void,
    epsilon: f64,
    savedMean: *const ::core::ffi::c_void,
    savedInvVariance: *const ::core::ffi::c_void,
) -> cudnnStatus_t {
    match get_tensor_nd_info(xDesc) {
        Ok((_, dims, _)) => {
            RTSigFuel::add_fuel_used(dims.into_iter().map(|x| x as u64).product::<u64>() * 3);
        }
        Err(e) => return e.0,
    };
    RTSigFuel::mix_runtime_signature(0x34C16E93BE2C8A33);
    let _ = epsilon; // the saved statistics already include it, as in cuDNN
    status((|| {
        let (n, c) = bn_shape(mode, &[xDesc, dyDesc, dxDesc], dBnScaleBiasDesc)?;
        if any_null(&[
            alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff,
            x, dy, dx as _, bnScale, dBnScaleResult as _, dBnBiasResult as _,
        ]) {
            return Err(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM);
        }
        if any_null(&[savedMean, savedInvVariance]) {
            return Err(cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED);
        }
        let (kern, stream) = ready(handle)?;
        let ad = *(alphaDataDiff as *const f32);
        let bd = *(betaDataDiff as *const f32);
        let ap = *(alphaParamDiff as *const f32);
        let bp = *(betaParamDiff as *const f32);
        let mut params = [
            p(&n), p(&c), p(&ad), p(&bd), p(&ap), p(&bp), p(&x), p(&dy), p(&dx), p(&bnScale),
            p(&dBnScaleResult), p(&dBnBiasResult), p(&savedMean), p(&savedInvVariance),
        ];
        let grid = (blocks(c as usize, 128), 1, 1);
        launched(kern.launch(&kern.bn_bwd, grid, (128, 1, 1), stream as _, &mut params))
    })())
}
