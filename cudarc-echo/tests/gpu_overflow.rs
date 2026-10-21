//! Task 6: accumulator overflow. In its own file so Cargo runs it in a fresh process:
//! ECHO_ACC_CAPACITY is read when an accumulator table is allocated (first module load in a
//! context), and CudaContext::new reuses the GPU's primary context.
//!   cargo test --features cuda-version-from-build-system --test gpu_overflow -- --ignored

use cudarc::driver::{sys, CudaContext, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const ATOMICS: &str = include_str!("../../echo-kernels/fixtures/atomics.ptx");
const N: i32 = 1000;

/// Launches the atomics fixture with `k` targets (N adds of 1.0); does not synchronize.
fn launch(stream: &Arc<CudaStream>, module: &Arc<CudaModule>, k: i32) -> CudaSlice<f32> {
    let f = module.load_function("atomics").unwrap();
    let dv = stream.memcpy_stod(&vec![1.0f32; N as usize]).unwrap();
    let mut out = stream.alloc_zeros::<f32>(k as usize).unwrap();
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut out)
            .arg(&dv)
            .arg(&N)
            .arg(&k)
            .launch(LaunchConfig::for_num_elems(N as u32))
    }
    .unwrap();
    out
}

#[test]
#[ignore]
fn full_table_errors_at_synchronize() {
    std::env::set_var("ECHO_ACC_CAPACITY", "8");
    let illegal = Err(DriverError(sys::cudaError_enum::CUDA_ERROR_ILLEGAL_STATE));
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Phase A: synchronize detects the overflow (it reads the mapped host error flag that
    // the reset kernel sets).
    let module = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
    // Control: 8 distinct targets fit in 8 slots.
    let out = launch(&stream, &module, 8);
    assert_eq!(stream.synchronize(), Ok(()), "8 targets must fit a capacity-8 table");
    assert_eq!(stream.memcpy_dtov(&out).unwrap(), vec![125.0f32; 8], "1000 adds of 1.0 over 8 targets");
    // 100 distinct targets do not fit: the next synchronize must fail.
    let out = launch(&stream, &module, 100);
    assert_eq!(stream.synchronize(), illegal, "stream synchronize after overflow");
    assert_eq!(ctx.synchronize(), illegal, "context synchronize after overflow");
    assert_eq!(stream.memcpy_dtov(&out).map(|_| ()), illegal, "memcpy_dtov after overflow");
    drop((out, module)); // frees the table; the next module gets a fresh one

    // Phase B: a data-returning copy detects the overflow by itself (no synchronize before it).
    let module = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
    let out = launch(&stream, &module, 100);
    assert_eq!(stream.memcpy_dtov(&out).map(|_| ()), illegal, "memcpy_dtov is the first check after overflow");
    let mut host = vec![0f32; 100];
    assert_eq!(stream.memcpy_dtoh(&out, &mut host), illegal, "memcpy_dtoh after overflow");
}
