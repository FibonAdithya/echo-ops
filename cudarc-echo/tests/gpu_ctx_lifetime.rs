//! Task 6 fix round 1: the accumulator table must not outlive its CUDA context. In its own
//! file so it runs in a fresh process where nothing else retains the primary context
//! (dropping the last CudaContext then resets it and frees all of its memory).
//!   cargo test --features cuda-version-from-build-system --test gpu_ctx_lifetime -- --ignored

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use echo_core::exact::ExactAcc;

const ATOMICS: &str = include_str!("../../echo-kernels/fixtures/atomics.ptx");

#[test]
#[ignore]
fn table_does_not_outlive_context() {
    let (k, n) = (37usize, 20_000usize);
    let v: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32 * 2f32.powi(-20)).collect();
    let init: Vec<f32> = (0..k).map(|j| j as f32 - 5.0).collect();
    let want: Vec<u32> = (0..k)
        .map(|j| {
            let mut acc = ExactAcc::new();
            acc.add(init[j]);
            for i in (j..n).step_by(k) {
                acc.add(v[i]);
            }
            acc.to_f32().to_bits()
        })
        .collect();
    for round in 0..3 {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
        let f = module.load_function("atomics").unwrap();
        let mut out = stream.memcpy_stod(&init).unwrap();
        let dv = stream.memcpy_stod(&v).unwrap();
        let (n_i, k_i) = (n as i32, k as i32);
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&mut out)
                .arg(&dv)
                .arg(&n_i)
                .arg(&k_i)
                .launch(LaunchConfig::for_num_elems(n as u32))
        }
        .unwrap_or_else(|e| panic!("round {round}: launch: {e:?}"));
        stream.synchronize().unwrap_or_else(|e| panic!("round {round}: synchronize: {e:?}"));
        let got: Vec<u32> = stream.memcpy_dtov(&out).unwrap().iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, want, "round {round}: not ExactAcc");
        drop((f, module, out, dv, stream));
        drop(ctx); // last reference: the primary context is released (and reset)
    }
}
