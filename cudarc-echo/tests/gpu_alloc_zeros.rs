//! Final fix R42: `alloc_zeros` on one stream must be ordered before a reader on another
//! stream (the slice's write event is recorded after the zeroing memset).
//!   cargo test --features cuda-version-from-build-system --test gpu_alloc_zeros -- --ignored

use cudarc::driver::{result, CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// `out[i] = src[i]` for `i < n` (plain loads and stores, no atomics, so it may run on
/// any stream). Hand-written PTX in nvcc's style.
const COPY_U32: &str = "\
.version 7.0
.target sm_70
.address_size 64

.visible .entry copy_u32(
\t.param .u64 copy_u32_param_0,
\t.param .u64 copy_u32_param_1,
\t.param .u32 copy_u32_param_2
)
{
\t.reg .pred \t%p<2>;
\t.reg .b32 \t%r<7>;
\t.reg .b64 \t%rd<9>;

\tld.param.u64 \t%rd1, [copy_u32_param_0];
\tld.param.u64 \t%rd2, [copy_u32_param_1];
\tld.param.u32 \t%r2, [copy_u32_param_2];
\tmov.u32 \t%r3, %ctaid.x;
\tmov.u32 \t%r4, %ntid.x;
\tmov.u32 \t%r5, %tid.x;
\tmad.lo.s32 \t%r1, %r3, %r4, %r5;
\tsetp.ge.u32 \t%p1, %r1, %r2;
\t@%p1 bra \t$L__BB0_2;
\tcvta.to.global.u64 \t%rd3, %rd2;
\tcvta.to.global.u64 \t%rd4, %rd1;
\tmul.wide.u32 \t%rd5, %r1, 4;
\tadd.s64 \t%rd6, %rd3, %rd5;
\tld.global.u32 \t%r6, [%rd6];
\tadd.s64 \t%rd7, %rd4, %rd5;
\tst.global.u32 \t[%rd7], %r6;
$L__BB0_2:
\tret;
}
";

/// Host function enqueued on s1: holds s1 so the zeroing memset behind it runs late.
unsafe extern "C" fn hold_stream(_: *mut std::ffi::c_void) {
    std::thread::sleep(std::time::Duration::from_millis(500));
}

#[test]
#[ignore]
fn alloc_zeros_is_ordered_before_reader_on_other_stream() {
    let ctx = CudaContext::new(0).unwrap();
    let s1 = ctx.new_stream().unwrap();
    let s2 = ctx.new_stream().unwrap();
    let module = ctx.load_module(Ptx::from_src(COPY_U32)).unwrap();
    let f = module.load_function("copy_u32").unwrap();
    let n = (64usize << 20) / 4;
    let mut out = s2.alloc_zeros::<u32>(n).unwrap();
    s2.synchronize().unwrap();

    for round in 0..3 {
        // Leave stale non-zero bytes in a block of s1's pool, then free it on s1
        // (stream-ordered reuse keeps the contents, see alloc_returns_zeroed_memory).
        let a = unsafe { s1.alloc::<u32>(n) }.unwrap();
        let ptr_a = a.device_ptr(&s1).0;
        unsafe { result::memset_d8_async(ptr_a, 0x3F, n * 4, s1.cu_stream()) }.unwrap();
        drop(a);
        unsafe {
            result::stream::launch_host_function(
                s1.cu_stream(),
                hold_stream,
                std::ptr::null_mut(),
            )
        }
        .unwrap();
        // The zeroing memset of `b` is queued on s1 behind the 500 ms hold.
        let b = s1.alloc_zeros::<u32>(n).unwrap();
        assert_eq!(b.device_ptr(&s1).0, ptr_a, "round {round}: could not provoke memory reuse");
        // Immediately read `b` on s2. The launch makes s2 wait on b's write event; without
        // that event being recorded after the memset, the copy sees the stale 0x3F bytes.
        let n_u32 = n as u32;
        unsafe {
            s2.launch_builder(&f)
                .arg(&mut out)
                .arg(&b)
                .arg(&n_u32)
                .launch(LaunchConfig::for_num_elems(n_u32))
        }
        .unwrap();
        let host = s2.memcpy_dtov(&out).unwrap();
        let nonzero = host.iter().filter(|&&w| w != 0).count();
        assert_eq!(nonzero, 0, "round {round}: {nonzero} of {n} words read on s2 are not zero");
        drop(b);
        s1.synchronize().unwrap();
    }
}
