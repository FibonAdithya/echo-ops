//! Task 6: ECHO runtime integration (GPU). Every test is `#[ignore]`; run on a GPU box with
//!   cargo test --features cuda-version-from-build-system --test gpu_runtime -- --ignored --test-threads=1
//! The fixtures are built with TIG's nvcc flags by echo-kernels/build_fixtures.sh.

use cudarc::driver::{
    result, sys, CudaContext, CudaStream, DevicePtr, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use echo_core::exact::ExactAcc;
use std::sync::Arc;

const FASTMATH: &str = include_str!("../../echo-kernels/fixtures/fastmath.ptx");
const ATOMICS: &str = include_str!("../../echo-kernels/fixtures/atomics.ptx");
const ATOMIC_USED: &str = include_str!("../../echo-kernels/fixtures/atomic_used.ptx");

fn gpu() -> (Arc<CudaContext>, Arc<CudaStream>) {
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    (ctx, stream)
}

#[test]
#[ignore]
fn prop_is_canonical() {
    let prop = cudarc::runtime::result::device::get_device_prop(0).expect("get_device_prop(0)");
    let name: Vec<u8> = prop.name.iter().map(|&c| c as u8).collect();
    assert_eq!(&name[..5], b"ECHO\0", "name");
    assert!(name.iter().skip(4).all(|&c| c == 0), "name is NUL-padded");
    assert_eq!((prop.major, prop.minor), (7, 0));
    assert_eq!(prop.multiProcessorCount, 128);
    assert_eq!(prop.warpSize, 32);
    assert_eq!(prop.maxThreadsPerBlock, 1024);
    assert_eq!(prop.maxThreadsDim, [1024, 1024, 64]);
    assert_eq!(prop.maxGridSize, [2147483647, 65535, 65535]);
    assert_eq!(prop.sharedMemPerBlock, 49152);
    assert_eq!(prop.sharedMemPerBlockOptin, 65536);
    assert_eq!(prop.regsPerBlock, 65536);
    assert_eq!(prop.totalConstMem, 65536);
    assert_eq!(prop.totalGlobalMem, 4294967296);
    // a sample of the fields that must be zero
    assert_eq!(prop.l2CacheSize, 0);
    assert_eq!(prop.pciBusID, 0);
    assert!(prop.uuid.bytes.iter().all(|&b| b == 0), "uuid");
    let bytes = unsafe {
        std::slice::from_raw_parts(
            &prop as *const _ as *const u8,
            std::mem::size_of_val(&prop),
        )
    };
    println!("prop sha256 = {} ({} bytes)", sha256_hex(bytes), bytes.len());
}

#[test]
#[ignore]
fn alloc_returns_zeroed_memory() {
    let (_ctx, stream) = gpu();
    let n = (64usize << 20) / 4;
    let a = unsafe { stream.alloc::<u32>(n) }.unwrap();
    let ptr_a = a.device_ptr(&stream).0;
    unsafe { result::memset_d8_async(ptr_a, 0xFF, n * 4, stream.cu_stream()) }.unwrap();
    let head = stream.memcpy_dtov(&a.slice(0..16)).unwrap();
    assert!(head.iter().all(|&w| w == u32::MAX), "fill with 0xFF failed");
    // No synchronize between free and alloc: with the default memory pool (release
    // threshold 0) a synchronize returns the freed pages to the driver, and the next
    // allocation at the same address gets fresh, driver-zeroed pages, which would make this
    // test pass without ECHO (observed on the L40 at commit 3293ec1). Stream-ordered reuse
    // keeps the 0xFF contents.
    drop(a);

    let b = unsafe { stream.alloc::<u32>(n) }.unwrap();
    let ptr_b = b.device_ptr(&stream).0;
    assert_eq!(ptr_b, ptr_a, "could not provoke memory reuse");
    let host = stream.memcpy_dtov(&b).unwrap();
    let nonzero = host.iter().filter(|&&w| w != 0).count();
    assert_eq!(nonzero, 0, "{nonzero} of {n} words are not zero after alloc");
}

/// Deterministic inputs in [lo, hi).
fn lcg_f32(seed: &mut u64, lo: f32, hi: f32) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = (*seed >> 40) as f32 / (1u64 << 24) as f32;
    lo + (hi - lo) * u
}

#[test]
#[ignore]
fn fastmath_fixture_has_no_hazards_after_load() {
    assert!(
        !echo_core::ptx::hazards(FASTMATH).is_empty(),
        "the fixture itself must contain hazards, or this test proves nothing"
    );
    let (ctx, stream) = gpu();
    let module = ctx.load_module(Ptx::from_src(FASTMATH)).unwrap();
    let loaded = module.echo_ptx();
    let hz = echo_core::ptx::hazards(loaded);
    assert!(hz.is_empty(), "hazards left after load: {hz:?}");
    for f in ["__echo_ex2f", "__echo_lg2f", "__echo_sinf", "__echo_cosf"] {
        assert!(loaded.contains(&format!(".visible .func  (.param .b32 func_retval0) {f}(")), "library {f} not linked");
    }

    let n = 1000usize;
    let mut seed = 7u64;
    let x: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, 0.05, 8.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, 0.5, 2.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, -1.0, 1.0)).collect();
    let dx = stream.memcpy_stod(&x).unwrap();
    let dy = stream.memcpy_stod(&y).unwrap();
    let dz = stream.memcpy_stod(&z).unwrap();
    let mut out = stream.alloc_zeros::<f32>(8 * n).unwrap();
    let f = module.load_function("fastmath").unwrap();
    let n_i32 = n as i32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&dx)
            .arg(&dy)
            .arg(&dz)
            .arg(&mut out)
            .arg(&n_i32)
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .unwrap();
    stream.synchronize().unwrap();
    let got = stream.memcpy_dtov(&out).unwrap();
    for i in 0..n {
        let (a, b, c) = (x[i] as f64, y[i] as f64, z[i] as f64);
        let want = [a.exp(), a.ln(), a.sin(), a.cos(), a.sqrt(), 1.0 / a.sqrt(), a / b, a * b + c];
        for k in 0..8 {
            let g = got[8 * i + k] as f64;
            let tol = 1e-5 * want[k].abs().max(1.0);
            assert!((g - want[k]).abs() <= tol, "i={i} k={k} x={a} got {g} want {}", want[k]);
        }
    }
    let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    println!("fastmath out sha256 = {}", sha256_hex(&bytes));
}

fn atomics_inputs(n: usize) -> Vec<f32> {
    // Mixed magnitudes and signs, so float addition order matters.
    let mut seed = 11u64;
    (0..n)
        .map(|i| {
            let m = lcg_f32(&mut seed, 1.0, 2.0);
            let e = (lcg_f32(&mut seed, -24.0, 24.0)) as i32;
            let s = if i % 3 == 0 { -1.0 } else { 1.0 };
            s * m * 2f32.powi(e)
        })
        .collect()
}

fn run_atomics(stream: &Arc<CudaStream>, module: &Arc<cudarc::driver::CudaModule>, init: &[f32], v: &[f32], block: u32) -> Vec<f32> {
    let f = module.load_function("atomics").unwrap();
    let mut out = stream.memcpy_stod(init).unwrap();
    let dv = stream.memcpy_stod(v).unwrap();
    let (n, k) = (v.len() as i32, init.len() as i32);
    let cfg = LaunchConfig {
        grid_dim: ((v.len() as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { stream.launch_builder(&f).arg(&mut out).arg(&dv).arg(&n).arg(&k).launch(cfg) }.unwrap();
    stream.synchronize().unwrap();
    stream.memcpy_dtov(&out).unwrap()
}

#[test]
#[ignore]
fn atomics_fixture_is_launch_shape_independent() {
    let (ctx, stream) = gpu();
    let module = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
    assert!(module.echo_ptx().contains("__echo_atomic_add_f32"), "atomic not rewritten");
    let k = 37usize;
    let n = 50_000usize;
    let v = atomics_inputs(n);
    let init: Vec<f32> = (0..k).map(|j| j as f32 * 0.25 - 3.0).collect();
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
    let mut first: Option<Vec<u32>> = None;
    for block in [32u32, 256, 1024] {
        let got: Vec<u32> = run_atomics(&stream, &module, &init, &v, block).iter().map(|x| x.to_bits()).collect();
        for j in 0..k {
            assert_eq!(got[j], want[j], "block {block} out[{j}]: got {:e} want ExactAcc {:e}", f32::from_bits(got[j]), f32::from_bits(want[j]));
        }
        if let Some(f) = &first {
            assert_eq!(&got, f, "block {block} differs from block 32");
        }
        first = Some(got);
    }
    // Rounding the f64 sum would differ from ExactAcc in some slot only by chance; the check
    // that matters is above. This line shows the sums are not trivially representable.
    println!("atomics out[0..4] = {:?}", &first.unwrap()[..4].iter().map(|b| f32::from_bits(*b)).collect::<Vec<_>>());
}

fn exact_sums(init: &[f32], v: &[f32]) -> Vec<u32> {
    let k = init.len();
    (0..k)
        .map(|j| {
            let mut acc = ExactAcc::new();
            acc.add(init[j]);
            for i in (j..v.len()).step_by(k) {
                acc.add(v[i]);
            }
            acc.to_f32().to_bits()
        })
        .collect()
}

#[test]
#[ignore]
fn atomics_launch_cooperative_gets_finalize() {
    let (ctx, stream) = gpu();
    let module = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
    let f = module.load_function("atomics").unwrap();
    let (k, n) = (37usize, 50_000usize);
    let v = atomics_inputs(n);
    let init: Vec<f32> = (0..k).map(|j| j as f32 * 0.25 - 3.0).collect();
    let mut out = stream.memcpy_stod(&init).unwrap();
    let dv = stream.memcpy_stod(&v).unwrap();
    let (n_i, k_i) = (n as i32, k as i32);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut out)
            .arg(&dv)
            .arg(&n_i)
            .arg(&k_i)
            .launch_cooperative(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<u32> = stream.memcpy_dtov(&out).unwrap().iter().map(|x| x.to_bits()).collect();
    assert_eq!(got, exact_sums(&init, &v), "cooperative launch result is not ExactAcc");
}

#[test]
#[ignore]
fn second_stream_flagged_launch_is_refused() {
    let (ctx, _default) = gpu();
    let s1 = ctx.new_stream().unwrap();
    let s2 = ctx.new_stream().unwrap();
    let module = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
    let f = module.load_function("atomics").unwrap();
    let (n, k) = (1000i32, 8i32);
    let v = vec![1.0f32; n as usize];
    let dv1 = s1.memcpy_stod(&v).unwrap();
    let dv2 = s2.memcpy_stod(&v).unwrap();
    let mut out1 = s1.alloc_zeros::<f32>(k as usize).unwrap();
    let mut out2 = s2.alloc_zeros::<f32>(k as usize).unwrap();
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe { s1.launch_builder(&f).arg(&mut out1).arg(&dv1).arg(&n).arg(&k).launch(cfg) }
        .expect("first stream claims");
    let r = unsafe { s2.launch_builder(&f).arg(&mut out2).arg(&dv2).arg(&n).arg(&k).launch(cfg) };
    assert!(
        matches!(r, Err(DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_SUPPORTED))),
        "flagged launch on a second stream: {:?}",
        r.map(|_| ())
    );
    // nothing ran on s2, and s1 is unaffected
    assert_eq!(s2.memcpy_dtov(&out2).unwrap(), vec![0.0f32; k as usize]);
    assert_eq!(s1.memcpy_dtov(&out1).unwrap(), vec![125.0f32; k as usize]);
    // the claiming stream keeps working
    unsafe { s1.launch_builder(&f).arg(&mut out1).arg(&dv1).arg(&n).arg(&k).launch(cfg) }.unwrap();
    assert_eq!(s1.memcpy_dtov(&out1).unwrap(), vec![250.0f32; k as usize]);
}

#[test]
#[ignore]
fn finalize_is_ordered_before_slice_events() {
    // A flagged kernel on s1 writes `out`; s2 copies `out` after waiting on its write event.
    // The copy must see finalized values, so the event has to be recorded after finalize.
    let (ctx, _default) = gpu();
    let s1 = ctx.new_stream().unwrap();
    let s2 = ctx.new_stream().unwrap();
    let module = ctx.load_module(Ptx::from_src(ATOMICS)).unwrap();
    let f = module.load_function("atomics").unwrap();
    let (k, n) = (1usize << 19, 1usize << 20);
    let init: Vec<f32> = (0..k).map(|j| j as f32).collect();
    let v = vec![1.0f32; n];
    let dv = s1.memcpy_stod(&v).unwrap();
    for round in 0..3 {
        let mut out = s1.memcpy_stod(&init).unwrap();
        let (n_i, k_i) = (n as i32, k as i32);
        unsafe {
            s1.launch_builder(&f)
                .arg(&mut out)
                .arg(&dv)
                .arg(&n_i)
                .arg(&k_i)
                .launch(LaunchConfig::for_num_elems(n as u32))
        }
        .unwrap();
        let got = s2.memcpy_dtov(&out).unwrap();
        let bad = (0..k).filter(|&j| got[j] != init[j] + 2.0).count();
        assert_eq!(bad, 0, "round {round}: {bad} of {k} values read on s2 were not finalized");
    }
}

/// Runs in a child process (spawned by `atomic_used_fixture_is_refused`) so its stderr can be read.
#[test]
#[ignore]
fn atomic_used_child() {
    if std::env::var_os("ECHO_T6_CHILD").is_none() {
        return;
    }
    let (ctx, _stream) = gpu();
    match ctx.load_module(Ptx::from_src(ATOMIC_USED)) {
        Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_PTX)) => {}
        Err(e) => panic!("expected CUDA_ERROR_INVALID_PTX, got {e:?}"),
        Ok(_) => panic!("load_module accepted a used float atomicAdd return value"),
    }
}

#[test]
#[ignore]
fn atomic_used_fixture_is_refused() {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args(["--ignored", "--exact", "atomic_used_child", "--nocapture", "--test-threads=1"])
        .env("ECHO_T6_CHILD", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "child failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("1 passed"), "child did not run\n{stdout}");
    assert!(stderr.contains("ECHO: atomic_used:") && stderr.contains("return value"), "stderr:\n{stderr}");
    println!("child stderr: {}", stderr.trim());
}

#[test]
#[ignore]
fn binary_image_is_refused() {
    let (ctx, _stream) = gpu();
    let dir = std::env::temp_dir().join(format!("echo_t6_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("fake.cubin");
    std::fs::write(&path, b"\x7fELF\x02\x01\x01\x00\x00\x00\xff\xfe binary").unwrap();
    let r = ctx.load_module(Ptx::from_file(&path));
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        matches!(r, Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_PTX))),
        "{:?}",
        r.map(|_| ())
    );
}

/// Minimal SHA-256 (FIPS 180-4), so the test needs no extra crate.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[test]
fn sha256_known_answers() {
    assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    let long = vec![b'a'; 1000];
    assert_eq!(sha256_hex(&long), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}
