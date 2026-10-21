# ECHO

A drop-in replacement for the Rust [`cudarc`](https://crates.io/crates/cudarc) crate (0.16) that
makes GPU results bit-identical across NVIDIA GPUs. Your code does not change: you point Cargo
at ECHO's copy of `cudarc` and rebuild.

What ECHO does under the same API:

- cuBLAS and cuDNN calls run fixed-order ECHO kernels instead of NVIDIA's libraries, so the
  result does not depend on which kernel the library picks for a GPU.
- Every kernel loaded with `load_module` is rewritten to use correctly rounded maths (no
  fast-math approximations, no fused multiply-add the driver chooses on its own).
- Float `atomicAdd` sums are accumulated exactly and rounded once, so thread order does not
  matter.
- The reported SM count is fixed (128), so grid sizes computed from it are the same everywhere.

## Swap it in

Add one `[patch]` section to the `Cargo.toml` of the project (or workspace root) that builds
your binary. Pick the one that matches where your `cudarc` comes from.

From crates.io (`cudarc = "0.16"`):

```toml
[patch.crates-io]
cudarc = { git = "https://github.com/FibonAdithya/echo-ops" }
```

From TIG's fork (`tig-foundation/cudarc`, as in tig-monorepo):

```toml
[patch."https://github.com/tig-foundation/cudarc.git"]
cudarc = { git = "https://github.com/FibonAdithya/echo-ops" }
```

Or without editing any file, for a single build:

```sh
cargo --config 'patch.crates-io.cudarc.git="https://github.com/FibonAdithya/echo-ops"' build --release
```

Then rebuild and confirm the swap took effect:

```sh
cargo update -p cudarc
cargo tree -i cudarc     # should show cudarc v0.16.4 (https://github.com/FibonAdithya/echo-ops...)
```

To use a local clone instead, replace `git = "..."` with
`path = "/path/to/echo-ops/cudarc-echo"`.

## Requirements and limits

- Rust projects using `cudarc` **0.16.x** only. Other CUDA code (C++, Python, other crates that
  link CUDA themselves) is not affected.
- Only cudarc's safe API is covered: `CudaContext`, `CudaStream`, `load_module`, launches,
  copies, `cublas`, `cudnn`. Raw calls through `cudarc::driver::sys` or `driver::result` bypass ECHO.
- The `curand`, `cublaslt`, `nccl`, `cusolver` and `cusparse` modules are removed. Their
  features still exist so dependents compile, but code that uses those modules will not build.
- Kernels must be loaded as PTX text. Cubin and fatbin images are refused, and so is PTX that
  ECHO cannot make deterministic; the reason is printed to stderr starting with `ECHO:`.
- Changed behaviour: a float `atomicAdd` result becomes visible only after the kernel finishes.
  A kernel that reads the target of its own `atomicAdd` in the same launch will see the old value.
- An NVIDIA driver new enough for your kernels' PTX version (per NVIDIA's tables, CUDA 12.8 PTX needs driver
  570.26 or newer).
- Tested on RTX 3080 Ti, RTX 3090 (sm_86) and L40 (sm_89). Other generations are untested.

## Repository layout

| path | what it is |
|---|---|
| `cudarc-echo/` | The replacement crate (package name `cudarc`, version 0.16.4), forked from [tig-foundation/cudarc](https://github.com/tig-foundation/cudarc) at `b3fccf5`. |
| `echo-core/` | PTX rewriter and exact-arithmetic code, used by `cudarc-echo`. No CUDA needed. |
| `echo-kernels/` | Source of ECHO's GPU kernels. `build.sh` regenerates `ptx/echo_lib.ptx` (needs `nvcc`). |

## Tests

```sh
cd echo-core && cargo test --release                                  # CPU only
cd cudarc-echo && cargo test --features cuda-12080,cudnn -- --ignored --test-threads=1   # needs a GPU
```

## License

MIT, see [`LICENSE`](LICENSE). `cudarc-echo/` is derived from cudarc and keeps its original
MIT OR Apache-2.0 license.
