//! Per-context accumulator table and per-module globals binding.
//!
//! `CudaContext::new` retains the GPU's primary context, so several `CudaContext` objects
//! can share one CUDA context. The table is therefore keyed by the raw `CUcontext`. It is
//! allocated at the first module load in that context and owned by the loaded modules
//! (`EchoModule` holds an `Arc<Table>`; the global map holds only a `Weak`). Every module
//! also holds its context, so the context outlives the table, and the table's memory is
//! freed when the last module is dropped. A context created later with the same handle
//! (after the primary context was reset) therefore gets a fresh table, never stale pointers.
//!
//! Ruling R24: float-atomic kernels of one context run on one stream. The first flagged
//! launch claims its stream; a flagged launch on another stream is refused before it runs.
//!
//! All device memory here comes from `result::malloc_sync`, not `CudaStream::alloc*`, so it
//! is not counted by `RTSigFuel::add_device_memory_used` and cannot trip the memory limit.
//!
//! Ruling R35: the overflow flag reaches the host without a device-to-host copy. Each table
//! owns one u32 of pinned, device-mapped host memory (`cuMemHostAlloc` with
//! `CU_MEMHOSTALLOC_DEVICEMAP`), bound to every module's `__echo_acc_host_error`.
//! `__echo_atomic_reset`, enqueued after every finalize, sets it to 1 when the module's
//! `__echo_acc_error` is set; the host reads it with a volatile load after the stream or
//! context synchronization that the checking call already performs. It is never cleared.

use crate::driver::{
    result::{self, DriverError},
    sys,
};
use std::collections::{BTreeSet, HashMap};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Debug counter: finalize launches enqueued so far (process-wide). Read only by tests;
/// kept in release builds because a relaxed atomic increment per launch is negligible.
/// It has no effect on results.
pub(crate) static FINALIZE_LAUNCHES: AtomicU64 = AtomicU64::new(0);
/// Debug counter: zeroing memsets issued by `CudaStream::alloc` (process-wide). Read only
/// by tests; same cost and effect as `FINALIZE_LAUNCHES`.
pub(crate) static ALLOC_MEMSETS: AtomicU64 = AtomicU64::new(0);

/// Table slots per context.
///
/// Test knob: the environment variable `ECHO_ACC_CAPACITY` (a positive integer) overrides it,
/// so tests can provoke a table overflow with a small table. It does not change results: a
/// launch either fits in the table and is exact, or overflows and fails at the next check.
/// An unparsable or zero value is ignored with a message on stderr (never a panic).
pub(crate) const DEFAULT_CAPACITY: u32 = 1 << 20;
/// i64 cells per slot (ExactAcc), as in echo_atomic.cu (ECHO_ACC_CELLS).
const CELLS: usize = 10;
/// Threads per block of `__echo_atomic_finalize`.
const FINALIZE_BLOCK: u32 = 256;

pub(crate) struct Table {
    pub(crate) capacity: u32,
    cu_ctx: usize,
    /// The stream of the first flagged launch (R24); `None` until then.
    stream: Mutex<Option<usize>>,
    /// Set once an overflow was seen; every later check fails without a device read.
    failed: AtomicBool,
    keys: sys::CUdeviceptr,
    cells: sys::CUdeviceptr,
    flags: sys::CUdeviceptr,
    list: sys::CUdeviceptr,
    /// Pinned, device-mapped host u32: 1 once any module's `__echo_acc_error` was set.
    host_error: *mut u32,
    /// Device address of `host_error`, stored in every module's `__echo_acc_host_error`.
    host_error_dev: sys::CUdeviceptr,
}

unsafe impl Send for Table {}
unsafe impl Sync for Table {}

fn tables() -> &'static Mutex<HashMap<usize, Weak<Table>>> {
    static TABLES: OnceLock<Mutex<HashMap<usize, Weak<Table>>>> = OnceLock::new();
    TABLES.get_or_init(|| Mutex::new(HashMap::new()))
}

impl Drop for Table {
    fn drop(&mut self) {
        // The modules that owned this table still hold their context, so it is alive here.
        for p in [self.keys, self.cells, self.flags, self.list] {
            let _ = unsafe { result::free_sync(p) };
        }
        let _ = unsafe { result::free_host(self.host_error as *mut std::ffi::c_void) };
        let mut map = tables().lock().unwrap();
        if map.get(&self.cu_ctx).is_some_and(|w| w.strong_count() == 0) {
            map.remove(&self.cu_ctx);
        }
    }
}

pub(crate) fn capacity_from_env() -> u32 {
    capacity_from(std::env::var("ECHO_ACC_CAPACITY").ok().as_deref())
}

/// `DEFAULT_CAPACITY`, or the value of the test knob `ECHO_ACC_CAPACITY` if it is a positive
/// integer. Anything else is ignored with a message (no panic in production).
pub(crate) fn capacity_from(var: Option<&str>) -> u32 {
    match var {
        None => DEFAULT_CAPACITY,
        Some(s) => match s.trim().parse::<u32>() {
            Ok(c) if c > 0 => c,
            _ => {
                eprintln!("ECHO: ignoring invalid ECHO_ACC_CAPACITY={s:?} (using {DEFAULT_CAPACITY})");
                DEFAULT_CAPACITY
            }
        },
    }
}

/// The table of `cu_ctx`, if one was allocated.
pub(crate) fn existing(cu_ctx: sys::CUcontext) -> Option<Arc<Table>> {
    tables().lock().unwrap().get(&(cu_ctx as usize)).and_then(Weak::upgrade)
}

/// The table of `cu_ctx`, allocating it on first use. `cu_ctx` must be current.
pub(crate) fn table_for(cu_ctx: sys::CUcontext) -> Result<Arc<Table>, DriverError> {
    let mut map = tables().lock().unwrap();
    if let Some(t) = map.get(&(cu_ctx as usize)).and_then(Weak::upgrade) {
        return Ok(t);
    }
    let capacity = capacity_from_env();
    let cap = capacity as usize;
    let sizes = [cap * 8, cap * 8 * CELLS, cap * 4, cap * 4];
    let mut ptrs: Vec<sys::CUdeviceptr> = Vec::with_capacity(4);
    for bytes in sizes {
        let r = unsafe { result::malloc_sync(bytes) }.and_then(|p| {
            unsafe { result::memset_d8_sync(p, 0, bytes) }.map(|()| p)
        });
        match r {
            Ok(p) => ptrs.push(p),
            Err(e) => {
                for p in ptrs {
                    let _ = unsafe { result::free_sync(p) };
                }
                return Err(e);
            }
        }
    }
    let host = match unsafe { mapped_host_flag() } {
        Ok(h) => h,
        Err(e) => {
            for p in ptrs {
                let _ = unsafe { result::free_sync(p) };
            }
            return Err(e);
        }
    };
    let t = Arc::new(Table {
        capacity,
        cu_ctx: cu_ctx as usize,
        stream: Mutex::new(None),
        failed: AtomicBool::new(false),
        keys: ptrs[0],
        cells: ptrs[1],
        flags: ptrs[2],
        list: ptrs[3],
        host_error: host.0,
        host_error_dev: host.1,
    });
    map.insert(cu_ctx as usize, Arc::downgrade(&t));
    Ok(t)
}

/// Allocates the zeroed, pinned, device-mapped host flag: (host pointer, device pointer).
/// The current context must be the table's.
unsafe fn mapped_host_flag() -> Result<(*mut u32, sys::CUdeviceptr), DriverError> {
    let h = result::malloc_host(std::mem::size_of::<u32>(), sys::CU_MEMHOSTALLOC_DEVICEMAP)? as *mut u32;
    std::ptr::write_volatile(h, 0);
    let mut d: sys::CUdeviceptr = 0;
    if let Err(e) = sys::cuMemHostGetDevicePointer_v2(&mut d, h as *mut std::ffi::c_void, 0).result() {
        let _ = result::free_host(h as *mut std::ffi::c_void);
        return Err(e);
    }
    Ok((h, d))
}

/// What ECHO keeps for one loaded module.
#[derive(Debug)]
pub(crate) struct EchoModule {
    /// The normalized, linked PTX that was loaded.
    pub(crate) ptx: String,
    pub(crate) finalize_kernels: BTreeSet<String>,
    finalize: sys::CUfunction,
    reset: sys::CUfunction,
    capacity: u32,
    /// Keeps the table (and its memory, which this module's globals point at) alive.
    table: Arc<Table>,
}

unsafe impl Send for EchoModule {}
unsafe impl Sync for EchoModule {}

impl std::fmt::Debug for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Table(capacity {})", self.capacity)
    }
}

fn global(module: sys::CUmodule, name: &str) -> Result<(sys::CUdeviceptr, usize), DriverError> {
    let n = CString::new(name).unwrap();
    let mut ptr: sys::CUdeviceptr = 0;
    let mut bytes: usize = 0;
    unsafe { sys::cuModuleGetGlobal_v2(&mut ptr, &mut bytes, module, n.as_ptr()) }.result()?;
    Ok((ptr, bytes))
}

fn set<T: Copy>(module: sys::CUmodule, name: &str, value: T) -> Result<sys::CUdeviceptr, DriverError> {
    let (ptr, bytes) = global(module, name)?;
    if bytes != std::mem::size_of::<T>() {
        eprintln!("ECHO: global {name} has {bytes} bytes, expected {}", std::mem::size_of::<T>());
        return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_PTX));
    }
    unsafe { result::memcpy_htod_sync(ptr, std::slice::from_ref(&value)) }?;
    Ok(ptr)
}

/// Binds the `__echo_acc_*` globals of a freshly loaded module to the context's table and
/// to its mapped host error flag. `cu_ctx` must be current.
pub(crate) fn bind(
    cu_ctx: sys::CUcontext,
    cu_module: sys::CUmodule,
    n: echo_core::ptx::Normalized,
) -> Result<EchoModule, DriverError> {
    let table = table_for(cu_ctx)?;
    set(cu_module, "__echo_acc_keys", table.keys)?;
    set(cu_module, "__echo_acc_cells", table.cells)?;
    set(cu_module, "__echo_acc_flags", table.flags)?;
    set(cu_module, "__echo_acc_list", table.list)?;
    set(cu_module, "__echo_acc_used", 0u32)?;
    set(cu_module, "__echo_acc_capacity", table.capacity)?;
    set(cu_module, "__echo_acc_error", 0u32)?;
    // Always set here (set() fails on a missing global); __echo_atomic_reset skips a null
    // __echo_acc_host_error only for modules bound by hand (echo-gpu-tests).
    set(cu_module, "__echo_acc_host_error", table.host_error_dev)?;
    let finalize = unsafe {
        result::module::get_function(cu_module, CString::new("__echo_atomic_finalize").unwrap())
    }?;
    let reset =
        unsafe { result::module::get_function(cu_module, CString::new("__echo_atomic_reset").unwrap()) }?;
    Ok(EchoModule {
        ptx: n.ptx,
        finalize_kernels: n.finalize_kernels,
        finalize,
        reset,
        capacity: table.capacity,
        table,
    })
}

impl EchoModule {
    /// R24: claims `stream` for this context's float-atomic kernels, or refuses the launch
    /// if another stream already holds the claim. Called before the kernel is launched.
    pub(crate) fn claim_stream(&self, stream: sys::CUstream) -> Result<(), DriverError> {
        let mut s = self.table.stream.lock().unwrap();
        match *s {
            None => {
                *s = Some(stream as usize);
                Ok(())
            }
            Some(first) if first == stream as usize => Ok(()),
            Some(first) => {
                eprintln!(
                    "ECHO: a kernel with float atomics was launched on stream {:#x}, but this context's \
                     float-atomic kernels run on stream {first:#x}; ECHO allows one such stream per context (ruling R24)",
                    stream as usize
                );
                Err(DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_SUPPORTED))
            }
        }
    }

    /// Enqueues `__echo_atomic_finalize` (grid ceil(capacity/256), block 256) and then
    /// `__echo_atomic_reset` on `stream`, after a launch of a kernel in `finalize_kernels`.
    pub(crate) fn enqueue_finalize(&self, stream: sys::CUstream) -> Result<(), DriverError> {
        let grid = self.capacity.div_ceil(FINALIZE_BLOCK);
        FINALIZE_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        unsafe {
            result::launch_kernel(self.finalize, (grid, 1, 1), (FINALIZE_BLOCK, 1, 1), 0, stream, &mut [])?;
            result::launch_kernel(self.reset, (1, 1, 1), (1, 1, 1), 0, stream, &mut [])
        }
    }
}

fn overflow(table: &Table) -> DriverError {
    table.failed.store(true, Ordering::Relaxed);
    eprintln!("ECHO: atomic accumulator full (capacity {})", table.capacity);
    DriverError(sys::cudaError_enum::CUDA_ERROR_ILLEGAL_STATE)
}

/// Reads the context's mapped host error flag (volatile load, no device access). The caller
/// has just synchronized the stream or context whose work it returns, so every reset kernel
/// enqueued before that point has completed and its store is visible. Fails with
/// CUDA_ERROR_ILLEGAL_STATE if the flag is set, now or earlier.
pub(crate) fn check_errors(cu_ctx: sys::CUcontext) -> Result<(), DriverError> {
    let Some(table) = existing(cu_ctx) else {
        return Ok(());
    };
    // No fence needed: the driver synchronization the caller just completed (stream/context
    // sync, or a pageable copy that returned) orders the reset kernels' stores before this load.
    if table.failed.load(Ordering::Relaxed) || unsafe { std::ptr::read_volatile(table.host_error) } != 0 {
        return Err(overflow(&table));
    }
    Ok(())
}

/// Fails if an overflow was already seen in `cu_ctx` (no device access, no synchronization).
pub(crate) fn check_failed(cu_ctx: sys::CUcontext) -> Result<(), DriverError> {
    match existing(cu_ctx) {
        Some(t) if t.failed.load(Ordering::Relaxed) => Err(overflow(&t)),
        _ => Ok(()),
    }
}
