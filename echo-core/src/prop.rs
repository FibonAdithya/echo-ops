//! Canonical device properties (Task 6).
//!
//! Under ECHO, `get_device_prop` returns the same `cudaDeviceProp` on every GPU, so an
//! algorithm that sizes its launches from the properties launches the same shapes everywhere.
//! Every field of the struct is zero except the ones below (values from the plan's Global
//! Constraints). The CUDA struct itself is built in cudarc-echo (`src/echo/mod.rs`), because
//! echo-core has no CUDA dependency.

/// `name`, NUL-padded into the 256-byte field.
pub const NAME: &str = "ECHO";
pub const MAJOR: i32 = 7;
pub const MINOR: i32 = 0;
pub const MULTI_PROCESSOR_COUNT: i32 = 128;
pub const WARP_SIZE: i32 = 32;
pub const MAX_THREADS_PER_BLOCK: i32 = 1024;
pub const MAX_THREADS_DIM: [i32; 3] = [1024, 1024, 64];
pub const MAX_GRID_SIZE: [i32; 3] = [2147483647, 65535, 65535];
pub const SHARED_MEM_PER_BLOCK: u64 = 49152;
pub const SHARED_MEM_PER_BLOCK_OPTIN: u64 = 65536;
pub const REGS_PER_BLOCK: i32 = 65536;
pub const TOTAL_CONST_MEM: u64 = 65536;
pub const TOTAL_GLOBAL_MEM: u64 = 4294967296;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_match_global_constraints() {
        assert_eq!(NAME, "ECHO");
        assert_eq!((MAJOR, MINOR), (7, 0));
        assert_eq!(MULTI_PROCESSOR_COUNT, 128);
        assert_eq!(WARP_SIZE, 32);
        assert_eq!(MAX_THREADS_PER_BLOCK, 1024);
        assert_eq!(MAX_THREADS_DIM, [1024, 1024, 64]);
        assert_eq!(MAX_GRID_SIZE, [i32::MAX, 65535, 65535]);
        assert_eq!(SHARED_MEM_PER_BLOCK, 48 * 1024);
        assert_eq!(SHARED_MEM_PER_BLOCK_OPTIN, 64 * 1024);
        assert_eq!(REGS_PER_BLOCK, 64 * 1024);
        assert_eq!(TOTAL_CONST_MEM, 64 * 1024);
        assert_eq!(TOTAL_GLOBAL_MEM, 4 << 30);
        assert!(NAME.len() < 256);
    }
}
