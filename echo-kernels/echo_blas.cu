// ECHO BLAS: fixed-order single-precision GEMM replacing cublasSgemm_v2 (Task 7).
// Bit-identical to echo_core::oracle::sgemm; every float op is an explicit .rn intrinsic.
//
// KERNEL INTERFACE (launched from cudarc-echo, Task 7b)
// -----------------------------------------------------
// __echo_sgemm(int transa, int transb, int m, int n, int k, float alpha,
//              const float* A, int lda, const float* B, int ldb,
//              float beta, float* C, int ldc)
//   C = alpha * op(A) * op(B) + beta * C, column-major like cuBLAS
//   (element (i,j) at i + j*ld). op(A) is m x k, op(B) is k x n, C is m x n.
//   transa/transb: 0 = N (no transpose), 1 = T (transpose), as cublasOperation_t.
//   alpha/beta: by value (the host dereferences cuBLAS's host pointers).
//   With beta == 0, C is never read (it may hold NaN). Rows m..ldc-1 of C are not written.
//   Requires m, n, k >= 0; lda >= (transa ? k : m); ldb >= (transb ? n : k); ldc >= m;
//   all >= 1. m == 0 or n == 0: launch nothing.
//   Launch: block (16, 16, 1), grid (ceil(m/16), ceil(n/16), 1), dynamic shared mem 0
//   (2 KB static). Thread (tx, ty) of block (bx, by) owns C[bx*16+tx, by*16+ty].
//
// Fixed operation order (binding, see the Task 7 brief): per C[i,j],
//   acc = 0; for p in 0..k ascending: acc = fma(opA[i,p], opB[p,j], acc)
//   C[i,j] = beta == 0 ? alpha*acc : fma(alpha, acc, beta*C[i,j])
// The shared-memory tiling only changes where operands are loaded from, never the order
// of the fma chain, and the inner loop stops exactly at k (an extra fma(0, 0, -0) would
// turn -0 into +0).

#define ECHO_GEMM_TILE 16

// ECHO-EXPORT: __echo_sgemm
extern "C" __global__ void __echo_sgemm(int transa, int transb, int m, int n, int k, float alpha,
                                        const float* A, int lda, const float* B, int ldb,
                                        float beta, float* C, int ldc)
{
    __shared__ float As[ECHO_GEMM_TILE][ECHO_GEMM_TILE + 1];  // As[p][i]
    __shared__ float Bs[ECHO_GEMM_TILE][ECHO_GEMM_TILE + 1];  // Bs[j][p]
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i0 = blockIdx.x * ECHO_GEMM_TILE, j0 = blockIdx.y * ECHO_GEMM_TILE;
    const int i = i0 + tx, j = j0 + ty;
    float acc = 0.0f;
    for (int p0 = 0; p0 < k; p0 += ECHO_GEMM_TILE) {
        // Loads are arranged so consecutive tx read consecutive addresses.
        if (transa == 0) {  // opA[r, p] = A[r + p*lda]
            int r = i0 + tx, p = p0 + ty;
            As[ty][tx] = (r < m && p < k) ? A[(long long)p * lda + r] : 0.0f;
        } else {            // opA[r, p] = A[p + r*lda]
            int r = i0 + ty, p = p0 + tx;
            As[tx][ty] = (r < m && p < k) ? A[(long long)r * lda + p] : 0.0f;
        }
        if (transb == 0) {  // opB[p, s] = B[p + s*ldb]
            int p = p0 + tx, s = j0 + ty;
            Bs[ty][tx] = (p < k && s < n) ? B[(long long)s * ldb + p] : 0.0f;
        } else {            // opB[p, s] = B[s + p*ldb]
            int p = p0 + ty, s = j0 + tx;
            Bs[tx][ty] = (p < k && s < n) ? B[(long long)p * ldb + s] : 0.0f;
        }
        __syncthreads();
        const int pend = (k - p0 < ECHO_GEMM_TILE) ? (k - p0) : ECHO_GEMM_TILE;
        for (int pp = 0; pp < pend; ++pp) {
            acc = __fmaf_rn(As[pp][tx], Bs[ty][pp], acc);
        }
        __syncthreads();
    }
    if (i < m && j < n) {
        float* c = C + (long long)j * ldc + i;
        if (beta == 0.0f) {
            *c = __fmul_rn(alpha, acc);
        } else {
            *c = __fmaf_rn(alpha, acc, __fmul_rn(beta, *c));
        }
    }
}
