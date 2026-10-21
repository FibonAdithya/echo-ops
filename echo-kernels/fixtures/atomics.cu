// Task 6 fixture: float atomicAdd whose return value is unused. ECHO rewrites it to the
// exact accumulator, and the host enqueues finalize + reset after every launch.
extern "C" __global__ void atomics(float* out, const float* v, int n, int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&out[i % K], v[i]);
}
