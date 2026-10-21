// Task 6 fixture: float atomicAdd whose return value is used. The value depends on thread
// order, so ECHO must refuse to load this module.
extern "C" __global__ void atomic_used(float* p, const float* v, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float old = atomicAdd(p, v[i]);
  out[i] = old;
}
