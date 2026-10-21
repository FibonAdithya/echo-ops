// Task 6 fixture: fast-math intrinsics that ECHO must rewrite at load_module.
// Built with TIG's flags (build_fixtures.sh); the committed PTX is fastmath.ptx.
// out[8*i + k]: k = 0 expf, 1 logf, 2 sinf, 3 cosf, 4 sqrtf, 5 rsqrtf, 6 x/y, 7 x*y+z.
extern "C" __global__ void fastmath(const float* x, const float* y, const float* z, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = x[i], b = y[i], c = z[i];
  out[8 * i + 0] = expf(a);
  out[8 * i + 1] = logf(a);
  out[8 * i + 2] = sinf(a);
  out[8 * i + 3] = cosf(a);
  out[8 * i + 4] = sqrtf(a);
  out[8 * i + 5] = rsqrtf(a);
  out[8 * i + 6] = a / b;
  out[8 * i + 7] = a * b + c;
}
