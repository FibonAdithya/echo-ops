// ECHO batch norm: fixed-order kernels replacing cuDNN's SPATIAL batch norm on
// NCHW float tensors of shape N x C x 1 x 1 (Task 7). Bit-identical to
// echo_core::oracle::{bn_forward_training, bn_forward_inference, bn_backward};
// every float op is an explicit .rn intrinsic.
//
// Layout: element (n, c) of x/y/dy/dx is at n*C + c; per-channel arrays have length C.
// alpha/beta: by value (the host dereferences cuDNN's host pointers).
// eps, factor: f32, converted on the host from cuDNN's double arguments with Rust `as f32`
//   (round to nearest even), exactly as the oracle does.
// blend(v, old, a, b) = b == 0 ? a*v : fma(a, v, b*old); with b == 0 old is never read.
// All pointers must be non-null and N >= 1, C >= 1 (the host returns an error otherwise).
// In-place (y == x, dx == dy) is safe: each element is read and then written by the same
// thread, and no element is read after it has been written.
//
// KERNEL INTERFACE (launched from cudarc-echo, Task 7b)
// -----------------------------------------------------
// __echo_bn_fwd_train(int N, int C, float alpha, float beta,
//                     const float* x, float* y, const float* bnScale, const float* bnBias,
//                     float factor, float* runningMean, float* runningVar, float eps,
//                     float* savedMean, float* savedInvStd)
//   cudnnBatchNormalizationForwardTraining. factor = (f32)exponentialAverageFactor.
//   Launch: block (128, 1, 1), grid (ceil(C/128), 1, 1), shared mem 0.
//   One thread per channel c = blockIdx.x*128 + threadIdx.x.
//
// __echo_bn_fwd_infer(int N, int C, float alpha, float beta,
//                     const float* x, float* y, const float* bnScale, const float* bnBias,
//                     const float* estimatedMean, const float* estimatedVar, float eps)
//   cudnnBatchNormalizationForwardInference.
//   Launch: block (256, 1, 1), grid (ceil(N*C/256), 1, 1), shared mem 0.
//   One thread per element e = blockIdx.x*256 + threadIdx.x (N*C < 2^31).
//
// __echo_bn_bwd(int N, int C, float alphaData, float betaData, float alphaParam, float betaParam,
//               const float* x, const float* dy, float* dx, const float* bnScale,
//               float* dBnScale, float* dBnBias, const float* savedMean, const float* savedInvStd)
//   cudnnBatchNormalizationBackward with saved statistics (epsilon is not used).
//   Launch: block (128, 1, 1), grid (ceil(C/128), 1, 1), shared mem 0.
//   One thread per channel c = blockIdx.x*128 + threadIdx.x.
//
// Fixed operation order: see the Task 7 brief and echo_core::oracle (n ascending throughout).

__device__ __forceinline__ float echo_bn_blend(float v, const float* old, float a, float b)
{
    return (b == 0.0f) ? __fmul_rn(a, v) : __fmaf_rn(a, v, __fmul_rn(b, *old));
}

// ECHO-EXPORT: __echo_bn_fwd_train
extern "C" __global__ void __echo_bn_fwd_train(int N, int C, float alpha, float beta,
                                               const float* x, float* y,
                                               const float* bnScale, const float* bnBias,
                                               float factor, float* runningMean, float* runningVar,
                                               float eps, float* savedMean, float* savedInvStd)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float nf = (float)N;  // exact for N < 2^24
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) sum = __fadd_rn(sum, x[(long long)n * C + c]);
    const float mean = __fdiv_rn(sum, nf);
    float ss = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float d = __fsub_rn(x[(long long)n * C + c], mean);
        ss = __fmaf_rn(d, d, ss);
    }
    const float var = __fdiv_rn(ss, nf);
    const float invstd = __frcp_rn(__fsqrt_rn(__fadd_rn(var, eps)));
    const float g = bnScale[c], bc = bnBias[c];
    for (int n = 0; n < N; ++n) {
        const long long e = (long long)n * C + c;
        const float v = __fmaf_rn(g, __fmul_rn(__fsub_rn(x[e], mean), invstd), bc);
        y[e] = echo_bn_blend(v, &y[e], alpha, beta);
    }
    const float one_minus = __fsub_rn(1.0f, factor);
    runningMean[c] = __fmaf_rn(factor, mean, __fmul_rn(one_minus, runningMean[c]));
    const float unbiased = (N > 1) ? __fdiv_rn(ss, (float)(N - 1)) : var;
    runningVar[c] = __fmaf_rn(factor, unbiased, __fmul_rn(one_minus, runningVar[c]));
    savedMean[c] = mean;
    savedInvStd[c] = invstd;
}

// ECHO-EXPORT: __echo_bn_fwd_infer
extern "C" __global__ void __echo_bn_fwd_infer(int N, int C, float alpha, float beta,
                                               const float* x, float* y,
                                               const float* bnScale, const float* bnBias,
                                               const float* estimatedMean, const float* estimatedVar,
                                               float eps)
{
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)N * C) return;
    const int c = (int)(e % C);
    const float invstd = __frcp_rn(__fsqrt_rn(__fadd_rn(estimatedVar[c], eps)));
    const float v = __fmaf_rn(bnScale[c], __fmul_rn(__fsub_rn(x[e], estimatedMean[c]), invstd), bnBias[c]);
    y[e] = echo_bn_blend(v, &y[e], alpha, beta);
}

// ECHO-EXPORT: __echo_bn_bwd
extern "C" __global__ void __echo_bn_bwd(int N, int C, float alphaData, float betaData,
                                         float alphaParam, float betaParam,
                                         const float* x, const float* dy, float* dx,
                                         const float* bnScale, float* dBnScale, float* dBnBias,
                                         const float* savedMean, const float* savedInvStd)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float nf = (float)N;
    const float m = savedMean[c], s = savedInvStd[c];
    float db = 0.0f, dg = 0.0f;
    for (int n = 0; n < N; ++n) {
        const long long e = (long long)n * C + c;
        const float xhat = __fmul_rn(__fsub_rn(x[e], m), s);
        const float g = dy[e];
        db = __fadd_rn(db, g);
        dg = __fmaf_rn(g, xhat, dg);
    }
    const float kk = __fdiv_rn(__fmul_rn(bnScale[c], s), nf);
    for (int n = 0; n < N; ++n) {
        const long long e = (long long)n * C + c;
        const float xhat = __fmul_rn(__fsub_rn(x[e], m), s);
        const float t = __fsub_rn(__fsub_rn(__fmul_rn(nf, dy[e]), db), __fmul_rn(xhat, dg));
        dx[e] = echo_bn_blend(__fmul_rn(kk, t), &dx[e], alphaData, betaData);
    }
    dBnScale[c] = echo_bn_blend(dg, &dBnScale[c], alphaParam, betaParam);
    dBnBias[c] = echo_bn_blend(db, &dBnBias[c], alphaParam, betaParam);
}
