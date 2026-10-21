// ECHO deterministic maths (Task 4): ex2, lg2, sin, cos for f32.
//
// Built only from correctly rounded IEEE-754 binary32 operations (__fmul_rn, __fadd_rn, __fsub_rn,
// __fmaf_rn, __fdiv_rn, rintf), integer/bit operations and comparisons, so every GPU computes the
// same bits. The CPU oracle echo-core/src/math.rs performs the same operations in the same order;
// any change here must be mirrored there (and the GPU test echo-gpu-tests/tests/gpu_math.rs checks it).
//
// Special values: NaN results are the canonical NaN 0x7fffffff.
//   ex2: NaN -> NaN, x >= 128 -> +inf, x < -150 -> +0.
//   lg2: NaN or x < 0 -> NaN, +-0 -> -inf, +inf -> +inf.
//   sin/cos: NaN, +-inf -> NaN; sin(+-0) = +-0. <= 2 ulp over the whole finite range: Cody-Waite
//   reduction for |x| <= 105615, exact Payne-Hanek reduction above (Ruling R9).
//
// Call ABI (used by the PTX normalizer): call (r), __echo_ex2f, (p); with .param .b32 for both.
//
// ECHO-EXPORT: __echo_ex2f
// ECHO-EXPORT: __echo_lg2f
// ECHO-EXPORT: __echo_sinf
// ECHO-EXPORT: __echo_cosf

#define ECHO_NAN __int_as_float(0x7fffffff)

extern "C" __device__ __noinline__ float __echo_ex2f(float x)
{
    if (x != x) return ECHO_NAN;
    if (x >= 128.0f) return __int_as_float(0x7f800000);
    if (x < -150.0f) return 0.0f;
    float n = rintf(x);
    float f = __fsub_rn(x, n);  // exact, |f| <= 0.5
    // c_k = (ln 2)^k / k! rounded to nearest f32
    float p = __int_as_float(0x377fe5fe);
    p = __fmaf_rn(p, f, __int_as_float(0x39218489));
    p = __fmaf_rn(p, f, __int_as_float(0x3aaec3ff));
    p = __fmaf_rn(p, f, __int_as_float(0x3c1d955b));
    p = __fmaf_rn(p, f, __int_as_float(0x3d635847));
    p = __fmaf_rn(p, f, __int_as_float(0x3e75fdf0));
    p = __fmaf_rn(p, f, __int_as_float(0x3f317218));
    p = __fmaf_rn(p, f, __int_as_float(0x3f800000));
    // n in [-150, 128]: both halves in [-75, 64], so only the final multiply rounds.
    int ni = (int)n;
    int h = ni / 2;
    float s1 = __int_as_float((h + 127) << 23);
    float s2 = __int_as_float((ni - h + 127) << 23);
    return __fmul_rn(__fmul_rn(p, s1), s2);
}

extern "C" __device__ __noinline__ float __echo_lg2f(float x)
{
    if (x != x || x < 0.0f) return ECHO_NAN;
    if (x == 0.0f) return __int_as_float(0xff800000);
    if (x == __int_as_float(0x7f800000)) return x;
    float xs = x;
    int e = 0;
    if (x < __int_as_float(0x00800000)) {  // subnormal: scale by 2^23 (exact)
        xs = __fmul_rn(x, 8388608.0f);
        e = -23;
    }
    unsigned int b = (unsigned int)__float_as_int(xs);
    e += (int)(b >> 23) - 127;
    unsigned int mb = (b & 0x007fffffu) | 0x3f800000u;  // m in [1, 2)
    if (mb > 0x3fb504f3u) {                              // m > largest f32 below sqrt(2)
        mb -= 0x00800000u;                               // m / 2 in [sqrt(1/2), 1)
        e += 1;
    }
    float m = __int_as_float((int)mb);
    // log2 m = K*atanh-series(s), s = (m-1)/(m+1) = s + s_lo; K = 2/ln 2 = K_HI + K_LO.
    float u = __fsub_rn(m, 1.0f);                         // exact
    float v = __fadd_rn(m, 1.0f);                         // rounded
    float v_lo = __fsub_rn(m, __fsub_rn(v, 1.0f));        // exact: m + 1 = v + v_lo
    float s = __fdiv_rn(u, v);
    float rem = __fmaf_rn(-s, v, u);                      // exact remainder u - s*v
    float s_lo = __fdiv_rn(__fmaf_rn(-s, v_lo, rem), v);
    float z = __fmul_rn(s, s);
    float p = __int_as_float(0x3ea4258a);                 // K/9
    p = __fmaf_rn(p, z, __int_as_float(0x3ed30bb1));      // K/7
    p = __fmaf_rn(p, z, __int_as_float(0x3f13bb63));      // K/5
    p = __fmaf_rn(p, z, __int_as_float(0x3f76384f));      // K/3
    float tail = __fmaf_rn(__fmul_rn(s, z), p,
                           __fmaf_rn(__int_as_float(0x33257060), s,                    // K_LO
                                     __fmul_rn(__int_as_float(0x4038aa3b), s_lo)));    // K_HI
    float l = __fmaf_rn(__int_as_float(0x4038aa3b), s, tail);
    return __fadd_rn((float)e, l);
}

// Cody-Waite reduction: r = x - k*pi/2 with pi/2 = C1 + C2 + C3; returns r, sets *quadrant = k & 3.
static __device__ __forceinline__ float __echo_reduce(float x, int *quadrant)
{
    float k = rintf(__fmul_rn(x, __int_as_float(0x3f22f983)));  // 2/pi
    float nk = -k;
    float r = __fmaf_rn(nk, __int_as_float(0x3fc90fdb), x);
    r = __fmaf_rn(nk, __int_as_float(0xb33bbd2e), r);
    r = __fmaf_rn(nk, __int_as_float(0xa6f72ced), r);
    // |k| >= 2^31 is a multiple of 2^8 (k mod 4 = 0); below that the conversion is exact.
    int q = (k < 2147483648.0f && k > -2147483648.0f) ? (int)k : 0;
    *quadrant = q & 3;
    return r;
}

// Bits of 2/pi after the binary point; word j holds bit positions 32j+1 ..= 32j+32. Selected with
// compares (no memory table), so the library has no global/const data to relocate when spliced.
static __device__ __forceinline__ unsigned int __echo_2opi_word(int j)
{
    return j == 0 ? 0xa2f9836eu : j == 1 ? 0x4e441529u : j == 2 ? 0xfc2757d1u : j == 3 ? 0xf534ddc0u
         : j == 4 ? 0xdb629599u : j == 5 ? 0x3c439041u : 0xfe5163abu;
}

// The 160-bit product is kept in five scalars and words are picked with compares, so nothing is
// indexed dynamically and the functions need no local memory.
struct __echo_u160 { unsigned int w0, w1, w2, w3, w4; };

static __device__ __forceinline__ unsigned int __echo_word160(const __echo_u160 &p, unsigned int i)
{
    return i == 0u ? p.w0 : i == 1u ? p.w1 : i == 2u ? p.w2 : i == 3u ? p.w3 : p.w4;
}

static __device__ __forceinline__ unsigned int __echo_bit160(const __echo_u160 &p, unsigned int k)
{
    return (__echo_word160(p, k >> 5) >> (k & 31u)) & 1u;
}

static __device__ __forceinline__ unsigned long long __echo_bits64(const __echo_u160 &p, unsigned int pos)
{
    unsigned int i = pos >> 5;
    unsigned int off = pos & 31u;
    unsigned long long lo = ((unsigned long long)__echo_word160(p, i + 1u) << 32) | (unsigned long long)__echo_word160(p, i);
    if (off == 0u) return lo;
    return (lo >> off) | ((unsigned long long)__echo_word160(p, i + 2u) << (64u - off));
}

// Exact Payne-Hanek reduction for finite |x| > 105615 (mirrors reduce_big in echo-core/src/math.rs).
static __device__ __forceinline__ float __echo_reduce_big(float x, int *quadrant)
{
    unsigned int xb = (unsigned int)__float_as_int(x);
    int ebits = (int)((xb >> 23) & 0xffu);
    unsigned int m = (xb & 0x007fffffu) | 0x00800000u;
    int sh = ebits - 150;
    int j0 = (sh >= 2) ? (sh - 2) / 32 : 0;
    int s = sh - 32 * j0;  // in [-7, 33]
    unsigned int w0 = __echo_2opi_word(j0);
    unsigned int w1 = __echo_2opi_word(j0 + 1);
    unsigned int w2 = __echo_2opi_word(j0 + 2);
    unsigned int w3 = __echo_2opi_word(j0 + 3);
    __echo_u160 p;
    unsigned long long t;
    t = (unsigned long long)m * w3;             p.w0 = (unsigned int)t;
    t = (unsigned long long)m * w2 + (t >> 32); p.w1 = (unsigned int)t;
    t = (unsigned long long)m * w1 + (t >> 32); p.w2 = (unsigned int)t;
    t = (unsigned long long)m * w0 + (t >> 32); p.w3 = (unsigned int)t;
    p.w4 = (unsigned int)(t >> 32);
    unsigned int q0 = (unsigned int)(128 - s);  // binary point, in [95, 135]
    int q = (int)(__echo_bit160(p, q0) | (__echo_bit160(p, q0 + 1u) << 1));
    unsigned long long f = __echo_bits64(p, q0 - 64u);  // fraction in units of 2^-64
    q += (int)(f >> 63);                                 // round to the nearest quadrant
    bool neg = (f >> 63) != 0ull;
    unsigned long long u = neg ? (0ull - f) : f;
    float r = 0.0f;
    if (u != 0ull) {
        int lz = __clzll((long long)u);
        unsigned long long n = u << lz;
        unsigned int h = (unsigned int)(n >> 40);
        unsigned int l = (unsigned int)((n >> 16) & 0x00ffffffull);
        float sh_h = __int_as_float((127 - 24 - lz) << 23);
        float sh_l = __int_as_float((127 - 48 - lz) << 23);
        float fh = __fmul_rn((float)h, sh_h);
        float fl = __fmul_rn((float)l, sh_l);
        if (neg) { fh = -fh; fl = -fl; }
        r = __fmaf_rn(fh, __int_as_float(0x3fc90fdb),
                      __fmaf_rn(fh, __int_as_float(0xb33bbd2e), __fmul_rn(fl, __int_as_float(0x3fc90fdb))));
    }
    if (xb >> 31) {  // x = -(q*pi/2 + r)
        *quadrant = (-q) & 3;
        return -r;
    }
    *quadrant = q & 3;
    return r;
}

// Cody-Waite for |x| <= 105615 (bits unchanged from the original fast path), Payne-Hanek above.
static __device__ __forceinline__ float __echo_reduce_any(float x, int *quadrant)
{
    if (__int_as_float(__float_as_int(x) & 0x7fffffff) > 105615.0f) return __echo_reduce_big(x, quadrant);
    return __echo_reduce(x, quadrant);
}

static __device__ __forceinline__ float __echo_sin_poly(float r)
{
    float z = __fmul_rn(r, r);
    float p = __int_as_float(0x3638ef1d);              // 1/362880
    p = __fmaf_rn(p, z, __int_as_float(0xb9500d01));   // -1/5040
    p = __fmaf_rn(p, z, __int_as_float(0x3c088889));   // 1/120
    p = __fmaf_rn(p, z, __int_as_float(0xbe2aaaab));   // -1/6
    float t = __fmul_rn(z, r);
    return __fmaf_rn(t, p, r);
}

static __device__ __forceinline__ float __echo_cos_poly(float r)
{
    float z = __fmul_rn(r, r);
    float p = __int_as_float(0x37d00d01);              // 1/40320
    p = __fmaf_rn(p, z, __int_as_float(0xbab60b61));   // -1/720
    p = __fmaf_rn(p, z, __int_as_float(0x3d2aaaab));   // 1/24
    p = __fmaf_rn(p, z, __int_as_float(0xbf000000));   // -1/2
    return __fmaf_rn(p, z, 1.0f);
}

static __device__ __forceinline__ float __echo_canon(float y)
{
    return (y != y) ? ECHO_NAN : y;
}

extern "C" __device__ __noinline__ float __echo_sinf(float x)
{
    if ((__float_as_int(x) & 0x7f800000) == 0x7f800000) return ECHO_NAN;
    if (x == 0.0f) return x;  // keeps the sign of zero
    int q;
    float r = __echo_reduce_any(x, &q);
    float y;
    if (q == 0) y = __echo_sin_poly(r);
    else if (q == 1) y = __echo_cos_poly(r);
    else if (q == 2) y = -__echo_sin_poly(r);
    else y = -__echo_cos_poly(r);
    return __echo_canon(y);
}

extern "C" __device__ __noinline__ float __echo_cosf(float x)
{
    if ((__float_as_int(x) & 0x7f800000) == 0x7f800000) return ECHO_NAN;
    int q;
    float r = __echo_reduce_any(x, &q);
    float y;
    if (q == 0) y = __echo_cos_poly(r);
    else if (q == 1) y = -__echo_sin_poly(r);
    else if (q == 2) y = -__echo_cos_poly(r);
    else y = __echo_sin_poly(r);
    return __echo_canon(y);
}
