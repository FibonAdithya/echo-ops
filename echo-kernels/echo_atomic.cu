// ECHO exact float accumulator: the device replacement for float atomicAdd (plan Task 5).
// The CPU oracle is echo-core/src/exact.rs (ExactAcc); echo_acc_add and echo_acc_to_bits below
// mirror ExactAcc::add and ExactAcc::to_f32 operation for operation and must agree bit for bit.
// Only integer atomics and integer arithmetic are used: no float instruction appears here.
//
// Protocol per launch of a rewritten kernel (the host side is Task 6):
//   1. the kernel calls __echo_atomic_add_f32(addr, v) instead of atom.add.f32;
//   2. the host launches __echo_atomic_finalize with >= __echo_acc_capacity threads
//      (it stores ExactAcc(old *addr, all v for that addr) to every touched addr and clears
//      the entry), then __echo_atomic_reset (one thread), which copies a set __echo_acc_error
//      to the context's pinned, device-mapped host flag *__echo_acc_host_error; the host
//      reads that flag after its stream/context synchronization (no device-to-host copy).
// Every address that received at least one add (including adds of 0, subnormals, NaN, inf) is
// rewritten, so its final value is always ExactAcc over the old value and all contributions.
//
// Table: open addressing on the generic address, linear probing, key 0 = empty. A slot is
// claimed with atom.cas.b64 and its index appended to __echo_acc_list. At most
// ECHO_ACC_MAX_PROBE slots are probed; if none is free or matching, __echo_acc_error is set to
// 1 and the contribution is dropped (the host must treat that launch as failed).
// Limit: each cell absorbs at most 2^31 contributions per launch (see exact.rs).
//
// ECHO-EXPORT: __echo_atomic_add_f32
// ECHO-EXPORT: __echo_atomic_finalize
// ECHO-EXPORT: __echo_atomic_reset

#define ECHO_ACC_CELLS 10
#define ECHO_ACC_MAX_PROBE 1024u
#define ECHO_ACC_FLAG_NAN 1u
#define ECHO_ACC_FLAG_POS_INF 2u
#define ECHO_ACC_FLAG_NEG_INF 4u

extern "C" {
__device__ unsigned long long* __echo_acc_keys;  // [capacity] generic float* or 0
__device__ long long* __echo_acc_cells;          // [capacity * 10]
__device__ unsigned int* __echo_acc_flags;       // [capacity] ECHO_ACC_FLAG_* bits
__device__ unsigned int* __echo_acc_list;        // [capacity] claimed slot indices
__device__ unsigned int __echo_acc_used;         // entries in __echo_acc_list
__device__ unsigned int __echo_acc_capacity;     // slots in the table
__device__ unsigned int __echo_acc_error;        // 1 = a contribution was dropped
__device__ unsigned int* __echo_acc_host_error;  // mapped host u32 (per context), or 0
}

struct echo_acc_state {
  long long cells[ECHO_ACC_CELLS];
  bool nan, pos_inf, neg_inf;
};

// Mirror of ExactAcc::add.
static __device__ __forceinline__ void echo_acc_add(echo_acc_state* s, float v) {
  unsigned int bits = __float_as_uint(v);
  bool neg = (bits >> 31) != 0u;
  unsigned int e = (bits >> 23) & 0xffu;
  unsigned int f = bits & 0x7fffffu;
  if (e == 0u) return;  // zero or subnormal: flushed to zero
  if (e == 255u) {
    if (f != 0u) s->nan = true;
    else if (neg) s->neg_inf = true;
    else s->pos_inf = true;
    return;
  }
  unsigned long long m = (unsigned long long)(f | 0x800000u);
  unsigned int k = e - 1u;
  unsigned int cell = k / 32u;
  unsigned long long x = m << (k % 32u);
  long long lo = (long long)(x & 0xffffffffull);
  long long hi = (long long)(x >> 32);
#pragma unroll
  for (int i = 0; i + 1 < ECHO_ACC_CELLS; ++i) {
    if ((unsigned int)i == cell) {
      if (neg) {
        s->cells[i] = (long long)((unsigned long long)s->cells[i] - (unsigned long long)lo);
        s->cells[i + 1] = (long long)((unsigned long long)s->cells[i + 1] - (unsigned long long)hi);
      } else {
        s->cells[i] = (long long)((unsigned long long)s->cells[i] + (unsigned long long)lo);
        s->cells[i + 1] = (long long)((unsigned long long)s->cells[i + 1] + (unsigned long long)hi);
      }
    }
  }
}

// Mirror of ExactAcc::to_bits.
static __device__ __forceinline__ unsigned int echo_acc_to_bits(const echo_acc_state* s) {
  const unsigned int INF = 0x7f800000u;
  if (s->nan || (s->pos_inf && s->neg_inf)) return 0x7fc00000u;
  if (s->pos_inf) return INF;
  if (s->neg_inf) return 0x80000000u | INF;

  // Carry-normalize into a two's-complement integer: 10 u32 limbs plus a signed top word.
  unsigned int limbs[ECHO_ACC_CELLS];
  long long carry = 0;
#pragma unroll
  for (int i = 0; i < ECHO_ACC_CELLS; ++i) {
    long long c = s->cells[i];
    long long t = (c & 0xffffffffll) + carry;
    limbs[i] = (unsigned int)t;
    carry = (c >> 32) + (t >> 32);
  }
  bool neg = carry < 0;
  unsigned int sign = neg ? 0x80000000u : 0u;
  if (carry != 0 && carry != -1) return sign | INF;
  if (neg) {
    unsigned long long c = 1ull;
#pragma unroll
    for (int i = 0; i < ECHO_ACC_CELLS; ++i) {
      unsigned long long t = (unsigned long long)(~limbs[i]) + c;
      limbs[i] = (unsigned int)t;
      c = t >> 32;
    }
    if (c != 0ull) return sign | INF;
  }

  int top = ECHO_ACC_CELLS - 1;
  while (top >= 0 && limbs[top] == 0u) --top;
  if (top < 0) return 0u;  // exact zero -> +0.0
  int p = 32 * top + 31 - __clz(limbs[top]);
  if (p < 23) return 0u;  // magnitude < 2^-126 -> +0.0
  int exp = p - 22;
  int shift = p - 23;
  int w = shift / 32;
  int b = shift % 32;
  unsigned long long next = (w + 1 < ECHO_ACC_CELLS) ? (unsigned long long)limbs[w + 1] : 0ull;
  unsigned int m = (unsigned int)((((unsigned long long)limbs[w]) | (next << 32)) >> b) & 0xffffffu;
  if (shift > 0) {
    int r = shift - 1;
    int rw = r / 32;
    int rb = r % 32;
    bool round = ((limbs[rw] >> rb) & 1u) != 0u;
    bool sticky = (limbs[rw] & ((1u << rb) - 1u)) != 0u;
    for (int i = 0; i < rw; ++i) sticky |= limbs[i] != 0u;
    if (round && (sticky || (m & 1u) != 0u)) {
      m += 1u;
      if (m == (1u << 24)) {
        m >>= 1;
        exp += 1;
      }
    }
  }
  if (exp >= 255) return sign | INF;
  return sign | ((unsigned int)exp << 23) | (m & 0x7fffffu);
}

// Replacement for atom.add.f32 / red.add.f32 on a global (generic) address. Call ABI:
//   call __echo_atomic_add_f32, (param_b64_generic_ptr, param_b32_float);
extern "C" __device__ __noinline__ void __echo_atomic_add_f32(float* addr, float v) {
  unsigned long long key = (unsigned long long)addr;
  unsigned int cap = __echo_acc_capacity;
  unsigned long long* keys = __echo_acc_keys;
  unsigned int limit = cap < ECHO_ACC_MAX_PROBE ? cap : ECHO_ACC_MAX_PROBE;
  unsigned int slot = cap == 0u ? 0u : (unsigned int)(((key >> 2) * 0x9E3779B97F4A7C15ull) >> 32) % cap;
  bool found = false;
  for (unsigned int i = 0u; i < limit; ++i) {
    unsigned long long k = *(volatile unsigned long long*)&keys[slot];
    if (k == 0ull) {
      k = atomicCAS(&keys[slot], 0ull, key);
      if (k == 0ull) {
        unsigned int idx = atomicAdd(&__echo_acc_used, 1u);
        __echo_acc_list[idx] = slot;
        found = true;
        break;
      }
    }
    if (k == key) {
      found = true;
      break;
    }
    slot = (slot + 1u == cap) ? 0u : slot + 1u;
  }
  if (!found) {
    atomicExch(&__echo_acc_error, 1u);
    return;
  }

  unsigned int bits = __float_as_uint(v);
  bool neg = (bits >> 31) != 0u;
  unsigned int e = (bits >> 23) & 0xffu;
  unsigned int f = bits & 0x7fffffu;
  if (e == 0u) return;  // zero or subnormal: flushed to zero (address still registered)
  if (e == 255u) {
    unsigned int flag = f != 0u ? ECHO_ACC_FLAG_NAN : (neg ? ECHO_ACC_FLAG_NEG_INF : ECHO_ACC_FLAG_POS_INF);
    atomicOr(&__echo_acc_flags[slot], flag);
    return;
  }
  unsigned long long m = (unsigned long long)(f | 0x800000u);
  unsigned int k = e - 1u;
  unsigned long long x = m << (k % 32u);
  unsigned long long lo = x & 0xffffffffull;
  unsigned long long hi = x >> 32;
  if (neg) {
    lo = 0ull - lo;
    hi = 0ull - hi;
  }
  unsigned long long* c = (unsigned long long*)(__echo_acc_cells + (unsigned long long)slot * ECHO_ACC_CELLS + k / 32u);
  atomicAdd(c, lo);
  if (hi != 0ull) atomicAdd(c + 1, hi);
}

// One thread per list index < __echo_acc_used: fold the entry into *addr and clear it.
extern "C" __global__ void __echo_atomic_finalize() {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= __echo_acc_used) return;
  unsigned int slot = __echo_acc_list[i];
  float* addr = (float*)__echo_acc_keys[slot];
  long long* cells = __echo_acc_cells + (unsigned long long)slot * ECHO_ACC_CELLS;
  unsigned int flags = __echo_acc_flags[slot];
  echo_acc_state s;
#pragma unroll
  for (int j = 0; j < ECHO_ACC_CELLS; ++j) s.cells[j] = cells[j];
  s.nan = (flags & ECHO_ACC_FLAG_NAN) != 0u;
  s.pos_inf = (flags & ECHO_ACC_FLAG_POS_INF) != 0u;
  s.neg_inf = (flags & ECHO_ACC_FLAG_NEG_INF) != 0u;
  echo_acc_add(&s, *addr);
  *addr = __uint_as_float(echo_acc_to_bits(&s));
#pragma unroll
  for (int j = 0; j < ECHO_ACC_CELLS; ++j) cells[j] = 0;
  __echo_acc_flags[slot] = 0u;
  __echo_acc_keys[slot] = 0ull;
}

// One thread: clear the list and publish a set error flag to the mapped host flag (sticky:
// it is only ever set to 1). A null host pointer (a module bound by hand, as in the
// echo-gpu-tests harness) is skipped.
extern "C" __global__ void __echo_atomic_reset() {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    __echo_acc_used = 0u;
    unsigned int* host = __echo_acc_host_error;
    if (*(volatile unsigned int*)&__echo_acc_error != 0u && host != 0) *(volatile unsigned int*)host = 1u;
  }
}
