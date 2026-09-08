// Reproduces the exact loop structure of search_kernel on the CPU and checks:
//   1. every offset in the range is visited exactly once
//   2. the X coordinate produced for each offset equals ((K0+offset)*G).x
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include "../src/secp256k1.h"

#ifndef HSIZE
#define HSIZE 16
#endif
#define GRP_SIZE (2 * HSIZE)

static uint64_t GX[HSIZE + 1][4], GY[HSIZE + 1][4];

static void add64_256(uint64_t v[4], uint64_t a) {
  uint64_t c = a;
  for (int i = 0; i < 4 && c; i++) { uint64_t o = v[i]; v[i] += c; c = (v[i] < o) ? 1 : 0; }
}

int main() {
  for (int i = 0; i < HSIZE; i++) {
    uint64_t k[4] = {(uint64_t)(i + 1), 0, 0, 0};
    ec_scalar_mul_g(GX[i], GY[i], k);
  }
  uint64_t kj[4] = {GRP_SIZE, 0, 0, 0};
  ec_scalar_mul_g(GX[HSIZE], GY[HSIZE], kj);

  // arbitrary 27-byte prefix, 8 unknown bits for a quick exhaustive check
  uint64_t K0[4] = {0x1122334455667788ULL, 0x99AABBCCDDEEFF00ULL,
                    0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
  K0[0] &= ~0xFFULL;  // clear the unknown low bits
  #ifndef UBITS
#define UBITS 8
#endif
  const int unknownBits = UBITS;
  const uint64_t totalKeys = 1ULL << unknownBits;

  #ifndef NTHR
#define NTHR 4
#endif
  const uint32_t nThreads = NTHR;
  const uint64_t keysPerThread = totalKeys / nThreads;
  const uint64_t groups = keysPerThread / GRP_SIZE;

  std::vector<int> visits(totalKeys, 0);
  int errs = 0;

  for (uint32_t gid = 0; gid < nThreads; gid++) {
    const uint64_t threadStart = (uint64_t)gid * keysPerThread;

    uint64_t startK[4]; fe_set(startK, K0);
    add64_256(startK, threadStart + HSIZE);
    uint64_t X[4], Y[4];
    ec_scalar_mul_g(X, Y, startK);

    uint64_t pp[HSIZE + 1][4];

    for (uint64_t g = 0; g < groups; g++) {
      const uint64_t centre = threadStart + g * GRP_SIZE + HSIZE;

      uint64_t acc[4]; fe_one(acc);
      for (int i = 0; i <= HSIZE; i++) {
        fe_set(pp[i], acc);
        uint64_t d[4]; fe_sub(d, GX[i], X);
        fe_mul(acc, acc, d);
      }
      uint64_t inv[4]; fe_inv(inv, acc);

      auto emit = [&](const uint64_t x[4], uint64_t off) {
        if (off >= totalKeys) { printf("  offset %llu out of range\n",
                                       (unsigned long long)off); errs++; return; }
        visits[off]++;
        uint64_t k[4]; fe_set(k, K0); add64_256(k, off);
        uint64_t ex[4], ey[4];
        ec_scalar_mul_g(ex, ey, k);
        if (!fe_eq(x, ex)) {
          printf("  X mismatch at offset %llu\n", (unsigned long long)off);
          errs++;
        }
      };

      emit(X, centre);

      uint64_t jx[4], jy[4];
      for (int i = HSIZE; i >= 0; i--) {
        uint64_t d[4], di[4];
        fe_sub(d, GX[i], X);
        fe_mul(di, inv, pp[i]);
        fe_mul(inv, inv, d);

        if (i == HSIZE) {
          ec_add_affine_pre(jx, jy, X, Y, GX[i], GY[i], di);
          continue;
        }
        const uint64_t step = (uint64_t)(i + 1);
        {
          uint64_t dy[4], s[4], t[4], nx[4];
          fe_add(dy, Y, GY[i]);
          fe_mul(s, dy, di);
          fe_sqr(t, s);
          fe_sub(t, t, X);
          fe_sub(nx, t, GX[i]);
          emit(nx, centre - step);
        }
        if (step < HSIZE) {
          uint64_t dy[4], s[4], t[4], nx[4];
          fe_sub(dy, GY[i], Y);
          fe_mul(s, dy, di);
          fe_sqr(t, s);
          fe_sub(t, t, X);
          fe_sub(nx, t, GX[i]);
          emit(nx, centre + step);
        }
      }
      fe_set(X, jx);
      fe_set(Y, jy);
    }
  }

  int missing = 0, dup = 0;
  for (uint64_t i = 0; i < totalKeys; i++) {
    if (visits[i] == 0) missing++;
    else if (visits[i] > 1) dup++;
  }
  printf("range 2^%d = %llu keys, %u threads, %llu groups/thread\n",
         unknownBits, (unsigned long long)totalKeys, nThreads,
         (unsigned long long)groups);
  printf("missing offsets : %d\n", missing);
  printf("duplicate       : %d\n", dup);
  printf("X errors        : %d\n", errs);
  if (!missing && !dup && !errs) { printf("\nKERNEL LOGIC OK\n"); return 0; }
  printf("\nKERNEL LOGIC FAILED\n");
  return 1;
}
