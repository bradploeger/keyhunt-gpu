// Host-side validation of secp256k1.h against Python-generated reference vectors.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../src/secp256k1.h"
#include "vectors.h"

static int fails = 0;

static void hex(const uint64_t v[4], char *out) {
  sprintf(out, "%016llX%016llX%016llX%016llX",
          (unsigned long long)v[3], (unsigned long long)v[2],
          (unsigned long long)v[1], (unsigned long long)v[0]);
}

static void check(const char *what, int idx, const uint64_t got[4], const uint64_t want[4]) {
  if (!fe_eq(got, want)) {
    char a[80], b[80];
    hex(got, a); hex(want, b);
    printf("  FAIL %s[%d]\n    got  %s\n    want %s\n", what, idx, a, b);
    fails++;
  }
}

int main() {
  printf("field multiply (%d vectors)\n", kMulN);
  for (int i = 0; i < kMulN; i++) {
    uint64_t r[4];
    fe_mul(r, kMul[i].a, kMul[i].b);
    check("mul", i, r, kMul[i].r);
  }

  printf("field add / sub / inverse (%d vectors)\n", kAddN);
  for (int i = 0; i < kAddN; i++) {
    uint64_t r[4];
    fe_add(r, kAdd[i].a, kAdd[i].b); check("add", i, r, kAdd[i].s);
    fe_sub(r, kAdd[i].a, kAdd[i].b); check("sub", i, r, kAdd[i].d);
    fe_inv(r, kAdd[i].b);            check("inv", i, r, kAdd[i].i);
  }

  printf("scalar multiplication k*G (%d vectors)\n", kEcN);
  for (int i = 0; i < kEcN; i++) {
    uint64_t x[4], y[4];
    ec_scalar_mul_g(x, y, kEc[i].k);
    check("ec.x", i, x, kEc[i].x);
    check("ec.y", i, y, kEc[i].y);
  }

  // Round trip: batch-inversion style affine addition must agree with scalar mul.
  printf("incremental affine addition vs scalar multiply\n");
  {
    uint64_t gx[4], gy[4];
    ec_generator(gx, gy);
    uint64_t px[4], py[4];
    uint64_t two[4] = {2, 0, 0, 0};
    ec_scalar_mul_g(px, py, two);   // start at 2G: P+G is never a doubling
    for (uint64_t k = 3; k <= 300; k++) {
      uint64_t dx[4], dxi[4];
      fe_sub(dx, gx, px);
      fe_inv(dxi, dx);
      uint64_t nx[4], ny[4];
      ec_add_affine_pre(nx, ny, px, py, gx, gy, dxi);
      fe_set(px, nx); fe_set(py, ny);
      uint64_t kk[4] = {k, 0, 0, 0}, ex[4], ey[4];
      ec_scalar_mul_g(ex, ey, kk);
      if (!fe_eq(px, ex) || !fe_eq(py, ey)) {
        printf("  FAIL at k=%llu\n", (unsigned long long)k);
        fails++;
        break;
      }
    }
  }

  // Batch (Montgomery) inversion must match individual inversions.
  printf("batch inversion\n");
  {
    const int N = 64;
    uint64_t v[N][4], pp[N][4], acc[4];
    for (int i = 0; i < N; i++) {
      v[i][0] = 0x9E3779B97F4A7C15ULL * (i + 1) + 7;
      v[i][1] = 0xC2B2AE3D27D4EB4FULL ^ (uint64_t)i;
      v[i][2] = 0x165667B19E3779F9ULL + i;
      v[i][3] = 0x0000000000000001ULL * i;
    }
    fe_one(acc);
    for (int i = 0; i < N; i++) { fe_set(pp[i], acc); fe_mul(acc, acc, v[i]); }
    uint64_t inv[4];
    fe_inv(inv, acc);
    for (int i = N - 1; i >= 0; i--) {
      uint64_t vi[4];
      fe_mul(vi, inv, pp[i]);
      fe_mul(inv, inv, v[i]);
      uint64_t ref[4];
      fe_inv(ref, v[i]);
      check("batchinv", i, vi, ref);
    }
  }

  if (fails == 0) printf("\nALL TESTS PASSED\n");
  else printf("\n%d FAILURES\n", fails);
  return fails ? 1 : 0;
}
