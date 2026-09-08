// secp256k1.h -- 256-bit field and elliptic curve arithmetic.
// Compiles as plain C++ (host) and as CUDA device code.
//
// Field: p = 2^256 - 2^32 - 977
// Numbers are 4 x uint64 limbs, little-endian (d[0] = least significant).
#pragma once
#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>          // __umulh, for the MSVC host path
#endif

#if defined(__CUDACC__)
  #define FD __host__ __device__ __forceinline__
#else
  #define FD inline
#endif

// ---------------------------------------------------------------- constants
#define P0 0xFFFFFFFEFFFFFC2FULL
#define P1 0xFFFFFFFFFFFFFFFFULL
#define P2 0xFFFFFFFFFFFFFFFFULL
#define P3 0xFFFFFFFFFFFFFFFFULL
// 2^256 mod p
#define PC 0x1000003D1ULL

// group order n
#define N0 0xBFD25E8CD0364141ULL
#define N1 0xBAAEDCE6AF48A03BULL
#define N2 0xFFFFFFFFFFFFFFFEULL
#define N3 0xFFFFFFFFFFFFFFFFULL

// ---------------------------------------------------------------- primitives
// High 64 bits of a 64x64 product. MSVC has no __int128, so it needs __umulh.
FD uint64_t mulhi64(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
  return __umul64hi(a, b);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
  return __umulh(a, b);
#elif defined(_MSC_VER)
#error "64-bit build required: 32-bit MSVC has no 64x64 high-multiply intrinsic"
#else
  return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
#endif
}

FD uint64_t addc64(uint64_t a, uint64_t b, uint64_t cin, uint64_t *cout) {
  uint64_t s = a + b;
  uint64_t c = (uint64_t)(s < a);
  uint64_t s2 = s + cin;
  c += (uint64_t)(s2 < s);
  *cout = c;
  return s2;
}

FD uint64_t subb64(uint64_t a, uint64_t b, uint64_t bin, uint64_t *bout) {
  uint64_t d = a - b;
  uint64_t br = (uint64_t)(a < b);
  uint64_t d2 = d - bin;
  br += (uint64_t)(d < bin);
  *bout = br;
  return d2;
}

FD uint64_t u256_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t c = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) r[i] = addc64(a[i], b[i], c, &c);
  return c;
}

FD uint64_t u256_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t b_ = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) r[i] = subb64(a[i], b[i], b_, &b_);
  return b_;
}

FD bool u256_ge_p(const uint64_t a[4]) {
  if (a[3] != P3) return a[3] > P3;
  if (a[2] != P2) return a[2] > P2;
  if (a[1] != P1) return a[1] > P1;
  return a[0] >= P0;
}

FD bool fe_is_zero(const uint64_t a[4]) {
  return (a[0] | a[1] | a[2] | a[3]) == 0ULL;
}

FD bool fe_eq(const uint64_t a[4], const uint64_t b[4]) {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

FD void fe_set(uint64_t r[4], const uint64_t a[4]) {
#pragma unroll
  for (int i = 0; i < 4; i++) r[i] = a[i];
}

FD void fe_zero(uint64_t r[4]) { r[0] = r[1] = r[2] = r[3] = 0; }

FD void fe_one(uint64_t r[4]) { r[0] = 1; r[1] = r[2] = r[3] = 0; }

// ---------------------------------------------------------------- field ops
FD void fe_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t c = u256_add(r, a, b);
  if (c) {  // 2^256 == PC (mod p); cannot carry again
    uint64_t t[4] = {PC, 0, 0, 0}, dummy[4];
    u256_add(dummy, r, t);
    fe_set(r, dummy);
  }
  if (u256_ge_p(r)) {
    uint64_t pp[4] = {P0, P1, P2, P3};
    u256_sub(r, r, pp);
  }
}

FD void fe_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t brw = u256_sub(r, a, b);
  if (brw) {
    uint64_t pp[4] = {P0, P1, P2, P3};
    u256_add(r, r, pp);
  }
}

FD void fe_neg(uint64_t r[4], const uint64_t a[4]) {
  if (fe_is_zero(a)) { fe_zero(r); return; }
  uint64_t pp[4] = {P0, P1, P2, P3};
  u256_sub(r, pp, a);
}

// r[5] = a[4] * b   (single limb multiplier)
FD void mul256x64(uint64_t r[5], const uint64_t a[4], uint64_t b) {
  uint64_t l0 = a[0] * b, h0 = mulhi64(a[0], b);
  uint64_t l1 = a[1] * b, h1 = mulhi64(a[1], b);
  uint64_t l2 = a[2] * b, h2 = mulhi64(a[2], b);
  uint64_t l3 = a[3] * b, h3 = mulhi64(a[3], b);
  uint64_t c = 0;
  r[0] = l0;
  r[1] = addc64(h0, l1, 0, &c);
  r[2] = addc64(h1, l2, c, &c);
  r[3] = addc64(h2, l3, c, &c);
  r[4] = h3 + c;
}

// out[8] = a[4] * b[4]
FD void u256_mul(uint64_t out[8], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t t[5];
  mul256x64(t, a, b[0]);
  out[0] = t[0]; out[1] = t[1]; out[2] = t[2]; out[3] = t[3]; out[4] = t[4];
  out[5] = 0; out[6] = 0; out[7] = 0;
#pragma unroll
  for (int j = 1; j < 4; j++) {
    mul256x64(t, a, b[j]);
    uint64_t c = 0;
    out[j + 0] = addc64(out[j + 0], t[0], c, &c);
    out[j + 1] = addc64(out[j + 1], t[1], c, &c);
    out[j + 2] = addc64(out[j + 2], t[2], c, &c);
    out[j + 3] = addc64(out[j + 3], t[3], c, &c);
    out[j + 4] = addc64(out[j + 4], t[4], c, &c);
    if (j + 5 < 8) out[j + 5] += c;  // for j==3 the carry is provably 0
  }
}

// Fold a 512-bit value down to a canonical 256-bit residue mod p.
FD void fe_reduce512(uint64_t r[4], const uint64_t t[8]) {
  uint64_t s[5];
  mul256x64(s, t + 4, PC);           // s = hi * 2^256 (mod p)

  uint64_t c = 0;
  s[0] = addc64(s[0], t[0], c, &c);
  s[1] = addc64(s[1], t[1], c, &c);
  s[2] = addc64(s[2], t[2], c, &c);
  s[3] = addc64(s[3], t[3], c, &c);
  s[4] += c;                          // s[4] < 2^34, no overflow

  // fold s[4] * 2^256 back in
  uint64_t f[4];
  f[0] = s[4] * PC;
  f[1] = mulhi64(s[4], PC);
  f[2] = 0; f[3] = 0;
  r[0] = s[0]; r[1] = s[1]; r[2] = s[2]; r[3] = s[3];
  uint64_t c2 = u256_add(r, r, f);
  if (c2) {                           // result is now tiny; adding PC cannot carry
    uint64_t g[4] = {PC, 0, 0, 0};
    u256_add(r, r, g);
  }
  if (u256_ge_p(r)) {
    uint64_t pp[4] = {P0, P1, P2, P3};
    u256_sub(r, r, pp);
  }
}

FD void fe_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t t[8];
  u256_mul(t, a, b);
  fe_reduce512(r, t);
}

FD void fe_sqr(uint64_t r[4], const uint64_t a[4]) { fe_mul(r, a, a); }

FD void fe_sqr_n(uint64_t r[4], const uint64_t a[4], int n) {
  fe_sqr(r, a);
  for (int i = 1; i < n; i++) fe_sqr(r, r);
}

// r = a^(p-2) = a^-1  (libsecp256k1 addition chain: 255 squarings, 15 mults)
FD void fe_inv(uint64_t r[4], const uint64_t a[4]) {
  uint64_t x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4], x88[4];
  uint64_t x176[4], x220[4], x223[4], t[4];

  fe_sqr(x2, a);       fe_mul(x2, x2, a);
  fe_sqr(x3, x2);      fe_mul(x3, x3, a);
  fe_sqr_n(x6, x3, 3); fe_mul(x6, x6, x3);
  fe_sqr_n(x9, x6, 3); fe_mul(x9, x9, x3);
  fe_sqr_n(x11, x9, 2);   fe_mul(x11, x11, x2);
  fe_sqr_n(x22, x11, 11); fe_mul(x22, x22, x11);
  fe_sqr_n(x44, x22, 22); fe_mul(x44, x44, x22);
  fe_sqr_n(x88, x44, 44); fe_mul(x88, x88, x44);
  fe_sqr_n(x176, x88, 88);fe_mul(x176, x176, x88);
  fe_sqr_n(x220, x176, 44); fe_mul(x220, x220, x44);
  fe_sqr_n(x223, x220, 3);  fe_mul(x223, x223, x3);

  fe_sqr_n(t, x223, 23); fe_mul(t, t, x22);
  fe_sqr_n(t, t, 5);     fe_mul(t, t, a);
  fe_sqr_n(t, t, 3);     fe_mul(t, t, x2);
  fe_sqr_n(t, t, 2);     fe_mul(t, t, a);
  fe_set(r, t);
}

// ---------------------------------------------------------------- curve
// secp256k1 generator
FD void ec_generator(uint64_t gx[4], uint64_t gy[4]) {
  gx[0] = 0x59F2815B16F81798ULL; gx[1] = 0x029BFCDB2DCE28D9ULL;
  gx[2] = 0x55A06295CE870B07ULL; gx[3] = 0x79BE667EF9DCBBACULL;
  gy[0] = 0x9C47D08FFB10D4B8ULL; gy[1] = 0xFD17B448A6855419ULL;
  gy[2] = 0x5DA4FBFC0E1108A8ULL; gy[3] = 0x483ADA7726A3C465ULL;
}

// Jacobian point. Z == 0 means point at infinity.
struct JPoint { uint64_t x[4], y[4], z[4]; };

FD void j_set_infinity(JPoint *P) { fe_one(P->x); fe_one(P->y); fe_zero(P->z); }
FD bool j_is_infinity(const JPoint *P) { return fe_is_zero(P->z); }

FD void j_double(JPoint *R, const JPoint *P) {
  if (j_is_infinity(P)) { j_set_infinity(R); return; }
  uint64_t A[4], B[4], C[4], D[4], t1[4], t2[4];
  fe_sqr(A, P->y);                     // A = Y^2
  fe_mul(B, P->x, A); fe_add(B, B, B); fe_add(B, B, B);   // B = 4*X*A
  fe_sqr(C, A); fe_add(C, C, C); fe_add(C, C, C); fe_add(C, C, C); // C = 8*A^2
  fe_sqr(t1, P->x); fe_add(D, t1, t1); fe_add(D, D, t1);  // D = 3*X^2  (a = 0)
  fe_sqr(t1, D); fe_sub(t1, t1, B); fe_sub(t1, t1, B);    // X' = D^2 - 2B
  fe_mul(t2, P->y, P->z); fe_add(t2, t2, t2);             // Z' = 2*Y*Z
  uint64_t nx[4]; fe_set(nx, t1);
  fe_sub(t1, B, nx); fe_mul(t1, D, t1); fe_sub(t1, t1, C); // Y' = D*(B-X') - C
  fe_set(R->x, nx); fe_set(R->y, t1); fe_set(R->z, t2);
}

// R = P + Q where Q is affine (qz == 1)
FD void j_add_affine(JPoint *R, const JPoint *P, const uint64_t qx[4], const uint64_t qy[4]) {
  if (j_is_infinity(P)) {
    fe_set(R->x, qx); fe_set(R->y, qy); fe_one(R->z); return;
  }
  uint64_t z2[4], u2[4], s2[4], h[4], r_[4], h2[4], h3[4], u1h2[4], t[4];
  fe_sqr(z2, P->z);
  fe_mul(u2, qx, z2);                 // U2 = X2 * Z1^2
  fe_mul(s2, qy, z2); fe_mul(s2, s2, P->z);  // S2 = Y2 * Z1^3
  fe_sub(h, u2, P->x);
  fe_sub(r_, s2, P->y);
  if (fe_is_zero(h)) {
    if (fe_is_zero(r_)) { JPoint tmp = *P; j_double(R, &tmp); return; }
    j_set_infinity(R); return;
  }
  fe_sqr(h2, h); fe_mul(h3, h2, h);
  fe_mul(u1h2, P->x, h2);
  fe_sqr(t, r_); fe_sub(t, t, h3); fe_sub(t, t, u1h2); fe_sub(t, t, u1h2);
  uint64_t nx[4]; fe_set(nx, t);
  fe_sub(t, u1h2, nx); fe_mul(t, r_, t);
  uint64_t ty[4]; fe_mul(ty, P->y, h3); fe_sub(t, t, ty);
  uint64_t nz[4]; fe_mul(nz, P->z, h);
  fe_set(R->x, nx); fe_set(R->y, t); fe_set(R->z, nz);
}

FD void j_to_affine(uint64_t ax[4], uint64_t ay[4], const JPoint *P) {
  if (j_is_infinity(P)) { fe_zero(ax); fe_zero(ay); return; }
  uint64_t zi[4], zi2[4], zi3[4];
  fe_inv(zi, P->z);
  fe_sqr(zi2, zi);
  fe_mul(zi3, zi2, zi);
  fe_mul(ax, P->x, zi2);
  fe_mul(ay, P->y, zi3);
}

// R = k * G, k given as 4 limbs. Simple, constant-ish, used off the hot path.
FD void ec_scalar_mul_g(uint64_t rx[4], uint64_t ry[4], const uint64_t k[4]) {
  uint64_t gx[4], gy[4];
  ec_generator(gx, gy);
  JPoint acc; j_set_infinity(&acc);
  for (int i = 255; i >= 0; i--) {
    JPoint t = acc;
    j_double(&acc, &t);
    if ((k[i >> 6] >> (i & 63)) & 1ULL) {
      JPoint t2 = acc;
      j_add_affine(&acc, &t2, gx, gy);
    }
  }
  j_to_affine(rx, ry, &acc);
}

// Affine addition, inverse of (x2-x1) supplied by the caller (batch inversion).
FD void ec_add_affine_pre(uint64_t rx[4], uint64_t ry[4],
                          const uint64_t x1[4], const uint64_t y1[4],
                          const uint64_t x2[4], const uint64_t y2[4],
                          const uint64_t dxinv[4]) {
  uint64_t dy[4], s[4], t[4];
  fe_sub(dy, y2, y1);
  fe_mul(s, dy, dxinv);
  fe_sqr(t, s);
  fe_sub(t, t, x1);
  fe_sub(t, t, x2);
  uint64_t nx[4]; fe_set(nx, t);
  fe_sub(t, x1, nx);
  fe_mul(t, s, t);
  fe_sub(t, t, y1);
  fe_set(rx, nx); fe_set(ry, t);
}

// X coordinate only (the search loop never needs Y).
FD void ec_addx_affine_pre(uint64_t rx[4],
                           const uint64_t x1[4], const uint64_t y1[4],
                           const uint64_t x2[4], const uint64_t y2[4],
                           const uint64_t dxinv[4]) {
  uint64_t dy[4], s[4], t[4];
  fe_sub(dy, y2, y1);
  fe_mul(s, dy, dxinv);
  fe_sqr(t, s);
  fe_sub(t, t, x1);
  fe_sub(rx, t, x2);
}
