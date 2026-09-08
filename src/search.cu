// search.cu -- GPU search over a private-key range defined by a fixed high-order
// prefix, matching generated compressed public keys against a target set.
//
// Build: see Makefile.  Run with --help.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <fstream>

#if defined(_WIN32)
#include <io.h>          // _isatty, _fileno
#else
#include <unistd.h>      // isatty, fileno
#endif

#include "progress.h"
#include <sstream>

#include "targets.h"


// ----------------------------------------------------------------- tunables
#ifndef HSIZE
#define HSIZE 128                   // half group size; GRP_SIZE keys per batch inversion
#endif
#define GRP_SIZE (2 * HSIZE)


#define MAX_RESULTS 4096

#define CUDA_CHECK(x) do {                                                     \
  cudaError_t _e = (x);                                                        \
  if (_e != cudaSuccess) {                                                     \
    fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #x, __FILE__, __LINE__,  \
            cudaGetErrorString(_e));                                           \
    exit(1);                                                                   \
  }                                                                            \
} while (0)

struct Result { uint64_t offset; uint32_t tidx; uint32_t pad; };

// Table index i in [0, HSIZE) holds (i+1)*G.  Index HSIZE holds GRP_SIZE*G,
// the jump used to advance the group centre.  Read uniformly across a warp,
// so __constant__ broadcast is the right storage class.
__constant__ uint64_t c_gx[HSIZE + 1][4];
__constant__ uint64_t c_gy[HSIZE + 1][4];

// ------------------------------------------------------------------ matching
__device__ __forceinline__ void check_x(
    const uint64_t x[4], uint64_t off,
    const uint32_t *s_filter, const uint64_t *table, const uint32_t *tidx,
    uint32_t mask, Result *out, uint32_t *outCount) {

  uint32_t bit = filter_bit(x);
  if (((s_filter[bit >> 5] >> (bit & 31)) & 1u) == 0u) return;   // ~87% exit here

  uint64_t tag = x_tag(x);
  uint32_t slot = table_slot(x, mask);
  for (;;) {
    uint64_t v = table[slot];
    if (v == 0ULL) return;
    if (v == tag) {
      uint32_t k = atomicAdd(outCount, 1u);
      if (k < MAX_RESULTS) { out[k].offset = off; out[k].tidx = tidx[slot]; }
      return;
    }
    slot = (slot + 1) & mask;
  }
}

// Kept out of line so the inversion's temporaries don't inflate register
// allocation for the whole kernel.  Try removing __noinline__ when tuning.
__device__ __noinline__ void fe_inv_ni(uint64_t r[4], const uint64_t a[4]) {
  fe_inv(r, a);
}

// -------------------------------------------------------------------- kernel
__global__ void __launch_bounds__(256) search_kernel(
    uint64_t *pointX, uint64_t *pointY,
    const uint32_t * __restrict__ g_filter,
    const uint64_t * __restrict__ table,
    const uint32_t * __restrict__ tidx,
    uint32_t mask,
    uint64_t groups, uint64_t groupBase, uint64_t keysPerThread,
    Result *out, uint32_t *outCount) {

  extern __shared__ uint32_t s_filter[];
  for (uint32_t i = threadIdx.x; i < FILTER_WORDS; i += blockDim.x)
    s_filter[i] = g_filter[i];
  __syncthreads();

  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t threadStart = (uint64_t)gid * keysPerThread;

  uint64_t X[4], Y[4];
#pragma unroll
  for (int i = 0; i < 4; i++) { X[i] = pointX[gid * 4 + i]; Y[i] = pointY[gid * 4 + i]; }

  uint64_t pp[HSIZE + 1][4];        // partial products (local memory, coalesced)

  for (uint64_t g = 0; g < groups; g++) {
    const uint64_t centre = threadStart + (groupBase + g) * GRP_SIZE + HSIZE;

    // ---- forward pass: accumulate prod(dx_i), storing running products
    uint64_t acc[4];
    fe_one(acc);
#pragma unroll 1
    for (int i = 0; i <= HSIZE; i++) {
      fe_set(pp[i], acc);
      uint64_t d[4];
      fe_sub(d, c_gx[i], X);
      fe_mul(acc, acc, d);
    }
    if (fe_is_zero(acc)) break;     // degenerate (needs a doubling); cannot happen
                                    // for realistic ranges, see README

    uint64_t inv[4];
    fe_inv_ni(inv, acc);            // one inversion per GRP_SIZE keys

    // ---- the group centre itself
    check_x(X, centre, s_filter, table, tidx, mask, out, outCount);

    // ---- backward pass: recover each dx_i^-1 and use it immediately
    uint64_t jx[4], jy[4];
#pragma unroll 1
    for (int i = HSIZE; i >= 0; i--) {
      uint64_t d[4], di[4];
      fe_sub(d, c_gx[i], X);
      fe_mul(di, inv, pp[i]);       // di = dx_i^-1
      fe_mul(inv, inv, d);

      if (i == HSIZE) {
        // advance the centre by GRP_SIZE*G; needs the full point, so keep Y
        ec_add_affine_pre(jx, jy, X, Y, c_gx[i], c_gy[i], di);
        continue;
      }

      const uint64_t step = (uint64_t)(i + 1);

      // P - (i+1)G.  Negating Y flips the sign of the slope, and the new X
      // depends only on s^2, so the negation itself can be skipped.
      {
        uint64_t dy[4], s[4], t[4], nx[4];
        fe_add(dy, Y, c_gy[i]);
        fe_mul(s, dy, di);
        fe_sqr(t, s);
        fe_sub(t, t, X);
        fe_sub(nx, t, c_gx[i]);
        check_x(nx, centre - step, s_filter, table, tidx, mask, out, outCount);
      }

      // P + (i+1)G.  step == HSIZE would land on centre+HSIZE, which belongs to
      // the next group's range, so it is skipped here.
      if (step < HSIZE) {
        uint64_t dy[4], s[4], t[4], nx[4];
        fe_sub(dy, c_gy[i], Y);
        fe_mul(s, dy, di);
        fe_sqr(t, s);
        fe_sub(t, t, X);
        fe_sub(nx, t, c_gx[i]);
        check_x(nx, centre + step, s_filter, table, tidx, mask, out, outCount);
      }
    }

    fe_set(X, jx);
    fe_set(Y, jy);
  }

#pragma unroll
  for (int i = 0; i < 4; i++) { pointX[gid * 4 + i] = X[i]; pointY[gid * 4 + i] = Y[i]; }
}

// -------------------------------------------------- device arithmetic selftest
__global__ void selftest_kernel(const uint64_t *k, uint64_t *outX, uint64_t *outY, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t x[4], y[4];
  ec_scalar_mul_g(x, y, k + i * 4);
#pragma unroll
  for (int j = 0; j < 4; j++) { outX[i * 4 + j] = x[j]; outY[i * 4 + j] = y[j]; }
}

// ==========================================================================
//                              host side
// ==========================================================================


// -------------------------------------------------------------- start points
// centre_t = (K0 + groupBase*GRP_SIZE + HSIZE + t*keysPerThread) * G
static void compute_start_points(const uint64_t K0[4], uint64_t groupBase,
                                 uint64_t keysPerThread, uint32_t nThreads,
                                 std::vector<uint64_t> &hx, std::vector<uint64_t> &hy) {
  uint64_t base[4];
  fe_set(base, K0);
  add64_256(base, groupBase * GRP_SIZE + HSIZE);

  uint64_t bx[4], by[4];
  ec_scalar_mul_g(bx, by, base);

  uint64_t stepK[4] = {keysPerThread, 0, 0, 0}, sx[4], sy[4];
  ec_scalar_mul_g(sx, sy, stepK);

  std::vector<JPoint> jp(nThreads);
  fe_set(jp[0].x, bx); fe_set(jp[0].y, by); fe_one(jp[0].z);
  for (uint32_t t = 1; t < nThreads; t++) j_add_affine(&jp[t], &jp[t - 1], sx, sy);

  // batch-normalise all Z values with a single inversion
  std::vector<uint64_t> pp((size_t)nThreads * 4);
  uint64_t acc[4]; fe_one(acc);
  for (uint32_t t = 0; t < nThreads; t++) {
    fe_set(&pp[(size_t)t * 4], acc);
    fe_mul(acc, acc, jp[t].z);
  }
  uint64_t inv[4]; fe_inv(inv, acc);

  hx.assign((size_t)nThreads * 4, 0);
  hy.assign((size_t)nThreads * 4, 0);
  for (int t = (int)nThreads - 1; t >= 0; t--) {
    uint64_t zi[4], zi2[4], zi3[4];
    fe_mul(zi, inv, &pp[(size_t)t * 4]);
    fe_mul(inv, inv, jp[t].z);
    fe_sqr(zi2, zi);
    fe_mul(zi3, zi2, zi);
    fe_mul(&hx[(size_t)t * 4], jp[t].x, zi2);
    fe_mul(&hy[(size_t)t * 4], jp[t].y, zi3);
  }
}

static void upload_gtable() {
  uint64_t gx[(HSIZE + 1) * 4], gy[(HSIZE + 1) * 4];
  for (int i = 0; i < HSIZE; i++) {
    uint64_t k[4] = {(uint64_t)(i + 1), 0, 0, 0};
    ec_scalar_mul_g(gx + i * 4, gy + i * 4, k);
  }
  uint64_t kj[4] = {(uint64_t)GRP_SIZE, 0, 0, 0};
  ec_scalar_mul_g(gx + HSIZE * 4, gy + HSIZE * 4, kj);
  CUDA_CHECK(cudaMemcpyToSymbol(c_gx, gx, sizeof(gx)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_gy, gy, sizeof(gy)));
}

// ------------------------------------------------------------------ reporting
static void report(const Config &cfg, const Target &tg,
                   const uint64_t priv[4], bool negated) {
  std::string ph = hex256(priv);
  printf("\n");
  printf("========================================================================\n");
  printf("  MATCH FOUND\n");
  printf("  compressed public key : %s\n", tg.pub.c_str());
  printf("  private key (256 bit) : %s\n", ph.c_str());
  if (negated)
    printf("  note: X matched with opposite Y parity, so the key is n - k\n");
  printf("========================================================================\n");
  fflush(stdout);

  FILE *f = fopen(cfg.outFile.c_str(), "a");
  if (f) {
    fprintf(f, "pubkey=%s privkey=%s%s\n", tg.pub.c_str(), ph.c_str(),
            negated ? " (n-k)" : "");
    fflush(f);
    fclose(f);
  }
  if (!cfg.notifyCmd.empty()) {
    std::string c = cfg.notifyCmd;
    size_t p;
    while ((p = c.find("%P")) != std::string::npos) c.replace(p, 2, tg.pub);
    while ((p = c.find("%K")) != std::string::npos) c.replace(p, 2, ph);
    if (system(c.c_str()) != 0) fprintf(stderr, "notify command returned non-zero\n");
  }
}

// Recompute from scratch and confirm before announcing anything.
static bool verify_and_report(const Config &cfg, const std::vector<Target> &tg,
                              const std::map<std::string, size_t> &byX,
                              const uint64_t K0[4], uint64_t offset) {
  uint64_t priv[4];
  fe_set(priv, K0);
  add64_256(priv, offset);

  uint64_t x[4], y[4];
  ec_scalar_mul_g(x, y, priv);
  auto it = byX.find(hex256(x));
  if (it == byX.end()) return false;             // tag collision

  const Target &t = tg[it->second];
  bool parityMatches = ((y[0] & 1ULL) == t.parity);
  if (parityMatches) { report(cfg, t, priv, false); }
  else { uint64_t nk[4]; order_minus(nk, priv); report(cfg, t, nk, true); }
  return true;
}

// ---------------------------------------------------------------- the search
static int run_search(const Config &cfg, const std::vector<Target> &targets,
                      const uint64_t K0[4], int unknownBits, bool quiet) {
  const uint64_t totalKeys = (unknownBits >= 64) ? 0 : (1ULL << unknownBits);
  uint32_t nThreads = (uint32_t)cfg.blocks * (uint32_t)cfg.threads;
  uint64_t keysPerThread = totalKeys / nThreads;
  uint64_t groupsPerThread = keysPerThread / GRP_SIZE;

  if (!quiet) {
    printf("range          : %s .. +2^%d\n", hex256(K0).c_str(), unknownBits);
    printf("targets        : %zu\n", targets.size());
    printf("grid           : %d blocks x %d threads = %u threads\n",
           cfg.blocks, cfg.threads, nThreads);
    printf("keys/thread    : %" PRIu64 "  (%" PRIu64 " groups of %d)\n",
           keysPerThread, groupsPerThread, GRP_SIZE);
    fflush(stdout);
  }

  TargetTable tt = build_table(targets);
  std::map<std::string, size_t> byX;
  for (size_t i = 0; i < targets.size(); i++) byX[hex256(targets[i].x)] = i;

  // The prefilter bitmap lives in dynamic shared memory: FILTER_WORDS*4 bytes.
  // At FILTER=20 that is 128 KB, well past the 48 KB default, so we must opt in
  // via cudaFuncAttributeMaxDynamicSharedMemorySize. If the card cannot supply
  // that much (the opt-in max is ~99 KB on Ampere, ~227 KB on Hopper), fail with
  // a clear message rather than a cryptic launch error.
  const size_t sharedBytes = (size_t)FILTER_WORDS * 4;
  {
    cudaFuncAttributes fa;
    CUDA_CHECK(cudaFuncGetAttributes(&fa, search_kernel));
    cudaError_t se = cudaFuncSetAttribute(
        search_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);
    if (se != cudaSuccess) {
      fprintf(stderr,
        "\nThis GPU cannot provide %zu KB of shared memory per block, which the\n"
        "prefilter needs at FILTER_LOG2_BITS=%d. Rebuild with a smaller FILTER,\n"
        "e.g. `make FILTER=17` (32 KB) or `make FILTER=15` (8 KB).\n",
        sharedBytes / 1024, FILTER_LOG2_BITS);
      return -1;
    }
  }

  uint32_t *d_filter; uint64_t *d_slots; uint32_t *d_idx;
  CUDA_CHECK(cudaMalloc(&d_filter, tt.filter.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_slots, tt.slots.size() * 8));
  CUDA_CHECK(cudaMalloc(&d_idx, tt.idx.size() * 4));
  CUDA_CHECK(cudaMemcpy(d_filter, tt.filter.data(), tt.filter.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_slots, tt.slots.data(), tt.slots.size() * 8, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_idx, tt.idx.data(), tt.idx.size() * 4, cudaMemcpyHostToDevice));

  uint64_t groupBase = 0;
  if (cfg.resume && !cfg.ckptFile.empty()) {
    std::ifstream cf(cfg.ckptFile);
    if (cf) { cf >> groupBase; printf("resuming at group %" PRIu64 "\n", groupBase); }
  }

  std::vector<uint64_t> hx, hy;
  if (!quiet) { printf("computing start points... "); fflush(stdout); }
  compute_start_points(K0, groupBase, keysPerThread, nThreads, hx, hy);
  if (!quiet) printf("done\n");

  uint64_t *d_px, *d_py;
  CUDA_CHECK(cudaMalloc(&d_px, hx.size() * 8));
  CUDA_CHECK(cudaMalloc(&d_py, hy.size() * 8));
  CUDA_CHECK(cudaMemcpy(d_px, hx.data(), hx.size() * 8, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py, hy.data(), hy.size() * 8, cudaMemcpyHostToDevice));

  Result *d_res; uint32_t *d_cnt;
  CUDA_CHECK(cudaMalloc(&d_res, sizeof(Result) * MAX_RESULTS));
  CUDA_CHECK(cudaMalloc(&d_cnt, 4));
  CUDA_CHECK(cudaMemset(d_cnt, 0, 4));

  int found = 0;
  auto t0 = std::chrono::steady_clock::now();
  uint64_t doneKeys = groupBase * GRP_SIZE * nThreads;

  bool istty =
#if defined(_WIN32)
      (_isatty(_fileno(stdout)) != 0);
#else
      (isatty(fileno(stdout)) != 0);
#endif
  Progress meter(totalKeys, 30, istty);

  while (groupBase < groupsPerThread) {
    uint64_t chunk = std::min(cfg.groupsPerLaunch, groupsPerThread - groupBase);
    search_kernel<<<cfg.blocks, cfg.threads, sharedBytes>>>(
        d_px, d_py, d_filter, d_slots, d_idx, tt.mask,
        chunk, groupBase, keysPerThread, d_res, d_cnt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    groupBase += chunk;
    doneKeys += chunk * GRP_SIZE * nThreads;

    uint32_t cnt = 0;
    CUDA_CHECK(cudaMemcpy(&cnt, d_cnt, 4, cudaMemcpyDeviceToHost));
    if (cnt) {
      std::vector<Result> r(std::min(cnt, (uint32_t)MAX_RESULTS));
      CUDA_CHECK(cudaMemcpy(r.data(), d_res, r.size() * sizeof(Result), cudaMemcpyDeviceToHost));
      for (auto &x : r) if (verify_and_report(cfg, targets, byX, K0, x.offset)) found++;
      CUDA_CHECK(cudaMemset(d_cnt, 0, 4));
    }

    if (!cfg.ckptFile.empty()) {
      std::ofstream cf(cfg.ckptFile, std::ios::trunc);
      cf << groupBase << "\n";
    }

    if (!quiet) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      meter.print(doneKeys, el, found);
    }
  }
  if (!quiet) { meter.finish(); printf("search complete, %d match(es)\n", found); }

  cudaFree(d_filter); cudaFree(d_slots); cudaFree(d_idx);
  cudaFree(d_px); cudaFree(d_py); cudaFree(d_res); cudaFree(d_cnt);
  return found;
}

// ------------------------------------------------------------------ selftest
static int run_selftest(Config cfg) {
  printf("[1/2] device field and curve arithmetic\n");
  const int N = 32;
  std::vector<uint64_t> k(N * 4), gx(N * 4), gy(N * 4);
  srand(12345);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < 4; j++)
      k[i * 4 + j] = ((uint64_t)rand() << 48) ^ ((uint64_t)rand() << 32) ^
                     ((uint64_t)rand() << 16) ^ (uint64_t)rand();
  for (int i = 0; i < N; i++) k[i * 4 + 3] &= 0x7FFFFFFFFFFFFFFFULL;

  uint64_t *dk, *dx, *dy;
  CUDA_CHECK(cudaMalloc(&dk, N * 32)); CUDA_CHECK(cudaMalloc(&dx, N * 32));
  CUDA_CHECK(cudaMalloc(&dy, N * 32));
  CUDA_CHECK(cudaMemcpy(dk, k.data(), N * 32, cudaMemcpyHostToDevice));
  selftest_kernel<<<1, N>>>(dk, dx, dy, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(gx.data(), dx, N * 32, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gy.data(), dy, N * 32, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < N; i++) {
    uint64_t hxv[4], hyv[4];
    ec_scalar_mul_g(hxv, hyv, &k[i * 4]);
    if (!fe_eq(hxv, &gx[i * 4]) || !fe_eq(hyv, &gy[i * 4])) {
      printf("  MISMATCH at vector %d\n", i);
      bad++;
    }
  }
  if (bad) { printf("  FAILED (%d/%d)\n", bad, N); return 1; }
  printf("  OK (%d/%d scalar multiplications match the host)\n", N, N);

  printf("[2/2] end-to-end search over a 2^24 range with a planted key\n");
  uint64_t K0[4];
  parse_hex("A1B2C3D4E5F60718293A4B5C6D7E8F90A1B2C3D4E5F60718293A4B5C", K0); // 28 bytes
  shl256(K0, 32);
  int unknownBits = 24;
  // put the planted key well inside the range
  uint64_t planted[4]; fe_set(planted, K0); add64_256(planted, 0xABCDEFULL);
  uint64_t px[4], py[4];
  ec_scalar_mul_g(px, py, planted);

  std::vector<Target> tg;
  Target t; fe_set(t.x, px); t.parity = (uint8_t)(py[0] & 1);
  t.pub = compressed_hex(px, py);
  tg.push_back(t);
  // pad with decoys so the table is exercised
  for (int i = 1; i <= 5000; i++) {
    uint64_t kk[4] = {(uint64_t)(0x1000000 + i), 0, 0, 0}, ax[4], ay[4];
    ec_scalar_mul_g(ax, ay, kk);
    Target d; fe_set(d.x, ax); d.parity = (uint8_t)(ay[0] & 1);
    d.pub = compressed_hex(ax, ay);
    tg.push_back(d);
  }
  printf("  planted key %s\n  expecting   %s\n", hex256(planted).c_str(), t.pub.c_str());

  cfg.blocks = 32; cfg.threads = 64; cfg.ckptFile.clear(); cfg.resume = false;
#if defined(_WIN32)
  cfg.outFile = "NUL";
#else
  cfg.outFile = "/dev/null";
#endif
  int f = run_search(cfg, tg, K0, unknownBits, false);
  if (f < 1) { printf("  FAILED: planted key was not found\n"); return 1; }
  printf("  OK\n\nSELFTEST PASSED\n");
  return 0;
}

// ---------------------------------------------------------------------- main
static void usage() {
  printf(
"keyhunt-gpu -- search a prefixed private-key range for matching public keys\n"
"\n"
"  --prefix HEX        high-order bytes of the private key (even length).\n"
"                      27 bytes / 54 hex chars leaves a 2^40 search space.\n"
"  --targets FILE      one compressed pubkey per line (02/03 + 64 hex),\n"
"                      or uncompressed (04 + 128 hex). '#' starts a comment.\n"
"  --out FILE          append matches here (default found.txt)\n"
"  --device N          CUDA device index (default 0)\n"
"  --blocks N          grid size, power of two (default: 8 x SM count)\n"
"  --threads N         block size, power of two (default 256)\n"
"  --groups N          groups per kernel launch, controls report interval\n"
"  --checkpoint FILE   write progress here after every launch\n"
"  --resume            start from the checkpoint file\n"
"  --notify-cmd CMD    run on a match; %%P -> pubkey, %%K -> private key\n"
"  --selftest          validate device arithmetic and find a planted key\n"
"  --help\n");
}

int main(int argc, char **argv) {
  Config cfg;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--prefix") cfg.prefixHex = next();
    else if (a == "--targets") cfg.targetFile = next();
    else if (a == "--out") cfg.outFile = next();
    else if (a == "--device") cfg.device = atoi(next().c_str());
    else if (a == "--blocks") cfg.blocks = atoi(next().c_str());
    else if (a == "--threads") cfg.threads = atoi(next().c_str());
    else if (a == "--groups") cfg.groupsPerLaunch = strtoull(next().c_str(), 0, 10);
    else if (a == "--checkpoint") cfg.ckptFile = next();
    else if (a == "--resume") cfg.resume = true;
    else if (a == "--notify-cmd") cfg.notifyCmd = next();
    else if (a == "--selftest") cfg.selftest = true;
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else { fprintf(stderr, "unknown option %s\n", argv[i]); usage(); return 1; }
  }

  CUDA_CHECK(cudaSetDevice(cfg.device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  printf("device %d: %s, %d SMs, %.1f GB\n", cfg.device, prop.name,
         prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);

  if (cfg.blocks == 0) {
    int b = prop.multiProcessorCount * 8, p = 1;
    while (p * 2 <= b) p *= 2;
    cfg.blocks = p;
  }
  if ((cfg.blocks & (cfg.blocks - 1)) || (cfg.threads & (cfg.threads - 1))) {
    fprintf(stderr, "--blocks and --threads must be powers of two\n");
    return 1;
  }

  upload_gtable();

  if (cfg.selftest) return run_selftest(cfg);

  if (cfg.prefixHex.empty() || cfg.targetFile.empty()) { usage(); return 1; }
  if (cfg.prefixHex.size() % 2 || cfg.prefixHex.size() >= 64) {
    fprintf(stderr, "--prefix must be an even number of hex chars, under 64\n");
    return 1;
  }
  int unknownBits = 256 - 4 * (int)cfg.prefixHex.size();
  if (unknownBits > 63) { fprintf(stderr, "prefix too short: %d unknown bits\n", unknownBits); return 1; }

  uint64_t K0[4];
  if (!parse_hex(cfg.prefixHex, K0)) { fprintf(stderr, "bad prefix hex\n"); return 1; }
  shl256(K0, unknownBits);

  const uint64_t nn[4] = {N0, N1, N2, N3};
  uint64_t top[4]; fe_set(top, K0); add64_256(top, (1ULL << unknownBits) - 1);
  if (cmp256(top, nn) >= 0)
    fprintf(stderr, "warning: range extends past the group order n; "
                    "keys above n are not valid private keys\n");
  if (K0[3] == 0 && K0[2] == 0 && K0[1] == 0)
    fprintf(stderr, "warning: very small key range, the batch-inversion "
                    "fast path assumes no point doublings occur\n");

  std::vector<Target> targets;
  if (!load_targets(cfg.targetFile, targets)) return 1;
  if (targets.empty()) { fprintf(stderr, "no targets loaded\n"); return 1; }

  uint32_t nThreads = (uint32_t)cfg.blocks * (uint32_t)cfg.threads;
  if (((uint64_t)nThreads * GRP_SIZE) > (1ULL << unknownBits)) {
    fprintf(stderr, "grid too large for a 2^%d range; reduce --blocks/--threads\n", unknownBits);
    return 1;
  }

  run_search(cfg, targets, K0, unknownBits, false);
  return 0;
}
