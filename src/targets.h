// targets.h -- host-side target-set handling. No CUDA, no hashing.
//
// Targets are raw compressed public keys (P2PK). Matching is on the X
// coordinate directly; there is no SHA-256, RIPEMD-160, or address encoding
// anywhere in this program.
#pragma once
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include "secp256k1.h"

#ifndef FILTER_LOG2_BITS
#define FILTER_LOG2_BITS 18         // shared-memory prefilter: 2^18 bits = 32 KB / block
#endif
#define FILTER_BITS  (1u << FILTER_LOG2_BITS)
#define FILTER_WORDS (FILTER_BITS / 32)

// -------------------------------------------- pubkey lookup-table conventions
// These index the target set by the public key's X coordinate. They are NOT
// cryptographic hashes and have nothing to do with Bitcoin addresses -- there
// is no SHA-256 / RIPEMD-160 anywhere in this program. Matching is done on the
// compressed public key directly, as P2PK recovery requires.
// Host and device must agree exactly on these three functions.
FD uint32_t filter_bit(const uint64_t x[4]) {
  return (uint32_t)x[1] & (FILTER_BITS - 1);
}
FD uint32_t table_slot(const uint64_t x[4], uint32_t mask) {
  return (uint32_t)(x[1] >> 32) & mask;
}
FD uint64_t x_tag(const uint64_t x[4]) {
  return x[0] | 1ULL;               // 0 is reserved as the "empty slot" marker
}


static const char *HEXD = "0123456789abcdef";

static std::string hex256(const uint64_t v[4]) {
  char b[65];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 16; j++)
      b[i * 16 + j] = HEXD[(v[3 - i] >> (60 - 4 * j)) & 0xF];
  b[64] = 0;
  return std::string(b);
}

static std::string compressed_hex(const uint64_t x[4], const uint64_t y[4]) {
  return std::string((y[0] & 1ULL) ? "03" : "02") + hex256(x);
}

static bool parse_hex(const std::string &s, uint64_t out[4]) {
  if (s.empty() || s.size() > 64) return false;
  out[0] = out[1] = out[2] = out[3] = 0;
  int bit = 0;
  for (int i = (int)s.size() - 1; i >= 0; i--) {
    char c = s[i];
    int v;
    if (c >= '0' && c <= '9') v = c - '0';
    else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
    else return false;
    out[bit >> 6] |= ((uint64_t)v) << (bit & 63);
    bit += 4;
  }
  return true;
}

static void shl256(uint64_t v[4], int n) {
  while (n >= 64) { v[3] = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = 0; n -= 64; }
  if (n == 0) return;
  v[3] = (v[3] << n) | (v[2] >> (64 - n));
  v[2] = (v[2] << n) | (v[1] >> (64 - n));
  v[1] = (v[1] << n) | (v[0] >> (64 - n));
  v[0] = v[0] << n;
}

static void add64_256(uint64_t v[4], uint64_t a) {
  uint64_t c = a;
  for (int i = 0; i < 4 && c; i++) { uint64_t o = v[i]; v[i] += c; c = (v[i] < o) ? 1 : 0; }
}

static int cmp256(const uint64_t a[4], const uint64_t b[4]) {
  for (int i = 3; i >= 0; i--) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  return 0;
}

// n - k, used when a match is found with the opposite Y parity
static void order_minus(uint64_t r[4], const uint64_t k[4]) {
  const uint64_t nn[4] = {N0, N1, N2, N3};
  uint64_t brw = 0;
  for (int i = 0; i < 4; i++) r[i] = subb64(nn[i], k[i], brw, &brw);
}

// ------------------------------------------------- GLV endomorphism (host only)
// secp256k1 has an efficiently computable endomorphism: constants beta (mod p)
// and lambda (mod n), with beta^3 = 1 mod p, lambda^3 = 1 mod n, such that for
// any point P = (x, y):   lambda * P = (beta * x, y).
//
// So the three points T, lambda*T, lambda^2*T share the y-coordinate (hence the
// compressed parity) and have x-coordinates x, beta*x, beta^2*x, with private
// keys t, lambda*t, lambda^2*t. By also storing beta*x and beta^2*x of every
// target in the lookup table, each generated point's single X implicitly tests
// all three -- up to 3x the targets covered per key computed, with no change to
// the device kernel. On a hit against image j, the matched point P = k*G equals
// +/- lambda^j * T, so the target's key is t = lambda^(-j) * k mod n (and the
// host re-derives T from t to confirm before reporting, so a wrong constant or
// a tag collision can never produce a false result).
//
// These helpers run on the host only -- at table-build time and, for the scalar
// reduction mod n, once per confirmed match -- so they favour clarity over
// speed. The field multiply (fe_mul, mod p) comes from secp256k1.h.

// Reduce (carry:r), known to be < 2n, to r mod n in place.
static void glv__reduce_once_n(uint64_t r[4], uint64_t carry) {
  const uint64_t nn[4] = {N0, N1, N2, N3};
  if (carry || cmp256(r, nn) >= 0) {
    uint64_t t[4];
    u256_sub(t, r, nn);   // when carry==1, r<n is guaranteed; the borrow wrap
    fe_set(r, t);         // makes this the correct (2^256 + r - n) low word
  }
}

static void glv__modadd_n(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t c = u256_add(r, a, b);
  glv__reduce_once_n(r, c);
}

// r = a * b mod n, binary (LSB-first) double-and-add. Inputs need not be reduced.
static void mulmod_n(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  const uint64_t nn[4] = {N0, N1, N2, N3};
  uint64_t base[4];
  fe_set(base, a);
  if (cmp256(base, nn) >= 0) { uint64_t t[4]; u256_sub(t, base, nn); fe_set(base, t); }
  uint64_t acc[4] = {0, 0, 0, 0};
  for (int i = 0; i < 256; i++) {
    if ((b[i >> 6] >> (i & 63)) & 1ULL) glv__modadd_n(acc, acc, base);
    glv__modadd_n(base, base, base);
  }
  fe_set(r, acc);
}

struct Glv {
  uint64_t beta[4], beta2[4];   // beta, beta^2  (mod p)
  uint64_t mult[3][4];          // key-recovery multiplier for image j: lambda^(-j) mod n
};

// beta and lambda are the standard secp256k1 endomorphism constants. Parsed from
// hex rather than hand-transcribed into limbs, and cross-checked at startup and
// in test_glv against the curve itself (lambda*G has x == beta*x_G).
static Glv glv_init() {
  Glv g;
  uint64_t lambda[4], lambda2[4];
  parse_hex("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee", g.beta);
  parse_hex("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72", lambda);
  fe_mul(g.beta2, g.beta, g.beta);      // beta^2 mod p
  mulmod_n(lambda2, lambda, lambda);    // lambda^2 mod n  (= lambda^-1)
  // image 0 -> *1 ; image 1 -> *lambda^-1 = lambda^2 ; image 2 -> *lambda^-2 = lambda
  g.mult[0][0] = 1; g.mult[0][1] = g.mult[0][2] = g.mult[0][3] = 0;
  fe_set(g.mult[1], lambda2);
  fe_set(g.mult[2], lambda);
  return g;
}

struct Target { uint64_t x[4]; uint8_t parity; std::string pub; };

struct Config {
  std::string prefixHex, targetFile, outFile = "found.txt", notifyCmd, ckptFile;
  std::string serverPubFile = "server.pub";
  unsigned char recipientX[32] = {0};   // server X25519 public key, loaded at startup
  bool glv = true;                      // search endomorphism images too (3x coverage)
  int device = 0, blocks = 0, threads = 256;
  uint64_t groupsPerLaunch = 256;
  bool selftest = false, resume = false;
};

// ------------------------------------------------------------- table building
struct TargetTable {
  std::vector<uint32_t> filter;
  std::vector<uint64_t> slots;
  std::vector<uint32_t> idx;
  uint32_t mask;
};

template <class Row>
static TargetTable build_table(const std::vector<Row> &tg) {
  TargetTable t;
  t.filter.assign(FILTER_WORDS, 0);
  uint32_t n = 1;
  while (n < tg.size() * 4) n <<= 1;
  if (n < 1024) n = 1024;
  t.mask = n - 1;
  t.slots.assign(n, 0);
  t.idx.assign(n, 0);

  for (size_t i = 0; i < tg.size(); i++) {
    uint32_t bit = filter_bit(tg[i].x);
    t.filter[bit >> 5] |= 1u << (bit & 31);
    uint64_t tag = x_tag(tg[i].x);
    uint32_t s = table_slot(tg[i].x, t.mask);
    while (t.slots[s] != 0 && t.slots[s] != tag) s = (s + 1) & t.mask;
    t.slots[s] = tag;
    t.idx[s] = (uint32_t)i;
  }
  return t;
}

// A single device-matchable entry: one x-coordinate the kernel looks for. With
// GLV each target expands to three (the target and its two endomorphism images,
// jimg = 0,1,2); without it, one. parity is the target's compressed parity (the
// endomorphism leaves y, hence parity, unchanged); tgt indexes the real target.
struct MatchEntry { uint64_t x[4]; uint8_t parity; uint8_t jimg; uint32_t tgt; };

static std::vector<MatchEntry> build_entries(const std::vector<Target> &tg,
                                             const Glv &g, bool glv_on) {
  std::vector<MatchEntry> e;
  e.reserve(tg.size() * (glv_on ? 3 : 1));
  for (uint32_t i = 0; i < tg.size(); i++) {
    MatchEntry m; m.parity = tg[i].parity; m.tgt = i;
    m.jimg = 0; fe_set(m.x, tg[i].x);            e.push_back(m);
    if (glv_on) {
      m.jimg = 1; fe_mul(m.x, tg[i].x, g.beta);  e.push_back(m);
      m.jimg = 2; fe_mul(m.x, tg[i].x, g.beta2); e.push_back(m);
    }
  }
  return e;
}

static bool load_targets(const std::string &path, std::vector<Target> &out) {
  std::ifstream f(path);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
  std::string line;
  size_t lineno = 0, bad = 0;
  while (std::getline(f, line)) {
    lineno++;
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    // Accept a raw P2PK scriptPubKey and unwrap it to the bare pubkey:
    //   21 <33-byte compressed> ac   ->  compressed key
    //   41 <65-byte uncompressed> ac ->  uncompressed key
    // (0x21/0x41 = push length, 0xac = OP_CHECKSIG)
    auto ends_ac = [](const std::string &s) {
      return s.size() >= 2 && (s[s.size()-2]=='a'||s[s.size()-2]=='A')
                           && (s[s.size()-1]=='c'||s[s.size()-1]=='C');
    };
    if (line.size() == 70 && line.compare(0, 2, "21") == 0 && ends_ac(line))
      line = line.substr(2, 66);
    else if (line.size() == 134 && line.compare(0, 2, "41") == 0 && ends_ac(line))
      line = line.substr(2, 130);

    Target t;
    if (line.size() == 66 && (line.compare(0, 2, "02") == 0 || line.compare(0, 2, "03") == 0)) {
      if (!parse_hex(line.substr(2), t.x)) { bad++; continue; }
      t.parity = (uint8_t)(line[1] == '3' ? 1 : 0);
    } else if (line.size() == 130 && line.compare(0, 2, "04") == 0) {
      uint64_t y[4];
      if (!parse_hex(line.substr(2, 64), t.x) || !parse_hex(line.substr(66), y)) { bad++; continue; }
      t.parity = (uint8_t)(y[0] & 1);
    } else { bad++; continue; }
    t.pub = std::string(t.parity ? "03" : "02") + hex256(t.x);
    out.push_back(t);
  }
  if (bad) fprintf(stderr, "warning: skipped %zu unparseable line(s)\n", bad);
  return true;
}

