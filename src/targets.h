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

struct Target { uint64_t x[4]; uint8_t parity; std::string pub; };

struct Config {
  std::string prefixHex, targetFile, outFile = "found.txt", notifyCmd, ckptFile;
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

static TargetTable build_table(const std::vector<Target> &tg) {
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

