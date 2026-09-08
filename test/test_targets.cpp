// Validates target parsing and the lookup-table indexing. No GPU needed.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../src/targets.h"

static int fails = 0;
static void ok(const char *what, bool cond) {
  if (!cond) { printf("  FAIL %s\n", what); fails++; }
}

int main() {
  // A real key so the vectors are self-consistent.
  uint64_t k[4] = {0xABCDEF, 0, 0, 0}, x[4], y[4];
  ec_scalar_mul_g(x, y, k);
  std::string comp = compressed_hex(x, y);
  std::string uncomp = "04" + hex256(x) + hex256(y);

  const char *path = "/tmp/kh_targets_test.txt";
  FILE *f = fopen(path, "w");
  fprintf(f, "# a comment line\n");
  fprintf(f, "\n");
  fprintf(f, "%s\n", comp.c_str());                      // bare compressed
  fprintf(f, "%s\n", uncomp.c_str());                    // bare uncompressed
  fprintf(f, "21%sac\n", comp.c_str());                  // P2PK scriptPubKey, compressed
  fprintf(f, "21%sAC\n", comp.c_str());                  // uppercase OP_CHECKSIG
  fprintf(f, "41%sac\n", uncomp.c_str());                // P2PK scriptPubKey, uncompressed
  fprintf(f, "deadbeef\n");                              // garbage, must be skipped
  fprintf(f, "02%s\n", std::string(64, 'z').c_str());    // bad hex, must be skipped
  fclose(f);

  std::vector<Target> tg;
  ok("load returned true", load_targets(path, tg));
  printf("parsed %zu targets\n", tg.size());
  ok("five valid targets parsed", tg.size() == 5);

  // Every accepted form must decode to the same X, and to the same parity.
  for (size_t i = 0; i < tg.size(); i++) {
    ok("X matches", fe_eq(tg[i].x, x));
    ok("parity matches", tg[i].parity == (uint8_t)(y[0] & 1));
    ok("canonical pubkey string", tg[i].pub == comp);
  }

  // Lookup table: every target must be findable by the same three index
  // functions the kernel uses, and a non-target must not collide into a hit.
  {
    std::vector<Target> many;
    for (int i = 1; i <= 20000; i++) {
      uint64_t kk[4] = {(uint64_t)(0x500000 + i), 0, 0, 0}, ax[4], ay[4];
      ec_scalar_mul_g(ax, ay, kk);
      Target t; fe_set(t.x, ax); t.parity = (uint8_t)(ay[0] & 1);
      t.pub = compressed_hex(ax, ay);
      many.push_back(t);
    }
    TargetTable tt = build_table(many);

    auto probe = [&](const uint64_t xx[4]) -> bool {
      uint32_t bit = filter_bit(xx);
      if (((tt.filter[bit >> 5] >> (bit & 31)) & 1u) == 0u) return false;
      uint64_t tag = x_tag(xx);
      uint32_t s = table_slot(xx, tt.mask);
      for (;;) {
        if (tt.slots[s] == 0) return false;
        if (tt.slots[s] == tag) return true;
        s = (s + 1) & tt.mask;
      }
    };

    int missed = 0;
    for (auto &t : many) if (!probe(t.x)) missed++;
    ok("all 20000 targets found by the kernel's probe", missed == 0);
    if (missed) printf("  %d missed\n", missed);

    int falsehits = 0, n = 20000;
    for (int i = 1; i <= n; i++) {
      uint64_t kk[4] = {(uint64_t)(0x900000 + i), 0, 0, 0}, ax[4], ay[4];
      ec_scalar_mul_g(ax, ay, kk);
      if (probe(ax)) falsehits++;
    }
    printf("false positives on %d non-targets: %d\n", n, falsehits);
    ok("no false positives", falsehits == 0);

    // Confirm the prefilter is actually doing work.
    int passed = 0;
    for (int i = 1; i <= n; i++) {
      uint64_t kk[4] = {(uint64_t)(0x900000 + i), 0, 0, 0}, ax[4], ay[4];
      ec_scalar_mul_g(ax, ay, kk);
      uint32_t bit = filter_bit(ax);
      if ((tt.filter[bit >> 5] >> (bit & 31)) & 1u) passed++;
    }
    printf("prefilter pass rate: %.1f%% (expect ~%.1f%% for %zu targets)\n",
           100.0 * passed / n, 100.0 * (1.0 - exp(-(double)many.size() / FILTER_BITS)),
           many.size());
  }

  remove(path);
  if (!fails) { printf("\nTARGET PARSING OK\n"); return 0; }
  printf("\n%d FAILURES\n", fails);
  return 1;
}
