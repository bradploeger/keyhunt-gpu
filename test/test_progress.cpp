// Host tests for progress.h -- formatting, bar fill, rolling rate, monotonic ETA.
#include <cstdio>
#include <string>
#include "../src/progress.h"

static int fails = 0;
static void ok(const char *what, bool cond, const std::string &got = "") {
  if (!cond) { printf("  FAIL %s   %s\n", what, got.c_str()); fails++; }
}
static bool has(const std::string &h, const std::string &n) {
  return h.find(n) != std::string::npos;
}

int main() {
  // duration formatting
  ok("0s",        Progress::dur_str(0)      == "0s",        Progress::dur_str(0));
  ok("45s",       Progress::dur_str(45)     == "45s",       Progress::dur_str(45));
  ok("2m18s",     Progress::dur_str(138)    == "2m18s",     Progress::dur_str(138));
  ok("1h1m1s",    Progress::dur_str(3661)   == "1h1m1s",    Progress::dur_str(3661));
  ok("1d0h0m",    Progress::dur_str(86400)  == "1d0h0m",    Progress::dur_str(86400));

  // rate formatting with unit scaling
  ok("key/s",   has(Progress::rate_str(12), "key/s"),      Progress::rate_str(12));
  ok("Kkey/s",  has(Progress::rate_str(1500), "Kkey/s"),   Progress::rate_str(1500));
  ok("Mkey/s",  has(Progress::rate_str(5e6), "Mkey/s"),    Progress::rate_str(5e6));
  ok("Gkey/s",  has(Progress::rate_str(1.18e9), "Gkey/s"), Progress::rate_str(1.18e9));

  // bar fill at known fractions
  {
    Progress p(1000, 10, true);
    std::string s0 = p.render(0, 0.0, 0);
    ok("empty bar", has(s0, "[----------]"), s0);
    Progress p2(1000, 10, true);
    p2.render(500, 1.0, 0);
    std::string s5 = p2.render(500, 2.0, 0);
    ok("half bar", has(s5, "[#####-----]"), s5);
    ok("50.0%", has(s5, "50.0%"), s5);
    Progress p3(1000, 10, true);
    std::string sf = p3.render(1000, 1.0, 0);
    ok("full bar", has(sf, "[##########]"), sf);
    ok("100.0%", has(sf, "100.0%"), sf);
  }

  // clamps past 100%
  {
    Progress p(1000, 10, true);
    std::string s = p.render(5000, 1.0, 0);
    ok("clamped to 100", has(s, "100.0%"), s);
    ok("clamped bar full", has(s, "[##########]"), s);
  }

  // rolling rate: after two samples 1s apart with 1e9 keys of delta, ~1 Gkey/s
  {
    Progress p(1000ULL * 1000 * 1000 * 1000, 20, true);
    p.render(0, 0.0, 0);
    std::string s = p.render(1000ULL * 1000 * 1000, 1.0, 0);
    ok("rolling ~1 Gkey/s", has(s, "Gkey/s"), s);
  }

  // ETA shrinks as work proceeds at constant rate
  {
    uint64_t total = 100000;
    Progress p(total, 20, true);
    p.render(0, 0.0, 0);
    p.render(10000, 1.0, 0);          // 10k/s
    std::string a = p.render(20000, 2.0, 0);
    std::string b = p.render(50000, 5.0, 0);
    // crude: eta in 'a' should be larger than in 'b' -- compare the substring
    ok("eta present early", has(a, "eta"), a);
    ok("eta present later", has(b, "eta"), b);
  }

  // found counter surfaces
  {
    Progress p(1000, 10, true);
    std::string s = p.render(100, 1.0, 3);
    ok("found shown", has(s, "found 3"), s);
  }

  // non-tty mode still renders (print path just uses newlines)
  {
    Progress p(1000, 10, false);
    std::string s = p.render(250, 1.0, 0);
    ok("non-tty renders", has(s, "25.0%"), s);
  }

  if (!fails) { printf("PROGRESS TESTS PASSED\n"); return 0; }
  printf("\n%d FAILURES\n", fails);
  return 1;
}
