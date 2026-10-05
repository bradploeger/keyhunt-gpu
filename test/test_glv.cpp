// test_glv.cpp -- host validation of the GLV endomorphism used to expand the
// target table (3x coverage) and to recover a key from an image match.
//
// No GPU and no OpenSSL needed. Checks the beta/lambda constants against the
// curve itself, the cube-root identities, and -- the important part -- that the
// exact recovery the search performs (t = lambda^(-j) * k mod n, with the
// reflection resolved by re-deriving the pubkey) reproduces every target from a
// match on any of its three images, in both Y parities.
#include "../src/targets.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int FAILS = 0;
static void check(const char *name, bool cond) {
  printf(cond ? "  ok  %s\n" : "  FAIL %s\n", name);
  if (!cond) FAILS++;
}

// A pseudo-random scalar comfortably below n (top nibble cleared -> < 2^252).
static void rand_scalar(uint64_t k[4], unsigned seed) {
  srand(seed);
  for (int i = 0; i < 4; i++)
    k[i] = ((uint64_t)rand() << 48) ^ ((uint64_t)rand() << 32) ^
           ((uint64_t)rand() << 16) ^ (uint64_t)rand();
  k[3] &= 0x0FFFFFFFFFFFFFFFULL;
  if (k[0] == 0 && k[1] == 0 && k[2] == 0 && k[3] == 0) k[0] = 1;
}

int main() {
  Glv g = glv_init();
  const uint64_t *lambda  = g.mult[2];   // glv_init stores lambda in mult[2]
  const uint64_t *lambda2 = g.mult[1];   // and lambda^2 (= lambda^-1) in mult[1]

  printf("[cube-root identities]\n");
  {
    uint64_t t[4];
    fe_mul(t, g.beta2, g.beta);                 // beta^3 mod p
    check("beta^3 == 1 (mod p)", fe_eq(t, (const uint64_t[]){1, 0, 0, 0}));
    mulmod_n(t, lambda2, lambda);               // lambda^3 mod n
    check("lambda^3 == 1 (mod n)", fe_eq(t, (const uint64_t[]){1, 0, 0, 0}));
    mulmod_n(t, lambda, lambda);                // lambda^2 recomputed
    check("lambda^2 is consistent", fe_eq(t, lambda2));
  }

  printf("\n[constants agree with the curve: lambda*G == (beta*x_G, y_G)]\n");
  {
    uint64_t gx[4], gy[4];
    ec_generator(gx, gy);
    uint64_t lx[4], ly[4];
    ec_scalar_mul_g(lx, ly, lambda);            // lambda * G
    uint64_t bx[4];
    fe_mul(bx, gx, g.beta);                      // beta * x_G
    check("x(lambda*G) == beta * x_G", fe_eq(lx, bx));
    check("y(lambda*G) == y_G", fe_eq(ly, gy));
  }

  printf("\n[build_entries expands each target to its 3 images]\n");
  {
    uint64_t t[4]; rand_scalar(t, 1);
    uint64_t tx[4], ty[4]; ec_scalar_mul_g(tx, ty, t);
    Target tg; fe_set(tg.x, tx); tg.parity = (uint8_t)(ty[0] & 1);
    tg.pub = compressed_hex(tx, ty);
    std::vector<Target> one(1, tg);

    std::vector<MatchEntry> e = build_entries(one, g, true);
    check("3 entries for 1 target", e.size() == 3);
    uint64_t b1[4], b2[4];
    fe_mul(b1, tx, g.beta);
    fe_mul(b2, tx, g.beta2);
    check("image 0 is the target x", e.size() == 3 && fe_eq(e[0].x, tx) && e[0].jimg == 0);
    check("image 1 is beta*x",       e.size() == 3 && fe_eq(e[1].x, b1) && e[1].jimg == 1);
    check("image 2 is beta^2*x",     e.size() == 3 && fe_eq(e[2].x, b2) && e[2].jimg == 2);

    std::vector<MatchEntry> off = build_entries(one, g, false);
    check("without GLV, 1 entry", off.size() == 1 && off[0].jimg == 0);
  }

  printf("\n[end-to-end recovery from every image, both parities]\n");
  {
    // forward scalar that lands on image j of target t is lambda^j * t
    const uint64_t one_s[4] = {1, 0, 0, 0};
    const uint64_t *fwd[3] = {one_s, lambda, lambda2};
    int cases = 0;
    for (unsigned s = 1; s <= 6; s++) {
      uint64_t t[4]; rand_scalar(t, s);
      uint64_t tx[4], ty[4]; ec_scalar_mul_g(tx, ty, t);
      std::string pub = compressed_hex(tx, ty);

      for (int j = 0; j < 3; j++) {
        for (int refl = 0; refl < 2; refl++) {
          // k = scalar of the generated point matching image j (reflected = n-k)
          uint64_t k[4]; mulmod_n(k, fwd[j], t);
          if (refl) { uint64_t nk[4]; order_minus(nk, k); fe_set(k, nk); }

          // replicate verify_and_report's recovery exactly
          uint64_t key[4]; mulmod_n(key, g.mult[j], k);
          uint64_t vx[4], vy[4]; ec_scalar_mul_g(vx, vy, key);
          if (compressed_hex(vx, vy) != pub) {
            uint64_t nk[4]; order_minus(nk, key); fe_set(key, nk);
            ec_scalar_mul_g(vx, vy, key);
          }
          bool ok = (compressed_hex(vx, vy) == pub);
          if (!ok) { printf("    recovery failed: seed %u image %d refl %d\n", s, j, refl); FAILS++; }
          cases++;
        }
      }
    }
    check("all image/parity recoveries reproduce the target", cases == 36 && FAILS == 0);
  }

  printf("\n%s\n", FAILS ? "GLV TESTS FAILED" : "ALL GLV TESTS PASSED");
  return FAILS ? 1 : 0;
}
