// Emulate the exact PTX instruction sequence of the device u256_mul on the host,
// with a carry-flag model matching PTX semantics, and check it against a known
// -good schoolbook multiply over many random inputs. If the PTX sequence is
// wrong, this diverges. (This validates the *sequence*; the real hardware still
// needs --selftest, but this catches logic errors I can't otherwise see.)
//
// IMPORTANT: the ptx_mul() below must mirror the asm() in src/secp256k1.h
// instruction-for-instruction. If you edit that asm, edit this to match, or the
// test is meaningless.
#include <cstdio>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>   // __umulh
#endif

// 64x64 -> high 64 bits, portable across MSVC / GCC / Clang.
static inline uint64_t HIp(uint64_t a, uint64_t b) {
#if defined(_MSC_VER)
  return __umulh(a, b);
#else
  return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
#endif
}
static inline uint64_t LOp(uint64_t a, uint64_t b) { return a * b; }  // low 64 wraps

// Sum three 64-bit values, returning the low 64 in *lo and the carry (0/1/2)
// as the return value. Used to model PTX add-with-carry without a 128-bit type:
// each accumulator step adds at most three 64-bit quantities, so the carry out
// never exceeds 2 and always fits in the flag chain the way PTX defines it.
static inline uint64_t add3(uint64_t x, uint64_t y, uint64_t z, uint64_t *lo) {
  uint64_t s = x + y;
  uint64_t c = (s < x);
  uint64_t s2 = s + z;
  c += (s2 < s);
  *lo = s2;
  return c;
}

// PTX carry-flag machine. CF holds the carry out of the previous .cc op.
struct M {
  uint64_t CF = 0;
  // add.cc: d = a+b, set CF
  uint64_t add_cc(uint64_t a, uint64_t b) { uint64_t lo; CF = add3(a, b, 0, &lo); return lo; }
  // addc.cc: d = a+b+CF, set CF
  uint64_t addc_cc(uint64_t a, uint64_t b) { uint64_t lo; CF = add3(a, b, CF, &lo); return lo; }
  // addc: d = a+b+CF (then CF updated)
  uint64_t addc(uint64_t a, uint64_t b) { uint64_t lo; CF = add3(a, b, CF, &lo); return lo; }
  // mad.lo.cc: d = lo(a*b)+c, set CF
  uint64_t madlo_cc(uint64_t a, uint64_t b, uint64_t c) { uint64_t lo; CF = add3(LOp(a, b), c, 0, &lo); return lo; }
  // madc.lo.cc: d = lo(a*b)+c+CF, set CF
  uint64_t madclo_cc(uint64_t a, uint64_t b, uint64_t c) { uint64_t lo; CF = add3(LOp(a, b), c, CF, &lo); return lo; }
  // mad.hi.cc: d = hi(a*b)+c, set CF
  uint64_t madhi_cc(uint64_t a, uint64_t b, uint64_t c) { uint64_t lo; CF = add3(HIp(a, b), c, 0, &lo); return lo; }
  // madc.hi.cc: d = hi(a*b)+c+CF, set CF
  uint64_t madchi_cc(uint64_t a, uint64_t b, uint64_t c) { uint64_t lo; CF = add3(HIp(a, b), c, CF, &lo); return lo; }
  // madc.hi: d = hi(a*b)+c+CF (then CF updated)
  uint64_t madchi(uint64_t a, uint64_t b, uint64_t c) { uint64_t lo; CF = add3(HIp(a, b), c, CF, &lo); return lo; }
};

// Emulate the PTX sequence EXACTLY as written in secp256k1.h
static void ptx_mul(uint64_t out[8], const uint64_t a[4], const uint64_t b[4]){
  uint64_t a0=a[0],a1=a[1],a2=a[2],a3=a[3],b0=b[0],b1=b[1],b2=b[2],b3=b[3];
  uint64_t r0,r1,r2,r3,r4,r5,r6,r7;
  M m;
  // column pass by b0
  r0 = LOp(a0,b0);
  r1 = HIp(a0,b0);
  r1 = m.madlo_cc(a1,b0,r1);
  r2 = m.madchi(a1,b0,0);        // madc.hi.u64 %2, a1,b0,0
  r2 = m.madlo_cc(a2,b0,r2);
  r3 = m.madchi(a2,b0,0);
  r3 = m.madlo_cc(a3,b0,r3);
  r4 = m.madchi(a3,b0,0);
  // column pass by b1
  r1 = m.madlo_cc(a0,b1,r1);
  r2 = m.madchi_cc(a0,b1,r2);
  r3 = m.madchi_cc(a1,b1,r3);
  r4 = m.madchi_cc(a2,b1,r4);
  r5 = m.madchi(a3,b1,0);
  r2 = m.madlo_cc(a1,b1,r2);
  r3 = m.madclo_cc(a2,b1,r3);
  r4 = m.madclo_cc(a3,b1,r4);
  r5 = m.addc(r5,0);
  // column pass by b2
  r2 = m.madlo_cc(a0,b2,r2);
  r3 = m.madchi_cc(a0,b2,r3);
  r4 = m.madchi_cc(a1,b2,r4);
  r5 = m.madchi_cc(a2,b2,r5);
  r6 = m.madchi(a3,b2,0);
  r3 = m.madlo_cc(a1,b2,r3);
  r4 = m.madclo_cc(a2,b2,r4);
  r5 = m.madclo_cc(a3,b2,r5);
  r6 = m.addc(r6,0);
  // column pass by b3
  r3 = m.madlo_cc(a0,b3,r3);
  r4 = m.madchi_cc(a0,b3,r4);
  r5 = m.madchi_cc(a1,b3,r5);
  r6 = m.madchi_cc(a2,b3,r6);
  r7 = m.madchi(a3,b3,0);
  r4 = m.madlo_cc(a1,b3,r4);
  r5 = m.madclo_cc(a2,b3,r5);
  r6 = m.madclo_cc(a3,b3,r6);
  r7 = m.addc(r7,0);
  out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;out[4]=r4;out[5]=r5;out[6]=r6;out[7]=r7;
}

// reference: full 512-bit schoolbook, computed with 64-bit hi/lo pieces and
// explicit carry propagation -- deliberately a different accumulation order
// than ptx_mul above, so agreement is a real cross-check, not a copy.
static void ref_mul(uint64_t out[8], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t res[8] = {0,0,0,0,0,0,0,0};
  for (int i = 0; i < 4; i++) {
    uint64_t carry = 0;
    for (int j = 0; j < 4; j++) {
      uint64_t lo = LOp(a[i], b[j]);
      uint64_t hi = HIp(a[i], b[j]);
      // column = res[i+j] + lo + carry, a value up to ~2^66; keep it as
      // (chi:t0) so no 64-bit add can silently overflow.
      uint64_t t0, chi = add3(res[i + j], lo, carry, &t0);
      res[i + j] = t0;
      // next carry = hi + chi; chi <= 2 and hi <= 2^64-2, but add safely anyway
      uint64_t nc, cc = add3(hi, chi, 0, &nc);
      carry = nc;
      // cc is the carry out of hi+chi (0 or 1); fold it one limb higher
      if (cc) {
        for (int k = i + j + 2; k < 8; k++) {
          uint64_t s = res[k] + 1; res[k] = s;
          if (s != 0) break;   // no further carry
        }
      }
    }
    // add the running carry into the next limb, propagating if needed
    uint64_t k = i + 4;
    uint64_t s = res[k] + carry; 
    uint64_t prop = (s < res[k]) ? 1 : 0;
    res[k] = s;
    for (k = i + 5; prop && k < 8; k++) {
      uint64_t s2 = res[k] + 1; res[k] = s2; prop = (s2 == 0) ? 1 : 0;
    }
  }
  for (int i = 0; i < 8; i++) out[i] = res[i];
}

static uint64_t st=0x123456789ULL;
static uint64_t rnd(){st^=st<<13;st^=st>>7;st^=st<<17;return st;}

int main(){
  int fails=0;
  for(int i=0;i<500000;i++){
    uint64_t a[4]={rnd(),rnd(),rnd(),rnd()},b[4]={rnd(),rnd(),rnd(),rnd()};
    uint64_t o1[8],o2[8]; ptx_mul(o1,a,b); ref_mul(o2,a,b);
    if(memcmp(o1,o2,64)){
      printf("MUL MISMATCH i=%d\n",i);
      for(int k=7;k>=0;k--) printf("%016llx",(unsigned long long)o1[k]);
      printf("  ptx\n");
      for(int k=7;k>=0;k--) printf("%016llx",(unsigned long long)o2[k]);
      printf("  ref\n");
      if(++fails>3) break;
    }
  }
  // edge cases
  uint64_t edges[][4]={{~0ULL,~0ULL,~0ULL,~0ULL},{0,0,0,0},{1,0,0,0},
     {~0ULL,0,0,0},{0,0,0,~0ULL},{~0ULL,~0ULL,0,0}};
  for(auto&a:edges)for(auto&b:edges){
    uint64_t o1[8],o2[8];ptx_mul(o1,a,b);ref_mul(o2,a,b);
    if(memcmp(o1,o2,64)){printf("EDGE MUL MISMATCH\n");fails++;}
  }
  if(!fails)printf("PTX MULTIPLY SEQUENCE OK (500k random + edges)\n");
  else printf("%d FAILURES\n",fails);
  return fails?1:0;
}
