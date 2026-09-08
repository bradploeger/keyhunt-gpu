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

typedef unsigned __int128 u128;
static inline uint64_t LO(u128 x){return (uint64_t)x;}
static inline uint64_t HIp(uint64_t a,uint64_t b){return (uint64_t)(((u128)a*b)>>64);}
static inline uint64_t LOp(uint64_t a,uint64_t b){return (uint64_t)((u128)a*b);}

// PTX carry-flag machine. CF is a single bit.
struct M {
  uint64_t CF=0;
  // add.cc: d = a+b, set CF
  uint64_t add_cc(uint64_t a,uint64_t b){ u128 s=(u128)a+b; CF=(uint64_t)(s>>64); return (uint64_t)s; }
  // addc.cc: d = a+b+CF, set CF
  uint64_t addc_cc(uint64_t a,uint64_t b){ u128 s=(u128)a+b+CF; CF=(uint64_t)(s>>64); return (uint64_t)s; }
  // addc: d = a+b+CF, CF unchanged-after (we don't read it again)
  uint64_t addc(uint64_t a,uint64_t b){ u128 s=(u128)a+b+CF; CF=(uint64_t)(s>>64); return (uint64_t)s; }
  // mad.lo.cc: d = lo(a*b)+c, set CF
  uint64_t madlo_cc(uint64_t a,uint64_t b,uint64_t c){ u128 s=(u128)LOp(a,b)+c; CF=(uint64_t)(s>>64); return (uint64_t)s; }
  // madc.lo.cc: d = lo(a*b)+c+CF, set CF
  uint64_t madclo_cc(uint64_t a,uint64_t b,uint64_t c){ u128 s=(u128)LOp(a,b)+c+CF; CF=(uint64_t)(s>>64); return (uint64_t)s; }
  // mad.hi.cc: d = hi(a*b)+c, set CF
  uint64_t madhi_cc(uint64_t a,uint64_t b,uint64_t c){ u128 s=(u128)HIp(a,b)+c; CF=(uint64_t)(s>>64); return (uint64_t)s; }
  // madc.hi.cc: d = hi(a*b)+c+CF, set CF
  uint64_t madchi_cc(uint64_t a,uint64_t b,uint64_t c){ u128 s=(u128)HIp(a,b)+c+CF; CF=(uint64_t)(s>>64); return (uint64_t)s; }
  // madc.hi: d = hi(a*b)+c+CF
  uint64_t madchi(uint64_t a,uint64_t b,uint64_t c){ u128 s=(u128)HIp(a,b)+c+CF; CF=(uint64_t)(s>>64); return (uint64_t)s; }
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

// reference: full 512-bit schoolbook via u128
static void ref_mul(uint64_t out[8], const uint64_t a[4], const uint64_t b[4]){
  u128 acc[8]={0};
  // accumulate into a wide array of 128-bit columns with carry propagation
  uint64_t tmp[8]={0};
  u128 carry=0;
  // simplest correct: bignum multiply
  unsigned __int128 res[9]={0};
  for(int i=0;i<4;i++){
    unsigned __int128 c=0;
    for(int j=0;j<4;j++){
      unsigned __int128 cur = (unsigned __int128)res[i+j] + (unsigned __int128)a[i]*b[j] + c;
      res[i+j] = (uint64_t)cur;
      c = cur>>64;
    }
    res[i+4] += c;
  }
  for(int i=0;i<8;i++) out[i]=(uint64_t)res[i];
}

static uint64_t st=0x123456789ULL;
static uint64_t rnd(){st^=st<<13;st^=st>>7;st^=st<<17;return st;}

int main(){
  int fails=0;
  for(int i=0;i<500000;i++){
    uint64_t a[4]={rnd(),rnd(),rnd(),rnd()},b[4]={rnd(),rnd(),rnd(),rnd()};
    uint64_t o1[8],o2[8]; ptx_mul(o1,a,b); ref_mul(o2,a,b);
    if(memcmp(o1,o2,64)){ printf("MUL MISMATCH i=%d\n",i);
      for(int k=7;k>=0;k--)printf("%016llx",(unsigned long long)o1[k]);printf("  ptx\n");
      for(int k=7;k>=0;k--)printf("%016llx",(unsigned long long)o2[k]);printf("  ref\n");
      if(++fails>3)break;}
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
