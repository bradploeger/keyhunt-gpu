// test_seal.cpp -- host tests for the sealed-box match-reporting layer.
//
// Validates that a private key sealed by the search tool can only be opened
// with the recipient's X25519 secret, that tampering and wrong keys are
// rejected, and that server.pub parsing works. The wire format is identical to
// protocol.py's seal_secret()/open_sealed() in the node/server repos; interop
// between the two is covered there.
#include "../src/seal.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <openssl/evp.h>

static int FAILS = 0;
static void check(const char *name, bool cond) {
  printf(cond ? "  ok  %s\n" : "  FAIL %s\n", name);
  if (!cond) FAILS++;
}

// Generate an X25519 keypair, returning raw public and secret (32 bytes each).
static bool gen_x25519(unsigned char pub[32], unsigned char sec[32]) {
  EVP_PKEY *k = NULL;
  EVP_PKEY_CTX *c = EVP_PKEY_CTX_new_id(EVP_PKEY_X25519, NULL);
  bool ok = c && EVP_PKEY_keygen_init(c) > 0 && EVP_PKEY_keygen(c, &k) > 0;
  size_t pl = 32, sl = 32;
  ok = ok && EVP_PKEY_get_raw_public_key(k, pub, &pl) > 0 && pl == 32;
  ok = ok && EVP_PKEY_get_raw_private_key(k, sec, &sl) > 0 && sl == 32;
  if (c) EVP_PKEY_CTX_free(c);
  if (k) EVP_PKEY_free(k);
  return ok;
}

int main() {
  std::string err;

  printf("[hex round trip]\n");
  {
    unsigned char in[5] = {0x00, 0x7f, 0x80, 0xab, 0xff};
    std::vector<unsigned char> out;
    check("decode(encode) is identity",
          kh_from_hex(kh_to_hex(in, 5), out) && out.size() == 5 &&
          memcmp(in, out.data(), 5) == 0);
    check("odd-length hex rejected", !kh_from_hex("abc", out));
    check("non-hex rejected", !kh_from_hex("zz", out));
  }

  unsigned char pub[32], sec[32];
  if (!gen_x25519(pub, sec)) { printf("could not generate a keypair\n"); return 1; }

  // A fixed 32-byte "private key" payload, big-endian like hex256 output.
  unsigned char key[32];
  for (int i = 0; i < 32; i++) key[i] = (unsigned char)(i * 7 + 1);

  printf("\n[seal / open round trip]\n");
  std::string blob;
  check("seal succeeds", kh_seal(pub, key, 32, blob, err));
  check("blob is epk(32)+nonce(12)+ct(32)+tag(16) = 92 bytes", blob.size() == 92 * 2);
  {
    std::vector<unsigned char> got;
    check("open succeeds with the right secret", kh_open(sec, blob, got, err));
    check("recovered key matches", got.size() == 32 && memcmp(got.data(), key, 32) == 0);
  }

  printf("\n[two seals of the same key differ (fresh ephemeral + nonce)]\n");
  {
    std::string blob2;
    kh_seal(pub, key, 32, blob2, err);
    check("ciphertexts are not identical", blob2 != blob);
  }

  printf("\n[tampering is rejected]\n");
  {
    std::vector<unsigned char> raw;
    kh_from_hex(blob, raw);
    std::vector<unsigned char> got;
    // flip a bit in the ciphertext region
    raw[80] ^= 0x01;
    check("flipped ciphertext fails the tag", !kh_open(sec, kh_to_hex(raw.data(), raw.size()), got, err));
    // flip a bit in the ephemeral public key
    kh_from_hex(blob, raw);
    raw[0] ^= 0x01;
    check("altered epk fails to open", !kh_open(sec, kh_to_hex(raw.data(), raw.size()), got, err));
    check("truncated blob rejected", !kh_open(sec, "deadbeef", got, err));
  }

  printf("\n[wrong recipient cannot open]\n");
  {
    unsigned char pub2[32], sec2[32];
    gen_x25519(pub2, sec2);
    std::vector<unsigned char> got;
    check("different secret fails", !kh_open(sec2, blob, got, err));
  }

  printf("\n[server.pub parsing]\n");
  {
    const char *path = "/tmp/kh_seal_test_server.pub";
    FILE *f = fopen(path, "w");
    fprintf(f, "{\n  \"ed25519\": \"%s\",\n  \"x25519\": \"%s\"\n}\n",
            "57213694ea6adaf39f41f2b86afe6610c3763973e24424601ea4dfbfd496496c",
            "2c684a4190113faf5bd6ea895f1df46a63e91fd8148b85a8d16fbef361fc4628");
    fclose(f);
    unsigned char x[32];
    check("parses x25519 from a server.pub", kh_load_recipient_x(path, x, err));
    check("parsed value is correct",
          kh_to_hex(x, 32) == "2c684a4190113faf5bd6ea895f1df46a63e91fd8148b85a8d16fbef361fc4628");
    check("missing file reported", !kh_load_recipient_x("/no/such/server.pub", x, err));

    const char *bad = "/tmp/kh_seal_test_bad.pub";
    f = fopen(bad, "w");
    fprintf(f, "{\"ed25519\":\"ab\"}\n");   // no x25519 field
    fclose(f);
    check("file without x25519 reported", !kh_load_recipient_x(bad, x, err));
  }

  printf("\n%s\n", FAILS ? "SEAL TESTS FAILED" : "ALL SEAL TESTS PASSED");
  return FAILS ? 1 : 0;
}
