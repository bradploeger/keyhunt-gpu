// seal.h -- one-shot "sealed box" for a recovered private key, encrypted to an
// X25519 public key only. Host-only (uses OpenSSL); never compiled as device
// code. The wire format is identical to protocol.py's seal_secret()/open_sealed()
// in the keyhunt-node / keyhunt-coord-server repositories, so a key sealed here
// is opened there with the server's X25519 secret:
//
//   ephemeral X25519 -> ECDH -> HKDF-SHA256(salt=nonce, info) -> ChaCha20-Poly1305
//   info = "KHSEAL1" || epk(32) || recipient_x(32)
//   blob = epk(32) || nonce(12) || ciphertext(plaintext + 16-byte tag)   (hex)
//
// There is no sender identity and no signature: anyone may produce a sealed box,
// but only the holder of the recipient's X25519 secret can open it. That is
// exactly what we want for a found key -- the search node never has to hold,
// print, or transmit the private key in the clear.
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include <openssl/evp.h>
#include <openssl/kdf.h>
#include <openssl/rand.h>
#include <openssl/err.h>

static const char KH_SEAL_MAGIC[] = {'K', 'H', 'S', 'E', 'A', 'L', '1'};  // 7 bytes, no NUL

// ------------------------------------------------------------------ hex utils
static std::string kh_to_hex(const unsigned char *p, size_t n) {
  static const char *d = "0123456789abcdef";
  std::string s;
  s.resize(n * 2);
  for (size_t i = 0; i < n; i++) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 0xF]; }
  return s;
}

static bool kh_from_hex(const std::string &s, std::vector<unsigned char> &out) {
  if (s.size() % 2) return false;
  out.clear();
  out.reserve(s.size() / 2);
  auto nyb = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (size_t i = 0; i < s.size(); i += 2) {
    int hi = nyb(s[i]), lo = nyb(s[i + 1]);
    if (hi < 0 || lo < 0) return false;
    out.push_back((unsigned char)((hi << 4) | lo));
  }
  return true;
}

// Pull the "x25519" 64-hex value out of a server.pub JSON file. Minimal on
// purpose: the file is a flat {"ed25519":"..","x25519":".."} object written by
// keygen.py. Returns false (with a message) if it is missing or malformed.
static bool kh_load_recipient_x(const std::string &path, unsigned char out_x[32],
                                std::string &err) {
  std::ifstream f(path);
  if (!f) { err = "cannot open " + path; return false; }
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  size_t k = s.find("\"x25519\"");
  if (k == std::string::npos) { err = "no x25519 field in " + path; return false; }
  size_t colon = s.find(':', k);
  if (colon == std::string::npos) { err = "malformed x25519 field"; return false; }
  size_t q1 = s.find('"', colon);
  if (q1 == std::string::npos) { err = "malformed x25519 value"; return false; }
  size_t q2 = s.find('"', q1 + 1);
  if (q2 == std::string::npos) { err = "unterminated x25519 value"; return false; }
  std::string hex = s.substr(q1 + 1, q2 - q1 - 1);
  std::vector<unsigned char> raw;
  if (!kh_from_hex(hex, raw) || raw.size() != 32) {
    err = "x25519 value is not 32 bytes of hex";
    return false;
  }
  memcpy(out_x, raw.data(), 32);
  return true;
}

// ------------------------------------------------------------------- internals
static bool kh__hkdf_sha256(const unsigned char *ikm, size_t ikmlen,
                            const unsigned char *salt, size_t saltlen,
                            const unsigned char *info, size_t infolen,
                            unsigned char out[32]) {
  EVP_PKEY_CTX *c = EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF, NULL);
  if (!c) return false;
  bool ok = EVP_PKEY_derive_init(c) > 0 &&
            EVP_PKEY_CTX_set_hkdf_md(c, EVP_sha256()) > 0 &&
            EVP_PKEY_CTX_set1_hkdf_salt(c, salt, (int)saltlen) > 0 &&
            EVP_PKEY_CTX_set1_hkdf_key(c, ikm, (int)ikmlen) > 0 &&
            EVP_PKEY_CTX_add1_hkdf_info(c, info, (int)infolen) > 0;
  size_t outlen = 32;
  ok = ok && EVP_PKEY_derive(c, out, &outlen) > 0 && outlen == 32;
  EVP_PKEY_CTX_free(c);
  return ok;
}

// info = MAGIC || epk || recipient_x   (used as both HKDF info and AEAD AAD)
static std::vector<unsigned char> kh__info(const unsigned char epk[32],
                                           const unsigned char rx[32]) {
  std::vector<unsigned char> v;
  v.insert(v.end(), KH_SEAL_MAGIC, KH_SEAL_MAGIC + sizeof(KH_SEAL_MAGIC));
  v.insert(v.end(), epk, epk + 32);
  v.insert(v.end(), rx, rx + 32);
  return v;
}

// ---------------------------------------------------------------------- seal
// Seal `ptlen` bytes to recipient_x (32-byte X25519 public key). Returns the
// hex blob in out_hex. Thread-unused; called only on the host when a match is
// confirmed.
static bool kh_seal(const unsigned char recipient_x[32],
                    const unsigned char *pt, size_t ptlen,
                    std::string &out_hex, std::string &err) {
  EVP_PKEY *rpub = EVP_PKEY_new_raw_public_key(EVP_PKEY_X25519, NULL, recipient_x, 32);
  if (!rpub) { err = "bad recipient key"; return false; }

  EVP_PKEY *eph = NULL;
  EVP_PKEY_CTX *kc = EVP_PKEY_CTX_new_id(EVP_PKEY_X25519, NULL);
  if (!kc || EVP_PKEY_keygen_init(kc) <= 0 || EVP_PKEY_keygen(kc, &eph) <= 0) {
    err = "ephemeral keygen failed";
    if (kc) EVP_PKEY_CTX_free(kc);
    EVP_PKEY_free(rpub);
    return false;
  }
  EVP_PKEY_CTX_free(kc);

  unsigned char epk[32]; size_t epklen = 32;
  unsigned char shared[32]; size_t sharedlen = 32;
  bool ok = EVP_PKEY_get_raw_public_key(eph, epk, &epklen) > 0 && epklen == 32;

  EVP_PKEY_CTX *dc = EVP_PKEY_CTX_new(eph, NULL);
  ok = ok && dc && EVP_PKEY_derive_init(dc) > 0 &&
       EVP_PKEY_derive_set_peer(dc, rpub) > 0 &&
       EVP_PKEY_derive(dc, shared, &sharedlen) > 0 && sharedlen == 32;
  if (dc) EVP_PKEY_CTX_free(dc);
  EVP_PKEY_free(eph);
  EVP_PKEY_free(rpub);
  if (!ok) { err = "ECDH failed"; return false; }

  unsigned char nonce[12];
  if (RAND_bytes(nonce, 12) != 1) { err = "RNG failed"; return false; }

  std::vector<unsigned char> info = kh__info(epk, recipient_x);
  unsigned char key[32];
  if (!kh__hkdf_sha256(shared, 32, nonce, 12, info.data(), info.size(), key)) {
    err = "HKDF failed";
    return false;
  }

  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx) { err = "cipher ctx failed"; return false; }
  std::vector<unsigned char> ct(ptlen);
  unsigned char tag[16];
  int len = 0;
  ok = EVP_EncryptInit_ex(ctx, EVP_chacha20_poly1305(), NULL, NULL, NULL) > 0 &&
       EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, 12, NULL) > 0 &&
       EVP_EncryptInit_ex(ctx, NULL, NULL, key, nonce) > 0 &&
       EVP_EncryptUpdate(ctx, NULL, &len, info.data(), (int)info.size()) > 0 &&
       EVP_EncryptUpdate(ctx, ct.data(), &len, pt, (int)ptlen) > 0;
  int flen = 0;
  ok = ok && EVP_EncryptFinal_ex(ctx, ct.data() + len, &flen) > 0 &&
       EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, 16, tag) > 0;
  EVP_CIPHER_CTX_free(ctx);
  if (!ok) { err = "AEAD encrypt failed"; return false; }

  std::vector<unsigned char> blob;
  blob.insert(blob.end(), epk, epk + 32);
  blob.insert(blob.end(), nonce, nonce + 12);
  blob.insert(blob.end(), ct.begin(), ct.end());
  blob.insert(blob.end(), tag, tag + 16);
  out_hex = kh_to_hex(blob.data(), blob.size());
  return true;
}

// ---------------------------------------------------------------------- open
// Open a sealed blob with the recipient's 32-byte X25519 secret scalar. Only
// used by the test suite; the search tool never opens, it only seals.
static bool kh_open(const unsigned char recipient_x_secret[32],
                    const std::string &blob_hex,
                    std::vector<unsigned char> &out, std::string &err) {
  std::vector<unsigned char> blob;
  if (!kh_from_hex(blob_hex, blob) || blob.size() < 32 + 12 + 16) {
    err = "malformed sealed blob";
    return false;
  }
  const unsigned char *epk = blob.data();
  const unsigned char *nonce = blob.data() + 32;
  const unsigned char *ct = blob.data() + 44;
  size_t ctlen = blob.size() - 44;            // ciphertext + 16-byte tag
  size_t ptlen = ctlen - 16;

  EVP_PKEY *mine = EVP_PKEY_new_raw_private_key(EVP_PKEY_X25519, NULL,
                                               recipient_x_secret, 32);
  if (!mine) { err = "bad recipient secret"; return false; }
  unsigned char rx[32]; size_t rxlen = 32;
  EVP_PKEY *peer = EVP_PKEY_new_raw_public_key(EVP_PKEY_X25519, NULL, epk, 32);
  unsigned char shared[32]; size_t sharedlen = 32;
  bool ok = EVP_PKEY_get_raw_public_key(mine, rx, &rxlen) > 0 && rxlen == 32 && peer;
  EVP_PKEY_CTX *dc = EVP_PKEY_CTX_new(mine, NULL);
  ok = ok && dc && EVP_PKEY_derive_init(dc) > 0 &&
       EVP_PKEY_derive_set_peer(dc, peer) > 0 &&
       EVP_PKEY_derive(dc, shared, &sharedlen) > 0 && sharedlen == 32;
  if (dc) EVP_PKEY_CTX_free(dc);
  EVP_PKEY_free(peer);
  EVP_PKEY_free(mine);
  if (!ok) { err = "ECDH failed"; return false; }

  std::vector<unsigned char> info = kh__info(epk, rx);
  unsigned char key[32];
  if (!kh__hkdf_sha256(shared, 32, nonce, 12, info.data(), info.size(), key)) {
    err = "HKDF failed";
    return false;
  }

  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (!ctx) { err = "cipher ctx failed"; return false; }
  out.assign(ptlen, 0);
  int len = 0;
  ok = EVP_DecryptInit_ex(ctx, EVP_chacha20_poly1305(), NULL, NULL, NULL) > 0 &&
       EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, 12, NULL) > 0 &&
       EVP_DecryptInit_ex(ctx, NULL, NULL, key, nonce) > 0 &&
       EVP_DecryptUpdate(ctx, NULL, &len, info.data(), (int)info.size()) > 0 &&
       EVP_DecryptUpdate(ctx, out.data(), &len, ct, (int)ptlen) > 0 &&
       EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG, 16,
                           (void *)(ct + ptlen)) > 0;
  int flen = 0;
  ok = ok && EVP_DecryptFinal_ex(ctx, out.data() + len, &flen) > 0;
  EVP_CIPHER_CTX_free(ctx);
  if (!ok) { err = "decryption failed"; out.clear(); return false; }
  return true;
}
