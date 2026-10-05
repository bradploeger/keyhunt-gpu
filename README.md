# keyhunt-gpu

Searches every private key sharing a fixed high-order prefix, derives the
compressed public key for each, and reports any that appear in a target list.

A 27-byte (216-bit) prefix leaves 40 unknown bits — **1,099,511,627,776 keys**.
On a modern GPU that is a run of roughly 5 minutes to an hour.

On a match it prints the 33-byte compressed public key but **never the private
key**: the private key is sealed to the coordination server's X25519 key (a
one-shot sealed box) and only that ciphertext is printed, appended to the out
file, and passed to an optional notify command. Only the holder of the server's
X25519 secret can open it, so the search node never holds, prints, or transmits
a recovered key in the clear. A `server.pub` (the same file the node and server
use) must be present at startup.

To run this across many machines, see the companion repositories
[`keyhunt-coord-server`](#) (hands out random 216-bit prefix blocks and collects
verified results) and [`keyhunt-node`](#) (registers a GPU node with the server
and drives this binary). This repo is the search engine itself and runs
standalone; the other two are optional coordination on top.

---

## Status

The arithmetic and the search logic are covered by tests that run without a GPU:

```
make test
```

That checks the field arithmetic, `k*G`, batch inversion, and the affine
addition path against reference vectors generated independently in Python, and
then replays the exact loop structure of the CUDA kernel on the CPU to confirm
every offset in a range is visited exactly once with the correct X coordinate.
It also cross-checks the dedicated squaring against multiply-based squaring,
parses every accepted target format, probes 20,000 targets through the same
index functions the kernel uses, and re-implements the device PTX multiply
sequence under a carry-flag model to check it against an independent bignum
multiply. All of it passes.

The CUDA path itself cannot be compiled or run without a GPU. Its hot arithmetic
uses hand-written PTX carry chains and a dedicated squaring; the host tests above
validate the *logic* of both (the squaring directly, the PTX multiply via an
instruction-for-instruction emulation), but PTX on real silicon can still differ
from the model. **Run `./keyhunt-gpu --selftest` on the target GPU first.** It
validates device arithmetic against the host and then plants a known key in a
small range and confirms the search finds it. If that passes, the pipeline is
sound end to end.

---

## Build

Needs CUDA 11 or newer, and OpenSSL (libcrypto) for sealing found keys.

On Debian/Ubuntu: `sudo apt-get install libssl-dev`. On Windows, install OpenSSL
(e.g. via vcpkg: `vcpkg install openssl`) so the linker can find `libcrypto`.

### Linux

```
make                  # autodetects your GPU architecture
make ARCH=sm_86       # or name it explicitly (sm_75 Turing, sm_86 Ampere, sm_89 Ada, sm_90 Hopper)
make test
```

### Windows 11

Install Visual Studio 2022 with the "Desktop development with C++" workload
**before** the CUDA Toolkit — the CUDA installer wires itself into whatever
Visual Studio it finds, and installing in the other order means nvcc cannot
locate a host compiler.

Open the **x64 Native Tools Command Prompt for VS 2022** (not plain cmd — it
sets up cl.exe) and run:

You need to know your cards compute capacity before compiling, find it with:

```
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

The pass your compute capability without the dot to the build.bat file.\

```
build.bat 86
```

The script builds `keyhunt-gpu.exe`, builds the host tests, and runs them. Or use CMake, 
which works on both platforms: 

```
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
ctest --test-dir build
```

Tunables, all optional:

```
make HSIZE=256        # keys per batch inversion = 2*HSIZE (default 512)
make FILTER=17        # log2 of the prefilter bits (default 19 = 64 KB shared)
```

The default `FILTER=19` prefilter needs 64 KB of shared memory per block, past
the 48 KB default, so the tool opts into it at launch (`cudaFuncAttributeMax`
`DynamicSharedMemorySize`), which Ampere and newer support comfortably (their
opt-in ceiling is ~99 KB). If the card cannot supply that much it fails with a
clear message telling you to rebuild with a smaller `FILTER` (e.g.
`make FILTER=17` for 16 KB). Larger `HSIZE` and `FILTER` trade
more shared/local memory for fewer inversions and a tighter prefilter; measure
occupancy with `-Xptxas -v` after changing them.

## Usage

```
./keyhunt-gpu --prefix <54 hex chars> --targets targets.txt
```

Full options:

```
--prefix HEX        high-order bytes of the private key, even length.
                    54 hex chars (27 bytes) leaves a 2^40 space.
--targets FILE      one pubkey per line. Accepts bare compressed
                    (02/03 + 64 hex), bare uncompressed (04 + 128 hex), or a
                    raw P2PK scriptPubKey (21...ac / 41...ac), which is
                    unwrapped automatically. '#' starts a comment.
--out FILE          append sealed matches here (default found.txt)
--server-info FILE  server public-key json (default server.pub); its
                    x25519 key seals any found private key
--device N          CUDA device index
--blocks N          grid size, power of two (default 8 x SM count)
--threads N         block size, power of two (default 256)
--groups N          groups per launch; controls progress/checkpoint interval
--checkpoint FILE   save progress after every launch
--resume            start from the checkpoint
--notify-cmd CMD    run on a match. %P becomes the pubkey, %K the sealed blob
--no-glv            disable the endomorphism expansion (search 1x, not 3x)
--selftest          validate the device and find a planted key
```

### Progress display

While searching, a live meter updates in place on one line:

```
[###################-----------]  62.5%    1.18 Gkey/s  9m42s / eta 5m49s  found 1
```

The bar fills left to right; then percent complete, the current search rate
(rolling average over the last few seconds, so it tracks real speed rather than
being dragged down by startup), elapsed time, an ETA that counts down, and the
running match count. Rate auto-scales through key/s, Kkey/s, Mkey/s, Gkey/s.

When stdout is not a terminal (redirected to a file or piped), the meter prints
one line per update with newlines instead of overwriting, so logs stay clean.
`--selftest` and normal runs both show it; there is nothing to enable.

Example:

```
./keyhunt-gpu \
  --prefix A1B2C3D4E5F60718293A4B5C6D7E8F90A1B2C3D4E5F60718293A4B5C \
  --targets targets.txt \
  --checkpoint state.txt --resume \
  --notify-cmd 'curl -d "found %K" ntfy.sh/my-topic'
```

Output on a hit:

```
========================================================================
  MATCH FOUND
  compressed public key : 02aedf62f361689b22568ad68a55899733563c88...
  private key           : sealed to server.pub, not shown
  sealed (open server-side with the X25519 secret):
  7b3b...<184 hex chars>...e1
========================================================================
```

The `sealed=` blob (epk ‖ nonce ‖ ChaCha20-Poly1305 ciphertext, hex) is what
lands in the out file and is handed to the node, which forwards it to the
coordination server. The server opens it with its X25519 secret and re-verifies
the key before trusting it. The format matches `protocol.py`'s
`seal_secret()` / `open_sealed()` in the node and server repos.

To build a test target list, `tools/make_test_targets.py` generates decoys and
can plant a real key at a chosen offset:

```
python3 tools/make_test_targets.py --count 35000 \
    --plant A1B2C3D4E5F60718293A4B5C6D7E8F90A1B2C3D4E5F60718293A4B5C:0x8000000000 \
    > targets.txt
```

---

## How the speed is achieved

The naive approach — one scalar multiplication per candidate — costs about 256
point doublings and additions, so roughly 1,500 field multiplications per key.
This does about **4.5**, a 300x reduction, using four ideas:

**1. Walk the range instead of multiplying.** Consecutive private keys differ by
one, so their public keys differ by `G`. The search never repeats a scalar
multiplication; it steps a point along the range.

**2. Batch the modular inversions.** Affine point addition needs one modular
inverse, which is by far the most expensive field operation. Montgomery's trick
converts *n* inversions into one inversion plus about 3n multiplications. Each
thread holds a group centre `P` and computes `P ± iG` for `i = 1..HSIZE` from
precomputed multiples of `G`, so all the denominators are independent and invert
together. Default `HSIZE=256` gives 512 keys per inversion.

**3. Both signs share one inverse.** `P + iG` and `P − iG` have the same
denominator `(x_P − x_iG)`, so one inverse produces two keys. Better still, the
new X depends only on the *square* of the slope, so the sign flip is free.

**4. Never compute Y.** A compressed key is a parity byte plus X, and matching
happens on X alone. Y is only needed to advance the group centre, once per 256
keys. That removes a multiplication per key.

The inversion uses the standard 255-squaring / 15-multiplication addition chain
for `a^(p-2)`, amortised to about one multiplication per key.

Matching is two-stage, because 1.1e12 lookups into a table will otherwise become
the bottleneck. First a 64 KB bitmap held in **shared memory**, indexed by 19
bits of X — with 35,000 targets this rejects about 94% of candidates at
register speed. Survivors probe an open-addressed hash table in global memory
holding 64 bits of X taken verbatim. None of this is hashing — the three index
functions are literal bit-slices of the X coordinate (`x[1]` low bits, `x[1]`
high bits, `x[0]`). Every reported hit is recomputed from scratch on the host
before it is announced, so a tag collision can never produce a false report.

Because matching is on X only, each generated point implicitly tests both `k`
and `n − k`. If a target matches with the opposite Y parity, the tool reports
`n − k` as the private key and says so.

### What it costs

| stage | field multiplies per key |
|---|---|
| batch inversion (forward + backward) | ~3.0 |
| the point addition itself | ~2.0 (per two keys, shared inverse) |
| the amortised inversion | ~1.0 |
| **total** | **~4.5** |

Expect somewhere between 300 Mkey/s and 1.5 Gkey/s depending on the card. At
500 Mkey/s a 2^40 range takes about 37 minutes; at 1.2 Gkey/s, about 15.

### Endomorphism (GLV) coverage — on by default

secp256k1 has an efficiently computable endomorphism: constants `beta` (mod p)
and `lambda` (mod n) with `beta^3 = 1`, `lambda^3 = 1`, such that for any point
`lambda*P = (beta*x, y)`. So a target `T = t*G` and its two images `lambda*T`,
`lambda^2*T` share a y-coordinate and have x-coordinates `x`, `beta*x`,
`beta^2*x`, with private keys `t`, `lambda*t`, `lambda^2*t`.

The tool stores all three x-coordinates of every target in the lookup table, so
**each point the kernel generates implicitly tests three targets at once** — up
to a 3x gain in targets covered per key computed, for free in the inner loop:
the kernel is byte-for-byte unchanged, matching is still a single X probe, and
the only added work is building a 3x-larger table and, on a hit, one modular
multiply and an independent re-derivation of the key.

On a hit against image *j*, the generated point `k*G` equals `±lambda^j * T`, so
the recovered key is `t = lambda^(-j) * k mod n`. The host re-derives the target
public key from `t` (trying the `n - t` reflection) and reports **nothing**
unless it reproduces the target exactly — so a bad constant or a tag collision
can never yield a false hit. The recovered key is the matched target's genuine
private key; note that for an image hit (`j != 0`) it does **not** lie in the
searched prefix range (it is a `lambda`-multiple of the offset), which is why
this helps only when the goal is "recover any matched pubkey's key," the same
sense in which the tool already reports `n - k`. Pass `--no-glv` to turn the
expansion off and match targets by their own x only.

Because the table now holds 3x as many x-coordinates, the shared-memory
prefilter fills up 3x faster, so its reject rate drops at a given `FILTER`. If
the profiler shows matching (not arithmetic) becoming the bottleneck with GLV
on, raise `FILTER` by one or two (e.g. `make FILTER=20`) to restore the reject
rate — subject to the shared-memory budget noted above.

---

## Tuning

Build with `-Xptxas -v` (already on) and watch the register count. If you see
spills, that is the first thing to fix.

- **`HSIZE`** — larger means fewer inversions but more local memory per thread
  (`(HSIZE+1) * 32` bytes). 128 and 256 are both reasonable; measure.
- **`--blocks` / `--threads`** — start at 8x SM count and 256. More blocks helps
  until local memory pressure bites.
- **`FILTER`** — the default 19 gives a 64 KB bitmap rejecting ~94% of
  candidates (at 35k targets). Dropping to 17 (16 KB) or 15 (4 KB) frees shared
  memory for more resident blocks per SM at a lower reject rate (~77% / ~34%);
  worth trying if the profiler shows you are occupancy-limited rather than
  ALU-bound. Going the other way (20 = 128 KB bitmap) reaches ~97% but needs a
  Hopper-class shared-memory budget.
- **`--groups`** — larger means less host round-tripping, but a coarser progress
  bar and checkpoint interval.

### Device arithmetic: PTX carry chains and dedicated squaring

Both of these are now implemented in `secp256k1.h`.

**PTX carry chains.** The device paths of `u256_add`, `u256_sub`, and the
256×256 `u256_mul` are hand-written `add.cc`/`addc.cc` and `mad.lo.cc`/
`madc.hi.cc` chains, each kept inside a single `asm` block so the hardware carry
flag is never dropped between instructions. The portable C versions remain as
the `#else` branch and are what the host build and the test suite run.

**Dedicated squaring.** `fe_sqr` no longer calls `fe_mul`. `u256_sqr` computes
the six off-diagonal products once and doubles them, then adds the four diagonal
squares — 10 multiplies instead of 16. Squaring dominates the 255-squaring
inverse chain and every point doubling, so this is a broad win.

Because the CUDA path cannot be exercised without a GPU, both changes are
validated on the host in ways that catch logic errors regardless:

- `test_math` cross-checks `fe_sqr` against `fe_mul(a,a)` over 100k random
  inputs plus edge cases near `p`.
- `test_ptx_sequence` re-implements the exact PTX multiply instruction sequence
  under a carry-flag model matching PTX semantics and checks it against an
  independent bignum multiply over 500k inputs. If the asm sequence is wrong,
  this test diverges. It must be kept in sync with the asm by hand.

This is not a substitute for real hardware: **run `--selftest` on the target
GPU** before a real search. It re-derives known keys on the device and will
catch anything the host cross-checks cannot (a bad opcode, an assembler quirk, a
compiler/driver mismatch).

Expected combined speedup over the portable path is roughly 1.3–1.8×, hardware
dependent; measure on your card.

After any change to the arithmetic, re-run `make test` and `--selftest`.

---

## Notes and limitations

- **Unknown bits must be 63 or fewer.** A 27-byte prefix gives 40, which is
  fine. Below about 22 bytes of prefix the search is not finishing in your
  lifetime regardless of how fast the inner loop is — the cost is exponential in
  the missing bits, and no amount of GPU throughput changes that.
- `--blocks` and `--threads` must be powers of two, and
  `blocks * threads * 2 * HSIZE` must not exceed the range size.
- The batch-inversion fast path assumes no point doubling occurs, i.e. that no
  group centre lands within `HSIZE` of `±kG` for tiny `k`. For any realistic
  prefix this cannot happen; the kernel checks for it anyway and the tool warns
  if you give it a near-zero range.
- If the range crosses the group order `n`, keys above `n` are not valid private
  keys. The tool warns rather than refusing, since the rest of the range is
  still searched correctly.
- Matches raw compressed public keys directly (P2PK), which is exactly what old
  pay-to-pubkey UTXOs expose in their scriptPubKey. There is no address hashing
  in this program — no SHA-256, no RIPEMD-160 — so nothing slows the inner loop
  down on that account.

## Layout

```
src/secp256k1.h              field + curve arithmetic, host and device
src/targets.h                target parsing and lookup table, host and device
src/search.cu                kernel and CLI
test/test_math.cpp           arithmetic vs Python vectors; squaring cross-check
test/test_kernel_logic.cpp   CPU replay of the kernel loop, coverage proof
test/test_targets.cpp        target parsing and lookup-table probe
test/test_progress.cpp       progress-meter formatting and rate/ETA logic
test/test_ptx_sequence.cpp   device PTX multiply sequence vs bignum reference
tools/gen_vectors.py         regenerates test/vectors.h
tools/make_test_targets.py   builds target files, optionally with a planted key
```
