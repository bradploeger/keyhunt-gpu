# keyhunt-gpu

Searches every private key sharing a fixed high-order prefix, derives the
compressed public key for each, and reports any that appear in a target list.

A 27-byte (216-bit) prefix leaves 40 unknown bits — **1,099,511,627,776 keys**.
On a modern GPU that is a run of roughly 15 minutes to an hour.

On a match it prints the 33-byte compressed public key and the full 256-bit
private key, both in hex, appends them to a file, and can run a command of your
choosing (for email, Telegram, ntfy, whatever you use).

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
It also parses every accepted target format and probes 20,000 targets through
the same index functions the kernel uses. All of it passes.

I could not compile or run the CUDA path — no `nvcc` and no GPU on the machine
I built this on. The device code shares the same header the tests exercise, but
**run `./keyhunt-gpu --selftest` first.** It validates device arithmetic against
the host and then plants a known key in a small range and confirms the search
finds it. If that passes, the pipeline is sound end to end.

---

## Build

Needs CUDA 11 or newer.

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

```
build.bat 86
```

The argument is your compute capability without the dot. Find it with:

```
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

The script builds `keyhunt-gpu.exe`, builds the three host tests, and runs
them. Or use CMake, which works on both platforms:

```
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
ctest --test-dir build
```

Tunables, all optional:

```
make HSIZE=256        # keys per batch inversion = 2*HSIZE (default 128)
make FILTER=19        # log2 of the shared-memory prefilter bits (default 18 = 32 KB)
```

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
--out FILE          append matches here (default found.txt)
--device N          CUDA device index
--blocks N          grid size, power of two (default 8 x SM count)
--threads N         block size, power of two (default 256)
--groups N          groups per launch; controls progress/checkpoint interval
--checkpoint FILE   save progress after every launch
--resume            start from the checkpoint
--notify-cmd CMD    run on a match. %P becomes the pubkey, %K the private key
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
  private key (256 bit) : a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4...
========================================================================
```

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
together. Default `HSIZE=128` gives 256 keys per inversion.

**3. Both signs share one inverse.** `P + iG` and `P − iG` have the same
denominator `(x_P − x_iG)`, so one inverse produces two keys. Better still, the
new X depends only on the *square* of the slope, so the sign flip is free.

**4. Never compute Y.** A compressed key is a parity byte plus X, and matching
happens on X alone. Y is only needed to advance the group centre, once per 256
keys. That removes a multiplication per key.

The inversion uses the standard 255-squaring / 15-multiplication addition chain
for `a^(p-2)`, amortised to about one multiplication per key.

Matching is two-stage, because 1.1e12 lookups into a table will otherwise become
the bottleneck. First a 32 KB bitmap held in **shared memory**, indexed by 18
bits of X — with 35,000 targets this rejects about 87% of candidates at
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

---

## Tuning

Build with `-Xptxas -v` (already on) and watch the register count. If you see
spills, that is the first thing to fix.

- **`HSIZE`** — larger means fewer inversions but more local memory per thread
  (`(HSIZE+1) * 32` bytes). 128 and 256 are both reasonable; measure.
- **`--blocks` / `--threads`** — start at 8x SM count and 256. More blocks helps
  until local memory pressure bites.
- **`FILTER`** — 19 gives a 64 KB bitmap and rejects ~93% instead of ~87%, at
  the cost of fewer resident blocks per SM. Worth trying if the profiler shows
  you are memory-bound rather than ALU-bound.
- **`--groups`** — larger means less host round-tripping, but a coarser progress
  bar and checkpoint interval.

### Where the remaining performance is

Two things I left on the table, both worth roughly 1.3–1.8x together:

**PTX carry chains.** `secp256k1.h` propagates carries portably, with
`s = a + b; carry = (s < a)`. The compiler recognises much of this, but
hand-written `add.cc.u64` / `addc.cc.u64` chains are meaningfully tighter. The
place to start is `mul256x64` and `u256_add`:

```cpp
asm("{\n\t"
    "add.cc.u64  %0, %5, %9;\n\t"
    "addc.cc.u64 %1, %6, %10;\n\t"
    "addc.cc.u64 %2, %7, %11;\n\t"
    "addc.cc.u64 %3, %8, %12;\n\t"
    "addc.u64    %4, 0, 0;\n\t}"
    : "=l"(r[0]),"=l"(r[1]),"=l"(r[2]),"=l"(r[3]),"=l"(carry)
    : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
      "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
```

Keep each carry chain inside a **single** `asm` block. Splitting one chain
across several statements relies on the carry flag surviving between them, which
is not something the compiler guarantees.

**Dedicated squaring.** `fe_sqr` currently calls `fe_mul`. A real squaring
routine computes the off-diagonal products once and doubles them, saving about a
third of the work. Squaring is roughly one operation in three here, and
dominates the inversion chain.

After either change, re-run `make test` and `--selftest`.

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
test/test_math.cpp           arithmetic vs Python reference vectors
test/test_kernel_logic.cpp   CPU replay of the kernel loop, coverage proof
test/test_targets.cpp        target parsing and lookup-table probe
tools/gen_vectors.py         regenerates test/vectors.h
tools/make_test_targets.py   builds target files, optionally with a planted key
```
