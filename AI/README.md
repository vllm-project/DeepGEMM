# AI/ — SM120 port working notes

This directory is the honest status of the `sm120-port` branch. Read this file before you
trust anything green in it.

| File | What it is |
|---|---|
| `2026-09-12-sm120-port-design.md` | Design, and the empirical findings behind it |
| `2026-09-12-sm120-port-plan.md` | The implementation plan. Historical: where execution proved a step wrong, the step carries an inline `CORRECTED DURING EXECUTION` note. |
| `sm120_touchpoints.md` | **Every edit to an upstream-owned file, plus the rebase procedure. Read before rebasing.** |
| `tools/check_sm120.sh` | Device compile gate — any CUDA>=13 machine, no sm120 GPU needed |
| `tools/check_sm120_host.sh` | Host-header compile gate |
| `tools/check_sm120_cuda_guard.sh` | Checks the CUDA>=13 guard fires on 12.x and only there |
| `tools/sm120_headers.txt` | Device headers the gate compiles standalone |
| `tools/sm120_host_headers.txt` | Host headers the gate compiles standalone |
| `tools/sm120_tu/` | One known-good instantiation per kernel, plus the SASS opcode it must emit |

## What this branch is

Ported from `origin/nv_dev` @ `572557e` (the 26/07 lineage) onto `66081d4` (Public Release
26/09). It adds 21 `sm120_*` host launchers across 5 families — FP8/FP4 GEMM, BF16 GEMM, MQA
logits, einsum, TF32 hyperconnection — behind 12 sm120 device headers.

```bash
grep -hcE "^static (void|torch::Tensor|auto) sm120_" csrc/jit_kernels/impls/sm120_*.hpp | paste -sd+ | bc   # 21
wc -l < AI/tools/sm120_headers.txt                                                                          # 12
```

There is no SM120 MegaMoE. It does not exist upstream either.

## NOTHING HERE HAS BEEN EXECUTED

**No sm120 kernel in this branch has ever run. Not once.**

The development host is 4× NVIDIA GB200 — compute capability 10.0. `nvcc` cross-compiles
`sm_120a` from any host, which is why the compile gates work; nothing here can *run* an sm120
binary. Every green result in this port means exactly one of three things:

- **it compiled** — `nvcc -cubin --gpu-architecture=sm_120a` exited 0, or `g++ -fsyntax-only`
  did;
- **the command exited zero** — e.g. `setup.py build_ext --inplace`;
- **the SASS contained opcode X** — `cuobjdump -sass` output matched a string in
  `AI/tools/sm120_tu/*.expect`.

None of those is a numerical result. There are **no** numerics, **no** performance numbers, and
**no** kernel launches anywhere in this work. Every heuristic (block sizes, stage counts,
split-K factors, swizzle choices, the AB-swap thresholds) is an untested guess inherited from
`nv_dev` or derived analytically.

**Do not enable sm120 by default in any release until numerics are confirmed on real SM120
hardware.**

## The single most useful fact for whoever gets hardware

**Of the 37 dispatch edit points this branch makes in upstream-owned `csrc/` files, exactly 2
have downstream detection. The other 35 fail silently.**

The two that are detectable are the attention constants `sm120::kMqaBlockKv` and
`sm120::kPagedSplitKv`: reverting either to 256 trips `DG_HOST_ASSERT(split_kv == 128)` in
`csrc/jit_kernels/impls/sm120_mqa_logits.hpp` or the `DG_STATIC_ASSERT(BLOCK_KV == kNumMathWarps * MMA_M)`
in `deep_gemm/include/deep_gemm/impls/sm120_fp8_mqa_logits.cuh`.

The other 35 are predicate widenings (`arch_major == 10` becoming
`(arch_major == 10 or arch_major == 12)`) and `else if (arch_major == 12)` dispatch arms —
runtime conditions inside function bodies. Drop one on a rebase and it stays invisible through:
compilation, both compile gates, `setup.py build_ext --inplace`, **and the entire test suite on
any sm90/sm100 machine**. It surfaces only on real SM120 silicon, as a `DG_HOST_UNREACHABLE`
abort, a `DG_HOST_ASSERT` failure, or — for `einsum.hpp`'s `fp8_bmm` arm — a silent misroute
into the SM90 path.

`sm120_touchpoints.md` tabulates all 37 per site, with the re-derivation commands. That manual
diff is the only defence that exists.

**And the enumeration itself has a blind spot.** Those 37 all live in the six files the Task 13
sweep scoped. A **thirteenth** upstream-owned file, `csrc/jit_kernels/impls/smxx_layout.hpp`,
was found later: a `get_arch_major() == 10` assert that needed widening, reached indirectly from
`csrc/apis/layout.hpp`. Because it contained no `== 12`, every re-derivation grep in
`sm120_touchpoints.md` returned clean while both k-grouped FP8 entry points aborted on SM120.
Rebase step 4 in that document now greps the `== 10` predicates for the same class of miss. A
rebaser who runs only the `== 12` greps will repeat the mistake.

## Deliberate divergences from `nv_dev`

Three, all intentional. Do not "fix" them by re-syncing from `nv_dev` without reading why.

### 1. PDL is off by default (performance only)

`nv_dev`'s `LaunchArgs` defaulted `enable_pdl = true`. This branch has no per-launch PDL flag —
only the global `set_pdl` (`csrc/apis/config.hpp:24`) — so **sm120 runs without PDL unless a
user switches it on**. That matches every sm100 implementation in this tree, so it is a
consistency choice, not an oversight, and it is performance-only: no numerical difference.

Note that the sm120 split-K pair *is* PDL-aware:
`cudaTriggerProgrammaticLaunchCompletion()` at `sm120_fp8_fp4_gemm_1d1d.cuh:1351` is paired
with `cudaGridDependencySynchronize()` at `sm120_split_k_reduce.cuh:13`. That path goes live
only if a user calls `set_pdl(True)`, and it **has never been exercised**.

### 2. `heuristics/runtime.hpp` mk-alignment left unwired (performance only)

`nv_dev` generalised `get_contiguous_mk_alignment` with a per-arch spec table, returning a
`{128, 64, 64}` BLOCK_M search spec for arch 12. This branch leaves that unwired: **sm120 gets
the legacy 128-byte alignment, the same as sm90.**

Wiring it would mean porting that machinery into a shared, upstream-owned file, buying
permanent rebase surface for an untestable performance heuristic. Verified performance-only:
`get_theoretical_mk_alignment_for_contiguous_layout` has exactly one consumer in `csrc/` — the
pybind shim at `csrc/apis/layout.hpp:156` — and appears in no dispatch path. Revisit if sm120
hardware shows contiguous-layout alignment mattering.

### 3. `SM120ArchSpec::get_split_k_factor` repairs an upstream defect (a fix, not a port)

**Do not re-sync this function from `nv_dev`.** Upstream's version establishes an SF-alignment
invariant with a search loop and then breaks it: two `std::min` clamps can lower `split_k` onto
a value that no longer satisfies the invariant the loop had just established, with nothing
re-checking. This branch hoists the invariant into a lambda used by both the search and a
post-clamp re-check, walks `split_k` back down, and asserts. Over a 2,007,040-point parameter
sweep: **6594 violating points → 0**, with exactly those 6594 points changing their selected
factor (all to `1`) and no other point moving.

Diverging costs nothing here: `csrc/jit_kernels/heuristics/sm120.hpp` is Category A —
sm120-exclusive, upstream never opens it.

## Upstream comments make hardware claims this port never exercised

Ported code carries `nv_dev`'s original comments, including empirical ones. For example
`deep_gemm/include/deep_gemm/mma/sm120.cuh:81`:

```
// Uses mxf8f6f4 with e2m1 A and e4m3 B. Verified on RTX PRO 6000 (sm_120a).
```

That is **NVIDIA's claim in its original context**, preserved because the port is faithful.
**It is not a claim this branch makes.** Nothing in this branch has been verified on an RTX PRO
6000 or on any other sm_120a device. Treat every such comment as provenance, not as evidence.

## The CUDA >= 13 guard

`deep_gemm/include/deep_gemm/common/sm120_utils.cuh` carries a compile-time `#error` for
`__CUDACC_VER_MAJOR__ < 13`. `nv_dev`'s `csrc/jit/compiler.hpp` reported that
`--gpu-architecture=sm_120a` makes ptxas fall back to `sm_120` and lose `block_scale` — which
would produce wrong numbers rather than fail to build. Design doc §3.1 disproved that **on
13.x** (both spellings produce byte-identical cubins that still emit `QMMA.SF...`) and reduced
it to a minimum-toolkit requirement of CUDA >= 13.0. It could not be tested on 12.x — no such
toolkit is installed here — so the guard is precautionary, and deliberately so: the failure it
guards against is silent wrong numerics. DeepJIT's own floor is 12.9; this tightens it for
sm120 only.

It lives in that device header, not in `csrc/apis/sm120_dispatch.hpp`, for two reasons: the
header is Category A, so the guard costs zero rebase surface; and being inside the
`__CUDA_ARCH__ >= 1200` block it fires at exactly the right moment — when the JIT invokes nvcc
for an sm120 kernel — rather than at extension-build time. The host extension never includes
it, so sm90/sm100 builds are unaffected. There is deliberately **no** runtime version check:
DeepJIT exposes no public nvcc-version accessor, so one would be a no-op wearing a seatbelt.

Coverage is the 8 sm120 kernels that issue MMA. `impls/sm120_split_k_reduce.cuh` and
`scheduler/sm120_paged_mqa_logits.cuh` do not include the header — they are a plain FP32 reduce
and a metadata kernel, with no block-scaled MMA between them.

`AI/tools/check_sm120_cuda_guard.sh` verifies it fires on a 12.x sm120 device pass and stays
silent on 13.x, on host passes, and on sm90/sm100 device passes. This host has no CUDA 12.x
toolkit, so those five legs drive `gcc -E` with `__CUDA_ARCH__` and `__CUDACC_VER_MAJOR__` set
explicitly.

That is not a stand-in for nvcc's device pass — it **is** that pass. `nvcc --dryrun` for
`-cubin --gpu-architecture=sm_120a` shows the device preprocessing step is literally
`gcc -std=c++20 -D__CUDA_ARCH__=1200 ... -E -x c++ ... -D__CUDACC_VER_MAJOR__=13`, emitting a
`.cpp1.ii` that is then handed to `cicc`; cicc never sees a preprocessor directive. The same
observation explains why `nvcc -D__CUDACC_VER_MAJOR__=12` does *not* simulate a 12.x toolkit:
on that generated command line the user's `-D "__CUDACC_VER_MAJOR__=12"` appears **before**
nvcc's own `-D__CUDACC_VER_MAJOR__=13`, and the last `-D` wins.

Two further legs: a real CUDA 12.x compile wherever such a toolkit exists (skipped here), and a
sensitivity probe — the same `#error` in the same place with its comparison inverted to `>= 13`,
which must abort a real `nvcc --gpu-architecture=sm_120a` compile. Without that leg every check
would be `gcc -E` and nothing would show the guard can fail the toolchain it exists to fail.
Removing the guard from a scratch copy of the tree makes the gate report `pass=4 fail=2` and
exit 1, as it should.

## Test gating

`tests/{generators,test_attention,test_fp8_fp4,test_einsum}.py` gained arch-12 enumeration rows.
Each row names, in a comment, the `DG_HOST_ASSERT` or `DG_STATIC_ASSERT` it restates; those
asserts are the source of truth. Two rows have no assert behind them and say so: the paged
`next_n` range (justified instead by the `kPadOddN` code path, which exists only for odd
`next_n >= 3`) and the einsum arch-12 tolerance (inherited from `nv_dev`, no evidence here).
Every widened value was additionally compile-probed for `sm_120a` with real nvcc before being
enumerated. See `sm120_touchpoints.md` → "Category B, part 3".

**These rows are dead code on this host** (capability 10.0) and were verified not to change the
sm90/sm100 enumerations. They make no sm120 test runnable. No sm120 test case in this repo has
been executed.

The tests are **standalone scripts, not pytest** — there is no pytest in this environment.
Running them needs `PYTHONPATH` pinned to the repo root (else a stale site-packages
`deep_gemm 2.3.0` is imported) and the two `deep_gemm/include/{cute,cutlass}` symlinks that
`develop.sh` creates (else every JIT compile fails with `fatal error: cute/numeric/math.hpp: No
such file or directory`, which mimics a code regression exactly). Both failure modes look like
bugs in this branch and are not. `sm120_touchpoints.md` → "Verification for this part" has the
exact invocation.

Four scripts cannot run on this host for reasons unrelated to the port, and fail identically at
the pre-Task-14 tree: `test_mega_gate.py` (no `tile_kernels`), `test_mega_mhc.py` and
`test_sanitizer.py` (no `tilelang`), `test_mega_moe.py` (needs 8 ranks; this host has 4 GPUs).

## Known residual risks

- **`check_grouped_ab_fp8_fp4` widening is argued-safe but untested at runtime.**
  `tests/test_mega_moe.py` is the only runtime exercise of it and cannot run here (8 ranks, 4
  GPUs). The widening adds a disjunct and cannot change the arch-10 result, but that is an
  argument, not an observation. Re-run that script on an 8-GPU host after any rebase touching
  `csrc/utils/layout.hpp`.
- **The attention A/B used a seeded 40-case sample, not the full sweep.**
  `tests/test_attention.py` honours `DG_MQA_NUM_CASES`, sampling from a per-section seeded RNG,
  so the same N selects the identical case set on both sides of an A/B — but 40 of 2304 + 4320 +
  161 cases is a sample, not coverage.
- **The arch-12 test enumerations are derived from asserts and compile probes, not from runs.**
  An assert says what the code *refuses*, not what it *computes correctly*, and a successful
  `sm_120a` compile says less still. A row that is too narrow silently loses coverage; one that
  is too wide aborts on hardware. Neither is detectable here. Two rows rest on weaker ground
  than the rest: the paged `next_n` range (no assert bounds it) and the einsum arch-12
  tolerance (`nv_dev`'s numbers, no local evidence).
- **The k-grouped SF-layout list for arch 12 is wider than `nv_dev`'s.** `nv_dev` restricts
  arch 12 to `[(32,128),(128,128)]`; this branch keeps the full non-SM90 list, because its own
  asserts accept any `k_alignment % 128 == 0` and no code here justifies the narrower set. If
  `nv_dev` had a hardware reason, this branch will enumerate cases SM120 rejects.
- **Every performance heuristic is unvalidated**, as is every numerical result. This bears
  repeating because it is the whole risk surface: the port is a compile-verified transcription.

## A consistency note (not a defect)

`csrc/jit_kernels/impls/sm120_mqa_logits.hpp` uses five named runtime classes, one per entry
point, where this branch's sm90/sm100 MQA-logits headers are flat `static void` launchers.
The other sm120 host headers match their sm100 counterparts one-for-one:

| Header | sm120 runtime classes | sm100 counterpart |
|---|---|---|
| `*_fp8_fp4_gemm_1d1d.hpp` | 2 | 1 — the extra is `SM120SplitKReduceRuntime`, a kernel sm100 does not have |
| `*_bf16_gemm.hpp` | 1 | 1 |
| `*_bmk_bnk_mn.hpp` | 0 | 0 |
| `*_tf32_hc_prenorm_gemm.hpp` | 0 | 0 |
| `*_mqa_logits.hpp` | **5** | **0** |

So MQA logits is the sole departure. It is a style inconsistency, not a defect, and nothing
depends on it. If it is ever cleaned up, the direction is MQA-logits → flat, matching the rest
of the tree — never the reverse.

## Verification, in full

```bash
CUDA_HOME=/usr/local/cuda-13.1 ./AI/tools/check_sm120.sh              # pass=23 fail=0
CUDA_HOME=/usr/local/cuda-13.1 ./AI/tools/check_sm120_host.sh         # pass=7  fail=0
CUDA_HOME=/usr/local/cuda-13.1 ./AI/tools/check_sm120_cuda_guard.sh   # pass=6  fail=0 (7 with a CUDA 12.x toolkit)
touch csrc/python_api.cpp                                             # REQUIRED -- see sm120_touchpoints.md
CUDA_HOME=/usr/local/cuda-13.1 python setup.py build_ext --inplace
PYTHONPATH="$PWD" python tests/test_fp8_fp4.py                        # and the other standalone scripts
```

| Layer | Covers | Can it catch a dropped dispatch arm? |
|---|---|---|
| `check_sm120.sh` | 12 device headers + 11 instantiations, correct MMA opcode | no |
| `check_sm120_host.sh` | 7 sm120 host headers against torch + DeepJIT | no |
| `check_sm120_cuda_guard.sh` | the CUDA>=13 `#error`: fires, does not false-positive, and can still abort a real `sm_120a` compile | n/a |
| `setup.py build_ext --inplace` | every Category B edit compiles and type-checks in situ | no |
| Scheduler no-regression check | `kSplitKFactor == 1` is inert (4 steps, incl. a sensitivity probe) | n/a |
| the standalone `tests/` scripts | no sm90/sm100 regression | no |
| **Numerics** | — | **UNAVAILABLE — needs sm120 hardware** |
| **Performance** | — | **UNAVAILABLE — needs sm120 hardware** |

The right-hand column is the point. The only thing that catches a dropped arm is the manual
re-derivation in `sm120_touchpoints.md`.
