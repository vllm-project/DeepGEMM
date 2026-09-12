# SM120 Port Design — DeepGEMM

**Date:** 2026-09-12
**Target branch:** `dev` (= `origin/main` @ `66081d4`, "Public Release 26/09")
**Source branch:** `origin/nv_dev` @ `572557e` ("Add SM90 FP8 MegaMoE support")
**Merge base:** `559d79f` ("Public release 26/07")

## 1. Goal

Port NVIDIA's SM120 (consumer Blackwell: RTX 50-series, RTX PRO 6000, DGX Spark)
kernel set from `nv_dev` onto the `main` lineage, structured so that:

1. Future rebases onto `upstream/main` stay cheap even when upstream lands large refactors.
2. The branch remains maintainable — a reader can tell what is ours and what is upstream's.

The port itself is mechanical. The design problem is **minimizing the conflict surface**.

## 2. Situation

`nv_dev` forked at the 26/07 release and never took the 26/09 refactor. The infrastructure
diverged substantially:

| Concern | `nv_dev` (26/07 base) | `dev` (26/09) |
|---|---|---|
| JIT layer | in-tree `csrc/jit/*` (6 headers) | external `third-party/deep_jit` submodule |
| Language | C++17 | **C++20** |
| Formatting | `fmt::format`, bundled `third-party/fmt` | `std::format` |
| Kernel host class | CRTP `LaunchRuntime<T>` (`generate_impl` + `launch_impl`) | plain class, single `compile_and_launch` |
| Compile entry | `compiler->build(...)` | `jit->compile(tag, src)` |
| Launch args | `LaunchArgs` | `deep_jit::cuda::LaunchOptions` |
| Device query | `device_runtime->get_arch_major()` | `jit->device.get_arch_major()`, `runtime->get_num_sms()` |
| Epilogue | — | `EpilogueInput` + `csrc/jit_kernels/impls/epilogue.hpp` |
| nvrtc | linked | dropped (driver API via DeepJIT) |
| ArchSpec concept | 6 statics | + `get_num_tma_store_stages` |

Consequence: **device-side `.cuh` files port nearly verbatim; host-side `.hpp` files need a
full mechanical rewrite.** `csrc/jit_kernels/impls/sm100_bf16_gemm.hpp` exists on both
branches and its cross-branch diff is the exact conversion recipe.

## 3. Verified findings

All of the following were established empirically on this machine (4x GB200, sm_100,
`nvcc` 13.1 at `/usr/local/cuda-13.1/bin/nvcc`, torch 2.11.0+cu130). nvcc cross-compiles
sm_120a without sm120 hardware present.

### 3.1 The `-gencode` / `block_scale` blocker does not exist on CUDA 13.x

`nv_dev`'s `csrc/jit/compiler.hpp` carries:

> SM120a requires `-gencode` (`--gpu-architecture` makes ptxas fall back to sm_120,
> losing block_scale and other arch-specific features)

Tested directly with `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X...`:

- `--gpu-architecture=sm_120a` and `-gencode=arch=compute_120a,code=sm_120a` produce
  **byte-identical cubins**, both emitting `QMMA.SF.16832.F32.E4M3.E4M3.E8`
  (the `.SF` suffix confirms the scale-factor/block-scaled variant survived).
- `sm_120a`, `sm_120f`, `sm_121a`, `sm_121f` all compile it correctly.

**Therefore DeepJIT needs no patch.** Its hardcoded `--gpu-architecture=sm_<arch>`
(`include/deep_jit/backend/cuda/options.hpp`) is fine, and its emission of `121f`/`121a`
for sm_12.1 (where `nv_dev` remapped to the `120f` family) is also fine. Both apparent
blockers reduce to a **minimum toolkit requirement: CUDA >= 13.0**.

### 3.2 Device side compiles against `dev` headers

All 12 sm120 device headers pass inclusion-compile for sm_120a against `dev`'s header tree.
Template instantiation then exposed exactly **two** missing dependencies:

1. `sched::Scheduler` — `nv_dev` passes a trailing `kSplitKFactor` that `dev`'s
   `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` does not declare.
2. `ptx::tensor_map_replace_global_dim_in_smem` — present in `nv_dev`'s
   `deep_gemm/include/deep_gemm/ptx/tma.cuh`, absent from `dev`'s.

With `nv_dev`'s `scheduler/gemm.cuh` and `ptx/tma.cuh` supplied, **all 9 kernel
instantiations compile clean** and emit the correct arch-specific MMA:

| Instantiation | SASS |
|---|---|
| `sm120_fp8_fp4_gemm_1d1d_impl` (FP8) | `QMMA.SF.16832.F32.E4M3.E4M3.E8` |
| `sm120_fp8_fp4_gemm_1d1d_impl` (FP4) | `OMMA.SF.16864.F32.E2M1.E2M1.E8` |
| `sm120_bf16_gemm_impl` | `HMMA.16816.F32.BF16` |
| `sm120_tf32_hc_prenorm_gemm_impl` | `HMMA.1688.F32.TF32` |
| `sm120_fp8_mqa_logits` / `sm120_fp8_paged_mqa_logits` | `QMMA.16832.F32.E4M3.E4M3` |
| `sm120_fp4_mqa_logits` / `sm120_fp4_paged_mqa_logits` | `OMMA.SF.16864.F32.E2M1.E2M1.E8` |
| `sm120_bmn_bnk_mn_gemm_impl`, `sm120_split_k_reduce_impl` | (no MMA expected) |

Known-good template arguments for each are recorded in section 7.2.

### 3.3 Split-K is Normal/Batched-only

In `nv_dev`'s scheduler, `num_blocks = num_mn_blocks * kSplitKFactor` appears **only** in the
`GemmType::Normal or GemmType::Batched` branch. All other GemmTypes are untouched. This is
what makes the additive approach in section 5.2 viable.

`kSFKSpan`'s default differs between branches (`nv_dev` 512u, `dev` 128u), but
`sm120_fp8_fp4_gemm_1d1d.cuh` passes it explicitly, so the default is irrelevant to us.

## 4. Conflict-surface taxonomy

The whole design rests on this split.

### Category A — sm120-exclusive files. Zero rebase risk.

Upstream never touches a path named `sm120_*`. 18 files, ~7,000 lines.

**Device side (12 files, ~4,500 lines) — port verbatim:**

| File | Lines |
|---|---|
| `deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh` | 1362 |
| `deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh` | 488 |
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_paged_mqa_logits.cuh` | 370 |
| `deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh` | 359 |
| `deep_gemm/include/deep_gemm/impls/sm120_fp4_mqa_logits.cuh` | 344 |
| `deep_gemm/include/deep_gemm/impls/sm120_fp8_mqa_logits.cuh` | 326 |
| `deep_gemm/include/deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh` | 283 |
| `deep_gemm/include/deep_gemm/impls/sm120_bmk_bnk_mn.cuh` | 236 |
| `deep_gemm/include/deep_gemm/impls/sm120_split_k_reduce.cuh` | 32 |
| `deep_gemm/include/deep_gemm/scheduler/sm120_paged_mqa_logits.cuh` | 281 |
| `deep_gemm/include/deep_gemm/common/sm120_utils.cuh` | 257 |
| `deep_gemm/include/deep_gemm/mma/sm120.cuh` | 163 |

**Host side (6 files, ~2,456 lines) — rewrite to the 26/09 infra:**

| File | Lines | Entry points |
|---|---|---|
| `csrc/jit_kernels/impls/sm120_mqa_logits.hpp` | 691 | 7 |
| `csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp` | 682 | 6 |
| `csrc/jit_kernels/impls/sm120_bf16_gemm.hpp` | 494 | 6 |
| `csrc/jit_kernels/heuristics/sm120.hpp` | 333 | `SM120ArchSpec` |
| `csrc/jit_kernels/impls/sm120_bmk_bnk_mn.hpp` | 130 | 1 |
| `csrc/jit_kernels/impls/sm120_tf32_hc_prenorm_gemm.hpp` | 126 | 1 |

21 host entry points across 5 families:

1. **FP8/FP4 GEMM 1d1d** — `sm120_fp8_fp4_gemm_1d1d`, `_k_grouped_`,
   `_m_grouped_contiguous_`, `_m_grouped_masked_`, `sm120_fp8_fp4_bmm`, `sm120_split_k_reduce`
2. **BF16 GEMM** — `sm120_bf16_gemm`, `_m_grouped_contiguous`, `_m_grouped_masked`,
   `_k_grouped`, `sm120_bf16_bhr_hdr_bhd`, `sm120_bf16_bhd_hdr_bhr`
3. **MQA logits** — `sm120_fp8_mqa_logits`, `sm120_fp4_mqa_logits`, `sm120_mqa_logits`,
   `sm120_paged_mqa_logits_metadata`, `sm120_fp8_paged_mqa_logits`,
   `sm120_fp4_paged_mqa_logits`, `sm120_paged_mqa_logits`
4. **Einsum BMK x BNK -> MN** — `sm120_bmn_bnk_mn_gemm`
5. **TF32 hyperconnection prenorm** — `sm120_tf32_hc_prenorm_gemm`

There is **no SM120 MegaMoE** on `nv_dev`. The 26/09 release's headline feature has no
sm120 path; that is out of scope here (section 9).

### Category B — upstream-owned files we must edit. All rebase risk lives here.

If left as `nv_dev` writes them, this is ~7 files and ~200 lines of edits inside upstream's
hot paths:

| File | `nv_dev`'s edit | Risk |
|---|---|---|
| `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` | split-K restructure (`num_mn_blocks`/`mn_block_idx`) | **High** — shared by sm90/sm100/sm120 |
| `csrc/apis/gemm.hpp` | arch-12 branch + `sm120_to_k_major` + `fp8_fp4_gemm_nt_sm120` (~75 lines) | **High** |
| `csrc/apis/einsum.hpp` | 4 dispatch branches + small-M AB-swap block (~40 lines) | Medium |
| `csrc/apis/attention.hpp` | 4 dispatch branches + `split_kv = 128` vs 256 | Medium |
| `csrc/apis/layout.hpp` | widen `arch_major == 10` predicates to include 12; K-major SF | Low |
| `csrc/apis/hyperconnection.hpp` | 1 dispatch branch | Low |
| `deep_gemm/include/deep_gemm/ptx/tma.cuh` | +1 function (4 lines) | Low |
| `csrc/jit_kernels/heuristics/config.hpp` | +3 struct fields used only by sm120 | Medium |

## 5. Design: isolation-by-indirection

**Principle: every Category B file gets the smallest possible, most re-appliable edit —
ideally one line. All real logic lives in Category A.**

### 5.1 Dispatch indirection

`nv_dev` inlines sm120 flows directly into `csrc/apis/*.hpp`. Instead, introduce
**`csrc/apis/sm120_dispatch.hpp`** (Category A) holding every sm120-specific host flow:
the K-major coercion, the AB-swap-before-SF-transform ordering, the small-M swap
heuristic, the `split_kv` choice.

Each Category B api header then carries exactly one added line per dispatch site:

```cpp
} else if (arch_major == 12) {
    return sm120::fp8_fp4_gemm_nt(a, b, d, c, recipe, recipe_a, recipe_b,
                                  compiled_dims, disable_ue8m0_cast, major_a, major_b, m, n, k);
}
```

Rationale: a one-line insertion into an `if/else` chain is the cheapest merge there is. If
upstream restructures the chain, you re-add one line instead of re-porting 75. This also
keeps the sm120 ordering constraints (K-major-only MMA; swap must precede the single SF
transform) documented in one place rather than smeared across five upstream files.

### 5.2 The two shared-infra additions

**`ptx/tma.cuh` -> avoid entirely.** `tensor_map_replace_global_dim_in_smem` is 4 lines of
inline PTX consumed only by `sm120_fp8_fp4_gemm_1d1d.cuh`. Define it in
`deep_gemm/include/deep_gemm/common/sm120_utils.cuh` (Category A) under a `sm120::` namespace
instead of adding it to upstream's `ptx/tma.cuh`. **This touchpoint drops to zero.**

Do **not** also pull in `nv_dev`'s other `ptx/tma.cuh` deltas — its cache-hint defaults
differ (`EVICT_FIRST` vs `EVICT_NORMAL`), and changing those would silently alter sm90/sm100
behavior.

**`scheduler/gemm.cuh` -> additive, not restructured.** This one cannot be avoided without
forking the scheduler. Two options were considered:

- *Fork:* give sm120 its own `scheduler/sm120_gemm.cuh`. Per-arch schedulers are already
  idiomatic here (`sm90_paged_mqa_logits.cuh`, `sm100_paged_mqa_logits.cuh`,
  `sm120_paged_mqa_logits.cuh`), so this would not look foreign. **Rejected** as the default:
  it silently forgoes upstream improvements to tile swizzling and group iteration.
- *Additive (chosen):* append a **trailing defaulted** `uint32_t kSplitKFactor = 1` to
  `Scheduler` — matching `nv_dev`'s parameter order exactly — add a `num_mn_blocks` field
  plus the split-K state fields, multiply `num_blocks` in the `Normal or Batched` branch
  only, and derive `(mn_block_idx, split_k_idx)` inside that branch of `get_next_block`
  under `if constexpr (kSplitKFactor > 1)`.

Because split-K is Normal/Batched-only (section 3.3), the additive form does **not** require
`nv_dev`'s restructure of every GemmType branch. Scope, measured against `dev`'s actual
`get_next_block`:

- ~6 added lines (the `if constexpr` split, the state fields, the trailing template param)
- **~2-3 modified lines** — `dev`'s Normal branch passes `next_block_idx` directly into
  `is_peer_cta_alive` and `get_swizzled_block_idx`; both must take `mn_block_idx`, and the
  peer-alive bound becomes `num_mn_blocks`. The bounds check stays on the raw index against
  the inflated `num_blocks`, or termination breaks.

So this is *near*-additive, not purely additive: behavior at the default `kSplitKFactor = 1`
is unchanged, but a few existing lines are rewritten rather than only appended. Modified
lines conflict more readily than added ones, which makes this **the one Category B edit most
likely to need hand-merging** — and therefore the place to reach for the fork fallback
(`scheduler/sm120_gemm.cuh`) if upstream churns this function.


#### 5.2.1 `heuristics/config.hpp` struct fields

A third shared-file touchpoint, missed on first pass. `nv_dev` adds three fields consumed
**only** by `heuristics/sm120.hpp` and `impls/sm120_fp8_fp4_gemm_1d1d.hpp`:

| Field | Struct | Default | Purpose |
|---|---|---|---|
| `max_gran_k` | `GemmDesc` | `128` | SF granularity for split-K alignment: `max(gran_k_a, gran_k_b)` |
| `cd_n_contiguous` | `Layout` | `true` | False for AB-swap transposed output; gates the TMA-store epilogue |
| `split_k_factor` | `GemmConfig` | `1` | Chosen split-K factor |

All three are **defaulted**, so adding them is additive and leaves sm90/sm100 behavior
unchanged. They cannot be relocated to Category A — they are fields of upstream's structs.

Conversely, `dev` added `PipelineConfig::num_tma_store_stages` (no initializer).
`SM120ArchSpec::get_pipeline_config` must set it explicitly — use `2`, matching `sm100.hpp`'s
non-k-grouped value — or the field is left indeterminate.

`dev` also added `GemmDesc::is_mxf4_mma()`, `get_smem_pack_factor()`, and `MmaKind::MXF4`,
which `nv_dev`'s `SM120ArchSpec` predates. Review `SM120ArchSpec`'s `get_smem_per_stage` and
`get_storage_config` against these: sm120 packs FP4 via its own `.b4x16_p64` path, so the new
`get_smem_pack_factor()` likely must **not** be applied to sm120. Verify during Task 7.

### 5.3 Resulting conflict surface

Counted precisely against `nv_dev`'s api headers (`arch_major == 12` occurrences), the
Category B surface is **larger than a first read suggests** — 33 api-header edit sites plus
two shared-struct/scheduler changes, across 7 files:

| File | Dispatch arms | Predicate widenings / constants | Extractable body |
|---|---|---|---|
| `csrc/apis/gemm.hpp` | 9 | — | `sm120_to_k_major` (~7 lines) + `fp8_fp4_gemm_nt_sm120` (~70 lines) |
| `csrc/apis/attention.hpp` | 4 | 7 (5 asserts, `block_kv`, `split_kv`) | — |
| `csrc/apis/einsum.hpp` | 3 | 4 | small-M AB-swap block (~34 lines) |
| `csrc/apis/layout.hpp` | — | 5 (4 widenings + 1 K-major SF block) | — |
| `csrc/apis/hyperconnection.hpp` | 1 | — | — |
| `scheduler/gemm.cuh` | — | ~6 added + ~2-3 modified | — |
| `heuristics/config.hpp` | — | 3 defaulted fields added | — |
| `ptx/tma.cuh` | — | **none** (moved to `common/sm120_utils.cuh`) | — |

So indirection does **not** reduce the site *count* — those ~20 dispatch arms and ~16
predicate/constant edits are irreducible; they *are* the seam between upstream's dispatch and
our kernels. What it does buy:

1. **~111 lines of extractable bodies** leave upstream files entirely. These are the edits
   most likely to conflict destructively, because they are real logic sitting inside
   functions upstream actively refactors.
2. Every remaining dispatch arm becomes a **uniform 1-3 line call** into `sm120::`, so a
   conflict is re-added mechanically from the manifest rather than re-derived.
3. The sm120 ordering constraints (K-major-only MMA; AB-swap must precede the single SF
   transform; `split_kv = 128` not 256) are stated once in Category A instead of being
   rediscovered from five upstream diffs.

The honest summary: **~7,000 lines become rebase-immune, ~111 lines of logic move out of
harm's way, and 33 small mechanical api-header edit sites plus one scheduler change remain,
all tracked by checklist.**
That is the realistic ceiling given DeepGEMM dispatches on `arch_major` with inline `if/else`
chains in headers and offers no plugin seam.

Non-header touchpoints, low risk: `setup.py` (no arch flags needed, but a CUDA >= 13 guard),
`csrc/python_api.cpp` if any entry point is newly exported, and the sm120 branches in
`tests/generators.py`, `tests/test_attention.py`, `tests/test_fp8_fp4.py`,
`tests/test_einsum.py`.

### 5.4 Touchpoint manifest

**`AI/sm120_touchpoints.md`** — a checklist enumerating every Category B edit with its exact
anchor (file, enclosing function, the upstream line it attaches to) and the one-line patch.
A rebase becomes: reapply the manifest, run the compile gate. This is the primary
maintainability artifact; it must be updated in the same commit as any Category B change.

### 5.5 Compile gate

**`AI/tools/check_sm120.sh`** — generalizes the harness built during investigation. Drives
`nvcc -cubin --gpu-architecture=sm_120a` over every sm120 device header (inclusion-compile)
plus one known-good instantiation per kernel, asserting the expected MMA opcode appears in
the SASS.

This runs **on any machine with CUDA >= 13**, including this GB200 box. Given no sm120
hardware, it is the only automated verification available, and it catches the entire class
of failure the port is most exposed to: upstream changing a shared device header out from
under a kernel we cannot run. Wire it into CI.

## 6. Testing strategy

No sm120 hardware is available, so verification is layered explicitly:

| Layer | Covers | Available now |
|---|---|---|
| Inclusion-compile, all 12 device headers, sm_120a | include graph, API drift | **Yes** |
| Instantiation + SASS opcode assert, 9 kernels | template compat, correct MMA selected | **Yes** |
| Host-side build (`setup.py build_ext`) | the 6 rewritten host headers compile against DeepJIT | **Yes** |
| Python import + dispatch reachability | `arch_major == 12` branches wired correctly | Partial — needs arch injection or sm120 HW |
| Numerics vs reference | kernel correctness | **No — requires sm120 HW** |
| Performance / heuristics tuning | `SM120ArchSpec` quality | **No — requires sm120 HW** |

The plan must therefore mark the numerics and perf surface as **explicitly unvalidated**, in
both the manifest and the commit history, so whoever gets sm120 hardware knows exactly what
was never run. Do not enable sm120 paths by default in any released artifact until numerics
are confirmed on hardware.

DeepJIT has no arch-override env var (`JIT_*` vars cover cache dir, debug, ptxas, dumps, and
C++ standard only), so host-side dispatch cannot be exercised on GB200 without either a
local DeepJIT patch adding one or real hardware. Prefer waiting for hardware over patching
the submodule.

## 7. Component interfaces

### 7.1 `csrc/apis/sm120_dispatch.hpp`

Namespace `deep_gemm::sm120`. Depends on the five sm120 `jit_kernels/impls` headers,
`heuristics/sm120.hpp`, and `apis/layout.hpp`. Depended on by the five api headers, each via
a single call. Exposes one function per dispatch site, taking the same argument list the
enclosing upstream api function already has in scope — so the hook line needs no argument
marshalling and no new types.

### 7.2 Host-header conversion recipe

Mechanical, applied to all 6 host files. Derived from the `sm100_bf16_gemm.hpp` cross-branch diff:

| `nv_dev` | `dev` |
|---|---|
| `#include "../../jit/{compiler,device_runtime,kernel_runtime}.hpp"` | `#include "../../runtime/runtime.hpp"` |
| `#include "../../utils/format.hpp"` | `#include <format>` |
| `class X final: public LaunchRuntime<X>` | `class X final` |
| `LaunchArgs launch_args;` | `deep_jit::cuda::LaunchOptions options;` |
| `static std::string generate_impl(const Args&)` + `static void launch_impl(...)` | single `static void compile_and_launch(const std::string& tag, const Args&)` |
| `fmt::format(R"(...)")` | `jit->compile(tag, std::format(R"(...)"))` |
| `DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, ...))` | `jit->launch(kernel, args.options, ...)` |
| `device_runtime->get_num_sms()` / `get_tc_util()` | `runtime->get_num_sms()` / `runtime->get_tc_util()` |
| `device_runtime->get_arch_major()` | `jit->device.get_arch_major()` |
| (absent) | add `EpilogueInput epilogue;` to `Args`, `#include "epilogue.hpp"`, thread `args.epilogue.type` / `args.epilogue.args` |

`dev`'s ArchSpec concept (from `heuristics/common.hpp`) requires exactly six members:
`compare`, `get_layout_candidates`, `get_layout_info`, `get_storage_config`,
`get_pipeline_config`, `get_launch_config`. **`nv_dev`'s `SM120ArchSpec` already provides all
six** — no additions needed. (`get_num_tma_store_stages` is private to `sm100.hpp`, not part
of the concept.) What *is* needed is struct-field work in `heuristics/config.hpp`; see 5.2.1.

Known-good instantiation parameters (for the compile gate):

- `sm120_fp8_fp4_gemm_1d1d_impl`: FP8 `<0,4096,7168, 128,128, 1, 128,128,128, 128,128, 128, 3, 128,256, 148, Normal,false, bfloat16_t, EpilogueIdentity, false,false,false, true,false, 128, 1>`; FP4 same but gran `32,32` and `kIsFP4=true, kBIsFP4=false, kAIsFP4=false`
- `sm120_bf16_gemm_impl`: `<0,4096,7168, 1, 128,128,64, 128,128, 128, 3, 128,256, 148, Normal,false, bfloat16_t, EpilogueIdentity, true, 1>`
- `sm120_bmn_bnk_mn_gemm_impl`: `<0,128,128, 128,64,64, 1, 128, 3, 128,256>` (requires `BLOCK_M == 128`)
- `sm120_tf32_hc_prenorm_gemm_impl`: `<128,128, 128,64,64, 1, 3, 256,128>` (requires `BLOCK_K == 64`, `BLOCK_M == warps * tiles * 16`)
- `sm120_{fp8,fp4}_mqa_logits`: `<32,128, false, 64,128, 2,4, 148, 128,256, float>` (requires `BLOCK_KV == (math_threads/32) * 16`)
- `sm120_{fp8,fp4}_paged_mqa_logits`: `<2,32, 128,64, false,false, 2,4, 128, 128,256, float>`
- `sm120_split_k_reduce_impl`: `<bfloat16_t, 2>`

## 8. Risks and open items

| Risk | Mitigation |
|---|---|
| Numerics never validated | Explicit in manifest + commits; do not enable by default in releases |
| Heuristics (`SM120ArchSpec`) untuned for real sm120 | Ship as-is from `nv_dev`; flag as needing on-HW tuning |
| Upstream churns `scheduler/gemm.cuh` badly | Documented fallback: fork to `scheduler/sm120_gemm.cuh` |
| CUDA < 13 users | Gate sm120 paths on toolkit version with a clear error |
| `apis/attention.hpp` edits collide with in-flight local MQA-logits work | Dispatch indirection keeps our edit to one line; sequence after that work settles |
| Host-side conversion introduces silent bugs no test can catch | Convert one family at a time, each gated on the compile gate |

## 9. Out of scope

- **SM120 MegaMoE** — does not exist on `nv_dev`; would be new kernel development.
- **Backporting split-K to sm90/sm100** — the shared-scheduler change makes it *possible*
  (`kSplitKFactor` is arch-agnostic) but wiring it into those ArchSpecs is separate work.
- **Patching DeepJIT** — shown unnecessary (section 3.1). Revisit only if an arch-override
  env var is wanted for host-side testing without hardware.
- **`sm_121` (DGX Spark / GB10) specific tuning** — compiles correctly; untuned.
