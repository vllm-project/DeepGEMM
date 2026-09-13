# SM120 Category B Touchpoints

Every edit this branch makes to an **upstream-owned** file. On rebase: reapply each row,
then run `AI/tools/check_sm120.sh` and the no-regression check below.

**Rule: any change to an upstream-owned file updates this file in the same commit.**

## Rebase procedure

Category A files never conflict — upstream does not have them:
`deep_gemm/include/deep_gemm/{mma,common,impls,scheduler}/sm120_*.cuh`,
`deep_gemm/include/deep_gemm/mma/sm120.cuh`, `csrc/jit_kernels/heuristics/sm120.hpp`,
`csrc/jit_kernels/impls/sm120_*.hpp`, `csrc/apis/sm120_dispatch.hpp`, and everything under
`AI/`. Everything that *can* conflict is tabulated in this file.

1. Rebase or merge onto the new upstream.
2. For each conflict in an upstream-owned file, find its section below and reapply those rows.
   **Do not re-derive the edit from `origin/nv_dev`.** That branch is on the 26/07 infra; its
   version of these files will not apply, and its test enumerations encode different kernel
   constraints — see "Deviations from `nv_dev`, collected".
3. Re-derive the site list from the new upstream and diff it against the tables here **by
   hand** — see "How to re-derive this list after a rebase". This is the only defence against a
   silently dropped dispatch arm; "READ THIS BEFORE TRUSTING A GREEN CHECKMARK" explains why no
   gate in this repo catches one.
4. `CUDA_HOME=/usr/local/cuda-13.1 ./AI/tools/check_sm120.sh` — every device kernel compiles and
   emits its expected MMA opcode. Was `pass=23 fail=0` when this was written.
5. `CUDA_HOME=/usr/local/cuda-13.1 ./AI/tools/check_sm120_host.sh` — every sm120 host header
   compiles. Was `pass=7 fail=0`.
6. `CUDA_HOME=/usr/local/cuda-13.1 ./AI/tools/check_sm120_cuda_guard.sh` — the CUDA>=13 guard in
   `deep_gemm/common/sm120_utils.cuh` still fires on 12.x and stays silent on 13.x, on host
   passes, and on sm90/sm100 device passes. Was `pass=5 fail=0` (6 where a CUDA 12.x toolkit is
   installed; that one check is skipped otherwise).
7. `touch csrc/python_api.cpp && CUDA_HOME=/usr/local/cuda-13.1 python setup.py build_ext --inplace`
   — the extension must build. The `touch` is **required**; see "Host-side no-regression check —
   and its false-green trap".
8. If `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` conflicted, re-run the **four-step**
   no-regression check in "No-regression check" below — all four steps, including step 3, the
   sensitivity probe.

   **Do not substitute a bare `cmp` of two cubins for steps 1-3.** nvcc 13.1 is not
   byte-deterministic: it embeds a per-compilation cookie, so two compiles of *identical*
   source already differ and `cmp` reports a difference that means nothing. The recipe below
   normalises the cookie (or compares `.text` directly), and the sensitivity probe is what
   proves the comparison is capable of reporting a difference at all. A probe that reports
   `IDENTICAL` means the comparison is broken, not that the change is inert.
9. Re-run the standalone test scripts under the conditions in "Verification for this part"
   (`PYTHONPATH` pinned to the repo root, plus the two `deep_gemm/include` symlinks). They are
   **not** pytest, and four of them cannot run on a 4-GPU host at all — that table lists which.
10. If a new field appears in `PipelineConfig`, `Layout`, `GemmDesc`, or the `ArchSpec`
    interface, check whether `SM120ArchSpec` (`csrc/jit_kernels/heuristics/sm120.hpp`) must
    populate it. A field it does not set takes a silent default; it is not a compile error.
11. If `tests/generators.py`, `tests/test_attention.py`, `tests/test_fp8_fp4.py` or
    `tests/test_einsum.py` conflicted, recheck each arch-12 enumeration row against the host
    assert it cites. Those asserts are the source of truth; the enumeration only restates them.

**Still unverified on real hardware: all numerics and all performance heuristics.** Read
`AI/README.md` before drawing any conclusion from a green run.

## Inventory

| Upstream file | Edits | Added by |
|---|---|---|
| `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` | 5 | Task 5 (split-K) |
| `csrc/jit_kernels/heuristics/config.hpp` | 3 | Task 7 (heuristics) |
| `csrc/apis/gemm.hpp` | 10 | Task 13 (dispatch) |
| `csrc/apis/attention.hpp` | 11 | Task 13 (dispatch) |
| `csrc/apis/einsum.hpp` | 10 | Task 13 (dispatch) |
| `csrc/apis/layout.hpp` | 5 | Task 13 (dispatch) |
| `csrc/apis/hyperconnection.hpp` | 2 | Task 13 (dispatch) |
| `csrc/utils/layout.hpp` | 3 | Task 13 (dispatch) |
| `tests/generators.py` | 2 | Task 14 (test gating) |
| `tests/test_attention.py` | 2 | Task 14 (test gating) |
| `tests/test_fp8_fp4.py` | 3 | Task 14 (test gating) |
| `tests/test_einsum.py` | 4 | Task 14 (test gating) |

Everything else this branch adds is a **new** file (Category A: `deep_gemm/{mma,common,impls,scheduler}/sm120_*.cuh`,
`csrc/jit_kernels/heuristics/sm120.hpp`, `csrc/jit_kernels/impls/sm120_*.hpp`,
`csrc/apis/sm120_dispatch.hpp`, `AI/tools/**`) and
cannot conflict on rebase. As of this commit, the twelve files above are the only upstream-owned
files the branch modifies. Verify that claim after any rebase with:

```bash
git diff --diff-filter=M --name-only <upstream-base>...HEAD
```

Anything listed there that is not in the table above is an undocumented touchpoint — add it.

## READ THIS BEFORE TRUSTING A GREEN CHECKMARK

**Most of the edits in this file fail silently if a rebase drops them. Nothing in this repo
detects that.**

The majority of this branch's Category B surface is **predicate widenings** —
`arch_major == 10` becoming `(arch_major == 10 or arch_major == 12)` — and **dispatch arms**
guarded by `else if (arch_major == 12)`. Both are *runtime conditions inside function bodies*.
Drop one and:

| Check | Still passes? | Why |
|---|---|---|
| `setup.py build_ext --inplace` | **yes** | the call still type-checks; a missing branch is not a type error |
| `AI/tools/check_sm120_host.sh` | **yes** | `-fsyntax-only`; it never evaluates a branch |
| `AI/tools/check_sm120.sh` | **yes** | compiles device TUs; knows nothing about host dispatch |
| the whole `tests/` suite | **yes** | on sm90/sm100 hardware nothing ever sets `arch_major == 12` |

The failure appears **only when an SM120 device runs an sm120 kernel**, as a
`DG_HOST_UNREACHABLE` abort or a `DG_HOST_ASSERT` failure — and **this project has no sm120
hardware**, so no check that exists here can reach it.

What the green gates actually prove: that the arms which are *present* compile and type-check
against the `sm120_*` launcher signatures, and that sm90/sm100 still behave as before. They
prove **nothing** about arms that are *absent*. The only defence is the per-site tables below.
After any rebase, re-derive the site list from upstream and diff it against those tables by
hand:

```bash
for f in gemm attention einsum hyperconnection layout; do
  echo "=== $f"; grep -cE "arch_major == 12|== 12\)" csrc/apis/$f.hpp
done
grep -cE "arch_major == 12|== 12\)" csrc/utils/layout.hpp    # must be 3
```

Expected counts for this branch (textual occurrences, which exceed the edit-row counts where
one edit contains two mentions): gemm **9**, attention **10**, einsum **9**,
hyperconnection **1**, apis/layout **6**, utils/layout **3**.

Use the **same two-branch pattern** here as in the upstream re-derivation above. Today every
arch-12 mention in our files goes through a local named `arch_major`, so the narrow pattern
happens to give identical counts — but a rebase that pulls in upstream's
`get_arch_major() == 12` phrasing would make the narrow pattern undercount here exactly as it
does against `nv_dev`. Keep the patterns identical so the two sides stay comparable.

## deep_gemm/include/deep_gemm/scheduler/gemm.cuh

| # | Anchor | Edit |
|---|---|---|
| 1 | `Scheduler` template parameter list, after `kNum1DBlocksPerGroup` | add `uint32_t kSplitKFactor = 1` |
| 2 | state fields, after `num_n_blocks` | add `num_mn_blocks` and `split_k_idx` (exactly two fields — see note below) |
| 3 | constructor, `Normal or Batched` branch | hoist `num_mn_blocks = num_m_blocks * num_n_blocks;` above the `if constexpr`; `num_blocks = num_mn_blocks * kSplitKFactor` |
| 4 | `get_next_block`, final `else` branch | derive `mn_block_idx`/`split_k_idx` under `if constexpr (kSplitKFactor > 1)`; feed `mn_block_idx` to `is_peer_cta_alive` and `get_swizzled_block_idx`; bounds check stays on the raw index |
| 5 | `Scheduler` body, after the existing `kKAlignment` `DG_STATIC_ASSERT` | add `DG_STATIC_ASSERT(kSplitKFactor == 1 or kGemmType != GemmType::Batched, ...)` |

**Do not re-add `k_partition_start` / `k_partition_end`.** nv_dev declares them but never writes
or reads them anywhere; they are vestigial upstream. They were dropped deliberately to keep the
Category B diff minimal — two dead `uint32_t` in a shared struct are pure rebase-conflict
surface for zero function. If a future nv_dev sync shows them becoming live, re-add them then.

**Touchpoint 5 rationale.** The constructor inflates `num_blocks` for `Normal` **or** `Batched`,
but `get_next_block`'s `Batched` branch then derives `current_group_idx = next_block_idx /
num_blocks` from that inflated value and never sets `split_k_idx`. That combination silently
produces wrong group indexing. The same defect exists upstream in nv_dev. We deliberately do
**not** rewrite the `Batched` branch — diverging there buys an untestable correctness fix at
permanent rebase cost. The assert converts a silent wrong result into a compile error instead.
This is reachable, not theoretical: `sm120_fp8_fp4_bmm` is a `Batched` entry point.
`Batched` at the default `kSplitKFactor == 1` is unaffected and still compiles.

`kSplitKFactor` **must stay the last template parameter**, after `kNum1DBlocksPerGroup` — this
matches nv_dev's order, and `sm120_fp8_fp4_gemm_1d1d.cuh` passes it positionally.

### Why this is inert at `kSplitKFactor == 1`

- Edit 3 hoists `num_mn_blocks` out of the `if constexpr`, so it is now assigned for every
  `GemmType`. For every type other than `Normal`/`Batched`/`MGroupedContiguous` it is a dead
  store (never read) and is eliminated.
- Edit 4's final `else` branch is reached by **both** `Normal` and `MGroupedContiguous`. For
  `Normal` at factor 1, `num_blocks == num_mn_blocks * 1`. For `MGroupedContiguous` the
  constructor sets `num_blocks = num_m_blocks * num_n_blocks`, which is also `== num_mn_blocks`.
  So substituting `num_mn_blocks` for `num_blocks` and `mn_block_idx` for `next_block_idx` is an
  identity in both cases.
- Edit 5 is a `static_assert` only — it emits no code, and its condition is true for every
  instantiation that exists today (`kSplitKFactor` defaults to 1 everywhere except
  `sm120_fp8_fp4_gemm_1d1d.cuh`, which is `Normal`).
- No other `GemmType` branch is touched.

## No-regression check

The change must be inert at the default. Four steps, **all four required** — skipping step 3
leaves a check that cannot fail, which is worthless:

1. compile TU 1 (sm100 `Normal`) before vs after → must be identical
2. compile TU 2 (sm90 `MGroupedContiguous`) before vs after → must be identical
3. sensitivity probe → must **differ**, proving steps 1-2 can detect a change at all
4. `Batched` + `kSplitKFactor > 1` → must **fail to compile** (touchpoint 5)

**Never use a bare `cmp` for steps 1-3.** See the caveat below.

**Caveat — nvcc is not byte-deterministic.** nvcc 13.1 embeds a per-compilation cookie
(`_sm100_cu_<hash>_<8 digits>`, a monotonically increasing timestamp) into `.shstrtab` /
`.strtab` for anonymous-namespace mangling. Two compiles of a *byte-identical, unmodified*
source produce cubins that differ at those offsets. A bare `cmp` of two cubins therefore
**always** reports a difference and proves nothing. Normalize the cookie first, or compare the
`.text` section directly. Both are lossless — nothing but the cookie is masked.

Build a pristine include tree, compile the same TU against each, compare:

```bash
# pristine "before" tree
mkdir -p /tmp/beforetree && cp -r deep_gemm/include /tmp/beforetree/include
git show <upstream-base>:deep_gemm/include/deep_gemm/scheduler/gemm.cuh \
  > /tmp/beforetree/include/deep_gemm/scheduler/gemm.cuh

FLAGS="-std=c++20 -cubin --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress 177,550"
CUTLASS="-Ithird-party/cutlass/include -I/usr/local/cuda-13.1/include/cccl"

# ... for each TU below, compile twice:
/usr/local/cuda-13.1/bin/nvcc $FLAGS --gpu-architecture=<arch> \
  -I/tmp/beforetree/include $CUTLASS -o /tmp/before.cubin /tmp/tu.cu
/usr/local/cuda-13.1/bin/nvcc $FLAGS --gpu-architecture=<arch> \
  -Ideep_gemm/include        $CUTLASS -o /tmp/after.cubin  /tmp/tu.cu

# compare with the build cookie normalized
python3 -c "
import hashlib, re, sys
pat = re.compile(rb'_sm[0-9]+_cu_[0-9a-f]{8}_[0-9]+')
h = [hashlib.sha256(pat.sub(b'_ID_', open(f,'rb').read())).hexdigest() for f in sys.argv[1:]]
print('IDENTICAL' if h[0] == h[1] else 'REGRESSION: cubins differ')
" /tmp/before.cubin /tmp/after.cubin
```

**TU 1 — sm100, `GemmType::Normal`** (`--gpu-architecture=sm_100a`):

```cpp
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
using namespace deep_gemm;
static void f(){auto p=reinterpret_cast<void*>(&sm100_bf16_gemm_impl<
    cute::UMMA::Major::K, cute::UMMA::Major::K,
    0, 4096, 7168,
    128, 128, 64,
    1,
    128, 128, 128,
    5,
    128, 128,
    1, false,
    148,
    128,
    false, false,
    GemmType::Normal, false,
    cutlass::bfloat16_t,
    epilogue::transform::EpilogueIdentity,
    100
>);(void)p;}
```

**TU 2 — sm90, `GemmType::MGroupedContiguous`** (`--gpu-architecture=sm_90a`). This is the case
edit 4 changes only by the `num_blocks == num_mn_blocks` identity, and the only arch where
`is_peer_cta_alive` and the `#if __CUDA_ARCH__ < 1000` multicast fixup are live, so it must be
checked too:

```cpp
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>
using namespace deep_gemm;
static void f(){auto p=reinterpret_cast<void*>(&sm90_bf16_gemm_impl<
    cute::UMMA::Major::K, cute::UMMA::Major::K,
    0, 4096, 7168,
    4,
    128, 128, 64,
    128, 128, 128,
    4,
    128, 256,
    2, false,
    132,
    GemmType::MGroupedContiguous, false,
    cutlass::bfloat16_t
>);(void)p;}
```

**Step 3 (REQUIRED) — sensitivity probe.** An `IDENTICAL` from steps 1-2 means nothing until you
have shown the comparison can report a difference. Perturb the header so the split-K path goes
live, recompile TU 1, and confirm:

```bash
sed -i 's/uint32_t kSplitKFactor = 1>/uint32_t kSplitKFactor = 2>/' \
  deep_gemm/include/deep_gemm/scheduler/gemm.cuh
# recompile TU 1 -> must now report "REGRESSION: cubins differ" (the cubin also grows)
sed -i 's/uint32_t kSplitKFactor = 2>/uint32_t kSplitKFactor = 1>/' \
  deep_gemm/include/deep_gemm/scheduler/gemm.cuh
```

If the probe reports `IDENTICAL`, the comparison is broken — fix it before trusting steps 1-2.

**Step 4 (REQUIRED) — the `Batched` guard still fires.** Compile this in a scratch dir (do not
add it to the gate); it must fail with the touchpoint-5 message:

```cpp
#include <deep_gemm/scheduler/gemm.cuh>
using namespace deep_gemm;
static void f(){
    using S = sched::Scheduler<GemmType::Batched, 128, 128, 4, 1, false, 148,
                               true, 128u, 128u,
                               sched::get_num_1d_blocks_per_group<GemmType::Batched,128,128,148,false>(),
                               /*kSplitKFactor=*/2>;
    static_assert(sizeof(S) > 0);
}
```

Re-run it with `kSplitKFactor = 1` as a negative control: that **must** still compile, or the
assert is over-broad and has broken every existing `Batched` kernel.

Recorded result at the time of the Task 5 commit, on nvcc 13.1 / `CUDA_HOME=/usr/local/cuda-13.1`.
`before` is the pre-Task-5 tree (`5d588d0`), `after` is the tree with touchpoints 1-5 applied:

| Step | TU | arch | before | after | verdict |
|---|---|---|---|---|---|
| 1 | sm100 `Normal` | `sm_100a` | 44320 B | 44320 B | identical (`.text` sha256 `0d9cdd61…`, normalized file `98e737e3…`) |
| 2 | sm90 `MGroupedContiguous` | `sm_90a` | 22096 B | 22096 B | identical (normalized file `c9f07012…`; `cuobjdump -sass` textually identical) |
| 3 | sensitivity probe (`kSplitKFactor = 2`) | `sm_100a` | 44320 B | 44408 B | **differs**, as required |
| 4 | `Batched` + `kSplitKFactor = 2` | `sm_120a` | — | — | **compile error**, as required (exit 1) |
| 4 | `Batched` + `kSplitKFactor = 1` (control) | `sm_120a` | — | — | compiles, exit 0 |

Two same-source control compiles (no edit at all) differed from each other only in the cookie,
confirming the cookie is build nondeterminism rather than an effect of the change.

These hashes are specific to nvcc 13.1 and to the `_ID_` normalization token used in the
snippet above; they will change with any toolchain update. Compare before-vs-after within a
single run rather than against these recorded values.

### Host-side no-regression check — and its false-green trap

Any change under `csrc/**` (for example the `config.hpp` touchpoint below) must be shown not to
break the sm90/sm100 host build:

```bash
touch csrc/python_api.cpp   # REQUIRED -- see below
CUDA_HOME=/usr/local/cuda-13.1 python setup.py build_ext --inplace
```

**`setup.py build_ext --inplace` on its own is a false green for any header-only change.**
setuptools decides what to rebuild by comparing each `.cpp` mtime against its `.o` mtime and
does **not** track header dependencies. Every host header in this repo — `heuristics/*.hpp`,
`impls/*.hpp`, `utils/*.hpp` — reaches the extension only by inclusion from the single TU
`csrc/python_api.cpp`. So editing a header rebuilds **nothing**: the command prints an
8-line log containing no compiler invocation at all and **exits 0**.

Confirm the run was real before believing it. A genuine rebuild shows two
`aarch64-linux-gnu-g++` invocations (compile, then link), a `copying build/... -> deep_gemm`
line, and a freshly-bumped mtime on
`build/temp.linux-*/csrc/python_api.o`. An exit code of 0 alone proves nothing.

## csrc/jit_kernels/heuristics/config.hpp

Shared by sm90, sm100 and sm120. **All three fields are defaulted**, so every existing
construction site — including the designated-initializer `GemmDesc{...}` / `GemmConfig{...}`
aggregates in `csrc/jit_kernels/impls/sm{90,100}_*.hpp` and in
`heuristics/common.hpp::get_best_config` — is unaffected and needs no edit.

| # | Anchor | Edit | Rationale |
|---|---|---|---|
| 1 | `struct GemmDesc`, after `compiled_dims`, before `ensure_zero_padding` | add `int max_gran_k = 128;` | SF granularity `max(gran_k_a, gran_k_b)`. `SM120ArchSpec::get_split_k_factor` needs it to size the SF tile (`kSFTileKBlocks = 4 * max_gran_k / block_k`) so each K partition starts on an SF-aligned boundary. sm90/sm100 never read it. |
| 2 | `struct GemmDesc`, immediately after #1 | add `bool cd_n_contiguous = true;` | False for AB-swap output (transposed, `stride_cd_n != 1`), which the TMA-store epilogue cannot express. `SM120ArchSpec::get_storage_config` reads it to force `swizzle_cd_mode = 0`, selecting the strided-store epilogue. sm90/sm100 never read it. |
| 3 | `struct GemmConfig`, after `launch_config` | add `int split_k_factor = 1;` | Number of K partitions. Set by `impls/sm120_fp8_fp4_gemm_1d1d.hpp` *after* `get_best_config` returns, and read by the kernel-arg builder and the split-K reduce launch. |

**Placement of #1/#2 is deliberate.** Both go on `GemmDesc`, not `Layout`, and in this order —
matching `nv_dev`'s declaration order. Two reasons: `SM120ArchSpec` reads them as
`desc.max_gran_k` / `desc.cd_n_contiguous`, and C++20 requires designated initializers to appear
in declaration order, so `nv_dev`'s `GemmDesc{ ..., .max_gran_k = ..., .cd_n_contiguous = ... }`
sites in `impls/sm120_fp8_fp4_gemm_1d1d.hpp` port across unchanged. `cd_n_contiguous` could
**not** live on `Layout`: `Layout` values are produced by `ArchSpec::get_layout_candidates`, never
supplied by the caller, so a caller has no way to set it there and the field would be dead.

**#3 is deliberately absent from `GemmConfig::operator<<`.** `nv_dev` prints it; we do not.
`DG_PRINT_CONFIGS` dumps the config from inside `get_best_config`, which returns *before* the
sm120 impl assigns `split_k_factor` — printing it there would always show the default `1` and
mislead. Keeping it out also keeps the diff to three added lines total.

**If upstream adds a field to `PipelineConfig`, `StorageConfig`, `LaunchConfig` or `LayoutInfo`,
`SM120ArchSpec` must set it.** Those four are returned by brace-init from
`heuristics/sm120.hpp`; a field upstream adds but sm120 does not initialize is left
indeterminate. This already happened once: `dev` added `PipelineConfig::num_tma_store_stages`
after `nv_dev` forked, and `SM120ArchSpec::get_pipeline_config` now sets it explicitly to `2`
(sm100's non-k-grouped value; sm120 has no k-grouped TMA-store special case).

### Deliberate divergence from `nv_dev`: `SM120ArchSpec::get_split_k_factor`

**Do not re-sync this function from `nv_dev`.** Upstream's version establishes an SF-alignment
invariant with a search loop — `num_k_blocks % split_k == 0` and
`(num_k_blocks / split_k) % kSFTileKBlocks == 0`, so every K partition starts on an SF tile
boundary — and then **breaks it**: the two `std::min` clamps that follow (minimum SF tiles per
partition, and the 32 MB workspace cap) can lower `split_k` onto a value that no longer
satisfies it, with nothing re-checking. Example: `num_k_blocks = 10`, `kSFTileKBlocks = 2` —
the search lands on 5, the first clamp yields `min(5, 10 / 4) = 2`, and `10 / 2 = 5` is not a
multiple of 2, so partition 1 starts mid-SF-tile.

This branch hoists the invariant into a local `is_sf_aligned` lambda (used by both the search
and a post-clamp re-check, so the two cannot drift), walks `split_k` back down after clamping,
and closes with `DG_HOST_ASSERT(is_sf_aligned(split_k))`. Walking down is monotonically safe:
a smaller factor can only cost performance, and `1` is always valid.

Diverging here is free — `heuristics/sm120.hpp` is **Category A** (sm120-exclusive, upstream
never opens it), so unlike the `scheduler/gemm.cuh` situation there is no rebase surface to
protect. Measured over a 2,007,040-point parameter sweep: violations `6594 -> 0`, and exactly
those 6594 points changed their selected `split_k` (all to `1`); no other point moved, and the
maximum selected factor is unchanged at 64.

### Related non-touchpoint: `get_byte_addressable_element_size`

`nv_dev` defined this in the shared `heuristics/common.hpp`; `dev` has no such function.
Rather than re-add it to a shared file, it is a static member of `SM120ArchSpec` — zero
upstream-owned surface, so it is **not** a touchpoint. Note that
`dev` also added `MmaKind::MXF4`, `GemmDesc::is_mxf4_mma()` and `GemmDesc::get_smem_pack_factor()`
after `nv_dev` forked, and `GemmDesc::get_mma_kind()` now returns `MXF4` for FP4 x FP4 where
`nv_dev` returned `MXFP8FP4`. SM120 must **not** adopt sm100's `get_smem_pack_factor` packing:
`SM120ArchSpec::get_smem_bytes_per_k` already halves packed FP4, and SM120 loads FP4 through its
own `.b4x16_p64` padded-SMEM path. See the comment on that function in `heuristics/sm120.hpp`.

## Gate

`AI/tools/check_sm120_host.sh` compiles every header listed in `AI/tools/sm120_host_headers.txt`
as a standalone TU against torch + DeepJIT + CUTLASS (`-fsyntax-only`, no GPU). It covers the
host side — `heuristics/sm120.hpp` and, from Task 8 on, `impls/sm120_*.hpp`. Because
`SM120ArchSpec` is a plain struct rather than a template, its member bodies are fully type-checked
by the bare `#include`. The gate does **not** instantiate `get_best_config<SM120ArchSpec>`, so it
alone does not prove the six-member ArchSpec concept is satisfied; that is covered from Task 8 on,
when `impls/sm120_fp8_fp4_gemm_1d1d.hpp` calls it and enters the gate list.

`AI/tools/check_sm120.sh` compiles `AI/tools/sm120_tu/sched_splitk.cu`, which instantiates
`Scheduler` with a trailing `kSplitKFactor = 4`. If a rebase drops edit 1 or reorders the
template parameters, that entry fails with "too many arguments for class template".
The gate does **not** cover inertness at factor 1, nor touchpoint 5's `Batched` guard — those
are steps 1-4 of the no-regression check above, which must be run by hand on rebase.

## Fallback

If upstream churns `get_next_block` badly enough that edits 3/4 no longer apply cleanly, fork to
`scheduler/sm120_gemm.cuh` and drop this touchpoint entirely (design doc section 5.2). That
removes all five edits at once and eliminates this file's Category B status.

# Category B, part 2 — the `arch_major == 12` dispatch (Task 13)

This is where the port becomes **reachable**: before this commit nothing `#include`d the sm120
headers at all. Everything below lives in an upstream-owned file, so every row is permanent
rebase surface. Re-apply each row after a rebase, then rerun both gates and the no-regression
check above.

## How to re-derive this list after a rebase

```bash
for f in gemm attention einsum hyperconnection layout; do
  echo "=== $f"; git show <upstream-sm120-ref>:csrc/apis/$f.hpp | grep -nE "arch_major == 12|== 12\)"
done
git show <upstream-sm120-ref>:csrc/utils/layout.hpp | grep -nE "arch_major == 12|== 12\)"
```

Against `origin/nv_dev` that prints **9 / 11 / 7 / 1 / 5** for the api headers and **3** for
`csrc/utils/layout.hpp` — 36 upstream sites. This branch realizes them in **37** edit points,
because `dev` has diverged from `nv_dev` in several places (see "Deviations from nv_dev" below).

**The second alternation branch `== 12\)` is load-bearing — do not simplify this pattern to
just `arch_major == 12`.** `nv_dev` writes one site as
`const int block_kv = (device_runtime->get_arch_major() == 12) ? 128 : 256;`
(`nv_dev:csrc/apis/attention.hpp:157`), where the `== 12` is attached to the *call* rather than
to a local named `arch_major`. The narrow pattern silently misses it and prints
**9 / 10 / 7 / 1 / 5 = 32**, one short — which makes every count quoted in this document fail
to reconcile and makes the whole enumeration look unverifiable. Verified against
`origin/nv_dev`: the corrected pattern prints 9 / 11 / 7 / 1 / 5 = 33, the narrow one 32.

## Include discipline

Exactly **one** line is added to each api header that needs it:

```cpp
#include "sm120_dispatch.hpp"
```

Do **not** add the five individual `impls/sm120_*.hpp` includes that `nv_dev` adds — the
dispatch header (Category A, Task 12) pulls them in. That is what holds each api header's
include delta to a single line.

`csrc/apis/layout.hpp` and `csrc/utils/layout.hpp` get **no** include: `sm120_dispatch.hpp`
itself includes `csrc/apis/layout.hpp`, so adding it there would be a cycle. This is also why
the SM120 K-major SF pre-transform (row 4 of the `apis/layout.hpp` table) must be inlined
rather than extracted.

## csrc/apis/gemm.hpp — 1 include + 9 dispatch arms

| # | Line | Enclosing function | Edit |
|---|---|---|---|
| 1 | 14 | (include block, after `#include "layout.hpp"`) | add `#include "sm120_dispatch.hpp"` |
| 2 | 113-118 | `fp8_fp4_gemm_nt` | insert `if (arch_major == 12) { ... return; }` **before** the shared `transform_sf_pair_into_required_layout` call: assert `not alpha.has_value()`, then `sm120::fp8_fp4_gemm_nt(a, b, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, major_a, major_b, m, n, k)` |
| 3 | 248-256 | `m_grouped_fp8_fp4_gemm_nt_contiguous` | `} else if (arch_major == 12 and sfa.scalar_type() == torch::kInt) {` — `sm120::to_k_major(b.first, major_b, n)`, `nv_dev`'s `is_mixed_fp4` / `k % 128` guard, then `sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d(..., major_a, cute::UMMA::Major::K, compiled_dims, use_psum_layout, expected_m_for_psum_layout)` |
| 4 | 321-324 | `m_grouped_fp8_fp4_gemm_nt_masked` | `} else if (arch_major == 12 and sfa.scalar_type() == torch::kInt) {` then `sm120_m_grouped_fp8_fp4_gemm_masked_1d1d(...)` |
| 5 | 376-387 | `k_grouped_fp8_gemm_tn_contiguous` | `} else if (arch_major == 12) {` — assert `not use_psum_layout` and `ks_cpu` non-empty; run the two `transform_k_grouped_sf_into_required_layout` calls inside the arm (upstream keeps them inside the arch-10 arm); allocate the tensormap buffer; call `sm120_k_grouped_fp8_fp4_gemm_1d1d(a.first.t().contiguous(), sfa, b.first.t().contiguous(), sfb, ..., gran_k, gran_k, K, K, compiled_dims, /*k_grouped_constant_stride=*/true, sum_k)` |
| 6 | 441-446 | `k_grouped_fp8_gemm_nt_contiguous` | `} else if (arch_major == 12) {` then `sm120_k_grouped_fp8_fp4_gemm_1d1d(..., std::get<2>(recipe), std::get<2>(recipe), K, K, compiled_dims)` |
| 7 | 546-550 | `bf16_gemm_nt` | `} else if (arch_major == 12) {` — assert `not alpha.has_value()`, then `sm120_bf16_gemm(sm120::to_k_major(a, major_a, m), sm120::to_k_major(b, major_b, n), c, d, m, n, k, K, K, compiled_dims)` |
| 8 | 632-636 | `m_grouped_bf16_gemm_nt_contiguous` | `} else if (arch_major == 12) {` then `sm120_m_grouped_bf16_gemm_contiguous(...)` (no `ensure_zero_padding` parameter) |
| 9 | 683-686 | `m_grouped_bf16_gemm_nt_masked` | `} else if (arch_major == 12) {` then `sm120_m_grouped_bf16_gemm_masked(...)` |
| 10 | 731-736 | `k_grouped_bf16_gemm_tn_contiguous` | `} else if (arch_major == 12) {` — assert `c.has_value()` and `not use_psum_layout` and `ks_cpu` non-empty, then `sm120_bf16_k_grouped_gemm(a, b, c, d, m, n, ks_cpu.value(), grouped_layout, MN, MN, compiled_dims)` |

Rows 2 and 7 carry an extra `DG_HOST_ASSERT(not alpha.has_value())` that `nv_dev` does not have:
`dev` added an `alpha` epilogue parameter to `fp8_fp4_gemm_nt` / `bf16_gemm_nt` after `nv_dev`
forked, and no sm120 launcher accepts it. Without the assert an `alpha` would be silently dropped.

**Row 2 must stay above the SF transform.** SM120 decides the AB-swap first, and the swap
changes *which* SF tensor is transformed as A and which as B, so it cannot reuse the shared
transform that the sm90/sm100 path performs beforehand. `nv_dev` achieves the same by wrapping
the whole sm90/sm100 body in `if (arch_major == 9 or arch_major == 10) { ... }`; the early-return
form used here is behaviourally identical and reindents nothing.

**Our `k_grouped_fp4_gemm_nt_contiguous` gets no arm.** That entry point does not exist in
`nv_dev`, so there is no upstream sm120 arm to port. It stays SM100-only.

## csrc/apis/attention.hpp — 1 include + 4 arms + 4 predicate widenings + 2 constants

| # | Line | Enclosing function | Edit |
|---|---|---|---|
| 1 | 18 | (include block, after `#include "layout.hpp"`) | add `#include "sm120_dispatch.hpp"` |
| 2 | 74-77 | `fp8_gemm_nt_skip_head_mid` | `} else if (arch_major == 12 and sfa.scalar_type() == torch::kInt) {` then `sm120_fp8_fp4_gemm_1d1d(..., 128, 128, major_a, major_b, compiled_dims, epilogue_type)` |
| 3 | 107 | `fp8_fp4_mqa_logits` (SF-Q check) | `arch_major == 10` becomes `arch_major == 10 or (arch_major == 12 and is_fp4)` |
| 4 | 146-147 | `fp8_fp4_mqa_logits` | `constexpr int block_kv = 256;` becomes `const int block_kv = (arch_major == 12) ? sm120::kMqaBlockKv : 256;` (plus one comment line) |
| 5 | 181-192 | `fp8_fp4_mqa_logits` (dispatch) | `} else if (arch_major == 12) {` — assert `not clean_logits`, `not schedule_meta.has_value()`, qk dtype, `num_heads` in {16,32,64}, FP32 weights; then `sm120_mqa_logits(..., is_mx_sf, qk_dtype)` |
| 6 | 407-416 | `get_paged_mqa_logits_metadata` | insert `if (arch_major == 12) { ... } else if (is_varlen) {` **as the first branch** — assert `block_kv` in {32,64} and the varlen invariants; derive `next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1`; call `sm120_paged_mqa_logits_metadata(..., (next_n + next_n_atom - 1) / next_n_atom, is_varlen, indices_ptr_or_nullptr)` |
| 7 | 470 | `fp8_fp4_paged_mqa_logits` (SF-Q check) | `arch_major == 10` becomes `arch_major == 10 or (arch_major == 12 and is_fp4)` |
| 8 | 480-483 | `fp8_fp4_paged_mqa_logits` (fused KV cache check) | add `or (arch_major == 12 and ((is_fp4 and (block_kv == 32 or block_kv == 64)) or (not is_fp4 and block_kv == 64)))` |
| 9 | 523 | `fp8_fp4_paged_mqa_logits` (indices check) | `arch_major == 10` becomes `(arch_major == 10 or arch_major == 12)` |
| 10 | 546-547 | `fp8_fp4_paged_mqa_logits` | `constexpr int split_kv = 256;` becomes `const int split_kv = (arch_major == 12) ? sm120::kPagedSplitKv : 256;` (plus one comment line) |
| 11 | 570-578 | `fp8_fp4_paged_mqa_logits` (dispatch) | `} else if (arch_major == 12) {` — assert qk dtype, `num_heads` in {16,32,64}, FP32 weights; then `sm120_paged_mqa_logits(..., split_kv, is_mx_sf, qk_dtype)` |

**Row 6 must be the first branch, and this is a deliberate divergence from `nv_dev`'s ordering.**
`nv_dev` tests `is_varlen` only to *validate* and then dispatches on arch (10 / 9 / 12); `dev`
restructured the function so the `is_varlen` branch itself *dispatches*, unconditionally to
SM100. Leaving the sm120 arm at the end would let a varlen arch-12 call fall into the SM100
branch. Placing it first keeps the arch-9 and arch-10 behaviour bit-for-bit unchanged (both
branches are still reached under exactly the same conditions), and the varlen invariants that
`nv_dev` asserts at its line 242 are folded into this arm instead of existing as a separate
widening — which is why this table has 10 edit rows for `nv_dev`'s 11 sites.

**Row 5's `not clean_logits` assert has no `nv_dev` counterpart.** `nv_dev` cleans logits with a
standalone `smxx_clean_logits` kernel after dispatch; `dev` deleted that kernel and fused
cleaning into the sm90/sm100 kernels. The sm120 kernels have no fused cleaning (Task 10
confirmed exactly one `CUTLASS_GLOBAL` per sm120 mqa `.cuh`, so there is no clean entry point),
so the only honest option is to refuse the request rather than silently return dirty logits.
`schedule_meta` is refused for the same reason — `sm120_mqa_logits` has no such parameter.

## csrc/apis/einsum.hpp — 1 include + 4 arms + 1 swap predicate + 2 guard widenings + 2 K-major coercions

| # | Line | Enclosing function | Edit |
|---|---|---|---|
| 1 | 19 | (include block) | add `#include "sm120_dispatch.hpp"` |
| 2 | 54-55 | `bmk_bnk_mn` | `} else if (arch_major == 12) { sm120_bmn_bnk_mn_gemm(a, b, d, s, m, n, k); }` — inserted **before** the arch-10 arm, matching `nv_dev`'s ordering |
| 3 | 79-80 | `bhr_hdr_bhd` | `} else if (arch_major == 12) { sm120_bf16_bhr_hdr_bhd(...); }` — before the arch-10 arm |
| 4 | 104-105 | `bhd_hdr_bhr` | `} else if (arch_major == 12) { sm120_bf16_bhd_hdr_bhr(...); }` — before the arch-10 arm |
| 5 | 210-215 | `fp8_bmm` | insert, **after** `early_return` and **before** the SF transform: `if (sm120::bmm_swap_ab_eligible(m, major_a, major_b, d_tensor, c.has_value())) { sm120::fp8_fp4_bmm_swapped(a, sfa, b, sfb, c, d_tensor, batch_size, m, n, k, major_a, major_b, compiled_dims, recipe); return; }` |
| 6 | 223-225 | `fp8_bmm` (dispatch) | `if (arch_major == 12) { sm120_fp8_fp4_bmm(a, transformed_sfa, b, transformed_sfb, c, d_tensor, batch_size, m, n, k, gran_k_a, gran_k_b, major_a, major_b, compiled_dims); } else if (arch_major == 10) {` — 12 before 10, matching `nv_dev` |
| 7 | 260 | `fp8_einsum`, `"bhd,hdr->bhr"` | `and arch_major == 10` becomes `and (arch_major == 10 or arch_major == 12)` |
| 8 | 266 | `fp8_einsum`, `"bhd,hdr->bhr"` | `perm_b` becomes `arch_major == 12 ? b.first.permute({0, 2, 1}).contiguous() : b.first.permute({0, 2, 1})` |
| 9 | 271 | `fp8_einsum`, `"bhd,bhr->hdr"` | `and arch_major == 10` becomes `and (arch_major == 10 or arch_major == 12)` |
| 10 | 274-277 | `fp8_einsum`, `"bhd,bhr->hdr"` | same ternary `.contiguous()` treatment for **both** `perm_a` and `perm_b` |

Rows 7 and 9 have no `nv_dev` counterpart: `nv_dev` matches those two expressions
unconditionally, while `dev` added an `and arch_major == 10` guard after the fork. Without the
widening, rows 8 and 10 would be dead code and arch 12 would hit `DG_HOST_UNREACHABLE`.

Rows 8 and 10 exist because SM120's MMA consumes K-major operands: after the permute these
tensors are MN-major, and the MN-major path is a ~3x-slower scalar fallback (A MN-major is not
supported at all). `nv_dev` writes these as `if (arch_major == 12) { perm_x = perm_x.contiguous(); }`
over non-const locals; the ternary form here keeps the locals `const` as `dev` declares them.

**Row 5 takes the operands unswapped.** `sm120::fp8_fp4_bmm_swapped` performs the A/B swap
*internally*, because the swap must precede the single SF transform; it also derives
`gran_k_a`/`gran_k_b` itself, so it takes neither those nor `recipe_a`/`recipe_b`.
`sm120::bmm_swap_ab_eligible` tests `arch_major == 12` internally — **do not** duplicate that
test at the call site. An FP8 `(d, sfd)` output pair cannot reach either row: the pre-existing
`DG_HOST_ASSERT(jit->device.get_arch_major() == 10 ...)` in the `sfd.has_value()` block fires
first, which is correct since no sm120 launcher accepts an SFD.

## csrc/apis/hyperconnection.hpp — 1 include + 1 arm

| # | Line | Enclosing function | Edit |
|---|---|---|---|
| 1 | 7 | (include block) | add `#include "sm120_dispatch.hpp"` |
| 2 | 48-51 | `tf32_hc_prenorm_gemm` | insert `if (arch_major == 12) { sm120_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1); } else if (arch_major == 9) {` — **12 before 9**, matching `nv_dev`'s ordering |

## csrc/apis/layout.hpp — 4 predicate widenings + 1 inlined body (no include)

| # | Line | Enclosing function | Edit |
|---|---|---|---|
| 1 | 45-46 | `transform_sf_into_required_layout` | `(FP32, x, gran_k)` path: `arch_major == 10` becomes `(arch_major == 10 or arch_major == 12)`; comment `SM100` becomes `SM100/SM120` |
| 2 | 53-54 | `transform_sf_into_required_layout` | `(INT, 1, gran_k)` path: same widening and comment update |
| 3 | 100-101 | `transform_k_grouped_sf_into_required_layout` | arch/granularity gate: `(arch_major == 10 and ...)` becomes `((arch_major == 10 or arch_major == 12) and ...)` |
| 4 | 107-126 | `transform_k_grouped_sf_into_required_layout` | FP32 path: widen the predicate **and inline the SM120-only K-major SF pre-transform** — `sf_contiguous`, the `expected_sf_k` loop over `ks_cpu`, and the conditional `.t().contiguous()`; the packer is then called with `sf_input` instead of `sf` |
| 5 | 128-129 | `transform_k_grouped_sf_into_required_layout` | INT pre-packed path: `arch_major == 10` becomes `(arch_major == 10 or arch_major == 12)`; the `gran_k == 32` restriction is **kept** |

**Row 4 is a real body, not a widening**, and it is the one place in this task where extraction
into `sm120_dispatch.hpp` is impossible: that header already includes this one (its line 27), so
moving the block there would be a circular include. It is permanent Category B surface: the
`if (arch_major == 12)` block itself is 11 lines (ours line 114-124), plus the `sf_input`
local it needs and the comment explaining why it is stranded here. It exists because SM120 also accepts K-major operands, whose SF tensor is
`[mn, sf_k]` while the common packer consumes `[sf_k, mn]`; the `expected_sf_k` sum decides
which orientation was handed in.

**Row 3 has no `nv_dev` counterpart.** `dev` added this hard arch/granularity/alignment assert
after the fork; without widening it, every arch-12 k-grouped call would abort before reaching
row 4.

**Row 3 retains a second strictness, also deliberately.** The widened assert keeps `dev`'s
`k_alignment % 128 == 0` where `nv_dev` requires only `k_alignment % 32 == 0`
(`nv_dev:csrc/apis/layout.hpp:103`), so arch 12 is stricter here than upstream. Loosening it
would change the **SM100** path too, which this task must not do. In practice the retention is
unreachable: every caller of `transform_k_grouped_sf_into_required_layout` already guarantees a
multiple of 128 before calling — `csrc/apis/gemm.hpp:347` and `:705` assert
`k_alignment % 128 == 0`, `:471` asserts `% 256 == 0`, and `:427` / `:428` pass the literal
`128`. No call can therefore produce a `k_alignment` that `nv_dev` would accept and this branch
rejects. If a future caller passes a 32- or 64-aligned value, loosen it for **both** arches in
one deliberate change rather than special-casing arch 12.

**Row 5 deliberately keeps `gran_k == 32`, where `nv_dev` has no granularity restriction on this
path.** Dropping it would change the SM100 path too, and this task must not alter sm90/sm100
behaviour. An sm120 INT (pre-packed UE8M0) k-grouped SF at `gran_k == 128` therefore still
falls through to `DG_HOST_UNREACHABLE` — exactly as it does on SM100 today. If sm120 hardware
later needs that case, drop the restriction for **both** arches in one deliberate change.

## csrc/utils/layout.hpp — 3 predicate widenings (no include)

**These three are missing from the Task 13 design/brief, which scoped the task to
`csrc/apis/*.hpp`. They are required for the dispatch to function**, and `nv_dev` carries all
three at its lines 58 / 67 / 80.

| # | Line | Enclosing function | Edit |
|---|---|---|---|
| 1 | 64 | `check_ab_fp8_fp4` | `DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and arch_major == 10)` becomes `... and (arch_major == 10 or arch_major == 12))` |
| 2 | 73 | `check_grouped_ab_fp8_fp4` | same widening |
| 3 | 86 | `get_default_recipe` | `} else if (arch_major == 10) {` becomes `} else if (arch_major == 10 or arch_major == 12) {` |

Row 3 is load-bearing, and its blast radius is wider than `sm120_dispatch.hpp`. **Three**
callers of `get_default_recipe` are reachable from an arch-12 request:

| Caller | Reached by |
|---|---|
| `csrc/apis/sm120_dispatch.hpp:111` (`sm120::fp8_fp4_gemm_nt`) | every default-recipe `fp8_fp4_gemm_nt` / `_nn` / `_tn` / `_tt` call on arch 12 |
| `csrc/apis/sm120_dispatch.hpp:182` (`sm120::fp8_fp4_bmm_swapped`) | the small-M AB-swapped batched path |
| **`csrc/apis/layout.hpp:73`** (`transform_sf_pair_into_required_layout`) | **every** default-recipe arch-12 m-grouped contiguous and m-grouped masked GEMM, plus the non-swapped `fp8_bmm` arm in `einsum.hpp` |

The third is the one easiest to overlook: it is not in a file named `sm120_*`, and it is
reached indirectly, because `transform_sf_pair_into_required_layout` fills in a default recipe
for any caller that passes neither `recipe` nor `recipe_a`. Without the widening all three
abort with `DG_HOST_UNREACHABLE("Unknown recipe")`.

Rows 1 and 2 gate packed-FP4 operands; without them every FP4 shape check on arch 12 aborts,
making the sm120 FP4 kernels unreachable through the public API. Row 2
(`check_grouped_ab_fp8_fp4`) is additionally reached from `csrc/apis/mega_moe.hpp:190` and
`:192`.

### Dropping these three rows fails at RUNTIME, not at compile time — nothing here catches it

This is the most dangerous rebase hazard in the whole branch, so it is worth stating flatly.
All three rows are **runtime `if`/assert conditions inside function bodies**. A rebase that
drops them:

- still **compiles**: `csrc/apis/sm120_dispatch.hpp` calls `get_default_recipe(...)` by name
  with matching argument types either way, so the call type-checks;
- still passes **`check_sm120_host.sh`**, which is `-fsyntax-only` and never evaluates a branch;
- still passes **`setup.py build_ext --inplace`**, for the same reason;
- still passes the **whole `tests/` suite on any sm90/sm100 box**, because none of those paths
  ever sets `arch_major == 12`.

The failure appears only when an SM120 device runs an sm120 kernel, as a
`DG_HOST_UNREACHABLE("Unknown recipe")` abort or a packed-FP4 `DG_HOST_ASSERT` failure. Since
this project has **no sm120 hardware**, that means a dropped row here is invisible to every
check that exists. Re-verify these three by hand after every rebase:

```bash
grep -nE "arch_major == 12|== 12\)" csrc/utils/layout.hpp   # must print 3 lines
```

The same reasoning applies to the `csrc/apis/layout.hpp` rows and to every predicate widening
in the attention table: widenings are runtime conditions and are invisible to the gates. The
build is a real type-check of the arms that are *present*; it can say nothing about arms that
are *absent*.

**How loudly a dropped edit fails varies, and a rebaser should not expect one signature.**
Three distinct behaviours:

| Dropped edit | What happens on SM120 |
|---|---|
| A dispatch arm whose `if` / `else if` chain ends in `DG_HOST_UNREACHABLE` — every arm in `gemm.hpp` and `attention.hpp`, and `einsum.hpp`'s three BF16 arms | clean `DG_HOST_UNREACHABLE("Unsupported architecture")` abort |
| **`einsum.hpp`'s `fp8_bmm` arm (row 6 of that table)** | **no clean abort.** That chain ends in an *unguarded* `else` calling `sm90_fp8_bmm` (ours, `csrc/apis/einsum.hpp:228-232`), so arch 12 silently routes into the **SM90** path and surfaces later as a JIT / device-compile failure. Loud, but not the clean abort the rest of this section promises. The code is faithful to `nv_dev` here; only the failure mode differs. |
| Either attention constant (`block_kv`, `split_kv`) | **self-detecting downstream**, not silent: `csrc/jit_kernels/impls/sm120_mqa_logits.hpp:487` and `:625` assert `split_kv == 128`, and `deep_gemm/include/deep_gemm/impls/sm120_fp8_mqa_logits.cuh:58` static-asserts `BLOCK_KV == kNumMathWarps * MMA_M`. Reverting either constant to 256 trips one of those. |

All three are still **runtime and sm120-only** — none reaches a host without SM120 silicon, so
none of them changes the conclusion above. The table exists only so a rebaser knows that the
`fp8_bmm` arm misroutes rather than aborting, and that the two attention constants are the one
category of dropped edit this port can actually detect.

## Deviations from `nv_dev`, collected

`dev` has moved since `nv_dev` forked, so some `nv_dev` sites do not exist here and some
`dev`-only sites do. Net: 36 upstream sites are realized as 37 edit points (plus the 4 one-line includes).

| Where | What `nv_dev` does | What this branch does | Why |
|---|---|---|---|
| `apis/attention.hpp` `get_paged_mqa_logits_metadata` | validates `is_varlen`, then dispatches 10 / 9 / 12 | the sm120 arm goes **first**, absorbing the varlen validation | `dev`'s `is_varlen` branch dispatches to SM100 itself; a trailing arm would be unreachable for varlen |
| `apis/attention.hpp` `fp8_fp4_mqa_logits` | cleans logits via a standalone `smxx_clean_logits` kernel | asserts `not clean_logits` in the sm120 arm | `dev` deleted `smxx_clean_logits`; sm120 kernels have no fused cleaning |
| `apis/gemm.hpp` `fp8_fp4_gemm_nt` / `bf16_gemm_nt` | no `alpha` parameter | extra `DG_HOST_ASSERT(not alpha.has_value())` | `dev` added `alpha`; no sm120 launcher accepts it |
| `apis/gemm.hpp` `fp8_fp4_gemm_nt` | wraps the sm90/sm100 body in `if (arch_major == 9 or arch_major == 10)` | early-returns on arch 12 before the SF transform | behaviourally identical, reindents nothing |
| `apis/einsum.hpp` `fp8_einsum` | matches two expressions unconditionally | widens `dev`'s `and arch_major == 10` guards | `dev`-only guard added after the fork |
| `apis/layout.hpp` k-grouped | no arch/granularity assert | widens `dev`'s assert | `dev`-only assert added after the fork |
| `apis/layout.hpp` INT k-grouped path | no `gran_k` restriction | keeps `dev`'s `gran_k == 32` | dropping it would change SM100 behaviour |
| `apis/gemm.hpp` `k_grouped_fp4_gemm_nt_contiguous` | entry point does not exist | no arm added | nothing upstream to port |

## Sites NOT wired, and why

- **`csrc/jit_kernels/heuristics/runtime.hpp`.** `nv_dev:53` returns a `{128, 64, 64}` BLOCK_M
  search spec for arch 12 inside `get_contiguous_mk_alignment`, a helper `dev` does not have —
  `dev`'s `get_theoretical_mk_alignment_for_contiguous_layout` has a different shape entirely and
  returns the legacy 128 for every arch other than 10. This is a **performance/alignment
  heuristic**, not dispatch reachability, and heuristics were Task 7's scope. Arch 12 gets the
  legacy 128 alignment, which every sm120 arm accepts. Revisit if sm120 hardware shows the
  k-grouped/contiguous alignment mattering.
- **`csrc/python_api.cpp`.** Confirmed unchanged and unchanged-needed:
  `grep -cE "sm90|sm100|sm120|arch_major" csrc/python_api.cpp` returns 0. sm120 is reached
  through the existing generic entry points, which dispatch internally on `arch_major`.

## Verification for this part

```bash
touch csrc/python_api.cpp                                       # REQUIRED, see the trap above
CUDA_HOME=/usr/local/cuda-13.1 python setup.py build_ext --inplace
CUDA_HOME=/usr/local/cuda-13.1 ./AI/tools/check_sm120_host.sh   # pass=7  fail=0
CUDA_HOME=/usr/local/cuda-13.1 ./AI/tools/check_sm120.sh        # pass=23 fail=0
```

Unlike Tasks 2-12, this task's edits are inside the single translation unit that
`csrc/python_api.cpp` actually compiles, so the build is a real syntax and type check of every
arm — including the `sm120::` and `sm120_*` call signatures.

For the runtime no-regression check, note that `tests/` are standalone scripts, **not** pytest
files (there is no pytest in this environment), and that they import `deep_gemm` from
site-packages unless told otherwise. Run them as:

```bash
ln -sfn "$PWD/third-party/cutlass/include/cutlass" deep_gemm/include/cutlass   # as develop.sh does
ln -sfn "$PWD/third-party/cutlass/include/cute"    deep_gemm/include/cute
cd tests && PYTHONPATH=.. python test_fp8_fp4.py
```

Without the two `deep_gemm/include` symlinks every JIT compile fails with
`fatal error: cute/numeric/math.hpp: No such file or directory`, which looks like a code
regression and is not one. Without `PYTHONPATH`, the scripts import a stale `deep_gemm 2.3.0`
from site-packages and fail with `ImportError: cannot import name ...`, which likewise looks
like a regression and is not one. Normalise both before comparing anything.

`tests/test_attention.py` reads **`DG_MQA_NUM_CASES`** to sample N cases per section
(`prefill` / `paged` / `sparse`) from a `random.Random` seeded per section with a fixed
constant. The same N therefore selects the **identical** case set on both sides of an A/B, so
it is safe to bound that script rather than run all 2304 + 4320 + 161 cases. Task 13 used
`DG_MQA_NUM_CASES=40`.

**Four scripts cannot run on this host at all**, for reasons unrelated to the port. Do not
record them as passes:

| Script | Why it cannot run |
|---|---|
| `tests/test_mega_gate.py` | `ModuleNotFoundError: No module named 'tile_kernels'` |
| `tests/test_mega_mhc.py` | `ModuleNotFoundError: No module named 'tilelang'` |
| `tests/test_sanitizer.py` | `ModuleNotFoundError: No module named 'tilelang'` |
| `tests/test_mega_moe.py` | spawns 8 distributed ranks; this host has 4 GPUs (`ValueError: device_id cuda:5 is out of range`) |

`test_mega_moe.py` is the only one of the four that would exercise an edited function
(`csrc/apis/mega_moe.hpp` line 190/192 calls `check_grouped_ab_fp8_fp4`, widened above). That
widening adds a disjunct and cannot change the arch-10 result, but it is **not** covered by any
runnable test here. Re-run that script on an 8-GPU host after any rebase that touches
`csrc/utils/layout.hpp`.

---

# Category B, part 3 — test gating (Task 14)

Four `tests/*.py` files are upstream-owned and gain arch-12 enumeration rows. Unlike the
dispatch edits, these rows do not change any `csrc/` behaviour: dropping one can only make the
suite enumerate a case SM120 rejects (a loud `DG_HOST_ASSERT` on sm120 hardware) or stop
enumerating a case SM120 supports (lost coverage). Neither is visible on sm90/sm100.

**Every arch-12 row restates a host assert or a `DG_STATIC_ASSERT`; those are the source of
truth.** The comment on each row names the file and condition it restates. On rebase, recheck
the assert before the enumeration.

## tests/generators.py — 2 edits

| # | Anchor | Edit |
|---|---|---|
| 1 | `QuantConfig.get_list_from_dtype`, after the `arch_major == 10` branch | add `elif get_arch_major() == 12:` appending `(32, 32, True, True)` (FP4xFP4), `(128, 32, False, True)` (FP8_A x FP4_B) and `(32, 128, True, False)` (FP4_A x FP8_B, the swapAB / `kAIsFP4` orientation) |
| 2 | `enumerate_k_grouped_contiguous` | scalar `major_a, major_b` becomes a `major_pairs` list; arch 12 + FP8 yields **both** `(MNMajor, MNMajor)` (the TN entry point) and `(KMajor, KMajor)` (the NT entry point). The body is wrapped in `for major_a, major_b in major_pairs:`, and the K-major FP8 case narrows to `gran_k == 128` and `cd_options == [(True, torch.float)]` because `k_grouped_fp8_gemm_nt_contiguous` pins `recipe == (1, 1, 128)` and requires a `c` (`csrc/apis/gemm.hpp`). |

Edit 2 is inert for every other arch: SM90 FP8 and FP4 already yielded a single K-major pair,
everything else a single MN-major pair, so the added loop has length 1 and the narrowing is a
no-op (SM90's SF-layout list is already `[(128, 128)]` and its `cd_options` already
`[(True, torch.float)]`).

Verified by A/B against the pre-edit file, with `get_arch_major` monkeypatched so all three
arches are comparable on this (arch-10) host:

| Enumerator | arch 9 | arch 10 | arch 12 |
|---|---|---|---|
| `QuantConfig.get_list_from_dtype` (FP8) | identical | identical | 1 → 4 configs |
| `enumerate_k_grouped_contiguous` (FP8) | identical | identical | 126 → 147 rows |
| `enumerate_k_grouped_contiguous` (BF16, FP4) | identical | identical | identical |
| `enumerate_normal` (FP8, BF16) | — | identical | — |

The 21 extra arch-12 k-grouped rows are exactly the new K-major/NT arm: 7 shapes × 3 SF layouts
with `gran_k == 128` × 1 `cd_option`. `enumerate_normal` was compared on this host only, where
it runs for real; it is untouched by these edits.

## tests/test_attention.py — 2 edits

| # | Anchor | Edit |
|---|---|---|
| 1 | `test_mqa_logits.enumerate_mqa_logits` | arch 12 gets `fmts = ('mxfp4', 'fp8')` (no MXFP8 kernel); `clean_logits` is forced off; FP4 `head_dims` narrows to `(128, )` |
| 2 | `test_paged_mqa_logits.enumerate_paged_mqa_logits` | varlen widens to `arch_major in (10, 12)`; arch 12 gets `fmts = ('mxfp4', 'fp8')`, `block_kvs = (32, 64)` for FP4 and `(64, )` for FP8, and FP4 `head_dims = (128, )` |

The asserts each row restates:

| Row | Restates |
|---|---|
| MXFP8 excluded on arch 12 | `DG_HOST_ASSERT(arch_major == 10 or (arch_major == 12 and is_fp4))` guarding the MX scaling factor, `csrc/apis/attention.hpp` (dense and paged) |
| `clean_logits` off on arch 12 | `DG_HOST_ASSERT(not clean_logits)` in the arch-12 dense arm — see "Deviations from `nv_dev`": this lineage deleted `smxx_clean_logits` and the sm120 kernels have no fused cleaning. **`nv_dev` does not have this restriction**; it is specific to this branch. |
| FP4 `head_dim == 128` | `DG_STATIC_ASSERT(kHeadDim == 128, "FP4 MQA only supports head_dim=128")` in `sm120_fp4_mqa_logits.cuh:61` and `sm120_fp4_paged_mqa_logits.cuh:63`, and `DG_HOST_ASSERT(head_dim == 128)` in `csrc/jit_kernels/impls/sm120_mqa_logits.hpp:249` |
| varlen on arch 12 | `DG_HOST_ASSERT((arch_major == 10 or arch_major == 12) and next_n == 1)` in the paged varlen block |
| `block_kv` on arch 12 | the `arch_major == 12` clause of the fused-KV-cache assert (`(is_fp4 and (block_kv == 32 or block_kv == 64)) or (not is_fp4 and block_kv == 64)`), plus `DG_HOST_ASSERT(block_kv == 64)` in the FP8 paged launcher |

Both edits are inert on arch 9 and 10: every ternary keeps its existing arm for those arches,
and the only non-arch-12 rewrite is `block_kvs`, whose `else` branch is `(64, )` — the literal
the original expression produced for every non-10 arch. Verified by A/B against the pre-edit
file, with `arch_major` stubbed (the enumerators are closures, so their source is lifted out by
text and exec'd — no GPU needed):

| Enumerator | arch 9 | arch 10 | arch 12 |
|---|---|---|---|
| `enumerate_mqa_logits` | 192 → 192, identical | 2304 → 2304, identical | 192 → 128 |
| `enumerate_paged_mqa_logits` | 24 → 24, identical | 4320 → 4320, identical | 24 → 120 |

The arch-12 dense count *falls* because `clean_logits` halves the case set and MXFP4 re-adds
only `head_dim == 128`; the paged count rises because varlen, MXFP4 and `block_kv == 32` are
all newly reachable.

## tests/test_fp8_fp4.py — 3 edits

| # | Anchor | Edit |
|---|---|---|
| 1 | import block | add `MajorTypeAB` to the `from generators import (...)` list |
| 2 | `test_gemm`, after `recipe, recipe_a, recipe_b = ...` | `continue` when `is_mixed_fp4 and get_arch_major() == 12 and k % 128 != 0` |
| 3 | `test_k_grouped_gemm_contiguous` | the FP8 entry point is selected **per case** from `major_a` (`select_fp8_gemm`) instead of per arch; the FP4 option becomes a `lambda` that ignores `major_a` |

Edit 2 restates `DG_HOST_ASSERT(!is_mixed_fp4 or k % 128 == 0)` in `csrc/apis/sm120_dispatch.hpp`
(and the identical assert in the arch-12 arm of `m_grouped_fp8_fp4_gemm_nt_contiguous`,
`csrc/apis/gemm.hpp`). It is reachable: the backward shapes in `enumerate_normal` put `k` at
2112 and 576, neither a multiple of 128.

Edit 3 is behaviour-preserving for arch 9 and 10. Arch 9 FP8 yields only K-major, so
`select_fp8_gemm` returns `k_grouped_fp8_gemm_nt_contiguous` exactly where the old
`arch_major == 9` test did; arch 10 FP8 yields only MN-major, so it returns the TN entry point;
the FP4 option is unchanged.

## tests/test_einsum.py — 4 edits

| # | Anchor | Edit |
|---|---|---|
| 1 | module scope, above `test_bmk_bnk_mn` | add `assert_bf16_einsum_close(z, ref_z, fp32_ref_fn=None)` |
| 2 | `test_bhr_hdr_bhd` | replace `assert calc_diff(z, ref_z) < 1e-10` with the helper, passing an FP32-truth thunk |
| 3 | `test_bhd_hdr_bhr` | same |
| 4 | `test_bhd_bhr_hdr` | same, with no thunk — `ref_z` there is already the FP32 truth |

On any arch other than 12 the helper is exactly the old `assert calc_diff(z, ref_z) < 1e-10`
and the thunk is never called, so there is no extra work and no tolerance change on sm90/sm100.
On arch 12 it relaxes the BF16-vs-BF16 comparison to `1e-7` and adds a `1e-5` check against an
FP32 reference: DeepGEMM and the cuBLAS/torch reference differ only in FP32 accumulation order,
which is a few ULPs past `1e-10`. **That tolerance choice is inherited from `nv_dev` and has
never been observed on hardware from this branch.**

## Re-deriving the test rows after a rebase

```bash
for f in tests/generators.py tests/test_attention.py tests/test_fp8_fp4.py tests/test_einsum.py; do
  printf '%-26s %s\n' "$f" \
    "$(grep -vE '^[[:space:]]*#' "$f" | grep -cE 'arch_major(\(\))? *(==|!=) *12|arch_major in \(10, 12\)')"
done
```

Expected for this branch (code lines only, comments excluded): `generators.py` **2**,
`test_attention.py` **7**, `test_fp8_fp4.py` **1**, `test_einsum.py` **1**. `test_fp8_fp4.py`
edits 1 and 3 and `test_einsum.py` edits 2-4 carry no literal `12`, so the count is a lower
bound — check the tables above as well.

## What these rows are NOT

They do not make any sm120 test runnable here. This host reports capability 10.0; every
arch-12 row above is dead code on it, and the arch-9/arch-10 enumerations are identical to their
pre-Task-14 form (verified above). **No sm120 test case in this repo has ever been executed.**

Runtime confirmation on this host (arch 10), with `PYTHONPATH` and the include symlinks set:
`test_einsum.py`, `test_attention.py` (`DG_MQA_NUM_CASES=40`), `test_fp8_fp4.py`,
`test_bf16.py`, `test_layout.py`, `test_legacy.py`, `test_hyperconnection.py` and
`test_lazy_init.py` all exit `0`. That proves no sm90/sm100 regression; it proves nothing about
arch 12.
