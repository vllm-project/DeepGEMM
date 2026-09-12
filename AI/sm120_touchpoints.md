# SM120 Category B Touchpoints

Every edit this branch makes to an **upstream-owned** file. On rebase: reapply each row,
then run `AI/tools/check_sm120.sh` and the no-regression check below.

**Rule: any change to an upstream-owned file updates this file in the same commit.**

## Inventory

| Upstream file | Edits | Added by |
|---|---|---|
| `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` | 4 | Task 5 (split-K) |

Everything else this branch adds is a **new** file (Category A: `deep_gemm/{mma,common,impls,scheduler}/sm120_*.cuh`,
`AI/tools/**`) and cannot conflict on rebase. As of this commit, `scheduler/gemm.cuh` is the
only upstream-owned file the branch modifies. Verify that claim after any rebase with:

```bash
git diff --diff-filter=M --name-only <upstream-base>...HEAD
```

Anything listed there that is not in the table above is an undocumented touchpoint — add it.

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

## Gate

`AI/tools/check_sm120.sh` compiles `AI/tools/sm120_tu/sched_splitk.cu`, which instantiates
`Scheduler` with a trailing `kSplitKFactor = 4`. If a rebase drops edit 1 or reorders the
template parameters, that entry fails with "too many arguments for class template".
The gate does **not** cover inertness at factor 1, nor touchpoint 5's `Batched` guard — those
are steps 1-4 of the no-regression check above, which must be run by hand on rebase.

## Fallback

If upstream churns `get_next_block` badly enough that edits 3/4 no longer apply cleanly, fork to
`scheduler/sm120_gemm.cuh` and drop this touchpoint entirely (design doc section 5.2). That
removes all five edits at once and eliminates this file's Category B status.
