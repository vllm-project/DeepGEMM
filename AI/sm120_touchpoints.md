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
| 2 | state fields, after `num_n_blocks` | add `num_mn_blocks`, `split_k_idx`, `k_partition_start`, `k_partition_end` |
| 3 | constructor, `Normal or Batched` branch | hoist `num_mn_blocks = num_m_blocks * num_n_blocks;` above the `if constexpr`; `num_blocks = num_mn_blocks * kSplitKFactor` |
| 4 | `get_next_block`, final `else` branch | derive `mn_block_idx`/`split_k_idx` under `if constexpr (kSplitKFactor > 1)`; feed `mn_block_idx` to `is_peer_cta_alive` and `get_swizzled_block_idx`; bounds check stays on the raw index |

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
- No other `GemmType` branch is touched.

## No-regression check

The change must be inert at the default. Two kernels that consume `sched::Scheduler` are
compiled against the pre-change and post-change headers and compared.

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

**Confirm the check is not vacuous.** Before trusting an `IDENTICAL`, perturb the header so the
split-K path goes live and confirm the comparison reports a difference:

```bash
sed -i 's/uint32_t kSplitKFactor = 1>/uint32_t kSplitKFactor = 2>/' \
  deep_gemm/include/deep_gemm/scheduler/gemm.cuh
# recompile TU 1 -> must now report "REGRESSION: cubins differ" (cubin also grows)
sed -i 's/uint32_t kSplitKFactor = 2>/uint32_t kSplitKFactor = 1>/' \
  deep_gemm/include/deep_gemm/scheduler/gemm.cuh
```

Recorded result at the time of the Task 5 commit, on nvcc 13.1 / `CUDA_HOME=/usr/local/cuda-13.1`:

| TU | arch | before | after | verdict |
|---|---|---|---|---|
| sm100 `Normal` | `sm_100a` | 44320 B | 44320 B | identical (`.text` sha256 `0d9cdd61…`, normalized file `98e737e3…`) |
| sm90 `MGroupedContiguous` | `sm_90a` | 22096 B | 22096 B | identical (normalized file `c9f07012…`; `cuobjdump -sass` textually identical) |
| sensitivity probe (`kSplitKFactor = 2`) | `sm_100a` | 44320 B | 44408 B | **differs**, as required |

Two same-source control compiles (no edit at all) differed from each other only in the cookie,
confirming the cookie is build nondeterminism rather than an effect of the change.

These hashes are specific to nvcc 13.1 and to the `_ID_` normalization token used in the
snippet above; they will change with any toolchain update. Compare before-vs-after within a
single run rather than against these recorded values.

## Gate

`AI/tools/check_sm120.sh` compiles `AI/tools/sm120_tu/sched_splitk.cu`, which instantiates
`Scheduler` with a trailing `kSplitKFactor = 4`. If a rebase drops edit 1 or reorders the
template parameters, that entry fails with "too many arguments for class template".
The gate does **not** cover inertness at factor 1 — that is the no-regression check above.

## Fallback

If upstream churns `get_next_block` badly enough that edits 3/4 no longer apply cleanly, fork to
`scheduler/sm120_gemm.cuh` and drop this touchpoint entirely (design doc section 5.2).
