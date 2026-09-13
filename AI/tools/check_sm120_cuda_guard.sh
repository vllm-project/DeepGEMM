#!/usr/bin/env bash
# Verifies the CUDA>=13 guard in deep_gemm/include/deep_gemm/common/sm120_utils.cuh:
# that it fires where it must, and stays silent everywhere else.
#
# The guard sits *before* the includes, inside the `__CUDA_ARCH__ >= 1200` block, so its
# behaviour is decidable by the preprocessor alone -- which is what makes it checkable on a
# host that has no CUDA 12.x toolkit installed (this one does not). `gcc -E` does not
# predefine __CUDA_ARCH__ or __CUDACC_VER_MAJOR__, so both can be set exactly, simulating any
# (toolkit version, compilation pass) pair. Forcing them through real nvcc does NOT work:
# `nvcc -D__CUDACC_VER_MAJOR__=12` only redefines the macro for the host pass (with a
# redefinition warning); cicc re-predefines it in the device pass, so the guard never sees 12.
#
# If a real CUDA 12.x toolkit IS present, check 6 additionally runs it for real.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MSG='require CUDA 13.0 or newer'
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
printf '#include <deep_gemm/common/sm120_utils.cuh>\n' > "$WORK/probe.cu"
INC="-I$REPO/deep_gemm/include -I$REPO/third-party/cutlass/include -I$CUDA_HOME/include -I$CUDA_HOME/include/cccl"
PASS=0; FAIL=0

# $1 = human label, $2 = "fires"|"silent", rest = -D flags
probe() {
  local label="$1" want="$2"; shift 2
  # The guard precedes the includes, so an unrelated "No such file" from a later include is
  # expected and irrelevant -- only the presence of the guard message is asserted on.
  local out; out="$(gcc -E -x c++ "$@" $INC "$WORK/probe.cu" -o /dev/null 2>&1)"
  local got="silent"; grep -qF -- "$MSG" <<<"$out" && got="fires"
  if [ "$got" = "$want" ]; then
    printf '%-52s %-6s (want %-6s) OK\n' "$label" "$got" "$want"; PASS=$((PASS+1))
  else
    printf '%-52s %-6s (want %-6s) FAIL\n' "$label" "$got" "$want"; FAIL=$((FAIL+1))
    sed -n '1,10p' <<<"$out"
  fi
}

probe "sm120 device pass, CUDA 12.x"  fires  -D__CUDA_ARCH__=1200 -D__CUDACC_VER_MAJOR__=12
probe "sm120 device pass, CUDA 13.x"  silent -D__CUDA_ARCH__=1200 -D__CUDACC_VER_MAJOR__=13
probe "host pass, CUDA 12.x"          silent -D__CUDACC_VER_MAJOR__=12
probe "sm100 device pass, CUDA 12.x"  silent -D__CUDA_ARCH__=1000 -D__CUDACC_VER_MAJOR__=12
probe "sm90 device pass, CUDA 12.x"   silent -D__CUDA_ARCH__=900  -D__CUDACC_VER_MAJOR__=12

# --- 6. real CUDA 12.x toolkit, if one is installed
NVCC12=""
for d in /usr/local/cuda-12*; do [ -x "$d/bin/nvcc" ] && NVCC12="$d/bin/nvcc"; done
if [ -n "$NVCC12" ]; then
  # nvcc needs a real input file with a recognised extension; process substitution
  # (/dev/fd/63) is rejected, hence the temp .cu above.
  out="$("$NVCC12" -std=c++20 -cubin --gpu-architecture=sm_120a \
         -I"$REPO/deep_gemm/include" -I"$REPO/third-party/cutlass/include" \
         -o /dev/null "$WORK/probe.cu" 2>&1)"
  if grep -qF -- "$MSG" <<<"$out"; then
    printf '%-52s %-6s (want %-6s) OK\n' "real nvcc 12.x ($NVCC12)" fires fires; PASS=$((PASS+1))
  else
    printf '%-52s %-6s (want %-6s) FAIL\n' "real nvcc 12.x ($NVCC12)" silent fires
    sed -n '1,10p' <<<"$out"; FAIL=$((FAIL+1))
  fi
else
  printf '%-52s SKIPPED (no CUDA 12.x toolkit installed)\n' "real nvcc 12.x"
fi

echo "-----"; echo "pass=$PASS fail=$FAIL"
[ "$PASS" -eq 0 ] && { echo "FATAL: gate ran zero checks -- nothing was verified"; exit 2; }
[ "$FAIL" -eq 0 ]
