#!/usr/bin/env bash
# Verifies the CUDA>=13 guard in deep_gemm/include/deep_gemm/common/sm120_utils.cuh:
# that it fires where it must, and stays silent everywhere else.
#
# The guard sits *before* the includes, inside the `__CUDA_ARCH__ >= 1200` block, so its
# behaviour is decidable by the preprocessor alone -- which is what makes it checkable on a
# host that has no CUDA 12.x toolkit installed (this one does not).
#
# `gcc -E` is not a stand-in for nvcc's device pass: it IS that pass. `nvcc --dryrun` for
# `-cubin --gpu-architecture=sm_120a` shows the device preprocessing step is literally
#
#   gcc -std=c++20 -D__CUDA_ARCH__=1200 ... -E -x c++ ... -D__CUDACC_VER_MAJOR__=13 ...
#
# emitting a .cpp1.ii that is then handed to cicc -- cicc never sees a preprocessor directive
# at all. So driving `gcc -E` with __CUDA_ARCH__ and __CUDACC_VER_MAJOR__ set explicitly
# reproduces any (toolkit version, compilation pass) pair exactly.
#
# This is also why `nvcc -D__CUDACC_VER_MAJOR__=12` does NOT simulate a 12.x toolkit: on that
# same command line the user's `-D "__CUDACC_VER_MAJOR__=12"` appears BEFORE nvcc's own
# `-D__CUDACC_VER_MAJOR__=13`, and the last -D wins. The guard never sees 12.
#
# Checks 1-5 are preprocessor probes. Check 6 runs a real CUDA 12.x toolkit if one exists.
# Check 7 is the sensitivity leg: it proves that an `#error` nested inside the
# `__CUDA_ARCH__ >= 1200` block really does abort a real `nvcc --gpu-architecture=sm_120a`
# compile. Without it, every check here is `gcc -E` and nothing in this repo shows the guard
# can fail the toolchain it is meant to fail.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MSG='require CUDA 13.0 or newer'
ARCH_REAL="${SM120_ARCH:-sm_120a}"
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

# --- 7. sensitivity leg: the same `#error`, in the same place, MUST be able to fail a real
#        sm_120a compile. Build a shadow include tree whose only difference is the inverted
#        comparison (`>= 13` instead of `< 13`), so the guard fires on THIS toolkit, and
#        compile the probe against it with the real nvcc. Checks 1-5 are all `gcc -E`; this is
#        the only leg that exercises the real device-compile path, and if it ever reports
#        "compiles" the guard is inert and checks 1-5 are measuring nothing.
NVCC="$CUDA_HOME/bin/nvcc"
if [ -x "$NVCC" ]; then
  mkdir -p "$WORK/shadow"
  cp -r "$REPO/deep_gemm/include/deep_gemm" "$WORK/shadow/deep_gemm"
  H="$WORK/shadow/deep_gemm/common/sm120_utils.cuh"
  sed -i 's/(__CUDACC_VER_MAJOR__ < 13)/(__CUDACC_VER_MAJOR__ >= 13)/' "$H"
  if ! grep -q '__CUDACC_VER_MAJOR__ >= 13' "$H"; then
    printf '%-52s %-6s (want %-6s) FAIL\n' "inverted-guard probe (patch did not apply)" - -
    FAIL=$((FAIL+1))
  else
    out="$("$NVCC" -std=c++20 -cubin --gpu-architecture="$ARCH_REAL" \
           -I"$WORK/shadow" -I"$REPO/third-party/cutlass/include" -I"$CUDA_HOME/include/cccl" \
           --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress 177,550 \
           -o /dev/null "$WORK/probe.cu" 2>&1)"
    if grep -qF -- "$MSG" <<<"$out"; then
      printf '%-52s %-6s (want %-6s) OK\n' "inverted guard, real nvcc $ARCH_REAL" fires fires
      PASS=$((PASS+1))
    else
      printf '%-52s %-6s (want %-6s) FAIL\n' "inverted guard, real nvcc $ARCH_REAL" silent fires
      sed -n '1,10p' <<<"$out"; FAIL=$((FAIL+1))
    fi
  fi
else
  printf '%-52s SKIPPED (no nvcc at %s)\n' "inverted-guard sensitivity probe" "$NVCC"
fi

echo "-----"; echo "pass=$PASS fail=$FAIL"
[ "$PASS" -eq 0 ] && { echo "FATAL: gate ran zero checks -- nothing was verified"; exit 2; }
[ "$FAIL" -eq 0 ]
