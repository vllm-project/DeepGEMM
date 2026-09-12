#!/usr/bin/env bash
# SM120 device compile gate. Runs on ANY machine with CUDA >= 13 -- no sm120
# hardware required. nvcc cross-compiles sm_120a from any host.
#
# Verifies, for every sm120 device header:
#   1. it compiles standalone (include-graph / API-drift check)
#   2. one known-good instantiation emits the expected MMA opcode in SASS
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_HOME/bin/nvcc"; CUOBJDUMP="$CUDA_HOME/bin/cuobjdump"
ARCH="${SM120_ARCH:-sm_120a}"
[ -x "$NVCC" ] || { echo "FATAL: no nvcc at $NVCC (set CUDA_HOME)"; exit 2; }
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
INC="-I$REPO/deep_gemm/include -I$REPO/third-party/cutlass/include -I$CUDA_HOME/include/cccl"
FLAGS="-std=c++20 -cubin --gpu-architecture=$ARCH --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress 177,550"
PASS=0; FAIL=0

# --- 1. standalone header compiles (one path per line)
HEADERS="$(cat "$REPO/AI/tools/sm120_headers.txt" 2>/dev/null || true)"
for h in $HEADERS; do
  [ -f "$REPO/deep_gemm/include/$h" ] || { printf '%-58s MISSING\n' "$h"; FAIL=$((FAIL+1)); continue; }
  echo "#include <$h>" > "$WORK/c.cu"
  if $NVCC $FLAGS $INC -o /dev/null "$WORK/c.cu" 2>"$WORK/e"; then
    printf '%-58s include OK\n' "$h"; PASS=$((PASS+1))
  else
    printf '%-58s include FAIL\n' "$h"; sed -n '1,6p' "$WORK/e"; FAIL=$((FAIL+1))
  fi
done

# --- 2. instantiations: each AI/tools/sm120_tu/<name>.cu has a sibling
#        <name>.expect containing a substring that must appear in the SASS
#        (empty .expect = compile-only, no MMA expected)
for tu in "$REPO"/AI/tools/sm120_tu/*.cu; do
  [ -e "$tu" ] || break
  n="$(basename "$tu" .cu)"; exp="$(cat "${tu%.cu}.expect" 2>/dev/null || true)"
  if ! $NVCC $FLAGS $INC -o "$WORK/$n.cubin" "$tu" 2>"$WORK/e"; then
    printf '%-58s inst FAIL\n' "$n"; sed -n '1,6p' "$WORK/e"; FAIL=$((FAIL+1)); continue
  fi
  sass="$($CUOBJDUMP -sass "$WORK/$n.cubin" 2>/dev/null || true)"
  if [ -z "$exp" ]; then
    printf '%-58s inst OK (no MMA expected)\n' "$n"; PASS=$((PASS+1))
  elif grep -q -- "$exp" <<<"$sass"; then
    printf '%-58s inst OK [%s]\n' "$n" "$exp"; PASS=$((PASS+1))
  else
    printf '%-58s inst BAD SASS (want %s)\n' "$n" "$exp"
    grep -oE '[QHIO]MMA[^ ]*' <<<"$sass" | sort -u | sed 's/^/    got: /'
    FAIL=$((FAIL+1))
  fi
done

echo "-----"; echo "pass=$PASS fail=$FAIL arch=$ARCH"
[ "$FAIL" -eq 0 ]
