#!/usr/bin/env bash
# SM120 host-header compile gate. Compiles each sm120 csrc header in isolation
# against torch + DeepJIT + CUTLASS. No GPU required.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
TORCH_INC="$(python -c 'import torch.utils.cpp_extension as e; print(" ".join("-I"+p for p in e.include_paths()))')"
ABI="$(python -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')"
PY_INC="$(python -c 'import sysconfig; print("-I"+sysconfig.get_paths()["include"])')"
INC="-I$REPO -I$REPO/deep_gemm/include -I$REPO/third-party/cutlass/include \
     -I$REPO/third-party/deep_jit/include -I$CUDA_HOME/include -I$CUDA_HOME/include/cccl \
     $TORCH_INC $PY_INC"
FLAGS="-std=c++20 -fsyntax-only -fPIC -D_GLIBCXX_USE_CXX11_ABI=$ABI -Wno-deprecated-declarations -Wno-abi"
PASS=0; FAIL=0
HEADERS="$(cat "$REPO/AI/tools/sm120_host_headers.txt" 2>/dev/null || true)"
[ -n "$(printf '%s' "$HEADERS" | tr -d '[:space:]')" ] || { echo "FATAL: AI/tools/sm120_host_headers.txt is missing, empty, or unreadable -- gate would verify nothing"; exit 2; }
for h in $HEADERS; do
  [ -f "$REPO/$h" ] || { printf '%-56s MISSING\n' "$h"; FAIL=$((FAIL+1)); continue; }
  echo "#include \"$REPO/$h\"" > "$WORK/c.cpp"
  if g++ $FLAGS $INC "$WORK/c.cpp" 2>"$WORK/e"; then
    printf '%-56s OK\n' "$h"; PASS=$((PASS+1))
  else
    printf '%-56s FAIL\n' "$h"; sed -n '1,8p' "$WORK/e"; FAIL=$((FAIL+1))
  fi
done
echo "-----"; echo "pass=$PASS fail=$FAIL"
[ "$PASS" -eq 0 ] && [ "$FAIL" -eq 0 ] && { echo "FATAL: gate ran zero checks (pass=0 fail=0) -- nothing was verified"; exit 2; }
[ "$FAIL" -eq 0 ]
