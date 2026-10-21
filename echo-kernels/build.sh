#!/bin/bash
# Builds ECHO's device library: every echo-kernels/*.cu (sorted) -> one PTX file.
# Usage: echo-kernels/build.sh [out.ptx]   (default: echo-kernels/ptx/echo_lib.ptx)
# Each exported symbol is marked in its .cu file with a comment line:  // ECHO-EXPORT: <name>
# The build fails if an exported symbol has no .visible .func/.entry line, or if the PTX
# contains anything nondeterministic (see check_hazards.py).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$DIR/ptx/echo_lib.ptx}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
shopt -s nullglob
SRCS=("$DIR"/*.cu)
if [ ${#SRCS[@]} -eq 0 ]; then echo "no .cu sources in $DIR" >&2; exit 1; fi
for f in $(printf '%s\n' "${SRCS[@]}" | sort); do cat "$f"; echo; done > "$TMP/echo_lib.cu"
nvcc -ptx -rdc=true -arch compute_70 -code sm_70 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false \
  -Wno-deprecated-gpu-targets "$TMP/echo_lib.cu" -o "$TMP/echo_lib.ptx"
missing=0
for sym in $(grep -h -o 'ECHO-EXPORT: *[A-Za-z_0-9]*' "${SRCS[@]}" | sed 's/ECHO-EXPORT: *//' | sort -u); do
  if ! grep -q -E "^\.visible \.(func|entry).*[ )]$sym\(" "$TMP/echo_lib.ptx"; then
    echo "missing exported symbol in PTX: $sym" >&2; missing=1
  fi
done
[ $missing -eq 0 ] || exit 1
python3 -I "$DIR/check_hazards.py" "$TMP/echo_lib.ptx"
mkdir -p "$(dirname "$OUT")"
cp "$TMP/echo_lib.ptx" "$OUT"
echo "wrote $OUT ($(wc -c < "$OUT") bytes, sha256 $(sha256sum "$OUT" | cut -c1-16))"
