#!/bin/bash
# Builds the Task 6 runtime test fixtures: echo-kernels/fixtures/*.cu -> fixtures/<name>.ptx,
# with the nvcc flags of tig-monorepo's tig-binary/scripts/build_ptx (CUDA 12.8).
# Usage: echo-kernels/build_fixtures.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)/fixtures"
shopt -s nullglob
for f in "$DIR"/*.cu; do
  out="${f%.cu}.ptx"
  nvcc -ptx "$f" -o "$out" -arch compute_70 -code sm_70 --use_fast_math -dopt=on
  echo "wrote $out ($(wc -c < "$out") bytes, sha256 $(sha256sum "$out" | cut -c1-16))"
done
