#!/usr/bin/env bash
# Build llama.cpp CUDA binaries for the GGUF arm of the PTQ-MT replication.
# Compilation needs only nvcc + cmake (no GPU), so it runs on the login node.
# Produces: llama-quantize, llama-imatrix, llama-cli under ~/llama.cpp/build/bin
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CMAKE="$REPO_ROOT/.venv/bin/cmake"
export CUDA_HOME=/apps/cudatoolkit/12.8.1
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

# A100 (sm_80) is our primary partition; sm_90 covers H100/H200 fallbacks.
CUDA_ARCHS="${CUDA_ARCHS:-80;90}"

if [ ! -d "$LLAMA_DIR/.git" ]; then
  echo ">> cloning llama.cpp into $LLAMA_DIR"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
echo ">> llama.cpp at commit $(git rev-parse --short HEAD)"

# GGML_NATIVE=OFF: the login node and compute nodes have different CPUs, so a
# `-march=native` build on the login node SIGILLs on compute nodes. A portable
# baseline is fine — compute is offloaded to the GPU (-ngl 99).
"$CMAKE" -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

"$CMAKE" --build build --config Release -j 8 \
  --target llama-quantize llama-imatrix llama-cli llama-server

echo ">> built binaries:"
ls -la "$LLAMA_DIR"/build/bin/llama-quantize "$LLAMA_DIR"/build/bin/llama-imatrix \
       "$LLAMA_DIR"/build/bin/llama-cli "$LLAMA_DIR"/build/bin/llama-server
echo ">> DONE"
