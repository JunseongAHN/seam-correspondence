#!/usr/bin/env bash
# Both builds, exactly as they were run and validated on this machine.
#   ./build.sh native
#   ./build.sh wasm
#   ./build.sh            (both)
set -euo pipefail
cd "$(dirname "$0")"

# Eigen 5.0.1-dev, header-only, already on this machine.  Not vendored, not fetched.
EIGEN=${EIGEN:-C:/repos/eigen}

# Any C++17 compiler works.  This machine had no toolchain, so a standalone
# WinLibs MinGW-w64 GCC 16.2.0 was unpacked into the scratchpad; point CXX at
# whatever you have (g++, clang++, or use the CMakeLists with MSVC).
: "${CXX:=g++}"

FLAGS="-std=c++17 -O3 -DEIGEN_NO_DEBUG -I src -I $EIGEN"
# NOT -ffast-math: the monotonicity gate in solve() and the 1e-300 guards in
# best_rotations depend on IEEE semantics.

build_native() {
  mkdir -p build-native
  echo "native: $CXX"
  $CXX $FLAGS -static src/sim.cpp src/io.cpp src/main.cpp -o build-native/simcpp.exe
  echo "  -> build-native/simcpp.exe"
}

build_wasm() {
  export EMSDK=/c/repos/emsdk
  export EMSDK_PYTHON="C:/repos/emsdk/python/3.13.3_64bit/python.exe"
  export PATH="/c/repos/emsdk:/c/repos/emsdk/upstream/emscripten:$PATH"
  mkdir -p build-wasm

  # (a) the JS-callable module: C API + embind, no filesystem
  em++ $FLAGS src/sim.cpp src/io.cpp src/wasm.cpp -o build-wasm/simcpp.js \
    -lembind -sMODULARIZE=1 -sEXPORT_NAME=createSim -sALLOW_MEMORY_GROWTH=1 \
    -sINITIAL_MEMORY=268435456 -sSTACK_SIZE=1048576 \
    -sEXPORTED_FUNCTIONS='["_sim_build","_sim_solve","_sim_n_hinges","_sim_mono_violations","_sim_seconds","_sim_energy","_sim_free","_malloc","_free"]' \
    -sEXPORTED_RUNTIME_METHODS='["HEAPF64","HEAP32","HEAPU8","ccall","cwrap"]' \
    -sENVIRONMENT=web,worker,node

  # (b) the same CLI as the native binary, for running the dumps under node
  em++ $FLAGS src/sim.cpp src/io.cpp src/main.cpp -o build-wasm/simcpp_node.js \
    -sNODERAWFS=1 -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=536870912 -sSTACK_SIZE=1048576
  echo "  -> build-wasm/simcpp.js  build-wasm/simcpp_node.js"
}

case "${1:-all}" in
  native) build_native ;;
  wasm)   build_wasm ;;
  all)    build_native; build_wasm ;;
  *) echo "usage: $0 [native|wasm|all]" >&2; exit 2 ;;
esac
