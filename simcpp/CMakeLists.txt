cmake_minimum_required(VERSION 3.16)
project(simcpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen is header-only and already on this machine; do not vendor or fetch it.
set(EIGEN3_INCLUDE_DIR "C:/repos/eigen" CACHE PATH "Eigen source root (header-only)")
if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
  message(FATAL_ERROR "Eigen not found at ${EIGEN3_INCLUDE_DIR} -- pass -DEIGEN3_INCLUDE_DIR=...")
endif()

set(SIM_CORE ${CMAKE_CURRENT_SOURCE_DIR}/src/sim.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/io.cpp)

add_library(simcore STATIC ${SIM_CORE})
target_include_directories(simcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR})
target_compile_definitions(simcore PUBLIC EIGEN_NO_DEBUG EIGEN_MPL2_ONLY)
if(MSVC)
  target_compile_options(simcore PUBLIC /O2 /fp:precise)
else()
  # -O3 only.  NOT -ffast-math: the solver's monotonicity gate and the 1e-300
  # guards in best_rotations depend on IEEE semantics.
  target_compile_options(simcore PUBLIC -O3)
endif()

if(EMSCRIPTEN)
  add_executable(simcpp ${CMAKE_CURRENT_SOURCE_DIR}/src/wasm.cpp)
  target_link_libraries(simcpp PRIVATE simcore)
  set_target_properties(simcpp PROPERTIES
    OUTPUT_NAME simcpp
    SUFFIX ".js"
    LINK_FLAGS "-lembind -sMODULARIZE=1 -sEXPORT_NAME=createSim -sEXPORT_ES6=0 \
-sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 -sSTACK_SIZE=1048576 \
-sEXPORTED_FUNCTIONS=['_sim_build','_sim_solve','_sim_n_hinges','_sim_mono_violations','_sim_seconds','_sim_energy','_sim_free','_malloc','_free'] \
-sEXPORTED_RUNTIME_METHODS=['HEAPF64','HEAP32','HEAPU8','ccall','cwrap'] \
-sENVIRONMENT=web,worker,node -O3")
else()
  add_executable(simcpp ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
  target_link_libraries(simcpp PRIVATE simcore)
endif()
