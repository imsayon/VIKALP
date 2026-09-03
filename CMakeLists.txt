cmake_minimum_required(VERSION 3.20)
project(vikalp LANGUAGES CXX)

# ── Shared contracts library (owned by integrator — do not modify) ────────────
add_library(vikalp_contracts src/contracts/Model.cpp)
target_include_directories(vikalp_contracts PUBLIC include)
target_compile_features(vikalp_contracts PUBLIC cxx_std_20)

# ── Flow C: CPU execution backend ─────────────────────────────────────────────
add_library(vikalp_backend STATIC src/backend/cpu_ops.cpp)
target_link_libraries(vikalp_backend PUBLIC vikalp_contracts)
target_compile_features(vikalp_backend PUBLIC cxx_std_20)

# ── Optional CUDA backend (Phase 5) ──────────────────────────────────────────
# CUDA is only compiled when real hardware AND toolkit are confirmed present.
# Set -DVIKALP_ENABLE_CUDA=ON on the cmake command line to enable.
option(VIKALP_ENABLE_CUDA "Build CUDA execution backend" OFF)
if(VIKALP_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(vikalp_backend PRIVATE cuda/cuda_ops.cu)
    target_link_libraries(vikalp_backend PRIVATE CUDA::cudart CUDA::cusparse)
    target_compile_definitions(vikalp_backend PUBLIC VIKALP_CUDA_ENABLED)

    if(BUILD_TESTING)
        add_executable(vikalp_cuda_consistency_test tests/test_cuda_consistency.cpp)
        target_link_libraries(vikalp_cuda_consistency_test PRIVATE vikalp_backend)
        add_test(NAME cuda_consistency COMMAND vikalp_cuda_consistency_test)
    endif()
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
include(CTest)
if(BUILD_TESTING)
    # Existing contracts test — unchanged
    add_executable(vikalp_contracts_test tests/contracts/contracts_test.cpp)
    target_link_libraries(vikalp_contracts_test PRIVATE vikalp_contracts)
    add_test(NAME contracts COMMAND vikalp_contracts_test)

    # Flow C backend correctness tests
    add_executable(vikalp_backend_test tests/test_backend.cpp)
    target_link_libraries(vikalp_backend_test PRIVATE vikalp_backend)
    add_test(NAME backend COMMAND vikalp_backend_test)
endif()
