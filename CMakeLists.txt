cmake_minimum_required(VERSION 3.20)
project(vikalp LANGUAGES CXX)

include(CTest)

# ── Shared contracts library (owned by integrator — do not modify) ────────────
add_library(vikalp_contracts src/contracts/Model.cpp)
target_include_directories(vikalp_contracts PUBLIC include)
target_compile_features(vikalp_contracts PUBLIC cxx_std_20)

# ── Flow B: CPU execution backend ─────────────────────────────────────────────
add_library(vikalp_backend STATIC src/backend/cpu_ops.cpp)
target_link_libraries(vikalp_backend PUBLIC vikalp_contracts)
target_compile_features(vikalp_backend PUBLIC cxx_std_20)

# ── Optional Flow B CUDA backend ──────────────────────────────────────────────
# CUDA is only compiled when real hardware AND toolkit are confirmed present.
# Set -DVIKALP_ENABLE_CUDA=ON on the cmake command line to enable.
option(VIKALP_ENABLE_CUDA "Build CUDA execution backend" OFF)
if(VIKALP_ENABLE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        # RTX 3050 (Ampere); callers may override for different hardware.
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(vikalp_backend PRIVATE cuda/cuda_ops.cu)
    target_link_libraries(vikalp_backend PRIVATE CUDA::cudart CUDA::cusparse)
    target_compile_definitions(vikalp_backend PUBLIC VIKALP_CUDA_ENABLED)
    set_target_properties(vikalp_backend PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
    )

    if(BUILD_TESTING)
        add_executable(vikalp_cuda_consistency_test tests/test_cuda_consistency.cpp)
        target_link_libraries(vikalp_cuda_consistency_test PRIVATE vikalp_backend)
        add_test(NAME cuda_consistency COMMAND vikalp_cuda_consistency_test)
    endif()
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    # Existing contracts test — unchanged
    add_executable(vikalp_contracts_test tests/contracts/contracts_test.cpp)
    target_link_libraries(vikalp_contracts_test PRIVATE vikalp_contracts)
    add_test(NAME contracts COMMAND vikalp_contracts_test)

    add_executable(vikalp_model_validation_test tests/model/model_validation_test.cpp)
    target_link_libraries(vikalp_model_validation_test PRIVATE vikalp_contracts)
    add_test(NAME model_validation COMMAND vikalp_model_validation_test)

    # Flow B backend correctness tests
    add_executable(vikalp_backend_test tests/test_backend.cpp)
    target_link_libraries(vikalp_backend_test PRIVATE vikalp_backend)
    add_test(NAME backend COMMAND vikalp_backend_test)
endif()
