cmake_minimum_required(VERSION 3.20)
project(vikalp LANGUAGES CXX)

include(CTest)

if(BUILD_TESTING)
    if(MSVC)
        add_compile_options(/UNDEBUG)
    else()
        add_compile_options(-UNDEBUG)
    endif()
endif()

# ── Shared contracts library (owned by integrator — do not modify) ────────────
add_library(vikalp_contracts src/contracts/Model.cpp)
target_include_directories(vikalp_contracts PUBLIC include)
target_compile_features(vikalp_contracts PUBLIC cxx_std_20)

add_library(vikalp_io STATIC src/io/MpsReader.cpp)
target_link_libraries(vikalp_io PUBLIC vikalp_contracts)
target_include_directories(vikalp_io PUBLIC include)
target_compile_features(vikalp_io PUBLIC cxx_std_20)

# ── Flow B: CPU execution backend ─────────────────────────────────────────────
add_library(vikalp_backend STATIC src/backend/cpu_ops.cpp)
target_link_libraries(vikalp_backend PUBLIC vikalp_contracts)
target_compile_features(vikalp_backend PUBLIC cxx_std_20)

# ── Flow C: continuous solvers ───────────────────────────────────────────────
add_library(vikalp_solvers STATIC
    src/solver/lp/pdhg.cpp
    src/solver/qp/pdqp.cpp
    src/solver/nlp/ipm.cpp
)
target_include_directories(vikalp_solvers PUBLIC include)
target_link_libraries(vikalp_solvers PUBLIC vikalp_backend)
target_compile_features(vikalp_solvers PUBLIC cxx_std_20)

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

# ── Flow D: Mixed-integer engine ──────────────────────────────────────────────
add_library(vikalp_mip STATIC
    src/solver/mip/BranchAndBound.cpp
    src/solver/mip/CutGenerator.cpp
    src/solver/mip/OuterApproximation.cpp
    src/solver/Solver.cpp
    src/examples/RefineryModel.cpp
)
target_include_directories(vikalp_mip PUBLIC include)
target_link_libraries(vikalp_mip PUBLIC vikalp_contracts vikalp_solvers)
target_compile_features(vikalp_mip PUBLIC cxx_std_20)

add_executable(vikalp_benchmark tools/benchmark_runner.cpp)
target_link_libraries(vikalp_benchmark PRIVATE vikalp_mip)
target_compile_features(vikalp_benchmark PUBLIC cxx_std_20)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_executable(vikalp_contracts_test tests/contracts/contracts_test.cpp)
    target_link_libraries(vikalp_contracts_test PRIVATE vikalp_contracts)
    add_test(NAME contracts COMMAND vikalp_contracts_test)

    add_executable(vikalp_model_validation_test tests/model/model_validation_test.cpp)
    target_link_libraries(vikalp_model_validation_test PRIVATE vikalp_contracts)
    add_test(NAME model_validation COMMAND vikalp_model_validation_test)

    foreach(model_test IN ITEMS milp minlp miqp nlp qp)
        add_executable(vikalp_${model_test}_model_test tests/model/${model_test}_model_test.cpp)
        target_link_libraries(vikalp_${model_test}_model_test PRIVATE vikalp_contracts)
        add_test(NAME ${model_test}_model COMMAND vikalp_${model_test}_model_test)
    endforeach()

    add_executable(vikalp_mps_reader_test tests/io/mps_reader_test.cpp)
    target_link_libraries(vikalp_mps_reader_test PRIVATE vikalp_io)
    add_test(NAME mps_reader COMMAND vikalp_mps_reader_test)
    set_tests_properties(mps_reader PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(vikalp_backend_test tests/test_backend.cpp)
    target_link_libraries(vikalp_backend_test PRIVATE vikalp_backend)
    add_test(NAME backend COMMAND vikalp_backend_test)

    add_executable(vikalp_continuous_test tests/solver/continuous/test_pdhg.cpp)
    target_link_libraries(vikalp_continuous_test PRIVATE vikalp_solvers)
    add_test(NAME continuous COMMAND vikalp_continuous_test)

    add_executable(vikalp_mip_test tests/solver/integer/test_mip.cpp)
    target_link_libraries(vikalp_mip_test PRIVATE vikalp_mip)
    add_test(NAME mip COMMAND vikalp_mip_test)

    add_executable(vikalp_integration_test tests/solver/integration/test_solver.cpp)
    target_link_libraries(vikalp_integration_test PRIVATE vikalp_io vikalp_mip)
    add_test(NAME integration COMMAND vikalp_integration_test)
    set_tests_properties(integration PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
