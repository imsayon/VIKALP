# VIKALP

**Versatile Indigenous Kernel for Accelerated Large-scale Programming**

VIKALP is a C++20 mathematical optimization engine for linear, quadratic,
nonlinear, and mixed-integer models. It provides a sparse in-memory model,
an MPS reader, CPU execution, and an optional CUDA backend.

## Requirements

- CMake 3.20 or newer
- A C++20 compiler
- Ninja or another CMake-supported build tool
- Optional: NVIDIA CUDA Toolkit for GPU execution

## Build

Configure and build a release binary:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

To build with CUDA, provide the architecture supported by the target GPU. The
default is `86`, suitable for the NVIDIA RTX 3050.

```bash
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVIKALP_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda
```

## Test

Build and run the complete CPU test suite:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

For a CUDA build, run the same command against `build-cuda`. This includes the
CPU/CUDA consistency test when CUDA support is enabled.

```bash
ctest --test-dir build-cuda --output-on-failure
```

## Use the engine

Add VIKALP to a CMake project and link the solver target:

```cmake
add_subdirectory(path/to/vikalp)
target_link_libraries(your_target PRIVATE vikalp_mip)
```

Construct a model and pass it to the public solver entry point. Models use a
canonical minimization form with CSR constraint and quadratic matrices.

```cpp
#include <vikalp/solver/Solver.hpp>

#include <iostream>

int main() {
    vikalp::Model model;
    model.name = "bounded_lp";
    model.linear_objective = {-1.0};
    model.variable_lower = {0.0};
    model.variable_upper = {10.0};
    model.variable_types = {vikalp::VariableType::Continuous};
    model.constraint_matrix.pattern = {0, 1, {0}, {}};

    vikalp::SolverOptions options;
    options.time_limit_seconds = 10.0;

    const auto result = vikalp::solve(model, options);
    if (result.status != vikalp::SolveStatus::Optimal) {
        std::cerr << result.message << '\n';
        return 1;
    }

    std::cout << "objective: " << result.objective_value << '\n';
    std::cout << "x: " << result.primal_solution[0] << '\n';
}
```

Select CUDA explicitly when using a CUDA-enabled build:

```cpp
vikalp::SolverOptions options;
options.backend = vikalp::BackendPreference::CUDA;
const auto result = vikalp::solve(model, options);
```

### Read an MPS model

```cpp
#include <vikalp/io/MpsReader.hpp>
#include <vikalp/solver/Solver.hpp>

#include <iostream>

vikalp::MpsReader reader;
vikalp::Model model;

if (!reader.read("model.mps", model)) {
    for (const auto& error : reader.errors()) {
        std::cerr << error.line << ": " << error.message << '\n';
    }
    return 1;
}

const auto result = vikalp::solve(model);
```

## Run the benchmark

The bundled benchmark runner executes deterministic refinery model cases and
can write machine-readable results.

```bash
./build/vikalp_benchmark
./build/vikalp_benchmark \
  --csv benchmarks/results/latest.csv \
  --json benchmarks/results/latest.json
```

Run `./build/vikalp_benchmark --help` for the available options.

## Scope

VIKALP is a solver core, not a graphical modeling environment. The current MPS
reader and solver interfaces are under active development and should be
evaluated against the requirements of each production workload.
