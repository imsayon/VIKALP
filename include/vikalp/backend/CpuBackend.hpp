#pragma once

// Flow C — CPU execution backend factory.
// Flow A should call make_cpu_backend() and hold the result as ExecutionBackend*.
// The concrete CpuVector/CpuMatrix types are private to src/backend/cpu_ops.cpp.

#include "vikalp/contracts/ExecutionBackend.hpp"

#include <memory>

namespace vikalp {

/// Returns a CPU-resident ExecutionBackend.
/// All operations are double-precision, single-threaded, and deterministic.
/// Thread safety: none — create one backend per thread if needed.
[[nodiscard]] std::unique_ptr<ExecutionBackend> make_cpu_backend();

} // namespace vikalp
