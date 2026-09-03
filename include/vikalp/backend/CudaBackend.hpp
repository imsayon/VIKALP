#pragma once

// Flow B — CUDA execution backend factory.
// Only available when VIKALP_CUDA_ENABLED is defined (cmake -DVIKALP_ENABLE_CUDA=ON).
// Solver code holds the result through the frozen ExecutionBackend contract.
// The concrete CudaVector/CudaMatrix types are private to cuda/cuda_ops.cu.

#include "vikalp/contracts/ExecutionBackend.hpp"

#include <memory>

namespace vikalp {

/// Returns a CUDA-resident ExecutionBackend.
/// Requires: CUDA device 0 to be present and accessible.
/// All operations are FP64. Device memory is RAII.
/// Thread safety: none — create one backend per thread if needed.
#ifdef VIKALP_CUDA_ENABLED
[[nodiscard]] std::unique_ptr<ExecutionBackend> make_cuda_backend();
#endif

} // namespace vikalp
