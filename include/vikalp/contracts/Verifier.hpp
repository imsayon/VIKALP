#pragma once

#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/SolverOptions.hpp"

#include <string>

namespace vikalp {

struct VerificationResult {
    bool accepted = false;
    Scalar primal_residual = Model::infinity();
    Scalar stationarity_residual = Model::infinity();
    Scalar complementarity_residual = Model::infinity();
    Scalar integrality_residual = Model::infinity();
    std::string message;
};

class Verifier {
public:
    virtual ~Verifier() = default;
    [[nodiscard]] virtual VerificationResult verify(
        const Model &model,
        const SolveResult &result,
        const SolverOptions &options) const = 0;
};

} // namespace vikalp
