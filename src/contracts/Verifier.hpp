#ifndef VIKALP_VERIFIER_HPP
#define VIKALP_VERIFIER_HPP

#include "Model.hpp"
#include "SolveResult.hpp"

namespace vikalk {

// Verifies whether a candidate solution satisfies the model requirements.
class Verifier {
public:
    virtual ~Verifier() = default;

    virtual bool verify(
        const Model& model,
        const SolveResult& result) const = 0;
};

}  // namespace vikalk

#endif  // VIKALP_VERIFIER_HPP