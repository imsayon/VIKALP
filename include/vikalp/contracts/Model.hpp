#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace vikalp {

using Index = std::int64_t;
using Scalar = double;

class NonlinearOracle;

struct CsrPattern {
    Index rows = 0;
    Index columns = 0;
    std::vector<Index> row_offsets{0};
    std::vector<Index> column_indices;
};

struct CsrMatrix {
    CsrPattern pattern;
    std::vector<Scalar> values;
};

enum class VariableType { Continuous,
                          Integer,
                          Binary };
enum class ProblemClass { LP,
                          QP,
                          NLP,
                          MILP,
                          MIQP,
                          MINLP };

// Canonical minimization model:
//   min c'x + 0.5*x'Qx + f(x)
//   s.t. row_lower <= Ax + g(x) <= row_upper
//        variable_lower <= x <= variable_upper
// Q is stored as a full symmetric CSR matrix. An absent Q is the default 0x0 matrix.
struct Model {
    std::string name;
    std::vector<Scalar> linear_objective;
    Scalar objective_offset = 0.0;
    CsrMatrix quadratic_objective;
    CsrMatrix constraint_matrix;
    std::vector<Scalar> constraint_lower;
    std::vector<Scalar> constraint_upper;
    std::vector<Scalar> variable_lower;
    std::vector<Scalar> variable_upper;
    std::vector<VariableType> variable_types;
    std::shared_ptr<const NonlinearOracle> nonlinear;

    [[nodiscard]] Index num_variables() const noexcept;
    [[nodiscard]] Index num_constraints() const noexcept;
    [[nodiscard]] bool has_quadratic_objective() const noexcept;
    [[nodiscard]] bool has_integer_variables() const noexcept;
    [[nodiscard]] ProblemClass problem_class() const noexcept;
    [[nodiscard]] std::vector<std::string> validate() const;

    static constexpr Scalar infinity() noexcept {
        return std::numeric_limits<Scalar>::infinity();
    }
};

} // namespace vikalp
