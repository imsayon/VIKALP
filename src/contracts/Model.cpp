#include "vikalp/contracts/Model.hpp"

#include "vikalp/contracts/NonlinearOracle.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace vikalp {
namespace {

bool validate_pattern(const CsrPattern &pattern, const std::string &name,
                      std::vector<std::string> &errors) {
    if (pattern.rows < 0 || pattern.columns < 0) {
        errors.push_back(name + " dimensions must be non-negative");
        return false;
    }
    if (pattern.row_offsets.size() != static_cast<std::size_t>(pattern.rows + 1)) {
        errors.push_back(name + " row_offsets must contain rows + 1 entries");
        return false;
    }
    if (pattern.row_offsets.empty() || pattern.row_offsets.front() != 0 ||
        pattern.row_offsets.back() !=
            static_cast<Index>(pattern.column_indices.size())) {
        errors.push_back(name + " row_offsets do not describe column_indices");
        return false;
    }

    for (Index row = 0; row < pattern.rows; ++row) {
        const auto begin = pattern.row_offsets[static_cast<std::size_t>(row)];
        const auto end = pattern.row_offsets[static_cast<std::size_t>(row + 1)];
        if (begin < 0 || begin > end ||
            end > static_cast<Index>(pattern.column_indices.size())) {
            errors.push_back(name + " row_offsets must be ordered and in range");
            return false;
        }
        Index previous = -1;
        for (Index position = begin; position < end; ++position) {
            const Index column =
                pattern.column_indices[static_cast<std::size_t>(position)];
            if (column < 0 || column >= pattern.columns) {
                errors.push_back(name + " contains an out-of-range column");
                return false;
            }
            if (column <= previous) {
                errors.push_back(name + " columns must be strictly ordered per row");
                return false;
            }
            previous = column;
        }
    }
    return true;
}

bool validate_matrix(const CsrMatrix &matrix, const std::string &name,
                     std::vector<std::string> &errors) {
    bool valid = validate_pattern(matrix.pattern, name, errors);
    if (matrix.values.size() != matrix.pattern.column_indices.size()) {
        errors.push_back(name + " values and column_indices sizes differ");
        valid = false;
    }
    if (std::any_of(matrix.values.begin(), matrix.values.end(),
                    [](Scalar value) { return !std::isfinite(value); })) {
        errors.push_back(name + " coefficients must be finite");
        valid = false;
    }
    return valid;
}

void validate_bounds(const std::vector<Scalar> &lower,
                     const std::vector<Scalar> &upper,
                     const std::string &name,
                     std::vector<std::string> &errors) {
    if (lower.size() != upper.size()) {
        errors.push_back(name + " lower and upper sizes differ");
        return;
    }
    for (std::size_t i = 0; i < lower.size(); ++i) {
        if (std::isnan(lower[i]) || std::isnan(upper[i]) || lower[i] > upper[i]) {
            errors.push_back(name + " contains an invalid bound at index " +
                             std::to_string(i));
        }
    }
}

} // namespace

Index Model::num_variables() const noexcept {
    return static_cast<Index>(linear_objective.size());
}

Index Model::num_constraints() const noexcept {
    return static_cast<Index>(constraint_lower.size());
}

bool Model::has_quadratic_objective() const noexcept {
    return !quadratic_objective.values.empty();
}

bool Model::has_integer_variables() const noexcept {
    return std::any_of(variable_types.begin(), variable_types.end(),
                       [](VariableType type) {
                           return type != VariableType::Continuous;
                       });
}

ProblemClass Model::problem_class() const noexcept {
    if (nonlinear) {
        return has_integer_variables() ? ProblemClass::MINLP : ProblemClass::NLP;
    }
    if (has_quadratic_objective()) {
        return has_integer_variables() ? ProblemClass::MIQP : ProblemClass::QP;
    }
    return has_integer_variables() ? ProblemClass::MILP : ProblemClass::LP;
}

std::vector<std::string> Model::validate() const {
    std::vector<std::string> errors;
    const Index variables = num_variables();
    const Index constraints = num_constraints();

    if (!std::isfinite(objective_offset) ||
        std::any_of(linear_objective.begin(), linear_objective.end(),
                    [](Scalar value) { return !std::isfinite(value); })) {
        errors.push_back("objective coefficients must be finite");
    }
    if (constraint_upper.size() != constraint_lower.size()) {
        errors.push_back("constraint lower and upper sizes differ");
    }
    if (variable_lower.size() != static_cast<std::size_t>(variables) ||
        variable_upper.size() != static_cast<std::size_t>(variables) ||
        variable_types.size() != static_cast<std::size_t>(variables)) {
        errors.push_back("variable metadata sizes must equal the objective size");
    }
    validate_bounds(constraint_lower, constraint_upper, "constraint bounds", errors);
    validate_bounds(variable_lower, variable_upper, "variable bounds", errors);

    validate_matrix(constraint_matrix, "constraint matrix", errors);
    if (constraint_matrix.pattern.rows != constraints ||
        constraint_matrix.pattern.columns != variables) {
        errors.push_back("constraint matrix dimensions do not match the model");
    }

    const bool quadratic_absent =
        quadratic_objective.pattern.rows == 0 &&
        quadratic_objective.pattern.columns == 0 &&
        quadratic_objective.pattern.column_indices.empty() &&
        quadratic_objective.values.empty();
    const bool quadratic_valid =
        validate_matrix(quadratic_objective, "quadratic objective", errors);
    if (!quadratic_absent &&
        (quadratic_objective.pattern.rows != variables ||
         quadratic_objective.pattern.columns != variables)) {
        errors.push_back("quadratic objective must be square and match the variables");
    }
    if (!quadratic_absent && quadratic_valid &&
        quadratic_objective.pattern.rows == variables &&
        quadratic_objective.pattern.columns == variables) {
        std::map<std::pair<Index, Index>, Scalar> entries;
        for (Index row = 0; row < quadratic_objective.pattern.rows; ++row) {
            for (Index position =
                     quadratic_objective.pattern.row_offsets[static_cast<std::size_t>(row)];
                 position < quadratic_objective.pattern.row_offsets[static_cast<std::size_t>(row + 1)];
                 ++position) {
                entries[{row, quadratic_objective.pattern.column_indices[static_cast<std::size_t>(position)]}] =
                    quadratic_objective.values[static_cast<std::size_t>(position)];
            }
        }
        for (const auto &[coordinate, value] : entries) {
            const auto transpose = entries.find({coordinate.second, coordinate.first});
            if (transpose == entries.end() ||
                std::abs(value - transpose->second) >
                    1e-12 * std::max({1.0, std::abs(value),
                                      std::abs(transpose->second)})) {
                errors.push_back("quadratic objective must be stored symmetrically");
                break;
            }
        }
    }

    if (nonlinear) {
        if (nonlinear->variables() != variables ||
            nonlinear->constraints() != constraints) {
            errors.push_back("nonlinear oracle dimensions do not match the model");
        }
        const auto &jacobian = nonlinear->jacobian_pattern();
        validate_pattern(jacobian, "nonlinear Jacobian", errors);
        if (jacobian.rows != constraints || jacobian.columns != variables) {
            errors.push_back("nonlinear Jacobian dimensions do not match the model");
        }
        const auto &hessian = nonlinear->hessian_pattern();
        validate_pattern(hessian, "nonlinear Hessian", errors);
        if (hessian.rows != variables || hessian.columns != variables) {
            errors.push_back("nonlinear Hessian dimensions do not match the model");
        }
    }
    return errors;
}

} // namespace vikalp
