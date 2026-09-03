#pragma once

#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/SolverOptions.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace vikalp::solver::detail {

using Clock = std::chrono::steady_clock;

struct Bounds {
    std::vector<Scalar> lower;
    std::vector<Scalar> upper;
};

struct ScaledLinearModel {
    CsrPattern pattern;
    std::vector<Scalar> values;
    std::vector<Scalar> objective;
    std::vector<Scalar> row_lower;
    std::vector<Scalar> row_upper;
    std::vector<Scalar> variable_lower;
    std::vector<Scalar> variable_upper;
    std::vector<Scalar> row_scale;
    std::vector<Scalar> column_scale;
};

inline SolveResult invalid_result(const std::string &message) {
    SolveResult result;
    result.status = SolveStatus::InvalidModel;
    result.message = message;
    return result;
}

inline SolveResult unsupported_result(const std::string &message) {
    SolveResult result;
    result.status = SolveStatus::UnsupportedModel;
    result.message = message;
    return result;
}

inline bool valid_options(const SolverOptions &options) {
    return options.iteration_limit >= 0 && std::isfinite(options.primal_tolerance) &&
           std::isfinite(options.dual_tolerance) && options.primal_tolerance >= 0.0 &&
           options.dual_tolerance >= 0.0 &&
           (!std::isfinite(options.time_limit_seconds) ||
            options.time_limit_seconds >= 0.0);
}

inline bool timed_out(const Clock::time_point started,
                      const SolverOptions &options) {
    return std::isfinite(options.time_limit_seconds) &&
           std::chrono::duration<Scalar>(Clock::now() - started).count() >=
               options.time_limit_seconds;
}

inline bool finite_vector(std::span<const Scalar> values) {
    return std::all_of(values.begin(), values.end(),
                       [](Scalar value) { return std::isfinite(value); });
}

inline bool valid_bound_vector(std::span<const Scalar> values) {
    return std::all_of(values.begin(), values.end(),
                       [](Scalar value) { return !std::isnan(value); });
}

inline bool valid_override(std::span<const Scalar> values, Index size) {
    return values.empty() ||
           (static_cast<Index>(values.size()) == size && valid_bound_vector(values));
}

inline bool copy_bounds(const Model &model,
                        std::span<const Scalar> lower_override,
                        std::span<const Scalar> upper_override,
                        Bounds &bounds, std::string &message) {
    const Index n = model.num_variables();
    if (model.variable_lower.size() != static_cast<std::size_t>(n) ||
        model.variable_upper.size() != static_cast<std::size_t>(n)) {
        message = "model variable bounds must match the objective size";
        return false;
    }
    if (!valid_override(lower_override, n) ||
        !valid_override(upper_override, n)) {
        message = "variable bound overrides must be empty or sized and non-NaN";
        return false;
    }
    bounds.lower = lower_override.empty()
                       ? model.variable_lower
                       : std::vector<Scalar>(lower_override.begin(), lower_override.end());
    bounds.upper = upper_override.empty()
                       ? model.variable_upper
                       : std::vector<Scalar>(upper_override.begin(), upper_override.end());
    for (Index i = 0; i < n; ++i) {
        const auto lower = bounds.lower[static_cast<std::size_t>(i)];
        const auto upper = bounds.upper[static_cast<std::size_t>(i)];
        if (std::isnan(lower) || std::isnan(upper) || lower > upper) {
            message = "variable bound overrides contain an invalid interval";
            return false;
        }
    }
    return true;
}

inline SolveResult validate_continuous(const Model &model, ProblemClass expected,
                                       const Bounds &bounds) {
    const auto errors = model.validate();
    if (!errors.empty()) {
        return invalid_result(errors.front());
    }
    if (model.problem_class() != expected) {
        return unsupported_result("model class does not match the selected continuous solver");
    }
    if (bounds.lower.size() != static_cast<std::size_t>(model.num_variables()) ||
        bounds.upper.size() != static_cast<std::size_t>(model.num_variables())) {
        return invalid_result("variable bounds do not match the model");
    }
    return {};
}

inline Scalar scale_bound(Scalar value, Scalar scale, bool multiply) {
    if (!std::isfinite(value)) return value;
    return multiply ? value * scale : value / scale;
}

inline ScaledLinearModel scale_linear_model(const Model &model,
                                            const Bounds &bounds) {
    const auto &a = model.constraint_matrix;
    const Index m = a.pattern.rows;
    const Index n = a.pattern.columns;
    ScaledLinearModel scaled;
    scaled.pattern = a.pattern;
    scaled.values = a.values;
    scaled.objective.resize(static_cast<std::size_t>(n));
    scaled.row_lower.resize(static_cast<std::size_t>(m));
    scaled.row_upper.resize(static_cast<std::size_t>(m));
    scaled.variable_lower.resize(static_cast<std::size_t>(n));
    scaled.variable_upper.resize(static_cast<std::size_t>(n));
    scaled.row_scale.assign(static_cast<std::size_t>(m), 1.0);
    scaled.column_scale.assign(static_cast<std::size_t>(n), 1.0);

    for (Index row = 0; row < m; ++row) {
        Scalar maximum = 0.0;
        for (Index k = a.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < a.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            maximum = std::max(maximum,
                               std::abs(a.values[static_cast<std::size_t>(k)]));
        }
        if (maximum > 1.0) {
            scaled.row_scale[static_cast<std::size_t>(row)] = 1.0 / maximum;
        }
    }
    // ponytail: column equilibration scans CSR columns; use CSC metadata if this
    // becomes a measured preprocessing bottleneck.
    for (Index column = 0; column < n; ++column) {
        Scalar maximum = 0.0;
        for (Index row = 0; row < m; ++row) {
            for (Index k = a.pattern.row_offsets[static_cast<std::size_t>(row)];
                 k < a.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
                if (a.pattern.column_indices[static_cast<std::size_t>(k)] == column) {
                    maximum = std::max(maximum,
                                       std::abs(a.values[static_cast<std::size_t>(k)]));
                }
            }
        }
        if (maximum > 1.0) {
            scaled.column_scale[static_cast<std::size_t>(column)] = 1.0 / maximum;
        }
    }
    // Reapply both diagonal factors after column equilibration.
    scaled.values = a.values;
    for (Index row = 0; row < m; ++row) {
        for (Index k = a.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < a.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto column = a.pattern.column_indices[static_cast<std::size_t>(k)];
            scaled.values[static_cast<std::size_t>(k)] *=
                scaled.row_scale[static_cast<std::size_t>(row)] *
                scaled.column_scale[static_cast<std::size_t>(column)];
        }
    }
    for (Index i = 0; i < n; ++i) {
        const auto scale = scaled.column_scale[static_cast<std::size_t>(i)];
        scaled.objective[static_cast<std::size_t>(i)] =
            model.linear_objective[static_cast<std::size_t>(i)] * scale;
        scaled.variable_lower[static_cast<std::size_t>(i)] =
            scale_bound(bounds.lower[static_cast<std::size_t>(i)], scale, false);
        scaled.variable_upper[static_cast<std::size_t>(i)] =
            scale_bound(bounds.upper[static_cast<std::size_t>(i)], scale, false);
    }
    for (Index row = 0; row < m; ++row) {
        const auto scale = scaled.row_scale[static_cast<std::size_t>(row)];
        scaled.row_lower[static_cast<std::size_t>(row)] =
            scale_bound(model.constraint_lower[static_cast<std::size_t>(row)], scale, true);
        scaled.row_upper[static_cast<std::size_t>(row)] =
            scale_bound(model.constraint_upper[static_cast<std::size_t>(row)], scale, true);
    }
    return scaled;
}

inline Scalar scaled_operator_bound(const CsrPattern &pattern,
                                    std::span<const Scalar> values) {
    Scalar max_row = 0.0;
    std::vector<Scalar> columns(static_cast<std::size_t>(pattern.columns), 0.0);
    for (Index row = 0; row < pattern.rows; ++row) {
        Scalar row_sum = 0.0;
        for (Index k = pattern.row_offsets[static_cast<std::size_t>(row)];
             k < pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            row_sum += std::abs(values[position]);
            columns[static_cast<std::size_t>(pattern.column_indices[position])] +=
                std::abs(values[position]);
        }
        max_row = std::max(max_row, row_sum);
    }
    Scalar max_column = 0.0;
    for (const auto value : columns) max_column = std::max(max_column, value);
    return std::sqrt(max_row * max_column);
}

inline void csr_to_dense(const CsrMatrix &matrix, std::vector<Scalar> &dense) {
    const auto rows = static_cast<std::size_t>(matrix.pattern.rows);
    const auto columns = static_cast<std::size_t>(matrix.pattern.columns);
    dense.assign(rows * columns, 0.0);
    for (Index row = 0; row < matrix.pattern.rows; ++row) {
        for (Index k = matrix.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < matrix.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            dense[static_cast<std::size_t>(row) * columns +
                  static_cast<std::size_t>(matrix.pattern.column_indices[position])] =
                matrix.values[position];
        }
    }
}

inline bool positive_semidefinite(const CsrMatrix &matrix) {
    // ponytail: dense PSD validation keeps the baseline small; use a sparse
    // factorization/eigensolver when very large Q matrices become supported.
    const auto n = static_cast<std::size_t>(matrix.pattern.rows);
    std::vector<Scalar> a;
    csr_to_dense(matrix, a);
    const Scalar tolerance = 1e-10;
    for (std::size_t k = 0; k < n; ++k) {
        Scalar pivot = a[k * n + k];
        for (std::size_t j = 0; j < k; ++j) {
            pivot -= a[k * n + j] * a[k * n + j];
        }
        if (pivot < -tolerance * std::max<Scalar>(1.0, std::abs(a[k * n + k]))) {
            return false;
        }
        if (pivot <= tolerance) {
            for (std::size_t i = k + 1; i < n; ++i) {
                if (std::abs(a[i * n + k]) > tolerance) return false;
                a[i * n + k] = 0.0;
            }
            continue;
        }
        const Scalar diagonal = std::sqrt(pivot);
        a[k * n + k] = diagonal;
        for (std::size_t i = k + 1; i < n; ++i) {
            Scalar value = a[i * n + k];
            for (std::size_t j = 0; j < k; ++j) {
                value -= a[i * n + j] * a[k * n + j];
            }
            a[i * n + k] = value / diagonal;
        }
    }
    return true;
}

inline Scalar dot_host(std::span<const Scalar> left,
                       std::span<const Scalar> right) {
    Scalar result = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) result += left[i] * right[i];
    return result;
}

inline Scalar objective_value(const Model &model, std::span<const Scalar> x) {
    Scalar value = model.objective_offset +
                   dot_host(model.linear_objective, x);
    std::vector<Scalar> qx;
    csr_to_dense(model.quadratic_objective, qx);
    if (!qx.empty()) {
        const auto n = static_cast<std::size_t>(model.num_variables());
        Scalar quadratic = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t column = 0; column < n; ++column) {
                quadratic += x[row] * qx[row * n + column] * x[column];
            }
        }
        value += 0.5 * quadratic;
    }
    return value;
}

inline void linear_values(const Model &model, std::span<const Scalar> x,
                          std::vector<Scalar> &result) {
    const auto &matrix = model.constraint_matrix;
    result.assign(static_cast<std::size_t>(matrix.pattern.rows), 0.0);
    for (Index row = 0; row < matrix.pattern.rows; ++row) {
        for (Index k = matrix.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < matrix.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            result[static_cast<std::size_t>(row)] +=
                matrix.values[position] *
                x[static_cast<std::size_t>(matrix.pattern.column_indices[position])];
        }
    }
}

inline Scalar box_violation(std::span<const Scalar> values,
                            std::span<const Scalar> lower,
                            std::span<const Scalar> upper) {
    Scalar result = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::isfinite(lower[i])) result = std::max(result, lower[i] - values[i]);
        if (std::isfinite(upper[i])) result = std::max(result, values[i] - upper[i]);
    }
    return std::max<Scalar>(0.0, result);
}

inline bool finite_solution(std::span<const Scalar> values) {
    return finite_vector(values);
}

} // namespace vikalp::solver::detail
