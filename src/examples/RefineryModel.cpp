// Flow D — Parametric refinery model implementation.

#include "vikalp/examples/RefineryModel.hpp"

#include <algorithm>
#include <map>
#include <vector>

namespace vikalp {

RefineryConfig refinery_small() {
    return RefineryConfig{3, 2, 2, 1, 1.0, false};
}

RefineryConfig refinery_medium() {
    return RefineryConfig{6, 4, 4, 2, 1.5, false};
}

RefineryConfig refinery_large() {
    return RefineryConfig{12, 8, 8, 2, 2.0, false};
}

RefineryConfig refinery_xl() {
    return RefineryConfig{20, 12, 12, 3, 3.0, false};
}

std::vector<RefineryDemoCase> refinery_demo_family() {
    auto small_miqp = refinery_small();
    small_miqp.include_quadratic = true;
    auto medium_miqp = refinery_medium();
    medium_miqp.include_quadratic = true;
    return {
        {"Refinery Small MILP", refinery_small()},
        {"Refinery Medium MILP", refinery_medium()},
        {"Refinery Large MILP", refinery_large()},
        {"Refinery XL MILP", refinery_xl()},
        {"Refinery Small MIQP", small_miqp},
        {"Refinery Medium MIQP", medium_miqp},
    };
}

Model build_refinery_model(const RefineryConfig &config) {
    Model model;
    model.name = "refinery_c" + std::to_string(config.num_crudes) +
                 "_u" + std::to_string(config.num_units) +
                 "_p" + std::to_string(config.num_products);

    const int C = std::max(1, config.num_crudes);
    const int U = std::max(1, config.num_units);
    const int P = std::max(1, config.num_products);
    const int Q = std::max(0, config.num_quality);

    // Variable indexing:
    // 0 .. C-1: y_c (binary, crude selection)
    // C .. 2*C-1: x_c (continuous, crude volume)
    // 2*C .. 2*C+U-1: z_u (binary, unit activation)
    // 2*C+U .. 2*C+U+P-1: p_k (continuous, product output)
    // 2*C+U+P .. 2*C+U+P + C*P - 1: b_cp (continuous, crude c blended into product p)
    const Index offset_yc = 0;
    const Index offset_xc = C;
    const Index offset_zu = 2 * C;
    const Index offset_pk = 2 * C + U;
    const Index offset_b  = 2 * C + U + P;
    const Index total_vars = offset_b + C * P;

    model.linear_objective.assign(static_cast<std::size_t>(total_vars), 0.0);
    model.variable_lower.assign(static_cast<std::size_t>(total_vars), 0.0);
    model.variable_upper.assign(static_cast<std::size_t>(total_vars), 0.0);
    model.variable_types.assign(static_cast<std::size_t>(total_vars), VariableType::Continuous);

    // y_c: binary, cost = 50 + 5*c
    for (int c = 0; c < C; ++c) {
        Index idx = offset_yc + c;
        model.variable_types[static_cast<std::size_t>(idx)] = VariableType::Binary;
        model.variable_lower[static_cast<std::size_t>(idx)] = 0.0;
        model.variable_upper[static_cast<std::size_t>(idx)] = 1.0;
        model.linear_objective[static_cast<std::size_t>(idx)] = 50.0 + 5.0 * c;
    }

    // x_c: continuous, cost = 20 + 2*c, upper bound = 500
    for (int c = 0; c < C; ++c) {
        Index idx = offset_xc + c;
        model.variable_types[static_cast<std::size_t>(idx)] = VariableType::Continuous;
        model.variable_lower[static_cast<std::size_t>(idx)] = 0.0;
        model.variable_upper[static_cast<std::size_t>(idx)] = 500.0;
        model.linear_objective[static_cast<std::size_t>(idx)] = 20.0 + 2.0 * c;
    }

    // z_u: binary, cost = 120 + 15*u
    for (int u = 0; u < U; ++u) {
        Index idx = offset_zu + u;
        model.variable_types[static_cast<std::size_t>(idx)] = VariableType::Binary;
        model.variable_lower[static_cast<std::size_t>(idx)] = 0.0;
        model.variable_upper[static_cast<std::size_t>(idx)] = 1.0;
        model.linear_objective[static_cast<std::size_t>(idx)] = 120.0 + 15.0 * u;
    }

    // p_k: continuous, demand minimum = 30 * demand_factor
    for (int p = 0; p < P; ++p) {
        Index idx = offset_pk + p;
        model.variable_types[static_cast<std::size_t>(idx)] = VariableType::Continuous;
        model.variable_lower[static_cast<std::size_t>(idx)] = 30.0 * config.demand_factor;
        model.variable_upper[static_cast<std::size_t>(idx)] = 1000.0;
        model.linear_objective[static_cast<std::size_t>(idx)] = -(10.0 + 3.0 * p); // product revenue
    }

    // b_cp: continuous, blend amounts
    for (int c = 0; c < C; ++c) {
        for (int p = 0; p < P; ++p) {
            Index idx = offset_b + c * P + p;
            model.variable_types[static_cast<std::size_t>(idx)] = VariableType::Continuous;
            model.variable_lower[static_cast<std::size_t>(idx)] = 0.0;
            model.variable_upper[static_cast<std::size_t>(idx)] = 500.0;
            model.linear_objective[static_cast<std::size_t>(idx)] = 0.5 * (c + p); // small handling cost
        }
    }

    // Constraints:
    // 1. Linking x_c <= 500 * y_c  -->  x_c - 500 * y_c <= 0  (C constraints)
    // 2. Crude mass balance: sum_p b_cp - x_c = 0             (C constraints)
    // 3. Product mass balance: sum_c b_cp - p_k = 0           (P constraints)
    // 4. Quality: sum_c quality_cq * b_cp - limit_q * p_k <= 0
    // 5. Processing capacity: sum_c x_c - sum_u Cap_u * z_u <= 0 (1 constraint)
    //    where Cap_u = 300 + 100 * u
    // 6. Unit activation limit: sum_u z_u >= 1                (1 constraint)

    struct RowDef {
        std::map<Index, Scalar> entries;
        Scalar lower;
        Scalar upper;
    };
    std::vector<RowDef> rows;

    // 1. Linking x_c - 500 y_c <= 0
    for (int c = 0; c < C; ++c) {
        RowDef row;
        row.entries[offset_xc + c] = 1.0;
        row.entries[offset_yc + c] = -500.0;
        row.lower = -Model::infinity();
        row.upper = 0.0;
        rows.push_back(std::move(row));
    }

    // 2. sum_p b_cp - x_c = 0
    for (int c = 0; c < C; ++c) {
        RowDef row;
        row.entries[offset_xc + c] = -1.0;
        for (int p = 0; p < P; ++p) {
            row.entries[offset_b + c * P + p] = 1.0;
        }
        row.lower = 0.0;
        row.upper = 0.0;
        rows.push_back(std::move(row));
    }

    // 3. sum_c b_cp - p_k = 0
    for (int p = 0; p < P; ++p) {
        RowDef row;
        row.entries[offset_pk + p] = -1.0;
        for (int c = 0; c < C; ++c) {
            row.entries[offset_b + c * P + p] = 1.0;
        }
        row.lower = 0.0;
        row.upper = 0.0;
        rows.push_back(std::move(row));
    }

    // 4. Product quality upper bounds. The coefficients are deterministic
    // crude-quality surrogates; the rows keep the generated family feasible.
    for (int p = 0; p < P; ++p) {
        for (int q = 0; q < Q; ++q) {
            RowDef row;
            const Scalar limit = 1.8 + 0.03 * q;
            row.entries[offset_pk + p] = -limit;
            for (int c = 0; c < C; ++c) {
                row.entries[offset_b + c * P + p] = 0.8 + 0.05 * c + 0.03 * q;
            }
            row.lower = -Model::infinity();
            row.upper = 0.0;
            rows.push_back(std::move(row));
        }
    }

    // 5. Unit capacity: sum_c x_c - sum_u (300 + 100*u) * z_u <= 0
    {
        RowDef row;
        for (int c = 0; c < C; ++c) {
            row.entries[offset_xc + c] = 1.0;
        }
        for (int u = 0; u < U; ++u) {
            row.entries[offset_zu + u] = -(300.0 + 100.0 * u);
        }
        row.lower = -Model::infinity();
        row.upper = 0.0;
        rows.push_back(std::move(row));
    }

    // 6. At least one unit on: sum_u z_u >= 1
    {
        RowDef row;
        for (int u = 0; u < U; ++u) {
            row.entries[offset_zu + u] = 1.0;
        }
        row.lower = 1.0;
        row.upper = static_cast<Scalar>(U);
        rows.push_back(std::move(row));
    }

    // Convert rows to CSR constraint matrix
    const Index num_rows = static_cast<Index>(rows.size());
    model.constraint_lower.resize(static_cast<std::size_t>(num_rows));
    model.constraint_upper.resize(static_cast<std::size_t>(num_rows));
    model.constraint_matrix.pattern.rows = num_rows;
    model.constraint_matrix.pattern.columns = total_vars;
    model.constraint_matrix.pattern.row_offsets = {0};

    for (std::size_t i = 0; i < rows.size(); ++i) {
        model.constraint_lower[i] = rows[i].lower;
        model.constraint_upper[i] = rows[i].upper;
        for (const auto &[col, val] : rows[i].entries) {
            model.constraint_matrix.pattern.column_indices.push_back(col);
            model.constraint_matrix.values.push_back(val);
        }
        model.constraint_matrix.pattern.row_offsets.push_back(
            static_cast<Index>(model.constraint_matrix.pattern.column_indices.size()));
    }

    // Quadratic objective handling
    if (config.include_quadratic) {
        // Diagonal quadratic penalty on p_k to model diminishing returns
        model.quadratic_objective.pattern.rows = total_vars;
        model.quadratic_objective.pattern.columns = total_vars;
        model.quadratic_objective.pattern.row_offsets = {0};

        for (Index var = 0; var < total_vars; ++var) {
            if (var >= offset_pk && var < offset_pk + P) {
                model.quadratic_objective.pattern.column_indices.push_back(var);
                model.quadratic_objective.values.push_back(0.1);
            }
            model.quadratic_objective.pattern.row_offsets.push_back(
                static_cast<Index>(model.quadratic_objective.pattern.column_indices.size()));
        }
    }

    return model;
}

} // namespace vikalp
