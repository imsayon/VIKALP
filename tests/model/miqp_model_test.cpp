#include "vikalp/contracts/Model.hpp"

#include <cassert>

int main() {
    vikalp::Model model;

    model.name = "SIMPLE_MIQP";

    model.linear_objective = {1.0, 2.0};

    model.quadratic_objective.pattern.rows = 2;
    model.quadratic_objective.pattern.columns = 2;
    model.quadratic_objective.pattern.row_offsets = {0, 1, 2};
    model.quadratic_objective.pattern.column_indices = {0, 1};
    model.quadratic_objective.values = {2.0, 4.0};

    model.constraint_matrix.pattern.rows = 1;
    model.constraint_matrix.pattern.columns = 2;
    model.constraint_matrix.pattern.row_offsets = {0, 2};
    model.constraint_matrix.pattern.column_indices = {0, 1};
    model.constraint_matrix.values = {1.0, 1.0};

    model.constraint_lower = {0.0};
    model.constraint_upper = {5.0};

    model.variable_lower = {0.0, 0.0};
    model.variable_upper = {10.0, 10.0};

    model.variable_types = {
        vikalp::VariableType::Integer,
        vikalp::VariableType::Continuous
    };

    assert(model.validate().empty());

    assert(model.num_variables() == 2);
    assert(model.num_constraints() == 1);

    assert(model.problem_class() ==
           vikalp::ProblemClass::MIQP);

    return 0;
}
