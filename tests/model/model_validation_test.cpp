#include "vikalp/contracts/Model.hpp"

#include <cassert>

int main() {
    vikalp::Model model;

    model.linear_objective = {1.0};

    model.constraint_matrix.pattern.rows = 1;
    model.constraint_matrix.pattern.columns = 1;
    model.constraint_matrix.pattern.row_offsets = {0, 1};
    model.constraint_matrix.pattern.column_indices = {0};
    model.constraint_matrix.values = {1.0};

    model.constraint_lower = {0.0};
    model.constraint_upper = {1.0};

    model.variable_lower = {0.0};
    model.variable_upper = {1.0};
    model.variable_types = {vikalp::VariableType::Continuous};

    assert(model.validate().empty());
    assert(model.num_variables() == 1);
    assert(model.num_constraints() == 1);
    assert(model.problem_class() == vikalp::ProblemClass::LP);

    return 0;
}