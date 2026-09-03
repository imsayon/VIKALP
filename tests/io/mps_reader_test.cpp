#include "vikalp/io/MpsReader.hpp"

#include <cassert>
#include <iostream>

int main() {
    vikalp::MpsReader reader;
    vikalp::Model model;

    const bool success =
        reader.read("tests/io/simple_lp.mps", model);

    if (!success) {
        for (const auto &error : reader.errors()) {
            std::cerr << "Line " << error.line
                      << ": " << error.message << '\n';
        }
        return 1;
    }

    assert(model.name == "SIMPLELP");

    assert(model.num_variables() == 1);
    assert(model.num_constraints() == 1);

    assert(model.linear_objective.size() == 1);
    assert(model.linear_objective[0] == 1.0);

    assert(model.variable_lower[0] == 0.0);
    assert(model.variable_upper[0] == 10.0);

    assert(model.constraint_lower[0] ==
           -vikalp::Model::infinity());

    assert(model.constraint_upper[0] == 5.0);

    assert(model.constraint_matrix.values.size() == 1);
    assert(model.constraint_matrix.values[0] == 1.0);

    assert(model.problem_class() == vikalp::ProblemClass::LP);

    return 0;
}
