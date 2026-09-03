#include "MpsReader.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <utility>

namespace vikalp {

bool MpsReader::read(const std::string &path, Model &model) {
    errors_.clear();
    variable_names_.clear();
    row_names_.clear();
    row_types_.clear();
    objective_row_.clear();
    column_entries_.clear();

    std::ifstream input(path);

    if (!input) {
        errors_.push_back(
            {0, "unable to open MPS file: " + path});
        return false;
    }

    model = Model{};

    std::string line;
    std::size_t line_number = 0;

    bool saw_name = false;
    bool saw_rows = false;
    bool saw_columns = false;
    bool saw_rhs = false;
    bool saw_endata = false;

    bool in_rows = false;
    bool in_columns = false;
    bool in_rhs = false;
    bool in_bounds = false;

    std::map<std::string, Scalar> rhs_values;

    struct BoundEntry {
        std::string type;
        std::string variable;
        Scalar value = 0.0;
    };

    std::vector<BoundEntry> bound_entries;

    while (std::getline(input, line)) {
        ++line_number;

        if (line.empty()) {
            continue;
        }

        std::istringstream stream(line);

        std::string first;
        stream >> first;

        if (first == "NAME") {
            saw_name = true;
            stream >> model.name;

            in_rows = false;
            in_columns = false;
            in_rhs = false;
            in_bounds = false;
        }
        else if (first == "ROWS") {
            saw_rows = true;

            in_rows = true;
            in_columns = false;
            in_rhs = false;
            in_bounds = false;
        }
        else if (first == "COLUMNS") {
            saw_columns = true;

            in_rows = false;
            in_columns = true;
            in_rhs = false;
            in_bounds = false;
        }
        else if (first == "RHS") {
            saw_rhs = true;

            in_rows = false;
            in_columns = false;
            in_rhs = true;
            in_bounds = false;
        }
        else if (first == "BOUNDS") {
            in_rows = false;
            in_columns = false;
            in_rhs = false;
            in_bounds = true;
        }
        else if (first == "ENDATA") {
            saw_endata = true;
            break;
        }
        else if (in_rows) {
            std::string row_type = first;
            std::string row_name;

            stream >> row_name;

            if (row_name.empty()) {
                errors_.push_back(
                    {line_number,
                     "missing row name"});
                continue;
            }

            if (row_type == "N") {
                objective_row_ = row_name;
            }
            else if (row_type == "L" ||
                     row_type == "G" ||
                     row_type == "E") {

                row_names_.push_back(row_name);
                row_types_.push_back(row_type[0]);
            }
            else {
                errors_.push_back(
                    {line_number,
                     "unsupported ROWS type '" +
                         row_type + "'"});
            }
        }
        else if (in_columns) {
            std::string variable_name = first;
            std::string row_name;
            double value = 0.0;

            stream >> row_name >> value;

            if (variable_name.empty() ||
                row_name.empty() ||
                stream.fail()) {

                errors_.push_back(
                    {line_number,
                     "invalid COLUMNS entry"});
                continue;
            }

            variable_names_.push_back(variable_name);

            column_entries_.push_back(
                {variable_name,
                 row_name,
                 value});

            std::string second_row;
            double second_value = 0.0;

            if (stream >> second_row >>
                second_value) {

                column_entries_.push_back(
                    {variable_name,
                     second_row,
                     second_value});
            }
        }
        else if (in_rhs) {
            std::string rhs_name = first;
            std::string row_name;
            double value = 0.0;

            stream >> row_name >> value;

            if (rhs_name.empty() ||
                row_name.empty() ||
                stream.fail()) {

                errors_.push_back(
                    {line_number,
                     "invalid RHS entry"});
                continue;
            }

            rhs_values[row_name] = value;

            std::string second_row;
            double second_value = 0.0;

            if (stream >> second_row >>
                second_value) {

                rhs_values[second_row] =
                    second_value;
            }
        }
        else if (in_bounds) {
            std::string bound_type = first;
            std::string bound_name;
            std::string variable_name;
            double value = 0.0;

            stream >> bound_name >> variable_name;

            if (bound_type.empty() ||
                bound_name.empty() ||
                variable_name.empty()) {

                errors_.push_back(
                    {line_number,
                     "invalid BOUNDS entry"});
                continue;
            }

            if (bound_type == "LO" ||
                bound_type == "UP" ||
                bound_type == "FX") {

                if (!(stream >> value)) {
                    errors_.push_back(
                        {line_number,
                         "missing bound value"});
                    continue;
                }

                bound_entries.push_back(
                    {bound_type,
                     variable_name,
                     value});
            }
            else if (bound_type == "BV") {
                bound_entries.push_back(
                    {bound_type,
                     variable_name,
                     1.0});
            }
            else if (bound_type == "FR") {
                bound_entries.push_back(
                    {bound_type,
                     variable_name,
                     0.0});
            }
            else {
                errors_.push_back(
                    {line_number,
                     "unsupported BOUNDS type '" +
                         bound_type + "'"});
            }
        }
    }

    if (!saw_name) {
        errors_.push_back(
            {0, "missing NAME section"});
    }

    if (!saw_rows) {
        errors_.push_back(
            {0, "missing ROWS section"});
    }

    if (!saw_columns) {
        errors_.push_back(
            {0, "missing COLUMNS section"});
    }

    if (!saw_rhs) {
        errors_.push_back(
            {0, "missing RHS section"});
    }

    if (!saw_endata) {
        errors_.push_back(
            {line_number,
             "missing ENDATA section"});
    }

    if (!errors_.empty()) {
        return false;
    }

    std::vector<std::string> variables;

    for (const auto &name : variable_names_) {
        bool already_present = false;

        for (const auto &existing : variables) {
            if (existing == name) {
                already_present = true;
                break;
            }
        }

        if (!already_present) {
            variables.push_back(name);
        }
    }

    const Index number_of_variables =
        static_cast<Index>(variables.size());

    const Index number_of_constraints =
        static_cast<Index>(row_names_.size());

    model.linear_objective.assign(
        static_cast<std::size_t>(number_of_variables),
        0.0);

    model.variable_lower.assign(
        static_cast<std::size_t>(number_of_variables),
        0.0);

    model.variable_upper.assign(
        static_cast<std::size_t>(number_of_variables),
        Model::infinity());

    model.variable_types.assign(
        static_cast<std::size_t>(number_of_variables),
        VariableType::Continuous);

    model.constraint_lower.assign(
        static_cast<std::size_t>(number_of_constraints),
        -Model::infinity());

    model.constraint_upper.assign(
        static_cast<std::size_t>(number_of_constraints),
        Model::infinity());

    std::map<std::string, Index> variable_index;

    for (Index i = 0;
         i < number_of_variables;
         ++i) {

        variable_index[
            variables[
                static_cast<std::size_t>(i)]] = i;
    }

    std::map<std::string, Index> row_index;

    for (Index i = 0;
         i < number_of_constraints;
         ++i) {

        row_index[
            row_names_[
                static_cast<std::size_t>(i)]] = i;
    }

    std::map<std::pair<Index, Index>, Scalar>
        matrix_entries;

    for (const auto &entry : column_entries_) {
        const auto variable_it =
            variable_index.find(entry.variable);

        if (variable_it ==
            variable_index.end()) {

            errors_.push_back(
                {0,
                 "unknown variable in COLUMNS: " +
                     entry.variable});
            continue;
        }

        const Index variable =
            variable_it->second;

        if (entry.row == objective_row_) {
            model.linear_objective[
                static_cast<std::size_t>(variable)] =
                entry.value;

            continue;
        }

        const auto row_it =
            row_index.find(entry.row);

        if (row_it == row_index.end()) {
            errors_.push_back(
                {0,
                 "unknown row in COLUMNS: " +
                     entry.row});
            continue;
        }

        const Index row = row_it->second;

        matrix_entries[
            {row, variable}] = entry.value;
    }

    if (!errors_.empty()) {
        return false;
    }

    for (Index row = 0;
         row < number_of_constraints;
         ++row) {

        const std::string &row_name =
            row_names_[
                static_cast<std::size_t>(row)];

        const char row_type =
            row_types_[
                static_cast<std::size_t>(row)];

        const auto rhs_it =
            rhs_values.find(row_name);

        const Scalar rhs =
            (rhs_it == rhs_values.end())
                ? 0.0
                : rhs_it->second;

        if (row_type == 'L') {
            model.constraint_lower[
                static_cast<std::size_t>(row)] =
                -Model::infinity();

            model.constraint_upper[
                static_cast<std::size_t>(row)] =
                rhs;
        }
        else if (row_type == 'G') {
            model.constraint_lower[
                static_cast<std::size_t>(row)] =
                rhs;

            model.constraint_upper[
                static_cast<std::size_t>(row)] =
                Model::infinity();
        }
        else if (row_type == 'E') {
            model.constraint_lower[
                static_cast<std::size_t>(row)] =
                rhs;

            model.constraint_upper[
                static_cast<std::size_t>(row)] =
                rhs;
        }
    }

    for (const auto &bound : bound_entries) {
        const auto variable_it =
            variable_index.find(bound.variable);

        if (variable_it ==
            variable_index.end()) {

            errors_.push_back(
                {0,
                 "unknown variable in BOUNDS: " +
                     bound.variable});
            continue;
        }

        const Index variable =
            variable_it->second;

        const std::size_t index =
            static_cast<std::size_t>(variable);

        if (bound.type == "LO") {
            model.variable_lower[index] =
                bound.value;
        }
        else if (bound.type == "UP") {
            model.variable_upper[index] =
                bound.value;
        }
        else if (bound.type == "FX") {
            model.variable_lower[index] =
                bound.value;

            model.variable_upper[index] =
                bound.value;
        }
        else if (bound.type == "FR") {
            model.variable_lower[index] =
                -Model::infinity();

            model.variable_upper[index] =
                Model::infinity();
        }
        else if (bound.type == "BV") {
            model.variable_lower[index] = 0.0;
            model.variable_upper[index] = 1.0;

            model.variable_types[index] =
                VariableType::Binary;
        }
    }

    if (!errors_.empty()) {
        return false;
    }

    model.constraint_matrix.pattern.rows =
        number_of_constraints;

    model.constraint_matrix.pattern.columns =
        number_of_variables;

    model.constraint_matrix.pattern.row_offsets.clear();
    model.constraint_matrix.pattern.column_indices.clear();
    model.constraint_matrix.values.clear();

    model.constraint_matrix.pattern.row_offsets.push_back(0);

    for (Index row = 0;
         row < number_of_constraints;
         ++row) {

        for (Index variable = 0;
             variable < number_of_variables;
             ++variable) {

            const auto it =
                matrix_entries.find(
                    {row, variable});

            if (it != matrix_entries.end()) {
                model.constraint_matrix
                    .pattern.column_indices
                    .push_back(variable);

                model.constraint_matrix.values
                    .push_back(it->second);
            }
        }

        model.constraint_matrix
            .pattern.row_offsets.push_back(
                static_cast<Index>(
                    model.constraint_matrix
                        .pattern.column_indices
                        .size()));
    }

    return model.validate().empty();
}

const std::vector<MpsError> &
MpsReader::errors() const noexcept {
    return errors_;
}

} // namespace vikalp
