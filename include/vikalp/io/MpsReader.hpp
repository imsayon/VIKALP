#pragma once

#include "vikalp/contracts/Model.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace vikalp {

struct MpsError {
    std::size_t line = 0;
    std::string message;
};

class MpsReader {
public:
    bool read(const std::string &path, Model &model);

    [[nodiscard]] const std::vector<MpsError> &errors() const noexcept;

private:
    std::vector<MpsError> errors_;

    std::vector<std::string> variable_names_;
    std::vector<std::string> row_names_;
    std::vector<char> row_types_;
    std::string objective_row_;

    struct ColumnEntry {
        std::string variable;
        std::string row;
        double value;
    };

    std::vector<ColumnEntry> column_entries_;
};

} // namespace vikalp
