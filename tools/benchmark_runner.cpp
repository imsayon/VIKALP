// Flow D — deterministic benchmark and external-oracle runner.

#include "vikalp/contracts/RelaxationOracle.hpp"
#include "vikalp/examples/RefineryModel.hpp"
#include "vikalp/solver/Solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using vikalp::Index;
using vikalp::Model;
using vikalp::Scalar;

struct BenchmarkRecord {
    std::string name;
    std::string source;
    std::string status;
    Index variables = 0;
    Index constraints = 0;
    Index nodes = 0;
    double milliseconds = 0.0;
    Scalar objective = Model::infinity();
    Scalar dual_bound = -Model::infinity();
    Scalar relative_gap = Model::infinity();
};

const char *status_name(vikalp::SolveStatus status) {
    switch (status) {
    case vikalp::SolveStatus::Optimal: return "Optimal";
    case vikalp::SolveStatus::LocallyOptimal: return "LocallyOptimal";
    case vikalp::SolveStatus::Feasible: return "Feasible";
    case vikalp::SolveStatus::Infeasible: return "Infeasible";
    case vikalp::SolveStatus::NodeLimit: return "NodeLimit";
    case vikalp::SolveStatus::TimeLimit: return "TimeLimit";
    case vikalp::SolveStatus::NumericalFailure: return "NumericalFailure";
    default: return "Other";
    }
}

Scalar evaluate_objective(const Model &model, const std::vector<Scalar> &x) {
    Scalar result = model.objective_offset;
    for (std::size_t i = 0; i < x.size(); ++i) {
        result += model.linear_objective[i] * x[i];
    }
    for (Index row = 0; row < model.quadratic_objective.pattern.rows; ++row) {
        for (Index position = model.quadratic_objective.pattern.row_offsets[
                 static_cast<std::size_t>(row)];
             position < model.quadratic_objective.pattern.row_offsets[
                 static_cast<std::size_t>(row + 1)]; ++position) {
            const auto pos = static_cast<std::size_t>(position);
            result += 0.5 * model.quadratic_objective.values[pos] *
                      x[static_cast<std::size_t>(row)] *
                      x[static_cast<std::size_t>(model.quadratic_objective.pattern
                                                       .column_indices[pos])];
        }
    }
    return result;
}

class RefineryOracle final : public vikalp::RelaxationOracle {
public:
    explicit RefineryOracle(vikalp::RefineryConfig config)
        : config_(config) {}

    [[nodiscard]] vikalp::SolveResult solve(
        const Model &model, std::span<const Scalar> lower,
        std::span<const Scalar> upper, std::span<const Scalar>,
        const vikalp::SolverOptions &) const override {
        const int crude_count = std::max(1, config_.num_crudes);
        const int unit_count = std::max(1, config_.num_units);
        const int product_count = std::max(1, config_.num_products);
        const Index crude_offset = 0;
        const Index volume_offset = crude_count;
        const Index unit_offset = 2 * crude_count;
        const Index product_offset = unit_offset + unit_count;
        const Index blend_offset = product_offset + product_count;
        const auto n = static_cast<std::size_t>(model.num_variables());
        const auto lo = [&](Index index) { return lower[static_cast<std::size_t>(index)]; };
        const auto hi = [&](Index index) { return upper[static_cast<std::size_t>(index)]; };

        vikalp::SolveResult result;
        std::vector<Scalar> x(n, 0.0);
        for (Index i = 0; i < model.num_variables(); ++i) {
            x[static_cast<std::size_t>(i)] = lo(i);
        }

        Scalar total_demand = 0.0;
        for (int p = 0; p < product_count; ++p) {
            const Index index = product_offset + p;
            if (lo(index) > hi(index)) {
                result.status = vikalp::SolveStatus::Infeasible;
                return result;
            }
            total_demand += lo(index);
        }

        Scalar remaining = total_demand;
        std::vector<Scalar> crude_volume(static_cast<std::size_t>(crude_count), 0.0);
        for (int c = 0; c < crude_count && remaining > 0.0; ++c) {
            const Index y = crude_offset + c;
            const Index volume = volume_offset + c;
            const Scalar capacity = std::min(hi(volume), 500.0 * hi(y));
            const Scalar amount = std::min(remaining, std::max(0.0, capacity));
            if (amount < lo(volume)) continue;
            crude_volume[static_cast<std::size_t>(c)] = amount;
            x[static_cast<std::size_t>(volume)] = amount;
            x[static_cast<std::size_t>(y)] = std::max(lo(y), amount / 500.0);
            if (x[static_cast<std::size_t>(y)] > hi(y) + 1e-9) continue;
            remaining -= amount;
        }
        if (remaining > 1e-8) {
            result.status = vikalp::SolveStatus::Infeasible;
            return result;
        }

        Scalar total_capacity = 0.0;
        Scalar total_activation = 0.0;
        for (int u = 0; u < unit_count; ++u) {
            const Index index = unit_offset + u;
            const Scalar amount = std::min(1.0, hi(index));
            if (amount < lo(index)) continue;
            x[static_cast<std::size_t>(index)] = amount;
            total_activation += amount;
            total_capacity += amount * (300.0 + 100.0 * u);
        }
        if (total_activation + 1e-9 < 1.0 ||
            total_capacity + 1e-9 < total_demand) {
            result.status = vikalp::SolveStatus::Infeasible;
            return result;
        }

        for (int p = 0; p < product_count; ++p) {
            const Scalar demand = x[static_cast<std::size_t>(product_offset + p)];
            Scalar assigned = 0.0;
            for (int c = 0; c < crude_count; ++c) {
                const Scalar share = c + 1 == crude_count
                    ? demand - assigned
                    : demand * crude_volume[static_cast<std::size_t>(c)] /
                          std::max(total_demand, 1.0);
                const Index blend = blend_offset + c * product_count + p;
                if (share > hi(blend) + 1e-8) {
                    result.status = vikalp::SolveStatus::Infeasible;
                    return result;
                }
                x[static_cast<std::size_t>(blend)] = share;
                assigned += share;
            }
        }

        if (std::any_of(x.begin(), x.end(), [](Scalar value) {
                return !std::isfinite(value);
            })) {
            result.status = vikalp::SolveStatus::NumericalFailure;
            return result;
        }
        result.status = vikalp::SolveStatus::Feasible;
        result.primal_solution = std::move(x);
        result.objective_value = evaluate_objective(model, result.primal_solution);
        return result;
    }

private:
    vikalp::RefineryConfig config_;
};

BenchmarkRecord run_internal(const vikalp::RefineryDemoCase &demo,
                             const Model &model) {
    vikalp::SolverOptions options;
    options.node_limit = 2000;
    options.time_limit_seconds = 5.0;
    RefineryOracle oracle(demo.config);
    const auto start = std::chrono::steady_clock::now();
    const auto result = vikalp::solve(model, oracle, options);
    const auto milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return {demo.name, "vikalp", status_name(result.status), model.num_variables(),
            model.num_constraints(), result.nodes, milliseconds,
            result.objective_value, result.dual_bound, result.relative_gap};
}

std::string shell_quote(const std::string &value) {
#ifdef _WIN32
    return "\"" + value + "\"";
#else
    std::string quoted = "'";
    for (char character : value) {
        if (character == '\'') quoted += "'\\''";
        else quoted += character;
    }
    return quoted + "'";
#endif
}

std::filesystem::path write_model_file(const Model &model, const std::string &name) {
    const auto path = std::filesystem::temp_directory_path() /
                      ("vikalp_" + std::to_string(std::hash<std::string>{}(name)) + ".model");
    std::ofstream output(path);
    output << "VIKALP_MODEL_V1\n" << model.name << '\n';
    output << model.num_variables() << ' ' << model.num_constraints() << '\n';
    for (auto value : model.linear_objective) output << value << ' ';
    output << '\n';
    for (auto value : model.variable_lower) output << value << ' ';
    output << '\n';
    for (auto value : model.variable_upper) output << value << ' ';
    output << '\n';
    for (auto value : model.variable_types) output << static_cast<int>(value) << ' ';
    output << '\n';
    for (std::size_t row = 0; row < model.constraint_lower.size(); ++row) {
        output << model.constraint_lower[row] << ' ' << model.constraint_upper[row] << ' '
               << model.constraint_matrix.pattern.row_offsets[row + 1] -
                      model.constraint_matrix.pattern.row_offsets[row] << ' ';
        for (Index position = model.constraint_matrix.pattern.row_offsets[row];
             position < model.constraint_matrix.pattern.row_offsets[row + 1]; ++position) {
            const auto pos = static_cast<std::size_t>(position);
            output << model.constraint_matrix.pattern.column_indices[pos] << ':'
                   << model.constraint_matrix.values[pos] << ' ';
        }
        output << '\n';
    }
    return path;
}

BenchmarkRecord run_external(const vikalp::RefineryDemoCase &demo,
                             const Model &model, const std::string &command) {
    BenchmarkRecord record{demo.name, "external", "ExternalFailure", model.num_variables(),
                           model.num_constraints()};
    const auto model_path = write_model_file(model, demo.name);
    const std::string full_command = command + " " + shell_quote(model_path.string()) + " 2>&1";
#ifdef _WIN32
    FILE *pipe = _popen(full_command.c_str(), "r");
#else
    FILE *pipe = popen(full_command.c_str(), "r");
#endif
    if (!pipe) {
        std::filesystem::remove(model_path);
        return record;
    }
    std::string output;
    char buffer[256];
    while (std::fgets(buffer, sizeof(buffer), pipe)) output += buffer;
#ifdef _WIN32
    const int exit_code = _pclose(pipe);
#else
    const int exit_code = pclose(pipe);
#endif
    std::filesystem::remove(model_path);
    if (exit_code == 0) {
        std::istringstream parsed(output);
        parsed >> record.status >> record.objective >> record.dual_bound;
        if (parsed) {
            record.relative_gap = std::max(0.0, record.objective - record.dual_bound) /
                                  std::max(1.0, std::abs(record.objective));
        } else {
            record.status = "ExternalFailure";
        }
    }
    return record;
}

std::string csv_quote(const std::string &value) {
    std::string result = "\"";
    for (char character : value) {
        if (character == '"') result += "\"\"";
        else result += character;
    }
    return result + "\"";
}

void write_csv(const std::filesystem::path &path,
               const std::vector<BenchmarkRecord> &records) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << "case,source,status,variables,constraints,nodes,milliseconds,objective,dual_bound,relative_gap\n";
    for (const auto &record : records) {
        output << csv_quote(record.name) << ',' << record.source << ',' << record.status << ','
               << record.variables << ',' << record.constraints << ',' << record.nodes << ','
               << std::setprecision(17) << record.milliseconds << ',' << record.objective << ','
               << record.dual_bound << ',' << record.relative_gap << '\n';
    }
}

void write_json(const std::filesystem::path &path,
                const std::vector<BenchmarkRecord> &records) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << "[\n";
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto &record = records[i];
        output << "  {\"case\":\"" << record.name << "\",\"source\":\""
               << record.source << "\",\"status\":\"" << record.status
               << "\",\"variables\":" << record.variables
               << ",\"constraints\":" << record.constraints
               << ",\"nodes\":" << record.nodes
               << ",\"milliseconds\":" << record.milliseconds;
        auto number = [&](const char *key, Scalar value) {
            output << ",\"" << key << "\":";
            if (std::isfinite(value)) output << std::setprecision(17) << value;
            else output << "null";
        };
        number("objective", record.objective);
        number("dual_bound", record.dual_bound);
        number("relative_gap", record.relative_gap);
        output << "}" << (i + 1 == records.size() ? "\n" : ",\n");
    }
    output << "]\n";
}

} // namespace

int main(int argc, char **argv) {
    std::string csv_path;
    std::string json_path;
    std::string external_command;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if ((argument == "--csv" || argument == "--json" ||
             argument == "--external") && i + 1 < argc) {
            auto &target = argument == "--csv" ? csv_path :
                           argument == "--json" ? json_path : external_command;
            target = argv[++i];
        } else if (argument == "--help") {
            std::cout << "usage: vikalp_benchmark [--csv PATH] [--json PATH] [--external COMMAND]\n";
            return 0;
        } else {
            std::cerr << "usage: vikalp_benchmark [--csv PATH] [--json PATH] [--external COMMAND]\n";
            return 2;
        }
    }

    std::vector<BenchmarkRecord> records;
    for (const auto &demo : vikalp::refinery_demo_family()) {
        const auto model = vikalp::build_refinery_model(demo.config);
        records.push_back(external_command.empty()
            ? run_internal(demo, model)
            : run_external(demo, model, external_command));
    }

    std::cout << "case | source | vars | rows | nodes | status | objective | rel_gap\n";
    for (const auto &record : records) {
        std::cout << record.name << " | " << record.source << " | " << record.variables
                  << " | " << record.constraints << " | " << record.nodes << " | "
                  << record.status << " | " << record.objective << " | "
                  << record.relative_gap << '\n';
    }
    if (!csv_path.empty()) write_csv(csv_path, records);
    if (!json_path.empty()) write_json(json_path, records);
    return std::any_of(records.begin(), records.end(), [](const auto &record) {
        return record.status == "ExternalFailure";
    }) ? 1 : 0;
}
