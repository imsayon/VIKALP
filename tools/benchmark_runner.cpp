// Flow D — Benchmark runner for VIKALP Mixed-Integer Optimization.
//
// Runs parametric benchmark families (refinery small/medium/large/xl)
// and reports execution statistics: node throughput, dual bound progression,
// optimality gap, and solution time.

#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/RelaxationOracle.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/SolverOptions.hpp"
#include "vikalp/examples/RefineryModel.hpp"
#include "vikalp/solver/Solver.hpp"
#include "vikalp/solver/mip/BranchAndBound.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

// Deterministic mock continuous relaxation oracle for benchmarking
class BenchmarkOracle final : public vikalp::RelaxationOracle {
public:
    [[nodiscard]] vikalp::SolveResult solve(
        const vikalp::Model &model,
        std::span<const vikalp::Scalar> lo_override,
        std::span<const vikalp::Scalar> hi_override,
        std::span<const vikalp::Scalar> /*warm_start*/,
        const vikalp::SolverOptions &/*options*/) const override {
        
        vikalp::SolveResult res;
        res.status = vikalp::SolveStatus::Optimal;
        const auto n = static_cast<std::size_t>(model.num_variables());
        res.primal_solution.assign(n, 0.0);

        auto eff_lo = lo_override.empty() ? model.variable_lower : std::vector<vikalp::Scalar>(lo_override.begin(), lo_override.end());
        auto eff_hi = hi_override.empty() ? model.variable_upper : std::vector<vikalp::Scalar>(hi_override.begin(), hi_override.end());

        vikalp::Scalar obj = model.objective_offset;
        for (std::size_t i = 0; i < n; ++i) {
            vikalp::Scalar val = 0.0;
            if (model.linear_objective[i] < 0.0) {
                val = std::min(eff_hi[i], 100.0);
            } else {
                val = eff_lo[i];
            }
            // If it's integer/binary and bounds are fixed or integer, keep it integer
            if (model.variable_types[i] != vikalp::VariableType::Continuous) {
                if (std::abs(eff_hi[i] - eff_lo[i]) < 1e-6) {
                    val = std::round(eff_lo[i]);
                } else {
                    // fractional relaxation point
                    val = eff_lo[i] + 0.5 * std::min(1.0, eff_hi[i] - eff_lo[i]);
                }
            }
            res.primal_solution[i] = val;
            obj += model.linear_objective[i] * val;
        }
        res.objective_value = obj;
        return res;
    }
};

void run_benchmark_case(const std::string &name, const vikalp::Model &model) {
    std::cout << "| " << std::left << std::setw(18) << name
              << " | " << std::right << std::setw(8) << model.num_variables()
              << " | " << std::setw(8) << model.num_constraints();

    BenchmarkOracle oracle;
    vikalp::SolverOptions opts;
    opts.node_limit = 2000;
    opts.time_limit_seconds = 5.0;

    auto t0 = std::chrono::high_resolution_clock::now();
    vikalp::SolveResult res = vikalp::solve(model, oracle, opts);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const char *status_str = "Unknown";
    switch (res.status) {
        case vikalp::SolveStatus::Optimal: status_str = "Optimal"; break;
        case vikalp::SolveStatus::Feasible: status_str = "Feasible"; break;
        case vikalp::SolveStatus::Infeasible: status_str = "Infeasible"; break;
        case vikalp::SolveStatus::NodeLimit: status_str = "NodeLimit"; break;
        case vikalp::SolveStatus::TimeLimit: status_str = "TimeLimit"; break;
        default: status_str = "Other"; break;
    }

    std::cout << " | " << std::setw(8) << res.nodes
              << " | " << std::setw(10) << std::fixed << std::setprecision(2) << ms
              << " | " << std::setw(10) << status_str
              << " | " << std::setw(12) << std::setprecision(4) << res.objective_value
              << " | " << std::setw(10) << std::scientific << std::setprecision(2) << res.relative_gap
              << " |\n";
}

} // namespace

int main() {
    std::cout << "\n========================================================================================\n";
    std::cout << "                VIKALP Flow D — Mixed-Integer Benchmark Suite                         \n";
    std::cout << "========================================================================================\n\n";

    std::cout << "| " << std::left << std::setw(18) << "Case Name"
              << " | " << std::right << std::setw(8) << "Vars"
              << " | " << std::setw(8) << "Rows"
              << " | " << std::setw(8) << "Nodes"
              << " | " << std::setw(10) << "Time (ms)"
              << " | " << std::setw(10) << "Status"
              << " | " << std::setw(12) << "Obj Value"
              << " | " << std::setw(10) << "Rel Gap"
              << " |\n";
    std::cout << "|:-------------------|---------:|---------:|---------:|-----------:|-----------:|-------------:|-----------:|\n";

    run_benchmark_case("Refinery Small", vikalp::build_refinery_model(vikalp::refinery_small()));
    run_benchmark_case("Refinery Medium", vikalp::build_refinery_model(vikalp::refinery_medium()));
    run_benchmark_case("Refinery Large", vikalp::build_refinery_model(vikalp::refinery_large()));
    
    auto miqp_cfg = vikalp::refinery_small();
    miqp_cfg.include_quadratic = true;
    run_benchmark_case("Refinery MIQP", vikalp::build_refinery_model(miqp_cfg));

    std::cout << "\nBenchmark run completed successfully.\n";
    return 0;
}
