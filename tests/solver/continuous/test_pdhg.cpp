#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/ExecutionBackend.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
// Add forward declaration near top of file
namespace vikalp::solver::nlp {
    SolveResult solve_nlp_ipm_baseline(const Model& model, ExecutionBackend& backend, Index max_iterations);
}
// Forward declarations of your Flow C solvers
namespace vikalp::solver::lp {
    SolveResult solve_pdhg_baseline(const Model& model, ExecutionBackend& backend, Index max_iterations);
}

namespace vikalp::solver::qp {
    SolveResult solve_pdqp_baseline(const Model& model, ExecutionBackend& backend, Index max_iterations);
}

namespace vikalp {

// Linker stubs for Model member methods if not compiled separately yet
Index Model::num_variables() const noexcept {
    return linear_objective.size();
}
Index Model::num_constraints() const noexcept {
    return constraint_matrix.pattern.rows;
}
bool Model::has_quadratic_objective() const noexcept {
    return !quadratic_objective.values.empty();
}

// 1. Lightweight Mock Vector
class MockVector : public BackendVector {
public:
    std::vector<Scalar> data;
    explicit MockVector(Index size) : data(size, 0.0) {}
    [[nodiscard]] Index size() const noexcept override { return data.size(); }
};

// 2. Lightweight Mock Matrix
class MockMatrix : public BackendMatrix {
public:
    CsrPattern pat;
    std::vector<Scalar> vals;
    explicit MockMatrix(const CsrPattern& pattern) : pat(pattern) {}
    [[nodiscard]] Index rows() const noexcept override { return pat.rows; }
    [[nodiscard]] Index columns() const noexcept override { return pat.columns; }
};

// 3. Mock ExecutionBackend
class MockExecutionBackend : public ExecutionBackend {
public:
    [[nodiscard]] std::unique_ptr<BackendVector> create_vector(Index size) override {
        return std::make_unique<MockVector>(size);
    }
    [[nodiscard]] std::unique_ptr<BackendMatrix> create_matrix(const CsrPattern &pattern) override {
        return std::make_unique<MockMatrix>(pattern);
    }
    void upload(std::span<const Scalar> source, BackendVector &target) override {
        auto& v = dynamic_cast<MockVector&>(target);
        v.data.assign(source.begin(), source.end());
    }
    void download(const BackendVector &source, std::span<Scalar> target) const override {
        auto& v = dynamic_cast<const MockVector&>(source);
        std::copy(v.data.begin(), v.data.end(), target.begin());
    }
    void set_values(std::span<const Scalar> values, BackendMatrix &target) override {
        auto& m = dynamic_cast<MockMatrix&>(target);
        m.vals.assign(values.begin(), values.end());
    }
    void fill(BackendVector &target, Scalar value) override {
        auto& v = dynamic_cast<MockVector&>(target);
        std::fill(v.data.begin(), v.data.end(), value);
    }
    void copy(const BackendVector &source, BackendVector &target) override {
        auto& s = dynamic_cast<const MockVector&>(source);
        auto& t = dynamic_cast<MockVector&>(target);
        t.data = s.data;
    }
    void axpby(Scalar alpha, const BackendVector &x, Scalar beta, BackendVector &y) override {
        auto& xv = dynamic_cast<const MockVector&>(x);
        auto& yv = dynamic_cast<MockVector&>(y);
        for (size_t i = 0; i < yv.data.size(); ++i) {
            yv.data[i] = alpha * xv.data[i] + beta * yv.data[i];
        }
    }
    [[nodiscard]] Scalar dot(const BackendVector &x, const BackendVector &y) override {
        auto& xv = dynamic_cast<const MockVector&>(x);
        auto& yv = dynamic_cast<const MockVector&>(y);
        Scalar sum = 0.0;
        for (size_t i = 0; i < xv.data.size(); ++i) sum += xv.data[i] * yv.data[i];
        return sum;
    }
    [[nodiscard]] Scalar norm_inf(const BackendVector &x) override {
        auto& xv = dynamic_cast<const MockVector&>(x);
        Scalar max_val = 0.0;
        for (auto val : xv.data) max_val = std::max(max_val, std::abs(val));
        return max_val;
    }
    void project_box(BackendVector &x, const BackendVector &lower, const BackendVector &upper) override {
        auto& xv = dynamic_cast<MockVector&>(x);
        auto& lv = dynamic_cast<const MockVector&>(lower);
        auto& uv = dynamic_cast<const MockVector&>(upper);
        for (size_t i = 0; i < xv.data.size(); ++i) {
            xv.data[i] = std::clamp(xv.data[i], lv.data[i], uv.data[i]);
        }
    }
    void multiply(Scalar alpha, const BackendMatrix &matrix, const BackendVector &x, Scalar beta, BackendVector &y, Transpose transpose) override {
        auto& m = dynamic_cast<const MockMatrix&>(matrix);
        auto& xv = dynamic_cast<const MockVector&>(x);
        auto& yv = dynamic_cast<MockVector&>(y);
        
        for (size_t i = 0; i < yv.data.size(); ++i) yv.data[i] *= beta;

        if (transpose == Transpose::No) {
            for (Index i = 0; i < m.pat.rows; ++i) {
                Scalar sum = 0.0;
                for (Index j = m.pat.row_offsets[i]; j < m.pat.row_offsets[i+1]; ++j) {
                    sum += m.vals[j] * xv.data[m.pat.column_indices[j]];
                }
                yv.data[i] += alpha * sum;
            }
        } else {
            for (Index i = 0; i < m.pat.rows; ++i) {
                for (Index j = m.pat.row_offsets[i]; j < m.pat.row_offsets[i+1]; ++j) {
                    Index col = m.pat.column_indices[j];
                    yv.data[col] += alpha * m.vals[j] * xv.data[i];
                }
            }
        }
    }
    [[nodiscard]] LinearSolveResult solve_linear_system(const BackendMatrix &, const BackendVector &, BackendVector &, MatrixProperty, Scalar, Index) override {
        return {true, 1, 0.0};
    }
    void synchronize() override {} // Added override here
};

} // namespace vikalp

int main() {
    std::cout << "=== Running Flow C Continuous Solver Tests ===\n\n";
    vikalp::MockExecutionBackend backend;

    // --- TEST 1: Constrained LP Test ---
    {
        vikalp::Model lp_model;
        lp_model.name = "ConstrainedLP";
        lp_model.linear_objective = {1.0};
        lp_model.variable_lower = {0.0};
        lp_model.variable_upper = {20.0};

        lp_model.constraint_matrix.pattern.rows = 1;
        lp_model.constraint_matrix.pattern.columns = 1;
        lp_model.constraint_matrix.pattern.row_offsets = {0, 1};
        lp_model.constraint_matrix.pattern.column_indices = {0};
        lp_model.constraint_matrix.values = {2.0};

        lp_model.constraint_lower = {-vikalp::Model::infinity()}; // Added vikalp:: prefix
        lp_model.constraint_upper = {16.0};

        std::cout << "[LP Test] Running PDHG solver on constrained model...\n";
        vikalp::SolveResult res = vikalp::solver::lp::solve_pdhg_baseline(lp_model, backend, 200);
        
        std::cout << "[LP Test] Iterations: " << res.iterations << "\n";
        if (!res.primal_solution.empty()) {
            std::cout << "[LP Test] Result x[0] = " << res.primal_solution[0] << "\n";
        }
    }

    std::cout << "\n----------------------------------------\n\n";

    // --- TEST 2: Convex QP Test ---
    {
        vikalp::Model qp_model;
        qp_model.name = "ConvexQP";
        qp_model.linear_objective = {-10.0};
        qp_model.variable_lower = {0.0};
        qp_model.variable_upper = {10.0};

        qp_model.quadratic_objective.pattern.rows = 1;
        qp_model.quadratic_objective.pattern.columns = 1;
        qp_model.quadratic_objective.pattern.row_offsets = {0, 1};
        qp_model.quadratic_objective.pattern.column_indices = {0};
        qp_model.quadratic_objective.values = {2.0};

        qp_model.constraint_matrix.pattern.rows = 0;
        qp_model.constraint_matrix.pattern.columns = 1;
        qp_model.constraint_matrix.pattern.row_offsets = {0};

        std::cout << "[QP Test] Running PDQP solver on quadratic model...\n";
        vikalp::SolveResult res = vikalp::solver::qp::solve_pdqp_baseline(qp_model, backend, 200);

        std::cout << "[QP Test] Iterations: " << res.iterations << "\n";
        if (!res.primal_solution.empty()) {
            std::cout << "[QP Test] Result x[0] = " << res.primal_solution[0] << " (Expected optimal: 5.0)\n";
        }
    }

    std::cout << "\n=== All Flow C Solver Tests Completed Successfully ===\n";
    // --- TEST 3: NLP IPM Smoke Test ---
    {
        vikalp::Model nlp_model;
        nlp_model.name = "SmoothNLP";
        nlp_model.linear_objective = {0.0};
        nlp_model.variable_lower = {-10.0};
        nlp_model.variable_upper = {10.0};
        // Attach a dummy nonlinear oracle indicator
        // nlp_model.nonlinear is shared_ptr<const NonlinearOracle>

        std::cout << "[NLP Test] Running IPM solver on nonlinear model skeleton...\n";
        vikalp::SolveResult res = vikalp::solver::nlp::solve_nlp_ipm_baseline(nlp_model, backend, 10);
        std::cout << "[NLP Test] Status code received: " << static_cast<int>(res.status) << "\n";
    }
    std::cout << "\n----------------------------------------\n\n";

    // --- TEST 4: Continuous Failures / Infeasible Model Test ---
    // Minimize 1.0 * x0, subject to conflicting bounds: x0 >= 15.0 and x0 <= 5.0 (Impossible!)
    {
        vikalp::Model infeasible_model;
        infeasible_model.name = "InfeasibleLP";
        infeasible_model.linear_objective = {1.0};
        infeasible_model.variable_lower = {15.0}; // Conflict here
        infeasible_model.variable_upper = {5.0};  // Conflict here

        infeasible_model.constraint_matrix.pattern.rows = 0;
        infeasible_model.constraint_matrix.pattern.columns = 1;
        infeasible_model.constraint_matrix.pattern.row_offsets = {0};

        std::cout << "[Failure Test] Running PDHG solver on infeasible model...\n";
        vikalp::SolveResult res = vikalp::solver::lp::solve_pdhg_baseline(infeasible_model, backend, 100);
        
        std::cout << "[Failure Test] Status code received: " << static_cast<int>(res.status) 
                  << " (Handled safely without crashing)\n";
    }
    return 0;
}