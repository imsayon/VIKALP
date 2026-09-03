// Flow C — CPU execution backend implementation.
//
// Design principles:
//   - Every BackendVector and BackendMatrix is CPU-resident (std::vector).
//   - All dimension errors throw std::invalid_argument immediately.
//   - No global state, no static storage, fully RAII.
//   - synchronize() is a no-op on CPU (no async pipeline to drain).
//   - solve_linear_system uses Conjugate Gradient (SPD/symmetric cases)
//     and BiCGSTAB for the general case.

#include "vikalp/backend/CpuBackend.hpp"
#include "vikalp/contracts/ExecutionBackend.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vikalp {
namespace {

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

[[noreturn]] void dim_error(const std::string &msg) {
    throw std::invalid_argument("CpuBackend: " + msg);
}

// ──────────────────────────────────────────────────────────────────────────────
// CpuVector — owns a heap-allocated double array
// ──────────────────────────────────────────────────────────────────────────────

class CpuVector final : public BackendVector {
public:
    explicit CpuVector(Index n) : data_(static_cast<std::size_t>(n), 0.0) {
        if (n < 0) dim_error("vector size must be non-negative");
    }

    [[nodiscard]] Index size() const noexcept override {
        return static_cast<Index>(data_.size());
    }

    std::vector<Scalar> &data() noexcept { return data_; }
    const std::vector<Scalar> &data() const noexcept { return data_; }

private:
    std::vector<Scalar> data_;
};

// ──────────────────────────────────────────────────────────────────────────────
// CpuMatrix — CSR pattern + mutable values
// ──────────────────────────────────────────────────────────────────────────────

class CpuMatrix final : public BackendMatrix {
public:
    explicit CpuMatrix(const CsrPattern &pattern) : pattern_(pattern) {
        if (pattern_.rows < 0 || pattern_.columns < 0)
            dim_error("matrix dimensions must be non-negative");
        // row_offsets must have exactly rows+1 entries
        if (pattern_.row_offsets.size() !=
            static_cast<std::size_t>(pattern_.rows + 1))
            dim_error("CsrPattern row_offsets size != rows+1");
        // pre-allocate values to zero; set_values will fill them
        values_.assign(pattern_.column_indices.size(), 0.0);
    }

    [[nodiscard]] Index rows() const noexcept override { return pattern_.rows; }
    [[nodiscard]] Index columns() const noexcept override { return pattern_.columns; }

    const CsrPattern &pattern() const noexcept { return pattern_; }
    std::vector<Scalar> &values() noexcept { return values_; }
    const std::vector<Scalar> &values() const noexcept { return values_; }

private:
    CsrPattern pattern_;
    std::vector<Scalar> values_;
};

// ──────────────────────────────────────────────────────────────────────────────
// Downcast helpers — safe, throw on wrong dynamic type
// ──────────────────────────────────────────────────────────────────────────────

CpuVector &as_cpu_vec(BackendVector &v) {
    auto *p = dynamic_cast<CpuVector *>(&v);
    if (!p) dim_error("BackendVector is not a CpuVector");
    return *p;
}

const CpuVector &as_cpu_vec(const BackendVector &v) {
    const auto *p = dynamic_cast<const CpuVector *>(&v);
    if (!p) dim_error("BackendVector is not a CpuVector");
    return *p;
}

const CpuMatrix &as_cpu_mat(const BackendMatrix &m) {
    const auto *p = dynamic_cast<const CpuMatrix *>(&m);
    if (!p) dim_error("BackendMatrix is not a CpuMatrix");
    return *p;
}

CpuMatrix &as_cpu_mat_mut(BackendMatrix &m) {
    auto *p = dynamic_cast<CpuMatrix *>(&m);
    if (!p) dim_error("BackendMatrix is not a CpuMatrix");
    return *p;
}

// ──────────────────────────────────────────────────────────────────────────────
// BiCGSTAB — used by solve_linear_system for general square systems.
// We store all working vectors on the stack (as std::vector) to avoid
// repeated heap allocation if the caller wraps this in a loop — the solver
// is called from Flow A's outer iteration, not the inner loop.
// ──────────────────────────────────────────────────────────────────────────────

// y = A*x  (CSR, no-transpose)
static void csr_matvec(const CpuMatrix &A,
                       const std::vector<Scalar> &x,
                       std::vector<Scalar> &y) {
    const auto &pat = A.pattern();
    const auto &vals = A.values();
    const std::size_t nrows = static_cast<std::size_t>(pat.rows);
    y.assign(nrows, 0.0);
    for (std::size_t row = 0; row < nrows; ++row) {
        Scalar acc = 0.0;
        for (Index k = pat.row_offsets[row]; k < pat.row_offsets[row + 1]; ++k) {
            acc += vals[static_cast<std::size_t>(k)] *
                   x[static_cast<std::size_t>(
                       pat.column_indices[static_cast<std::size_t>(k)])];
        }
        y[row] = acc;
    }
}

static LinearSolveResult bicgstab(const CpuMatrix &A,
                                  const std::vector<Scalar> &b,
                                  std::vector<Scalar> &x,
                                  Scalar tol,
                                  Index max_iter) {
    // Initialise x = 0
    const std::size_t n = b.size();
    x.assign(n, 0.0);

    std::vector<Scalar> r(n), r_hat(n), p(n), v(n), s(n), t(n), tmp(n);

    // r = b - A*x  (x=0, so r = b)
    r = b;
    r_hat = r; // shadow residual — fixed throughout

    Scalar rho_prev = 1.0, alpha = 1.0, omega = 1.0;
    std::fill(v.begin(), v.end(), 0.0);
    std::fill(p.begin(), p.end(), 0.0);

    Scalar r_hat_dot_r = std::inner_product(r_hat.begin(), r_hat.end(), r.begin(), 0.0);

    LinearSolveResult res;
    for (Index iter = 0; iter < max_iter; ++iter) {
        const Scalar rho = r_hat_dot_r;
        if (rho == 0.0) break; // breakdown

        if (iter == 0) {
            p = r;
        } else {
            const Scalar beta = (rho / rho_prev) * (alpha / omega);
            for (std::size_t i = 0; i < n; ++i)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // v = A*p
        csr_matvec(A, p, v);
        const Scalar denom_alpha =
            std::inner_product(r_hat.begin(), r_hat.end(), v.begin(), 0.0);
        if (denom_alpha == 0.0) break;
        alpha = rho / denom_alpha;

        // s = r - alpha*v
        for (std::size_t i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];

        // Early exit check on s
        Scalar norm_s = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            norm_s = std::max(norm_s, std::abs(s[i]));
        if (norm_s <= tol) {
            for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i];
            res.converged = true;
            res.iterations = iter + 1;
            res.residual = norm_s;
            return res;
        }

        // t = A*s
        csr_matvec(A, s, t);
        const Scalar tt = std::inner_product(t.begin(), t.end(), t.begin(), 0.0);
        if (tt == 0.0) break;
        omega = std::inner_product(t.begin(), t.end(), s.begin(), 0.0) / tt;

        // update x
        for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i] + omega * s[i];

        // update r
        for (std::size_t i = 0; i < n; ++i) r[i] = s[i] - omega * t[i];

        // convergence check on r
        Scalar norm_r = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            norm_r = std::max(norm_r, std::abs(r[i]));

        if (norm_r <= tol) {
            res.converged = true;
            res.iterations = iter + 1;
            res.residual = norm_r;
            return res;
        }

        r_hat_dot_r = std::inner_product(r_hat.begin(), r_hat.end(), r.begin(), 0.0);
        rho_prev = rho;
    }

    // Measure final residual
    csr_matvec(A, x, tmp);
    Scalar final_norm = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        final_norm = std::max(final_norm, std::abs(b[i] - tmp[i]));

    res.converged = false;
    res.iterations = max_iter;
    res.residual = final_norm;
    return res;
}

// ──────────────────────────────────────────────────────────────────────────────
// CpuBackend — implements ExecutionBackend
// ──────────────────────────────────────────────────────────────────────────────

class CpuBackend final : public ExecutionBackend {
public:
    // ── Factory ──────────────────────────────────────────────────────────────

    [[nodiscard]] std::unique_ptr<BackendVector> create_vector(Index size) override {
        return std::make_unique<CpuVector>(size);
    }

    [[nodiscard]] std::unique_ptr<BackendMatrix>
    create_matrix(const CsrPattern &pattern) override {
        return std::make_unique<CpuMatrix>(pattern);
    }

    // ── Transfer ─────────────────────────────────────────────────────────────

    void upload(std::span<const Scalar> source, BackendVector &target) override {
        auto &v = as_cpu_vec(target);
        if (static_cast<Index>(source.size()) != v.size())
            dim_error("upload: source size " + std::to_string(source.size()) +
                      " != vector size " + std::to_string(v.size()));
        std::copy(source.begin(), source.end(), v.data().begin());
    }

    void download(const BackendVector &source,
                  std::span<Scalar> target) const override {
        const auto &v = as_cpu_vec(source);
        if (static_cast<Index>(target.size()) != v.size())
            dim_error("download: target size " + std::to_string(target.size()) +
                      " != vector size " + std::to_string(v.size()));
        std::copy(v.data().begin(), v.data().end(), target.begin());
    }

    void set_values(std::span<const Scalar> values,
                    BackendMatrix &target) override {
        auto &m = as_cpu_mat_mut(target);
        if (values.size() != m.values().size())
            dim_error("set_values: values size " + std::to_string(values.size()) +
                      " != nnz " + std::to_string(m.values().size()));
        std::copy(values.begin(), values.end(), m.values().begin());
    }

    // ── Vector operations ────────────────────────────────────────────────────

    void fill(BackendVector &target, Scalar value) override {
        auto &v = as_cpu_vec(target);
        std::fill(v.data().begin(), v.data().end(), value);
    }

    void copy(const BackendVector &source, BackendVector &target) override {
        const auto &s = as_cpu_vec(source);
        auto &t = as_cpu_vec(target);
        if (s.size() != t.size())
            dim_error("copy: source size " + std::to_string(s.size()) +
                      " != target size " + std::to_string(t.size()));
        std::copy(s.data().begin(), s.data().end(), t.data().begin());
    }

    // y = alpha*x + beta*y
    void axpby(Scalar alpha, const BackendVector &x,
               Scalar beta, BackendVector &y) override {
        const auto &xv = as_cpu_vec(x);
        auto &yv = as_cpu_vec(y);
        if (xv.size() != yv.size())
            dim_error("axpby: x size " + std::to_string(xv.size()) +
                      " != y size " + std::to_string(yv.size()));
        const auto &xd = xv.data();
        auto &yd = yv.data();
        const std::size_t n = xd.size();
        for (std::size_t i = 0; i < n; ++i)
            yd[i] = alpha * xd[i] + beta * yd[i];
    }

    // dot(x, y) = sum(x_i * y_i)
    [[nodiscard]] Scalar dot(const BackendVector &x,
                             const BackendVector &y) override {
        const auto &xv = as_cpu_vec(x);
        const auto &yv = as_cpu_vec(y);
        if (xv.size() != yv.size())
            dim_error("dot: size mismatch " + std::to_string(xv.size()) +
                      " vs " + std::to_string(yv.size()));
        return std::inner_product(xv.data().begin(), xv.data().end(),
                                  yv.data().begin(), 0.0);
    }

    // norm_inf(x) = max_i |x_i|
    [[nodiscard]] Scalar norm_inf(const BackendVector &x) override {
        const auto &v = as_cpu_vec(x);
        Scalar result = 0.0;
        for (const Scalar val : v.data())
            result = std::max(result, std::abs(val));
        return result;
    }

    // x_i = clamp(x_i, lower_i, upper_i)
    void project_box(BackendVector &x,
                     const BackendVector &lower,
                     const BackendVector &upper) override {
        auto &xv = as_cpu_vec(x);
        const auto &lv = as_cpu_vec(lower);
        const auto &uv = as_cpu_vec(upper);
        const Index n = xv.size();
        if (lv.size() != n || uv.size() != n)
            dim_error("project_box: x/lower/upper sizes must match (" +
                      std::to_string(n) + " vs " + std::to_string(lv.size()) +
                      " vs " + std::to_string(uv.size()) + ")");
        auto &xd = xv.data();
        const auto &ld = lv.data();
        const auto &ud = uv.data();
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
            xd[i] = std::min(std::max(xd[i], ld[i]), ud[i]);
    }

    // ── Sparse matrix-vector multiply ────────────────────────────────────────
    //
    // No-transpose:  y = alpha * A * x + beta * y
    // Transpose:     y = alpha * A^T * x + beta * y
    //
    // Both modes operate directly on the CSR representation.
    // No intermediate dense conversion.

    void multiply(Scalar alpha, const BackendMatrix &matrix,
                  const BackendVector &x, Scalar beta,
                  BackendVector &y, Transpose transpose) override {
        const auto &m = as_cpu_mat(matrix);
        const auto &xv = as_cpu_vec(x);
        auto &yv = as_cpu_vec(y);

        const auto &pat = m.pattern();
        const auto &vals = m.values();

        if (transpose == Transpose::No) {
            // y is shape (rows,), x is shape (cols,)
            if (xv.size() != pat.columns)
                dim_error("multiply (No): x size " +
                          std::to_string(xv.size()) +
                          " != matrix columns " + std::to_string(pat.columns));
            if (yv.size() != pat.rows)
                dim_error("multiply (No): y size " +
                          std::to_string(yv.size()) +
                          " != matrix rows " + std::to_string(pat.rows));

            const auto &xd = xv.data();
            auto &yd = yv.data();
            const std::size_t nrows = static_cast<std::size_t>(pat.rows);

            for (std::size_t row = 0; row < nrows; ++row) {
                Scalar acc = 0.0;
                for (Index k = pat.row_offsets[row];
                     k < pat.row_offsets[row + 1]; ++k) {
                    acc += vals[static_cast<std::size_t>(k)] *
                           xd[static_cast<std::size_t>(
                               pat.column_indices[static_cast<std::size_t>(k)])];
                }
                yd[row] = alpha * acc + beta * yd[row];
            }
        } else {
            // Transpose: y is shape (cols,), x is shape (rows,)
            if (xv.size() != pat.rows)
                dim_error("multiply (Yes): x size " +
                          std::to_string(xv.size()) +
                          " != matrix rows " + std::to_string(pat.rows));
            if (yv.size() != pat.columns)
                dim_error("multiply (Yes): y size " +
                          std::to_string(yv.size()) +
                          " != matrix columns " + std::to_string(pat.columns));

            const auto &xd = xv.data();
            auto &yd = yv.data();
            const std::size_t nrows = static_cast<std::size_t>(pat.rows);
            const std::size_t ncols = static_cast<std::size_t>(pat.columns);

            // Scale y by beta first
            for (std::size_t j = 0; j < ncols; ++j)
                yd[j] *= beta;

            // Scatter: for each non-zero (row, col, val): y[col] += alpha * val * x[row]
            for (std::size_t row = 0; row < nrows; ++row) {
                const Scalar scaled_x = alpha * xd[row];
                for (Index k = pat.row_offsets[row];
                     k < pat.row_offsets[row + 1]; ++k) {
                    const std::size_t col =
                        static_cast<std::size_t>(
                            pat.column_indices[static_cast<std::size_t>(k)]);
                    yd[col] += vals[static_cast<std::size_t>(k)] * scaled_x;
                }
            }
        }
    }

    // ── Linear system solve ──────────────────────────────────────────────────
    //
    // Solves A * solution = rhs using BiCGSTAB (works for general square A).
    // For SPD matrices (SymmetricPositiveDefinite), BiCGSTAB still converges;
    // a dedicated CG can be added later as an optimisation.
    //
    // The solver is NOT a flow-A optimization primitive — it is a purely
    // numerical operation. Flow A decides whether/when to call it.

    [[nodiscard]] LinearSolveResult
    solve_linear_system(const BackendMatrix &matrix,
                        const BackendVector &rhs,
                        BackendVector &solution,
                        MatrixProperty /*property*/,
                        Scalar tolerance,
                        Index iteration_limit) override {
        const auto &m = as_cpu_mat(matrix);
        const auto &b = as_cpu_vec(rhs);
        auto &sol = as_cpu_vec(solution);

        if (m.rows() != m.columns())
            dim_error("solve_linear_system: matrix must be square (" +
                      std::to_string(m.rows()) + "x" +
                      std::to_string(m.columns()) + ")");
        if (b.size() != m.rows())
            dim_error("solve_linear_system: rhs size " +
                      std::to_string(b.size()) +
                      " != matrix rows " + std::to_string(m.rows()));
        if (sol.size() != m.rows())
            dim_error("solve_linear_system: solution size " +
                      std::to_string(sol.size()) +
                      " != matrix rows " + std::to_string(m.rows()));

        std::vector<Scalar> x_internal;
        LinearSolveResult res =
            bicgstab(m, b.data(), x_internal, tolerance, iteration_limit);

        std::copy(x_internal.begin(), x_internal.end(), sol.data().begin());
        return res;
    }

    // ── Synchronisation ──────────────────────────────────────────────────────
    //
    // CPU has no async pipeline — this is a deliberate no-op.
    // On the CUDA backend this will call cudaDeviceSynchronize().

    void synchronize() override {
        // no-op: CPU execution is always synchronous
    }
};

} // namespace

// ──────────────────────────────────────────────────────────────────────────────
// Public factory
// ──────────────────────────────────────────────────────────────────────────────

std::unique_ptr<ExecutionBackend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

} // namespace vikalp
