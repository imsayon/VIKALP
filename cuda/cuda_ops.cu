// Flow B — CUDA execution backend implementation.
//
// Architecture rules:
//   - Solver policy (PDHG, step sizes, convergence) stays OUTSIDE this file.
//   - This file provides only numerical primitives requested by the solver.
//   - All device memory is owned by CudaVector/CudaMatrix (RAII).
//   - Model data is uploaded ONCE; active vectors stay resident on GPU.
//   - Operation workspaces are retained after growing to the required size.
//   - All CUDA API calls are checked; errors throw std::runtime_error.
//   - Synchronize is called deliberately, not speculatively.
//   - FP64 throughout; no mixed precision.

#include "vikalp/backend/CudaBackend.hpp"
#include "vikalp/contracts/ExecutionBackend.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vikalp {
namespace {

// ──────────────────────────────────────────────────────────────────────────────
// Error checking helpers
// ──────────────────────────────────────────────────────────────────────────────

void check_cuda(cudaError_t err, const char *context) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CudaBackend [") + context +
                                 "]: " + cudaGetErrorString(err));
}

void check_cusparse(cusparseStatus_t stat, const char *context) {
    if (stat != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string("CudaBackend cuSPARSE [") +
                                 context + "]: status " +
                                 std::to_string(static_cast<int>(stat)));
}

void validate_pattern(const CsrPattern &pattern) {
    if (pattern.rows < 0 || pattern.columns < 0)
        throw std::invalid_argument("CudaBackend: matrix dimensions must be non-negative");
    if (pattern.rows == std::numeric_limits<Index>::max() ||
        pattern.row_offsets.size() != static_cast<std::size_t>(pattern.rows) + 1)
        throw std::invalid_argument("CudaBackend: CsrPattern row_offsets size != rows+1");
    if (pattern.row_offsets.front() != 0 ||
        pattern.row_offsets.back() != static_cast<Index>(pattern.column_indices.size()))
        throw std::invalid_argument(
            "CudaBackend: CsrPattern row_offsets do not describe column_indices");

    for (Index row = 0; row < pattern.rows; ++row) {
        const Index begin = pattern.row_offsets[static_cast<std::size_t>(row)];
        const Index end = pattern.row_offsets[static_cast<std::size_t>(row + 1)];
        if (begin < 0 || begin > end ||
            end > static_cast<Index>(pattern.column_indices.size()))
            throw std::invalid_argument(
                "CudaBackend: CsrPattern row_offsets must be ordered and in range");
        Index previous = -1;
        for (Index position = begin; position < end; ++position) {
            const Index column =
                pattern.column_indices[static_cast<std::size_t>(position)];
            if (column < 0 || column >= pattern.columns || column <= previous)
                throw std::invalid_argument(
                    "CudaBackend: CsrPattern columns must be ordered and in range");
            previous = column;
        }
    }
}

struct DnVecDescriptor {
    cusparseDnVecDescr_t value = nullptr;
    ~DnVecDescriptor() {
        if (value) cusparseDestroyDnVec(value);
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Device memory RAII wrapper
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
struct DeviceBuffer {
    T *ptr = nullptr;
    std::size_t count = 0;

    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t n) : count(n) {
        if (n > 0)
            check_cuda(cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)),
                       "DeviceBuffer cudaMalloc");
    }

    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr); // no-throw in destructor
    }

    // Non-copyable, movable
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr;
        o.count = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
        if (this != &o) {
            if (ptr) cudaFree(ptr);
            ptr = o.ptr;
            count = o.count;
            o.ptr = nullptr;
            o.count = 0;
        }
        return *this;
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// CUDA custom kernels — simple vector primitives
// ──────────────────────────────────────────────────────────────────────────────

// y[i] = alpha*x[i] + beta*y[i]
__global__ void kernel_axpby(double alpha, const double *__restrict__ x,
                              double beta, double *__restrict__ y,
                              int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + beta * y[i];
}

// fill: y[i] = val
__global__ void kernel_fill(double *__restrict__ y, double val, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = val;
}

// project_box: x[i] = clamp(x[i], lo[i], hi[i])
__global__ void kernel_project_box(double *__restrict__ x,
                                   const double *__restrict__ lo,
                                   const double *__restrict__ hi,
                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double v = x[i];
        if (v < lo[i]) v = lo[i];
        if (v > hi[i]) v = hi[i];
        x[i] = v;
    }
}

// abs-max reduction for norm_inf (two-pass via thrust or manual)
// Simple two-step: each block reduces to block_max; host reduces block results.
__global__ void kernel_abs_max(const double *__restrict__ x, int n,
                                double *__restrict__ block_max) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? fabs(x[i]) : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.x] = sdata[0];
}

// Dot product: per-block partial sums
__global__ void kernel_dot(const double *__restrict__ x,
                            const double *__restrict__ y,
                            int n, double *__restrict__ partial) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? (x[i] * y[i]) : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

static constexpr int BLOCK = 256;

static int grid(int n) { return (n + BLOCK - 1) / BLOCK; }

// ──────────────────────────────────────────────────────────────────────────────
// CudaVector
// ──────────────────────────────────────────────────────────────────────────────

class CudaVector final : public BackendVector {
public:
    explicit CudaVector(Index n) : n_(n) {
        if (n < 0 || n > std::numeric_limits<int>::max())
            throw std::invalid_argument("CudaVector: unsupported size");
        buf_ = DeviceBuffer<double>(static_cast<std::size_t>(n));
        // Zero-initialise on device
        if (n > 0)
            check_cuda(cudaMemset(buf_.ptr, 0, n * sizeof(double)),
                       "CudaVector zero init");
    }

    [[nodiscard]] Index size() const noexcept override { return n_; }
    double *device_ptr() noexcept { return buf_.ptr; }
    const double *device_ptr() const noexcept { return buf_.ptr; }

private:
    Index n_;
    DeviceBuffer<double> buf_;
};

// ──────────────────────────────────────────────────────────────────────────────
// CudaMatrix — CSR with cuSPARSE descriptor
// ──────────────────────────────────────────────────────────────────────────────

class CudaMatrix final : public BackendMatrix {
public:
    explicit CudaMatrix(const CsrPattern &pat)
        : rows_(pat.rows), cols_(pat.columns),
          nnz_(static_cast<Index>(pat.column_indices.size())) {
        validate_pattern(pat);
        d_row_offsets_ = DeviceBuffer<int64_t>(pat.row_offsets.size());
        d_col_indices_ = DeviceBuffer<int64_t>(pat.column_indices.size());
        d_values_ = DeviceBuffer<double>(pat.column_indices.size());

        // Upload row_offsets and column_indices (structure is fixed)
        check_cuda(cudaMemcpy(d_row_offsets_.ptr, pat.row_offsets.data(),
                              (pat.rows + 1) * sizeof(Index),
                              cudaMemcpyHostToDevice),
                   "CudaMatrix upload row_offsets");
        if (nnz_ > 0) {
            check_cuda(cudaMemcpy(d_col_indices_.ptr, pat.column_indices.data(),
                                  pat.column_indices.size() * sizeof(Index),
                                  cudaMemcpyHostToDevice),
                       "CudaMatrix upload col_indices");
            check_cuda(cudaMemset(d_values_.ptr, 0,
                                  static_cast<std::size_t>(nnz_) * sizeof(double)),
                       "CudaMatrix zero values");
        }

        if (nnz_ > 0) {
            check_cusparse(
                cusparseCreateCsr(
                    &desc_,
                    rows_, cols_, nnz_,
                    d_row_offsets_.ptr, d_col_indices_.ptr, d_values_.ptr,
                    CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                "cusparseCreateCsr");
        }
    }

    ~CudaMatrix() {
        if (desc_) cusparseDestroySpMat(desc_); // no-throw
    }

    // Non-copyable
    CudaMatrix(const CudaMatrix &) = delete;
    CudaMatrix &operator=(const CudaMatrix &) = delete;

    [[nodiscard]] Index rows() const noexcept override { return rows_; }
    [[nodiscard]] Index columns() const noexcept override { return cols_; }
    [[nodiscard]] Index nnz() const noexcept { return nnz_; }

    cusparseSpMatDescr_t descriptor() const noexcept { return desc_; }
    double *device_values() noexcept { return d_values_.ptr; }

private:
    Index rows_, cols_, nnz_;
    DeviceBuffer<int64_t> d_row_offsets_;
    DeviceBuffer<int64_t> d_col_indices_;
    DeviceBuffer<double> d_values_;
    cusparseSpMatDescr_t desc_ = nullptr;
};

// ──────────────────────────────────────────────────────────────────────────────
// Downcast helpers
// ──────────────────────────────────────────────────────────────────────────────

static CudaVector &as_cuda_vec(BackendVector &v) {
    auto *p = dynamic_cast<CudaVector *>(&v);
    if (!p) throw std::invalid_argument("CudaBackend: not a CudaVector");
    return *p;
}
static const CudaVector &as_cuda_vec(const BackendVector &v) {
    const auto *p = dynamic_cast<const CudaVector *>(&v);
    if (!p) throw std::invalid_argument("CudaBackend: not a CudaVector");
    return *p;
}
static CudaMatrix &as_cuda_mat(BackendMatrix &m) {
    auto *p = dynamic_cast<CudaMatrix *>(&m);
    if (!p) throw std::invalid_argument("CudaBackend: not a CudaMatrix");
    return *p;
}
static const CudaMatrix &as_cuda_mat(const BackendMatrix &m) {
    const auto *p = dynamic_cast<const CudaMatrix *>(&m);
    if (!p) throw std::invalid_argument("CudaBackend: not a CudaMatrix");
    return *p;
}

// ──────────────────────────────────────────────────────────────────────────────
// CudaBackend
// ──────────────────────────────────────────────────────────────────────────────

class CudaBackend final : public ExecutionBackend {
public:
    CudaBackend() {
        check_cuda(cudaSetDevice(0), "cudaSetDevice");
        check_cusparse(cusparseCreate(&sparse_handle_), "cusparseCreate");
    }

    ~CudaBackend() {
        cudaFree(d_reduce_buf_);
        cudaFree(d_spmv_workspace_);
        if (sparse_handle_) cusparseDestroy(sparse_handle_);
    }

    // ── Factory ──────────────────────────────────────────────────────────────

    [[nodiscard]] std::unique_ptr<BackendVector>
    create_vector(Index size) override {
        return std::make_unique<CudaVector>(size);
    }

    [[nodiscard]] std::unique_ptr<BackendMatrix>
    create_matrix(const CsrPattern &pattern) override {
        return std::make_unique<CudaMatrix>(pattern);
    }

    // ── Transfer ─────────────────────────────────────────────────────────────

    void upload(std::span<const Scalar> source, BackendVector &target) override {
        auto &v = as_cuda_vec(target);
        if (static_cast<Index>(source.size()) != v.size())
            dim_error("upload size mismatch");
        if (source.empty()) return;
        check_cuda(cudaMemcpy(v.device_ptr(), source.data(),
                              source.size() * sizeof(Scalar),
                              cudaMemcpyHostToDevice),
                   "upload");
    }

    void download(const BackendVector &source,
                  std::span<Scalar> target) const override {
        const auto &v = as_cuda_vec(source);
        if (static_cast<Index>(target.size()) != v.size())
            dim_error("download size mismatch");
        if (target.empty()) return;
        check_cuda(cudaMemcpy(target.data(), v.device_ptr(),
                              target.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
                   "download");
    }

    void set_values(std::span<const Scalar> values,
                    BackendMatrix &target) override {
        auto &m = as_cuda_mat(target);
        if (static_cast<Index>(values.size()) != m.nnz())
            dim_error("set_values size mismatch");
        if (values.empty()) return;
        check_cuda(cudaMemcpy(m.device_values(), values.data(),
                              values.size() * sizeof(Scalar),
                              cudaMemcpyHostToDevice),
                   "set_values");
    }

    // ── Vector operations ────────────────────────────────────────────────────

    void fill(BackendVector &target, Scalar value) override {
        auto &v = as_cuda_vec(target);
        const int n = static_cast<int>(v.size());
        if (n == 0) return;
        kernel_fill<<<grid(n), BLOCK>>>(v.device_ptr(), value, n);
        check_cuda(cudaGetLastError(), "fill kernel");
    }

    void copy(const BackendVector &source, BackendVector &target) override {
        const auto &s = as_cuda_vec(source);
        auto &t = as_cuda_vec(target);
        if (s.size() != t.size()) dim_error("copy size mismatch");
        if (s.size() == 0) return;
        check_cuda(cudaMemcpy(t.device_ptr(), s.device_ptr(),
                              static_cast<std::size_t>(s.size()) * sizeof(Scalar),
                              cudaMemcpyDeviceToDevice),
                   "copy");
    }

    void axpby(Scalar alpha, const BackendVector &x,
               Scalar beta, BackendVector &y) override {
        const auto &xv = as_cuda_vec(x);
        auto &yv = as_cuda_vec(y);
        if (xv.size() != yv.size()) dim_error("axpby size mismatch");
        const int n = static_cast<int>(xv.size());
        if (n == 0) return;
        kernel_axpby<<<grid(n), BLOCK>>>(
            alpha, xv.device_ptr(), beta, yv.device_ptr(), n);
        check_cuda(cudaGetLastError(), "axpby kernel");
    }

    [[nodiscard]] Scalar dot(const BackendVector &x,
                             const BackendVector &y) override {
        const auto &xv = as_cuda_vec(x);
        const auto &yv = as_cuda_vec(y);
        if (xv.size() != yv.size()) dim_error("dot size mismatch");
        const int n = static_cast<int>(xv.size());
        if (n == 0) return 0.0;

        const int blocks = grid(n);
        ensure_reduce_capacity(static_cast<std::size_t>(blocks));
        // Block results are copied to the host for the final reduction.
        kernel_dot<<<blocks, BLOCK, BLOCK * sizeof(double)>>>(
            xv.device_ptr(), yv.device_ptr(), n, d_reduce_buf_);
        check_cuda(cudaGetLastError(), "dot kernel");

        std::vector<double> host_buf(static_cast<std::size_t>(blocks));
        check_cuda(cudaMemcpy(host_buf.data(), d_reduce_buf_,
                              blocks * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "dot memcpy");
        double result = 0.0;
        for (double v : host_buf) result += v;
        return result;
    }

    [[nodiscard]] Scalar norm_inf(const BackendVector &x) override {
        const auto &xv = as_cuda_vec(x);
        const int n = static_cast<int>(xv.size());
        if (n == 0) return 0.0;

        const int blocks = grid(n);
        ensure_reduce_capacity(static_cast<std::size_t>(blocks));
        kernel_abs_max<<<blocks, BLOCK, BLOCK * sizeof(double)>>>(
            xv.device_ptr(), n, d_reduce_buf_);
        check_cuda(cudaGetLastError(), "norm_inf kernel");

        std::vector<double> host_buf(static_cast<std::size_t>(blocks));
        check_cuda(cudaMemcpy(host_buf.data(), d_reduce_buf_,
                              blocks * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "norm_inf memcpy");
        double result = 0.0;
        for (double v : host_buf) result = std::max(result, v);
        return result;
    }

    void project_box(BackendVector &x,
                     const BackendVector &lower,
                     const BackendVector &upper) override {
        auto &xv = as_cuda_vec(x);
        const auto &lv = as_cuda_vec(lower);
        const auto &uv = as_cuda_vec(upper);
        if (xv.size() != lv.size() || xv.size() != uv.size())
            dim_error("project_box size mismatch");
        const int n = static_cast<int>(xv.size());
        if (n == 0) return;
        kernel_project_box<<<grid(n), BLOCK>>>(
            xv.device_ptr(), lv.device_ptr(), uv.device_ptr(), n);
        check_cuda(cudaGetLastError(), "project_box kernel");
    }

    // ── cuSPARSE SpMV ────────────────────────────────────────────────────────
    //
    // y = alpha * op(A) * x + beta * y
    // Uses cusparseSpMV with CSR descriptor; handles both No and Yes transpose.

    void multiply(Scalar alpha, const BackendMatrix &matrix,
                  const BackendVector &x, Scalar beta,
                  BackendVector &y, Transpose transpose) override {
        const auto &m = as_cuda_mat(matrix);
        const auto &xv = as_cuda_vec(x);
        auto &yv = as_cuda_vec(y);

        cusparseOperation_t op = (transpose == Transpose::No)
                                     ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                     : CUSPARSE_OPERATION_TRANSPOSE;

        // Validate dimensions
        const Index expected_x = (transpose == Transpose::No) ? m.columns() : m.rows();
        const Index expected_y = (transpose == Transpose::No) ? m.rows() : m.columns();
        if (xv.size() != expected_x || yv.size() != expected_y)
            dim_error("multiply dimension mismatch");
        if (m.nnz() == 0) {
            axpby(0.0, y, beta, y);
            return;
        }

        // Create dense vector descriptors (lightweight, refer to existing device memory)
        DnVecDescriptor dx_desc;
        DnVecDescriptor dy_desc;
        check_cusparse(cusparseCreateDnVec(&dx_desc.value, xv.size(),
                                           const_cast<double *>(xv.device_ptr()),
                                           CUDA_R_64F),
                       "CreateDnVec x");
        check_cusparse(cusparseCreateDnVec(&dy_desc.value, yv.size(),
                                           yv.device_ptr(), CUDA_R_64F),
                       "CreateDnVec y");

        // Query buffer size
        std::size_t buf_size = 0;
        check_cusparse(cusparseSpMV_bufferSize(
                           sparse_handle_, op, &alpha, m.descriptor(),
                           dx_desc.value, &beta, dy_desc.value, CUDA_R_64F,
                           CUSPARSE_SPMV_CSR_ALG1, &buf_size),
                       "SpMV_bufferSize");

        // Use pre-allocated workspace if it fits, else allocate temporarily
        void *workspace = d_spmv_workspace_;
        if (buf_size > spmv_workspace_size_) {
            void *replacement = nullptr;
            check_cuda(cudaMalloc(&replacement, buf_size), "SpMV workspace alloc");
            if (d_spmv_workspace_) cudaFree(d_spmv_workspace_);
            d_spmv_workspace_ = replacement;
            spmv_workspace_size_ = buf_size;
            workspace = d_spmv_workspace_;
        }

        check_cusparse(cusparseSpMV(sparse_handle_, op, &alpha, m.descriptor(),
                                    dx_desc.value, &beta, dy_desc.value, CUDA_R_64F,
                                    CUSPARSE_SPMV_CSR_ALG1, workspace),
                       "cusparseSpMV");
    }

    // ── solve_linear_system ───────────────────────────────────────────────────
    //
    // BiCGSTAB on GPU. All working vectors stay device-resident.
    // Only the convergence scalars (dot products, norms) come to host.

    [[nodiscard]] LinearSolveResult
    solve_linear_system(const BackendMatrix &matrix,
                        const BackendVector &rhs,
                        BackendVector &solution,
                        MatrixProperty /*property*/,
                        Scalar tolerance,
                        Index iteration_limit) override {
        const auto &m = as_cuda_mat(matrix);
        if (m.rows() != m.columns())
            dim_error("solve_linear_system: matrix must be square");
        if (rhs.size() != m.rows() || solution.size() != m.rows())
            dim_error("solve_linear_system: rhs/solution size mismatch");
        if (!std::isfinite(tolerance) || tolerance < 0.0)
            dim_error("solve_linear_system: invalid tolerance");
        if (iteration_limit < 0)
            dim_error("solve_linear_system: invalid iteration limit");

        const Index n = m.rows();

        // Working vectors — all on device, allocated once
        auto r     = create_vector(n);
        auto r_hat = create_vector(n);
        auto p     = create_vector(n);
        auto v     = create_vector(n);
        auto s     = create_vector(n);
        auto t     = create_vector(n);

        // Initialise solution = 0, r = rhs
        fill(solution, 0.0);
        copy(rhs, *r);
        copy(*r, *r_hat);
        copy(*r, *p);

        Scalar rho_prev = 1.0, alpha = 1.0, omega = 1.0;
        fill(*v, 0.0);

        Scalar r_hat_dot_r = dot(*r_hat, *r);

        LinearSolveResult res;
        const Scalar initial_norm = norm_inf(*r);
        if (initial_norm <= tolerance)
            return LinearSolveResult{true, 0, initial_norm};

        for (Index iter = 0; iter < iteration_limit; ++iter) {
            res.iterations = iter + 1;
            const Scalar rho = r_hat_dot_r;
            if (rho == 0.0) break;

            if (iter > 0) {
                const Scalar beta = (rho / rho_prev) * (alpha / omega);
                // p = r + beta*(p - omega*v)
                axpby(-omega, *v, 1.0, *p);      // p = p - omega*v
                axpby(1.0, *r, beta, *p);         // p = r + beta*p
            }

            // v = A*p
            fill(*v, 0.0);
            multiply(1.0, matrix, *p, 0.0, *v, Transpose::No);

            const Scalar denom = dot(*r_hat, *v);
            if (denom == 0.0) break;
            alpha = rho / denom;

            // s = r - alpha*v
            copy(*r, *s);
            axpby(-alpha, *v, 1.0, *s);

            const Scalar norm_s = norm_inf(*s);
            if (norm_s <= tolerance) {
                axpby(alpha, *p, 1.0, solution);
                res.converged = true;
                res.iterations = iter + 1;
                res.residual = norm_s;
                return res;
            }

            // t = A*s
            fill(*t, 0.0);
            multiply(1.0, matrix, *s, 0.0, *t, Transpose::No);

            const Scalar tt = dot(*t, *t);
            if (tt == 0.0) break;
            omega = dot(*t, *s) / tt;
            if (omega == 0.0) {
                axpby(alpha, *p, 1.0, solution);
                copy(*s, *r);
                break;
            }

            // x += alpha*p + omega*s
            axpby(alpha, *p, 1.0, solution);
            axpby(omega, *s, 1.0, solution);

            // r = s - omega*t
            copy(*s, *r);
            axpby(-omega, *t, 1.0, *r);

            const Scalar norm_r = norm_inf(*r);
            if (norm_r <= tolerance) {
                res.converged = true;
                res.iterations = iter + 1;
                res.residual = norm_r;
                return res;
            }

            r_hat_dot_r = dot(*r_hat, *r);
            rho_prev = rho;
        }

        // Measure final residual
        auto tmp = create_vector(n);
        fill(*tmp, 0.0);
        multiply(1.0, matrix, solution, 0.0, *tmp, Transpose::No);
        axpby(1.0, rhs, -1.0, *tmp); // tmp = rhs - A*x
        res.residual = norm_inf(*tmp);
        res.converged = false;
        return res;
    }

    // ── Synchronise ───────────────────────────────────────────────────────────
    //
    // Called deliberately by the solver when it needs results on the host.
    // Not called speculatively.

    void synchronize() override {
        check_cuda(cudaDeviceSynchronize(), "synchronize");
    }

private:
    [[noreturn]] static void dim_error(const char *msg) {
        throw std::invalid_argument(std::string("CudaBackend: ") + msg);
    }

    void ensure_reduce_capacity(std::size_t required) {
        if (required <= reduce_capacity_) return;
        double *replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&replacement),
                              required * sizeof(double)),
                   "reduction workspace alloc");
        if (d_reduce_buf_) cudaFree(d_reduce_buf_);
        d_reduce_buf_ = replacement;
        reduce_capacity_ = required;
    }

    cusparseHandle_t sparse_handle_ = nullptr;

    double *d_reduce_buf_ = nullptr;
    std::size_t reduce_capacity_ = 0;

    // Lazy-growing SpMV workspace to avoid per-call malloc
    void *d_spmv_workspace_ = nullptr;
    std::size_t spmv_workspace_size_ = 0;
};

} // namespace

// ──────────────────────────────────────────────────────────────────────────────
// Public factories
// ──────────────────────────────────────────────────────────────────────────────

std::unique_ptr<ExecutionBackend> make_cuda_backend() {
    return std::make_unique<CudaBackend>();
}

} // namespace vikalp
