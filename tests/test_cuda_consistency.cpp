// Flow B — CPU/CUDA consistency tests.
//
// When VIKALP_CUDA_ENABLED is defined, this test runs the same deterministic
// numerical operations on both backends and compares results within tolerance.
//
// Tolerance: 1e-10 (well above FP64 round-off, well below any solver tolerance)
//
// This test is only compiled and run when cmake -DVIKALP_ENABLE_CUDA=ON.

#include "vikalp/backend/CpuBackend.hpp"
#include "vikalp/backend/CudaBackend.hpp"
#include "vikalp/contracts/ExecutionBackend.hpp"

#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

static int g_failures = 0;
static constexpr double kTol = 1e-10;

void check(const char *test, double cpu, double gpu) {
    if (!std::isfinite(cpu) || !std::isfinite(gpu) ||
        std::abs(cpu - gpu) > kTol) {
        std::printf("FAIL [%s]: CPU=%.17g  GPU=%.17g  diff=%.3e\n",
                    test, cpu, gpu, std::abs(cpu - gpu));
        ++g_failures;
    }
}

void check_vec(const char *prefix,
               vikalp::ExecutionBackend &cpu_be, const vikalp::BackendVector &cpu_v,
               vikalp::ExecutionBackend &gpu_be, const vikalp::BackendVector &gpu_v) {
    const vikalp::Index n = cpu_v.size();
    std::vector<double> cpu_buf(static_cast<std::size_t>(n));
    std::vector<double> gpu_buf(static_cast<std::size_t>(n));
    cpu_be.download(cpu_v, std::span<double>(cpu_buf));
    gpu_be.download(gpu_v, std::span<double>(gpu_buf));
    for (vikalp::Index i = 0; i < n; ++i) {
        const std::string name = std::string(prefix) + "[" + std::to_string(i) + "]";
        check(name.c_str(), cpu_buf[static_cast<std::size_t>(i)],
              gpu_buf[static_cast<std::size_t>(i)]);
    }
}

// Reference matrix A (2x3):
//   row 0: cols {0,1} values {1,2}
//   row 1: cols {1,2} values {3,4}
vikalp::CsrPattern mat_A_pattern() {
    return {2, 3, {0, 2, 4}, {0, 1, 1, 2}};
}

void run_comparison(vikalp::ExecutionBackend &cpu, vikalp::ExecutionBackend &gpu) {
    const std::vector<double> avals = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> xhost = {2.0, 1.0, 3.0};

    // ── axpby ────────────────────────────────────────────────────────────────
    {
        auto cx = cpu.create_vector(3); auto cy = cpu.create_vector(3);
        auto gx = gpu.create_vector(3); auto gy = gpu.create_vector(3);
        const std::vector<double> xv = {1.0, -2.0, 3.0};
        const std::vector<double> yv = {4.0, 5.0, -6.0};
        cpu.upload(std::span<const double>(xv), *cx);
        cpu.upload(std::span<const double>(yv), *cy);
        gpu.upload(std::span<const double>(xv), *gx);
        gpu.upload(std::span<const double>(yv), *gy);
        cpu.axpby(2.5, *cx, -1.0, *cy);
        gpu.axpby(2.5, *gx, -1.0, *gy);
        gpu.synchronize();
        check_vec("axpby", cpu, *cy, gpu, *gy);
    }

    // ── dot ──────────────────────────────────────────────────────────────────
    {
        auto cx = cpu.create_vector(4); auto cy = cpu.create_vector(4);
        auto gx = gpu.create_vector(4); auto gy = gpu.create_vector(4);
        const std::vector<double> xv = {1.0, -2.0, 0.5, 3.0};
        const std::vector<double> yv = {-1.0, 3.0, 2.0, -0.5};
        cpu.upload(std::span<const double>(xv), *cx);
        cpu.upload(std::span<const double>(yv), *cy);
        gpu.upload(std::span<const double>(xv), *gx);
        gpu.upload(std::span<const double>(yv), *gy);
        const double cpu_dot = cpu.dot(*cx, *cy);
        gpu.synchronize();
        const double gpu_dot = gpu.dot(*gx, *gy);
        check("dot", cpu_dot, gpu_dot);
    }

    // ── norm_inf ─────────────────────────────────────────────────────────────
    {
        auto cv = cpu.create_vector(5); auto gv = gpu.create_vector(5);
        const std::vector<double> vals = {1.0, -7.5, 3.0, -2.0, 0.0};
        cpu.upload(std::span<const double>(vals), *cv);
        gpu.upload(std::span<const double>(vals), *gv);
        const double cpu_ni = cpu.norm_inf(*cv);
        gpu.synchronize();
        const double gpu_ni = gpu.norm_inf(*gv);
        check("norm_inf", cpu_ni, gpu_ni);
    }

    // ── project_box ──────────────────────────────────────────────────────────
    {
        auto cx = cpu.create_vector(3); auto gx = gpu.create_vector(3);
        auto cl = cpu.create_vector(3); auto gl = gpu.create_vector(3);
        auto cu = cpu.create_vector(3); auto gu = gpu.create_vector(3);
        const std::vector<double> xv = {0.5, -1.0, 2.0};
        const std::vector<double> lv = {0.0, 0.0, 0.0};
        const std::vector<double> uv = {1.0, 1.0, 1.0};
        cpu.upload(std::span<const double>(xv), *cx);
        cpu.upload(std::span<const double>(lv), *cl);
        cpu.upload(std::span<const double>(uv), *cu);
        gpu.upload(std::span<const double>(xv), *gx);
        gpu.upload(std::span<const double>(lv), *gl);
        gpu.upload(std::span<const double>(uv), *gu);
        cpu.project_box(*cx, *cl, *cu);
        gpu.project_box(*gx, *gl, *gu);
        gpu.synchronize();
        check_vec("project_box", cpu, *cx, gpu, *gx);
    }

    // ── SpMV (No transpose) ──────────────────────────────────────────────────
    {
        auto pat = mat_A_pattern();
        auto cm = cpu.create_matrix(pat); auto gm = gpu.create_matrix(pat);
        cpu.set_values(std::span<const double>(avals), *cm);
        gpu.set_values(std::span<const double>(avals), *gm);
        auto cx = cpu.create_vector(3); auto cy = cpu.create_vector(2);
        auto gx = gpu.create_vector(3); auto gy = gpu.create_vector(2);
        cpu.upload(std::span<const double>(xhost), *cx);
        gpu.upload(std::span<const double>(xhost), *gx);
        cpu.fill(*cy, 0.0); gpu.fill(*gy, 0.0);
        cpu.multiply(1.0, *cm, *cx, 0.0, *cy, vikalp::Transpose::No);
        gpu.multiply(1.0, *gm, *gx, 0.0, *gy, vikalp::Transpose::No);
        gpu.synchronize();
        check_vec("spmv_no_T", cpu, *cy, gpu, *gy);
    }

    // ── SpMV (Transpose) ─────────────────────────────────────────────────────
    {
        auto pat = mat_A_pattern();
        auto cm = cpu.create_matrix(pat); auto gm = gpu.create_matrix(pat);
        cpu.set_values(std::span<const double>(avals), *cm);
        gpu.set_values(std::span<const double>(avals), *gm);
        const std::vector<double> xT = {2.0, 1.0}; // rows(A)=2
        auto cx = cpu.create_vector(2); auto cy = cpu.create_vector(3);
        auto gx = gpu.create_vector(2); auto gy = gpu.create_vector(3);
        cpu.upload(std::span<const double>(xT), *cx);
        gpu.upload(std::span<const double>(xT), *gx);
        cpu.fill(*cy, 0.0); gpu.fill(*gy, 0.0);
        cpu.multiply(1.0, *cm, *cx, 0.0, *cy, vikalp::Transpose::Yes);
        gpu.multiply(1.0, *gm, *gx, 0.0, *gy, vikalp::Transpose::Yes);
        gpu.synchronize();
        check_vec("spmv_T", cpu, *cy, gpu, *gy);
    }

    // ── solve_linear_system ───────────────────────────────────────────────────
    {
        // A = 2x2 identity
        vikalp::CsrPattern p{2, 2, {0, 1, 2}, {0, 1}};
        auto cm = cpu.create_matrix(p); auto gm = gpu.create_matrix(p);
        const std::vector<double> av = {1.0, 1.0};
        cpu.set_values(std::span<const double>(av), *cm);
        gpu.set_values(std::span<const double>(av), *gm);
        auto cb = cpu.create_vector(2); auto gb = gpu.create_vector(2);
        auto cs = cpu.create_vector(2); auto gs = gpu.create_vector(2);
        const std::vector<double> bv = {3.0, 7.0};
        cpu.upload(std::span<const double>(bv), *cb);
        gpu.upload(std::span<const double>(bv), *gb);
        cpu.fill(*cs, 0.0); gpu.fill(*gs, 0.0);
        cpu.solve_linear_system(*cm, *cb, *cs, vikalp::MatrixProperty::General, 1e-10, 100);
        gpu.solve_linear_system(*gm, *gb, *gs, vikalp::MatrixProperty::General, 1e-10, 100);
        gpu.synchronize();
        check_vec("solve_linear_system", cpu, *cs, gpu, *gs);
    }
}

} // namespace

int main() {
    std::printf("=== CPU/CUDA Consistency Tests ===\n");
    std::printf("Tolerance: %.1e\n\n", kTol);

    auto cpu = vikalp::make_cpu_backend();
    auto gpu = vikalp::make_cuda_backend();

    run_comparison(*cpu, *gpu);

    if (g_failures == 0) {
        std::printf("\nALL CONSISTENCY TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d CONSISTENCY TEST(S) FAILED\n", g_failures);
    return 1;
}
