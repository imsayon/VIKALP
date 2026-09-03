// Flow C — CPU backend correctness tests.
//
// All expected values are hand-computed independently of the implementation.
// No external test framework is required.
// Exit code: 0 = all pass, 1 = any failure.

#include "vikalp/backend/CpuBackend.hpp"
#include "vikalp/contracts/ExecutionBackend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ──────────────────────────────────────────────────────────────────────────────
// Tiny test framework
// ──────────────────────────────────────────────────────────────────────────────

int g_failures = 0;

void check_eq(const std::string &test, double got, double expected,
              double tol = 1e-12) {
    if (std::abs(got - expected) > tol) {
        std::printf("FAIL [%s]: got %.17g, expected %.17g (tol %.1e)\n",
                    test.c_str(), got, expected, tol);
        ++g_failures;
    }
}

void check_true(const std::string &test, bool cond) {
    if (!cond) {
        std::printf("FAIL [%s]: condition was false\n", test.c_str());
        ++g_failures;
    }
}

// Download vector to std::vector<double>
std::vector<double> dload(vikalp::ExecutionBackend &be,
                          const vikalp::BackendVector &v) {
    std::vector<double> buf(static_cast<std::size_t>(v.size()));
    be.download(v, std::span<double>(buf));
    return buf;
}

// ──────────────────────────────────────────────────────────────────────────────
// Reference matrix used across several tests:
//
//   A = [1 2 0]   (2x3)
//       [0 3 4]
//
//   x = [2, 1, 3]^T
//
// A*x  = [1*2 + 2*1 + 0*3, 0*2 + 3*1 + 4*3] = [4, 15]
// A^T*[2,1] = [1*2+0*1, 2*2+3*1, 0*2+4*1] = [2, 7, 4]
// ──────────────────────────────────────────────────────────────────────────────

vikalp::CsrPattern make_A_pattern() {
    // A is 2x3:
    //   row 0: cols {0,1} -> values {1,2}
    //   row 1: cols {1,2} -> values {3,4}
    return vikalp::CsrPattern{
        /*rows=*/2, /*columns=*/3,
        /*row_offsets=*/{0, 2, 4},
        /*column_indices=*/{0, 1, 1, 2}
    };
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

void test_create_vector(vikalp::ExecutionBackend &be) {
    auto v = be.create_vector(5);
    check_true("create_vector: size", v->size() == 5);

    auto v0 = be.create_vector(0);
    check_true("create_vector: zero size", v0->size() == 0);
}

void test_create_matrix(vikalp::ExecutionBackend &be) {
    auto pat = make_A_pattern();
    auto m = be.create_matrix(pat);
    check_true("create_matrix: rows", m->rows() == 2);
    check_true("create_matrix: cols", m->columns() == 3);
}

void test_upload_download(vikalp::ExecutionBackend &be) {
    auto v = be.create_vector(4);
    const std::vector<double> src = {1.0, -2.5, 0.0, 3.14};
    be.upload(std::span<const double>(src), *v);
    auto got = dload(be, *v);
    for (int i = 0; i < 4; ++i)
        check_eq("upload_download[" + std::to_string(i) + "]", got[i], src[i]);
}

void test_fill(vikalp::ExecutionBackend &be) {
    auto v = be.create_vector(3);
    be.fill(*v, 7.0);
    auto got = dload(be, *v);
    for (int i = 0; i < 3; ++i)
        check_eq("fill[" + std::to_string(i) + "]", got[i], 7.0);
}

void test_copy(vikalp::ExecutionBackend &be) {
    auto src = be.create_vector(3);
    auto dst = be.create_vector(3);
    be.fill(*src, -4.0);
    be.copy(*src, *dst);
    auto got = dload(be, *dst);
    for (int i = 0; i < 3; ++i)
        check_eq("copy[" + std::to_string(i) + "]", got[i], -4.0);
}

void test_axpby(vikalp::ExecutionBackend &be) {
    // y = 2*x + 3*y
    // x = [1, 2, 3], y = [4, 5, 6]
    // expected: [2*1+3*4, 2*2+3*5, 2*3+3*6] = [14, 19, 24]
    auto x = be.create_vector(3);
    auto y = be.create_vector(3);
    const std::vector<double> xv = {1.0, 2.0, 3.0};
    const std::vector<double> yv = {4.0, 5.0, 6.0};
    be.upload(std::span<const double>(xv), *x);
    be.upload(std::span<const double>(yv), *y);
    be.axpby(2.0, *x, 3.0, *y);
    auto got = dload(be, *y);
    check_eq("axpby[0]", got[0], 14.0);
    check_eq("axpby[1]", got[1], 19.0);
    check_eq("axpby[2]", got[2], 24.0);

    // Special case: beta=0, alpha=1 (copy)
    auto a = be.create_vector(2);
    auto b = be.create_vector(2);
    const std::vector<double> av = {5.0, -3.0};
    be.upload(std::span<const double>(av), *a);
    be.fill(*b, 100.0);
    be.axpby(1.0, *a, 0.0, *b);
    auto gb = dload(be, *b);
    check_eq("axpby_beta0[0]", gb[0], 5.0);
    check_eq("axpby_beta0[1]", gb[1], -3.0);
}

void test_dot(vikalp::ExecutionBackend &be) {
    // dot([1,2,3], [4,5,6]) = 4+10+18 = 32
    auto x = be.create_vector(3);
    auto y = be.create_vector(3);
    const std::vector<double> xv = {1.0, 2.0, 3.0};
    const std::vector<double> yv = {4.0, 5.0, 6.0};
    be.upload(std::span<const double>(xv), *x);
    be.upload(std::span<const double>(yv), *y);
    check_eq("dot", be.dot(*x, *y), 32.0);

    // Orthogonal
    auto a = be.create_vector(2);
    auto b = be.create_vector(2);
    const std::vector<double> av = {1.0, 0.0};
    const std::vector<double> bv = {0.0, 1.0};
    be.upload(std::span<const double>(av), *a);
    be.upload(std::span<const double>(bv), *b);
    check_eq("dot_orthogonal", be.dot(*a, *b), 0.0);

    // Negative values: dot([-1,-2],[3,4]) = -3-8 = -11
    const std::vector<double> nv = {-1.0, -2.0};
    const std::vector<double> pv = {3.0, 4.0};
    be.upload(std::span<const double>(nv), *a);
    be.upload(std::span<const double>(pv), *b);
    check_eq("dot_negative", be.dot(*a, *b), -11.0);
}

void test_norm_inf(vikalp::ExecutionBackend &be) {
    // max(|1|,|-5|,|3|) = 5
    auto v = be.create_vector(3);
    const std::vector<double> vals = {1.0, -5.0, 3.0};
    be.upload(std::span<const double>(vals), *v);
    check_eq("norm_inf", be.norm_inf(*v), 5.0);

    // Zero vector
    be.fill(*v, 0.0);
    check_eq("norm_inf_zero", be.norm_inf(*v), 0.0);

    // 1x1
    auto u = be.create_vector(1);
    const std::vector<double> u1 = {-7.5};
    be.upload(std::span<const double>(u1), *u);
    check_eq("norm_inf_1x1", be.norm_inf(*u), 7.5);
}

void test_project_box(vikalp::ExecutionBackend &be) {
    // x=[0.5, -1.0, 2.0], lo=[0,0,0], hi=[1,1,1]
    // projected: [0.5, 0, 1]
    auto x = be.create_vector(3);
    auto lo = be.create_vector(3);
    auto hi = be.create_vector(3);
    const std::vector<double> xv = {0.5, -1.0, 2.0};
    const std::vector<double> lv = {0.0, 0.0, 0.0};
    const std::vector<double> uv = {1.0, 1.0, 1.0};
    be.upload(std::span<const double>(xv), *x);
    be.upload(std::span<const double>(lv), *lo);
    be.upload(std::span<const double>(uv), *hi);
    be.project_box(*x, *lo, *hi);
    auto got = dload(be, *x);
    check_eq("project_box[0]", got[0], 0.5);
    check_eq("project_box[1]", got[1], 0.0);
    check_eq("project_box[2]", got[2], 1.0);

    // Already at boundary
    const std::vector<double> bv = {0.0, 1.0, 0.5};
    be.upload(std::span<const double>(bv), *x);
    be.project_box(*x, *lo, *hi);
    auto gb = dload(be, *x);
    check_eq("project_box_boundary[0]", gb[0], 0.0);
    check_eq("project_box_boundary[1]", gb[1], 1.0);
    check_eq("project_box_boundary[2]", gb[2], 0.5);

    // Negative domain: x=[-3,-2,-1], lo=[-5,-3,-2], hi=[-1,-1,-0.5]
    // projected: [-3, -2, -1]  (all in range)
    auto xn = be.create_vector(3);
    auto ln = be.create_vector(3);
    auto un = be.create_vector(3);
    const std::vector<double> xnv = {-3.0, -2.0, -1.0};
    const std::vector<double> lnv = {-5.0, -3.0, -2.0};
    const std::vector<double> unv = {-1.0, -1.0, -0.5};
    be.upload(std::span<const double>(xnv), *xn);
    be.upload(std::span<const double>(lnv), *ln);
    be.upload(std::span<const double>(unv), *un);
    be.project_box(*xn, *ln, *un);
    auto gn = dload(be, *xn);
    check_eq("project_box_neg[0]", gn[0], -3.0);
    check_eq("project_box_neg[1]", gn[1], -2.0);
    check_eq("project_box_neg[2]", gn[2], -1.0);
}

void test_set_values(vikalp::ExecutionBackend &be) {
    auto pat = make_A_pattern(); // 2x3, 4 nnz
    auto m = be.create_matrix(pat);
    // nnz = 4 entries: {1,2,3,4}
    const std::vector<double> vals = {1.0, 2.0, 3.0, 4.0};
    be.set_values(std::span<const double>(vals), *m);
    // Verification happens implicitly via SpMV tests below
    check_true("set_values: survived", true);
}

void test_spmv_no_transpose(vikalp::ExecutionBackend &be) {
    // A = [1 2 0 / 0 3 4],  x=[2,1,3]
    // A*x = [1*2+2*1, 3*1+4*3] = [4, 15]
    auto pat = make_A_pattern();
    auto m = be.create_matrix(pat);
    const std::vector<double> avals = {1.0, 2.0, 3.0, 4.0};
    be.set_values(std::span<const double>(avals), *m);

    auto x = be.create_vector(3);
    auto y = be.create_vector(2);
    const std::vector<double> xv = {2.0, 1.0, 3.0};
    be.upload(std::span<const double>(xv), *x);
    be.fill(*y, 0.0);

    // y = 1.0 * A * x + 0.0 * y
    be.multiply(1.0, *m, *x, 0.0, *y, vikalp::Transpose::No);
    auto got = dload(be, *y);
    check_eq("spmv_no_T[0]", got[0], 4.0);
    check_eq("spmv_no_T[1]", got[1], 15.0);

    // With non-trivial alpha/beta:
    // y_new = 2 * A * x + 3 * y_old  (y_old=[4,15] from above)
    // = [2*4+3*4, 2*15+3*15] = [20, 75]
    be.multiply(2.0, *m, *x, 3.0, *y, vikalp::Transpose::No);
    auto got2 = dload(be, *y);
    check_eq("spmv_no_T_alphabeta[0]", got2[0], 20.0);
    check_eq("spmv_no_T_alphabeta[1]", got2[1], 75.0);
}

void test_spmv_transpose(vikalp::ExecutionBackend &be) {
    // A^T * [2,1] = [1*2+0*1, 2*2+3*1, 0*2+4*1] = [2, 7, 4]
    auto pat = make_A_pattern();
    auto m = be.create_matrix(pat);
    const std::vector<double> avals = {1.0, 2.0, 3.0, 4.0};
    be.set_values(std::span<const double>(avals), *m);

    auto x = be.create_vector(2); // x has rows(A)=2 elements
    auto y = be.create_vector(3); // y has cols(A)=3 elements
    const std::vector<double> xv = {2.0, 1.0};
    be.upload(std::span<const double>(xv), *x);
    be.fill(*y, 0.0);

    be.multiply(1.0, *m, *x, 0.0, *y, vikalp::Transpose::Yes);
    auto got = dload(be, *y);
    check_eq("spmv_T[0]", got[0], 2.0);
    check_eq("spmv_T[1]", got[1], 7.0);
    check_eq("spmv_T[2]", got[2], 4.0);

    // With alpha=0.5, beta=2.0 and y=[2,7,4]:
    // y_new = 0.5*[2,7,4] + 2*[2,7,4] = [1+4, 3.5+14, 2+8] = [5, 17.5, 10]
    be.multiply(0.5, *m, *x, 2.0, *y, vikalp::Transpose::Yes);
    auto got2 = dload(be, *y);
    check_eq("spmv_T_alphabeta[0]", got2[0], 5.0);
    check_eq("spmv_T_alphabeta[1]", got2[1], 17.5);
    check_eq("spmv_T_alphabeta[2]", got2[2], 10.0);
}

void test_spmv_1x1(vikalp::ExecutionBackend &be) {
    // 1x1 matrix A=[7], x=[3] -> A*x=[21], A^T*[3]=[21]
    vikalp::CsrPattern p{1, 1, {0, 1}, {0}};
    auto m = be.create_matrix(p);
    const std::vector<double> avals = {7.0};
    be.set_values(std::span<const double>(avals), *m);

    auto x = be.create_vector(1);
    auto y = be.create_vector(1);
    const std::vector<double> xv = {3.0};
    be.upload(std::span<const double>(xv), *x);
    be.fill(*y, 0.0);

    be.multiply(1.0, *m, *x, 0.0, *y, vikalp::Transpose::No);
    auto got = dload(be, *y);
    check_eq("spmv_1x1_no_T", got[0], 21.0);

    be.fill(*y, 0.0);
    be.multiply(1.0, *m, *x, 0.0, *y, vikalp::Transpose::Yes);
    auto got2 = dload(be, *y);
    check_eq("spmv_1x1_T", got2[0], 21.0);
}

void test_spmv_empty_row(vikalp::ExecutionBackend &be) {
    // 3x2 matrix:
    //   row 0: [1, 0]  -> col 0, val 1
    //   row 1: []      -> empty
    //   row 2: [0, 5]  -> col 1, val 5
    // x=[2, 3]
    // A*x = [2, 0, 15]
    vikalp::CsrPattern p{3, 2, {0, 1, 1, 2}, {0, 1}};
    auto m = be.create_matrix(p);
    const std::vector<double> avals = {1.0, 5.0};
    be.set_values(std::span<const double>(avals), *m);

    auto x = be.create_vector(2);
    auto y = be.create_vector(3);
    const std::vector<double> xv = {2.0, 3.0};
    be.upload(std::span<const double>(xv), *x);
    be.fill(*y, 0.0);

    be.multiply(1.0, *m, *x, 0.0, *y, vikalp::Transpose::No);
    auto got = dload(be, *y);
    check_eq("spmv_empty_row[0]", got[0], 2.0);
    check_eq("spmv_empty_row[1]", got[1], 0.0);
    check_eq("spmv_empty_row[2]", got[2], 15.0);
}

void test_spmv_zero_vector(vikalp::ExecutionBackend &be) {
    // A*0 = 0
    auto pat = make_A_pattern();
    auto m = be.create_matrix(pat);
    const std::vector<double> avals = {1.0, 2.0, 3.0, 4.0};
    be.set_values(std::span<const double>(avals), *m);

    auto x = be.create_vector(3);
    auto y = be.create_vector(2);
    be.fill(*x, 0.0);
    be.fill(*y, 0.0);

    be.multiply(1.0, *m, *x, 0.0, *y, vikalp::Transpose::No);
    auto got = dload(be, *y);
    check_eq("spmv_zero_x[0]", got[0], 0.0);
    check_eq("spmv_zero_x[1]", got[1], 0.0);
}

void test_solve_linear_system(vikalp::ExecutionBackend &be) {
    // Solve A*x = b where A is the 2x2 identity
    // A = [1 0 / 0 1], b=[3,5]  -> x=[3,5]
    vikalp::CsrPattern p{2, 2, {0, 1, 2}, {0, 1}};
    auto m = be.create_matrix(p);
    const std::vector<double> avals = {1.0, 1.0};
    be.set_values(std::span<const double>(avals), *m);

    auto rhs = be.create_vector(2);
    auto sol = be.create_vector(2);
    const std::vector<double> bv = {3.0, 5.0};
    be.upload(std::span<const double>(bv), *rhs);
    be.fill(*sol, 0.0);

    auto res = be.solve_linear_system(*m, *rhs, *sol,
                                     vikalp::MatrixProperty::General,
                                     1e-10, 100);
    check_true("solve_identity: converged", res.converged);
    auto got = dload(be, *sol);
    check_eq("solve_identity[0]", got[0], 3.0, 1e-8);
    check_eq("solve_identity[1]", got[1], 5.0, 1e-8);

    // Solve A*x = b where A = [2 1 / 1 3], b=[4, 5]
    // Hand-solve: 2x+y=4, x+3y=5 => y=(5-2*x+x)/... using substitution:
    //   x = (4-y)/2; substitute: (4-y)/2 + 3y = 5 => 4-y+6y=10 => 5y=6 => y=1.2, x=1.4
    vikalp::CsrPattern p2{2, 2, {0, 2, 4}, {0, 1, 0, 1}};
    auto m2 = be.create_matrix(p2);
    const std::vector<double> av2 = {2.0, 1.0, 1.0, 3.0};
    be.set_values(std::span<const double>(av2), *m2);

    auto rhs2 = be.create_vector(2);
    auto sol2 = be.create_vector(2);
    const std::vector<double> bv2 = {4.0, 5.0};
    be.upload(std::span<const double>(bv2), *rhs2);
    be.fill(*sol2, 0.0);

    auto res2 = be.solve_linear_system(*m2, *rhs2, *sol2,
                                       vikalp::MatrixProperty::SymmetricPositiveDefinite,
                                       1e-10, 1000);
    check_true("solve_2x2: converged", res2.converged);
    auto got2 = dload(be, *sol2);
    check_eq("solve_2x2[0]", got2[0], 1.4, 1e-7);
    check_eq("solve_2x2[1]", got2[1], 1.2, 1e-7);
}

void test_invalid_dimensions(vikalp::ExecutionBackend &be) {
    // upload: wrong size
    auto v2 = be.create_vector(2);
    const std::vector<double> src3 = {1.0, 2.0, 3.0};
    bool threw = false;
    try { be.upload(std::span<const double>(src3), *v2); }
    catch (const std::invalid_argument &) { threw = true; }
    check_true("upload_dim_error", threw);

    // download: wrong size
    auto v3 = be.create_vector(3);
    std::vector<double> dst2(2);
    threw = false;
    try { be.download(*v3, std::span<double>(dst2)); }
    catch (const std::invalid_argument &) { threw = true; }
    check_true("download_dim_error", threw);

    // axpby: size mismatch
    auto x2 = be.create_vector(2);
    threw = false;
    try { be.axpby(1.0, *x2, 1.0, *v3); }
    catch (const std::invalid_argument &) { threw = true; }
    check_true("axpby_dim_error", threw);

    // dot: size mismatch
    threw = false;
    try { be.dot(*x2, *v3); }
    catch (const std::invalid_argument &) { threw = true; }
    check_true("dot_dim_error", threw);

    // copy: size mismatch
    threw = false;
    try { be.copy(*x2, *v3); }
    catch (const std::invalid_argument &) { threw = true; }
    check_true("copy_dim_error", threw);

    // multiply: wrong x dimension
    auto pat = make_A_pattern(); // 2x3
    auto m = be.create_matrix(pat);
    const std::vector<double> avals = {1.0, 2.0, 3.0, 4.0};
    be.set_values(std::span<const double>(avals), *m);
    auto xbad = be.create_vector(5); // should be 3
    auto ygood = be.create_vector(2);
    threw = false;
    try { be.multiply(1.0, *m, *xbad, 0.0, *ygood, vikalp::Transpose::No); }
    catch (const std::invalid_argument &) { threw = true; }
    check_true("multiply_x_dim_error", threw);

    // multiply: wrong y dimension
    auto xgood = be.create_vector(3);
    auto ybad = be.create_vector(5); // should be 2
    threw = false;
    try { be.multiply(1.0, *m, *xgood, 0.0, *ybad, vikalp::Transpose::No); }
    catch (const std::invalid_argument &) { threw = true; }
    check_true("multiply_y_dim_error", threw);

    // project_box: size mismatch
    auto lo = be.create_vector(2);
    auto hi = be.create_vector(2);
    auto x3 = be.create_vector(3);
    threw = false;
    try { be.project_box(*x3, *lo, *hi); }
    catch (const std::invalid_argument &) { threw = true; }
    check_true("project_box_dim_error", threw);
}

void test_synchronize(vikalp::ExecutionBackend &be) {
    // CPU synchronize is a no-op; just ensure it does not throw.
    bool threw = false;
    try { be.synchronize(); }
    catch (...) { threw = true; }
    check_true("synchronize_no_throw", !threw);
}

} // namespace

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

int main() {
    auto be = vikalp::make_cpu_backend();

    test_create_vector(*be);
    test_create_matrix(*be);
    test_upload_download(*be);
    test_fill(*be);
    test_copy(*be);
    test_axpby(*be);
    test_dot(*be);
    test_norm_inf(*be);
    test_project_box(*be);
    test_set_values(*be);
    test_spmv_no_transpose(*be);
    test_spmv_transpose(*be);
    test_spmv_1x1(*be);
    test_spmv_empty_row(*be);
    test_spmv_zero_vector(*be);
    test_solve_linear_system(*be);
    test_invalid_dimensions(*be);
    test_synchronize(*be);

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
