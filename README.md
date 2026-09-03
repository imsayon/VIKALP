# VIKALP

**Versatile Indigenous Kernel for Accelerated Large-scale Programming**

*A sovereign GPU-accelerated mathematical optimization solver.*

VIKALP is an early-stage effort for Smart India Hackathon problem statement **SIH26119**, proposed by Mangalore Refinery and Petrochemicals Limited (MRPL). Its purpose is to build a transparent mathematical optimization **solver core** from first principles for large, sparse industrial problems.

## What VIKALP is

The initial target is a numerical engine for:

- Linear Programming (LP)
- Mixed-Integer Linear Programming (MILP)
- Convex Quadratic Programming (QP)

The continuous and mixed-integer layers meet at the `RelaxationOracle`
contract. The built-in `solve(model, options)` entry point connects the current
LP/QP/NLP implementations to the MILP/MIQP/MINLP controllers; callers can still
inject a custom oracle for tests and external experiments.

## What VIKALP is not

VIKALP is not a GUI, refinery dashboard, modeling-language clone, or wrapper around CPLEX, Gurobi, HiGHS, SCIP, CBC, cuOpt, or another solver. Existing solvers are references and external benchmarks; they must not become VIKALP's hidden numerical engine.

## Engineering order

```text
Correctness
  -> Numerical robustness
  -> Reproducibility
  -> Scalability
  -> Performance
  -> GPU acceleration
```

GPU acceleration matters only where measurement shows a real benefit without weakening mathematical validity.

## Current status

- Gate 0 public solver contracts are frozen under `include/vikalp/contracts`
- Canonical sparse-model validation and a runnable contract check are implemented
- A deterministic CPU execution backend and CPU correctness tests are implemented
- An optional CUDA/cuSPARSE backend exists behind `VIKALP_ENABLE_CUDA`; hardware validation and profiling are pending
- Flow C continuous baselines are implemented for bounded LP (restarted/preconditioned PDHG), convex QP (PDQP), and smooth NLP (primal-dual IPM)
- Flow C analytic tests cover optimal solutions, row bounds, convexity rejection, warm starts, bound overrides, equality constraints, and residuals
- Flow D branch-and-bound, convex outer approximation, refinery cases, and reproducible benchmark tooling are implemented
- The built-in solver entry point has CPU integration coverage for LP, QP, NLP, MILP, MIQP, and MINLP
- Flow D tests use a mock/feasible relaxation oracle; they are not performance or production-solver claims
- `BackendPreference::Auto` currently selects the deterministic CPU backend; CUDA must be requested explicitly
- The MPS reader and in-memory solver API are early-stage and not yet compatibility-stable
- CUDA uses the same `ExecutionBackend` solver contract, but GPU runtime and performance claims remain unverified

Flow C deliberately remains a baseline: NLP requires a strictly feasible starting point, and its first KKT assembly uses a dense CSR pattern for small analytic cases. These are explicit limitations, not production-scale performance claims.

## Starting references

- [Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient](https://research.google/pubs/practical-large-scale-linear-programming-using-primal-dual-hybrid-gradient/)
- [Parallelizing the dual revised simplex method](https://link.springer.com/article/10.1007/s12532-017-0130-5)
- [cuPDLP-C](https://arxiv.org/abs/2312.14832)
- [Netlib LP problems](https://www.netlib.org/lp/data/)
- [MIPLIB 2017](https://miplib.zib.de/)
- [QPLIB](https://qplib.zib.de/)
