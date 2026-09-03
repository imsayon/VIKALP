# VIKALP

**Versatile Indigenous Kernel for Accelerated Large-scale Programming**

*A sovereign GPU-accelerated mathematical optimization solver.*

VIKALP is an early-stage effort for Smart India Hackathon problem statement **SIH26119**, proposed by Mangalore Refinery and Petrochemicals Limited (MRPL). Its purpose is to build a transparent mathematical optimization **solver core** from first principles for large, sparse industrial problems.

## What VIKALP is

The initial target is a numerical engine for:

- Linear Programming (LP)
- Mixed-Integer Linear Programming (MILP)
- Convex Quadratic Programming (QP)

These are goals, not current capabilities. The repository is presently in research and specification.

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
- No optimization algorithm implementation yet
- No supported file format or stable end-user API yet
- No GPU-performance or benchmark claims

## Starting references

- [Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient](https://research.google/pubs/practical-large-scale-linear-programming-using-primal-dual-hybrid-gradient/)
- [Parallelizing the dual revised simplex method](https://link.springer.com/article/10.1007/s12532-017-0130-5)
- [cuPDLP-C](https://arxiv.org/abs/2312.14832)
- [Netlib LP problems](https://www.netlib.org/lp/data/)
- [MIPLIB 2017](https://miplib.zib.de/)
- [QPLIB](https://qplib.zib.de/)
