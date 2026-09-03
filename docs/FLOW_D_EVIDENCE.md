# Flow D — Mixed-Integer Engine Evidence

Status: Flow D implementation and CPU integration complete. The PPT export
remains deliberately pending.

## Implemented scope

- deterministic best-bound branch-and-bound with node identity, bound
  overrides, incumbent tracking, infeasibility/integrality/bound pruning,
  absolute/relative gap termination, node/time limits, and warm starts;
- most-fractional branching followed by pseudocost branching with observations
  taken from the parent relaxation bound;
- bounded rounding heuristic with linear-feasibility and quadratic-objective
  checks;
- a root-only cut-generator hook and valid integer-row bound-rounding cuts;
- MILP/MIQP routing through the relaxation oracle and convex OA master,
  nonlinear subproblem, derivative linearization, and epigraph loop for MINLP;
- six deterministic refinery MILP/MIQP cases with crude selection, unit
  activation, blending, demand, capacity, and quality rows;
- benchmark output in stdout, CSV, and JSON formats, plus an external-oracle
  mode.

## Reproducible checks

```text
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
build/vikalp_mip_test
build/vikalp_benchmark --csv benchmarks/results/flow-d.csv \
  --json benchmarks/results/flow-d.json
```

The MIP executable reports 33 focused checks. The benchmark uses a deliberately
simple feasible `RelaxationOracle`, so its records are labelled `Feasible`; they
are smoke evidence, not optimality or performance claims.

The `integration` CTest target separately runs LP, QP, NLP, MILP, MIQP, and
MINLP through the built-in solver entry point and real continuous solvers.

## External-oracle protocol

```text
build/vikalp_benchmark --external "./my-oracle" \
  --csv benchmarks/results/external.csv
```

The runner writes each generated model to a temporary `VIKALP_MODEL_V1` text
file and appends that path to the command. The command must print three
whitespace-separated fields:

```text
<status> <objective> <dual_bound>
```

The external command is an evidence/benchmark boundary. It is never used as
VIKALP's runtime numerical backend.

## Truth boundary

`Optimal` is emitted only when the supplied relaxation or OA master provides a
certified bound and the configured gap is closed. `Feasible`, `NodeLimit`,
`TimeLimit`, `IterationLimit`, `UnsupportedModel`, and `NumericalFailure` remain
distinct. The convex OA implementation assumes the nonlinear oracle supplies
convex objective/constraint data consistent with the tangent-master contract.
