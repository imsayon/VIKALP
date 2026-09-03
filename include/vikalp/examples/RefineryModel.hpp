#pragma once

// Flow D — Parametric refinery model family.
//
// Generates realistic MILP/MIQP instances based on crude oil refinery operations:
//   - Crude blending: select crude types (binary), blend ratios (continuous)
//   - Unit scheduling: assign operations to processing units (binary)
//   - Quality constraints: sulfur, octane, viscosity bounds
//
// Models are parametric by:
//   - num_crudes: number of crude oil types (binary selection)
//   - num_units: number of processing units (binary on/off)
//   - num_products: number of output products (continuous blend ratios)
//   - num_quality: number of quality constraints per product
//
// All models are deterministic and reproducible for benchmarking.

#include "vikalp/contracts/Model.hpp"

#include <string>
#include <vector>

namespace vikalp {

/// Configuration for refinery model generation.
struct RefineryConfig {
    int num_crudes = 3;      // number of crude types to select from
    int num_units = 2;       // number of processing units
    int num_products = 2;    // number of output products
    int num_quality = 2;     // quality constraints per product
    Scalar demand_factor = 1.0;  // scales product demand
    bool include_quadratic = false;  // add quadratic blending terms
};

struct RefineryDemoCase {
    std::string name;
    RefineryConfig config;
};

/// Build a refinery MILP/MIQP model from configuration.
/// The model is a minimization of operating cost subject to
/// crude selection, blending, unit capacity, and quality constraints.
[[nodiscard]] Model build_refinery_model(const RefineryConfig &config);

/// Preset configurations for benchmarking.
[[nodiscard]] RefineryConfig refinery_small();   // 3 crudes, 2 units, 2 products
[[nodiscard]] RefineryConfig refinery_medium();  // 6 crudes, 4 units, 4 products
[[nodiscard]] RefineryConfig refinery_large();   // 12 crudes, 8 units, 8 products
[[nodiscard]] RefineryConfig refinery_xl();      // 20 crudes, 12 units, 12 products

/// Six deterministic MILP/MIQP cases used by the benchmark and demo runner.
[[nodiscard]] std::vector<RefineryDemoCase> refinery_demo_family();

} // namespace vikalp
