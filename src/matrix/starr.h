#pragma once
#include <hnswlib/hnswlib.h>

struct StableMigrationOptions {
    float z_tolerance = 2.0f; // Allowed standardized edge worsening before correlation scaling.
    size_t search_ef = 0;     // 0 means source M; query ef is independent.
    unsigned fine_repair_passes = 0; // Optional uncovered-node local repair; disabled by default.
};

struct StableMigrationStats {
    size_t edges = 0, removed_edges = 0, repaired = 0, refined = 0, bridges = 0;
    bool batched_stable_copy = false;
    double register_s = 0, filter_s = 0, reinsert_s = 0, refine_s = 0;
};

// Offline, single-threaded L2/float migration. Rows follow SOURCE INTERNAL-ID order.
// Preserves external labels, levels, M and maxM0. Does not modify the source.
// On success the caller owns both the returned index and *out_space; delete the
// index first. The returned vectors remain float32, with normal HNSW queries.
hnswlib::HierarchicalNSW<float> *migrateHnswIndexStable(
    hnswlib::HierarchicalNSW<float> *source, const std::vector<std::vector<float>> &target_vectors,
    hnswlib::SpaceInterface<float> **out_space, const StableMigrationOptions &options = {},
    StableMigrationStats *out_stats = nullptr);
