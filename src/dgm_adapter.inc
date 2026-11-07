// HNSW-to-DGM adapter with configurable candidate width, beam width, hops and diversity.
static Migration migrateDgmScanned(Graph &source, const Rows &target, bool local, size_t C,
                                   size_t L, size_t H, float alpha) {
    Migration out;
    auto t = Clock::now();
    out.target = allocateTarget(source, target);
    out.preparation_s += elapsed(t);
    auto &result = *out.target.graph;
    out.layer_samples.resize(source.maxlevel_ + 1);
    for (int h = source.maxlevel_; h >= 0; --h) {
        t = Clock::now();
        auto m = mapping(source, h);
        auto original = extract(source, h, m);
        size_t dim = target[0].size();
        std::vector<float> packed(m.local_to_global.size() * dim);
        for (size_t i = 0; i < m.local_to_global.size(); ++i)
            std::copy(target[m.local_to_global[i]].begin(), target[m.local_to_global[i]].end(),
                      packed.begin() + i * dim);
        dgm::EmbeddingSpace space(packed.data(), m.local_to_global.size(), dim,
                                  dgm::Metric::kSquaredL2, 1, result.fstdistfunc_,
                                  result.dist_func_param_);
        out.preparation_s += elapsed(t);
        t = Clock::now();
        dgm::MigrationResult changed;
        if (local) {
            dgm::LocalConfig c;
            c.max_degree = h ? source.maxM_ : source.maxM0_;
            c.shortlist_size = C;
            c.diversity_factor = alpha;
            c.calibration_sample_size = 4096;
            c.random_seed = 20250311;
            c.thread_count = 1;
            changed = dgm::RunDgmLocal(original, space, c);
            out.layer_samples[h] = std::min(size_t(4096), m.local_to_global.size());
        } else {
            dgm::SearchConfig c;
            c.max_degree = h ? source.maxM_ : source.maxM0_;
            c.beam_width = L;
            c.max_hops = H;
            c.diversity_factor = alpha;
            c.thread_count = 1;
            c.entry_point = m.global_to_local[source.enterpoint_node_];
            changed = dgm::RunDgmSearch(original, space, c);
        }
        out.core_s += elapsed(t);
        out.work.nodes_processed += changed.stats.nodes_processed;
        out.work.candidates_screened += changed.stats.candidates_screened;
        out.work.adjacency_expansions += changed.stats.adjacency_expansions;
        out.work.source_distance_evaluations += changed.stats.source_distance_evaluations;
        out.work.pruning_distance_evaluations += changed.stats.pruning_distance_evaluations;
        t = Clock::now();
        writeLayer(result, h, m, changed.graph);
        out.writeback_s += elapsed(t);
    }
    result.setEf(200);
    return out;
}
