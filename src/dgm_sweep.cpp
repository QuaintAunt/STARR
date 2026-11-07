// Reuse the benchmark data loader, query evaluation and graph validation helpers.
#define main benchmark_main
#include "benchmark.cpp"
#undef main
#include "dgm_adapter.inc"
#include <sched.h>

static void expected(const char *name, uint64_t actual) {
    if (const char *v = std::getenv(name))
        check(std::stoull(v) == actual, std::string("reference mismatch ") + name);
}
static void measureSweep(int argc, char **argv) {
    check(argc == 14, "run METHOD OLD OLD_DIM TARGET TARGET_DIM INDEX GT N C L H ALPHA");
    std::string method = argv[2];
    bool local = method == "DGM-Local";
    check(local || method == "DGM-Search", "invalid method");
    size_t olddim = std::stoull(argv[4]), newdim = std::stoull(argv[6]), n = std::stoull(argv[9]),
           C = std::stoull(argv[10]), L = std::stoull(argv[11]), H = std::stoull(argv[12]);
    double alpha = std::stod(argv[13]);
    check(alpha == 1.0 || alpha == 1.1 || alpha == 1.2, "alpha outside experiment grid");
    check(local ? (C == 64 || C == 128 || C == 256 || C == 400 || C == 800)
                : (L == 64 || L == 128 || L == 256 || L == 512),
          "width outside experiment grid");
    check(local ? (L == 0 && H == 0) : (C == 0 && (H == 2 || H == 3 || H == 4 || H == 6)),
          "inapplicable params or invalid hops");
    check(
        (!std::getenv("EXPECTED_CPU") || sched_getcpu() == std::stoi(std::getenv("EXPECTED_CPU"))),
        "CPU differs from EXPECTED_CPU");
    auto target = readRows(argv[5], n, newdim);
    auto q = actualQueryIds(n);
    auto gt = readTruth(argv[8], q, n, newdim);
    check(q.size() == 1000 && std::getenv("QUERY_IDS_BINARY"), "1,000 saved query IDs required");
    hnswlib::L2Space oldspace(olddim);
    Graph source(&oldspace, argv[7]);
    check(source.cur_element_count == n && source.max_elements_ == n && source.M_ == 16 &&
              source.maxM_ == 16 && source.maxM0_ == 32 && source.ef_construction_ == 200,
          "source config mismatch");
    Audit before;
    {
        auto old = readRows(argv[3], n, olddim);
        before = inspect(source, &old);
    }
    expected("EXPECTED_SOURCE_STRUCTURE_HASH", before.structure);
    expected("EXPECTED_SOURCE_VECTOR_HASH", before.vectors);
    std::cerr << "[migration] method=" << method << " C=" << C << " L=" << L << " H=" << H
              << " alpha=" << alpha << " original-source/all-layers/threads1" << std::endl;
    auto started = std::chrono::system_clock::now().time_since_epoch();
    auto t = Clock::now();
    auto changed = migrateDgmScanned(source, target, local, C, L, H, static_cast<float>(alpha));
    double migration = elapsed(t); // Full adapter call, not its DGM-only subtotal.
    std::cerr << "[migration] finished complete_seconds=" << std::setprecision(17) << migration
              << std::endl;
    auto &result = *changed.target.graph;
    sameMetadata(source, result, true);
    auto audit = inspect(result, &target);
    unchanged(before, inspect(source));
    size_t occupied = 0;
    for (const auto &layer : before.layers)
        occupied += layer.nodes;
    check(changed.work.nodes_processed == occupied, "not all occupied nodes/layers processed");
    if (local) {
        check(changed.layer_samples.size() == before.layers.size(), "missing MPS layer");
        for (size_t h = 0; h < before.layers.size(); ++h)
            check(changed.layer_samples[h] == std::min(size_t(4096), before.layers[h].nodes),
                  "MPS calibration mapping mismatch");
    }
    expected("EXPECTED_RESULT_STRUCTURE_HASH", audit.structure);
    expected("EXPECTED_RESULT_VECTOR_HASH", audit.vectors);
    expected("EXPECTED_RESULT_REACHABLE", audit.reachable);
    auto ev = evaluate(
        result, target, q,
        gt); // Warm up 1,000 queries and time one batch; count recall hits outside the timer.
    auto after = inspect(source), after_query = inspect(result, &target);
    unchanged(before, after);
    unchanged(audit, after_query);
    expected("EXPECTED_RESULT_HITS", ev.hits);
    check(
        (!std::getenv("EXPECTED_CPU") || sched_getcpu() == std::stoi(std::getenv("EXPECTED_CPU"))),
        "CPU affinity changed");
    std::cout << std::setprecision(17)
              << "RESULT {\"status\":\"complete\",\"method\":" << std::quoted(method)
              << ",\"n\":" << n << ",\"source_dim\":" << olddim << ",\"target_dim\":" << newdim
              << ",\"C\":" << (local ? std::to_string(C) : "null")
              << ",\"L\":" << (local ? "null" : std::to_string(L))
              << ",\"H\":" << (local ? "null" : std::to_string(H)) << ",\"alpha\":" << alpha
              << ",\"effective_alpha_float32\":" << static_cast<float>(alpha)
              << ",\"migration_s\":" << migration << ",\"query_count\":" << q.size()
              << ",\"total_hits\":" << ev.hits << ",\"recall_at_10_percent\":" << ev.recall
              << ",\"reachable\":" << audit.reachable << ",\"unreachable\":" << n - audit.reachable
              << ",\"query_ef\":200,\"k\":10,\"include_self\":true,\"warmup_queries\":1000,\"timed_"
                 "query_rounds\":1"
              << ",\"query_time_ms\":" << ev.ms << ",\"query_total_s\":" << ev.total_s
              << ",\"query_total_ns\":" << ev.total_ns
              << ",\"query_ids_loaded\":true,\"migration_threads\":1,\"query_threads\":1,\"pid\":"
              << getpid() << ",\"cpu\":" << sched_getcpu() << ",\"migration_started_unix_ns\":"
              << std::chrono::duration_cast<std::chrono::nanoseconds>(started).count()
              << ",\"source_unchanged\":true,\"index_unchanged_during_query\":true,\"metadata_"
                 "valid\":true,\"target_vectors_valid\":true,\"all_layers_processed\":true"
              << ",\"preparation_s\":" << changed.preparation_s
              << ",\"core_including_MPS_s\":" << changed.core_s
              << ",\"writeback_s\":" << changed.writeback_s
              << ",\"MPS_calibration_requested\":" << (local ? "4096" : "null")
              << ",\"MPS_seed\":" << (local ? "20250311" : "null")
              << ",\"work_block_size\":" << (local ? 64 : 8)
              << ",\"start\":\"original source graph with target vectors; no Local "
                 "precursor\",\"postprocessing\":\"none\",\"layer_order\":\"highest_to_base\","
                 "\"MPS_layer_samples\":";
    array(std::cout, changed.layer_samples);
    std::cout << ",\"work\":{\"nodes_processed\":" << changed.work.nodes_processed
              << ",\"candidates_screened\":" << changed.work.candidates_screened
              << ",\"adjacency_expansions\":" << changed.work.adjacency_expansions
              << ",\"source_distance_evaluations\":" << changed.work.source_distance_evaluations
              << ",\"pruning_distance_evaluations\":" << changed.work.pruning_distance_evaluations
              << "}" << ",\"source_before\":";
    auditJson(std::cout, before);
    std::cout << ",\"source_after\":";
    auditJson(std::cout, after);
    std::cout << ",\"output_before_query\":";
    auditJson(std::cout, audit);
    std::cout << ",\"output_after_query\":";
    auditJson(std::cout, after_query);
    std::cout << "}" << std::endl;
}
static int sweepSelfTest() {
    const size_t n = 128;
    Rows old(n, std::vector<float>(5)), target(n, std::vector<float>(7));
    std::mt19937 rng(70);
    std::normal_distribution<float> normal;
    for (auto &v : old)
        for (float &x : v)
            x = normal(rng);
    for (auto &v : target)
        for (float &x : v)
            x = normal(rng);
    hnswlib::L2Space sp(5);
    Graph source(&sp, n, 16, 200, 100);
    for (size_t i = 0; i < n; ++i)
        source.addPoint(old[i].data(), i * 7 + 101);
    auto before = inspect(source, &old, false);
    size_t count = 0;
    for (bool local : {true, false}) {
        auto ref = migrateDgm(source, target, local),
             sweep = migrateDgmScanned(source, target, local, 400, 256, 4, 1.1f);
        auto a = inspect(*ref.target.graph, &target, false),
             b = inspect(*sweep.target.graph, &target, false);
        unchanged(a, b);
        check(ref.work.nodes_processed == sweep.work.nodes_processed &&
                  ref.work.source_distance_evaluations == sweep.work.source_distance_evaluations &&
                  ref.work.pruning_distance_evaluations ==
                      sweep.work.pruning_distance_evaluations &&
                  ref.layer_samples == sweep.layer_samples,
              "default adapter differs from benchmark adapter");
    }
    for (size_t C : {64, 128, 256, 400, 800})
        for (float alpha : {1.f, 1.1f, 1.2f}) {
            auto output = migrateDgmScanned(source, target, true, C, 0, 0, alpha);
            sameMetadata(source, *output.target.graph, true);
            inspect(*output.target.graph, &target, false);
            unchanged(before, inspect(source, &old, false));
            ++count;
        }
    for (size_t L : {64, 128, 256, 512})
        for (size_t H : {2, 3, 4, 6})
            for (float alpha : {1.f, 1.1f, 1.2f}) {
                auto output = migrateDgmScanned(source, target, false, 0, L, H, alpha);
                sameMetadata(source, *output.target.graph, true);
                inspect(*output.target.graph, &target, false);
                unchanged(before, inspect(source, &old, false));
                ++count;
            }
    check(count == 63, "grid cardinality mismatch");
    std::cout << "PASS: 63 parameter combinations; default all-layer/work-stat equality with "
                 "benchmark adapter; dimensions 5->7; external labels; capacities/entry/levels; "
                 "source immutability; no postprocessing"
              << std::endl;
    return 0;
}
int main(int argc, char **argv) {
    try {
        check(argc >= 2 && (!std::getenv("EXPECTED_CPU") ||
                            sched_getcpu() == std::stoi(std::getenv("EXPECTED_CPU"))),
              "mode required; CPU must match EXPECTED_CPU when set");
        std::string mode = argv[1];
        if (mode == "self-test")
            return sweepSelfTest();
        if (mode == "fixture") {
            check(argc == 3, "fixture directory");
            makeFixture(argv[2]);
            return 0;
        }
        check(mode == "run", "invalid mode");
        measureSweep(argc, argv);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
