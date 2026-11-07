// Reuse the benchmark data loader, exact ground truth and graph validation helpers.
// The included benchmark entry point is unused by this executable.
#define main benchmark_main
#include "benchmark.cpp"
#undef main
#include <sched.h>

static const std::vector<std::string> variants = {"full",
                                                  "no_rho",
                                                  "no_restore",
                                                  "no_share",
                                                  "no_connect",
                                                  "no_share_no_connect",
                                                  "no_batch_reverse"};
static StableMigrationOptions ablationOptions(const std::string &variant) {
    check(std::find(variants.begin(), variants.end(), variant) != variants.end(),
          "unknown ablation");
    StableMigrationOptions o;
    o.z_tolerance = 2;
    o.search_ef = 16;
    o.fine_repair_passes = 0;
    if (variant == "no_rho")
        o.correlation_scaling = false;
    if (variant == "no_restore")
        o.restore_survivors = false;
    if (variant == "no_share" || variant == "no_share_no_connect")
        o.share_candidates = false;
    if (variant == "no_connect" || variant == "no_share_no_connect")
        o.repair_connectivity = false;
    if (variant == "no_batch_reverse")
        o.batch_stable_reverse = false;
    return o;
}
static void expectedHash(const char *variable, uint64_t actual) {
    if (const char *s = std::getenv(variable))
        check(std::stoull(s) == actual, std::string("reference mismatch: ") + variable);
}
static std::string measure(Graph &original, const Rows &target, const std::vector<size_t> &q,
                           const Truth &gt, const Audit &before, const std::string &variant) {
    auto options = ablationOptions(variant);
    StableMigrationStats stats;
    Owned output;
    hnswlib::SpaceInterface<float> *space = nullptr;
    int cpu = sched_getcpu();
    check((!std::getenv("EXPECTED_CPU") || cpu == std::stoi(std::getenv("EXPECTED_CPU"))),
          "query/migration CPU differs from EXPECTED_CPU");
    const auto wall_start = std::chrono::system_clock::now().time_since_epoch();
    auto start = Clock::now();
    auto *migrated = migrateHnswIndexStable(&original, target, &space, options, &stats);
    const double migration =
        elapsed(start); // Complete function call, including allocation and all actual stages.
    output.space.reset(space);
    output.graph.reset(migrated);
    migrated->setEf(200);
    sameMetadata(original, *migrated, false);
    auto a = inspect(*migrated, &target);
    unchanged(before, inspect(original));
    if (options.repair_connectivity)
        check(a.reachable == target.size(), "enabled connectivity repair failed");
    if (!options.share_candidates)
        check(stats.shared_edges == 0, "disabled candidate sharing executed");
    if (!options.repair_connectivity)
        check(stats.bridges == 0, "disabled connectivity repair executed");
    check(stats.refined == 0, "disabled optional local repair executed");
    check(stats.batched_stable_copy == (stats.batch_gate_active && options.batch_stable_reverse),
          "batch execution flag mismatch");
    if (variant == "full") {
        expectedHash("EXPECTED_FULL_STRUCTURE_HASH", a.structure);
        expectedHash("EXPECTED_FULL_VECTOR_HASH", a.vectors);
    }
    auto ev = evaluate(*migrated, target, q,
                       gt); // Warm up all queries, then time one complete query batch.
    auto after_query = inspect(*migrated, &target);
    unchanged(a, after_query);
    auto source_after = inspect(original);
    unchanged(before, source_after);
    if (variant == "full")
        expectedHash("EXPECTED_FULL_HITS", ev.hits);
    check(
        (!std::getenv("EXPECTED_CPU") || sched_getcpu() == std::stoi(std::getenv("EXPECTED_CPU"))),
        "CPU affinity changed");
    std::ostringstream out;
    out << std::setprecision(17) << "{\"status\":\"complete\",\"variant\":" << std::quoted(variant)
        << ",\"n\":" << target.size() << ",\"source_dim\":" << original.data_size_ / 4
        << ",\"target_dim\":" << target[0].size() << ",\"migration_s\":" << migration
        << ",\"query_count\":" << q.size() << ",\"total_hits\":" << ev.hits
        << ",\"recall_at_10_percent\":" << ev.recall << ",\"reachable\":" << a.reachable
        << ",\"unreachable\":" << target.size() - a.reachable
        << ",\"batch_gate_active\":" << int(stats.batch_gate_active)
        << ",\"batch_executed\":" << int(stats.batched_stable_copy)
        << ",\"query_ef\":200,\"k\":10,\"include_self\":true,\"warmup_queries\":" << q.size()
        << ",\"timed_query_rounds\":1" << ",\"query_time_ms\":" << ev.ms
        << ",\"query_total_s\":" << ev.total_s << ",\"query_total_ns\":" << ev.total_ns
        << ",\"migration_threads\":1,\"query_threads\":1,\"pid\":" << getpid() << ",\"cpu\":" << cpu
        << ",\"migration_started_unix_ns\":"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(wall_start).count()
        << ",\"query_ids_loaded\":" << (std::getenv("QUERY_IDS_BINARY") ? "true" : "false")
        << ",\"source_unchanged\":true,\"index_unchanged_during_query\":true,\"vectors_valid\":"
           "true,\"metadata_valid\":true"
        << ",\"options\":{\"tau\":2,\"B\":16,\"fine_repair_passes\":0,\"correlation_scaling\":"
        << int(options.correlation_scaling)
        << ",\"restore_survivors\":" << int(options.restore_survivors)
        << ",\"share_candidates\":" << int(options.share_candidates)
        << ",\"repair_connectivity\":" << int(options.repair_connectivity)
        << ",\"batch_stable_reverse\":" << int(options.batch_stable_reverse) << "}"
        << ",\"stats\":{\"repaired\":" << stats.repaired << ",\"edges\":" << stats.edges
        << ",\"removed_edges\":" << stats.removed_edges
        << ",\"shared_edges\":" << stats.shared_edges << ",\"bridges\":" << stats.bridges
        << ",\"refined\":" << stats.refined << ",\"register_s\":" << stats.register_s
        << ",\"filter_s\":" << stats.filter_s << ",\"reinsert_s\":" << stats.reinsert_s
        << ",\"refine_s\":" << stats.refine_s << ",\"connectivity_s\":" << stats.connectivity_s
        << "}" << ",\"source_before\":";
    auditJson(out, before);
    out << ",\"source_after\":";
    auditJson(out, source_after);
    out << ",\"output_before_query\":";
    auditJson(out, a);
    out << ",\"output_after_query\":";
    auditJson(out, after_query);
    out << "}";
    return out.str();
}
static int batch(int argc, char **argv) {
    check(argc == 10, "batch SOURCE SOURCE_DIM TARGET TARGET_DIM SOURCE_INDEX GT N OUTPUT_PREFIX");
    const size_t olddim = std::stoull(argv[3]), newdim = std::stoull(argv[5]),
                 n = std::stoull(argv[8]);
    auto target = readRows(argv[4], n, newdim);
    auto q = actualQueryIds(n);
    auto gt = readTruth(argv[7], q, n, newdim);
    hnswlib::L2Space oldspace(olddim);
    Graph source(&oldspace, argv[6]);
    check(source.cur_element_count == n && source.M_ == 16 && source.maxM_ == 16 &&
              source.maxM0_ == 32 && source.ef_construction_ == 200,
          "formal source parameters mismatch");
    Audit before;
    {
        auto old = readRows(argv[2], n, olddim);
        before = inspect(source, &old);
    }
    expectedHash("EXPECTED_SOURCE_STRUCTURE_HASH", before.structure);
    expectedHash("EXPECTED_SOURCE_VECTOR_HASH", before.vectors);
    check(q.size() == 1000 && std::getenv("QUERY_IDS_BINARY"), "1,000 explicit query IDs required");
    size_t complete = 0, failed = 0;
    for (const auto &variant : variants) {
        auto path = std::string(argv[9]) + "_" + variant + ".raw.json";
        check(!std::filesystem::exists(path), "refusing raw result overwrite");
        std::cerr << "[migration] starting " << variant
                  << " from source graph; tau=2 B=16 fine_repair_passes=0" << std::endl;
        std::string raw;
        try {
            raw = measure(source, target, q, gt, before, variant);
            ++complete;
        } catch (const std::exception &e) {
            std::ostringstream s;
            s << "{\"status\":\"failed\",\"variant\":" << std::quoted(variant)
              << ",\"reason\":" << std::quoted(e.what()) << "}";
            raw = s.str();
            ++failed;
        }
        std::ofstream f(path);
        check(bool(f), "raw file create failed");
        f << raw << '\n';
        f.flush();
        check(bool(f), "raw file write failed");
        std::cout << "VARIANT " << raw << std::endl;
        unchanged(before, inspect(source));
    }
    std::cout << "RESULT {\"completed\":" << complete << ",\"failed\":" << failed
              << ",\"source_unchanged\":true,\"n\":" << n << ",\"pid\":" << getpid() << "}"
              << std::endl;
    return 0;
}
static int ablationSelfTest() {
    const size_t n = 1500;
    Rows old(n, std::vector<float>(5)), target(n, std::vector<float>(7));
    std::mt19937 gen(9);
    std::normal_distribution<float> normal;
    for (auto &v : old)
        for (float &x : v)
            x = normal(gen);
    for (auto &v : target)
        for (float &x : v)
            x = normal(gen);
    hnswlib::L2Space sp(5);
    Graph src(&sp, n, 16, 200, 100);
    for (size_t i = 0; i < n; ++i)
        src.addPoint(old[i].data(), i);
    auto before = inspect(src, &old);
    auto q = actualQueryIds(n);
    auto gt = exact(target, q, 1);
    for (const auto &variant : variants) {
        auto o = ablationOptions(variant);
        unsigned disabled = !o.share_candidates + !o.repair_connectivity + !o.correlation_scaling +
                            !o.restore_survivors + !o.batch_stable_reverse;
        check(disabled == (variant == "full"                  ? 0
                           : variant == "no_share_no_connect" ? 2
                                                              : 1),
              "ablation switches mismatch");
        auto result = measure(src, target, q, gt, before, variant);
        check(result.find("\"status\":\"complete\"") != std::string::npos,
              "self test missing result");
    }
    for (const auto &variant : std::vector<std::string>{"full", "no_batch_reverse"}) {
        auto o = ablationOptions(variant);
        StableMigrationStats stats;
        Owned out;
        hnswlib::SpaceInterface<float> *s = nullptr;
        out.graph.reset(migrateHnswIndexStable(&src, old, &s, o, &stats));
        out.space.reset(s);
        check(stats.removed_edges == 0 && stats.batch_gate_active,
              "identity-space batch gate fixture failed");
        check(stats.batched_stable_copy == (variant == "full"),
              "gate active/executed distinction failed");
    }
    bool rejected = false;
    try {
        ablationOptions("invalid");
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected, "unknown variant accepted");
    std::cout << "PASS seven variants, independent original source, cross dimensions, all-layer "
                 "audits, query immutability, reachability and module flags, batch gate versus "
                 "execution, unknown variant rejection"
              << std::endl;
    return 0;
}
int main(int argc, char **argv) {
    try {
        check(argc >= 2, "mode required");
        check((!std::getenv("EXPECTED_CPU") ||
               sched_getcpu() == std::stoi(std::getenv("EXPECTED_CPU"))),
              "CPU differs from EXPECTED_CPU");
        std::string mode = argv[1];
        if (mode == "self-test")
            return ablationSelfTest();
        if (mode == "fixture") {
            check(argc == 3, "fixture output");
            makeFixture(argv[2]);
            return 0;
        }
        check(mode == "batch", "invalid mode");
        return batch(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
