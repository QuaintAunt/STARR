// Reuse the benchmark migration, binary I/O and graph validation helpers.
#define main benchmark_main
#include "benchmark.cpp"
#undef main
#include <sched.h>

static const std::vector<size_t> EF_GRID = {10,  20,  40,  80,  120,  160, 200,
                                            300, 400, 600, 800, 1200, 1600};
static int64_t unixNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
static void expected(const char *name, uint64_t actual) {
    if (const char *s = std::getenv(name))
        check(std::to_string(actual) == s, std::string("reference mismatch: ") + name);
}
struct Round {
    size_t ef, ef_order, repeat, sequence, hits;
    int64_t loop_ns, started_ns, ended_ns;
    int cpu_before, cpu_after;
    double load_before[3] = {}, load_after[3] = {};
};
static Round timedRound(const Graph &graph, const Rows &target, const std::vector<size_t> &q,
                        const Truth &gt, size_t ef, size_t order, size_t repeat, size_t sequence) {
    Round r{};
    r.ef = ef;
    r.ef_order = order;
    r.repeat = repeat;
    r.sequence = sequence;
    r.cpu_before = sched_getcpu();
    getloadavg(r.load_before, 3);
    std::vector<std::vector<Id>> results;
    results.reserve(q.size());
    r.started_ns = unixNs();
    auto start = Clock::now();
    for (size_t i = 0; i < q.size(); ++i) {
        auto heap = graph.searchKnn(target[q[i]].data(), 10);
        std::vector<Id> ids;
        while (!heap.empty()) {
            ids.push_back(heap.top().second);
            heap.pop();
        }
        results.push_back(std::move(ids));
    }
    r.loop_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    r.ended_ns = unixNs();
    r.cpu_after = sched_getcpu();
    getloadavg(r.load_after, 3);
    check(r.loop_ns > 0, "nonpositive query time");
    // Count recall hits outside the timed loop.
    for (size_t i = 0; i < q.size(); ++i) {
        std::set<Id> unique(results[i].begin(), results[i].end());
        check(unique.size() == results[i].size() && unique.size() <= 10, "duplicate query result");
        for (Id x : unique)
            r.hits += std::find(gt[i].begin(), gt[i].end(), x) != gt[i].end();
    }
    if (const char *cpu = std::getenv("EXPECTED_CPU"))
        check(r.cpu_before == std::stoi(cpu) && r.cpu_after == std::stoi(cpu),
              "query moved off pinned CPU");
    return r;
}
static void writeRound(std::ostream &out, const Round &r, size_t nquery,
                       const std::string &method) {
    double seconds = r.loop_ns / 1e9;
    out << std::setprecision(17) << "{\"method\":\"" << method << "\",\"pid\":" << getpid()
        << ",\"ef_search\":" << r.ef << ",\"ef_order\":" << r.ef_order << ",\"round\":" << r.repeat
        << ",\"execution_sequence\":" << r.sequence << ",\"query_count\":" << nquery
        << ",\"query_loop_time_ns\":" << r.loop_ns << ",\"query_loop_time_s\":" << seconds
        << ",\"qps\":" << nquery / seconds << ",\"mean_query_ms\":" << 1000 * seconds / nquery
        << ",\"total_hits\":" << r.hits
        << ",\"recall_at_10_percent\":" << 100. * r.hits / (nquery * 10)
        << ",\"started_unix_ns\":" << r.started_ns << ",\"ended_unix_ns\":" << r.ended_ns
        << ",\"cpu_before\":" << r.cpu_before << ",\"cpu_after\":" << r.cpu_after
        << ",\"loadavg_before\":[" << r.load_before[0] << ',' << r.load_before[1] << ','
        << r.load_before[2] << "],\"loadavg_after\":[" << r.load_after[0] << ',' << r.load_after[1]
        << ',' << r.load_after[2] << "]}\n";
}
static void curve(const std::string &method, const std::string &oldpath, size_t olddim,
                  const std::string &newpath, size_t newdim, const std::string &index,
                  const std::string &truth, size_t n, const std::string &rawpath) {
    check(!std::filesystem::exists(rawpath), "refuse raw overwrite");
    auto target = readRows(newpath, n, newdim);
    auto q = actualQueryIds(n);
    check(q.size() == 1000, "expected1000queries");
    auto gt = readTruth(truth, q, n, newdim);
    hnswlib::L2Space space(method == "Fresh" ? newdim : olddim);
    Graph original(&space, index);
    check(original.cur_element_count == n && original.M_ == 16 && original.maxM_ == 16 &&
              original.maxM0_ == 32 && original.ef_construction_ == 200,
          "source/Fresh parameters differ");
    Audit source_before;
    if (method == "Fresh")
        source_before = inspect(original, &target);
    else {
        auto old = readRows(oldpath, n, olddim);
        source_before = inspect(original, &old);
    }
    expected("EXPECTED_SOURCE_STRUCTURE_HASH", source_before.structure);
    expected("EXPECTED_SOURCE_VECTOR_HASH", source_before.vectors);
    Owned ours;
    Migration dgm;
    Graph *result = &original;
    size_t generation_count = 0;
    if (method == "Ours") {
        StableMigrationOptions opts;
        opts.z_tolerance = 2;
        opts.search_ef = 16;
        opts.fine_repair_passes = 0;
        hnswlib::SpaceInterface<float> *ptr = nullptr;
        auto *graph = migrateHnswIndexStable(&original, target, &ptr, opts);
        ours.space.reset(ptr);
        ours.graph.reset(graph);
        result = graph;
        ++generation_count;
        sameMetadata(original, *result, false);
    } else if (method != "Fresh") {
        check(method == "DGM-Local" || method == "DGM-Search", "unknown method");
        dgm = migrateDgm(original, target, method == "DGM-Local");
        result = dgm.target.graph.get();
        ++generation_count;
        sameMetadata(original, *result, true);
    }
    result->setEf(EF_GRID.front());
    auto before = inspect(*result, &target);
    unchanged(source_before, inspect(original));
    expected("EXPECTED_STRUCTURE_HASH", before.structure);
    expected("EXPECTED_VECTOR_HASH", before.vectors);
    if (method == "Ours")
        check(before.reachable == n, "Full reachability failure");
    std::cout << "INDEX_READY {\"method\":\"" << method << "\",\"pid\":" << getpid()
              << ",\"generation_count\":" << generation_count << ",\"structure_hash\":\""
              << before.structure << "\"}" << std::endl;
    std::ofstream output(rawpath);
    check(bool(output), "cannot create raw file");
    bool all_consistent = true;
    size_t sequence = 0;
    for (size_t i = 0; i < EF_GRID.size(); ++i) {
        const size_t ef = EF_GRID[i];
        result->setEf(ef); // Always explicit, including ef=200.
        const Graph &query_index = *result;
        for (size_t j = 0; j < q.size(); ++j) {
            auto heap = query_index.searchKnn(target[q[j]].data(), 10);
            std::vector<Id> ids;
            while (!heap.empty()) {
                ids.push_back(heap.top().second);
                heap.pop();
            }
        }
        std::vector<Round> rounds;
        for (size_t repeat = 1; repeat <= 5; ++repeat)
            rounds.push_back(timedRound(query_index, target, q, gt, ef, i + 1, repeat, ++sequence));
        bool same = true;
        for (const auto &r : rounds) {
            same = same && r.hits == rounds[0].hits;
            writeRound(output, r, q.size(), method);
        }
        output.flush();
        check(bool(output), "raw write failed");
        all_consistent = all_consistent && same;
        std::cout << "CONFIG_DONE {\"ef_search\":" << ef
                  << ",\"rounds\":5,\"recall_consistent\":" << (same ? "true" : "false") << "}"
                  << std::endl;
    }
    auto after = inspect(*result, &target), source_after = inspect(original);
    unchanged(before, after);
    unchanged(source_before, source_after);
    std::cout << std::setprecision(17) << "RESULT {\"method\":\"" << method
              << "\",\"pid\":" << getpid() << ",\"warmup_queries_per_config\":1000,\"n\":" << n
              << ",\"query_count\":1000,\"threads\":1,\"query_ids_loaded\":"
              << (std::getenv("QUERY_IDS_BINARY") ? "true" : "false")
              << ",\"timed_rounds\":" << sequence << ",\"generation_count\":" << generation_count
              << ",\"all_recall_consistent\":" << (all_consistent ? "true" : "false")
              << ",\"index_unchanged\":true,\"source_unchanged\":true,\"cpu\":" << sched_getcpu()
              << ",\"ef_grid\":";
    array(std::cout, EF_GRID);
    std::cout << ",\"output_before\":";
    auditJson(std::cout, before);
    std::cout << ",\"output_after\":";
    auditJson(std::cout, after);
    std::cout << ",\"source_before\":";
    auditJson(std::cout, source_before);
    std::cout << ",\"source_after\":";
    auditJson(std::cout, source_after);
    std::cout << "}" << std::endl;
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "self-test")
            return selfTest();
        check(argc == 11 && std::string(argv[1]) == "curve",
              "curve METHOD OLD OLD_DIM TARGET TARGET_DIM INDEX GT N RAW_JSONL");
        curve(argv[2], argv[3], std::stoull(argv[4]), argv[5], std::stoull(argv[6]), argv[7],
              argv[8], std::stoull(argv[9]), argv[10]);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
