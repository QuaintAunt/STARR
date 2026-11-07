// One target embedding per process: reuse its vectors, fresh graph and exact truth.
#include "evaluation.h"
#include "starr.h"
#include <iomanip>
#include <future>
#include <atomic>
#include <thread>

static int evaluationSelfTest(const fs::path &fixture) {
    constexpr size_t n = 1500;
    const auto target = loadPrefix((fixture / "target.bin").string(), 7, n, 0);
    const auto ids = actualQueryIds(n);
    if (ids.size() != kQueryCount)
        throw std::runtime_error("query count mismatch");
    const auto queries = sampleQueries(target.base, ids);
    const auto truths = makeTruths(target.base, queries);
    const auto path = fixture / "evaluation.gt";
    saveTruths(path, truths, ids);
    if (loadTruths(path, ids) != truths)
        throw std::runtime_error("truth cache roundtrip mismatch");
    auto reordered = ids;
    std::swap(reordered[0], reordered[1]);
    bool rejected = false;
    try {
        loadTruths(path, reordered);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("truth cache accepted mismatched query IDs");
    hnswlib::L2Space space(7);
    hnswlib::HierarchicalNSW<float> index(&space, (fixture / "target.hnsw").string());
    const auto result = evaluate(&index, queries, truths);
    if (result.recall < 0.99)
        throw std::runtime_error("query evaluation recall mismatch");
    std::cout << "RESULT {\"query_count\":" << ids.size()
              << ",\"warmup_queries\":" << queries.size()
              << ",\"recall_at_10_percent\":" << result.recall * 100 << ",\"query_ids\":[";
    for (size_t i = 0; i < ids.size(); ++i)
        std::cout << (i ? "," : "") << ids[i];
    std::cout << "]}" << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "self-test")
            return evaluationSelfTest(argv[2]);
        // TARGET TARGET_DIM TARGET_BUILD_S (OLD OLD_DIM OLD_BUILD_S)...
        if (argc < 7 || (argc - 4) % 3)
            throw std::invalid_argument(
                "usage: TARGET TARGET_DIM TARGET_BUILD_S OLD OLD_DIM OLD_BUILD_S ...");
        const std::string target_name = argv[1];
        const int target_dim = std::stoi(argv[2]);
        const double target_build_s = std::stod(argv[3]);
        constexpr size_t n = 1'000'000, q = kQueryCount;
        const fs::path vectors_dir = "data/MsMarco";
        const fs::path indexes_dir = "indexes/MsMarco";
        const auto target = loadPrefix((vectors_dir / ("msmarco_" + target_name + ".bin")).string(),
                                       target_dim, n, 0);
        const auto query_ids = actualQueryIds(n);
        const auto queries = sampleQueries(target.base, query_ids);
        const auto truth_dir = fs::path("truth/MsMarco");
        const auto truth_path = truth_dir / (target_name + "_1000_queries.bin");
        std::vector<std::unordered_set<hnswlib::labeltype>> truths;
        if (fs::exists(truth_path)) {
            truths = loadTruths(truth_path, query_ids);
        } else {
            truths.resize(q);
            std::atomic<size_t> next{0};
            std::vector<std::future<void>> workers;
            const size_t worker_count =
                std::min<size_t>(q, std::max(1u, std::thread::hardware_concurrency()));
            std::cerr << "[truth] " << target_name << " q=" << q << " workers=" << worker_count
                      << std::endl;
            for (size_t worker = 0; worker < worker_count; ++worker)
                workers.push_back(std::async(std::launch::async, [&] {
                    for (size_t i = next.fetch_add(1); i < q; i = next.fetch_add(1))
                        truths[i] = truth(target.base, queries[i], 10);
                }));
            for (auto &worker : workers)
                worker.get();
            fs::create_directories(truth_dir);
            saveTruths(truth_path, truths, query_ids);
        }
        hnswlib::L2Space target_space(target_dim);
        auto fresh = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            &target_space, (indexes_dir / ("msmarco_" + target_name + "_hnsw.bin")).string());
        const auto fresh_eval = evaluate(fresh.get(), queries, truths);
        std::cerr << "[target] " << target_name << " fresh_recall=" << fresh_eval.recall * 100
                  << std::endl;
        for (int a = 4; a < argc; a += 3) {
            const std::string old_name = argv[a];
            const int old_dim = std::stoi(argv[a + 1]);
            const double old_build_s = std::stod(argv[a + 2]);
            hnswlib::L2Space old_space(old_dim);
            auto old = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                &old_space, (indexes_dir / ("msmarco_" + old_name + "_hnsw.bin")).string());
            hnswlib::SpaceInterface<float> *migrated_space_raw = nullptr;
            StableMigrationStats stats;
            const auto start = Clock::now();
            auto *migrated_raw =
                migrateHnswIndexStable(old.get(), target.base, &migrated_space_raw, {}, &stats);
            const double migration_s = std::chrono::duration<double>(Clock::now() - start).count();
            std::unique_ptr<hnswlib::SpaceInterface<float>> migrated_space(migrated_space_raw);
            std::unique_ptr<hnswlib::HierarchicalNSW<float>> migrated(migrated_raw);
            const auto migrated_eval = evaluate(migrated.get(), queries, truths);
            std::cout << std::setprecision(10) << 2 << ",MsMarco," << old_name << ',' << target_name
                      << ',' << old_build_s << ',' << target_build_s << ',' << migration_s << ','
                      << fresh_eval.milliseconds << ',' << migrated_eval.milliseconds << ','
                      << fresh_eval.recall * 100 << ',' << migrated_eval.recall * 100 << ','
                      << overlap(migrated_eval, fresh_eval) * 100 << ','
                      << double(stats.repaired) * 100 / n << std::endl;
            std::cerr << "[done] " << old_name << " -> " << target_name
                      << " recall=" << migrated_eval.recall * 100 << " time=" << migration_s << "s"
                      << std::endl;
        }
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
