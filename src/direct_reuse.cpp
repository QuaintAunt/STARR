// Reuse the shared query IDs, truth-cache reader and evaluator.
// This executable only copies the graph and substitutes target vectors.
#include "matrix/evaluation.h"
#include <iomanip>
#include <limits>

using Hnsw = hnswlib::HierarchicalNSW<float>;
using FloatRows = std::vector<std::vector<float>>;

static void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct DirectIndex {
    std::unique_ptr<hnswlib::L2Space> space;
    std::unique_ptr<Hnsw> graph; // Destroy graph before its distance space.
};

// target rows are in source INTERNAL-ID order; formal preflight verifies row labels.
// Only allocate/copy. In particular, no addPoint/updatePoint or graph optimization.
static DirectIndex copyDirect(const Hnsw &source, const FloatRows &target) {
    DirectIndex result;
    result.space = std::make_unique<hnswlib::L2Space>(target.front().size());
    result.graph = std::make_unique<Hnsw>(result.space.get(), source.max_elements_, source.M_,
                                          source.ef_construction_);
    auto &graph = *result.graph;
    require(graph.maxM_ == source.maxM_ && graph.maxM0_ == source.maxM0_ &&
                graph.size_links_level0_ == source.size_links_level0_ &&
                graph.size_links_per_element_ == source.size_links_per_element_,
            "unsupported nonstandard source capacity/layout");
    for (tableint id = 0; id < source.cur_element_count; ++id) {
        std::memset(graph.data_level0_memory_ + id * graph.size_data_per_element_, 0,
                    graph.size_data_per_element_);
        std::memcpy(graph.get_linklist0(id), source.get_linklist0(id), source.size_links_level0_);
        std::memcpy(graph.getDataByInternalId(id), target[id].data(), graph.data_size_);
        graph.setExternalLabel(id, source.getExternalLabel(id));
        const int level = source.element_levels_[id];
        graph.element_levels_[id] = level;
        graph.linkLists_[id] = nullptr;
        if (level) {
            const size_t bytes = graph.size_links_per_element_ * level;
            graph.linkLists_[id] = static_cast<char *>(std::calloc(1, bytes + 1));
            if (!graph.linkLists_[id])
                throw std::bad_alloc();
            // Loaded HNSW allocates exactly bytes, not bytes+1, in the source.
            std::memcpy(graph.linkLists_[id], source.linkLists_[id], bytes);
        }
        ++graph.cur_element_count;
        graph.label_lookup_[source.getExternalLabel(id)] = id;
    }
    graph.enterpoint_node_ = source.enterpoint_node_;
    graph.maxlevel_ = source.maxlevel_;
    graph.mult_ = source.mult_;
    graph.revSize_ = source.revSize_;
    graph.setEf(200);
    return result;
}

struct Structure {
    uint64_t hash = 14695981039346656037ULL;
    size_t edges = 0, lists = 0, reachable = 0;
};

static void mix(uint64_t &hash, uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte) {
        hash = (hash ^ (value & 255)) * 1099511628211ULL;
        value >>= 8;
    }
}

static Structure inspect(Hnsw &graph, const FloatRows &expected, bool row_labels) {
    const size_t n = graph.cur_element_count;
    require(n == expected.size() && graph.num_deleted_ == 0, "node/deletion mismatch");
    require(graph.enterpoint_node_ < n, "invalid entry node");
    Structure result;
    for (uint64_t value :
         {uint64_t(n), uint64_t(graph.max_elements_), uint64_t(graph.M_), uint64_t(graph.maxM_),
          uint64_t(graph.maxM0_), uint64_t(graph.ef_construction_), uint64_t(graph.maxlevel_),
          uint64_t(graph.enterpoint_node_)})
        mix(result.hash, value);
    for (tableint id = 0; id < n; ++id) {
        const auto label = graph.getExternalLabel(id);
        if (row_labels)
            require(label == id, "source/target label is not the object row ID");
        const auto found = graph.label_lookup_.find(label);
        require(found != graph.label_lookup_.end() && found->second == id, "label lookup mismatch");
        require(expected[id].size() * sizeof(float) == graph.data_size_, "wrong data dimension");
        require(std::memcmp(graph.getDataByInternalId(id), expected[id].data(), graph.data_size_) ==
                    0,
                "index vector does not match object row");
        require(std::all_of(expected[id].begin(), expected[id].end(),
                            [](float x) { return std::isfinite(x); }),
                "nonfinite input vector");
        mix(result.hash, id);
        mix(result.hash, label);
        mix(result.hash, graph.element_levels_[id]);
        for (int level = 0; level <= graph.element_levels_[id]; ++level) {
            auto *list = graph.get_linklist_at_level(id, level);
            const size_t count = graph.getListCount(list);
            require(count <= (level ? graph.maxM_ : graph.maxM0_), "degree exceeds capacity");
            const auto *ids = reinterpret_cast<const tableint *>(list + 1);
            ++result.lists;
            result.edges += count;
            mix(result.hash, level);
            mix(result.hash, count);
            for (size_t j = 0; j < count; ++j) {
                require(ids[j] < n && graph.element_levels_[ids[j]] >= level,
                        "invalid neighbor ID/layer");
                mix(result.hash, ids[j]);
            }
        }
    }
    std::vector<unsigned char> seen(n, 0);
    std::vector<tableint> queue{graph.enterpoint_node_};
    seen[queue.front()] = 1;
    for (size_t j = 0; j < queue.size(); ++j) {
        auto *list = graph.get_linklist0(queue[j]);
        const auto *ids = reinterpret_cast<const tableint *>(list + 1);
        for (size_t k = 0; k < graph.getListCount(list); ++k)
            if (!seen[ids[k]]) {
                seen[ids[k]] = 1;
                queue.push_back(ids[k]);
            }
    }
    result.reachable = queue.size();
    return result;
}

static void compareStructures(Hnsw &source, Hnsw &reused) {
    require(source.cur_element_count == reused.cur_element_count &&
                source.max_elements_ == reused.max_elements_ && source.M_ == reused.M_ &&
                source.maxM_ == reused.maxM_ && source.maxM0_ == reused.maxM0_ &&
                source.ef_construction_ == reused.ef_construction_ &&
                source.maxlevel_ == reused.maxlevel_ &&
                source.enterpoint_node_ == reused.enterpoint_node_ &&
                source.mult_ == reused.mult_ && source.revSize_ == reused.revSize_,
            "global structure differs");
    for (tableint id = 0; id < source.cur_element_count; ++id) {
        require(source.getExternalLabel(id) == reused.getExternalLabel(id) &&
                    source.element_levels_[id] == reused.element_levels_[id],
                "node metadata differs");
        for (int level = 0; level <= source.element_levels_[id]; ++level) {
            auto *left = source.get_linklist_at_level(id, level);
            auto *right = reused.get_linklist_at_level(id, level);
            const size_t count = source.getListCount(left);
            require(count == reused.getListCount(right), "neighbor count differs");
            require(std::memcmp(left + 1, right + 1, count * sizeof(tableint)) == 0,
                    "neighbor IDs/order differ");
        }
    }
}

static int selfTest() {
    // Synthetic source fixture is manually populated: no real Fresh build/query.
    const size_t n = 6;
    FloatRows old(n, std::vector<float>(3)), target(n, std::vector<float>(5));
    hnswlib::L2Space space(3);
    Hnsw source(&space, n, 2, 200);
    for (tableint id = 0; id < n; ++id) {
        old[id] = {float(id), 1.f, 2.f};
        target[id] = {9.f, float(id), 3.f, 4.f, 5.f};
        std::memset(source.data_level0_memory_ + id * source.size_data_per_element_, 0,
                    source.size_data_per_element_);
        std::memcpy(source.getDataByInternalId(id), old[id].data(), source.data_size_);
        const size_t label = 100 + id * 7;
        source.setExternalLabel(id, label);
        source.label_lookup_[label] = id;
        source.element_levels_[id] = id < 3 ? 2 : 0;
        source.linkLists_[id] = nullptr;
        if (id < 3)
            source.linkLists_[id] =
                static_cast<char *>(std::calloc(1, 2 * source.size_links_per_element_));
        ++source.cur_element_count;
        auto *list = source.get_linklist0(id);
        auto *ids = reinterpret_cast<tableint *>(list + 1);
        ids[0] = (id + 2) % n;
        ids[1] = (id + 1) % n;
        source.setListCount(list, 2);
        if (id < 3)
            for (int level = 1; level <= 2; ++level) {
                list = source.get_linklist(id, level);
                ids = reinterpret_cast<tableint *>(list + 1);
                ids[0] = (id + 2) % 3;
                ids[1] = (id + 1) % 3;
                source.setListCount(list, 2);
            }
    }
    source.enterpoint_node_ = 1;
    source.maxlevel_ = 2;
    const auto before = inspect(source, old, false);
    auto copied = copyDirect(source, target);
    compareStructures(source, *copied.graph);
    const auto after = inspect(source, old, false), result = inspect(*copied.graph, target, false);
    require(before.hash == after.hash && before.hash == result.hash, "structure hash differs");
    auto *list = copied.graph->get_linklist(0, 2);
    auto *ids = reinterpret_cast<tableint *>(list + 1);
    std::swap(ids[0], ids[1]);
    bool detected = false;
    try {
        compareStructures(source, *copied.graph);
    } catch (const std::runtime_error &) {
        detected = true;
    }
    require(detected, "test failed to detect upper-layer neighbor order corruption");
    std::cout
        << "PASS: dimension-changing copy, all-layer neighbor order, arbitrary labels, "
           "entry/levels/capacities, target vectors, source immutability, corruption detection\n";
    return 0;
}

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test")
            return selfTest();
        require(argc == 8,
                "OLD_BIN OLD_DIM TARGET_BIN TARGET_DIM OLD_INDEX TARGET_INDEX TRUTH_CACHE");
        constexpr size_t n = 1000000, q = kQueryCount;
        const int old_dim = std::stoi(argv[2]), target_dim = std::stoi(argv[4]);
        const auto old_vectors = loadPrefix(argv[1], old_dim, n, 0);
        const auto target = loadPrefix(argv[3], target_dim, n, 0);
        const auto query_ids = actualQueryIds(n);
        const auto queries = sampleQueries(target.base, query_ids);
        const bool cached = fs::exists(argv[7]);
        const auto gt = cached ? loadTruths(argv[7], query_ids) : makeTruths(target.base, queries);
        if (!cached) {
            fs::create_directories(fs::path(argv[7]).parent_path());
            saveTruths(argv[7], gt, query_ids);
        }
        for (size_t i = 0; i < q; ++i) {
            require(gt[i].size() == 10 && gt[i].count(query_ids[i]),
                    "truth/query self-match mismatch");
            for (auto id : gt[i])
                require(id < n, "truth label out of bounds");
        }
        // Validate the target Fresh index without querying or rebuilding it.
        hnswlib::L2Space target_space(target_dim);
        {
            Hnsw identity_check(&target_space, argv[6]);
            require(identity_check.M_ == 16 && identity_check.maxM0_ == 32 &&
                        identity_check.ef_construction_ == 200,
                    "target index configuration mismatch");
            inspect(identity_check, target.base, true);
        }
        hnswlib::L2Space source_space(old_dim);
        Hnsw source(&source_space, argv[5]);
        require(source.M_ == 16 && source.maxM0_ == 32 && source.ef_construction_ == 200,
                "source index configuration mismatch");
        const auto before = inspect(source, old_vectors.base, true);
        std::cerr << "[preflight] all source/target row labels and vectors verified; cached truth "
                     "loaded\n";
        const auto start = Clock::now();
        auto direct = copyDirect(source, target.base);
        const double construction_s = std::chrono::duration<double>(Clock::now() - start).count();
        // Warm up all queries before timing searchKnn calls.
        const auto measured = evaluate(direct.graph.get(), queries, gt);
        compareStructures(source, *direct.graph);
        const auto copied = inspect(*direct.graph, target.base, true);
        const auto after = inspect(source, old_vectors.base, true);
        require(before.hash == copied.hash && before.hash == after.hash &&
                    before.reachable == copied.reachable,
                "structure/source preservation failed");
        std::cout << std::setprecision(17) << "RESULT {\"n\":" << n << ",\"query_count\":" << q
                  << ",\"warmup_queries\":" << q << ",\"source_dimension\":" << old_dim
                  << ",\"target_dimension\":" << target_dim
                  << ",\"construction_time_s\":" << construction_s
                  << ",\"recall_at_10_percent\":" << measured.recall * 100
                  << ",\"query_time_ms\":" << measured.milliseconds
                  << ",\"source_structure_hash\":\"" << before.hash
                  << "\",\"direct_structure_hash\":\"" << copied.hash
                  << "\",\"source_after_structure_hash\":\"" << after.hash
                  << "\",\"structure_hash_scheme\":\"fnv1a64_le_metadata_and_all_ordered_layers_no_"
                     "vectors\""
                  << ",\"all_structure_equal\":true,\"source_vectors_unchanged\":true"
                  << ",\"target_vectors_equal\":true,\"row_labels_verified\":true"
                  << ",\"source_reachable\":" << before.reachable
                  << ",\"direct_reachable\":" << copied.reachable
                  << ",\"all_layer_edge_count\":" << copied.edges
                  << ",\"all_layer_list_count\":" << copied.lists << ",\"query_ids\":[";
        for (size_t i = 0; i < q; ++i)
            std::cout << (i ? "," : "") << query_ids[i];
        std::cout << "],\"ground_truth\":[";
        for (size_t i = 0; i < q; ++i) {
            std::vector<size_t> labels(gt[i].begin(), gt[i].end());
            std::sort(labels.begin(), labels.end());
            std::cout << (i ? "," : "") << '[';
            for (size_t j = 0; j < labels.size(); ++j)
                std::cout << (j ? "," : "") << labels[j];
            std::cout << ']';
        }
        std::cout << "]}" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
