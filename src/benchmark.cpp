#include <cstdlib>
#include "query_ids.h"
#ifdef MATRIX_OURS
#include "matrix/starr.h"
#else
#include "starr.h"
#endif
#include <dgm/dgm.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <unistd.h>

using Graph = hnswlib::HierarchicalNSW<float>;
using Id = hnswlib::tableint;
using Rows = std::vector<std::vector<float>>;
using Clock = std::chrono::steady_clock;
static void check(bool b, const std::string &s) {
    if (!b)
        throw std::runtime_error(s);
}
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
static void mix(uint64_t &h, uint64_t v) {
    for (int j = 0; j < 8; ++j) {
        h = (h ^ (v & 255)) * 1099511628211ULL;
        v >>= 8;
    }
}
static void bytes(uint64_t &h, const void *p, size_t n) {
    auto a = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; ++i)
        h = (h ^ a[i]) * 1099511628211ULL;
}
static Rows readRows(const std::string &path, size_t n, size_t d) {
    check(std::filesystem::file_size(path) == n * d * 4, "vector file size mismatch");
    std::ifstream f(path, std::ios::binary);
    Rows v(n, std::vector<float>(d));
    for (auto &x : v) {
        check(bool(f.read(reinterpret_cast<char *>(x.data()), d * 4)), "short vector file");
        for (float y : x)
            check(std::isfinite(y), "nonfinite vector");
    }
    return v;
}
static std::vector<Id> neighbors(Graph &g, Id id, int h) {
    auto p = g.get_linklist_at_level(id, h);
    auto a = reinterpret_cast<Id *>(p + 1);
    return {a, a + g.getListCount(p)};
}
static void writeNeighbors(Graph &g, Id id, int h, const std::vector<Id> &ids) {
    check(ids.size() <= (h ? g.maxM_ : g.maxM0_), "write capacity");
    auto p = g.get_linklist_at_level(id, h);
    std::copy(ids.begin(), ids.end(), reinterpret_cast<Id *>(p + 1));
    g.setListCount(p, ids.size());
}
struct Layer {
    size_t nodes = 0, edges = 0, max_degree = 0;
};
struct Audit {
    uint64_t structure = 14695981039346656037ULL, vectors = 14695981039346656037ULL;
    size_t reachable = 0;
    std::vector<Layer> layers;
};
static Audit inspect(Graph &g, const Rows *expected = nullptr, bool row_labels = true) {
    const size_t n = g.cur_element_count;
    check(n > 0 && g.num_deleted_ == 0 && g.enterpoint_node_ < n, "index metadata");
    Audit a;
    a.layers.resize(g.maxlevel_ + 1);
    for (uint64_t v : {uint64_t(n), uint64_t(g.max_elements_), uint64_t(g.M_), uint64_t(g.maxM_),
                       uint64_t(g.maxM0_), uint64_t(g.ef_construction_), uint64_t(g.maxlevel_),
                       uint64_t(g.enterpoint_node_)})
        mix(a.structure, v);
    if (expected)
        check(expected->size() == n, "audit expected count");
    for (Id id = 0; id < n; ++id) {
        auto label = g.getExternalLabel(id);
        if (row_labels)
            check(label == id, "object row/label mismatch");
        check(g.label_lookup_.at(label) == id, "label lookup mismatch");
        check(g.element_levels_[id] >= 0 && g.element_levels_[id] <= g.maxlevel_, "invalid level");
        if (expected)
            check((*expected)[id].size() * 4 == g.data_size_ &&
                      std::memcmp(g.getDataByInternalId(id), (*expected)[id].data(),
                                  g.data_size_) == 0,
                  "target/source vectors mismatch");
        bytes(a.vectors, g.getDataByInternalId(id), g.data_size_);
        mix(a.structure, id);
        mix(a.structure, label);
        mix(a.structure, g.element_levels_[id]);
        for (int h = 0; h <= g.element_levels_[id]; ++h) {
            auto ids = neighbors(g, id, h);
            check(ids.size() <= (h ? g.maxM_ : g.maxM0_), "degree overflow");
            auto &l = a.layers[h];
            ++l.nodes;
            l.edges += ids.size();
            l.max_degree = std::max(l.max_degree, ids.size());
            mix(a.structure, h);
            mix(a.structure, ids.size());
            for (size_t j = 0; j < ids.size(); ++j) {
                Id v = ids[j];
                check(v < n && v != id && g.element_levels_[v] >= h,
                      "invalid/self/cross-layer edge");
                check(std::find(ids.begin(), ids.begin() + j, v) == ids.begin() + j,
                      "duplicate edge");
                mix(a.structure, v);
            }
        }
    }
    std::vector<unsigned char> seen(n);
    std::vector<Id> q{g.enterpoint_node_};
    seen[q[0]] = 1;
    for (size_t j = 0; j < q.size(); ++j)
        for (Id v : neighbors(g, q[j], 0))
            if (!seen[v]) {
                seen[v] = 1;
                q.push_back(v);
            }
    a.reachable = q.size();
    return a;
}
static void unchanged(const Audit &a, const Audit &b) {
    check(a.structure == b.structure && a.vectors == b.vectors, "source was mutated");
}
static void sameMetadata(Graph &a, Graph &b, bool same_ef) {
    check(a.cur_element_count == b.cur_element_count && a.max_elements_ == b.max_elements_ &&
              a.M_ == b.M_ && a.maxM_ == b.maxM_ && a.maxM0_ == b.maxM0_ &&
              a.maxlevel_ == b.maxlevel_ && a.enterpoint_node_ == b.enterpoint_node_,
          "migration metadata mismatch");
    if (same_ef)
        check(a.ef_construction_ == b.ef_construction_, "DGM efConstruction metadata mismatch");
    for (Id id = 0; id < a.cur_element_count; ++id)
        check(a.getExternalLabel(id) == b.getExternalLabel(id) &&
                  a.element_levels_[id] == b.element_levels_[id],
              "label/level mismatch");
}
struct Owned {
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<Graph> graph;
};
static Owned allocateTarget(Graph &src, const Rows &target) {
    Owned o;
    o.space = std::make_unique<hnswlib::L2Space>(target[0].size());
    o.graph =
        std::make_unique<Graph>(o.space.get(), src.max_elements_, src.M_, src.ef_construction_);
    auto &g = *o.graph;
    check(g.maxM_ == src.maxM_ && g.maxM0_ == src.maxM0_, "nonstandard source capacity");
    for (Id id = 0; id < src.cur_element_count; ++id) {
        std::memset(g.data_level0_memory_ + id * g.size_data_per_element_, 0,
                    g.size_data_per_element_);
        std::memcpy(g.getDataByInternalId(id), target[id].data(), g.data_size_);
        auto label = src.getExternalLabel(id);
        g.setExternalLabel(id, label);
        g.label_lookup_[label] = id;
        int h = src.element_levels_[id];
        g.element_levels_[id] = h;
        g.linkLists_[id] = nullptr;
        if (h) {
            g.linkLists_[id] =
                static_cast<char *>(std::calloc(1, h * g.size_links_per_element_ + 1));
            if (!g.linkLists_[id])
                throw std::bad_alloc();
        }
        ++g.cur_element_count;
    }
    g.maxlevel_ = src.maxlevel_;
    g.enterpoint_node_ = src.enterpoint_node_;
    g.mult_ = src.mult_;
    g.revSize_ = src.revSize_;
    return o;
}
// DGM processes each layer with local IDs; HNSW storage keeps global object IDs.
struct Mapping {
    std::vector<Id> local_to_global, global_to_local;
};
static Mapping mapping(Graph &src, int h) {
    Mapping m;
    m.global_to_local.assign(src.cur_element_count, std::numeric_limits<Id>::max());
    for (Id id = 0; id < src.cur_element_count; ++id)
        if (src.element_levels_[id] >= h) {
            m.global_to_local[id] = m.local_to_global.size();
            m.local_to_global.push_back(id);
        }
    return m;
}
static dgm::Graph extract(Graph &src, int h, const Mapping &m) {
    std::vector<dgm::Graph::AdjacencyList> rows(m.local_to_global.size());
    for (Id local = 0; local < rows.size(); ++local)
        for (Id global : neighbors(src, m.local_to_global[local], h)) {
            check(global < m.global_to_local.size() &&
                      m.global_to_local[global] != std::numeric_limits<Id>::max(),
                  "cross-layer input");
            rows[local].push_back(m.global_to_local[global]);
        }
    return dgm::Graph(std::move(rows));
}
static void writeLayer(Graph &target, int h, const Mapping &m, const dgm::Graph &output) {
    check(output.size() == m.local_to_global.size(), "DGM output layer size");
    for (Id id = 0; id < output.size(); ++id) {
        std::vector<Id> global;
        for (Id x : output.neighbors(id)) {
            check(x < m.local_to_global.size(), "DGM invalid local output");
            global.push_back(m.local_to_global[x]);
        }
        writeNeighbors(target, m.local_to_global[id], h, global);
    }
}
struct Migration {
    Owned target;
    double preparation_s = 0, core_s = 0, writeback_s = 0;
    dgm::MigrationStats work;
    std::vector<size_t> layer_samples;
};
static Migration migrateDgm(Graph &source, const Rows &target, bool local) {
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
            c.shortlist_size = 400;
            c.diversity_factor = 1.1f;
            c.calibration_sample_size = 4096;
            c.random_seed = 20250311;
            c.thread_count = 1;
            changed = dgm::RunDgmLocal(original, space, c);
            out.layer_samples[h] = std::min(size_t(4096), m.local_to_global.size());
        } else {
            dgm::SearchConfig c;
            c.max_degree = h ? source.maxM_ : source.maxM0_;
            c.beam_width = 256;
            c.max_hops = 4;
            c.diversity_factor = 1.1f;
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
static std::vector<size_t> sample(size_t n, size_t count) {
    check(count <= n, "too many queries");
    std::vector<size_t> all(n), q(count);
    std::iota(all.begin(), all.end(), 0);
    std::mt19937 gen(42);
    std::sample(all.begin(), all.end(), q.begin(), count, gen);
    return q;
}
using Truth = std::vector<std::vector<Id>>;
static Truth exact(const Rows &v, const std::vector<size_t> &q, unsigned workers) {
    // Indexed query objects remain eligible neighbors, including the query itself.
    Truth gt(q.size());
    std::atomic<size_t> next{0};
    std::vector<std::future<void>> jobs;
    for (unsigned w = 0; w < workers; ++w)
        jobs.push_back(std::async(std::launch::async, [&] {
            for (size_t j = next++; j < q.size(); j = next++) {
                std::priority_queue<std::pair<float, Id>> best;
                const auto &query = v[q[j]];
                for (Id id = 0; id < v.size(); ++id) {
                    float d = 0;
                    for (size_t k = 0; k < query.size(); ++k) {
                        float t = query[k] - v[id][k];
                        d += t * t;
                    }
                    std::pair<float, Id> item{d, id};
                    if (best.size() < 10 || item < best.top()) {
                        best.push(item);
                        if (best.size() > 10)
                            best.pop();
                    }
                }
                while (!best.empty()) {
                    gt[j].push_back(best.top().second);
                    best.pop();
                }
                std::sort(gt[j].begin(), gt[j].end());
                if ((j + 1) % 100 == 0)
                    std::cerr << "[truth] " << j + 1 << '/' << q.size() << std::endl;
            }
        }));
    for (auto &j : jobs)
        j.get();
    return gt;
}
static void writeTruth(const std::string &path, const Truth &gt, const std::vector<size_t> &q,
                       size_t n, size_t d) {
    check(!std::filesystem::exists(path), "refusing truth overwrite");
    std::ofstream f(path, std::ios::binary);
    uint64_t header[] = {0x44474d475431ULL, n, d, q.size(), 10};
    f.write(reinterpret_cast<char *>(header), sizeof(header));
    for (size_t x : q) {
        uint64_t y = x;
        f.write(reinterpret_cast<char *>(&y), 8);
    }
    for (auto &row : gt)
        f.write(reinterpret_cast<const char *>(row.data()), 40);
    check(bool(f), "truth write failed");
}
static Truth readTruth(const std::string &path, const std::vector<size_t> &q, size_t n, size_t d) {
    std::ifstream f(path, std::ios::binary);
    uint64_t h[5];
    check(bool(f.read(reinterpret_cast<char *>(h), 40)) && h[0] == 0x44474d475431ULL && h[1] == n &&
              h[2] == d && h[3] == q.size() && h[4] == 10,
          "GT header mismatch");
    for (size_t x : q) {
        uint64_t y;
        check(bool(f.read(reinterpret_cast<char *>(&y), 8)) && x == y, "GT query ID mismatch");
    }
    Truth gt(q.size(), std::vector<Id>(10));
    for (auto &row : gt) {
        check(bool(f.read(reinterpret_cast<char *>(row.data()), 40)), "GT truncated");
        std::set<Id> u(row.begin(), row.end());
        check(u.size() == 10 && *u.rbegin() < n, "GT invalid labels");
    }
    check(f.peek() == EOF, "GT trailing bytes");
    return gt;
}
struct Evaluation {
    double recall, ms, total_s;
    int64_t total_ns;
    size_t hits;
};
static Evaluation evaluate(Graph &g, const Rows &v, const std::vector<size_t> &q, const Truth &gt) {
    // Warm up before timing; count recall hits only after the timed query batch.
    g.setEf(200);
    auto search = [&](size_t id) {
        auto heap = g.searchKnn(v[q[id]].data(), 10);
        std::vector<Id> out;
        while (!heap.empty()) {
            out.push_back(heap.top().second);
            heap.pop();
        }
        return out;
    };
    for (size_t i = 0; i < q.size(); ++i)
        search(i);
    std::vector<std::vector<Id>> results;
    results.reserve(q.size());
    auto t = Clock::now();
    for (size_t i = 0; i < q.size(); ++i)
        results.push_back(search(i));
    auto duration = Clock::now() - t;
    double total_s = std::chrono::duration<double>(duration).count();
    int64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    double ms = total_s * 1000 / q.size();
    size_t hits = 0;
    for (size_t i = 0; i < q.size(); ++i) {
        std::set<Id> u(results[i].begin(), results[i].end());
        check(u.size() == results[i].size() && u.size() <= 10, "duplicate query result");
        for (Id x : u)
            hits += std::find(gt[i].begin(), gt[i].end(), x) != gt[i].end();
    }
    return {100. * hits / (10 * q.size()), ms, total_s, total_ns, hits};
}
template <class T> static void array(std::ostream &o, const std::vector<T> &a) {
    o << '[';
    for (size_t i = 0; i < a.size(); ++i) {
        if (i)
            o << ',';
        o << a[i];
    }
    o << ']';
}
static void auditJson(std::ostream &o, const Audit &a) {
    o << "{\"structure_hash\":\"" << a.structure << "\",\"vector_hash\":\"" << a.vectors
      << "\",\"reachable\":" << a.reachable << ",\"layers\":[";
    for (size_t h = 0; h < a.layers.size(); ++h) {
        if (h)
            o << ',';
        auto l = a.layers[h];
        o << "{\"layer\":" << h << ",\"nodes\":" << l.nodes << ",\"edges\":" << l.edges
          << ",\"avg_degree\":" << double(l.edges) / l.nodes << ",\"max_degree\":" << l.max_degree
          << '}';
    }
    o << "]}";
}
static void run(const std::string &method, const std::string &oldfile, size_t olddim,
                const std::string &newfile, size_t newdim, const std::string &indexfile,
                const std::string &truthfile, size_t n) {
    auto target = readRows(newfile, n, newdim);
    auto q = actualQueryIds(n);
    auto gt = readTruth(truthfile, q, n, newdim);
    hnswlib::L2Space space(method == "Fresh" ? newdim : olddim);
    Graph original(&space, indexfile);
    check(original.cur_element_count == n && original.M_ == 16 && original.maxM_ == 16 &&
              original.maxM0_ == 32 && original.ef_construction_ == 200,
          "HNSW experiment configuration mismatch");
    Audit before;
    if (method == "Fresh")
        before = inspect(original, &target);
    else {
        auto old = readRows(oldfile, n, olddim);
        before = inspect(original, &old);
    }
    std::cerr << "[preflight] full source vector/label/layer audit passed; method=" << method
              << std::endl;
    double migration = 0, prep = 0, core = 0, writeback = 0;
    Owned ours;
    Migration dgm;
    StableMigrationStats stats;
    Graph *result = &original;
    if (method != "Fresh") {
        auto t = Clock::now();
        if (method == "Ours") {
            StableMigrationOptions options;
            options.z_tolerance = 2;
            options.search_ef = 16;
            options.fine_repair_passes = 0;
            if (const char *x = std::getenv("STARR_TAU"))
                options.z_tolerance = std::stof(x);
            if (const char *x = std::getenv("STARR_B"))
                options.search_ef = std::stoull(x);
            hnswlib::SpaceInterface<float> *s = nullptr;
            auto g = migrateHnswIndexStable(&original, target, &s, options, &stats);
            ours.space.reset(s);
            ours.graph.reset(g);
            result = g;
            result->setEf(200);
        } else {
            check(method == "DGM-Local" || method == "DGM-Search", "invalid method");
            dgm = migrateDgm(original, target, method == "DGM-Local");
            result = dgm.target.graph.get();
            prep = dgm.preparation_s;
            core = dgm.core_s;
            writeback = dgm.writeback_s;
        }
        migration = elapsed(t);
        std::cerr << "[migration] completed seconds=" << migration << std::endl;
        sameMetadata(original, *result, method != "Ours");
    }
    auto a = inspect(*result, &target);
    auto after = inspect(original);
    unchanged(before, after);
    if (method == "Ours")
        check(a.reachable == n, "Ours full reachability invariant failed");
    auto ev = evaluate(*result, target, q, gt);
#ifdef MATRIX_OURS
    std::string shared_edges = "null";
    int batch_gate = int(stats.batched_stable_copy);
#else
    std::string shared_edges = std::to_string(stats.shared_edges);
    int batch_gate = int(stats.batch_gate_active);
#endif
    std::cout << std::setprecision(15) << "RESULT {\"method\":\"" << method << "\",\"n\":" << n
              << ",\"source_dim\":" << olddim << ",\"target_dim\":" << newdim
              << ",\"migration_s\":" << migration << ",\"recall_at_10_percent\":" << ev.recall
              << ",\"query_time_ms\":" << ev.ms << ",\"query_total_s\":" << ev.total_s
              << ",\"query_total_ns\":" << ev.total_ns << ",\"total_hits\":" << ev.hits
              << ",\"query_ids_loaded\":" << (std::getenv("QUERY_IDS_BINARY") ? "true" : "false")
              << ",\"query_count\":" << q.size()
              << ",\"query_ef\":200,\"query_warmup\":1000,\"target_metadata_efConstruction\":"
              << result->ef_construction_
              << ",\"source_unchanged\":true,\"metadata_valid\":true,\"target_vectors_valid\":true,"
                 "\"structure_valid\":true,\"prepare_s\":"
              << prep << ",\"dgm_core_including_MPS_s\":" << core
              << ",\"writeback_s\":" << writeback
              << ",\"nodes_processed\":" << dgm.work.nodes_processed
              << ",\"candidates_screened\":" << dgm.work.candidates_screened
              << ",\"adjacency_expansions\":" << dgm.work.adjacency_expansions
              << ",\"source_distance_evaluations\":" << dgm.work.source_distance_evaluations
              << ",\"pruning_distance_evaluations\":" << dgm.work.pruning_distance_evaluations
              << ",\"ours_repaired\":" << stats.repaired
              << ",\"ours_shared_edges\":" << shared_edges << ",\"ours_bridges\":" << stats.bridges
              << ",\"batch_gate_active\":" << batch_gate
              << ",\"batch_executed\":" << int(stats.batched_stable_copy)
              << ",\"MPS_layer_samples\":";
    array(std::cout, dgm.layer_samples);
    std::cout << ",\"source_before\":";
    auditJson(std::cout, before);
    std::cout << ",\"source_after\":";
    auditJson(std::cout, after);
    std::cout << ",\"output_audit\":";
    auditJson(std::cout, a);
    std::cout << "}" << std::endl;
}
static int selfTest();
static void makeFixture(const std::string &path);
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "self-test")
            return selfTest();
        check(argc >= 2, "mode required");
        std::string mode = argv[1];
        if (mode == "fixture") {
            check(argc == 3, "fixture DIRECTORY");
            makeFixture(argv[2]);
        } else if (mode == "truth") {
            check(argc == 6, "truth VECTORS DIM N OUTPUT");
            size_t d = std::stoull(argv[3]), n = std::stoull(argv[4]);
            auto v = readRows(argv[2], n, d);
            auto q = actualQueryIds(n);
            auto t = Clock::now();
            auto gt = exact(v, q, 8);
            double secs = elapsed(t);
            writeTruth(argv[5], gt, q, n, d);
            std::cout << "RESULT {\"n\":" << n << ",\"dimension\":" << d
                      << ",\"query_count\":1000,\"k\":10,\"threads\":8,\"ground_truth_seconds\":"
                      << secs << ",\"query_ids\":";
            array(std::cout, q);
            std::cout << "}" << std::endl;
        } else if (mode == "run") {
            check(argc == 10, "run METHOD OLD OLD_DIM TARGET TARGET_DIM INDEX TRUTH N");
            run(argv[2], argv[3], std::stoull(argv[4]), argv[5], std::stoull(argv[6]), argv[7],
                argv[8], std::stoull(argv[9]));
        } else
            throw std::runtime_error("unknown mode");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

static int selfTest() {
    const size_t n = 12;
    Rows old(n, std::vector<float>(5)), target(n, std::vector<float>(7));
    hnswlib::L2Space space(5);
    Graph source(&space, n, 2, 200);
    for (Id id = 0; id < n; ++id) {
        for (size_t j = 0; j < 5; ++j)
            old[id][j] = float(id * 3 + j) / 17;
        for (size_t j = 0; j < 7; ++j)
            target[id][j] = float((id * 7 + j * 3) % 19) / 19;
        std::memset(source.data_level0_memory_ + id * source.size_data_per_element_, 0,
                    source.size_data_per_element_);
        std::memcpy(source.getDataByInternalId(id), old[id].data(), 20);
        size_t label = 101 + id * 7;
        source.setExternalLabel(id, label);
        source.label_lookup_[label] = id;
        int h = id == 4 ? 2 : ((id == 1 || id == 9) ? 1 : 0);
        source.element_levels_[id] = h;
        source.linkLists_[id] =
            h ? static_cast<char *>(std::calloc(h * source.size_links_per_element_ + 1, 1))
              : nullptr;
        ++source.cur_element_count;
        writeNeighbors(source, id, 0, {Id((id + 3) % n), Id((id + 1) % n)});
    }
    source.maxlevel_ = 2;
    source.enterpoint_node_ = 4;
    writeNeighbors(source, 1, 1, {9, 4});
    writeNeighbors(source, 4, 1, {1, 9});
    writeNeighbors(source, 9, 1, {4, 1});
    writeNeighbors(source, 4, 2, {});
    auto before = inspect(source, &old, false);
    auto copy = allocateTarget(source, target);
    for (int h = 2; h >= 0; --h) {
        auto m = mapping(source, h);
        auto layer = extract(source, h, m);
        writeLayer(*copy.graph, h, m, layer);
        for (Id id : m.local_to_global)
            check(neighbors(source, id, h) == neighbors(*copy.graph, id, h),
                  "identity adapter changed ordered layer");
    }
    sameMetadata(source, *copy.graph, true);
    auto cp = inspect(*copy.graph, &target, false);
    check(cp.structure == before.structure, "identity topology hash differs");
    for (bool local : {true, false}) {
        auto migrated = migrateDgm(source, target, local);
        sameMetadata(source, *migrated.target.graph, true);
        inspect(*migrated.target.graph, &target, false);
        unchanged(before, inspect(source, &old, false));
        check(migrated.work.nodes_processed == 16, "wrong occupied-layer node count");
        for (int h = 2; h >= 0; --h) {
            auto m = mapping(source, h);
            auto input = extract(source, h, m);
            auto orig = input.csr_neighbors();
            std::vector<float> packed;
            for (Id id : m.local_to_global)
                packed.insert(packed.end(), target[id].begin(), target[id].end());
            hnswlib::L2Space l2(7);
            dgm::EmbeddingSpace es(packed.data(), m.local_to_global.size(), 7,
                                   dgm::Metric::kSquaredL2, 1, l2.get_dist_func(),
                                   l2.get_dist_func_param());
            for (Id i = 0; i < m.local_to_global.size(); ++i)
                for (Id j = 0; j < m.local_to_global.size(); ++j)
                    check(es.distance(i, j) == l2.get_dist_func()(packed.data() + i * 7,
                                                                  packed.data() + j * 7,
                                                                  l2.get_dist_func_param()),
                          "distance callback mismatch");
            dgm::MigrationResult ref;
            if (local) {
                dgm::LocalConfig c;
                c.max_degree = h ? 2 : 4;
                c.shortlist_size = 400;
                c.thread_count = 1;
                ref = dgm::RunDgmLocal(input, es, c);
            } else {
                dgm::SearchConfig c;
                c.max_degree = h ? 2 : 4;
                c.entry_point = m.global_to_local[4];
                c.thread_count = 1;
                ref = dgm::RunDgmSearch(input, es, c);
            }
            check(input.csr_neighbors() == orig, "DGM changed input graph");
            for (Id i = 0; i < m.local_to_global.size(); ++i) {
                std::vector<Id> want;
                for (Id x : ref.graph.neighbors(i))
                    want.push_back(m.local_to_global[x]);
                check(want == neighbors(*migrated.target.graph, m.local_to_global[i], h),
                      "layer local/global output mismatch");
            }
        }
        migrated.target.graph->setEf(200);
        auto heap = migrated.target.graph->searchKnn(target[3].data(), 10);
        check(!heap.empty(), "HNSW cannot query migrated storage");
        while (!heap.empty()) {
            check(source.label_lookup_.count(heap.top().second), "returned invalid external label");
            heap.pop();
        }
    }
    // Ours accepts arbitrary external labels and preserves layer membership too.
    StableMigrationOptions opts;
    opts.search_ef = 16;
    hnswlib::SpaceInterface<float> *sp = nullptr;
    Owned own;
    own.graph.reset(migrateHnswIndexStable(&source, target, &sp, opts));
    own.space.reset(sp);
    sameMetadata(source, *own.graph, false);
    check(inspect(*own.graph, &target, false).reachable == n, "Ours reachability");
    unchanged(before, inspect(source, &old, false));
    // Exact ground-truth tie rule agrees with exhaustive sort, and no self filtering.
    auto q = sample(n, n);
    auto gt = exact(target, q, 1);
    for (size_t i = 0; i < q.size(); ++i) {
        std::vector<std::pair<float, Id>> distances;
        for (Id id = 0; id < n; ++id) {
            float d = 0;
            for (size_t j = 0; j < 7; ++j) {
                float t = target[q[i]][j] - target[id][j];
                d += t * t;
            }
            distances.emplace_back(d, id);
        }
        std::sort(distances.begin(), distances.end());
        std::vector<Id> wanted;
        for (size_t j = 0; j < 10; ++j)
            wanted.push_back(distances[j].second);
        std::sort(wanted.begin(), wanted.end());
        check(wanted == gt[i], "exact truth mismatch");
    }
    auto bad = neighbors(*copy.graph, 1, 1);
    writeNeighbors(*copy.graph, 1, 1, {0});
    bool caught = false;
    try {
        inspect(*copy.graph, &target, false);
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught, "illegal cross-layer edge not caught");
    writeNeighbors(*copy.graph, 1, 1, bad);
    writeNeighbors(*copy.graph, 0, 0, {1, 1});
    caught = false;
    try {
        inspect(*copy.graph, &target, false);
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught, "duplicate edge not caught");
    // Tail-bit dimensions and small calibration populations are exercised above.
    std::cout << "PASS adapter: dimensions5->7, arbitrary labels, noncontiguous upper subsets, "
                 "singleton top, entry4, layer mapping, all-layer ordered roundtrip, source "
                 "immutability, callback squaredL2, reference-core equivalence, capacity/edge "
                 "rejection, HNSW query, Ours full, exactGT oracle\n";
    return 0;
}

// Only a small synthetic fixture for end-to-end protocol tests; never formal Fresh.
static void makeFixture(const std::string &path) {
    check(!std::filesystem::exists(path), "refuse fixture overwrite");
    std::filesystem::create_directory(path);
    const size_t n = 1500;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> uniform(-1, 1);
    for (bool target : {false, true}) {
        size_t dim = target ? 7 : 5;
        Rows v(n, std::vector<float>(dim));
        for (auto &row : v)
            for (float &x : row)
                x = uniform(rng);
        std::string name = path + (target ? "/target" : "/source");
        std::ofstream file(name + ".bin", std::ios::binary);
        for (auto &row : v)
            file.write(reinterpret_cast<const char *>(row.data()), dim * 4);
        file.close();
        hnswlib::L2Space space(dim);
        Graph g(&space, n, 16, 200, 100);
        for (size_t i = 0; i < n; ++i)
            g.addPoint(v[i].data(), i);
        g.saveIndex(name + ".hnsw");
    }
    std::cout << "synthetic fixture n=1500 only; no formal-data Fresh construction\n";
}
