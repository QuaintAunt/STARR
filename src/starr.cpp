#include "starr.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace {
using Index = hnswlib::HierarchicalNSW<float>;
using Id = hnswlib::tableint;
using Heap = std::priority_queue<std::pair<float, Id>, std::vector<std::pair<float, Id>>,
                                 Index::CompareByFirst>;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static std::vector<Id> adjacent(Index &graph, Id id, int level) {
    auto *list = graph.get_linklist_at_level(id, level);
    auto *data = reinterpret_cast<Id *>(list + 1);
    return {data, data + graph.getListCount(list)};
}

static float distance(Index &graph, Id left, Id right) {
    return graph.fstdistfunc_(graph.getDataByInternalId(left), graph.getDataByInternalId(right),
                              graph.dist_func_param_);
}

static std::vector<float> standardized(std::vector<float> distances) {
    if (distances.empty())
        return distances;
    const float mean = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
    float variance = 0;
    for (float value : distances)
        variance += (value - mean) * (value - mean);
    const float sd = std::sqrt(variance / distances.size());
    // Constant-distance lists use zero scores, including singleton lists.
    for (float &value : distances)
        value = sd > 0 ? (value - mean) / sd : 0;
    return distances;
}

// Collect reverse-edge proposals before pruning each recipient list once.
static void batchReverse(Index &graph, const std::vector<std::unordered_set<uint64_t>> &rejected) {
    const size_t n = graph.cur_element_count;
    std::vector<std::vector<Id>> incoming(n);
    for (int level = 0; level <= graph.maxlevel_; ++level) {
        for (Id id = 0; id < n; ++id) {
            if (graph.element_levels_[id] < level)
                continue;
            for (Id other : adjacent(graph, id, level)) {
                if (rejected[level].count((uint64_t(other) << 32) | id))
                    continue;
                auto *list = graph.get_linklist_at_level(other, level);
                const auto *data = reinterpret_cast<const Id *>(list + 1);
                if (std::find(data, data + graph.getListCount(list), id) ==
                    data + graph.getListCount(list))
                    incoming[other].push_back(id);
            }
        }
        for (Id id = 0; id < n; ++id) {
            if (incoming[id].empty())
                continue;
            auto selected = adjacent(graph, id, level);
            selected.insert(selected.end(), incoming[id].begin(), incoming[id].end());
            incoming[id].clear();
            const size_t capacity = level ? graph.maxM_ : graph.maxM0_;
            if (selected.size() > capacity) {
                Heap candidates;
                for (Id other : selected)
                    candidates.emplace(distance(graph, id, other), other);
                graph.getNeighborsByHeuristic2(candidates, capacity);
                selected.clear();
                while (!candidates.empty()) {
                    selected.push_back(candidates.top().second);
                    candidates.pop();
                }
            }
            auto *list = graph.get_linklist_at_level(id, level);
            auto *data = reinterpret_cast<Id *>(list + 1);
            std::copy(selected.begin(), selected.end(), data);
            graph.setListCount(list, selected.size());
        }
    }
}

static size_t ensureReachable(Index &graph) {
    const Id unseen = std::numeric_limits<Id>::max();
    // Parent edges preserve a directed spanning tree while bridges are added.
    std::vector<Id> parent(graph.cur_element_count, unseen), queue{graph.enterpoint_node_};
    parent[queue[0]] = queue[0];
    size_t cursor = 0, bridges = 0;
    auto flood = [&]() {
        while (cursor < queue.size()) {
            const Id id = queue[cursor++];
            for (Id other : adjacent(graph, id, 0))
                if (parent[other] == unseen) {
                    parent[other] = id;
                    queue.push_back(other);
                }
        }
    };
    flood();
    for (Id root = 0; root < graph.cur_element_count; ++root) {
        if (parent[root] != unseen)
            continue;
        auto candidates =
            graph.searchBaseLayer(graph.enterpoint_node_, graph.getDataByInternalId(root), 0);
        std::vector<std::pair<float, Id>> sorted;
        while (!candidates.empty()) {
            sorted.push_back(candidates.top());
            candidates.pop();
        }
        std::sort(sorted.begin(), sorted.end());
        auto attach = [&](Id anchor) {
            auto *list = graph.get_linklist0(anchor);
            auto *data = reinterpret_cast<Id *>(list + 1);
            const size_t count = graph.getListCount(list);
            size_t slot = count;
            if (count >= graph.maxM0_) {
                // Farthest non-tree edge is replaceable without sacrificing reachability.
                float farthest = -1;
                for (size_t j = 0; j < count; ++j)
                    if (parent[data[j]] != anchor) {
                        const float d = distance(graph, anchor, data[j]);
                        if (d > farthest) {
                            farthest = d;
                            slot = j;
                        }
                    }
                if (slot == count)
                    return false;
            } else
                graph.setListCount(list, count + 1);
            data[slot] = root;
            parent[root] = anchor;
            queue.push_back(root);
            ++bridges;
            return true;
        };
        bool attached = false;
        for (const auto &candidate : sorted)
            if (attach(candidate.second)) {
                attached = true;
                break;
            }
        // Rare fallback: any reachable leaf has a free or non-tree slot.
        if (!attached) {
            const size_t count = queue.size();
            for (size_t j = 0; j < count && !attached; ++j)
                attached = attach(queue[j]);
        }
        if (!attached)
            throw std::runtime_error("cannot connect an unreachable component");
        flood();
    }
    return bridges;
}

// Search the current target graph; beam width bounds retained candidates, not visits.
static Heap search(Index &graph, Id id, int level, const std::vector<Id> &seeds) {
    auto release = [&](hnswlib::VisitedList *p) {
        graph.visited_list_pool_->releaseVisitedList(p);
    };
    std::unique_ptr<hnswlib::VisitedList, decltype(release)> visited(
        graph.visited_list_pool_->getFreeVisitedList(), release);
    Heap best, pending;
    auto insert = [&](Id other) {
        if (visited->mass[other] == visited->curV)
            return;
        visited->mass[other] = visited->curV;
        const float d = distance(graph, id, other);
        if (best.size() < graph.ef_construction_ || d < best.top().first) {
            pending.emplace(-d, other);
            if (other != id)
                best.emplace(d, other);
            if (best.size() > graph.ef_construction_)
                best.pop();
        }
    };
    for (Id seed : seeds)
        insert(seed);
    while (!pending.empty()) {
        const auto [negative_d, current] = pending.top();
        if (best.size() >= graph.ef_construction_ && -negative_d > best.top().first)
            break;
        pending.pop();
        auto *list = graph.get_linklist_at_level(current, level);
        const auto *data = reinterpret_cast<const Id *>(list + 1);
        const size_t count = graph.getListCount(list);
        for (size_t j = 0; j < count; ++j) {
#ifdef USE_SSE
            if (j + 1 < count) {
                _mm_prefetch(graph.getDataByInternalId(data[j + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(visited->mass + data[j + 1]),
                             _MM_HINT_T0);
            }
#endif
            insert(data[j]);
        }
    }
    return best;
}

static void shareCandidates(Index &graph, Id id, const std::vector<Id> &candidates,
                            std::vector<unsigned char> *enriched, StableMigrationStats &stats) {
    for (Id neighbor : adjacent(graph, id, 0)) {
        auto *list = graph.get_linklist0(neighbor);
        auto *data = reinterpret_cast<Id *>(list + 1);
        const size_t count = graph.getListCount(list);
        if (count >= graph.maxM0_)
            continue;
        Id closest = neighbor;
        float best = std::numeric_limits<float>::infinity();
        for (Id candidate : candidates) {
            if (candidate == neighbor || std::find(data, data + count, candidate) != data + count)
                continue;
            const float d = distance(graph, neighbor, candidate);
            if (d < best) {
                best = d;
                closest = candidate;
            }
        }
        if (closest != neighbor) {
            ++stats.shared_edges;
            data[count] = closest;
            graph.setListCount(list, count + 1);
            if (enriched)
                (*enriched)[neighbor] = 1;
        }
    }
}

static void reconnect(Index &graph, Id id, std::vector<unsigned char> *enriched,
                      const StableMigrationOptions &options, StableMigrationStats &stats) {
    Id entry = graph.enterpoint_node_;
    float nearest = distance(graph, id, entry);
    for (int level = graph.maxlevel_; level > graph.element_levels_[id]; --level) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (Id other : adjacent(graph, entry, level)) {
                const float d = distance(graph, id, other);
                if (d < nearest) {
                    nearest = d;
                    entry = other;
                    changed = true;
                }
            }
        }
    }
    for (int level = graph.element_levels_[id]; level >= 0; --level) {
        // Keep pre-search neighbors available for restoration after diversity pruning.
        auto seeds = adjacent(graph, id, level);
        const size_t surviving = seeds.size();
        seeds.push_back(entry);
        auto candidates = search(graph, id, level, seeds);
        std::vector<Id> reusable;
        if (level == 0) {
            auto copy = candidates;
            while (!copy.empty()) {
                reusable.push_back(copy.top().second);
                copy.pop();
            }
        }
        if (candidates.empty())
            continue;
        entry = graph.mutuallyConnectNewElement(graph.getDataByInternalId(id), id, candidates,
                                                level, true);
        auto *list = graph.get_linklist_at_level(id, level);
        auto *data = reinterpret_cast<Id *>(list + 1);
        size_t count = graph.getListCount(list);
        const size_t capacity = level ? graph.maxM_ : graph.maxM0_;
        for (size_t j = 0; options.restore_survivors && j < surviving && count < capacity; ++j)
            if (std::find(data, data + count, seeds[j]) == data + count)
                data[count++] = seeds[j];
        graph.setListCount(list, count);
        if (level == 0 && options.share_candidates)
            shareCandidates(graph, id, reusable, enriched, stats);
    }
}

// Optional local repair: add shortcuts only where reinsertion/sharing did not act.
static bool growUncovered(Index &graph, Id id, Id preferred) {
    const auto own = adjacent(graph, id, 0);
    if (own.empty() || own.size() >= graph.maxM0_)
        return false;
    Id pivot = preferred;
    if (std::find(own.begin(), own.end(), pivot) == own.end()) {
        pivot = own.front();
        float nearest = distance(graph, id, pivot);
        for (Id other : own) {
            const float d = distance(graph, id, other);
            if (d < nearest) {
                nearest = d;
                pivot = other;
            }
        }
    }
    auto release = [&](hnswlib::VisitedList *p) {
        graph.visited_list_pool_->releaseVisitedList(p);
    };
    std::unique_ptr<hnswlib::VisitedList, decltype(release)> visited(
        graph.visited_list_pool_->getFreeVisitedList(), release);
    visited->mass[id] = visited->curV;
    for (Id other : own)
        visited->mass[other] = visited->curV;
    const size_t spare = graph.maxM0_ - own.size();
    Heap additions;
    auto *pivot_list = graph.get_linklist0(pivot);
    const auto *neighbors = reinterpret_cast<const Id *>(pivot_list + 1);
    const size_t neighbor_count = graph.getListCount(pivot_list);
    for (size_t j = 0; j < neighbor_count; ++j) {
#ifdef USE_SSE
        if (j + 1 < neighbor_count)
            _mm_prefetch(graph.getDataByInternalId(neighbors[j + 1]), _MM_HINT_T0);
#endif
        const Id other = neighbors[j];
        if (visited->mass[other] == visited->curV)
            continue;
        visited->mass[other] = visited->curV;
        const float d = distance(graph, id, other);
        if (additions.size() < spare || d < additions.top().first) {
            additions.emplace(d, other);
            if (additions.size() > spare)
                additions.pop();
        }
    }
    if (additions.empty())
        return false;
    auto *list = graph.get_linklist0(id);
    auto *data = reinterpret_cast<Id *>(list + 1);
    size_t count = own.size();
    while (!additions.empty()) {
        data[count++] = additions.top().second;
        additions.pop();
    }
    graph.setListCount(list, count);
    return true;
}
} // namespace

Index *migrateHnswIndexStable(Index *source, const std::vector<std::vector<float>> &vectors,
                              hnswlib::SpaceInterface<float> **out_space,
                              const StableMigrationOptions &options,
                              StableMigrationStats *out_stats) {
    if (!out_space)
        throw std::invalid_argument("out_space must not be null");
    *out_space = nullptr;
    const size_t n = vectors.size();
    if (!source || !n || n != source->cur_element_count || source->num_deleted_ ||
        vectors.front().empty())
        throw std::invalid_argument("need nonempty aligned vectors and an undeleted L2 source");
    if (!std::isfinite(options.z_tolerance) || options.z_tolerance < 0 ||
        (options.search_ef && options.search_ef < source->M_))
        throw std::invalid_argument("invalid stability tolerance or search ef");
    const size_t dim = vectors.front().size();
    for (const auto &vector : vectors)
        if (vector.size() != dim ||
            !std::all_of(vector.begin(), vector.end(), [](float x) { return std::isfinite(x); }))
            throw std::invalid_argument("invalid vector dimensions or non-finite data");
    StableMigrationStats stats;
    auto start = Clock::now();
    auto space = std::make_unique<hnswlib::L2Space>(dim);
    auto index = std::make_unique<Index>(space.get(), n, source->M_,
                                         options.search_ef ? options.search_ef : source->M_);
    Index &graph = *index;
    // Preserve object identity and layer membership while replacing vector storage.
    for (Id id = 0; id < n; ++id) {
        std::memset(graph.data_level0_memory_ + id * graph.size_data_per_element_, 0,
                    graph.size_data_per_element_);
        std::memcpy(graph.getDataByInternalId(id), vectors[id].data(), graph.data_size_);
        const auto label = source->getExternalLabel(id);
        graph.setExternalLabel(id, label);
        graph.label_lookup_[label] = id;
        const int level = source->element_levels_[id];
        graph.element_levels_[id] = level;
        graph.linkLists_[id] = nullptr;
        if (level) {
            graph.linkLists_[id] =
                static_cast<char *>(std::calloc(1, graph.size_links_per_element_ * level + 1));
            if (!graph.linkLists_[id])
                throw std::bad_alloc();
        }
        ++graph.cur_element_count;
    }
    graph.enterpoint_node_ = source->enterpoint_node_;
    graph.maxlevel_ = source->maxlevel_;
    stats.register_s = elapsed(start);
    start = Clock::now();
    std::vector<unsigned char> unstable(n, 0);
    std::vector<std::vector<uint64_t>> rejected_keys(graph.maxlevel_ + 1);
    std::vector<Id> pivots(options.fine_repair_passes ? n : 0, std::numeric_limits<Id>::max());
    std::vector<size_t> offsets(n + 1);
    for (Id id = 0; id < n; ++id)
        offsets[id + 1] = offsets[id] + source->getListCount(source->get_linklist0(id));
    std::vector<float> before_cache(offsets.back()), after_cache(offsets.back());
    for (Id id = 0; id < n; ++id) {
        for (int level = 0; level <= source->element_levels_[id]; ++level) {
            const auto neighbors = adjacent(*source, id, level);
            std::vector<float> old_distances, new_distances;
            old_distances.reserve(neighbors.size());
            new_distances.reserve(neighbors.size());
            for (size_t j = 0; j < neighbors.size(); ++j) {
#ifdef USE_SSE
                if (j + 1 < neighbors.size()) {
                    _mm_prefetch(source->getDataByInternalId(neighbors[j + 1]), _MM_HINT_T0);
                    _mm_prefetch(graph.getDataByInternalId(neighbors[j + 1]), _MM_HINT_T0);
                }
#endif
                const Id other = neighbors[j];
                float before = 0, after = 0;
                bool reused = false;
                if (level == 0 && other < id) {
                    auto *reverse = source->get_linklist0(other);
                    const auto *data = reinterpret_cast<const Id *>(reverse + 1);
                    const size_t count = source->getListCount(reverse);
                    const auto *found = std::find(data, data + count, id);
                    if (found != data + count) {
                        const size_t position = offsets[other] + (found - data);
                        before = before_cache[position];
                        after = after_cache[position];
                        reused = true;
                    }
                }
                if (!reused) {
                    before = std::sqrt(distance(*source, id, other));
                    after = std::sqrt(distance(graph, id, other));
                }
                if (level == 0) {
                    before_cache[offsets[id] + j] = before;
                    after_cache[offsets[id] + j] = after;
                }
                old_distances.push_back(before);
                new_distances.push_back(after);
            }
            // Compare both distance profiles over the same source neighbor list.
            const auto old_z = standardized(old_distances), new_z = standardized(new_distances);
            double correlation = 0;
            for (size_t j = 0; j < neighbors.size(); ++j)
                correlation += double(old_z[j]) * new_z[j];
            if (!neighbors.empty())
                correlation /= neighbors.size();
            const float tolerance =
                options.z_tolerance *
                (options.correlation_scaling ? std::clamp(correlation, 0.0, 1.0) : 1.0);
            auto *list = graph.get_linklist_at_level(id, level);
            auto *data = reinterpret_cast<Id *>(list + 1);
            size_t count = 0;
            float nearest = std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < neighbors.size(); ++j) {
                ++stats.edges;
                // Screening is one-sided: shorter standardized edges remain reusable.
                if (new_z[j] - old_z[j] <= tolerance) {
                    data[count++] = neighbors[j];
                    if (options.fine_repair_passes && level == 0 && new_distances[j] < nearest) {
                        nearest = new_distances[j];
                        pivots[id] = neighbors[j];
                    }
                } else {
                    ++stats.removed_edges;
                    rejected_keys[level].push_back((uint64_t(id) << 32) | neighbors[j]);
                    unstable[id] = 1;
                }
            }
            graph.setListCount(list, count);
            if (!count && n > 1)
                unstable[id] = 1;
        }
    }
    const size_t unstable_count = std::count(unstable.begin(), unstable.end(), 1);
    // An upper bound on directly changed LEVEL-0 lists, not a Recall guarantee.
    stats.batch_gate_active = unstable_count <= (n - 1) / (source->maxM0_ + 1);
    stats.batched_stable_copy = stats.batch_gate_active && options.batch_stable_reverse;
    if (stats.batched_stable_copy) {
        std::vector<std::unordered_set<uint64_t>> rejected(rejected_keys.size());
        for (size_t level = 0; level < rejected_keys.size(); ++level)
            rejected[level].insert(rejected_keys[level].begin(), rejected_keys[level].end());
        batchReverse(graph, rejected);
    }
    stats.filter_s = elapsed(start);
    start = Clock::now();
    std::vector<Id> repair_order;
    for (Id id = 0; id < n; ++id)
        if (unstable[id])
            repair_order.push_back(id);
    std::vector<unsigned char> enriched(options.fine_repair_passes ? n : 0, 0);
    // Repair higher vertices first so their upper-layer routes guide later searches.
    for (int level = graph.maxlevel_; level >= 0; --level)
        for (Id id : repair_order) {
            if (graph.element_levels_[id] != level)
                continue;
            reconnect(graph, id, options.fine_repair_passes ? &enriched : nullptr, options, stats);
            ++stats.repaired;
        }
    stats.reinsert_s = elapsed(start);
    start = Clock::now();
    for (unsigned pass = 0; pass < options.fine_repair_passes; ++pass)
        for (Id id = 0; id < n; ++id)
            if (!unstable[id] && !enriched[id] && growUncovered(graph, id, pivots[id]))
                ++stats.refined;
    const auto connectivity_start = Clock::now();
    if (options.repair_connectivity)
        stats.bridges = ensureReachable(graph);
    stats.connectivity_s = elapsed(connectivity_start);
    stats.refine_s = elapsed(start);
    if (out_stats)
        *out_stats = stats;
    *out_space = space.release();
    return index.release();
}
