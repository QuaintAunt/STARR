#pragma once
#include "../query_ids.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <hnswlib/hnswlib.h>

using tableint = hnswlib::tableint;
using linklistsizeint = hnswlib::linklistsizeint;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

struct Vectors {
    std::vector<std::vector<float>> base;
    std::vector<std::vector<float>> query;
};

static Vectors loadPrefix(const std::string &path, int dim, size_t base_count, size_t query_count) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open " + path);
    Vectors result;
    result.base.assign(base_count, std::vector<float>(dim));
    result.query.assign(query_count, std::vector<float>(dim));
    for (auto &vector : result.base)
        if (!file.read(reinterpret_cast<char *>(vector.data()), dim * sizeof(float)))
            throw std::runtime_error("not enough base vectors in " + path);
    for (auto &vector : result.query)
        if (!file.read(reinterpret_cast<char *>(vector.data()), dim * sizeof(float)))
            throw std::runtime_error("not enough query vectors in " + path);
    return result;
}

static std::vector<std::vector<float>> sampleQueries(const std::vector<std::vector<float>> &base,
                                                     const std::vector<size_t> &query_ids) {
    std::vector<std::vector<float>> queries;
    queries.reserve(query_ids.size());
    for (size_t index : query_ids)
        queries.push_back(base.at(index));
    return queries;
}

static float l2(const std::vector<float> &left, const std::vector<float> &right) {
    float sum = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        const float delta = left[i] - right[i];
        sum += delta * delta;
    }
    return std::sqrt(sum);
}

static std::unordered_set<hnswlib::labeltype> truth(const std::vector<std::vector<float>> &base,
                                                    const std::vector<float> &query, size_t k) {
    std::vector<std::pair<float, hnswlib::labeltype>> distances(base.size());
    for (size_t i = 0; i < base.size(); ++i)
        distances[i] = {l2(base[i], query), i};
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
    std::unordered_set<hnswlib::labeltype> result;
    for (size_t i = 0; i < k; ++i)
        result.insert(distances[i].second);
    return result;
}

struct Evaluation {
    double recall;
    double milliseconds;
    std::vector<std::unordered_set<hnswlib::labeltype>> labels;
};

static std::vector<std::unordered_set<hnswlib::labeltype>>
makeTruths(const std::vector<std::vector<float>> &base,
           const std::vector<std::vector<float>> &queries) {
    std::vector<std::unordered_set<hnswlib::labeltype>> result;
    result.reserve(queries.size());
    for (const auto &query : queries)
        result.push_back(truth(base, query, 10));
    return result;
}

static void saveTruths(const fs::path &path,
                       const std::vector<std::unordered_set<hnswlib::labeltype>> &truths,
                       const std::vector<size_t> &query_ids) {
    if (truths.size() != query_ids.size())
        throw std::invalid_argument("truth/query count mismatch");
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream file(temporary, std::ios::binary);
    const uint64_t query_count = truths.size();
    const uint64_t k = truths.empty() ? 0 : truths.front().size();
    file.write(reinterpret_cast<const char *>(&query_count), sizeof(query_count));
    file.write(reinterpret_cast<const char *>(&k), sizeof(k));
    for (size_t id : query_ids) {
        const uint64_t value = id;
        file.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }
    for (const auto &labels : truths)
        for (hnswlib::labeltype label : labels)
            file.write(reinterpret_cast<const char *>(&label), sizeof(label));
    if (!file)
        throw std::runtime_error("failed to save truths: " + path.string());
    file.close();
    fs::rename(temporary, path);
}

static std::vector<std::unordered_set<hnswlib::labeltype>>
loadTruths(const fs::path &path, const std::vector<size_t> &query_ids) {
    std::ifstream file(path, std::ios::binary);
    uint64_t query_count = 0, k = 0;
    file.read(reinterpret_cast<char *>(&query_count), sizeof(query_count));
    file.read(reinterpret_cast<char *>(&k), sizeof(k));
    if (!file || query_count != query_ids.size() || k != 10)
        throw std::runtime_error("invalid truth cache: " + path.string());
    for (size_t id : query_ids) {
        uint64_t value;
        if (!file.read(reinterpret_cast<char *>(&value), sizeof(value)) || value != id)
            throw std::runtime_error("truth cache query IDs mismatch: " + path.string());
    }
    std::vector<std::unordered_set<hnswlib::labeltype>> truths(query_count);
    for (auto &labels : truths) {
        for (uint64_t i = 0; i < k; ++i) {
            hnswlib::labeltype label;
            file.read(reinterpret_cast<char *>(&label), sizeof(label));
            labels.insert(label);
        }
    }
    if (!file || file.peek() != EOF)
        throw std::runtime_error("truncated truth cache: " + path.string());
    return truths;
}

static Evaluation evaluate(hnswlib::HierarchicalNSW<float> *index,
                           const std::vector<std::vector<float>> &queries,
                           const std::vector<std::unordered_set<hnswlib::labeltype>> &truths) {
    index->setEf(200);
    for (const auto &query : queries)
        index->searchKnn(query.data(), 10);
    size_t matches = 0;
    double milliseconds = 0;
    std::vector<std::unordered_set<hnswlib::labeltype>> labels(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto &query = queries[i];
        const auto start = Clock::now();
        auto found = index->searchKnn(query.data(), 10);
        milliseconds += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        while (!found.empty()) {
            matches += truths[i].count(found.top().second);
            labels[i].insert(found.top().second);
            found.pop();
        }
    }
    return {static_cast<double>(matches) / (queries.size() * 10), milliseconds / queries.size(),
            std::move(labels)};
}

static double overlap(const Evaluation &left, const Evaluation &right) {
    size_t matches = 0;
    for (size_t i = 0; i < left.labels.size(); ++i)
        for (hnswlib::labeltype label : left.labels[i])
            matches += right.labels[i].count(label);
    return static_cast<double>(matches) / (left.labels.size() * 10);
}
