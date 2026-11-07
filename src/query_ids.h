#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

constexpr size_t kQueryCount = 1000;

static std::vector<size_t> actualQueryIds(size_t n) {
    if (n < kQueryCount)
        throw std::invalid_argument("need at least 1,000 indexed objects");
    const char *path = std::getenv("QUERY_IDS_BINARY");
    std::vector<size_t> ids(kQueryCount);
    if (!path) {
        std::vector<size_t> all(n);
        std::iota(all.begin(), all.end(), 0);
        std::mt19937 rng(42);
        std::sample(all.begin(), all.end(), ids.begin(), kQueryCount, rng);
        return ids;
    }
    std::ifstream file(path, std::ios::binary);
    std::set<size_t> unique;
    for (auto &id : ids) {
        uint64_t value;
        if (!file.read(reinterpret_cast<char *>(&value), sizeof(value)) || value >= n ||
            !unique.insert(value).second)
            throw std::runtime_error("invalid saved query ID");
        id = value;
    }
    if (file.peek() != EOF)
        throw std::runtime_error("query ID file trailing bytes");
    return ids;
}
