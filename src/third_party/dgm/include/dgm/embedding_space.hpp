// Modified for the STARR HNSW adapter: optional squared-L2 distance callback.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "dgm/types.hpp"

namespace dgm {

// A non-owning row-major embedding matrix with precomputed cosine norms.
// The caller must keep `values` alive for the lifetime of this object.
class EmbeddingSpace {
public:
    EmbeddingSpace(const float* values,
                   std::size_t row_count,
                   std::size_t dimensions,
                   Metric metric = Metric::kCosine,
                   std::size_t thread_count = 0,
                   float (*distance_callback)(const void*, const void*, const void*) = nullptr,
                   const void* distance_context = nullptr);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t dimensions() const noexcept;
    [[nodiscard]] Metric metric() const noexcept;
    [[nodiscard]] const float* row(NodeId id) const;
    [[nodiscard]] float distance(NodeId lhs, NodeId rhs) const;

private:
    float (*distance_callback_)(const void*, const void*, const void*){nullptr};
    const void* distance_context_{nullptr};
    const float* values_{nullptr};
    std::size_t row_count_{0};
    std::size_t dimensions_{0};
    Metric metric_{Metric::kCosine};
    std::vector<float> inverse_norms_;
};

}  // namespace dgm
