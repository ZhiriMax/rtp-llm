#include "rtp_llm/cpp/embedding_engine/EmbeddingGatherLayout.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace rtp_llm {

namespace {

int clampReuseLen(int reuse_len, int max_len) {
    return std::max(0, std::min(reuse_len, max_len));
}

void check(bool value, const char* message) {
    if (!value) {
        throw std::invalid_argument(message);
    }
}

}  // namespace

EmbeddingGatherLayout buildEmbeddingGatherLayout(const std::vector<EmbeddingGatherRowView>& rows,
                                                 int                                        seq_size_per_block,
                                                 bool enable_in_batch_kv_prefix_dedup) {
    check(seq_size_per_block > 0, "seq_size_per_block must be positive");

    EmbeddingGatherLayout layout;
    layout.rows.resize(rows.size());

    std::vector<int> all_row_indices(rows.size());
    std::iota(all_row_indices.begin(), all_row_indices.end(), 0);

    int in_batch_shared_len = 0;
    if (enable_in_batch_kv_prefix_dedup && rows.size() > 1) {
        int min_len = rows[0].length;
        for (const auto& row : rows) {
            min_len = std::min(min_len, row.length);
        }

        int lcp = 0;
        if (min_len > 0) {
            const auto& owner = rows[0];
            for (; lcp < min_len; ++lcp) {
                bool same = true;
                for (size_t row_idx = 1; row_idx < rows.size(); ++row_idx) {
                    const auto& row = rows[row_idx];
                    if (row.token_ids[lcp] != owner.token_ids[lcp]
                        || row.token_type_ids[lcp] != owner.token_type_ids[lcp]) {
                        same = false;
                        break;
                    }
                }
                if (!same) {
                    break;
                }
            }
        }

        in_batch_shared_len = lcp / seq_size_per_block * seq_size_per_block;
        if (in_batch_shared_len > 0) {
            for (const auto& row : rows) {
                if (row.length == in_batch_shared_len) {
                    in_batch_shared_len = std::max(0, in_batch_shared_len - seq_size_per_block);
                    break;
                }
            }
        }
    }

    if (in_batch_shared_len > 0) {
        layout.prefix_groups.push_back(EmbeddingGatherPrefixGroup{0, in_batch_shared_len, 0, all_row_indices});
        for (size_t i = 0; i < rows.size(); ++i) {
            const bool owner = i == 0;
            layout.rows[i]  = EmbeddingGatherRowInfo{
                owner ? 0 : in_batch_shared_len,
                owner ? rows[i].length : rows[i].length - in_batch_shared_len,
                owner ? 0 : in_batch_shared_len,
            };
        }
    } else {
        for (size_t i = 0; i < rows.size(); ++i) {
            layout.prefix_groups.push_back(
                EmbeddingGatherPrefixGroup{static_cast<int>(i), 0, 0, {static_cast<int>(i)}});
            layout.rows[i] = EmbeddingGatherRowInfo{0, rows[i].length, 0};
        }
    }

    return layout;
}

EmbeddingGatherLayout buildEmbeddingGatherLayoutForRanges(const std::vector<EmbeddingGatherRowView>&  rows,
                                                          const std::vector<EmbeddingGatherRowRange>& ranges,
                                                          int                                        seq_size_per_block,
                                                          bool enable_in_batch_kv_prefix_dedup) {
    check(seq_size_per_block > 0, "seq_size_per_block must be positive");

    EmbeddingGatherLayout layout;
    layout.rows.resize(rows.size());
    if (ranges.empty()) {
        check(rows.empty(), "ranges must be provided when rows is non-empty");
        return layout;
    }

    for (const auto& range : ranges) {
        check(range.begin >= 0, "range begin must be non-negative");
        check(range.count >= 0, "range count must be non-negative");
        check(static_cast<size_t>(range.begin + range.count) <= rows.size(), "range exceeds rows size");
        if (range.count == 0) {
            continue;
        }

        std::vector<EmbeddingGatherRowView> range_rows(rows.begin() + range.begin,
                                                       rows.begin() + range.begin + range.count);
        auto range_layout =
            buildEmbeddingGatherLayout(range_rows, seq_size_per_block, enable_in_batch_kv_prefix_dedup);

        for (size_t local_row = 0; local_row < range_layout.rows.size(); ++local_row) {
            layout.rows[range.begin + local_row] = range_layout.rows[local_row];
        }
        for (auto group : range_layout.prefix_groups) {
            group.owner_row_idx += range.begin;
            for (auto& row_idx : group.row_indices) {
                row_idx += range.begin;
            }
            layout.prefix_groups.push_back(std::move(group));
        }
    }

    return layout;
}

void applyEmbeddingKVReuse(EmbeddingGatherLayout&                     layout,
                           const std::vector<EmbeddingGatherRowView>& rows,
                           size_t                                     group_idx,
                           int                                        reuse_len) {
    check(group_idx < layout.prefix_groups.size(), "group_idx is out of range");
    auto& group = layout.prefix_groups[group_idx];

    if (group.in_batch_shared_len > 0) {
        group.reuse_len = clampReuseLen(reuse_len, group.in_batch_shared_len);
        for (auto row_idx : group.row_indices) {
            const bool owner = row_idx == group.owner_row_idx;
            layout.rows[row_idx] = EmbeddingGatherRowInfo{
                owner ? group.reuse_len : group.in_batch_shared_len,
                rows[row_idx].length - (owner ? group.reuse_len : group.in_batch_shared_len),
                owner ? group.reuse_len : group.in_batch_shared_len,
            };
        }
        return;
    }

    check(group.row_indices.size() == 1, "non-shared group must contain exactly one row");
    const int row_idx = group.row_indices[0];
    group.reuse_len   = clampReuseLen(reuse_len, rows[row_idx].length);
    layout.rows[row_idx] =
        EmbeddingGatherRowInfo{group.reuse_len, rows[row_idx].length - group.reuse_len, group.reuse_len};
}

}  // namespace rtp_llm
