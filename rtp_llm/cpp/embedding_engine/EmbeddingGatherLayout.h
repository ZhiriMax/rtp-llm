#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtp_llm {

struct EmbeddingGatherRowView {
    const int32_t* token_ids      = nullptr;
    const int32_t* token_type_ids = nullptr;
    int            length        = 0;
};

struct EmbeddingGatherPrefixGroup {
    int              owner_row_idx       = 0;
    int              in_batch_shared_len = 0;
    int              reuse_len           = 0;
    std::vector<int> row_indices;
};

struct EmbeddingGatherRowInfo {
    int compute_begin      = 0;
    int compute_len        = 0;
    int kv_read_prefix_len = 0;
};

struct EmbeddingGatherRowRange {
    int begin = 0;
    int count = 0;
};

struct EmbeddingGatherLayout {
    std::vector<EmbeddingGatherPrefixGroup> prefix_groups;
    std::vector<EmbeddingGatherRowInfo>     rows;
};

EmbeddingGatherLayout buildEmbeddingGatherLayout(const std::vector<EmbeddingGatherRowView>& rows,
                                                 int                                        seq_size_per_block,
                                                 bool enable_in_batch_kv_prefix_dedup);

EmbeddingGatherLayout buildEmbeddingGatherLayoutForRanges(const std::vector<EmbeddingGatherRowView>&  rows,
                                                          const std::vector<EmbeddingGatherRowRange>& ranges,
                                                          int                                        seq_size_per_block,
                                                          bool enable_in_batch_kv_prefix_dedup);

void applyEmbeddingKVReuse(EmbeddingGatherLayout&                     layout,
                           const std::vector<EmbeddingGatherRowView>& rows,
                           size_t                                     group_idx,
                           int                                        reuse_len);

}  // namespace rtp_llm
