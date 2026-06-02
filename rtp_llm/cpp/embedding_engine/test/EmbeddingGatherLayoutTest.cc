#include "rtp_llm/cpp/embedding_engine/EmbeddingGatherLayout.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rtp_llm {
namespace {

EmbeddingGatherRowView rowView(const std::vector<int32_t>& tokens, const std::vector<int32_t>& types) {
    return EmbeddingGatherRowView{tokens.data(), types.data(), static_cast<int>(tokens.size())};
}

std::vector<int32_t> zeros(size_t size) {
    return std::vector<int32_t>(size, 0);
}

TEST(EmbeddingGatherLayoutTest, BuildsSharedPrefixGroupForAlignedCommonPrefix) {
    std::vector<int32_t> row0_tokens{1, 2, 3, 4, 5, 6};
    std::vector<int32_t> row1_tokens{1, 2, 3, 4, 7, 8};
    auto                 row0_types = zeros(row0_tokens.size());
    auto                 row1_types = zeros(row1_tokens.size());

    auto layout = buildEmbeddingGatherLayout(
        {rowView(row0_tokens, row0_types), rowView(row1_tokens, row1_types)}, 4, true);

    ASSERT_EQ(layout.prefix_groups.size(), 1);
    EXPECT_EQ(layout.prefix_groups[0].owner_row_idx, 0);
    EXPECT_EQ(layout.prefix_groups[0].in_batch_shared_len, 4);
    EXPECT_EQ(layout.prefix_groups[0].row_indices, std::vector<int>({0, 1}));

    ASSERT_EQ(layout.rows.size(), 2);
    EXPECT_EQ(layout.rows[0].compute_begin, 0);
    EXPECT_EQ(layout.rows[0].compute_len, 6);
    EXPECT_EQ(layout.rows[0].kv_read_prefix_len, 0);
    EXPECT_EQ(layout.rows[1].compute_begin, 4);
    EXPECT_EQ(layout.rows[1].compute_len, 2);
    EXPECT_EQ(layout.rows[1].kv_read_prefix_len, 4);
}

TEST(EmbeddingGatherLayoutTest, AlignsCommonPrefixToBlockSize) {
    std::vector<int32_t> row0_tokens{1, 2, 3, 4, 5, 6};
    std::vector<int32_t> row1_tokens{1, 2, 3, 4, 5, 8};
    auto                 row0_types = zeros(row0_tokens.size());
    auto                 row1_types = zeros(row1_tokens.size());

    auto layout = buildEmbeddingGatherLayout(
        {rowView(row0_tokens, row0_types), rowView(row1_tokens, row1_types)}, 4, true);

    ASSERT_EQ(layout.prefix_groups.size(), 1);
    EXPECT_EQ(layout.prefix_groups[0].in_batch_shared_len, 4);
}

TEST(EmbeddingGatherLayoutTest, TypeMismatchBreaksCommonPrefix) {
    std::vector<int32_t> row0_tokens{1, 2, 3, 4, 5};
    std::vector<int32_t> row1_tokens{1, 2, 3, 4, 6};
    std::vector<int32_t> row0_types{0, 0, 0, 0, 0};
    std::vector<int32_t> row1_types{0, 0, 1, 0, 0};

    auto layout = buildEmbeddingGatherLayout(
        {rowView(row0_tokens, row0_types), rowView(row1_tokens, row1_types)}, 4, true);

    ASSERT_EQ(layout.prefix_groups.size(), 2);
    EXPECT_EQ(layout.prefix_groups[0].in_batch_shared_len, 0);
    EXPECT_EQ(layout.prefix_groups[1].in_batch_shared_len, 0);
    EXPECT_EQ(layout.rows[0].compute_begin, 0);
    EXPECT_EQ(layout.rows[1].compute_begin, 0);
}

TEST(EmbeddingGatherLayoutTest, DropsLastSharedBlockWhenConsumerSuffixWouldBeEmpty) {
    std::vector<int32_t> row0_tokens{1, 2, 3, 4};
    std::vector<int32_t> row1_tokens{1, 2, 3, 4, 5};
    auto                 row0_types = zeros(row0_tokens.size());
    auto                 row1_types = zeros(row1_tokens.size());

    auto layout = buildEmbeddingGatherLayout(
        {rowView(row0_tokens, row0_types), rowView(row1_tokens, row1_types)}, 4, true);

    ASSERT_EQ(layout.prefix_groups.size(), 2);
    EXPECT_EQ(layout.rows[0].compute_len, 4);
    EXPECT_EQ(layout.rows[1].compute_len, 5);
}

TEST(EmbeddingGatherLayoutTest, AppliesBlockCacheReuseToOwnerOnlyForSharedGroup) {
    std::vector<int32_t> row0_tokens{1, 2, 3, 4, 5, 6};
    std::vector<int32_t> row1_tokens{1, 2, 3, 4, 7, 8};
    auto                 row0_types = zeros(row0_tokens.size());
    auto                 row1_types = zeros(row1_tokens.size());
    std::vector<EmbeddingGatherRowView> rows{rowView(row0_tokens, row0_types), rowView(row1_tokens, row1_types)};

    auto layout = buildEmbeddingGatherLayout(rows, 4, true);
    applyEmbeddingKVReuse(layout, rows, 0, 4);

    EXPECT_EQ(layout.prefix_groups[0].reuse_len, 4);
    EXPECT_EQ(layout.rows[0].compute_begin, 4);
    EXPECT_EQ(layout.rows[0].compute_len, 2);
    EXPECT_EQ(layout.rows[0].kv_read_prefix_len, 4);
    EXPECT_EQ(layout.rows[1].compute_begin, 4);
    EXPECT_EQ(layout.rows[1].compute_len, 2);
    EXPECT_EQ(layout.rows[1].kv_read_prefix_len, 4);
}

TEST(EmbeddingGatherLayoutTest, AppliesBlockCacheReuseToSingleRowGroup) {
    std::vector<int32_t> row0_tokens{1, 2, 3, 4, 5, 6};
    auto                 row0_types = zeros(row0_tokens.size());
    std::vector<EmbeddingGatherRowView> rows{rowView(row0_tokens, row0_types)};

    auto layout = buildEmbeddingGatherLayout(rows, 4, false);
    applyEmbeddingKVReuse(layout, rows, 0, 4);

    ASSERT_EQ(layout.prefix_groups.size(), 1);
    EXPECT_EQ(layout.prefix_groups[0].reuse_len, 4);
    EXPECT_EQ(layout.rows[0].compute_begin, 4);
    EXPECT_EQ(layout.rows[0].compute_len, 2);
    EXPECT_EQ(layout.rows[0].kv_read_prefix_len, 4);
}

TEST(EmbeddingGatherLayoutTest, BuildsSharedPrefixOnlyWithinEachRange) {
    std::vector<int32_t> row0_tokens{1, 2, 3, 4, 10};
    std::vector<int32_t> row1_tokens{1, 2, 3, 4, 11};
    std::vector<int32_t> row2_tokens{8, 7, 6, 5, 20};
    std::vector<int32_t> row3_tokens{8, 7, 6, 5, 21};
    auto                 row0_types = zeros(row0_tokens.size());
    auto                 row1_types = zeros(row1_tokens.size());
    auto                 row2_types = zeros(row2_tokens.size());
    auto                 row3_types = zeros(row3_tokens.size());
    std::vector<EmbeddingGatherRowView> rows{rowView(row0_tokens, row0_types),
                                             rowView(row1_tokens, row1_types),
                                             rowView(row2_tokens, row2_types),
                                             rowView(row3_tokens, row3_types)};

    auto layout = buildEmbeddingGatherLayoutForRanges(rows,
                                                      {EmbeddingGatherRowRange{0, 2}, EmbeddingGatherRowRange{2, 2}},
                                                      4,
                                                      true);

    ASSERT_EQ(layout.prefix_groups.size(), 2);
    EXPECT_EQ(layout.prefix_groups[0].owner_row_idx, 0);
    EXPECT_EQ(layout.prefix_groups[0].in_batch_shared_len, 4);
    EXPECT_EQ(layout.prefix_groups[0].row_indices, std::vector<int>({0, 1}));
    EXPECT_EQ(layout.prefix_groups[1].owner_row_idx, 2);
    EXPECT_EQ(layout.prefix_groups[1].in_batch_shared_len, 4);
    EXPECT_EQ(layout.prefix_groups[1].row_indices, std::vector<int>({2, 3}));

    EXPECT_EQ(layout.rows[0].compute_begin, 0);
    EXPECT_EQ(layout.rows[1].compute_begin, 4);
    EXPECT_EQ(layout.rows[2].compute_begin, 0);
    EXPECT_EQ(layout.rows[3].compute_begin, 4);
}

TEST(EmbeddingGatherLayoutTest, RejectsNonEmptyRowsWithoutRanges) {
    std::vector<int32_t> row0_tokens{1, 2, 3, 4};
    auto                 row0_types = zeros(row0_tokens.size());
    std::vector<EmbeddingGatherRowView> rows{rowView(row0_tokens, row0_types)};

    EXPECT_THROW(buildEmbeddingGatherLayoutForRanges(rows, {}, 4, true), std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
