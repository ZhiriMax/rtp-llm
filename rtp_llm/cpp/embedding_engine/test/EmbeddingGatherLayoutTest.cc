#include "rtp_llm/cpp/embedding_engine/EmbeddingGatherLayout.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

EmbeddingGatherRowView makeRow(const std::vector<int32_t>& token_ids,
                               const std::vector<int32_t>& token_type_ids) {
    return EmbeddingGatherRowView{token_ids.data(), token_type_ids.data(), static_cast<int>(token_ids.size())};
}

TEST(EmbeddingGatherLayoutTest, SharesOnlyRowsInSameRange) {
    const std::vector<int32_t> row0_tokens{1, 2, 3, 4, 5, 6};
    const std::vector<int32_t> row1_tokens{1, 2, 3, 4, 7, 8};
    const std::vector<int32_t> row2_tokens{1, 2, 3, 4, 9, 10};
    const std::vector<int32_t> token_types(row0_tokens.size(), 0);

    const std::vector<EmbeddingGatherRowView> rows{
        makeRow(row0_tokens, token_types),
        makeRow(row1_tokens, token_types),
        makeRow(row2_tokens, token_types),
    };
    const std::vector<EmbeddingGatherRowRange> ranges{
        EmbeddingGatherRowRange{0, 2},
        EmbeddingGatherRowRange{2, 1},
    };

    auto layout = buildEmbeddingGatherLayoutForRanges(rows, ranges, 4, true);

    ASSERT_EQ(layout.prefix_groups.size(), 2);
    EXPECT_EQ(layout.prefix_groups[0].in_batch_shared_len, 4);
    EXPECT_EQ(layout.prefix_groups[0].row_indices, (std::vector<int>{0, 1}));
    EXPECT_EQ(layout.prefix_groups[1].in_batch_shared_len, 0);
    EXPECT_EQ(layout.prefix_groups[1].row_indices, (std::vector<int>{2}));

    EXPECT_EQ(layout.rows[0].compute_begin, 0);
    EXPECT_EQ(layout.rows[0].compute_len, 6);
    EXPECT_EQ(layout.rows[1].compute_begin, 4);
    EXPECT_EQ(layout.rows[1].compute_len, 2);
    EXPECT_EQ(layout.rows[1].kv_read_prefix_len, 4);
    EXPECT_EQ(layout.rows[2].compute_begin, 0);
    EXPECT_EQ(layout.rows[2].compute_len, 6);
}

TEST(EmbeddingGatherLayoutTest, DropsSharedBlockWhenConsumerSuffixIsEmpty) {
    const std::vector<int32_t> row0_tokens{1, 2, 3, 4};
    const std::vector<int32_t> row1_tokens{1, 2, 3, 4};
    const std::vector<int32_t> token_types(row0_tokens.size(), 0);

    auto layout = buildEmbeddingGatherLayout(
        {makeRow(row0_tokens, token_types), makeRow(row1_tokens, token_types)}, 4, true);

    ASSERT_EQ(layout.prefix_groups.size(), 1);
    EXPECT_EQ(layout.prefix_groups[0].in_batch_shared_len, 0);
    EXPECT_EQ(layout.rows[0].compute_begin, 0);
    EXPECT_EQ(layout.rows[1].compute_begin, 0);
}

}  // namespace
}  // namespace rtp_llm
