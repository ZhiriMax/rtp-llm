#include "ATen/ops/ones.h"
#include "c10/core/ScalarType.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingExecutor.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include <ATen/TensorIndexing.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <set>

using namespace std;
using namespace at::indexing;

namespace rtp_llm {

namespace {

constexpr int64_t kEmbeddingCacheRequestId = 0;

std::string embeddingKVCacheModeFromEnv() {
    const char* mode = std::getenv("EMBEDDING_KV_CACHE_MODE");
    return mode ? std::string(mode) : std::string(kEmbeddingKVCacheModeOff);
}

std::string embeddingKVCacheCommitPolicyFromEnv() {
    const char* policy = std::getenv("EMBEDDING_KV_CACHE_COMMIT_POLICY");
    return policy ? std::string(policy) : std::string(kEmbeddingKVCacheCommitPolicyPrefixBlock);
}

bool isValidEmbeddingKVCacheMode(const std::string& mode) {
    return mode == kEmbeddingKVCacheModeOff || mode == kEmbeddingKVCacheModeBlock
           || mode == kEmbeddingKVCacheModeInBatch;
}

bool isValidEmbeddingKVCacheCommitPolicy(const std::string& policy) {
    return policy == kEmbeddingKVCacheCommitPolicyPrefixBlock
           || policy == kEmbeddingKVCacheCommitPolicyFullBlock;
}

bool isLastTokenSafeHandler(const py::object& handler) {
    if (!py::hasattr(handler, "is_last_token_safe_for_kv_cache")) {
        return false;
    }
    return py::cast<bool>(handler.attr("is_last_token_safe_for_kv_cache")());
}

std::vector<int32_t> makeCacheTokenVector(const int32_t* token_ids, const int32_t* token_type_ids, int length) {
    std::vector<int32_t> cache_tokens;
    if (length <= 0) {
        return cache_tokens;
    }
    cache_tokens.reserve(length);
    for (int i = 0; i < length; ++i) {
        const int32_t token_type = token_type_ids ? token_type_ids[i] : 0;
        cache_tokens.push_back(hashInt64Array(token_ids[i], &token_type, &token_type + 1));
    }
    return cache_tokens;
}

std::vector<int64_t> buildCacheKeys(const int32_t* token_ids,
                                    const int32_t* token_type_ids,
                                    int            length,
                                    int            seq_size_per_block) {
    std::vector<int64_t> cache_keys;
    if (seq_size_per_block <= 0 || length <= 0) {
        return cache_keys;
    }
    cache_keys.reserve(length / seq_size_per_block);
    int64_t hash = 0;
    for (int pos = 0; pos + seq_size_per_block <= length; pos += seq_size_per_block) {
        hash = hashInt64Array(hash, token_ids + pos, token_ids + pos + seq_size_per_block);
        hash = hashInt64Array(hash, token_type_ids + pos, token_type_ids + pos + seq_size_per_block);
        cache_keys.push_back(hash);
    }
    return cache_keys;
}

std::vector<int> uniqueBlocks(const std::vector<std::vector<int>>& row_blocks) {
    std::set<int> seen;
    std::vector<int> blocks;
    for (const auto& row : row_blocks) {
        for (auto block : row) {
            if (seen.insert(block).second) {
                blocks.push_back(block);
            }
        }
    }
    return blocks;
}

struct ScopeGuard {
    std::function<void()> fn;
    ~ScopeGuard() {
        if (fn) {
            fn();
        }
    }
    void dismiss() {
        fn = nullptr;
    }
};

}  // namespace

namespace HandlerArgs {

static const char* names[] = {
    "input_lengths",
    "hidden_states",
    "input_ids",
    "attention_mask",
    "moe_gating",
};
static_assert(sizeof(names) / sizeof(names[0]) <= NUM_INPUT_TYPES, "redundant handler arg name");
static_assert(sizeof(names) / sizeof(names[0]) >= NUM_INPUT_TYPES, "missing handler arg name");

static bool set_by_str(Flag& flag, const char* name) {
    for (size_t i = 0; i < NUM_INPUT_TYPES; ++i) {
        if (std::strcmp(names[i], name) == 0) {
            flag.set(i);
            return true;
        }
    }
    return false;
}

static const char* get_name(Arg idx) {
    return names[static_cast<size_t>(idx)];
}

static bool has_arg(const Flag& flag, Arg idx) {
    return flag.test(static_cast<size_t>(idx));
}

}  // namespace HandlerArgs

EmbeddingExecutor::EmbeddingExecutor(const EngineInitParams&              params,
                                     rtp_llm::DeviceBase*                device,
                                     py::object                          handler,
                                     const std::shared_ptr<CacheManager>& cache_manager,
                                     const rtp_llm::GptInitParameter*     embedding_params):
    handler_(handler),
    handler_args_(),
    device_(device),
    metrics_reporter_(params.metrics_reporter),
    params_(embedding_params ? *embedding_params : params.gpt_init_parameter),
    cache_manager_(cache_manager),
    embedding_kv_cache_mode_(embeddingKVCacheModeFromEnv()),
    embedding_kv_cache_commit_policy_(embeddingKVCacheCommitPolicyFromEnv()) {
    RTP_LLM_CHECK_WITH_INFO(isValidEmbeddingKVCacheMode(embedding_kv_cache_mode_),
                            "invalid EMBEDDING_KV_CACHE_MODE: %s",
                            embedding_kv_cache_mode_.c_str());
    RTP_LLM_CHECK_WITH_INFO(isValidEmbeddingKVCacheCommitPolicy(embedding_kv_cache_commit_policy_),
                            "invalid EMBEDDING_KV_CACHE_COMMIT_POLICY: %s",
                            embedding_kv_cache_commit_policy_.c_str());
    enable_embedding_kv_cache_ =
        cache_manager_
        && (embedding_kv_cache_mode_ == kEmbeddingKVCacheModeBlock
            || embedding_kv_cache_mode_ == kEmbeddingKVCacheModeInBatch);
    enable_in_batch_kv_prefix_dedup_ =
        enable_embedding_kv_cache_ && embedding_kv_cache_mode_ == kEmbeddingKVCacheModeInBatch;

    GptModelInitParams model_init_params({device_,
                                          params.gpt_weights,
                                          Executor::genModelDescription(params_),
                                          cache_manager_ ? ((optional<KVCacheAllocator::KVCacheBuffer>)
                                                               cache_manager_->kvCacheBuffer()) :
                                                           nullopt,
                                          params.model_id});

    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(model_init_params, params.py_model, true));
    } else {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model");
        model_.reset(new GptModel(model_init_params));
    }

    init_position_ids(params_.max_seq_len_);
    std::vector<std::string> handler_args;
    {
        py::gil_scoped_acquire acquire;
        torch_type_  = py::module::import("torch").attr("Tensor");
        handler_args = py::cast<std::vector<std::string>>(handler_.attr("extend_forward_args")());
    }

    for (const auto& name : handler_args) {
        if (!HandlerArgs::set_by_str(handler_args_, name.c_str())) {
            RTP_LLM_LOG_WARNING("unknown handler arg: \"%s\", ignored", name.c_str());
        }
    }

    allow_in_batch_kv_prefix_dedup_ = false;
    if (enable_in_batch_kv_prefix_dedup_) {
        try {
            py::gil_scoped_acquire acquire;
            allow_in_batch_kv_prefix_dedup_ = isLastTokenSafeHandler(handler_);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("failed to inspect embedding handler last-token safety: %s", e.what());
        }
        if (!allow_in_batch_kv_prefix_dedup_) {
            RTP_LLM_LOG_WARNING("EMBEDDING_KV_CACHE_MODE=in_batch is only enabled for last-token-safe handlers in "
                                "this version; fallback to block mode for this handler.");
        }
    }
}

void EmbeddingExecutor::init_position_ids(int max_seq_len) {
    max_position_ids_buf_ = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)max_seq_len}, rtp_llm::AllocationType::HOST}, {});
    int* position_ids = (int*)max_position_ids_buf_->data();
    for (int i = 0; i < max_seq_len; i++) {
        position_ids[i] = i;
    }
}

absl::StatusOr<EmbeddingExecutor::EmbeddingModelInput>
EmbeddingExecutor::gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const {
    EmbeddingModelInput result;
    if (!enable_embedding_kv_cache_) {
        CHECK_AND_RETURN_REF(model_input, gatherLegacyModelInput(streams));
        result.model_input = std::move(model_input);
        return result;
    }
    return gatherKVCacheModelInput(streams);
}

absl::StatusOr<GptModelInputs>
EmbeddingExecutor::gatherLegacyModelInput(const std::list<EmbeddingStreamPtr>& streams) const {
    int64_t token_num  = 0;
    int64_t batch_size = 0;
    calcTokenNum(streams, token_num, batch_size);
    GptModelInputs model_input;
    model_input.combo_tokens = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)token_num}, rtp_llm::AllocationType::HOST}, {});
    model_input.combo_tokens_type_ids = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)token_num}, rtp_llm::AllocationType::HOST}, {});
    model_input.combo_position_ids = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)token_num}, rtp_llm::AllocationType::HOST}, {});
    model_input.input_lengths = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)batch_size}, rtp_llm::AllocationType::HOST}, {});
    model_input.sequence_lengths =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {0}, rtp_llm::AllocationType::HOST}, {});
    model_input.prefix_lengths = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)batch_size}, rtp_llm::AllocationType::HOST}, {});
    memset(model_input.prefix_lengths->data(), 0, model_input.prefix_lengths->sizeBytes());
    int* merged_tokens         = model_input.combo_tokens->data<int>();
    int* input_lengths         = model_input.input_lengths->data<int>();
    int* merged_positon_ids    = model_input.combo_position_ids->data<int>();
    int* merged_token_type_ids = model_input.combo_tokens_type_ids->data<int>();
    int  token_idx             = 0;
    int  batch_idx             = 0;
    int  position_bias         = 0;
    if (params_.position_ids_style_ == 1) {
        position_bias = params_.special_tokens_.pad_token_id_ + 1;
    }
    std::vector<rtp_llm::BufferPtr> gathered_mm_features;
    std::vector<int>                new_locs;
    std::vector<int>                merged_text_mask;
    std::vector<rtp_llm::BufferPtr> gathered_input_embeddings;
    std::vector<int>                gathered_input_embeddings_locs;
    merged_text_mask.resize(token_num, 1);
    for (auto& stream : streams) {
        int         length     = stream->inputLength();
        int         batchSize  = stream->batchSize();
        const auto& mm_feature = stream->multimodalFeature();
        if (mm_feature.has_value()) {
            for (const auto& feature : mm_feature.value().features) {
                gathered_mm_features.emplace_back(torchTensor2Buffer(feature));
            }
            const auto mm_locs = mm_feature.value().locs;
            for (int i = 0; i < mm_locs->size(); ++i) {
                new_locs.push_back(*mm_locs->dataWithOffset<int>(i) + token_idx);
            }
            const auto text_token_mask = mm_feature.value().text_tokens_mask;
            memcpy(merged_text_mask.data() + token_idx, text_token_mask->data(), text_token_mask->size() * sizeof(int));
        }

        if (stream->embeddingInput()->input_embeddings.has_value()) {
            auto input_embedding_buffer = stream->embeddingInput()->input_embeddings.value();
            gathered_input_embeddings.emplace_back(
                device_->clone({*input_embedding_buffer, rtp_llm::AllocationType::HOST}));
            gathered_input_embeddings_locs.push_back(token_idx);
        }
        memcpy(merged_tokens + (int)token_idx, stream->embeddingInput()->token_ids->data(), length * sizeof(int32_t));
        memcpy(merged_token_type_ids + (int)token_idx,
               stream->embeddingInput()->token_type_ids->data(),
               length * sizeof(int32_t));
        memcpy(input_lengths + (int)batch_idx,
               stream->embeddingInput()->input_lengths->data(),
               stream->batchSize() * sizeof(int32_t));
        int length_idx = 0;
        for (int i = 0; i < batchSize; i++) {
            int seqLen = stream->embeddingInput()->input_lengths->data<int32_t>()[i];
            RTP_LLM_CHECK_WITH_INFO(seqLen + position_bias <= int(max_position_ids_buf_->shape()[0]),
                                    "position index exceed max_position_length");
            memcpy(merged_positon_ids + token_idx + length_idx,
                   max_position_ids_buf_->data<int32_t>() + position_bias,
                   seqLen * sizeof(int32_t));
            length_idx += seqLen;
        }
        if (length_idx != length) {
            return absl::InternalError("stream total_length not equal to sum of lengths");
        }
        batch_idx += stream->batchSize();
        token_idx += length;
    }
    if (!gathered_mm_features.empty()) {
        model_input.multimodal_features = std::move(gathered_mm_features);
        model_input.mm_features_locs    = device_->clone({*vector2Buffer(new_locs), rtp_llm::AllocationType::HOST});
        model_input.text_tokens_mask =
            device_->clone({*vector2Buffer(merged_text_mask), rtp_llm::AllocationType::HOST});
    }

    if (!gathered_input_embeddings.empty()) {
        model_input.input_embeddings = std::move(gathered_input_embeddings);
        model_input.input_embeddings_locs =
            device_->clone({*vector2Buffer(gathered_input_embeddings_locs), rtp_llm::AllocationType::HOST});
    }

    size_t max_seq_len = *std::max_element(input_lengths, input_lengths + batch_size);
    if (HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::MOE_GATING)) {
        model_input.need_moe_gating = true;
    }
    reportMetrics(batch_size, token_num, max_seq_len);
    return model_input;
}

absl::StatusOr<EmbeddingExecutor::EmbeddingModelInput>
EmbeddingExecutor::gatherKVCacheModelInput(const std::list<EmbeddingStreamPtr>& streams) const {
    struct RowRef {
        EmbeddingStreamPtr stream;
        int                local_row_idx = 0;
        int                token_offset  = 0;
        int                length        = 0;
    };

    const auto& cache_config       = cache_manager_->cacheConfig();
    const int   seq_size_per_block = static_cast<int>(cache_config.seq_size_per_block);
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be positive");

    int64_t legacy_token_num = 0;
    int64_t batch_size       = 0;
    calcTokenNum(streams, legacy_token_num, batch_size);

    for (const auto& stream : streams) {
        if (stream->multimodalFeature().has_value() || stream->embeddingInput()->input_embeddings.has_value()) {
            RTP_LLM_LOG_WARNING("embedding kv cache path does not support multimodal features or input_embeddings yet; "
                                "fallback to legacy gather");
            CHECK_AND_RETURN_REF(model_input, gatherLegacyModelInput(streams));
            EmbeddingModelInput result;
            result.model_input = std::move(model_input);
            return result;
        }
    }

    std::vector<RowRef>                 rows;
    std::vector<EmbeddingGatherRowView> row_views;
    std::vector<EmbeddingGatherRowRange> row_ranges;
    rows.reserve(batch_size);
    row_views.reserve(batch_size);
    row_ranges.reserve(streams.size());
    for (const auto& stream : streams) {
        int       offset      = 0;
        const int range_begin = static_cast<int>(rows.size());
        auto*     lengths     = stream->embeddingInput()->input_lengths->data<int32_t>();
        for (int local_row = 0; local_row < stream->batchSize(); ++local_row) {
            const int length = lengths[local_row];
            rows.push_back(RowRef{stream, local_row, offset, length});
            row_views.push_back(EmbeddingGatherRowView{
                stream->embeddingInput()->token_ids->data<int32_t>() + offset,
                stream->embeddingInput()->token_type_ids->data<int32_t>() + offset,
                length,
            });
            offset += length;
        }
        if (offset != stream->inputLength()) {
            return absl::InternalError("stream total_length not equal to sum of lengths");
        }
        row_ranges.push_back(EmbeddingGatherRowRange{range_begin, static_cast<int>(rows.size()) - range_begin});
    }

    auto layout = buildEmbeddingGatherLayoutForRanges(
        row_views,
        row_ranges,
        seq_size_per_block,
        enable_in_batch_kv_prefix_dedup_ && allow_in_batch_kv_prefix_dedup_);

    int max_original_len = 0;
    for (const auto& row : rows) {
        max_original_len = std::max(max_original_len, row.length);
    }

    EmbeddingModelInput result;
    result.model_input.sequence_lengths =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {0}, rtp_llm::AllocationType::HOST}, {});
    result.model_input.seq_size_per_block = cache_config.seq_size_per_block;
    result.model_input.k_block_size       = cache_config.k_block_stride;
    result.model_input.v_block_size       = cache_config.v_block_stride;
    result.model_input.scale_block_size   = cache_config.kv_scale_block_stride;

    const int max_blocks_num = (max_original_len + seq_size_per_block - 1) / seq_size_per_block;
    if (max_blocks_num > 0) {
        result.model_input.kv_cache_block_id = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32,
             {static_cast<size_t>(batch_size), static_cast<size_t>(max_blocks_num)},
             rtp_llm::AllocationType::HOST},
            {});
        memset(result.model_input.kv_cache_block_id->data(), 0, result.model_input.kv_cache_block_id->sizeBytes());
    }

    auto malloc_blocks = [&](int block_num) -> absl::StatusOr<std::vector<int>> {
        if (block_num <= 0) {
            return std::vector<int>{};
        }
        auto [success, resource] =
            cache_manager_->malloc(KVCacheAllocator::SimpleMallocInfo(kEmbeddingCacheRequestId, block_num));
        if (!success) {
            return absl::ResourceExhaustedError("embedding kv cache malloc failed");
        }
        return resource.block_id;
    };

    for (size_t group_idx = 0; group_idx < layout.prefix_groups.size(); ++group_idx) {
        const auto& group = layout.prefix_groups[group_idx];
        int         max_group_len = 0;
        for (auto row_idx : group.row_indices) {
            max_group_len = std::max(max_group_len, rows[row_idx].length);
        }
        const auto& owner  = rows[group.owner_row_idx];
        const int init_len = group.in_batch_shared_len > 0 ?
                                 std::min(owner.length, group.in_batch_shared_len + 1) :
                                 max_group_len;
        auto init_tokens =
            makeCacheTokenVector(owner.stream->embeddingInput()->token_ids->data<int32_t>() + owner.token_offset,
                                 owner.stream->embeddingInput()->token_type_ids->data<int32_t>() + owner.token_offset,
                                 init_len);
        auto init_cache_keys =
            buildCacheKeys(owner.stream->embeddingInput()->token_ids->data<int32_t>() + owner.token_offset,
                           owner.stream->embeddingInput()->token_type_ids->data<int32_t>() + owner.token_offset,
                           init_len,
                           seq_size_per_block);

        auto match_info = cache_manager_->mallocWithCache(
            CacheManager::AdvancedMallocInfo(kEmbeddingCacheRequestId, init_tokens, init_cache_keys));
        std::vector<int> group_blocks_to_free = match_info.cache_blocks;
        ScopeGuard group_guard{[this, &group_blocks_to_free]() {
            auto blocks = uniqueBlocks({group_blocks_to_free});
            if (!blocks.empty()) {
                cache_manager_->free(blocks);
            }
        }};
        applyEmbeddingKVReuse(layout, row_views, group_idx, match_info.reuse_length);

        const int shared_block_num = group.in_batch_shared_len / seq_size_per_block;
        std::vector<int> shared_blocks(match_info.cache_blocks.begin(),
                                       match_info.cache_blocks.begin()
                                           + std::min<int>(shared_block_num, match_info.cache_blocks.size()));
        if (group.in_batch_shared_len > 0 && static_cast<int>(shared_blocks.size()) < shared_block_num) {
            CHECK_AND_RETURN_REF(extra_shared_blocks,
                                 malloc_blocks(shared_block_num - static_cast<int>(shared_blocks.size())));
            shared_blocks.insert(
                shared_blocks.end(), extra_shared_blocks.begin(), extra_shared_blocks.end());
            group_blocks_to_free.insert(
                group_blocks_to_free.end(), extra_shared_blocks.begin(), extra_shared_blocks.end());
        }

        EmbeddingKVCacheAllocation allocation;
        allocation.prefix_group       = layout.prefix_groups[group_idx];
        allocation.seq_size_per_block = seq_size_per_block;
        allocation.row_block_indices.resize(group.row_indices.size());
        allocation.row_token_ids.resize(group.row_indices.size());
        allocation.row_cache_keys.resize(group.row_indices.size());
        allocation.row_lengths.reserve(group.row_indices.size());

        for (size_t local_row = 0; local_row < group.row_indices.size(); ++local_row) {
            const int   row_idx = group.row_indices[local_row];
            const auto& row     = rows[row_idx];
            const int   row_block_num = (row.length + seq_size_per_block - 1) / seq_size_per_block;

            std::vector<int> row_blocks;
            if (group.in_batch_shared_len > 0) {
                row_blocks = shared_blocks;
            } else {
                row_blocks = match_info.cache_blocks;
                if (static_cast<int>(row_blocks.size()) > row_block_num) {
                    row_blocks.resize(row_block_num);
                }
            }
            if (static_cast<int>(row_blocks.size()) < row_block_num) {
                CHECK_AND_RETURN_REF(extra_blocks, malloc_blocks(row_block_num - static_cast<int>(row_blocks.size())));
                row_blocks.insert(row_blocks.end(), extra_blocks.begin(), extra_blocks.end());
                group_blocks_to_free.insert(group_blocks_to_free.end(), extra_blocks.begin(), extra_blocks.end());
            }
            allocation.row_block_indices[local_row] = std::move(row_blocks);
            allocation.row_lengths.push_back(row.length);
            allocation.row_token_ids[local_row] =
                makeCacheTokenVector(row.stream->embeddingInput()->token_ids->data<int32_t>() + row.token_offset,
                                     row.stream->embeddingInput()->token_type_ids->data<int32_t>() + row.token_offset,
                                     row.length);
            allocation.row_cache_keys[local_row] =
                buildCacheKeys(row.stream->embeddingInput()->token_ids->data<int32_t>() + row.token_offset,
                               row.stream->embeddingInput()->token_type_ids->data<int32_t>() + row.token_offset,
                               row.length,
                               seq_size_per_block);

            if (max_blocks_num > 0) {
                auto* block_dst = result.model_input.kv_cache_block_id->data<int32_t>()
                                  + static_cast<size_t>(row_idx) * max_blocks_num;
                memcpy(block_dst,
                       allocation.row_block_indices[local_row].data(),
                       allocation.row_block_indices[local_row].size() * sizeof(int32_t));
            }
        }
        result.kv_cache_allocations.push_back(std::move(allocation));
        group_guard.dismiss();
    }

    int64_t token_num        = 0;
    int     max_compute_len  = 0;
    for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
        token_num += layout.rows[row_idx].compute_len;
        max_compute_len = std::max(max_compute_len, layout.rows[row_idx].compute_len);
    }

    result.model_input.combo_tokens = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {static_cast<size_t>(token_num)}, rtp_llm::AllocationType::HOST}, {});
    result.model_input.combo_tokens_type_ids = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {static_cast<size_t>(token_num)}, rtp_llm::AllocationType::HOST}, {});
    result.model_input.combo_position_ids = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {static_cast<size_t>(token_num)}, rtp_llm::AllocationType::HOST}, {});
    result.model_input.input_lengths = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {static_cast<size_t>(batch_size)}, rtp_llm::AllocationType::HOST}, {});
    result.model_input.prefix_lengths = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {static_cast<size_t>(batch_size)}, rtp_llm::AllocationType::HOST}, {});
    memset(result.model_input.prefix_lengths->data(), 0, result.model_input.prefix_lengths->sizeBytes());

    int32_t* merged_tokens         = result.model_input.combo_tokens->data<int32_t>();
    int32_t* merged_token_type_ids = result.model_input.combo_tokens_type_ids->data<int32_t>();
    int32_t* merged_position_ids   = result.model_input.combo_position_ids->data<int32_t>();
    int32_t* input_lengths         = result.model_input.input_lengths->data<int32_t>();
    int32_t* prefix_lengths        = result.model_input.prefix_lengths->data<int32_t>();

    int position_bias = 0;
    if (params_.position_ids_style_ == 1) {
        position_bias = params_.special_tokens_.pad_token_id_ + 1;
    }

    int token_idx = 0;
    for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
        const auto& row        = rows[row_idx];
        const auto& row_layout = layout.rows[row_idx];
        RTP_LLM_CHECK_WITH_INFO(row.length + position_bias <= int(max_position_ids_buf_->shape()[0]),
                                "position index exceed max_position_length");
        auto* src_tokens = row.stream->embeddingInput()->token_ids->data<int32_t>() + row.token_offset
                           + row_layout.compute_begin;
        auto* src_types = row.stream->embeddingInput()->token_type_ids->data<int32_t>() + row.token_offset
                          + row_layout.compute_begin;
        memcpy(merged_tokens + token_idx, src_tokens, row_layout.compute_len * sizeof(int32_t));
        memcpy(merged_token_type_ids + token_idx, src_types, row_layout.compute_len * sizeof(int32_t));
        memcpy(merged_position_ids + token_idx,
               max_position_ids_buf_->data<int32_t>() + position_bias + row_layout.compute_begin,
               row_layout.compute_len * sizeof(int32_t));
        input_lengths[row_idx]  = row_layout.compute_len;
        prefix_lengths[row_idx] = row_layout.kv_read_prefix_len;
        token_idx += row_layout.compute_len;
    }

    if (HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::MOE_GATING)) {
        result.model_input.need_moe_gating = true;
    }
    reportMetrics(batch_size, token_num, max_compute_len);
    return result;
}

ModelRequest EmbeddingExecutor::generateOldModelRequest(GptModelInputs& model_input) {
    ModelRequest model_request;
    model_request.generate_batch_size  = 0;
    model_request.context_batch_size   = model_input.input_lengths->shape()[0];
    model_request.combo_tokens         = model_input.combo_tokens;
    model_request.combo_position_ids   = model_input.combo_position_ids;
    model_request.combo_token_type_ids = model_input.combo_tokens_type_ids;
    model_request.input_lengths        = model_input.input_lengths;
    model_request.sequence_lengths     = model_input.sequence_lengths;
    model_request.prefix_lengths       = model_input.prefix_lengths;
    model_request.attention_mask       = model_input.attention_mask;
    return model_request;
}

void EmbeddingExecutor::calcTokenNum(const list<EmbeddingStreamPtr>& streams,
                                     int64_t&                        token_num,
                                     int64_t&                        batch_size) const {
    token_num  = 0;
    batch_size = 0;
    for (auto& stream : streams) {
        token_num += stream->inputLength();
        batch_size += stream->batchSize();
    }
}

void EmbeddingExecutor::freeKVCacheAllocations(const std::vector<EmbeddingKVCacheAllocation>& allocations) const {
    if (!cache_manager_) {
        return;
    }
    for (const auto& allocation : allocations) {
        auto blocks = uniqueBlocks(allocation.row_block_indices);
        if (!blocks.empty()) {
            cache_manager_->free(blocks);
        }
    }
}

void EmbeddingExecutor::commitKVCacheRow(const EmbeddingKVCacheAllocation& allocation,
                                         int                               local_row_idx,
                                         int                               commit_len) const {
    if (!cache_manager_ || allocation.seq_size_per_block <= 0 || local_row_idx < 0
        || local_row_idx >= static_cast<int>(allocation.row_block_indices.size())
        || local_row_idx >= static_cast<int>(allocation.row_token_ids.size())
        || local_row_idx >= static_cast<int>(allocation.row_cache_keys.size())) {
        return;
    }

    commit_len = commit_len / allocation.seq_size_per_block * allocation.seq_size_per_block;
    if (commit_len <= 0) {
        return;
    }
    const int commit_blocks = commit_len / allocation.seq_size_per_block;
    const auto& row_blocks = allocation.row_block_indices[local_row_idx];
    if (commit_blocks <= 0 || commit_blocks > static_cast<int>(row_blocks.size())) {
        return;
    }

    const int token_len = std::min<int>(allocation.row_token_ids[local_row_idx].size(), commit_len + 1);
    std::vector<int32_t> tokens(allocation.row_token_ids[local_row_idx].begin(),
                                allocation.row_token_ids[local_row_idx].begin() + token_len);
    std::vector<int64_t> cache_keys(allocation.row_cache_keys[local_row_idx].begin(),
                                    allocation.row_cache_keys[local_row_idx].begin()
                                        + std::min<int>(commit_blocks,
                                                        allocation.row_cache_keys[local_row_idx].size()));
    std::vector<int32_t> blocks(row_blocks.begin(), row_blocks.begin() + commit_blocks);
    CacheManager::FreeInfo free_info(kEmbeddingCacheRequestId, tokens, cache_keys, blocks);
    cache_manager_->freeWithCache(free_info);
}

void EmbeddingExecutor::commitKVCacheAllocations(const std::vector<EmbeddingKVCacheAllocation>& allocations) const {
    if (!cache_manager_) {
        return;
    }

    for (const auto& allocation : allocations) {
        std::set<int> committed_blocks;
        auto mark_committed = [&](int local_row_idx, int commit_len) {
            commit_len = commit_len / allocation.seq_size_per_block * allocation.seq_size_per_block;
            const int commit_blocks = commit_len / allocation.seq_size_per_block;
            if (local_row_idx < 0 || local_row_idx >= static_cast<int>(allocation.row_block_indices.size())) {
                return;
            }
            const auto& row_blocks = allocation.row_block_indices[local_row_idx];
            for (int i = 0; i < commit_blocks && i < static_cast<int>(row_blocks.size()); ++i) {
                committed_blocks.insert(row_blocks[i]);
            }
        };

        if (embedding_kv_cache_commit_policy_ == kEmbeddingKVCacheCommitPolicyFullBlock) {
            if (allocation.prefix_group.in_batch_shared_len > 0) {
                RTP_LLM_LOG_WARNING("EMBEDDING_KV_CACHE_COMMIT_POLICY=full_block only commits the owner row for an "
                                    "in-batch shared-prefix group on this old CacheManager path.");
                commitKVCacheRow(allocation, 0, allocation.row_lengths[0]);
                mark_committed(0, allocation.row_lengths[0]);
            } else {
                for (size_t local_row = 0; local_row < allocation.row_lengths.size(); ++local_row) {
                    commitKVCacheRow(allocation, static_cast<int>(local_row), allocation.row_lengths[local_row]);
                    mark_committed(static_cast<int>(local_row), allocation.row_lengths[local_row]);
                }
            }
        } else if (allocation.prefix_group.in_batch_shared_len > 0) {
            commitKVCacheRow(allocation, 0, allocation.prefix_group.in_batch_shared_len);
            mark_committed(0, allocation.prefix_group.in_batch_shared_len);
        } else {
            for (size_t local_row = 0; local_row < allocation.row_lengths.size(); ++local_row) {
                commitKVCacheRow(allocation, static_cast<int>(local_row), allocation.row_lengths[local_row]);
                mark_committed(static_cast<int>(local_row), allocation.row_lengths[local_row]);
            }
        }

        std::vector<int> uncommitted_blocks;
        std::set<int>    seen_uncommitted;
        for (const auto& row_blocks : allocation.row_block_indices) {
            for (auto block : row_blocks) {
                if (committed_blocks.count(block) == 0 && seen_uncommitted.insert(block).second) {
                    uncommitted_blocks.push_back(block);
                }
            }
        }
        if (!uncommitted_blocks.empty()) {
            cache_manager_->free(uncommitted_blocks);
        }
    }
}

unique_ptr<GptModelOutputs> EmbeddingExecutor::copyResultToCPU(th::Tensor gpu_outputs) const {
    auto output     = std::make_unique<GptModelOutputs>();
    auto buffer_ptr = torchTensor2Buffer(gpu_outputs);
    output->hidden_states =
        device_->allocateBuffer({buffer_ptr->type(), buffer_ptr->shape(), rtp_llm::AllocationType::HOST}, {});
    device_->copy({*(output->hidden_states), *(buffer_ptr)});
    return output;
}

absl::Status EmbeddingExecutor::sliceTensor(py::object                           tensor,
                                            const std::list<EmbeddingStreamPtr>& streams,
                                            int                                  total_batch_size) const {
    auto gpu_tensors = py::cast<torch::Tensor>(tensor);
    auto cpu_tensors = gpu_tensors.cpu();
    if (total_batch_size != cpu_tensors.size(0)) {
        std::ostringstream error_msg;
        error_msg << "total batch size not equal to output tensor at dim 0: " << total_batch_size << " vs "
                  << cpu_tensors.size(0);
        return absl::InternalError(error_msg.str());
    }
    int index = 0;
    for (auto& stream : streams) {
        if (index + stream->batchSize() > cpu_tensors.size(0)) {
            std::ostringstream error_msg;
            error_msg << "current index exceed output tensor at dim 0: " << index << ":" << index + stream->batchSize()
                      << "tensor size: " << cpu_tensors.size(0);
            return absl::InternalError(error_msg.str());
        }
        torch::Tensor sliced_tensor = cpu_tensors.slice(0, index, index + stream->batchSize(), 1);
        stream->updateTensorOutput(sliced_tensor);
        index += stream->batchSize();
    }
    return absl::OkStatus();
}

absl::Status EmbeddingExecutor::slicePyList(py::object                           gpu_outputs,
                                            const std::list<EmbeddingStreamPtr>& streams,
                                            int                                  total_batch_size) const {
    auto output_list = py::cast<py::list>(gpu_outputs);
    if (total_batch_size != output_list.size()) {
        std::ostringstream error_msg;
        error_msg << "total batch size not equal to output list: " << total_batch_size << " vs " << output_list.size();
        return absl::InternalError(error_msg.str());
    }
    int index = 0;
    for (auto& stream : streams) {
        if (index + stream->batchSize() > output_list.size()) {
            std::ostringstream error_msg;
            error_msg << "current index exceed output list max index: " << index << ":" << index + stream->batchSize()
                      << "list size: " << output_list.size();
            return absl::InternalError(error_msg.str());
        }
        py::slice slice(index, index + stream->batchSize(), 1);
        auto      res = pyListToTensorMapVec(output_list[slice]);
        stream->updateMapOutput(res);
        index += stream->batchSize();
    }
    return absl::OkStatus();
}

// embedding postprocess output has two vaild values:
// 1. tensor, which shape is [total_batch_size, ...]
// 2. list<map<str, tensor>, which shape is [total_batch_size]
absl::Status EmbeddingExecutor::updateStreams(py::object                           post_process_output,
                                              const std::list<EmbeddingStreamPtr>& streams,
                                              int                                  total_batch_size) const {
    if (pybind11::isinstance<py::list>(post_process_output)) {
        return slicePyList(post_process_output, streams, total_batch_size);
        // need use python class type to check
    } else if (py::isinstance(post_process_output, torch_type_)) {
        return sliceTensor(post_process_output, streams, total_batch_size);
    } else {
        return absl::InternalError("unknown output type");
    }
}

absl::StatusOr<py::object> EmbeddingExecutor::postProcess(const ModelRequest&    model_request,
                                                          const GptModelOutputs& gpu_outputs) {
    using namespace HandlerArgs;

    try {
        py::dict kwargs;
        if (has_arg(handler_args_, Arg::INPUT_LENGTHS)) {
            kwargs[get_name(Arg::INPUT_LENGTHS)] = Buffer2torchTensor(model_request.input_lengths, false);
        }
        if (has_arg(handler_args_, Arg::HIDDEN_STATES)) {
            kwargs[get_name(Arg::HIDDEN_STATES)] = Buffer2torchTensor(gpu_outputs.all_hidden_states, false);
        }
        if (has_arg(handler_args_, Arg::INPUT_IDS)) {
            kwargs[get_name(Arg::INPUT_IDS)] = Buffer2torchTensor(model_request.combo_tokens, false);
        }
        if (has_arg(handler_args_, Arg::ATTENTION_MASK)) {
            kwargs[get_name(Arg::ATTENTION_MASK)] = py::none();  // mark to be generated by python
        }
        if (has_arg(handler_args_, Arg::MOE_GATING)) {
            py::list moe_gating;
            for (const auto& gating : gpu_outputs.moe_gating) {
                if (gating != nullptr) {
                    moe_gating.append(Buffer2torchTensor(gating, false));
                } else {
                    moe_gating.append(py::none());
                }
            }
            kwargs[get_name(Arg::MOE_GATING)] = moe_gating;
        }

        py::object output = handler_.attr("extend_forward")(**kwargs);
        return output;
    } catch (const exception& e) {
        return absl::InternalError("meet error when run handler " + std::string(e.what()));
    }
}

absl::Status EmbeddingExecutor::process(const std::list<EmbeddingStreamPtr>& streams) {
    CHECK_AND_RETURN_REF(embedding_model_input, gatherModelInput(streams));
    auto&           model_input = embedding_model_input.model_input;
    ScopeGuard      kv_cache_allocation_guard{[this, &embedding_model_input]() {
        freeKVCacheAllocations(embedding_model_input.kv_cache_allocations);
    }};
    auto            merged_output = std::make_unique<MergedOutput>();
    GptModelOutputs model_output;
    ModelRequest    model_request    = generateOldModelRequest(model_input);
    auto            total_batch_size = model_request.context_batch_size;
    model_output                     = std::move(model_->forward(model_input));
    py::gil_scoped_acquire acquire;
    // for py::list, handler should ensure object to cpu in the python impl,
    // for torch::Tensor, we manually move it to cpu during updateStreams()
    CHECK_AND_RETURN_REF(post, postProcess(model_request, model_output));
    auto res = updateStreams(post, streams, total_batch_size);
    if (res.ok()) {
        commitKVCacheAllocations(embedding_model_input.kv_cache_allocations);
        kv_cache_allocation_guard.dismiss();
    }
    return res;
}

void EmbeddingExecutor::reportMetrics(size_t context_batch_size, size_t combo_token_num, size_t max_seq_len) const {
    if (metrics_reporter_) {
        RtpLLMExecutorMetricsCollector collector;
        collector.context_batch_size  = context_batch_size;
        collector.generate_batch_size = 0;
        collector.execute_token_size  = combo_token_num;
        collector.max_seq_len         = max_seq_len;
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
