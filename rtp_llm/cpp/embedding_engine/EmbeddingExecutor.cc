#include "ATen/ops/ones.h"
#include "c10/core/ScalarType.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingExecutor.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include <ATen/TensorIndexing.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <functional>
#include <numeric>
#include "rtp_llm/cpp/utils/DebugUtils.h"
using namespace std;
using namespace at::indexing;

namespace rtp_llm {

namespace {

constexpr int64_t kEmbeddingCacheRequestId = 0;

std::vector<int32_t> toInt32Vector(const std::vector<int>& values) {
    std::vector<int32_t> result;
    result.reserve(values.size());
    for (auto value : values) {
        result.push_back(static_cast<int32_t>(value));
    }
    return result;
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

struct ScopeGuard {
    std::function<void()> fn;
    ~ScopeGuard() {
        if (fn) {
            fn();
        }
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

EmbeddingExecutor::EmbeddingExecutor(const EngineInitParams&                params,
                                     py::object                            handler,
                                     const std::shared_ptr<KVCacheManager>& cache_manager):
    handler_(handler),
    handler_args_(),
    metrics_reporter_(params.metrics_reporter),
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    eplb_config(params.eplb_config),
    runtime_config_(params.runtime_config),
    cache_manager_(cache_manager) {
    RTP_LLM_CHECK_WITH_INFO(isValidEmbeddingKVCacheMode(runtime_config_.embedding_kv_cache_mode),
                            "invalid EMBEDDING_KV_CACHE_MODE: %s",
                            runtime_config_.embedding_kv_cache_mode.c_str());
    RTP_LLM_CHECK_WITH_INFO(isValidEmbeddingKVCacheCommitPolicy(runtime_config_.embedding_kv_cache_commit_policy),
                            "invalid EMBEDDING_KV_CACHE_COMMIT_POLICY: %s",
                            runtime_config_.embedding_kv_cache_commit_policy.c_str());
    enable_embedding_kv_cache_ =
        cache_manager_ && (runtime_config_.embedding_kv_cache_mode == kEmbeddingKVCacheModeBlock
                           || runtime_config_.embedding_kv_cache_mode == kEmbeddingKVCacheModeInBatch);
    enable_in_batch_kv_prefix_dedup_ =
        enable_embedding_kv_cache_ && runtime_config_.embedding_kv_cache_mode == kEmbeddingKVCacheModeInBatch;

    CacheConfig empty_cache_config;
    const auto& cache_config = cache_manager_ ? cache_manager_->cacheConfig() : empty_cache_config;
    auto        kv_cache_layer_to_group =
        cache_manager_ ? toInt32Vector(cache_config.layer_to_group_id) : std::vector<int32_t>();
    GptModelInitParams model_init_params({
        params.gpt_weights,
        Executor::genModelDescription(model_config_, parallelism_config, eplb_config, params.moe_config),
        cache_manager_ ? std::make_optional(cache_manager_->getMainModelCacheLayerLayout()) : std::nullopt,
        params.model_id,
        parallelism_config,
        params.hw_kernel_config,
        params.profiling_debug_logging_config,
        params.runtime_config,
        params.concurrency_config,
        params.sp_config,
        params.device_resource_config,
        params.model_config_.mla_ops_type,
        params.model_config_.max_seq_len,
        params.model_config_.hidden_size,
        cache_manager_ ? cache_config.seq_size_per_block : params.model_config_.attn_config.tokens_per_block,
        cache_manager_ ? cache_config.kernel_seq_size_per_block :
                         params.model_config_.attn_config.kernel_tokens_per_block,
        cache_manager_ ? cache_config.groupNums() : 1,
        kv_cache_layer_to_group,
        cache_manager_,
    });

    RTP_LLM_CHECK_WITH_INFO(!params.py_model.is_none(), "py_model must be provided, legacy C++ GptModel path removed");
    RTP_LLM_LOG_INFO("init executor with python model");
    model_.reset(new PyWrappedModel(model_init_params, params.py_model, true));

    init_position_ids(model_config_.max_seq_len);

    // Pre-cache constant pinned tensors for kv_cache_layer_to_group and kv_cache_group_types
    if (cache_manager_) {
        auto layer_to_group_vec = toInt32Vector(cache_config.layer_to_group_id);
        cached_kv_cache_layer_to_group_ =
            torch::from_blob(
                layer_to_group_vec.data(), {static_cast<int64_t>(layer_to_group_vec.size())}, torch::kInt32)
                .clone()
                .pin_memory();

        std::vector<int32_t> group_types_vec;
        group_types_vec.reserve(cache_config.group_types.size());
        for (auto group_type : cache_config.group_types) {
            group_types_vec.push_back(static_cast<int32_t>(group_type));
        }
        cached_kv_cache_group_types_ =
            torch::from_blob(
                group_types_vec.data(), {static_cast<int64_t>(group_types_vec.size())}, torch::kInt32)
                .clone()
                .pin_memory();
    }

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
    max_position_ids_tensor_ = torch::arange(max_seq_len, torch::kInt32);
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
    auto           i32_options = torch::TensorOptions(torch::kInt32).pinned_memory(true);

    model_input.combo_tokens          = torch::empty({static_cast<int64_t>(token_num)}, i32_options);
    model_input.combo_tokens_type_ids = torch::empty({static_cast<int64_t>(token_num)}, i32_options);
    model_input.combo_position_ids    = torch::empty({static_cast<int64_t>(token_num)}, i32_options);
    model_input.input_lengths         = torch::empty({static_cast<int64_t>(batch_size)}, i32_options);
    model_input.sequence_lengths      = torch::empty({0}, i32_options);
    model_input.prefix_lengths        = torch::zeros({static_cast<int64_t>(batch_size)}, i32_options);
    int* merged_tokens                = model_input.combo_tokens.data_ptr<int>();
    int* input_lengths                = model_input.input_lengths.data_ptr<int>();
    int* merged_positon_ids           = model_input.combo_position_ids.data_ptr<int>();
    int* merged_token_type_ids        = model_input.combo_tokens_type_ids.data_ptr<int>();
    int  token_idx                    = 0;
    int  batch_idx                    = 0;
    int  position_bias                = 0;
    if (model_config_.position_ids_style == 1) {
        position_bias = model_config_.special_tokens.pad_token_id + 1;
    }

    std::vector<torch::Tensor> gathered_mm_features;
    std::vector<int>           new_locs;
    std::vector<int>           merged_text_mask;
    std::vector<torch::Tensor> gathered_input_embeddings;
    std::vector<int>           gathered_input_embeddings_locs;
    merged_text_mask.resize(token_num, 1);
    for (auto& stream : streams) {
        int         length     = stream->inputLength();
        int         batchSize  = stream->batchSize();
        const auto& mm_feature = stream->multimodalFeature();
        if (mm_feature.has_value()) {
            for (const auto& feature : mm_feature.value().features) {
                gathered_mm_features.emplace_back(feature);
            }
            const auto mm_locs      = mm_feature.value().locs;
            auto       mm_locs_data = mm_locs.data_ptr<int>();
            for (int i = 0; i < mm_locs.numel(); ++i) {
                new_locs.push_back(mm_locs_data[i] + token_idx);
            }
            const auto text_token_mask = mm_feature.value().text_tokens_mask;
            memcpy(merged_text_mask.data() + token_idx,
                   text_token_mask.data_ptr<int>(),
                   text_token_mask.numel() * sizeof(int));
        }

        if (stream->embeddingInput()->input_embeddings.has_value()) {
            gathered_input_embeddings.emplace_back(stream->embeddingInput()->input_embeddings.value().cpu());
            gathered_input_embeddings_locs.push_back(token_idx);
        }
        memcpy(
            merged_tokens + (int)token_idx, stream->embeddingInput()->token_ids.data_ptr(), length * sizeof(int32_t));
        memcpy(merged_token_type_ids + (int)token_idx,
               stream->embeddingInput()->token_type_ids.data_ptr(),
               length * sizeof(int32_t));
        memcpy(input_lengths + (int)batch_idx,
               stream->embeddingInput()->input_lengths.data_ptr(),
               stream->batchSize() * sizeof(int32_t));
        int length_idx = 0;
        for (int i = 0; i < batchSize; i++) {
            int seqLen = stream->embeddingInput()->input_lengths.data_ptr<int32_t>()[i];
            RTP_LLM_CHECK_WITH_INFO(seqLen + position_bias <= (int)max_position_ids_tensor_.size(0),
                                    "seqlen(%d) + position_bias(%d) exceed max_position_length(%d)",
                                    int(seqLen),
                                    int(position_bias),
                                    (int)max_position_ids_tensor_.size(0));
            memcpy(merged_positon_ids + token_idx + length_idx,
                   max_position_ids_tensor_.data_ptr<int32_t>() + position_bias,
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
        model_input.mm_features_locs =
            torch::from_blob(new_locs.data(), {(int64_t)new_locs.size()}, torch::kInt32).clone();
        model_input.text_tokens_mask =
            torch::from_blob(merged_text_mask.data(), {(int64_t)merged_text_mask.size()}, torch::kInt32).clone();
    }

    if (!gathered_input_embeddings.empty()) {
        model_input.input_embeddings      = std::move(gathered_input_embeddings);
        model_input.input_embeddings_locs = torch::from_blob(gathered_input_embeddings_locs.data(),
                                                             {(int64_t)gathered_input_embeddings_locs.size()},
                                                             torch::kInt32)
                                                .clone();
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
        int                token_offset = 0;
        int                length = 0;
    };

    const auto& cache_config = cache_manager_->cacheConfig();
    const int   seq_size_per_block = static_cast<int>(cache_config.seq_size_per_block);
    const int   kernel_blocks_per_kv_block = static_cast<int>(cache_config.kernelBlocksPerKvBlock());
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be positive");

    int64_t legacy_token_num = 0;
    int64_t batch_size = 0;
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

    std::vector<RowRef> rows;
    rows.reserve(batch_size);
    std::vector<EmbeddingGatherRowView> row_views;
    row_views.reserve(batch_size);
    std::vector<EmbeddingGatherRowRange> row_ranges;
    row_ranges.reserve(streams.size());
    for (const auto& stream : streams) {
        int   offset = 0;
        const int range_begin = static_cast<int>(rows.size());
        auto* input_lengths = stream->embeddingInput()->input_lengths.data_ptr<int32_t>();
        for (int local_row = 0; local_row < stream->batchSize(); ++local_row) {
            const int length = input_lengths[local_row];
            rows.push_back(RowRef{stream, local_row, offset, length});
            row_views.push_back(EmbeddingGatherRowView{
                stream->embeddingInput()->token_ids.data_ptr<int32_t>() + offset,
                stream->embeddingInput()->token_type_ids.data_ptr<int32_t>() + offset,
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

    int     max_original_len = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        max_original_len = std::max(max_original_len, rows[i].length);
    }

    auto i32_options = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    EmbeddingModelInput result;
    result.model_input.sequence_lengths          = torch::empty({0}, i32_options);
    result.model_input.seq_size_per_block        = cache_config.seq_size_per_block;
    result.model_input.kernel_seq_size_per_block = cache_config.kernel_seq_size_per_block;
    result.model_input.kv_block_stride_bytes     = cache_config.kv_block_stride_bytes;
    result.model_input.kv_scale_stride_bytes     = cache_config.kv_scale_stride_bytes;

    const int max_blocks_num = (max_original_len + seq_size_per_block - 1) / seq_size_per_block;
    if (max_blocks_num > 0) {
        result.model_input.kv_cache_kernel_block_id =
            torch::zeros({static_cast<int64_t>(cache_config.groupNums()),
                          static_cast<int64_t>(batch_size),
                          static_cast<int64_t>(max_blocks_num * kernel_blocks_per_kv_block)},
                         i32_options);
        result.model_input.kv_cache_block_id = torch::zeros({static_cast<int64_t>(cache_config.groupNums()),
                                                             static_cast<int64_t>(batch_size),
                                                             static_cast<int64_t>(max_blocks_num)},
                                                            i32_options);

        result.model_input.kv_cache_layer_to_group = cached_kv_cache_layer_to_group_;
        result.model_input.kv_cache_group_types    = cached_kv_cache_group_types_;
    }

    auto copy_blocks_to_model_input = [&](const BatchKVCacheResourcePtr& resource,
                                          int                            local_row_idx,
                                          int                            model_row_idx) {
        if (max_blocks_num <= 0) {
            return;
        }
        for (int gid = 0; gid < resource->groupNums(); ++gid) {
            const auto& kernel_blocks = resource->kernelBlocks(local_row_idx, gid);
            auto* kernel_dst = result.model_input.kv_cache_kernel_block_id.data_ptr<int32_t>()
                               + (static_cast<size_t>(gid) * batch_size + static_cast<size_t>(model_row_idx))
                                     * max_blocks_num * kernel_blocks_per_kv_block;
            memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

            const auto& blocks = resource->blocks(local_row_idx, gid);
            auto* block_dst = result.model_input.kv_cache_block_id.data_ptr<int32_t>()
                              + (static_cast<size_t>(gid) * batch_size + static_cast<size_t>(model_row_idx))
                                    * max_blocks_num;
            memcpy(block_dst, blocks.data(), blocks.size() * sizeof(int32_t));
        }
    };

    for (size_t group_idx = 0; group_idx < layout.prefix_groups.size(); ++group_idx) {
        const auto& group = layout.prefix_groups[group_idx];
        int max_group_len = 0;
        for (auto row_idx : group.row_indices) {
            max_group_len = std::max(max_group_len, rows[row_idx].length);
        }
        auto generate_input = std::make_shared<GenerateInput>();
        // Shared groups allocate in two stages: first expose only the aligned shared prefix to cache
        // matching/common-block allocation, then expand to each row's full suffix blocks.
        const int init_len = group.in_batch_shared_len > 0 ? group.in_batch_shared_len : max_group_len;
        generate_input->generate_config = std::make_shared<GenerateConfig>();
        generate_input->input_ids = torch::empty({static_cast<int64_t>(init_len)}, torch::kInt32);
        if (init_len > 0) {
            const auto& owner = rows[group.owner_row_idx];
            memcpy(generate_input->input_ids.data_ptr<int32_t>(),
                   owner.stream->embeddingInput()->token_ids.data_ptr<int32_t>() + owner.token_offset,
                   init_len * sizeof(int32_t));
        }

        auto complete_token_ids = std::make_shared<CompleteTokenIds>(
            group.row_indices.size(), group.row_indices.size(), std::max(1, max_group_len), seq_size_per_block);
        complete_token_ids->init(generate_input);
        auto copy_full_rows_to_complete_token_ids = [&]() {
            for (size_t local_row = 0; local_row < group.row_indices.size(); ++local_row) {
                const auto& row = rows[group.row_indices[local_row]];
                memcpy(complete_token_ids->data(local_row),
                       row.stream->embeddingInput()->token_ids.data_ptr<int32_t>() + row.token_offset,
                       row.length * sizeof(int32_t));
            }
        };
        if (group.in_batch_shared_len == 0) {
            copy_full_rows_to_complete_token_ids();
            complete_token_ids->setSeqLength(max_group_len);
        }

        auto batch_kv_cache_resource = std::make_shared<BatchKVCacheResource>();
        batch_kv_cache_resource->resetBatchSize(group.row_indices.size());
        batch_kv_cache_resource->initGroups(cache_config.groupNums(),
                                            cache_config.layer_all_num,
                                            cache_config.layer_to_group_id,
                                            cache_config.kernelBlocksPerKvBlock(),
                                            cache_config.group_types);
        MallocInfo malloc_info;
        malloc_info.batch_kv_cache_resource = batch_kv_cache_resource;
        malloc_info.complete_token_ids = complete_token_ids;
        malloc_info.request_id = kEmbeddingCacheRequestId;
        malloc_info.reuse_cache = true;
        malloc_info.enable_device_cache = true;
        malloc_info.enable_remove_skipped_blocks = false;

        auto malloc_result = cache_manager_->malloc(malloc_info);
        if (!malloc_result.success) {
            freeKVCacheAllocations(result.kv_cache_allocations);
            return absl::ResourceExhaustedError("embedding kv cache malloc failed");
        }

        if (group.in_batch_shared_len > 0) {
            copy_full_rows_to_complete_token_ids();
            complete_token_ids->setSeqLength(max_group_len);
            auto suffix_malloc_result = cache_manager_->malloc(malloc_info);
            if (!suffix_malloc_result.success) {
                cache_manager_->free(FreeInfo{batch_kv_cache_resource, complete_token_ids, kEmbeddingCacheRequestId});
                freeKVCacheAllocations(result.kv_cache_allocations);
                return absl::ResourceExhaustedError("embedding kv cache malloc failed");
            }
        }

        applyEmbeddingKVReuse(layout, row_views, group_idx, malloc_result.reuse_len);

        for (size_t local_row = 0; local_row < group.row_indices.size(); ++local_row) {
            copy_blocks_to_model_input(batch_kv_cache_resource, local_row, group.row_indices[local_row]);
        }
        std::vector<int> row_lengths;
        row_lengths.reserve(group.row_indices.size());
        for (auto row_idx : group.row_indices) {
            row_lengths.push_back(rows[row_idx].length);
        }
        result.kv_cache_allocations.push_back(
            {batch_kv_cache_resource, complete_token_ids, layout.prefix_groups[group_idx], row_lengths,
             seq_size_per_block});
    }

    int64_t token_num = 0;
    int     max_compute_len = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        token_num += layout.rows[i].compute_len;
        max_compute_len = std::max(max_compute_len, layout.rows[i].compute_len);
    }

    result.model_input.combo_tokens          = torch::empty({static_cast<int64_t>(token_num)}, i32_options);
    result.model_input.combo_tokens_type_ids = torch::empty({static_cast<int64_t>(token_num)}, i32_options);
    result.model_input.combo_position_ids    = torch::empty({static_cast<int64_t>(token_num)}, i32_options);
    result.model_input.input_lengths         = torch::empty({static_cast<int64_t>(batch_size)}, i32_options);
    result.model_input.prefix_lengths        = torch::zeros({static_cast<int64_t>(batch_size)}, i32_options);

    int32_t* merged_tokens         = result.model_input.combo_tokens.data_ptr<int32_t>();
    int32_t* merged_token_type_ids = result.model_input.combo_tokens_type_ids.data_ptr<int32_t>();
    int32_t* merged_position_ids   = result.model_input.combo_position_ids.data_ptr<int32_t>();
    int32_t* input_lengths         = result.model_input.input_lengths.data_ptr<int32_t>();
    int32_t* prefix_lengths        = result.model_input.prefix_lengths.data_ptr<int32_t>();

    int position_bias = 0;
    if (model_config_.position_ids_style == 1) {
        position_bias = model_config_.special_tokens.pad_token_id + 1;
    }

    int token_idx = 0;
    for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
        const auto& row = rows[row_idx];
        const auto& row_layout = layout.rows[row_idx];
        RTP_LLM_CHECK_WITH_INFO(row.length + position_bias <= (int)max_position_ids_tensor_.size(0),
                                "seqlen(%d) + position_bias(%d) exceed max_position_length(%d)",
                                row.length,
                                position_bias,
                                (int)max_position_ids_tensor_.size(0));
        auto* src_tokens = row.stream->embeddingInput()->token_ids.data_ptr<int32_t>() + row.token_offset
                           + row_layout.compute_begin;
        auto* src_types = row.stream->embeddingInput()->token_type_ids.data_ptr<int32_t>() + row.token_offset
                          + row_layout.compute_begin;
        memcpy(merged_tokens + token_idx, src_tokens, row_layout.compute_len * sizeof(int32_t));
        memcpy(merged_token_type_ids + token_idx, src_types, row_layout.compute_len * sizeof(int32_t));
        memcpy(merged_position_ids + token_idx,
               max_position_ids_tensor_.data_ptr<int32_t>() + position_bias + row_layout.compute_begin,
               row_layout.compute_len * sizeof(int32_t));
        input_lengths[row_idx] = row_layout.compute_len;
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
    model_request.context_batch_size   = model_input.input_lengths.size(0);
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
        if (allocation.batch_kv_cache_resource && allocation.complete_token_ids) {
            cache_manager_->free(
                FreeInfo{allocation.batch_kv_cache_resource, allocation.complete_token_ids, kEmbeddingCacheRequestId});
        }
    }
}

void EmbeddingExecutor::commitKVCacheRow(const EmbeddingKVCacheAllocation& allocation,
                                         int                               local_row_idx,
                                         int                               commit_len) const {
    if (!cache_manager_ || !allocation.batch_kv_cache_resource || !allocation.complete_token_ids
        || allocation.seq_size_per_block <= 0) {
        return;
    }

    commit_len = commit_len / allocation.seq_size_per_block * allocation.seq_size_per_block;
    if (commit_len <= 0) {
        return;
    }
    const int commit_blocks = commit_len / allocation.seq_size_per_block;
    if (commit_blocks <= 0 || local_row_idx < 0
        || local_row_idx >= allocation.batch_kv_cache_resource->batchSize()) {
        return;
    }

    auto row_resource = std::make_shared<BatchKVCacheResource>();
    row_resource->resetBatchSize(1);
    const auto& cache_config = cache_manager_->cacheConfig();
    row_resource->initGroups(cache_config.groupNums(),
                             cache_config.layer_all_num,
                             cache_config.layer_to_group_id,
                             cache_config.kernelBlocksPerKvBlock(),
                             cache_config.group_types);
    for (int gid = 0; gid < allocation.batch_kv_cache_resource->groupNums(); ++gid) {
        const auto& blocks = allocation.batch_kv_cache_resource->blocks(local_row_idx, gid);
        if (static_cast<int>(blocks.size()) < commit_blocks) {
            return;
        }
        row_resource->setBatchBlocks(0, gid, BlockIndicesType(blocks.begin(), blocks.begin() + commit_blocks));
    }

    auto generate_input = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->input_ids       = torch::empty({static_cast<int64_t>(commit_len)}, torch::kInt32);
    if (commit_len > 0) {
        memcpy(generate_input->input_ids.data_ptr<int32_t>(),
               allocation.complete_token_ids->data(local_row_idx),
               commit_len * sizeof(int32_t));
    }
    auto row_complete_token_ids =
        std::make_shared<CompleteTokenIds>(1, 1, std::max(1, commit_len), allocation.seq_size_per_block);
    row_complete_token_ids->init(generate_input);
    initCacheKeys(row_resource, row_complete_token_ids, allocation.seq_size_per_block);
    row_resource->setLastBlockAligned(true);

    cache_manager_->insertIntoCache(InsertInfo{row_resource, row_complete_token_ids, false});
}

void EmbeddingExecutor::commitKVCacheAllocations(const std::vector<EmbeddingKVCacheAllocation>& allocations) const {
    if (!cache_manager_) {
        return;
    }

    for (const auto& allocation : allocations) {
        if (runtime_config_.embedding_kv_cache_commit_policy == kEmbeddingKVCacheCommitPolicyFullBlock) {
            for (size_t local_row = 0; local_row < allocation.row_lengths.size(); ++local_row) {
                commitKVCacheRow(allocation, static_cast<int>(local_row), allocation.row_lengths[local_row]);
            }
            continue;
        }

        if (allocation.prefix_group.in_batch_shared_len > 0) {
            commitKVCacheRow(allocation, 0, allocation.prefix_group.in_batch_shared_len);
        } else {
            for (size_t local_row = 0; local_row < allocation.row_lengths.size(); ++local_row) {
                commitKVCacheRow(allocation, static_cast<int>(local_row), allocation.row_lengths[local_row]);
            }
        }
    }
}

unique_ptr<GptModelOutputs> EmbeddingExecutor::copyResultToCPU(th::Tensor gpu_outputs) const {
    auto output           = std::make_unique<GptModelOutputs>();
    output->hidden_states = gpu_outputs.cpu();
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
            kwargs[get_name(Arg::INPUT_LENGTHS)] = model_request.input_lengths;
        }
        if (has_arg(handler_args_, Arg::HIDDEN_STATES)) {
            kwargs[get_name(Arg::HIDDEN_STATES)] = gpu_outputs.all_hidden_states;
        }
        if (has_arg(handler_args_, Arg::INPUT_IDS)) {
            kwargs[get_name(Arg::INPUT_IDS)] = model_request.combo_tokens;
        }
        if (has_arg(handler_args_, Arg::ATTENTION_MASK)) {
            kwargs[get_name(Arg::ATTENTION_MASK)] = py::none();  // mark to be generated by python
        }
        if (has_arg(handler_args_, Arg::MOE_GATING)) {
            py::list moe_gating;
            for (const auto& gating : gpu_outputs.moe_gating) {
                if (gating.defined()) {
                    moe_gating.append(gating);
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
    ScopeGuard kv_cache_allocation_guard{[this, &embedding_model_input]() {
        freeKVCacheAllocations(embedding_model_input.kv_cache_allocations);
    }};
    auto            merged_output = std::make_unique<MergedOutput>();
    GptModelOutputs model_output;
    ModelRequest    model_request    = generateOldModelRequest(model_input);
    auto            total_batch_size = model_request.context_batch_size;
    model_->releaseBuffers();
    model_output = std::move(model_->forward(model_input));
    py::gil_scoped_acquire acquire;
    // for py::list, handler should ensure object to cpu in the python impl,
    // for torch::Tensor, we manually move it to cpu during updateStreams()
    CHECK_AND_RETURN_REF(post, postProcess(model_request, model_output));
    auto res = updateStreams(post, streams, total_batch_size);
    if (res.ok()) {
        commitKVCacheAllocations(embedding_model_input.kv_cache_allocations);
    }
    model_->releaseBuffers();
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
