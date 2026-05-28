#include "ATen/ops/ones.h"
#include "c10/core/ScalarType.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingExecutor.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <ATen/TensorIndexing.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <numeric>
#include "rtp_llm/cpp/utils/DebugUtils.h"
using namespace std;
using namespace at::indexing;

namespace rtp_llm {

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

EmbeddingExecutor::EmbeddingExecutor(const EngineInitParams&         params,
                                     py::object                      handler,
                                     std::shared_ptr<KVCacheManager> cache_manager,
                                     int32_t                         kv_cache_group_num,
                                     std::vector<int32_t>            kv_cache_layer_to_group):
    handler_(handler),
    handler_args_(),
    metrics_reporter_(params.metrics_reporter),
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    eplb_config(params.eplb_config),
    cache_manager_(std::move(cache_manager)),
    kv_cache_group_num_(kv_cache_group_num),
    kv_cache_layer_to_group_(std::move(kv_cache_layer_to_group)) {
    if (cache_manager_) {
        cache_config_ = cache_manager_->cacheConfig();
    }

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
        model_config_.mla_ops_type,
        model_config_.max_seq_len,
        model_config_.hidden_size,
        model_config_.attn_config.tokens_per_block,
        model_config_.attn_config.kernel_tokens_per_block,
        kv_cache_group_num_,
        kv_cache_layer_to_group_,
        cache_manager_,
    });

    RTP_LLM_CHECK_WITH_INFO(!params.py_model.is_none(), "py_model must be provided, legacy C++ GptModel path removed");
    RTP_LLM_LOG_INFO("init executor with python model");
    model_.reset(new PyWrappedModel(model_init_params, params.py_model, true));

    init_position_ids(model_config_.max_seq_len);
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
}

void EmbeddingExecutor::init_position_ids(int max_seq_len) {
    max_position_ids_tensor_ = torch::arange(max_seq_len, torch::kInt32);
}

void EmbeddingExecutor::fillKVCacheMetadata(GptModelInputs& model_input, size_t max_blocks_num) const {
    if (!cache_manager_ || max_blocks_num == 0) {
        return;
    }
    auto i32_options = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    model_input.kv_cache_layer_to_group = torch::empty({(int64_t)kv_cache_layer_to_group_.size()}, i32_options);
    if (!kv_cache_layer_to_group_.empty()) {
        memcpy(model_input.kv_cache_layer_to_group.data_ptr<int32_t>(),
               kv_cache_layer_to_group_.data(),
               kv_cache_layer_to_group_.size() * sizeof(int32_t));
    }
    model_input.kv_cache_group_types = torch::empty({(int64_t)kv_cache_group_num_}, i32_options);
    auto* group_types                = model_input.kv_cache_group_types.data_ptr<int32_t>();
    for (int32_t gid = 0; gid < kv_cache_group_num_; ++gid) {
        auto type =
            cache_config_.group_types.empty() ? CacheGroupType::FULL : cache_config_.group_types[gid];
        group_types[gid] = static_cast<int32_t>(type);
    }
    model_input.kv_block_stride_bytes     = cache_config_.kv_block_stride_bytes;
    model_input.kv_scale_stride_bytes     = cache_config_.kv_scale_stride_bytes;
    model_input.seq_size_per_block        = cache_config_.seq_size_per_block;
    model_input.kernel_seq_size_per_block = cache_config_.kernel_seq_size_per_block;
}

namespace {

// Per-batch-slot kv-cache-block-id memcpy. Mirrors the layout that
// PyWrappedModel expects: [group, batch, max_blocks (× kernel_blocks_per_kv_block for kernel)].
void copyEmbeddingKVCacheBlocksToModelInput(GptModelInputs&             model_input,
                                            const BatchKVCacheResource& kv_cache,
                                            int                         stream_batch_idx,
                                            int                         model_batch_idx,
                                            size_t                      max_blocks_num,
                                            size_t                      kernel_blocks_per_kv_block) {
    if (!model_input.kv_cache_kernel_block_id.defined() || max_blocks_num == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_kernel_block_id.dim() == 3,
                            "embedding kv_cache_kernel_block_id must be 3-D");
    RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_id.dim() == 3, "embedding kv_cache_block_id must be 3-D");

    const size_t batch           = model_input.kv_cache_kernel_block_id.size(1);
    int32_t*     kernel_dst_base = model_input.kv_cache_kernel_block_id.data_ptr<int32_t>();
    int32_t*     store_dst_base  = model_input.kv_cache_block_id.data_ptr<int32_t>();

    for (int gid = 0; gid < kv_cache.groupNums(); ++gid) {
        const auto& kernel_blocks = kv_cache.kernelBlocks(stream_batch_idx, gid);
        int32_t*    kernel_dst    = kernel_dst_base
                                  + (static_cast<size_t>(gid) * batch + static_cast<size_t>(model_batch_idx))
                                        * max_blocks_num * kernel_blocks_per_kv_block;
        memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

        const auto& physical_blocks = kv_cache.blocks(stream_batch_idx, gid);
        int32_t*    store_dst       = store_dst_base
                                + (static_cast<size_t>(gid) * batch + static_cast<size_t>(model_batch_idx))
                                      * max_blocks_num;
        memcpy(store_dst, physical_blocks.data(), physical_blocks.size() * sizeof(int32_t));
    }
}

}  // namespace

absl::StatusOr<GptModelInputs> EmbeddingExecutor::gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const {
    int64_t token_num  = 0;
    int64_t batch_size = 0;
    calcTokenNum(streams, token_num, batch_size);
    GptModelInputs model_input;
    auto           i32_options = torch::TensorOptions(torch::kInt32).pinned_memory(true);

    // Determine whether ANY stream carries a kv_cache_resource (only the
    // synthetic prefix/suffix sub-streams built by processPrefixCacheStream do
    // — normal HTTP streams have nullptr kv_cache_resource and skip the paged
    // dispatch in Qwen3Model.forward via the has_blocks guard).
    size_t max_blocks_num = 0;
    for (auto& stream : streams) {
        const auto& input = stream->embeddingInput();
        if (input->kv_cache_resource) {
            max_blocks_num = std::max(max_blocks_num, (size_t)input->kv_cache_resource->curBlocksNum());
        }
    }

    model_input.combo_tokens          = torch::empty({token_num}, i32_options);
    model_input.combo_tokens_type_ids = torch::empty({token_num}, i32_options);
    model_input.combo_position_ids    = torch::empty({token_num}, i32_options);
    model_input.input_lengths         = torch::empty({batch_size}, i32_options);
    model_input.sequence_lengths      = torch::empty({0}, i32_options);
    // prefix_lengths stays per-batch; default 0, overwritten if a stream provides them.
    model_input.prefix_lengths        = torch::empty({batch_size}, i32_options);
    if (max_blocks_num > 0) {
        const auto kernel_blocks_per_kv_block = cache_config_.kernelBlocksPerKvBlock();
        model_input.kv_cache_kernel_block_id  = torch::zeros(
            {(int64_t)kv_cache_group_num_,
             (int64_t)batch_size,
             (int64_t)(max_blocks_num * kernel_blocks_per_kv_block)},
            i32_options);
        model_input.kv_cache_block_id = torch::zeros(
            {(int64_t)kv_cache_group_num_, (int64_t)batch_size, (int64_t)max_blocks_num}, i32_options);
        fillKVCacheMetadata(model_input, max_blocks_num);
    }

    int* merged_tokens         = model_input.combo_tokens.data_ptr<int>();
    int* input_lengths         = model_input.input_lengths.data_ptr<int>();
    int* prefix_lengths        = model_input.prefix_lengths.data_ptr<int>();
    int* merged_positon_ids    = model_input.combo_position_ids.data_ptr<int>();
    int* merged_token_type_ids = model_input.combo_tokens_type_ids.data_ptr<int>();
    int  token_idx             = 0;
    int  batch_idx             = 0;
    int  position_bias         = 0;
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

        auto embedding_input = stream->embeddingInput();
        if (embedding_input->input_embeddings.has_value()) {
            gathered_input_embeddings.emplace_back(embedding_input->input_embeddings.value().cpu());
            gathered_input_embeddings_locs.push_back(token_idx);
        }
        memcpy(merged_tokens + (int)token_idx, embedding_input->token_ids.data_ptr(), length * sizeof(int32_t));
        memcpy(merged_token_type_ids + (int)token_idx,
               embedding_input->token_type_ids.data_ptr(),
               length * sizeof(int32_t));
        memcpy(input_lengths + (int)batch_idx,
               embedding_input->input_lengths.data_ptr(),
               stream->batchSize() * sizeof(int32_t));
        if (embedding_input->prefix_lengths.defined() && embedding_input->prefix_lengths.numel() > 0) {
            RTP_LLM_CHECK_WITH_INFO(
                embedding_input->prefix_lengths.numel() == stream->batchSize(),
                "embedding_input->prefix_lengths.numel(%ld) != stream->batchSize(%d)",
                (long)embedding_input->prefix_lengths.numel(),
                stream->batchSize());
            memcpy(prefix_lengths + (int)batch_idx,
                   embedding_input->prefix_lengths.data_ptr(),
                   stream->batchSize() * sizeof(int32_t));
        } else {
            std::fill(prefix_lengths + (int)batch_idx,
                      prefix_lengths + (int)batch_idx + stream->batchSize(),
                      0);
        }
        int length_idx = 0;
        for (int i = 0; i < batchSize; i++) {
            int seqLen    = embedding_input->input_lengths.data_ptr<int32_t>()[i];
            int prefixLen = prefix_lengths[batch_idx + i];
            RTP_LLM_CHECK_WITH_INFO(seqLen + prefixLen + position_bias <= (int)max_position_ids_tensor_.size(0),
                                    "seqlen(%d) + prefix_len(%d) + position_bias(%d) exceed max_position_length(%d)",
                                    int(seqLen),
                                    int(prefixLen),
                                    int(position_bias),
                                    (int)max_position_ids_tensor_.size(0));
            // Suffix sub-stream of a prefix-kv-cache request starts its RoPE
            // positions at `prefixLen`, since logical positions [0, prefixLen)
            // were already consumed by the prefix forward.
            memcpy(merged_positon_ids + token_idx + length_idx,
                   max_position_ids_tensor_.data_ptr<int32_t>() + position_bias + prefixLen,
                   seqLen * sizeof(int32_t));
            if (embedding_input->kv_cache_resource) {
                copyEmbeddingKVCacheBlocksToModelInput(model_input,
                                                       *embedding_input->kv_cache_resource,
                                                       i,
                                                       batch_idx + i,
                                                       max_blocks_num,
                                                       cache_config_.kernelBlocksPerKvBlock());
            }
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

    size_t max_seq_len = 0;
    for (int i = 0; i < batch_size; ++i) {
        max_seq_len = std::max(max_seq_len, (size_t)(input_lengths[i] + prefix_lengths[i]));
    }
    if (HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::MOE_GATING)) {
        model_input.need_moe_gating = true;
    }
    reportMetrics(batch_size, token_num, max_seq_len);
    return model_input;
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

absl::Status EmbeddingExecutor::processNormal(const std::list<EmbeddingStreamPtr>& streams) {
    CHECK_AND_RETURN_REF(model_input, gatherModelInput(streams));
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
    model_->releaseBuffers();
    return res;
}

bool EmbeddingExecutor::shouldUsePrefixKVCache(const EmbeddingStreamPtr& stream) const {
    auto input = stream->embeddingInput();
    return cache_manager_ && input->enable_prefix_kv_cache && input->common_prefix_length > 0
           && !input->multimodal_features.has_value() && !input->input_embeddings.has_value()
           && stream->batchSize() > 1;
}

// Prefix KV cache split for shared-user-prefix Mainse-style ranker requests.
//
// Plan:
//   1. Allocate kv_resource for the prefix tokens only (batch=1) with
//      reuse_cache=true; run a forward pass on those tokens; insert the
//      resulting KV blocks into the system BlockCache (refcount keeps them
//      alive even after we free the prefix kv_resource).
//   2. Allocate kv_resource for the full N sequences with reuse_cache=true.
//      The cache_manager intra-batch reuse path (`initMallocForCommonLen`)
//      first hits the BlockCache for batch 0 (sharing the just-inserted
//      prefix blocks), then `reference()`s those block ids into batch 1..N-1
//      so all slots share the same physical prefix blocks.
//   3. Run a suffix forward (only the suffix tokens, with prefix_lengths=
//      reuse_len for every batch). The kernel reads K/V at positions
//      [0, reuse_len) from the shared prefix blocks and computes K/V at
//      positions [reuse_len, total_len) fresh.
//   4. Free both kv_resources. The prefix free was already done after
//      insertIntoCache; the suffix free decrements the cache's refcount
//      back to 1 (LRU-evictable when needed).
//
// Any failure in malloc/insertIntoCache falls back to processNormal so the
// request still completes correctly.
absl::Status EmbeddingExecutor::processPrefixCacheStream(const EmbeddingStreamPtr& stream) {
    auto input      = stream->embeddingInput();
    int  batch_size = (int)stream->batchSize();
    int  block_size = (int)cache_config_.seq_size_per_block;
    if (block_size <= 0) {
        return processNormal({stream});
    }
    int common_prefix_len = (int)input->common_prefix_length;
    int reuse_len         = common_prefix_len / block_size * block_size;
    if (reuse_len <= 0) {
        return processNormal({stream});
    }

    auto* token_ptr      = input->token_ids.data_ptr<int32_t>();
    auto* token_type_ptr = input->token_type_ids.data_ptr<int32_t>();
    auto* length_ptr     = input->input_lengths.data_ptr<int32_t>();

    std::vector<std::vector<int32_t>> full_token_rows;
    std::vector<std::vector<int32_t>> full_type_rows;
    full_token_rows.reserve(batch_size);
    full_type_rows.reserve(batch_size);
    int64_t offset = 0;
    for (int i = 0; i < batch_size; ++i) {
        int row_len = length_ptr[i];
        if (row_len < reuse_len) {
            return processNormal({stream});
        }
        full_token_rows.emplace_back(token_ptr + offset, token_ptr + offset + row_len);
        full_type_rows.emplace_back(token_type_ptr + offset, token_type_ptr + offset + row_len);
        offset += row_len;
    }

    const auto kernel_blocks_per_kv_block = cache_config_.kernelBlocksPerKvBlock();
    const auto group_types                = cache_config_.group_types;

    // ---------- Step 1: prefix forward (batch=1 over user tokens) ----------
    auto prefix_kv_resource = std::make_shared<BatchKVCacheResource>();
    prefix_kv_resource->resetBatchSize(1);
    prefix_kv_resource->initGroups(kv_cache_group_num_,
                                   cache_config_.layer_all_num,
                                   cache_config_.layer_to_group_id,
                                   kernel_blocks_per_kv_block,
                                   group_types);

    std::vector<int32_t> prefix_tokens(full_token_rows[0].begin(), full_token_rows[0].begin() + reuse_len);
    std::vector<int32_t> prefix_types(full_type_rows[0].begin(), full_type_rows[0].begin() + reuse_len);

    // CompleteTokenIds wraps a [batch, max_seq_len] int32 tensor; the
    // cache_manager reads token bytes from data(batch_idx) to compute the
    // rolling cache_keys. Use initFromRows() to allocate the backing buffer
    // and copy our prefix tokens — the default constructor only stores
    // metadata and leaves complete_token_ids_ undefined, which would fail
    // the size(1) check inside setSeqLength().
    auto prefix_complete_token_ids =
        std::make_shared<CompleteTokenIds>(/*batch_size=*/1, /*max_batch_size=*/1, reuse_len, block_size);
    prefix_complete_token_ids->initFromRows({prefix_tokens}, reuse_len);
    initCacheKeys(prefix_kv_resource, prefix_complete_token_ids, block_size);

    MallocInfo prefix_malloc_info;
    prefix_malloc_info.batch_kv_cache_resource = prefix_kv_resource;
    prefix_malloc_info.complete_token_ids      = prefix_complete_token_ids;
    prefix_malloc_info.request_id              = input->request_id;
    prefix_malloc_info.reuse_cache             = true;
    prefix_malloc_info.enable_device_cache     = true;
    auto prefix_malloc_result                  = cache_manager_->malloc(prefix_malloc_info);
    if (!prefix_malloc_result.success) {
        RTP_LLM_LOG_WARNING("[mainse-prefix] prefix malloc failed, fallback to normal embedding path");
        return processNormal({stream});
    }

    auto free_prefix = [&]() {
        FreeInfo info;
        info.batch_kv_cache_resource = prefix_kv_resource;
        info.complete_token_ids      = prefix_complete_token_ids;
        info.request_id              = input->request_id;
        cache_manager_->free(info);
    };

    // Build a 1-row sub-stream and run the prefix forward through processNormal's
    // helpers (gatherModelInput / forward / postProcess). We call model_->forward
    // directly since we don't need to run postProcess on the prefix.
    std::vector<int32_t> prefix_lengths_vec{reuse_len};
    auto prefix_input_obj =
        std::make_shared<EmbeddingInput>(prefix_tokens, prefix_types, prefix_lengths_vec, input->request_id);
    prefix_input_obj->prefix_lengths    = torch::zeros({1}, torch::kInt32);
    prefix_input_obj->kv_cache_resource = prefix_kv_resource;
    std::list<EmbeddingStreamPtr> prefix_streams{std::make_shared<EmbeddingStream>(prefix_input_obj)};

    auto prefix_status = gatherModelInput(prefix_streams);
    if (!prefix_status.ok()) {
        free_prefix();
        return prefix_status.status();
    }
    auto prefix_model_input = std::move(prefix_status.value());
    model_->releaseBuffers();
    RTP_LLM_LOG_INFO("[mainse-prefix] req=%ld start prefix forward batch=1 reuse_len=%d",
                     input->request_id, reuse_len);
    (void)model_->forward(prefix_model_input);
    RTP_LLM_LOG_INFO("[mainse-prefix] req=%ld prefix forward done", input->request_id);

    // ---------- Step 2: register the just-written prefix blocks in BlockCache ----------
    InsertInfo insert_info{prefix_kv_resource, prefix_complete_token_ids, /*is_resident=*/false};
    cache_manager_->insertIntoCache(insert_info);
    free_prefix();

    // ---------- Step 3: malloc full sequences (batch=N) with reuse_cache=true ----------
    auto suffix_kv_resource = std::make_shared<BatchKVCacheResource>();
    suffix_kv_resource->resetBatchSize(batch_size);
    suffix_kv_resource->initGroups(kv_cache_group_num_,
                                   cache_config_.layer_all_num,
                                   cache_config_.layer_to_group_id,
                                   kernel_blocks_per_kv_block,
                                   group_types);

    int max_row_len = 0;
    for (auto& row : full_token_rows) {
        max_row_len = std::max(max_row_len, (int)row.size());
    }
    auto suffix_complete_token_ids =
        std::make_shared<CompleteTokenIds>(batch_size, batch_size, max_row_len, block_size);
    // common_len = reuse_len so initMallocForCommonLen treats the first
    // reuse_len tokens as the shared-prefix portion to dedup across batches.
    suffix_complete_token_ids->initFromRows(full_token_rows, reuse_len);
    initCacheKeys(suffix_kv_resource, suffix_complete_token_ids, block_size);

    MallocInfo suffix_malloc_info;
    suffix_malloc_info.batch_kv_cache_resource = suffix_kv_resource;
    suffix_malloc_info.complete_token_ids      = suffix_complete_token_ids;
    suffix_malloc_info.request_id              = input->request_id;
    suffix_malloc_info.reuse_cache             = true;
    suffix_malloc_info.enable_device_cache     = true;
    auto suffix_malloc_result                  = cache_manager_->malloc(suffix_malloc_info);
    if (!suffix_malloc_result.success) {
        RTP_LLM_LOG_WARNING("[mainse-prefix] suffix malloc failed, fallback to normal embedding path");
        return processNormal({stream});
    }

    auto free_suffix = [&]() {
        FreeInfo info;
        info.batch_kv_cache_resource = suffix_kv_resource;
        info.complete_token_ids      = suffix_complete_token_ids;
        info.request_id              = input->request_id;
        cache_manager_->free(info);
    };

    // ---------- Step 4: suffix forward (only suffix tokens, prefix via cache) ----------
    std::vector<int32_t> suffix_tokens;
    std::vector<int32_t> suffix_types;
    std::vector<int32_t> suffix_input_lengths;
    std::vector<int32_t> suffix_prefix_lengths;
    suffix_input_lengths.reserve(batch_size);
    suffix_prefix_lengths.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        int suffix_len = (int)full_token_rows[i].size() - reuse_len;
        suffix_tokens.insert(
            suffix_tokens.end(), full_token_rows[i].begin() + reuse_len, full_token_rows[i].end());
        suffix_types.insert(
            suffix_types.end(), full_type_rows[i].begin() + reuse_len, full_type_rows[i].end());
        suffix_input_lengths.push_back(suffix_len);
        suffix_prefix_lengths.push_back(reuse_len);
    }
    auto suffix_input_obj =
        std::make_shared<EmbeddingInput>(suffix_tokens, suffix_types, suffix_input_lengths, input->request_id);
    suffix_input_obj->prefix_lengths    = torch::from_blob(suffix_prefix_lengths.data(),
                                                        {(int64_t)suffix_prefix_lengths.size()},
                                                        torch::kInt32)
                                           .clone();
    suffix_input_obj->kv_cache_resource = suffix_kv_resource;
    std::list<EmbeddingStreamPtr> suffix_streams{std::make_shared<EmbeddingStream>(suffix_input_obj)};

    auto suffix_status = gatherModelInput(suffix_streams);
    if (!suffix_status.ok()) {
        free_suffix();
        return suffix_status.status();
    }
    auto suffix_model_input = std::move(suffix_status.value());
    auto model_request      = generateOldModelRequest(suffix_model_input);
    auto total_batch_size   = model_request.context_batch_size;
    model_->releaseBuffers();
    int max_suffix = 0;
    for (auto v : suffix_input_lengths) max_suffix = std::max(max_suffix, v);
    RTP_LLM_LOG_INFO("[mainse-prefix] req=%ld start suffix forward batch=%d max_suffix_len=%d "
                     "reuse_len=%d suffix_malloc_reuse_len=%d",
                     input->request_id, batch_size, max_suffix, reuse_len,
                     suffix_malloc_result.reuse_len);
    auto model_output = std::move(model_->forward(suffix_model_input));
    RTP_LLM_LOG_INFO("[mainse-prefix] req=%ld suffix forward done", input->request_id);

    py::gil_scoped_acquire acquire;
    auto                   post_status = postProcess(model_request, model_output);
    if (!post_status.ok()) {
        free_suffix();
        model_->releaseBuffers();
        return post_status.status();
    }
    // updateStreams expects the original user-facing stream (not the synthetic
    // suffix sub-stream) so the per-item output tensors land on the right
    // EmbeddingOutput. The post-process op shapes `[batch_size, output_num]`
    // already matches the synthetic batch which is the same N items.
    auto res = updateStreams(post_status.value(), {stream}, total_batch_size);
    free_suffix();
    model_->releaseBuffers();
    return res;
}

absl::Status EmbeddingExecutor::process(const std::list<EmbeddingStreamPtr>& streams) {
    // Group streams: ones eligible for prefix-kv-cache split run one-at-a-time
    // through processPrefixCacheStream; the rest are batched through
    // processNormal preserving previous batching behaviour.
    std::list<EmbeddingStreamPtr> normal_streams;
    for (auto& stream : streams) {
        if (shouldUsePrefixKVCache(stream)) {
            if (!normal_streams.empty()) {
                auto status = processNormal(normal_streams);
                if (!status.ok()) {
                    return status;
                }
                normal_streams.clear();
            }
            auto status = processPrefixCacheStream(stream);
            if (!status.ok()) {
                return status;
            }
        } else {
            normal_streams.push_back(stream);
        }
    }
    if (!normal_streams.empty()) {
        return processNormal(normal_streams);
    }
    return absl::OkStatus();
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
