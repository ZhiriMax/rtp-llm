#pragma once

#include <memory>
#include <bitset>
#include <torch/python.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/all.h>
#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/embedding_engine/ModelRequest.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"

namespace rtp_llm {

namespace HandlerArgs {

enum class Arg : uint32_t {
    INPUT_LENGTHS,
    HIDDEN_STATES,
    INPUT_IDS,
    ATTENTION_MASK,
    MOE_GATING,
    // reserve as number marker
    NUM_INPUT_TYPES
};

constexpr size_t NUM_INPUT_TYPES = static_cast<size_t>(Arg::NUM_INPUT_TYPES);

using Flag = std::bitset<NUM_INPUT_TYPES>;

}  // namespace HandlerArgs

class EmbeddingExecutor {
public:
    EmbeddingExecutor(const EngineInitParams&          params,
                      py::object                       handler,
                      std::shared_ptr<KVCacheManager>  cache_manager           = nullptr,
                      int32_t                          kv_cache_group_num      = 1,
                      std::vector<int32_t>             kv_cache_layer_to_group = {});

    absl::Status process(const std::list<EmbeddingStreamPtr>& streams);

private:
    std::unique_ptr<ModelBase>      model_;
    py::object                      handler_;
    HandlerArgs::Flag               handler_args_;
    py::handle                      torch_type_;
    torch::Tensor                   max_position_ids_tensor_;
    kmonitor::MetricsReporterPtr    metrics_reporter_ = nullptr;
    ModelConfig                     model_config_;
    ParallelismConfig               parallelism_config;
    EPLBConfig                      eplb_config;
    std::shared_ptr<KVCacheManager> cache_manager_;
    CacheConfig                     cache_config_;
    int32_t                         kv_cache_group_num_{1};
    std::vector<int32_t>            kv_cache_layer_to_group_;

    ModelRequest                     generateOldModelRequest(GptModelInputs& model_input);
    absl::StatusOr<GptModelInputs>   gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const;
    std::unique_ptr<GptModelOutputs> copyResultToCPU(th::Tensor gpu_outputs) const;
    absl::Status                     updateStreams(py::object                           post_process_output,
                                                   const std::list<EmbeddingStreamPtr>& streams,
                                                   int                                  total_batch_size) const;
    absl::Status
    sliceTensor(py::object gpu_outputs, const std::list<EmbeddingStreamPtr>& streams, int total_batch_size) const;
    absl::Status
    slicePyList(py::object gpu_outputs, const std::list<EmbeddingStreamPtr>& streams, int total_batch_size) const;
    absl::StatusOr<py::object> postProcess(const ModelRequest& model_request, const GptModelOutputs& gpu_outputs);
    void calcTokenNum(const std::list<EmbeddingStreamPtr>& streams, int64_t& token_num, int64_t& batch_size) const;
    void init_position_ids(int max_seq_len);
    void reportMetrics(size_t context_batch_size, size_t combo_token_num, size_t max_seq_len) const;
    // Helper: fill the static parts of `model_input.kv_cache_*` (group→layer
    // mapping, group types, block strides). Only invoked when at least one
    // stream carries a `kv_cache_resource`. No-op when cache_manager is null.
    void fillKVCacheMetadata(GptModelInputs& model_input, size_t max_blocks_num) const;
    // Per-stream variant of `process`. Today this is the only path; once
    // prefix-kv-cache split is enabled it routes the prefix/suffix sub-streams.
    absl::Status processNormal(const std::list<EmbeddingStreamPtr>& streams);
    bool         shouldUsePrefixKVCache(const EmbeddingStreamPtr& stream) const;
    absl::Status processPrefixCacheStream(const EmbeddingStreamPtr& stream);
};

}  // namespace rtp_llm
