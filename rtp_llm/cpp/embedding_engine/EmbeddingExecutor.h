#pragma once

#include <memory>
#include <bitset>
#include <torch/python.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/all.h>
#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"
#include "rtp_llm/cpp/dataclass/EngineInitParameter.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/embedding_engine/ModelRequest.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingGatherLayout.h"

namespace rtp_llm {

inline constexpr const char* kEmbeddingKVCacheModeOff     = "off";
inline constexpr const char* kEmbeddingKVCacheModeBlock   = "block";
inline constexpr const char* kEmbeddingKVCacheModeInBatch = "in_batch";

inline constexpr const char* kEmbeddingKVCacheCommitPolicyPrefixBlock = "prefix_block";
inline constexpr const char* kEmbeddingKVCacheCommitPolicyFullBlock   = "full_block";

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
    explicit EmbeddingExecutor(const EngineInitParams&            params,
                               rtp_llm::DeviceBase*              device,
                               py::object                        handler,
                               const std::shared_ptr<CacheManager>& cache_manager = nullptr,
                               const rtp_llm::GptInitParameter*      embedding_params = nullptr);

    absl::Status process(const std::list<EmbeddingStreamPtr>& streams);

private:
    struct EmbeddingKVCacheAllocation {
        std::vector<std::vector<int>>     row_block_indices;
        std::vector<std::vector<int32_t>> row_token_ids;
        std::vector<std::vector<int64_t>> row_cache_keys;
        EmbeddingGatherPrefixGroup        prefix_group;
        std::vector<int>                  row_lengths;
        int                               seq_size_per_block = 0;
    };

    struct EmbeddingModelInput {
        GptModelInputs                         model_input;
        std::vector<EmbeddingKVCacheAllocation> kv_cache_allocations;
    };

    std::unique_ptr<GptModel>       model_;
    py::object                      handler_;
    HandlerArgs::Flag               handler_args_;
    py::handle                      torch_type_;
    rtp_llm::DeviceBase*            device_;
    rtp_llm::BufferPtr              max_position_ids_buf_;
    kmonitor::MetricsReporterPtr    metrics_reporter_ = nullptr;
    const rtp_llm::GptInitParameter params_;
    std::shared_ptr<CacheManager>    cache_manager_;
    std::string                      embedding_kv_cache_mode_;
    std::string                      embedding_kv_cache_commit_policy_;
    bool                             enable_embedding_kv_cache_       = false;
    bool                             enable_in_batch_kv_prefix_dedup_ = false;
    bool                             allow_in_batch_kv_prefix_dedup_  = false;

    ModelRequest                              generateOldModelRequest(GptModelInputs& model_input);
    absl::StatusOr<EmbeddingModelInput>       gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const;
    absl::StatusOr<GptModelInputs> gatherLegacyModelInput(const std::list<EmbeddingStreamPtr>& streams) const;
    absl::StatusOr<EmbeddingModelInput> gatherKVCacheModelInput(
        const std::list<EmbeddingStreamPtr>& streams) const;
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
    void freeKVCacheAllocations(const std::vector<EmbeddingKVCacheAllocation>& allocations) const;
    void commitKVCacheAllocations(const std::vector<EmbeddingKVCacheAllocation>& allocations) const;
    void commitKVCacheRow(const EmbeddingKVCacheAllocation& allocation, int local_row_idx, int commit_len) const;
};

}  // namespace rtp_llm
