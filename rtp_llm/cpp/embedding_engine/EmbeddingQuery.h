#pragma once
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <torch/extension.h>

namespace rtp_llm {

class EmbeddingInput {
public:
    explicit EmbeddingInput(const torch::Tensor&                    token_ids,
                            const torch::Tensor&                    token_type_ids,
                            const torch::Tensor&                    input_lengths,
                            int64_t                                 request_id,
                            const std::optional<MultimodalFeature>& multimodal_features = std::nullopt,
                            std::optional<torch::Tensor>            input_embeddings    = std::nullopt);

    explicit EmbeddingInput(const std::vector<int32_t>&             token_ids,
                            const std::vector<int32_t>&             token_type_ids,
                            const std::vector<int32_t>&             input_lengths,
                            int64_t                                 request_id,
                            const std::optional<MultimodalFeature>& multimodal_features = std::nullopt,
                            std::optional<torch::Tensor>            input_embeddings    = std::nullopt);

    torch::Tensor                               token_ids;
    torch::Tensor                               token_type_ids;
    torch::Tensor                               input_lengths;
    int64_t                                     total_length;
    int64_t                                     request_id;
    std::optional<std::vector<MultimodalInput>> multimodal_inputs;
    std::optional<MultimodalFeature>            multimodal_features;
    std::optional<torch::Tensor>                input_embeddings;
    // Mainse-style shared user-prefix cache flags. The renderer sets these per
    // request; the engine-side `shouldUsePrefixKVCache` performs the final
    // feasibility check (batch_size>1, prefix>=block_size, etc.) before
    // actually splitting the forward into a (prefix, suffix) pair.
    bool                                        enable_prefix_kv_cache{false};
    int64_t                                     common_prefix_length{0};

    // Per-batch prefix kv-cache prefill lengths and kv resource handle. These
    // are set internally by `processPrefixCacheStream` when it constructs the
    // synthetic prefix/suffix sub-streams, and consumed by `gatherModelInput`
    // to populate `GptModelInputs.kv_cache_kernel_block_id` and to offset
    // RoPE position ids by the cached prefix length.
    torch::Tensor                               prefix_lengths;
    BatchKVCacheResourcePtr                     kv_cache_resource;
    CompleteTokenIdsPtr                         complete_token_ids;

    void        checkVaild();
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "EmbeddingInput {"
                     << "token_ids: [" << token_ids.sizes() << "]"
                     << ", token_type_ids: [" << token_type_ids.sizes() << "]"
                     << ", input_lengths: [" << input_lengths.sizes() << "]"
                     << ", total_length: " << total_length << "}";
        if (input_embeddings.has_value()) {
            debug_string << ", input_embeddings: [" << input_embeddings.value().sizes() << "]";
        }
        return debug_string.str();
    }
};

class TypedOutput {
public:
    void setTensorOuput(torch::Tensor t) {
        isTensor = true;
        this->t  = t;
    }
    void setMapOutput(std::vector<std::map<std::string, at::Tensor>>& m) {
        isTensor  = false;
        this->map = std::move(m);
    }

    bool                                                          isTensor;
    std::optional<at::Tensor>                                     t;
    std::optional<std::vector<std::map<std::string, at::Tensor>>> map;
};

class EmbeddingOutput {
public:
    void setTensorOutput(torch::Tensor t) {
        output.setTensorOuput(t);
    }
    void setMapOutput(std::vector<std::map<std::string, torch::Tensor>>& m) {
        output.setMapOutput(m);
    }
    void setError(ErrorCode code, const std::string& error_msg) {
        error_info = ErrorInfo(code, error_msg);
    }

    TypedOutput output;
    ErrorInfo   error_info;
};

}  // namespace rtp_llm
