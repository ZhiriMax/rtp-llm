#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <exception>

using namespace std;
namespace rtp_llm {

namespace {

bool enableEmbeddingKVCache(const RuntimeConfig& runtime_config) {
    return runtime_config.embedding_kv_cache_mode == kEmbeddingKVCacheModeBlock
           || runtime_config.embedding_kv_cache_mode == kEmbeddingKVCacheModeInBatch;
}

bool supportEmbeddingKVCache(const ModelConfig& model_config, const CacheConfig& cache_config, std::string& reason) {
    if (cache_config.groupNums() != 1) {
        reason = "hybrid cache groups are not supported yet";
        return false;
    }
    if (cache_config.group_types.empty() || cache_config.group_types[0] != CacheGroupType::FULL) {
        reason = "linear cache groups are not supported yet";
        return false;
    }
    if (cache_config.use_mla || model_config.attn_config.use_mla) {
        reason = "MLA cache layout is not supported yet";
        return false;
    }
    if (!model_config.attn_config.is_causal) {
        reason = "non-causal attention cannot safely reuse prefix KV";
        return false;
    }
    if (!model_config.attn_config.need_rope_kv_cache) {
        reason = "attention path does not write KV cache before attention";
        return false;
    }
    return true;
}

}  // namespace

EmbeddingEngine::EmbeddingEngine(const EngineInitParams& params, py::object handler):
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    concurrency_config(params.concurrency_config),
    metrics_reporter_(params.metrics_reporter),
    step_profiler_(params.profiling_debug_logging_config.torch_cuda_profiler_dir,
                   params.parallelism_config.dp_rank * params.parallelism_config.tp_size
                       + params.parallelism_config.tp_rank) {
    {
        size_t device_id = params.parallelism_config.world_rank % params.parallelism_config.local_world_size;
        rtp_llm::initRuntime(device_id,
                             params.profiling_debug_logging_config.trace_memory,
                             params.device_resource_config.enable_comm_overlap,
                             params.model_config_.mla_ops_type);
    }
    warmupNoBlockCopy();
    if (enableEmbeddingKVCache(params.runtime_config)) {
        auto cache_config = CacheConfigCreator::createConfig(
            model_config_, parallelism_config, params.runtime_config, params.kv_cache_config, std::nullopt);
        std::string unsupported_reason;
        if (!supportEmbeddingKVCache(model_config_, cache_config, unsupported_reason)) {
            RTP_LLM_LOG_WARNING("embedding kv cache mode %s fallback to off: %s",
                                params.runtime_config.embedding_kv_cache_mode.c_str(),
                                unsupported_reason.c_str());
        } else {
            RTP_LLM_LOG_INFO("create embedding cache manager with config %s", cache_config.debugString().c_str());
            resource_context_.cache_manager = std::make_shared<KVCacheManager>(
                cache_config,
                false,
                metrics_reporter_,
                params.kv_cache_config,
                parallelism_config,
                params.runtime_config);
            if (!resource_context_.cache_manager->init()) {
                RTP_LLM_FAIL("init embedding kv cache manager failed");
            }
        }
    }
    executor_.reset(new EmbeddingExecutor(params, handler, resource_context_.cache_manager));
    scheduler_.reset(
        new EmbeddingScheduler(model_config_, concurrency_config, params.runtime_config, metrics_reporter_));

    (void)startLoop();
}

EmbeddingEngine::~EmbeddingEngine() {
    RTP_LLM_LOG_INFO("destory embedding engine");
    (void)stop();
}

absl::Status EmbeddingEngine::startLoop() {
    RTP_LLM_LOG_INFO("start embedding engine");
    running_     = true;
    loop_thread_ = std::thread(&EmbeddingEngine::loop, this);
    return absl::OkStatus();
}

absl::Status EmbeddingEngine::stop() {
    RTP_LLM_LOG_INFO("stop embedding engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void EmbeddingEngine::loop() {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_INFO("loop begin");
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

std::shared_ptr<EmbeddingOutput> EmbeddingEngine::decode(th::Tensor                       token_ids,
                                                         th::Tensor                       token_type_ids,
                                                         th::Tensor                       input_lengths,
                                                         int64_t                          request_id,
                                                         std::optional<MultimodalFeature> multimodal_features,
                                                         std::optional<th::Tensor>        input_embeddings) {
    auto input = std::make_shared<EmbeddingInput>(
        token_ids, token_type_ids, input_lengths, request_id, multimodal_features, input_embeddings);
    return decode(input);
}

std::shared_ptr<EmbeddingOutput> EmbeddingEngine::decode(std::shared_ptr<EmbeddingInput> input) {
    auto embedding_stream = std::make_shared<EmbeddingStream>(input);
    embedding_stream->setMetricReporter(metrics_reporter_);
    THROW_IF_STATUS_ERROR(enqueue(embedding_stream));
    embedding_stream->waitFinish();
    return embedding_stream->embeddingOutput();
}

absl::Status EmbeddingEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

absl::Status EmbeddingEngine::enqueue(EmbeddingStreamPtr streams) {
    return scheduler_->enqueue(streams);
}

absl::Status EmbeddingEngine::step() {
    cudaSyncAndCheck();
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    CHECK_AND_RETURN_REF(streams, scheduler_->scheduleNew());
    if (streams.empty()) {
        RTP_LLM_LOG_INFO("no query run and sleep");
        return absl::OkStatus();
    }
    step_profiler_.tick();
    try {
        auto status = executor_->process(streams);
        if (!status.ok()) {
            for (auto& stream : streams) {
                stream->setError(status.ToString());
                RTP_LLM_LOG_WARNING(
                    "error_stream_info: length: %d, exception: %s", stream->inputLength(), status.ToString().c_str());
            }
        }
    } catch (const exception& e) {
        std::string error_msg = e.what();
        RTP_LLM_LOG_WARNING("run engine failed, stream size: %d, error: %s", streams.size(), error_msg.c_str());
        for (auto& stream : streams) {
            stream->setError(error_msg);
            RTP_LLM_LOG_WARNING("error_stream_info: length: %d", stream->inputLength());
        }
        if (error_msg.find("CUDA Driver error") != string::npos || error_msg.find("CUDA error") != string::npos) {
            RTP_LLM_LOG_ERROR("detect CUDA error, do abort");
            abort();
        }
    }
    cudaSyncAndCheck();
    return absl::OkStatus();
}

}  // namespace rtp_llm
