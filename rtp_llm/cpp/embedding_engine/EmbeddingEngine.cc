#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include <exception>

using namespace std;
namespace rtp_llm {

EmbeddingEngine::EmbeddingEngine(const EngineInitParams& params, py::object handler):
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    concurrency_config(params.concurrency_config),
    metrics_reporter_(params.metrics_reporter),
    step_profiler_(params.profiling_debug_logging_config.torch_cuda_profiler_dir,
                   params.parallelism_config.dp_rank * params.parallelism_config.tp_size
                       + params.parallelism_config.tp_rank),
    runtime_config_(params.runtime_config),
    kv_cache_config_(params.kv_cache_config) {
    {
        size_t device_id = params.parallelism_config.world_rank % params.parallelism_config.local_world_size;
        rtp_llm::initRuntime(device_id,
                             params.profiling_debug_logging_config.trace_memory,
                             params.device_resource_config.enable_comm_overlap,
                             params.model_config_.mla_ops_type);
    }
    initCacheManagerIfNeeded(params, handler);
    warmupNoBlockCopy();
    executor_.reset(new EmbeddingExecutor(
        params, handler, resource_context_.cache_manager, kv_cache_group_num_, kv_cache_layer_to_group_));
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


void EmbeddingEngine::initCacheManagerIfNeeded(const EngineInitParams& params, py::object handler) {
    bool enable_prefix_kv_cache = false;
    {
        py::gil_scoped_acquire acquire;
        if (py::hasattr(handler, "enable_prefix_kv_cache")) {
            enable_prefix_kv_cache = py::cast<bool>(handler.attr("enable_prefix_kv_cache"));
        }
    }
    if (!enable_prefix_kv_cache) {
        return;
    }

    auto cache_config = CacheConfigCreator::createConfig(
        model_config_, parallelism_config, runtime_config_, kv_cache_config_, std::nullopt);
    RTP_LLM_LOG_INFO("create embedding prefix kv cache manager with config %s", cache_config.debugString().c_str());
    resource_context_.cache_manager = make_shared<KVCacheManager>(
        cache_config, false, metrics_reporter_, kv_cache_config_, parallelism_config, runtime_config_);
    resource_context_.role_type = params.pd_sep_config.role_type;
    if (!resource_context_.cache_manager->init()) {
        RTP_LLM_FAIL("init embedding prefix kv cache manager failed");
    }

    const auto& cache_cfg = resource_context_.cache_manager->cacheConfig();
    kv_cache_group_num_  = cache_cfg.groupNums();
    kv_cache_layer_to_group_.clear();
    kv_cache_layer_to_group_.reserve(cache_cfg.layer_to_group_id.size());
    for (const auto group_id : cache_cfg.layer_to_group_id) {
        kv_cache_layer_to_group_.push_back(static_cast<int32_t>(group_id));
    }
}

}  // namespace rtp_llm
