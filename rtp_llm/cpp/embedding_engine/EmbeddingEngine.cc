#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <exception>
#include <cstdlib>
#include <string>

using namespace std;
namespace rtp_llm {

namespace {

std::string embeddingKVCacheModeFromEnv() {
    const char* mode = std::getenv("EMBEDDING_KV_CACHE_MODE");
    return mode ? std::string(mode) : std::string(kEmbeddingKVCacheModeOff);
}

bool enableEmbeddingKVCache(const std::string& mode) {
    return mode == kEmbeddingKVCacheModeBlock || mode == kEmbeddingKVCacheModeInBatch;
}

bool supportEmbeddingKVCache(const GptInitParameter& params, std::string& reason) {
    if (params.use_mla_) {
        reason = "MLA cache layout is not supported yet";
        return false;
    }
    if (!params.is_causal_) {
        reason = "non-causal attention cannot safely reuse prefix KV";
        return false;
    }
    return true;
}

}  // namespace

EmbeddingEngine::EmbeddingEngine(const EngineInitParams& params, py::object handler):
    params_(params.gpt_init_parameter), metrics_reporter_(params.metrics_reporter) {
    rtp_llm::DeviceFactory::initDevices(params.gpt_init_parameter);
    auto* device = rtp_llm::DeviceFactory::getDefaultDevice();
    const auto cache_mode = embeddingKVCacheModeFromEnv();
    if (enableEmbeddingKVCache(cache_mode)) {
        std::string unsupported_reason;
        if (!supportEmbeddingKVCache(params_, unsupported_reason)) {
            RTP_LLM_LOG_WARNING("embedding kv cache mode %s fallback to off: %s",
                                cache_mode.c_str(),
                                unsupported_reason.c_str());
        } else {
            if (!params_.use_kvcache_) {
                RTP_LLM_LOG_INFO("force enable use_kvcache for embedding kv cache mode %s", cache_mode.c_str());
                params_.use_kvcache_ = true;
            }
            auto cache_config = CacheConfigCreator::createConfig(params_);
            RTP_LLM_LOG_INFO("create embedding cache manager with config %s", cache_config.debugString().c_str());
            resource_context_.cache_manager =
                std::make_shared<CacheManager>(cache_config, device, false, metrics_reporter_, params_);
            resource_context_.reuse_cache = true;
        }
    }
    executor_.reset(new EmbeddingExecutor(params, device, handler, resource_context_.cache_manager, &params_));
    scheduler_.reset(new EmbeddingScheduler(params_, metrics_reporter_));

    (void)startLoop();
}

EmbeddingEngine::~EmbeddingEngine() {
    RTP_LLM_LOG_INFO("destory embedding engine");
    (void)stop();
}

const rtp_llm::GptInitParameter& EmbeddingEngine::GetGptInitParameter() {
    return params_;
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
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    CHECK_AND_RETURN_REF(streams, scheduler_->scheduleNew());
    if (streams.empty()) {
        RTP_LLM_LOG_INFO("no query run and sleep");
        return absl::OkStatus();
    }
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
    return absl::OkStatus();
}

}  // namespace rtp_llm
