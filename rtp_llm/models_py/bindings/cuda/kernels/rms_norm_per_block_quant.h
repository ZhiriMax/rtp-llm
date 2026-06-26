#pragma once

#include <torch/library.h>
#include <torch/torch.h>

namespace rtp_llm {

void rms_norm_per_block_quant_fp8(torch::Tensor input,
                                  torch::Tensor output_q,
                                  torch::Tensor output_s,
                                  torch::Tensor weight,
                                  int64_t       group_size,
                                  double        rms_eps,
                                  double        quant_eps,
                                  double        min_8bit,
                                  double        max_8bit);

}
