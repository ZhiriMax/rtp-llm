#pragma once

#include <torch/all.h>

namespace torch_ext {

void rms_norm_per_block_quant_fp8(at::Tensor& input,
                                  at::Tensor& output_q,
                                  at::Tensor& output_s,
                                  at::Tensor& weight,
                                  int64_t     group_size,
                                  double      rms_eps,
                                  double      quant_eps,
                                  double      fp8_min,
                                  double      fp8_max);

}  // namespace torch_ext
