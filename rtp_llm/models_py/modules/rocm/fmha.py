import torch
import logging
from typing import Optional, Any, List
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.ops import PyAttentionInputs, FMHAType, KVCache
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from libth_transformer.rtp_llm_ops import FusedRopeKVCachePrefillOp, FusedRopeKVCacheDecodeOp
import aiter
from aiter import dtypes

import os
# Simple data structure for fmha_params
class FMHAParams:
    def __init__(self, batch_size: int, max_seq_len: int , seq_lens: Optional[torch.Tensor] = None,
                 kv_cache_block_id_host: Optional[torch.Tensor] = None,
                 kv_cache_block_id_device: Optional[torch.Tensor] = None,
                 input_lengths: Optional[torch.Tensor] = None,
                 prefix_lengths: Optional[torch.Tensor] = None):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.seq_lens = seq_lens
        self.kv_cache_block_id_host = kv_cache_block_id_host
        self.kv_cache_block_id_device = kv_cache_block_id_device
        self.input_lengths = input_lengths
        self.prefix_lengths = prefix_lengths
        self.cu_seqlens_q = None
        self.cu_seqlens_k = None
        self.max_seqlen_q = max_seq_len
        self.max_seqlen_k = max_seq_len
        self.token_q_num = 0
        if input_lengths is not None:
            input_lengths_gpu = input_lengths.to(torch.device("cuda"), dtype=torch.int32, non_blocking=True)
            self.cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_lengths_gpu.device)
            self.cu_seqlens_q[1:] = torch.cumsum(input_lengths_gpu, 0)
            self.token_q_num = int(input_lengths.sum().item())
            if prefix_lengths is not None and prefix_lengths.numel() > 0:
                prefix_lengths_gpu = prefix_lengths.to(input_lengths_gpu.device, dtype=torch.int32, non_blocking=True)
                kv_lengths = input_lengths_gpu + prefix_lengths_gpu
                self.max_seqlen_k = int(kv_lengths.max().item())
            else:
                kv_lengths = input_lengths_gpu
                self.max_seqlen_k = max_seq_len
            self.cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_lengths_gpu.device)
            self.cu_seqlens_k[1:] = torch.cumsum(kv_lengths, 0)

class FMHAPrefillImplBase(FMHAImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        attn_inputs: PyAttentionInputs,
        config: GptInitModelParameters
    ) -> None:
        super().__init__(fmha_impl, FusedRopeKVCachePrefillOp(config.gpt_init_params),
                         attn_inputs)
class FMHADecodeImplBase(FMHAImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        attn_inputs: PyAttentionInputs,
        config: GptInitModelParameters,
    ) -> None:
        super().__init__(fmha_impl, FusedRopeKVCacheDecodeOp(config.gpt_init_params), attn_inputs)

PREFILL_MHA_IMPS: List[type[FMHAPrefillImplBase]] = []
DECODE_MHA_IMPS: List[type[FMHADecodeImplBase]] = []


try:
    class AiterPrefillImpl(FMHAPrefillImplBase):
        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(AiterPrefillAttnOp(config), attn_inputs, config)

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.AITER_PREFILL


    PREFILL_MHA_IMPS.append(AiterPrefillImpl)
except ImportError:
    logging.info("AiterPrefillImpl not available, skipped.")

class AiterPrefillAttnOp():
    def __init__(
        self , config: GptInitModelParameters
    ):
        self.head_num = config.head_num
        self.head_dim = config.hidden_size // config.head_num
        self.head_num_kv = config.head_num_kv
        self.kv_cache_data_type = config.kv_cache_data_type

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Extract batch size and max sequence length from attention inputs
        batch_size = attn_inputs.input_lengths.size(0)
        max_seq_len = attn_inputs.input_lengths.max().item()
        prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
        
        # Create and return fmha_params with the required attributes
        self.fmha_params = FMHAParams(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            input_lengths=attn_inputs.input_lengths,
            prefix_lengths=prefix_lengths,
            kv_cache_block_id_host=attn_inputs.kv_cache_block_id_host,
            kv_cache_block_id_device=attn_inputs.kv_cache_block_id_device
        )
        return self.fmha_params

    def forward(self, qkv, kv_cache, fmha_params):

        # q_tensor: {batch_size, head_num, seq_len, head_dim}
        # k_tensor: {batch_size, head_num_kv, seq_len_with_prefix, head_dim}
        # v_tensor: {batch_size, head_num_kv, seq_len_with_prefix, head_dim}
        q_tensor, k_tensor, v_tensor = qkv[0],qkv[1],qkv[2]
        
        batch_size_actual, head_num_actual, seq_len, head_dim = q_tensor.shape
        has_prefix = (
            kv_cache is not None
            and fmha_params.prefix_lengths is not None
            and fmha_params.prefix_lengths.numel() > 0
            and fmha_params.prefix_lengths.max().item() > 0
            and fmha_params.kv_cache_block_id_device is not None
            and fmha_params.kv_cache_block_id_device.numel() > 0
        )
        if has_prefix:
            valid_queries = []
            input_lengths = fmha_params.input_lengths
            for batch_idx in range(batch_size_actual):
                actual_len = input_lengths[batch_idx].item()
                valid_queries.append(q_tensor[batch_idx, :, :actual_len, :].transpose(0, 1))
            q = torch.cat(valid_queries, dim=0)

            key_cache = kv_cache.k_cache_base
            value_cache = kv_cache.v_cache_base
            block_size = key_cache.shape[2]
            vec_size = 16 // key_cache.element_size()
            key_cache = key_cache.view(
                key_cache.shape[0],
                self.head_num_kv,
                self.head_dim // vec_size,
                block_size,
                vec_size,
            )
            value_cache = value_cache.view(
                value_cache.shape[0],
                self.head_num_kv,
                block_size // vec_size,
                self.head_dim,
                vec_size,
            )
            block_table = fmha_params.kv_cache_block_id_device.to(dtype=torch.int32, device=q.device)
            needed_cols = (fmha_params.max_seqlen_k + block_size - 1) // block_size + 1
            if block_table.shape[1] < needed_cols:
                pad = torch.zeros(
                    (block_table.shape[0], needed_cols - block_table.shape[1]),
                    dtype=block_table.dtype,
                    device=block_table.device,
                )
                block_table = torch.cat([block_table, pad], dim=1)
            seqlen_k = (fmha_params.cu_seqlens_k[1:] - fmha_params.cu_seqlens_k[:-1]).to(torch.int32)
            kv_indptr = torch.zeros(batch_size_actual + 1, dtype=torch.int32, device=q.device)
            kv_page_indices = torch.zeros(1, dtype=torch.int32, device=q.device)
            q_descale = None
            k_descale = None
            v_descale = None
            fp8_dtypes = tuple(
                dtype
                for dtype in (
                    getattr(torch, "float8_e4m3fnuz", None),
                    getattr(torch, "float8_e4m3fn", None),
                )
                if dtype is not None
            )
            if fp8_dtypes and key_cache.dtype in fp8_dtypes:
                q_descale = torch.ones(1, dtype=torch.float32, device=q.device)
                k_descale = torch.ones(1, dtype=torch.float32, device=q.device)
                v_descale = torch.ones(1, dtype=torch.float32, device=q.device)
            res = aiter.mha_batch_prefill_func(
                q,
                key_cache,
                value_cache,
                fmha_params.cu_seqlens_q,
                kv_indptr,
                kv_page_indices,
                fmha_params.max_seqlen_q,
                fmha_params.max_seqlen_k,
                causal=True,
                block_table=block_table,
                seqlen_k=seqlen_k,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
            return res.reshape(fmha_params.token_q_num, self.head_num * self.head_dim)
        
        # dimensions for aiter.flash_attn_func  {batch_size, seq_len, head_num, head_dim}
        q = q_tensor.transpose(1, 2)  # {batch_size, seq_len, head_num, head_dim}
        k = k_tensor.transpose(1, 2)  # {batch_size, seq_len_with_prefix, head_num_kv, head_dim}
        v = v_tensor.transpose(1, 2)  # {batch_size, seq_len_with_prefix, head_dim}

        res = aiter.flash_attn_func(q, k, v, dropout_p=0., softmax_scale=None, causal=True)
        
        input_lengths = fmha_params.input_lengths  # 每个 batch 的真实长度
        hidden_size = head_num_actual * head_dim
        
        valid_results = []
        for batch_idx in range(batch_size_actual):
            actual_len = input_lengths[batch_idx].item()
            batch_result = res[batch_idx, :actual_len, :, :]  # {actual_len, head_num, head_dim}
            batch_result = batch_result.reshape(actual_len, hidden_size)  # {actual_len, hidden_size}
            valid_results.append(batch_result)
        
        final_result = torch.cat(valid_results, dim=0)  # {total_token_num, hidden_size}
        
        return final_result

try:
    class AiterDecodeImpl(FMHADecodeImplBase):
        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(AiterDecodeAttnOp(config), attn_inputs, config)

    DECODE_MHA_IMPS.append(AiterDecodeImpl)
except ImportError:
    logging.info("PagedAttnDecodeOp not available, skipped.")
    
class AiterDecodeAttnOp():
    def __init__(
        self , config: GptInitModelParameters
    ):
        self.head_num = config.head_num
        self.head_dim = config.hidden_size // config.head_num
        self.head_num_kv = config.head_num_kv
        self.kv_cache_data_type = config.kv_cache_data_type

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Extract batch size and max sequence length from attention inputs
        batch_size = attn_inputs.input_lengths.size(0)
        max_seq_len = attn_inputs.input_lengths.max().item()
        seq_lens = attn_inputs.sequence_lengths.cpu() + 1
        seq_lens = seq_lens.cuda()
        kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host
        kv_cache_block_id_device = attn_inputs.kv_cache_block_id_device
        # Create and return fmha_params with the required attributes
        self.fmha_params = FMHAParams(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            seq_lens = seq_lens,
            kv_cache_block_id_host = kv_cache_block_id_host,
            kv_cache_block_id_device = kv_cache_block_id_device
        )
        return self.fmha_params

    def forward(self, query: torch.Tensor, kv_cache: Optional[KVCache] , fmha_params:Optional[Any]) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens
        max_seq_len = fmha_params.max_seq_len + 1
        key_cache = kv_cache.k_cache_base
        value_cache = kv_cache.v_cache_base
        block_tables_id_host = fmha_params.kv_cache_block_id_host
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        num_kv_heads = self.head_num_kv
        scale = 1.0 / (self.head_dim ** 0.5)
        alibi_slopes = None

        k_scale = kv_cache.k_scale_base if kv_cache and kv_cache.k_scale_base is not None else 1.0
        v_scale = kv_cache.v_scale_base if kv_cache and kv_cache.v_scale_base is not None else 1.0
        max_num_blocks = block_tables_id_device.shape[1]

        if os.environ.get('USE_ASM_PA'):
            output = torch.ops.aiter.pa_fwd_asm(
                query,
                key_cache,
                value_cache,
                block_tables_id_device,
                seq_lens,
                max_num_blocks,
                k_scale,
                v_scale,
            )
        else :
            num_seqs, num_heads, head_size = query.shape
            block_size = value_cache.shape[2]
            _PARTITION_SIZE_ROCM = 256
            
            # init output
            output = torch.empty_like(query)

            max_num_partitions = (max_seq_len + _PARTITION_SIZE_ROCM - 1) // _PARTITION_SIZE_ROCM
            assert _PARTITION_SIZE_ROCM % block_size == 0
            # init tmp_output
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            
            # init exp_sums 
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            fp8_out_scale=None
            cpa_fp8_out = False
            # init max_logits 
            max_logits = torch.ones_like(exp_sums)
            
            kv_cache_dtype ="auto"
            key_cache_reshaped = key_cache.permute(0,1,3,2)
            value_cache_reshaped = value_cache.permute(0,1,3,2)

            aiter.paged_attention_rocm(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache_reshaped,
                value_cache_reshaped,
                num_kv_heads,
                float(scale),
                block_tables_id_device,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,  # kv_cache_dtype
                k_scale,
                v_scale,
                fp8_out_scale if cpa_fp8_out else None,
                _PARTITION_SIZE_ROCM,
            )

        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped
