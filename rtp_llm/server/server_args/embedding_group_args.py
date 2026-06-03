def init_embedding_group_args(parser):
    ##############################################################################################################
    # Embedding Configuration
    ##############################################################################################################
    embedding_group = parser.add_argument_group("Embedding Configuration")
    embedding_group.add_argument(
        "--embedding_model",
        env_name="EMBEDDING_MODEL",
        type=int,
        default=0,
        help="嵌入模型的具体类型",
    )

    embedding_group.add_argument(
        "--extra_input_in_mm_embedding",
        env_name="EXTRA_INPUT_IN_MM_EMBEDDING",
        type=str,
        default=None,
        help='在多模态嵌入中使用额外的输入，可选值"INDEX"',
    )

    embedding_group.add_argument(
        "--embedding_kv_cache_mode",
        env_name="EMBEDDING_KV_CACHE_MODE",
        type=str,
        default="off",
        choices=["off", "block", "in_batch"],
        help="Embedding KV cache 模式",
    )

    embedding_group.add_argument(
        "--embedding_kv_cache_commit_policy",
        env_name="EMBEDDING_KV_CACHE_COMMIT_POLICY",
        type=str,
        default="prefix_block",
        choices=["prefix_block", "full_block"],
        help="Embedding KV cache 写回 block cache 的策略",
    )
