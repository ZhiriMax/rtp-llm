from rtp_llm.server.server_args.util import str2bool


def init_engine_group_args(parser, runtime_config):
    ##############################################################################################################
    # Engine Configuration
    # Fields merged from EngineConfig to RuntimeConfig (warm_up, warm_up_with_loss)
    ##############################################################################################################
    engine_group = parser.add_argument_group("Engine Configuration")
    engine_group.add_argument(
        "--warm_up",
        env_name="WARM_UP",
        bind_to=(runtime_config, 'warm_up'),
        type=str2bool,
        default=True,
        help="在服务启动时是否开启预热",
    )
    engine_group.add_argument(
        "--warm_up_with_loss",
        env_name="WARM_UP_WITH_LOSS",
        bind_to=(runtime_config, 'warm_up_with_loss'),
        type=str2bool,
        default=False,
        help="在服务启动时是否开启损失去预热",
    )
    engine_group.add_argument(
        "--embedding_kv_cache_mode",
        env_name="EMBEDDING_KV_CACHE_MODE",
        bind_to=(runtime_config, "embedding_kv_cache_mode"),
        type=str,
        choices=["off", "block", "in_batch"],
        default="off",
        help=(
            "Embedding Engine KV cache 模式: off=当前路径, block=仅接入 paged KV block, "
            "in_batch=启用批内公共前缀去重"
        ),
    )
    engine_group.add_argument(
        "--embedding_kv_cache_commit_policy",
        env_name="EMBEDDING_KV_CACHE_COMMIT_POLICY",
        bind_to=(runtime_config, "embedding_kv_cache_commit_policy"),
        type=str,
        choices=["prefix_block", "full_block"],
        default="prefix_block",
        help=(
            "Embedding Engine KV cache 提交策略: "
            "prefix_block=只提交共享前缀完整 block, full_block=提交每行完整 block"
        ),
    )
