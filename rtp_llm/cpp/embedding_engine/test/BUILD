load("//:def.bzl", "copts")

cc_test(
    name = "embedding_gather_layout_test",
    srcs = [
        "EmbeddingGatherLayoutTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/embedding_engine:embedding_gather_layout",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
