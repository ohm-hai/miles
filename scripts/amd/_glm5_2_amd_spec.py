"""Audited GLM-5.2 checkpoint identities and architecture invariants."""

MODEL_REVISIONS = {
    "GLM-5.2": "b4734de4facf877f85769a911abafc5283eab3d9",
    "GLM-5.2_5layer": "1c749139f70e158e4420ba67f342bef1de2e650d",
}
DATA_REVISION = "2e65612930298bde4c5d58fd97b3f23a483aaff9"
CHECKPOINT_INDEX_LAYOUTS = {
    # (number of weight entries, number of shard files, tensor payload bytes)
    "GLM-5.2": (59585, 282, 1506659919872),
    "GLM-5.2_5layer": (1618, 14, 45683868160),
}
SOURCE_CONFIG_ASSETS = {
    "GLM-5.2": (
        3732,
        "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a",
    ),
    "GLM-5.2_5layer": (
        1690,
        "f426922f0ad4efaaa6cae6dd24a6f0c59b0df1a6a1265857c4e9a949a5fdb020",
    ),
}
SOURCE_INDEX_ASSETS = {
    "GLM-5.2": (
        5408032,
        "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e",
    ),
    "GLM-5.2_5layer": (
        145189,
        "d898bb64c9258ed870a83d23fde8142cb71c0db5eba8dcacd9d2884b4c995ec9",
    ),
}
CRITICAL_CONFIG_VALUES = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "dtype": "bfloat16",
    "first_k_dense_replace": 3,
    "head_dim": 192,
    "hidden_size": 6144,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_skip_topk_offset": 3,
    "index_topk": 2048,
    "index_topk_freq": 4,
    "indexer_rope_interleave": True,
    "intermediate_size": 12288,
    "kv_lora_rank": 512,
    "model_type": "glm_moe_dsa",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_nextn_predict_layers": 1,
    "q_lora_rank": 2048,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "v_head_dim": 256,
    "vocab_size": 154880,
}

# These files are consumed at runtime by SGLang and dataset validation.
# Both pinned model repositories have the same audited tokenizer payloads.
RUNTIME_HF_ASSETS = {
    "chat_template.jinja": (
        5076,
        "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
    ),
    "generation_config.json": (
        194,
        "ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55",
    ),
    "tokenizer.json": (
        20217442,
        "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    ),
    "tokenizer_config.json": (
        761,
        "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
    ),
}
