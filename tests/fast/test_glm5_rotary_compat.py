from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


class _Dummy:
    pass


class _AttnMaskType:
    padding = object()
    causal = object()


def _stub_module(monkeypatch, dotted_name: str, **attrs) -> ModuleType:
    """Install a minimal importable module hierarchy for the isolated model import."""
    parts = dotted_name.split(".")
    for index in range(len(parts)):
        name = ".".join(parts[: index + 1])
        module = sys.modules.get(name)
        if module is None:
            module = ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
        if index:
            parent_name = ".".join(parts[:index])
            monkeypatch.setattr(sys.modules[parent_name], parts[index], module, raising=False)

    module = sys.modules[dotted_name]
    for name, value in attrs.items():
        monkeypatch.setattr(module, name, value, raising=False)
    return module


def _load_glm5_module(monkeypatch):
    tensor_type = type("Tensor", (), {})
    _stub_module(monkeypatch, "torch", Tensor=tensor_type)
    _stub_module(monkeypatch, "megatron.core.parallel_state")
    _stub_module(
        monkeypatch,
        "megatron.core.extensions.transformer_engine",
        TEColumnParallelLinear=_Dummy,
        TELinear=_Dummy,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.extensions.transformer_engine_spec_provider",
        TESpecProvider=_Dummy,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.models.common.embeddings",
        RotaryEmbedding=_Dummy,
        YarnRotaryEmbedding=_Dummy,
        _yarn_get_mscale=lambda *_args, **_kwargs: 1.0,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.models.gpt.gpt_layer_specs",
        get_gpt_decoder_block_spec=lambda *_args, **_kwargs: None,
    )
    _stub_module(monkeypatch, "megatron.core.post_training.modelopt.layers", Linear=_Dummy)
    _stub_module(monkeypatch, "megatron.core.tensor_parallel.layers", ColumnParallelLinear=_Dummy)
    _stub_module(
        monkeypatch,
        "megatron.core.tensor_parallel.mappings",
        gather_from_sequence_parallel_region=lambda value, **_kwargs: value,
        scatter_to_sequence_parallel_region=lambda value, **_kwargs: value,
    )
    _stub_module(monkeypatch, "megatron.core.transformer.attention", Attention=_Dummy)
    _stub_module(monkeypatch, "megatron.core.transformer.enums", AttnMaskType=_AttnMaskType)
    _stub_module(monkeypatch, "megatron.core.transformer.identity_op", IdentityOp=_Dummy)
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.moe.moe_utils",
        RouterGatingLinearFunction=_Dummy,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.spec_utils",
        ModuleSpec=_Dummy,
        build_module=lambda *_args, **_kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.transformer_block",
        get_num_layers_to_build=lambda *_args, **_kwargs: 0,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.transformer_config",
        MLATransformerConfig=_Dummy,
    )
    _stub_module(monkeypatch, "miles.utils.hf_config", load_hf_config=lambda *_args, **_kwargs: None)
    _stub_module(
        monkeypatch,
        "miles.utils.replay_base",
        indexer_replay_manager=SimpleNamespace(),
    )
    _stub_module(
        monkeypatch,
        "miles_plugins.models.glm5.ops.indexer",
        generate_varlen_mask_params=lambda *_args, **_kwargs: None,
        lighting_indexer=lambda *_args, **_kwargs: None,
    )
    _stub_module(monkeypatch, "miles_plugins.models.glm5.ops.sparse_mla", SparseMLA=_Dummy)

    module_path = Path(__file__).resolve().parents[2] / "miles_plugins" / "models" / "glm5" / "glm5.py"
    module_name = "miles_plugins.models.glm5._rotary_compat_test_target"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, tensor_type


def test_extract_rotary_pos_emb_accepts_current_and_legacy_returns(monkeypatch):
    glm5, tensor_type = _load_glm5_module(monkeypatch)
    rotary_pos_emb = tensor_type()

    assert glm5._extract_rotary_pos_emb(rotary_pos_emb) is rotary_pos_emb
    assert glm5._extract_rotary_pos_emb((rotary_pos_emb, 1.0)) is rotary_pos_emb
