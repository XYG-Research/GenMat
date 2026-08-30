from __future__ import annotations

import warnings

from genim.model import CausalTransformerLM, ModelConfig


def test_model_init_emits_no_nested_tensor_warning() -> None:
    cfg = ModelConfig(vocab_size=16, max_len=8, d_model=32, n_layers=2, n_heads=4, dropout=0.0, element_emb="token")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CausalTransformerLM(cfg)

    msgs = [str(w.message) for w in caught]
    assert not any("enable_nested_tensor is True" in msg for msg in msgs), msgs


def test_element_feature_sets_preserve_legacy_shape_and_expand_new_models() -> None:
    tokens = ["<PAD>", "<BOS>", "<EOS>", "E_Fe", "E_O"]
    legacy = CausalTransformerLM(
        ModelConfig(
            vocab_size=len(tokens),
            max_len=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
            element_emb="features",
        ),
        id_to_token=tokens,
    )
    periodic = CausalTransformerLM(
        ModelConfig(
            vocab_size=len(tokens),
            max_len=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
            element_emb="features",
            element_feature_set="periodic8",
        ),
        id_to_token=tokens,
    )
    assert legacy.elem_proj is not None and legacy.elem_proj.in_features == 4
    assert periodic.elem_proj is not None and periodic.elem_proj.in_features == 8
