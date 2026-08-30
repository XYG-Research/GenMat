from __future__ import annotations

from pathlib import Path

import torch
import pytest

from genim import ChemistryPolicy, GenMat, GenIM, SamplingConfig
from genim.checkpoints import CheckpointError, load_model_checkpoint, sha256_file
from genim.model import CausalTransformerLM, ModelConfig
from genim.train import train_lm


def test_genmat_is_primary_api_with_genim_compatibility() -> None:
    assert issubclass(GenIM, GenMat)


def _tiny_checkpoint(path: Path) -> None:
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "HALL_1",
        "LEN_0",
        "ANG_0",
        "E_Na",
        "E_Cl",
        "W_a",
        "COORD_0",
    ]
    vocab = {token: index for index, token in enumerate(tokens)}
    cfg = ModelConfig(
        vocab_size=len(tokens),
        max_len=14,
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        element_emb="token",
    )
    model = CausalTransformerLM(cfg, id_to_token=tokens)
    # No format_version: this intentionally exercises legacy v1 compatibility.
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": cfg.__dict__,
            "vocab": vocab,
            "id_to_token": tokens,
            "tokenize_config": {
                "coord_bins": 8,
                "len_bins": 8,
                "len_min": 2.0,
                "len_max": 10.0,
                "ang_bins": 8,
                "ang_min": 40.0,
                "ang_max": 140.0,
            },
            "pad_id": vocab["<PAD>"],
        },
        path,
    )


def test_legacy_checkpoint_load_has_hash_and_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    _tiny_checkpoint(path)
    loaded = load_model_checkpoint(path, device="cpu")
    assert loaded.format_version == 1
    assert loaded.sha256 == sha256_file(path)
    assert loaded.device.type == "cpu"
    with pytest.raises(CheckpointError, match="SHA256 mismatch"):
        load_model_checkpoint(path, device="cpu", expected_sha256="0" * 64)


def test_public_api_returns_provenance_even_for_rejected_candidates(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    _tiny_checkpoint(path)
    generator = GenIM.from_checkpoint(path, device="cpu")
    results = generator.sample(
        config=SamplingConfig(n=3, batch_size=3, max_sites=1, min_sites=1, top_k=1, seed=7),
        chemistry=ChemistryPolicy(mode="any", allowed_elements=["Na", "Cl"]),
    )
    assert len(results) == 3
    assert all(result.checkpoint_sha256 == sha256_file(path) for result in results)
    assert all(result.chemistry_policy["mode"] == "any" for result in results)
    assert all(result.validation.reason for result in results)


def test_public_api_forces_hall_token_for_requested_spacegroup(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    _tiny_checkpoint(path)
    generator = GenIM.from_checkpoint(path, device="cpu")
    results = generator.sample(
        config=SamplingConfig(
            n=2,
            batch_size=2,
            max_sites=1,
            min_sites=1,
            top_k=1,
            seed=7,
            fixed_spacegroup=1,
        ),
        chemistry=ChemistryPolicy(mode="any", allowed_elements=["Na", "Cl"]),
    )
    assert len(results) == 2
    assert all(result.tokens[1] == "HALL_1" for result in results)
    assert all(result.sampling["forced_hall_number"] == 1 for result in results)
    assert all(
        result.sampling["hall_resolution_source"] == "representative_hall_from_spglib"
        for result in results
    )


def test_public_api_rejects_spacegroup_absent_from_checkpoint_vocab(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    _tiny_checkpoint(path)
    generator = GenIM.from_checkpoint(path, device="cpu")
    with pytest.raises(ValueError, match="No Hall token for space group 225"):
        generator.sample(
            config=SamplingConfig(n=1, max_sites=1, fixed_spacegroup=225),
            chemistry=ChemistryPolicy(mode="any", allowed_elements=["Na", "Cl"]),
        )


def test_training_writes_v2_provenance_and_validation_loss(tmp_path: Path) -> None:
    tokens = ["<PAD>", "<BOS>", "<EOS>", "E_Fe", "E_O"]
    vocab = {token: index for index, token in enumerate(tokens)}
    data_path = tmp_path / "tokens.pt"
    torch.save(
        {
            "sequences": torch.tensor(
                [
                    [1, 3, 2, 0],
                    [1, 4, 2, 0],
                    [1, 3, 4, 2],
                    [1, 4, 3, 2],
                ],
                dtype=torch.long,
            ),
            "vocab": vocab,
            "id_to_token": tokens,
            "config": {
                "coord_bins": 8,
                "len_bins": 8,
                "len_min": 2.0,
                "len_max": 10.0,
                "ang_bins": 8,
                "ang_min": 40.0,
                "ang_max": 140.0,
            },
            "stats": {"kept": 4, "skipped": 0},
            "provenance": {"source_name": "tiny.jsonl", "source_sha256": "abc"},
        },
        data_path,
    )
    out = tmp_path / "trained.pt"
    train_lm(
        data_path=data_path,
        out_ckpt=out,
        steps=1,
        batch_size=2,
        lr=1e-3,
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        element_emb="features",
        element_feature_set="periodic8",
        seed=7,
        val_fraction=0.25,
    )
    loaded = load_model_checkpoint(out, device="cpu")
    assert loaded.format_version == 2
    assert loaded.model_config.element_feature_set == "periodic8"
    assert loaded.metadata["train_size"] == 3
    assert loaded.metadata["validation_size"] == 1
    assert loaded.metadata["validation_loss"] is not None
    assert loaded.metadata["dataset_name"] == "tokens.pt"
