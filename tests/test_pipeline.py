from __future__ import annotations

from pathlib import Path

from genmat.checkpoints import safe_torch_load
from genmat.decode import DecodeConfig, decode_tokens_to_atoms
from genmat.examples import make_examples_jsonl
from genmat.preprocess import preprocess_jsonl_to_tokens
from genmat.validate import validate_atoms


def test_examples_preprocess_and_decode(tmp_path: Path) -> None:
    jsonl = tmp_path / "examples.jsonl"
    tokens_pt = tmp_path / "examples.tokens.pt"

    make_examples_jsonl(jsonl)
    preprocess_jsonl_to_tokens(
        in_path=jsonl,
        out_path=tokens_pt,
        max_sites=15,
        symprec=1e-2,
        coord_bins=96,
        len_bins=180,
        len_min=2.0,
        len_max=20.0,
        ang_bins=181,
        ang_min=40.0,
        ang_max=140.0,
    )

    blob = safe_torch_load(tokens_pt)
    assert int(blob["stats"]["kept"]) > 0

    vocab = blob["vocab"]
    pad_id = int(vocab["<PAD>"])
    id_to_token = blob["id_to_token"]
    cfg = blob["config"]

    dec_cfg = DecodeConfig(
        coord_bins=int(cfg["coord_bins"]),
        len_bins=int(cfg["len_bins"]),
        len_min=float(cfg["len_min"]),
        len_max=float(cfg["len_max"]),
        ang_bins=int(cfg["ang_bins"]),
        ang_min=float(cfg["ang_min"]),
        ang_max=float(cfg["ang_max"]),
    )

    seq_ids = blob["sequences"][0].tolist()
    toks = [id_to_token[i] for i in seq_ids if i != pad_id]
    atoms = decode_tokens_to_atoms(toks, cfg=dec_cfg, max_sites=int(cfg["max_sites"]))

    ok, reason = validate_atoms(atoms, min_dist=0.5, symprec=1e-2)
    assert ok, reason
