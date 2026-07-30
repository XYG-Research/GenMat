from __future__ import annotations

from genim.config import DEFAULT_CONFIG


def test_default_output_base_is_output_root() -> None:
    assert DEFAULT_CONFIG["paths"]["out_dir"] == "output"


def test_default_chemistry_is_universal_with_legacy_mode_available() -> None:
    assert DEFAULT_CONFIG["mp_download"]["chemistry_filter"] == "any"
    assert DEFAULT_CONFIG["preprocess"]["chemistry_mode"] == "any"
    assert DEFAULT_CONFIG["generate"]["chemistry_mode"] == "any"
    assert DEFAULT_CONFIG["generate"]["random_pool"] == "chemistry"
