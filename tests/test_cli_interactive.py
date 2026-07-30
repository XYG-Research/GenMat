from __future__ import annotations

from pathlib import Path

import genim.cli as cli


def test_interactive_gen_accepts_explicit_percent_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "available_elements", lambda **kwargs: ["Fe", "Si", "Al"])
    monkeypatch.setattr(cli, "write_default_config", lambda path: None)

    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        captured["argv"] = argv
        return 0

    answers = iter(
        [
            "gen",
            "Fe Si",
            "3",
            "",
            "P 30 30",
            "",
        ]
    )

    monkeypatch.setattr(cli, "main", fake_main)
    monkeypatch.setattr(cli, "_prompt_line", lambda prompt: next(answers))

    rc = cli._interactive_menu()

    assert rc == 0
    assert captured["argv"] == [
        "gen",
        "--conf",
        str((tmp_path / "conf.yml").resolve()),
        "--elements",
        "Fe",
        "Si",
        "--nelements",
        "3",
        "--ratios",
        "30.0",
        "30.0",
        "--ratio-mode",
        "percent",
    ]


def test_interactive_gen_accepts_x_placeholders(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "available_elements", lambda **kwargs: ["Ni", "Fe", "Co", "Al"])
    monkeypatch.setattr(cli, "write_default_config", lambda path: None)

    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        captured["argv"] = argv
        return 0

    answers = iter(
        [
            "gen",
            "Ni Fe Co Al",
            "4",
            "",
            "R 1 1 X X",
            "",
        ]
    )

    monkeypatch.setattr(cli, "main", fake_main)
    monkeypatch.setattr(cli, "_prompt_line", lambda prompt: next(answers))

    rc = cli._interactive_menu()

    assert rc == 0
    assert captured["argv"] == [
        "gen",
        "--conf",
        str((tmp_path / "conf.yml").resolve()),
        "--elements",
        "Ni",
        "Fe",
        "Co",
        "Al",
        "--nelements",
        "4",
        "--ratios",
        "1.0",
        "1.0",
        "X",
        "X",
        "--ratio-mode",
        "ratio",
    ]


def test_interactive_gen_accepts_explicit_percent_for_full_numeric_input(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "available_elements", lambda **kwargs: ["Fe", "Co", "Ni", "Al"])
    monkeypatch.setattr(cli, "write_default_config", lambda path: None)

    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        captured["argv"] = argv
        return 0

    answers = iter(
        [
            "gen",
            "Fe Co Ni Al",
            "4",
            "20",
            "P 25 25 25 25",
            "",
        ]
    )

    monkeypatch.setattr(cli, "main", fake_main)
    monkeypatch.setattr(cli, "_prompt_line", lambda prompt: next(answers))

    rc = cli._interactive_menu()

    assert rc == 0
    assert captured["argv"] == [
        "gen",
        "--conf",
        str((tmp_path / "conf.yml").resolve()),
        "--elements",
        "Fe",
        "Co",
        "Ni",
        "Al",
        "--nelements",
        "4",
        "--n",
        "20",
        "--ratios",
        "25.0",
        "25.0",
        "25.0",
        "25.0",
        "--ratio-mode",
        "percent",
    ]
