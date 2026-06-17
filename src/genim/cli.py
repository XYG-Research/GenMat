from __future__ import annotations

import argparse
import copy
import shlex
import sys
import traceback as _tb
from pathlib import Path
from typing import Any

from ase.data import atomic_numbers

from .chem import allowed_intermetallic_elements
from .config import DEFAULT_CONFIG, SynthRequest, load_config, write_default_config
from .examples import make_examples_jsonl
from .generate import generate_cifs
from .inspect import InspectConfig, inspect_any
from .mp_download import download_mp_jsonl
from .preprocess import preprocess_jsonl_to_tokens
from .score import score_cif_dir
from .surface_screen import run_surface_screen, SurfaceScreenConfig
from .synth import synth_cifs
from .sym_seed import SymSeedConfig, write_symmetry_seeds_jsonl
from .train import train_lm
from .validate import validate_cif_dir


_RATIO_ARG_HELP = (
    "Optional composition control aligned with --elements order. "
    "Default interpretation is ratio, e.g. --ratios 1 1 1 1. "
    "Use --ratio-mode percent if the numbers are total-atom percentages, e.g. --ratios 25 25 25 25. "
    "Use X for listed elements whose composition should stay unconstrained, e.g. 1 1 X X or 25 25 X X."
)

_RATIO_MODE_HELP = (
    "Interpret --ratios: "
    "ratio=strict atomic-count ratio among listed elements; "
    "percent=listed elements' total-atom percentages; "
    "default is ratio. X means the corresponding listed element is left unconstrained."
)


def _p(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _try_load_config(conf_path: Path) -> dict[str, Any]:
    try:
        return load_config(conf_path)
    except FileNotFoundError:
        return copy.deepcopy(DEFAULT_CONFIG)
    except Exception as exc:
        print(f"[interactive] Failed to load config {conf_path}: {type(exc).__name__}: {exc}")
        return copy.deepcopy(DEFAULT_CONFIG)


def _find_latest_cif_dir(out_base: Path) -> Path | None:
    out_base = Path(out_base).expanduser().resolve()
    if not out_base.is_dir():
        return None
    cands: list[Path] = []
    for d in out_base.iterdir():
        if not d.is_dir():
            continue
        if any(d.glob("*.cif")) or any(d.glob("*.CIF")):
            cands.append(d)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _prompt_line(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _print_symbol_matrix(symbols: list[str], *, ncols: int = 5) -> None:
    symbols = [str(s).strip() for s in symbols if str(s).strip()]
    if not symbols:
        return
    ncols = max(1, int(ncols))
    col_w = max(len(s) for s in symbols) + 2
    for i in range(0, len(symbols), ncols):
        row = symbols[i : i + ncols]
        print("".join(s.ljust(col_w) for s in row).rstrip())


def _format_ratio_example(required_n: int, numeric_tokens: list[str]) -> str:
    vals = ["X"] * max(1, int(required_n))
    for i, token in enumerate(numeric_tokens[: len(vals)]):
        vals[i] = str(token)
    return " ".join(vals)


def _interactive_ratio_example(prefix: str, required_n: int, numeric_tokens: list[str]) -> str:
    return f"{prefix} {_format_ratio_example(required_n, numeric_tokens)}"


def _normalize_ratio_token(token: str) -> str:
    txt = str(token).strip()
    if not txt:
        raise ValueError("empty ratio token")
    if txt.lower() in {"x", "*"}:
        return "X"
    return str(float(txt))


def _parse_interactive_ratio_entry(text: str) -> tuple[list[str] | None, str | None]:
    txt = str(text).strip()
    if not txt:
        return None, None

    parts = txt.split()
    mode: str | None = None
    if parts and parts[0].lower() in {"r", "p"}:
        mode = "ratio" if parts[0].lower() == "r" else "percent"
        parts = parts[1:]
        if not parts:
            raise ValueError("missing ratio values after R/P prefix")

    values = [_normalize_ratio_token(x) for x in parts]
    return values, mode


def _print_ratio_input_guide(*, elements: list[str], nelements_total: int) -> None:
    required_n = len(elements)
    if required_n <= 0:
        return

    print("[ratio] Composition input is aligned with the --elements order.")
    ratio_full = _interactive_ratio_example("R", required_n, ["1"] * required_n)
    percent_token = "20" if nelements_total > required_n else "25"
    percent_full = _interactive_ratio_example("P", required_n, [percent_token] * required_n)
    ratio_partial = _interactive_ratio_example("R", required_n, ["1", "1"])
    percent_partial = _interactive_ratio_example("P", required_n, ["20" if nelements_total > required_n else "25", "20" if nelements_total > required_n else "25"])
    print("[ratio] Write one line as: R/P + values. Default is ratio if you omit the prefix.")
    print(f"[ratio]   ratio example:   {ratio_full}")
    print("[ratio]                    -> numeric entries are strict atomic-count ratios.")
    print(f"[ratio]   percent example: {percent_full}")
    print("[ratio]                    -> numeric entries are total-atom percentages.")
    print(f"[ratio]   with X:          {ratio_partial}")
    print(f"[ratio]                    or {percent_partial}")
    if nelements_total > required_n:
        print("[ratio]                    -> X-marked listed elements and extra elements share the remaining percentage.")
    else:
        print("[ratio]                    -> X-marked listed elements must appear, but their composition is unconstrained.")


def _interactive_menu() -> int:
    conf_path = _p("conf.yml")
    cfg = _try_load_config(conf_path)

    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    mp_jsonl = _p(str(paths.get("mp_jsonl", "data/mp_train.jsonl")))
    tokens_pt = _p(str(paths.get("tokens_pt", "data/mp_train.tokens.pt")))
    ckpt = _p(str(paths.get("ckpt", "checkpoints/mp_train_fullsg_60.pt")))
    out_base = _p(str(paths.get("out_dir", "output")))

    print("GenIM interactive mode")
    if conf_path.is_file():
        print(f"[config] Loaded: {conf_path}")
    else:
        print(f"[config] Not found: {conf_path} (some modules may fail; run `genim conf-init`)")
    print(f"[defaults] mp_jsonl={mp_jsonl}")
    print(f"[defaults] tokens_pt={tokens_pt}")
    print(f"[defaults] ckpt={ckpt}")
    print(f"[defaults] out_dir={out_base}")
    print("")

    menu: list[tuple[str, str]] = [
        ("gen", "Structure generation (alias: synth)"),
        ("mlip", "MLIP relax + Energy Above Hull CSV (alias: score)"),
        ("mp-download", "Download MP dataset (JSONL)"),
        ("preprocess", "Tokenize JSONL -> tokens.pt"),
        ("train", "Train Transformer LM"),
        ("inspect", "Inspect dataset (.jsonl or .pt)"),
        ("validate", "Validate CIFs under a directory"),
        ("snapshot-panel", "Render a 6-per-row perspective crystal snapshot panel from CIFs"),
        ("conf-init", "Write default conf.yml template"),
        ("generate", "Low-level sampling (writes CIFs)"),
        ("examples-make", "Create a tiny offline example dataset (JSONL)"),
        ("sym-seed", "Generate synthetic symmetry seed structures (JSONL)"),
    ]
    alias_to_cmd = {
        "synth": "gen",
        "score": "mlip",
        "exit": "quit",
        "q": "quit",
        "quit": "quit",
    }

    while True:
        print("Modules:")
        for i, (cmd, desc) in enumerate(menu, start=1):
            print(f"  {i:>2}. {cmd:<12} - {desc}")
        sel = _prompt_line("Select (number/name, q to quit): ").strip()
        if not sel:
            continue
        sel_norm = sel.lower().strip()
        sel_norm = alias_to_cmd.get(sel_norm, sel_norm)
        if sel_norm in {"quit"}:
            return 0

        cmd: str | None = None
        if sel_norm.isdigit():
            idx = int(sel_norm)
            if 1 <= idx <= len(menu):
                cmd = menu[idx - 1][0]
        else:
            for c, _ in menu:
                if sel_norm == c:
                    cmd = c
                    break
        if cmd is None:
            print(f"[interactive] Unknown selection: {sel!r}")
            print("")
            continue

        argv: list[str] | None = None
        hints: list[str] = []

        # --- Two "foolproof" modules ---
        if cmd == "gen":
            if not conf_path.is_file():
                write_default_config(conf_path)
                print(f"[interactive] Wrote default config: {conf_path}")

            inc_meta = True
            try:
                gen_sec = cfg.get("generate", {}) if isinstance(cfg, dict) else {}
                if isinstance(gen_sec, dict):
                    inc_meta = bool(gen_sec.get("include_metalloids", True))
            except Exception:
                inc_meta = True

            pool_all = allowed_intermetallic_elements(include_metalloids=inc_meta)
            pool = [s for s in pool_all if int(atomic_numbers.get(s, 999)) <= 84]  # up to Po (metalloid)

            rare: list[str] = []
            t3d: list[str] = []
            t4d: list[str] = []
            t5d: list[str] = []
            main_group: list[str] = []
            for sym in pool:
                z = int(atomic_numbers.get(sym, 0))
                if sym in {"Sc", "Y"} or (57 <= z <= 71):  # Sc/Y + lanthanides
                    rare.append(sym)
                elif sym in {"Zn", "Cd"}:  # treat as main-group (post-transition) for UI
                    main_group.append(f"({sym})")
                elif 21 <= z <= 30:  # 3d
                    t3d.append(sym)
                elif 39 <= z <= 48:  # 4d
                    t4d.append(sym)
                elif 72 <= z <= 80:  # 5d
                    t5d.append(sym)
                else:
                    main_group.append(sym)

            print(f"[interactive] Available intermetallic elements (include_metalloids={inc_meta})")
            n_trans = len(t3d) + len(t4d) + len(t5d)
            print(f"[Transition metals: 3d/4d/5d] (n={n_trans})")
            if t3d:
                print("3d:")
                _print_symbol_matrix(t3d, ncols=10)
            if t4d:
                print("4d:")
                _print_symbol_matrix(t4d, ncols=10)
            if t5d:
                print("5d:")
                _print_symbol_matrix(t5d, ncols=10)
            print("")
            print(f"[Main-group metals{' + metalloids' if inc_meta else ''}] (n={len(main_group)})")
            _print_symbol_matrix(main_group, ncols=10)
            print("")
            print(f"[Rare earth metals: Sc/Y/La-Lu] (n={len(rare)})")
            _print_symbol_matrix(rare, ncols=10)
            print("")

            elems = _prompt_line("Elements (space-separated, e.g. Fe Si): ").strip().split()
            if not elems:
                print("[interactive] Canceled.")
                print("")
                continue
            ne = _prompt_line("Total distinct element count (--nelements): ").strip()
            if not ne.isdigit():
                print("[interactive] Invalid --nelements.")
                print("")
                continue
            ne_total = int(ne)
            nmax = _prompt_line("Max unique CIFs (--n, blank=conf generate.n_max): ").strip()
            _print_ratio_input_guide(elements=elems, nelements_total=ne_total)
            ratios = _prompt_line(
                "Composition control (blank=none; format: R/P + values, e.g. `R 1 1 X X` or `P 20 20 20 20`): "
            ).strip()

            argv = ["gen", "--conf", str(conf_path), "--elements", *elems, "--nelements", str(ne_total)]
            if nmax:
                argv += ["--n", nmax]
            if ratios:
                try:
                    r, ratio_mode = _parse_interactive_ratio_entry(ratios)
                    if r is None:
                        raise ValueError("missing ratio values")
                except Exception:
                    print("[interactive] Invalid composition control; expected `R 1 1 X X` or `P 20 20 20 20`.")
                    print("")
                    continue
                argv += ["--ratios", *r]
                if ratio_mode is not None:
                    argv += ["--ratio-mode", ratio_mode]

            out_dir = out_base / f"{'-'.join(elems)}_{ne_total}el"
            hints = [f"Read conf: {conf_path}", f"Write CIFs under: {out_dir}"]

        elif cmd == "mlip":
            if not conf_path.is_file():
                write_default_config(conf_path)
                print(f"[interactive] Wrote default config: {conf_path}")

            suggested = _find_latest_cif_dir(out_base)
            default_cif_dir = str(suggested) if suggested is not None else ""
            cif_dir = _prompt_line(f"CIF directory (--cif-dir) [default: {default_cif_dir}]: ").strip()
            if not cif_dir:
                cif_dir = default_cif_dir
            if not cif_dir:
                print("[interactive] Missing --cif-dir.")
                print("")
                continue
            out_dir = _prompt_line("Score out dir (--out-dir, blank=default <cif_dir>/ml_score): ").strip()

            argv = ["mlip", "--conf", str(conf_path), "--cif-dir", cif_dir]
            if out_dir:
                argv += ["--out-dir", out_dir]
            default_score_dir = str((_p(cif_dir) / "ml_score").resolve())
            hints = [f"Read conf: {conf_path}", f"Read CIFs: {cif_dir}", f"Write scores under: {out_dir or default_score_dir}"]

        # --- Dataset / training pipeline ---
        elif cmd == "mp-download":
            outp = _prompt_line(f"Output JSONL (--out) [default: {mp_jsonl}]: ").strip()
            if not outp:
                outp = str(mp_jsonl)

            sec = cfg.get("mp_download", {}) if isinstance(cfg, dict) else {}
            if not isinstance(sec, dict):
                sec = {}

            argv = ["mp-download", "--out", outp]
            chemistry = sec.get("chemistry_filter", "intermetallic")
            argv += ["--chemistry", str(chemistry)]
            if sec.get("chemsys"):
                argv += ["--chemsys", str(sec["chemsys"])]
            if sec.get("elements"):
                argv += ["--elements", *[str(x) for x in sec["elements"]]]
            if sec.get("spacegroup_number") is not None:
                argv += ["--spacegroup-number", str(int(sec["spacegroup_number"]))]
            if bool(sec.get("balance_spacegroups", False)):
                argv += ["--balance-spacegroups"]
            argv += ["--per-spacegroup", str(int(sec.get("per_spacegroup", 50)))]
            argv += ["--max-atoms", str(int(sec.get("max_atoms", 200)))]
            argv += ["--eah-max", str(float(sec.get("eah_max", 1.0)))]
            argv += ["--nelements-min", str(int(sec.get("nelements_min", 2)))]
            argv += ["--nelements-max", str(int(sec.get("nelements_max", 5)))]
            argv += ["--limit", str(int(sec.get("limit", 100000)))]
            argv += ["--per-page", str(int(sec.get("per_page", 500)))]
            argv += ["--timeout", str(float(sec.get("timeout", 60.0)))]
            if bool(sec.get("include_metalloids", True)):
                argv += ["--include-metalloids"]

            hints = [f"Write JSONL: {outp}", f"Config source: {conf_path} (mp_download.*)"]

        elif cmd == "preprocess":
            inp = _prompt_line(f"Input JSONL (--in) [default: {mp_jsonl}]: ").strip()
            if not inp:
                inp = str(mp_jsonl)
            outp = _prompt_line(f"Output tokens (--out) [default: {tokens_pt}]: ").strip()
            if not outp:
                outp = str(tokens_pt)

            sec = cfg.get("preprocess", {}) if isinstance(cfg, dict) else {}
            if not isinstance(sec, dict):
                sec = {}

            argv = ["preprocess", "--in", inp, "--out", outp]
            argv += ["--max-sites", str(int(sec.get("max_sites", 25)))]
            argv += ["--symprec", str(float(sec.get("symprec", 1e-2)))]
            argv += ["--coord-bins", str(int(sec.get("coord_bins", 128)))]
            argv += ["--len-bins", str(int(sec.get("len_bins", 240)))]
            argv += ["--len-min", str(float(sec.get("len_min", 1.5)))]
            argv += ["--len-max", str(float(sec.get("len_max", 30.0)))]
            argv += ["--ang-bins", str(int(sec.get("ang_bins", 181)))]
            argv += ["--ang-min", str(float(sec.get("ang_min", 30.0)))]
            argv += ["--ang-max", str(float(sec.get("ang_max", 150.0)))]
            if bool(sec.get("seed_all_elements", True)):
                argv += ["--seed-all-elements"]
            if bool(sec.get("seed_all_hall", True)):
                argv += ["--seed-all-hall"]
            if bool(sec.get("include_metalloids", True)):
                argv += ["--include-metalloids"]

            hints = [f"Read JSONL: {inp}", f"Write tokens: {outp}", f"Config source: {conf_path} (preprocess.*)"]

        elif cmd == "train":
            data_in = _prompt_line(f"Input tokens (--data) [default: {tokens_pt}]: ").strip()
            if not data_in:
                data_in = str(tokens_pt)
            out_ckpt = _prompt_line(f"Output checkpoint (--out) [default: {ckpt}]: ").strip()
            if not out_ckpt:
                out_ckpt = str(ckpt)

            sec = cfg.get("train", {}) if isinstance(cfg, dict) else {}
            if not isinstance(sec, dict):
                sec = {}

            argv = ["train", "--data", data_in, "--out", out_ckpt]
            argv += ["--steps", str(int(sec.get("steps", 20000)))]
            argv += ["--batch", str(int(sec.get("batch", 64)))]
            argv += ["--lr", str(float(sec.get("lr", 3e-4)))]
            argv += ["--d-model", str(int(sec.get("d_model", 256)))]
            argv += ["--layers", str(int(sec.get("layers", 6)))]
            argv += ["--heads", str(int(sec.get("heads", 8)))]
            argv += ["--dropout", str(float(sec.get("dropout", 0.1)))]
            argv += ["--seed", str(int(sec.get("seed", 7)))]
            argv += ["--element-emb", str(sec.get("element_emb", "features"))]

            hints = [f"Read tokens: {data_in}", f"Write checkpoint: {out_ckpt}", f"Config source: {conf_path} (train.*)"]

        # --- Utils / advanced ---
        elif cmd == "inspect":
            inp = _prompt_line(f"Inspect path (--in) [default: {tokens_pt}]: ").strip()
            if not inp:
                inp = str(tokens_pt)
            topk = _prompt_line("Top-k (--top-k, blank=15): ").strip() or "15"
            argv = ["inspect", "--in", inp, "--top-k", topk]
            hints = [f"Read: {inp}"]

        elif cmd == "validate":
            suggested = _find_latest_cif_dir(out_base)
            default_cif_dir = str(suggested) if suggested is not None else ""
            cif_dir = _prompt_line(f"CIF directory (--cif-dir) [default: {default_cif_dir}]: ").strip()
            if not cif_dir:
                cif_dir = default_cif_dir
            if not cif_dir:
                print("[interactive] Missing --cif-dir.")
                print("")
                continue
            argv = ["validate", "--cif-dir", cif_dir]
            hints = [f"Read CIFs: {cif_dir}"]

        elif cmd == "snapshot-panel":
            suggested = _find_latest_cif_dir(out_base)
            default_cif_dir = str(suggested) if suggested is not None else ""
            cif_dir = _prompt_line(f"CIF directory (--cif-dir) [default: {default_cif_dir}]: ").strip()
            if not cif_dir:
                cif_dir = default_cif_dir
            if not cif_dir:
                print("[interactive] Missing --cif-dir.")
                print("")
                continue
            n_snap = _prompt_line("How many snapshots (--n, blank=12): ").strip() or "12"
            id_start = _prompt_line("ID start (--id-start, blank=none): ").strip()
            id_end = _prompt_line("ID end (--id-end, blank=none): ").strip()
            default_out = str((_p(cif_dir) / "snapshot_panel.png").resolve())
            outp = _prompt_line(f"Output PNG (--out) [default: {default_out}]: ").strip()
            argv = ["snapshot-panel", "--cif-dir", cif_dir, "--n", n_snap]
            if id_start:
                argv += ["--id-start", id_start]
            if id_end:
                argv += ["--id-end", id_end]
            if outp:
                argv += ["--out", outp]
            hints = [f"Read CIFs: {cif_dir}", f"Write panel: {outp or default_out}"]

        elif cmd == "conf-init":
            outp = _prompt_line("conf.yml path (--out, blank=conf.yml): ").strip()
            argv = ["conf-init"] + (["--out", outp] if outp else [])
            hints = [f"Write config: {outp or str(conf_path)}"]

        elif cmd == "generate":
            outp_default = out_base / "generate"
            ckpt_in = _prompt_line(f"Checkpoint (--ckpt) [default: {ckpt}]: ").strip()
            if not ckpt_in:
                ckpt_in = str(ckpt)
            n = _prompt_line("Max unique CIFs (--n, blank=200): ").strip() or "200"
            out_dir = _prompt_line(f"Output dir (--out-dir) [default: {outp_default}]: ").strip()
            if not out_dir:
                out_dir = str(outp_default)
            argv = ["generate", "--ckpt", ckpt_in, "--n", n, "--out-dir", out_dir]
            hints = [f"Read checkpoint: {ckpt_in}", f"Write CIFs: {out_dir}"]

        elif cmd == "examples-make":
            outp = _prompt_line("Output JSONL (--out, default=data/examples.jsonl): ").strip() or "data/examples.jsonl"
            argv = ["examples-make", "--out", outp]
            hints = [f"Write JSONL: {outp}"]

        elif cmd == "sym-seed":
            outp = _prompt_line("Output JSONL (--out): ").strip()
            if not outp:
                print("[interactive] Missing --out.")
                print("")
                continue
            sgs = _prompt_line("Spacegroups list (--spacegroups, e.g. 168 207): ").strip()
            if not sgs:
                print("[interactive] Missing --spacegroups.")
                print("")
                continue
            argv = ["sym-seed", "--out", outp, "--spacegroups", *sgs.split()]
            elems = _prompt_line("Elements (--elements, blank=Fe Si): ").strip()
            if elems:
                argv += ["--elements", *elems.split()]
            hints = [f"Write JSONL: {outp}"]

        else:
            print(f"[interactive] Unsupported cmd: {cmd}")
            print("")
            continue

        extra = _prompt_line("Extra args (optional, appended as-is): ").strip()
        if extra:
            try:
                argv += shlex.split(extra, posix=False)
            except Exception as exc:
                print(f"[interactive] Failed to parse extra args: {type(exc).__name__}: {exc}")
                print("")
                continue

        print("")
        for h in hints:
            print(f"[io] {h}")
        print(f"[run] genim {' '.join(argv)}")

        try:
            rc = main(argv=argv)
        except SystemExit as exc:
            rc = int(exc.code) if isinstance(exc.code, int) else 2
        except KeyboardInterrupt:
            print("[interactive] Interrupted.")
            print("")
            return 130
        except Exception as exc:
            print(f"[interactive] Failed: {type(exc).__name__}: {exc}")
            _tb.print_exc()
            print("")
            return 1

        print(f"[done] exit={rc}")
        print("")
        return int(rc)


def main(argv: list[str] | None = None) -> int:
    if argv is None and len(sys.argv) == 1:
        return _interactive_menu()

    parser = argparse.ArgumentParser(prog="genim", allow_abbrev=False)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("examples-make", help="Create a tiny offline example dataset (JSONL).", allow_abbrev=False)
    p_ex.add_argument("--out", required=True, type=_p)

    p_ins = sub.add_parser("inspect", help="Inspect a dataset (.jsonl or .pt) and print quick stats.", allow_abbrev=False)
    p_ins.add_argument("--in", dest="inp", required=True, type=_p)
    p_ins.add_argument("--top-k", type=int, default=15, help="Show top-k elements/hall numbers.")
    p_ins.add_argument("--wyckoff", action="store_true", help="For JSONL: run spglib and report Wyckoff/Hall stats (slower).")
    p_ins.add_argument("--symprec", type=float, default=1e-2)

    p_conf = sub.add_parser("conf-init", help="Write a default conf.yml template (all knobs in one place).", allow_abbrev=False)
    p_conf.add_argument("--out", type=_p, default=_p("conf.yml"))

    p_syn = sub.add_parser("synth", help="Generate intermetallic CIFs with minimal args (conf.yml-driven).", allow_abbrev=False)
    p_syn.add_argument("--conf", type=_p, default=_p("conf.yml"))
    p_syn.add_argument("--elements", nargs="+", required=True, help="Required element symbols, e.g. Fe Si")
    p_syn.add_argument("--nelements", type=int, required=True, help="Total distinct element count in the output")
    p_syn.add_argument("--n", type=int, default=None, help="Maximum number of *unique* CIFs to emit (overrides conf generate.n_max).")
    p_syn.add_argument("--ratios", nargs="+", default=None, help=_RATIO_ARG_HELP)
    p_syn.add_argument("--ratio-mode", choices=["ratio", "percent"], default=None, help=_RATIO_MODE_HELP)

    # Foolproof alias: "gen" == "synth"
    p_gen_alias = sub.add_parser("gen", help="Alias of synth (structure generation).", allow_abbrev=False)
    p_gen_alias.add_argument("--conf", type=_p, default=_p("conf.yml"))
    p_gen_alias.add_argument("--elements", nargs="+", required=True, help="Required element symbols, e.g. Fe Si")
    p_gen_alias.add_argument("--nelements", type=int, required=True, help="Total distinct element count in the output")
    p_gen_alias.add_argument("--n", type=int, default=None, help="Maximum number of *unique* CIFs to emit (overrides conf generate.n_max).")
    p_gen_alias.add_argument("--ratios", nargs="+", default=None, help=_RATIO_ARG_HELP)
    p_gen_alias.add_argument("--ratio-mode", choices=["ratio", "percent"], default=None, help=_RATIO_MODE_HELP)

    p_dl = sub.add_parser("mp-download", help="Download structures from Materials Project (JSONL).", allow_abbrev=False)
    p_dl.add_argument("--out", required=True, type=_p)
    p_dl.add_argument(
        "--chemistry",
        choices=["intermetallic", "any"],
        default="intermetallic",
        help="Local chemistry filter. 'intermetallic' enforces >=2 allowed elements (metals + optional metalloids). 'any' keeps any chemistry.",
    )
    p_dl.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help="Allowed elements set (local filter), e.g. Fe Ni Al. Use --chemsys for exact systems.",
    )
    p_dl.add_argument("--chemsys", default=None, help="Chemical system query like Fe-Ni-Al (server-side filter).")
    p_dl.add_argument("--max-atoms", type=int, default=80)
    p_dl.add_argument("--eah-max", type=float, default=0.25, help="energy_above_hull max (eV/atom) if field available.")
    p_dl.add_argument("--nelements-min", type=int, default=2)
    p_dl.add_argument("--nelements-max", type=int, default=4)
    p_dl.add_argument("--limit", type=int, default=5000, help="Max docs to download.")
    p_dl.add_argument("--per-page", type=int, default=200)
    p_dl.add_argument("--timeout", type=float, default=60.0)
    p_dl.add_argument("--include-metalloids", action="store_true")
    p_dl.add_argument("--spacegroup-number", type=int, default=None, help="Filter by spacegroup number (1..230).")
    p_dl.add_argument("--balance-spacegroups", action="store_true", help="Download a balanced dataset across all 230 spacegroups (uses spacegroup_number server-side filter).")
    p_dl.add_argument("--per-spacegroup", type=int, default=50, help="Target docs per spacegroup when --balance-spacegroups is set.")

    p_pp = sub.add_parser("preprocess", help="Tokenize JSONL structures into a Torch dataset.", allow_abbrev=False)
    p_pp.add_argument("--in", dest="inp", required=True, type=_p)
    p_pp.add_argument("--out", required=True, type=_p)
    p_pp.add_argument("--max-sites", type=int, default=15)
    p_pp.add_argument("--symprec", type=float, default=1e-2)
    p_pp.add_argument("--coord-bins", type=int, default=96)
    p_pp.add_argument("--len-bins", type=int, default=180)
    p_pp.add_argument("--len-min", type=float, default=2.0)
    p_pp.add_argument("--len-max", type=float, default=20.0)
    p_pp.add_argument("--ang-bins", type=int, default=181)
    p_pp.add_argument("--ang-min", type=float, default=40.0)
    p_pp.add_argument("--ang-max", type=float, default=140.0)
    p_pp.add_argument("--seed-all-elements", action="store_true", help="Seed vocab with all metal/metalloid element tokens (for better generalization).")
    p_pp.add_argument("--seed-all-hall", action="store_true", help="Seed vocab with all HALL_1..HALL_530 tokens (for broader symmetry coverage).")
    p_pp.add_argument("--include-metalloids", action="store_true", help="Only affects --seed-all-elements (whether to include metalloids).")

    p_ss = sub.add_parser("sym-seed", help="Generate synthetic symmetry seed structures (JSONL).", allow_abbrev=False)
    p_ss.add_argument("--out", required=True, type=_p)
    p_ss.add_argument("--spacegroups", nargs="+", type=int, required=True, help="Spacegroup numbers (1..230) to seed.")
    p_ss.add_argument("--elements", nargs="+", default=["Fe", "Si"], help="Element symbols to cycle through when assigning sites.")
    p_ss.add_argument("--n-sites", type=int, default=3, help="Number of Wyckoff representative sites per structure.")
    p_ss.add_argument("--n-per-sg", type=int, default=1, help="How many structures to generate per spacegroup.")
    p_ss.add_argument("--max-atoms", type=int, default=200)
    p_ss.add_argument("--min-dist", type=float, default=1.5)
    p_ss.add_argument("--symprec", type=float, default=1e-2)
    p_ss.add_argument("--seed", type=int, default=7)

    p_tr = sub.add_parser("train", help="Train a causal Transformer LM on token sequences.", allow_abbrev=False)
    p_tr.add_argument("--data", required=True, type=_p)
    p_tr.add_argument("--out", required=True, type=_p)
    p_tr.add_argument("--steps", type=int, default=2000)
    p_tr.add_argument("--batch", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--d-model", type=int, default=256)
    p_tr.add_argument("--layers", type=int, default=6)
    p_tr.add_argument("--heads", type=int, default=8)
    p_tr.add_argument("--dropout", type=float, default=0.1)
    p_tr.add_argument("--element-emb", choices=["token", "features"], default="features", help="How to embed/predict element tokens (generalization vs. capacity tradeoff).")
    p_tr.add_argument("--seed", type=int, default=7)

    p_gen = sub.add_parser("generate", help="Sample structures and write CIFs.", allow_abbrev=False)
    p_gen.add_argument("--ckpt", required=True, type=_p)
    p_gen.add_argument("--n", type=int, default=100, help="Maximum number of *unique* CIFs to emit.")
    p_gen.add_argument("--out-dir", required=True, type=_p)
    p_gen.add_argument("--max-sites", type=int, default=15)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-k", type=int, default=0)
    p_gen.add_argument("--elements", nargs="+", default=None, help="Restrict element tokens during sampling.")
    p_gen.add_argument("--nelements-min", type=int, default=2)
    p_gen.add_argument("--nelements-max", type=int, default=None)
    p_gen.add_argument("--include-metalloids", action="store_true")
    p_gen.add_argument("--substitute-elements", nargs="+", default=None, help="Post-hoc substitute the generated species set with these elements (keeps prototype, changes chemistry).")

    p_val = sub.add_parser("validate", help="Validate generated CIFs quickly.", allow_abbrev=False)
    p_val.add_argument("--cif-dir", required=True, type=_p)
    p_val.add_argument("--min-dist", type=float, default=1.5)
    p_val.add_argument("--symprec", type=float, default=1e-2)

    p_snap = sub.add_parser("snapshot-panel", help="Render a tiled perspective crystal snapshot panel from CIFs.", allow_abbrev=False)
    p_snap.add_argument("--cif-dir", required=True, type=_p)
    p_snap.add_argument("--out", type=_p, default=None, help="Output PNG path (default: <cif_dir>/snapshot_panel.png).")
    p_snap.add_argument("--n", type=int, default=12, help="How many structures to snapshot.")
    p_snap.add_argument("--seed", type=int, default=7, help="Random seed for default random sampling.")
    p_snap.add_argument("--id-start", type=int, default=None, help="Inclusive lower bound of structure ID.")
    p_snap.add_argument("--id-end", type=int, default=None, help="Inclusive upper bound of structure ID.")
    p_snap.add_argument("--cols", type=int, default=6, help="How many tiles per row.")
    p_snap.add_argument("--size", type=int, default=320, help="Square snapshot size in px.")
    p_snap.add_argument("--label-h", type=int, default=24, help="Label area height in px.")
    p_snap.add_argument("--pad", type=int, default=3, help="Gap between tiles in px.")
    p_snap.add_argument("--margin", type=int, default=4, help="Panel outer margin in px.")
    p_snap.add_argument("--supersample", type=int, default=3, help="Supersampling factor for cleaner lines.")
    p_snap.add_argument("--atom-scale", type=float, default=0.78, help="Atom sphere radius multiplier for the perspective view.")
    p_snap.add_argument("--cell-line-width", type=int, default=3, help="Cell line width in px.")
    p_snap.add_argument("--tiles-dir", type=_p, default=None, help="Optional directory to also write per-structure tiles.")

    p_sc = sub.add_parser("score", help="Relax CIFs with an MLIP and compute ML-based Energy Above Hull (CSV).", allow_abbrev=False)
    p_sc.add_argument("--conf", type=_p, default=_p("conf.yml"))
    p_sc.add_argument("--cif-dir", required=True, type=_p)
    p_sc.add_argument("--out-dir", type=_p, default=None)

    p_surf = sub.add_parser("surface-screen", help="Enumerate low-complexity facets/terminations and optionally MLIP-screen slabs.", allow_abbrev=False)
    p_surf.add_argument("--input-dir", type=_p, default=_p("."))
    p_surf.add_argument("--out-dir", type=_p, default=_p("output/surface_screen"))
    p_surf.add_argument("--recursive", action="store_true")
    p_surf.add_argument("--max-index", type=int, default=2)
    p_surf.add_argument("--min-d-hkl", type=float, default=1.2)
    p_surf.add_argument("--min-slab-size", type=float, default=12.0)
    p_surf.add_argument("--min-vacuum-size", type=float, default=15.0)
    p_surf.add_argument("--max-slab-thickness-a", type=float, default=15.0)
    p_surf.add_argument("--max-surface-area", type=float, default=160.0)
    p_surf.add_argument("--min-surface-area", type=float, default=1.0)
    p_surf.add_argument("--layer-tol-a", type=float, default=0.45)
    p_surf.add_argument("--max-terminations-per-facet", type=int, default=0)
    p_surf.add_argument("--write-all-initial-slabs", action="store_true")
    p_surf.add_argument("--max-compensated-slab-thickness-a", type=float, default=15.0)
    p_surf.add_argument("--no-modeling-slabs", action="store_true")
    p_surf.add_argument("--no-inplane-rectangularize", action="store_true")
    p_surf.add_argument("--max-inplane-area-multiplier", type=int, default=4)
    p_surf.add_argument("--orthogonal-angle-tol-deg", type=float, default=2.0)
    p_surf.add_argument("--run-mlip", action="store_true")
    p_surf.add_argument("--ml-model", type=str, default="eSEN-30M-MPtrj")
    p_surf.add_argument("--ml-checkpoint", type=str, default=None)
    p_surf.add_argument("--ml-device", type=str, default="auto")
    p_surf.add_argument("--ml-optimizer", type=str, default="LBFGS")
    p_surf.add_argument("--ml-fmax", type=float, default=0.08)
    p_surf.add_argument("--ml-steps", type=int, default=120)
    p_surf.add_argument("--ml-maxstep", type=float, default=0.04)
    p_surf.add_argument("--relax-surface-layers", type=int, default=2)
    p_surf.add_argument("--max-displacement-a", type=float, default=1.0)
    p_surf.add_argument("--surface-rmsd-a", type=float, default=0.6)
    p_surf.add_argument("--interlayer-rel-change", type=float, default=0.35)

    # Foolproof alias: "mlip" == "score"
    p_mlip_alias = sub.add_parser("mlip", help="Alias of score (MLIP relax + energy/hull CSV).", allow_abbrev=False)
    p_mlip_alias.add_argument("--conf", type=_p, default=_p("conf.yml"))
    p_mlip_alias.add_argument("--cif-dir", required=True, type=_p)
    p_mlip_alias.add_argument("--out-dir", type=_p, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "examples-make":
        make_examples_jsonl(args.out)
        return 0
    if args.cmd == "inspect":
        inspect_any(args.inp, cfg=InspectConfig(top_k=args.top_k, wyckoff=args.wyckoff, symprec=args.symprec))
        return 0
    if args.cmd == "conf-init":
        write_default_config(args.out)
        print(f"Wrote default config: {args.out}")
        return 0
    if args.cmd in {"synth", "gen"}:
        synth_cifs(
            conf_path=args.conf,
            req=SynthRequest(
                required_elements=args.elements,
                nelements_total=args.nelements,
                ratios=args.ratios,
                ratio_mode=args.ratio_mode,
                n_max=args.n,
            ),
        )
        return 0
    if args.cmd == "mp-download":
        stats = download_mp_jsonl(
            out_path=args.out,
            chemsys=args.chemsys,
            elements=args.elements,
            chemistry_filter=args.chemistry,
            max_atoms=args.max_atoms,
            eah_max=args.eah_max,
            nelements_min=args.nelements_min,
            nelements_max=args.nelements_max,
            limit=args.limit,
            per_page=args.per_page,
            timeout=args.timeout,
            include_metalloids=args.include_metalloids,
            spacegroup_number=args.spacegroup_number,
            balance_spacegroups=args.balance_spacegroups,
            per_spacegroup=args.per_spacegroup,
        )
        msg = f"MP download finished: kept={stats.kept} skipped={stats.skipped} out={args.out}"
        if stats.spacegroups_target is not None:
            msg += f" spacegroups_covered={stats.spacegroups_covered}/{stats.spacegroups_target}"
        print(msg)
        return 0
    if args.cmd == "preprocess":
        preprocess_jsonl_to_tokens(
            in_path=args.inp,
            out_path=args.out,
            max_sites=args.max_sites,
            symprec=args.symprec,
            coord_bins=args.coord_bins,
            len_bins=args.len_bins,
            len_min=args.len_min,
            len_max=args.len_max,
            ang_bins=args.ang_bins,
            ang_min=args.ang_min,
            ang_max=args.ang_max,
            seed_all_elements=args.seed_all_elements,
            seed_all_hall=args.seed_all_hall,
            include_metalloids=args.include_metalloids,
        )
        return 0
    if args.cmd == "sym-seed":
        write_symmetry_seeds_jsonl(
            out_path=args.out,
            spacegroups=[int(x) for x in args.spacegroups],
            elements=[str(x) for x in args.elements],
            cfg=SymSeedConfig(
                n_sites=int(args.n_sites),
                n_per_sg=int(args.n_per_sg),
                max_atoms=int(args.max_atoms),
                min_dist=float(args.min_dist),
                symprec=float(args.symprec),
                seed=int(args.seed),
            ),
        )
        print(f"Wrote symmetry seeds: {args.out}")
        return 0
    if args.cmd == "train":
        train_lm(
            data_path=args.data,
            out_ckpt=args.out,
            steps=args.steps,
            batch_size=args.batch,
            lr=args.lr,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            dropout=args.dropout,
            element_emb=args.element_emb,
            seed=args.seed,
        )
        return 0
    if args.cmd == "generate":
        generate_cifs(
            ckpt_path=args.ckpt,
            n_samples=args.n,
            out_dir=args.out_dir,
            max_sites=args.max_sites,
            temperature=args.temperature,
            top_k=args.top_k,
            restrict_elements=args.elements,
            nelements_min=args.nelements_min,
            nelements_max=args.nelements_max,
            include_metalloids=args.include_metalloids,
            substitute_elements=args.substitute_elements,
        )
        return 0
    if args.cmd == "validate":
        ok = validate_cif_dir(args.cif_dir, min_dist=args.min_dist, symprec=args.symprec)
        return 0 if ok else 2
    if args.cmd == "snapshot-panel":
        from .snapshot_panel import render_snapshot_panel

        out_path = render_snapshot_panel(
            cif_dir=args.cif_dir,
            out_path=args.out,
            n=args.n,
            seed=args.seed,
            id_start=args.id_start,
            id_end=args.id_end,
            cols=args.cols,
            size_px=args.size,
            label_h_px=args.label_h,
            pad_px=args.pad,
            margin_px=args.margin,
            supersample=args.supersample,
            atom_scale=args.atom_scale,
            cell_line_w_px=args.cell_line_width,
            tiles_dir=args.tiles_dir,
        )
        print(f"Wrote snapshot panel: {out_path}")
        return 0
    if args.cmd in {"score", "mlip"}:
        csv_path = score_cif_dir(conf_path=args.conf, cif_dir=args.cif_dir, out_dir=args.out_dir)
        print(f"Wrote scores: {csv_path}")
        return 0
    if args.cmd == "surface-screen":
        summary = run_surface_screen(
            SurfaceScreenConfig(
                input_dir=args.input_dir,
                out_dir=args.out_dir,
                recursive=bool(args.recursive),
                max_index=int(args.max_index),
                min_d_hkl=float(args.min_d_hkl),
                min_slab_size=float(args.min_slab_size),
                min_vacuum_size=float(args.min_vacuum_size),
                max_slab_thickness_a=float(args.max_slab_thickness_a),
                max_surface_area=float(args.max_surface_area),
                min_surface_area=float(args.min_surface_area),
                layer_tol_a=float(args.layer_tol_a),
                max_terminations_per_facet=int(args.max_terminations_per_facet),
                write_all_initial_slabs=bool(args.write_all_initial_slabs),
                max_compensated_slab_thickness_a=float(args.max_compensated_slab_thickness_a),
                write_modeling_slabs=not bool(args.no_modeling_slabs),
                inplane_rectangularize=not bool(args.no_inplane_rectangularize),
                max_inplane_area_multiplier=int(args.max_inplane_area_multiplier),
                orthogonal_angle_tol_deg=float(args.orthogonal_angle_tol_deg),
                run_mlip=bool(args.run_mlip),
                ml_model=str(args.ml_model),
                ml_checkpoint=args.ml_checkpoint,
                ml_device=str(args.ml_device),
                ml_optimizer=str(args.ml_optimizer),
                ml_fmax=float(args.ml_fmax),
                ml_steps=int(args.ml_steps),
                ml_maxstep=float(args.ml_maxstep),
                relax_surface_layers=int(args.relax_surface_layers),
                max_displacement_a=float(args.max_displacement_a),
                surface_rmsd_a=float(args.surface_rmsd_a),
                interlayer_rel_change=float(args.interlayer_rel_change),
            )
        )
        print(json.dumps(summary, indent=2))
        return 0

    raise RuntimeError(f"Unknown cmd: {args.cmd}")
