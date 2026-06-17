# GenIM: Generative Intermetallic Crystal Structures (MVP)

**English** | [简体中文](README.zh-CN.md)

GenIM trains a generative structure builder on Materials Project (MP) data to quickly produce physically plausible intermetallic crystal structures.

This repository is a **runnable MVP** that follows the core ideas of two related works (symmetry/Wyckoff representation + autoregressive Transformer sampling + post-generation filtering; see [References](#references)), packaged as a reusable Python library and CLI.

## Features

- Pull structures from the MP API (requires `MP_API_KEY`).
- Standardize structures with `spglib` and extract a **Hall number / Wyckoff site** representation.
- Encode each structure into a token sequence and train a **Causal Transformer LM** for autoregressive generation.
- Decode generated sequences back into 3D periodic structures (ASE `Atoms`) and run fast acceptance filtering (minimum interatomic distance, duplicates/clashes, recognizable symmetry, etc.).
- Generation enables **physically motivated geometric constraints** by default: connectivity checks + volume/bond-length autoscaling (avoids fragmented clusters / over-sparse lattices).

## Requirements

- Python 3.9+
- Core: `torch`, `ase`, `spglib`, `requests`, `numpy`, `pyyaml`, `tqdm`, `matplotlib`, `Pillow`
- Optional extras: `genim[mlip]` (fairchem-core for MLIP relaxation), `genim[hull]` (pymatgen + scipy for energy-above-hull and surface screening), `genim[all]` for everything.

Install (editable):

```bash
python -m pip install -e .
# full features:
python -m pip install -e ".[all]"
```

## Configuration (conf.yml)

All tunable parameters live in `conf.yml` (a default config is provided at the repo root). You can also regenerate a template:

```bash
genim conf-init --out conf.yml
```

Common groups:

- Dataset download: `mp_download.*`
- Preprocessing: `preprocess.*` (`seed_all_elements` / `seed_all_hall` improve cross-element/symmetry generalization)
- Training: `train.*` (`element_emb: features` uses periodic-table features for element embedding/prediction)
- Generation: `generate.*` (`n_max`, deduplication, space-group sampling, ...)
- Geometric enhancement: `generate.autoscale_cell`, `generate.prototype_mode`, `validate.max_dist_factor`, `validate.require_connected`

## Quick start (offline example)

Run the full pipeline on a few built-in example structures (no MP key needed):

```bash
genim examples-make --out data/examples.jsonl
genim preprocess --in data/examples.jsonl --out data/examples.tokens.pt
genim train --data data/examples.tokens.pt --out checkpoints/example.pt --steps 200
genim generate --ckpt checkpoints/example.pt --n 10 --out-dir output/cif
genim validate --cif-dir output/cif
```

## Minimal generation: elements + element count

Specify only the required elements and the total number of element species; everything else comes from `conf.yml`:

```bash
genim synth --elements Fe Si --nelements 2
genim synth --elements Fe Si --nelements 3
```

- `--nelements 2`: binary structures containing only Fe/Si.
- `--nelements 3`: ternary structures containing Fe/Si, with the third element drawn from the intermetallic element pool.
- `generate.n_max` is the maximum count; if there are not enough unique structures, it outputs as many unique ones as possible (no error).
- Strict deduplication is on by default (`generate.dedup.*`).
- Default `generate.prototype_mode: target` samples directly in the target element system (more reasonable lattice/bond lengths); set to `random` for more prototype diversity.
- Default `generate.hall_mode: model` (quality first); set to `uniform_230` to cover all 230 space groups.

### Composition ratios

`--ratios` is always written in `--elements` order:

```bash
# 1) nelements == len(elements): atom-count ratio
genim synth --elements Ni Fe Co Al --nelements 4 --ratios 1 1 1 1
# Ni:Fe:Co:Al = 1:1:1:1

# 2) some listed elements only required to appear, composition free: use X
genim synth --elements Ni Fe Co Al --nelements 4 --ratios 1 1 X X --ratio-mode ratio
# Ni:Fe strictly 1:1; Co and Al must appear but with free fractions

# 3) percent: numeric slots are total-atom percentages, X is free
genim synth --elements Ni Fe Co Al --nelements 4 --ratios 25 25 X X --ratio-mode percent
# Ni 25 at.%, Fe 25 at.%; Co and Al share the remaining 50 at.%

# 4) nelements > len(elements), extra elements free
genim synth --elements Fe Si --nelements 3 --ratios 30 30 --ratio-mode percent
# Fe 30 at.%, Si 30 at.%; the 3rd element gets the remaining 40 at.%

# 5) no composition constraint
genim synth --elements Fe Si --nelements 3
```

- The number of `--ratios` must equal the number of `--elements`.
- `X` means "this element must appear, but its fraction is not fixed".
- `--ratio-mode ratio`: constrain only the relative atom counts of numeric slots.
- `--ratio-mode percent`: numeric slots are total-atom percentages; all `X` slots and extra elements share the remainder.
- Default interpretation is `ratio`; numeric slots are treated as percentages only when you pass `--ratio-mode percent`.

## Dataset coverage check (recommended)

For more general intermetallic generation, inspect training-set coverage first (elements, Hall numbers, number of Wyckoff representative sites, ...):

```bash
genim inspect --in data/mp_train.jsonl
genim inspect --in data/mp_train.jsonl --wyckoff
genim inspect --in data/mp_train.tokens.pt
```

## Using Materials Project as the training set

1) Set the API key:

```bash
export MP_API_KEY="your-mp-key"   # or PMG_MAPI_KEY
```

To avoid leaving the key in your shell history, write it into a local `.mp_api_key` file (one key per line); the tool reads it automatically. This file is gitignored.

2) Download (example: restrict element set + structure size):

```bash
genim mp-download --elements Fe Ni Al --max-atoms 80 --limit 5000 --out data/mp.jsonl
```

3) Preprocess / train / generate as above.

### Intermetallic constraints at generation time

`genim generate` applies two post-generation filters by default:

- Fast geometric/symmetry acceptance (minimum interatomic distance, recognizable space group, ...).
- **Intermetallic filter**: requires **at least 2 elements** and excludes common non-metals/halogens (use `--include-metalloids` to allow metalloids).

Strict deduplication is on by default, and `--n` is interpreted as "max number of **unique** structures". If constraints are too strict to reach `--n`, the program outputs as many unique structures as possible and reports rejection-reason statistics (no exception).

To better match a binary/ternary intermetallic distribution:

```bash
genim generate --ckpt checkpoints/mp_train.pt --n 200 --out-dir output/mp_cif --nelements-max 3
```

### "Prototype generation → element substitution" (more general)

To generate element systems that are rare/absent in the training set (e.g. metalloids such as Si/Ge), a more robust approach is:

1) let the model generate **structural prototypes** (space group / Wyckoff / coordinates);
2) use `--substitute-elements` to replace the element set of the generated structures (keeping the prototype).

Example (generate Fe–Si binary prototypes and substitute FeSi):

```bash
genim generate --ckpt checkpoints/mp_train_fullsg_60.pt --n 20 --out-dir output/demo_FeSi --nelements-min 2 --nelements-max 2 --include-metalloids --substitute-elements Fe Si
```

## Notes (MVP trade-offs)

To get the end-to-end pipeline running first, this version discretizes continuous variables (lattice parameters and Wyckoff coordinates) into **binned tokens**. For stronger performance closer to the reference papers, the coordinates/lattice can be modeled with continuous density (Gaussian embedding + mixture density head, etc.).

For maximum generality / broader intermetallic coverage:

- `mp-download`: increase `--limit`, relax `--nelements-max/--max-atoms/--eah-max`, add `--include-metalloids` as needed.
- `preprocess`: increase `--max-sites` as needed (longer sequences, higher training cost); add `--seed-all-elements/--seed-all-hall` for better cross-element/symmetry generalization.
- `train`: use `--element-emb features` to enable periodic-table feature-based element embedding/prediction.

## MLIP relaxation + Energy Above Hull scoring

`genim score` (alias `genim mlip`) relaxes generated structures (cell + atoms) with **eSEN/OMAT24 (fairchem-core)**, automatically builds an ML reference set for the same chemical system (MP structures + the same MLIP energies), computes `Energy Above Hull (eV/atom)` per structure, and writes a CSV. The CSV also reports `composition_ratio`, `composition_percent`, and `composition_counts`.

```bash
genim synth --elements Fe Si --nelements 2
genim score --conf conf.yml --cif-dir output/Fe-Si_2el

# or the two-step aliases:
genim gen --elements Fe Si --nelements 2
genim mlip --conf conf.yml --cif-dir output/Fe-Si_2el
```

> Requires the optional extras: `pip install ".[all]"` (fairchem-core + pymatgen + scipy).

## Interactive mode

Run `genim` with no subcommand to enter an interactive menu. Press Enter to accept defaults (loaded from `conf.yml`).

```bash
genim
```

The menu prints which paths it reads/writes, runs one selected module, then exits. For composition control in interactive mode, write one line as `R/P + values`, e.g. `R 1 1 X X` or `P 20 20 20 20`.

## Structure snapshot panel

`genim snapshot-panel` samples a fixed number of structures from a generated CIF directory into a snapshot panel. Sampling is random by default, supports filtering by ID range, 6 per row, with labels like `009-Fe5Co5Ni5Al5` (the number is the actual atom count in the cell). Each tile is rendered in a perspective crystal view with atoms and unit-cell lattice lines visible at the same time.

```bash
genim snapshot-panel --cif-dir output/Fe-Co-Ni-Al_4el --n 12
genim snapshot-panel --cif-dir output/Fe-Co-Ni-Al_4el --n 12 --id-start 9 --id-end 40
```

Configuration lives in `conf.yml`:

- `ml.*`: choose the eSEN model/device; point `ml.checkpoint` at a local `esen_30m_oam.pt` to avoid HF download/gating.
- `relax.*`: bulk relaxation settings (ASE + ExpCellFilter/UnitCellFilter).
- `hull.*`: MP reference pool + reference relaxation; `stable_threshold_eV_per_atom: 0.2` is the 200 meV/atom stability threshold.

## Pretrained weights & data download

To keep the repository lightweight, the training data (`data/`) and pretrained weights (`checkpoints/`) are **not committed with the source**; they are distributed as [GitHub Releases](https://github.com/XYG-Research/GenIM/releases) assets:

- `mp_train_fullsg_60.pt`: Causal Transformer weights trained on the full 230-space-group coverage + intermetallic merged set.
- `mp_train.jsonl` / `mp_train.tokens.pt`: MP-derived training structures and their tokenized dataset (sourced from Materials Project, subject to its terms of use).

Download and place them back into the corresponding directories (default paths in `conf.yml` under `paths.*`):

```bash
mkdir -p checkpoints
mv mp_train_fullsg_60.pt checkpoints/
```

You can also **reproduce from scratch**: use `genim mp-download` (with your own `MP_API_KEY`), then `preprocess` → `train`. See [`ACCEPTANCE.md`](ACCEPTANCE.md).

## References

This project's methodology is inspired by the following works (both in *npj Computational Materials*):

1. DOI: [10.1038/s41524-025-01881-2](https://doi.org/10.1038/s41524-025-01881-2)
2. DOI: [10.1038/s41524-025-01940-8](https://doi.org/10.1038/s41524-025-01940-8)

## License

Released under the [BSD 3-Clause License](LICENSE).
