# GenIM: symmetry-aware generative crystal structures

**English** | [简体中文](README.zh-CN.md)

GenIM is a Python package and command-line toolkit for building, generating,
validating, benchmarking, and screening periodic crystal structures. Version
0.3 adds auditable multi-model proposal backends. Version 0.2 made the chemistry
domain explicit and general-purpose: oxides, nitrides,
halides, carbides, semiconductors, elemental solids, and intermetallics can use
the same Hall/Wyckoff generation pipeline.

The representation combines space-group symmetry, Wyckoff sites, discretized
lattice/coordinate tokens, and a causal Transformer. Generated candidates are
decoded to ASE `Atoms`, checked geometrically and crystallographically, deduplicated,
and optionally relaxed/scored with an ML interatomic potential.

## What changed in 0.3

- `GenerationBackend`, `GenerationConstraints`, `GeneratedCandidate`, and
  `EnsembleGenerator` define a model-neutral generation and provenance contract.
  The original 0.3 names remain compatibility aliases.
- `GenIMBackend` and the optional `MatraBackend` run through the same ASE
  conversion, validation, condition audit, and cross-model deduplication path.
- `AlgorithmicSeedBackend` and `GeneratorService` provide a reproducible,
  composition-exact default when no compatible checkpoint is configured,
  without presenting the result as a learned-model prediction.
- The optional HTTP API exposes `/v1/health`, `/v1/capabilities`, and
  `/v1/generate` with the same schema-version-2 candidate records used by Python.
- Matra checkpoints are loaded with PyTorch weights-only mode, SHA256
  verification, schema checks, and exact reconstructed-model weight matching.
- `genim generate-ensemble` produces selected CIFs, a complete `candidates.jsonl`
  audit trail, and a source-aware `ensemble-report.json`.
- Matra conditioning supports stability, exact elements, stoichiometry, space
  group, checkpoint-specific Wyckoff indices, and continuous hull targets.
  Unsupported or scientifically unevaluated constraints remain explicit in
  every candidate record.

## What changed in 0.2

- `ChemistryPolicy(mode="any")` is now the default. `metallic` and
  `intermetallic` are explicit compatibility modes.
- Vocabulary seeding covers all 118 elements in `any` mode and supports exact
  allow/deny lists.
- `GenIM.from_checkpoint(...)` provides a stable Python API with batched
  grammar-constrained sampling.
- New checkpoints have a versioned schema, SHA256 verification, training/data
  provenance, and strict weight/config compatibility checks. Legacy v1
  checkpoints remain loadable.
- New models default to `periodic8` element descriptors: atomic number,
  covalent radius, mass, period, group, and broad chemical-class indicators.
  Legacy feature checkpoints retain their original four-dimensional projection.
- `ValidationReport` exposes decision metrics, while `genim benchmark` reports
  validity, uniqueness, formula/element coverage, and space-group coverage.
- CI and a tracked regression suite cover chemistry policies, legacy checkpoint
  loading, batch sampling, generation provenance, and the existing workflows.

## Scientific scope

GenIM proposes symmetry-consistent candidates; it does **not** prove that a
candidate is synthesizable, dynamically stable, or the thermodynamic ground
state. Use the stages according to the question:

1. grammar/decode checks establish structural well-formedness;
2. geometric and symmetry validation rejects obvious failures;
3. deduplication measures novelty within the generated set;
4. MLIP relaxation and energy-above-hull provide a model-dependent screen;
5. DFT, phonons, finite-temperature analysis, and experimental judgment remain
   necessary for strong scientific claims.

An old checkpoint trained only on intermetallic data does not become a reliable
oxide generator merely because `chemistry_mode` is changed to `any`. General
chemistry requires a representative training corpus and a newly trained
checkpoint. See [Scientific scope](docs/SCIENTIFIC_SCOPE.md).

## Installation

Python 3.9 or newer is required.

```bash
python -m pip install -e .
python -m pip install -e ".[test]"       # tests
python -m pip install -e ".[api]"        # FastAPI service
python -m pip install -e ".[hull]"       # hull and surface workflows
python -m pip install -e ".[all]"        # MLIP + hull extras
```

Matra is optional and has a separate non-commercial research license. Install
it explicitly from an approved source; GenIM does not vendor Matra code or
weights:

```bash
python -m pip install -e ".[matra]"
python -m pip install -e ../matra-genoa-preview
```

## Chemistry policies

| Mode | Accepted domain | Default minimum species |
|---|---|---:|
| `any` | Every real chemical element | 1 |
| `metallic` | Metals and optional metalloids | 1 |
| `intermetallic` | Legacy metallic domain | 2 |

`allowed_elements` and `excluded_elements` narrow these domains exactly.
GenIM deliberately avoids an `inorganic` heuristic because that label cannot be
inferred unambiguously from an element set alone.

Example configuration:

```yaml
mp_download:
  chemistry_filter: any

preprocess:
  chemistry_mode: any
  seed_all_elements: true
  seed_all_hall: true

generate:
  chemistry_mode: any
  allowed_elements: [Na, Cl, K, Br]
  excluded_elements: null
  random_pool: chemistry
```

To reproduce historical behaviour:

```yaml
generate:
  chemistry_mode: intermetallic
  include_metalloids: true
  random_pool: intermetallic
```

## Offline smoke pipeline

The built-in examples include intermetallic, ionic, covalent, and semiconductor
structures.

```bash
genim examples-make --out data/examples.jsonl
genim preprocess --in data/examples.jsonl --out data/examples.tokens.pt \
  --seed-all-elements --seed-all-hall --chemistry any
genim train --data data/examples.tokens.pt --out checkpoints/example.pt \
  --steps 200 --element-emb features --element-feature-set periodic8
genim generate --ckpt checkpoints/example.pt --n 10 --out-dir output/cif \
  --chemistry any --nelements-min 1
genim validate --cif-dir output/cif
genim benchmark --cif-dir output/cif --out output/cif/benchmark.json
```

## Materials Project training data

Set `MP_API_KEY` (or `PMG_MAPI_KEY`), or place the key in the gitignored
`.mp_api_key` file.

```bash
genim mp-download --chemistry any --max-atoms 100 --eah-max 0.5 \
  --nelements-min 1 --nelements-max 5 --limit 50000 --out data/mp.jsonl
genim preprocess --in data/mp.jsonl --out data/mp.tokens.pt \
  --seed-all-elements --seed-all-hall --chemistry any
genim inspect --in data/mp.jsonl --wyckoff
genim inspect --in data/mp.tokens.pt
genim train --data data/mp.tokens.pt --out checkpoints/mp_general.pt \
  --steps 20000 --val-fraction 0.1 \
  --element-emb features --element-feature-set periodic8
```

Dataset balance matters. Report composition families, element frequency,
space-group coverage, cell-size distribution, train/validation/test split
policy, and any Materials Project filters together with model results. The
built-in seeded random validation split is a reproducible software baseline;
use composition- and prototype-held-out external splits for extrapolation claims.

## Default generation service

The service has working defaults and accepts arbitrary valid compositions. It
uses a configured checkpoint when available and otherwise returns an explicitly
labelled algorithmic starting geometry:

```bash
genim serve --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"formula":"LiFePO4","spacegroup_number":62,"n":4,"seed":7}'
```

Use `GENIM_MATRA_CHECKPOINT`, `GENIM_MATRA_SHA256`,
`GENIM_MODEL_CHECKPOINT`, and `GENIM_MODEL_SHA256` to configure checkpoint
backends. In a source checkout, the service can discover the verified sibling
`matra-v02-med.ckpt`. `backend="auto"` prefers an available checkpoint;
explicit unavailable checkpoint requests fail instead of silently falling back.
See [Public generation API](docs/PUBLIC_API.md).

## Python API

```python
from genim import ChemistryPolicy, GenIM, SamplingConfig

model = GenIM.from_checkpoint(
    "checkpoints/mp_general.pt",
    device="auto",
    # expected_sha256="..."  # recommended for published artifacts
)

results = model.sample(
    config=SamplingConfig(
        n=64,
        batch_size=16,
        max_sites=25,
        min_sites=2,
        temperature=0.9,
        seed=7,
    ),
    chemistry=ChemistryPolicy(
        mode="any",
        allowed_elements=["Na", "Cl", "K", "Br"],
        min_elements=2,
        max_elements=2,
    ),
)

paths = model.write_valid_cifs(results, "output/alkali_halides")
records = [result.to_record() for result in results]  # includes provenance
```

The current batch sampler performs one model forward pass per token position for
all active rows. KV-cache decoding is a future performance optimization; the
public result/provenance contract does not depend on it.

## Ensemble GenIM + Matra generation

The hybrid interface treats models as independent proposal sources. It does not
average or concatenate incompatible state dictionaries.

```bash
genim matra-checkpoint-info \
  --ckpt ../matra-genoa-preview/checkpoints/matra-v02-med.ckpt \
  --expected-sha256 4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa

genim generate-ensemble \
  --genim-ckpt checkpoints/mp_general.pt \
  --matra-ckpt ../matra-genoa-preview/checkpoints/matra-v02-med.ckpt \
  --matra-sha256 4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa \
  --elements Na Cl --stoichiometry 1 1 --stability stable \
  --n-per-backend 64 --seed 7 --out-dir output/hybrid-nacl
```

Equivalent Python API:

```python
from genim import (
    EnsembleGenerator, GenerationConstraints, GenerationSettings, GenIMBackend,
    MatraBackend, write_ensemble_run,
)

backends = [
    GenIMBackend.from_checkpoint("checkpoints/mp_general.pt"),
    MatraBackend.from_checkpoint(
        "../matra-genoa-preview/checkpoints/matra-v02-med.ckpt",
        expected_sha256="4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa",
    ),
]
run = EnsembleGenerator(backends).run(
    condition=GenerationConstraints(
        elements=("Na", "Cl"), stoichiometry=(1, 1), stability="stable",
    ),
    config=GenerationSettings(n=64, temperature=0.75, seed=7),
)
write_ensemble_run(run, "output/hybrid-nacl")
```

`selected` means structurally valid, constraint-not-disproved, and unique in the
run. It is not a claim of thermodynamic stability. Each requested constraint is
reported as `satisfied`, `violated`, or `not_evaluated` with its method and
evidence level. Unevaluated stability/hull targets require MLIP/DFT evidence. See
[Matra integration](docs/MATRA_INTEGRATION.md).

## Composition-constrained synthesis

`genim synth` reads `conf.yml` and supports exact ratio or atomic-percentage
constraints:

```bash
genim synth --elements Na Cl --nelements 2 --ratios 1 1
genim synth --elements Fe O --nelements 2 --ratios 2 3
genim synth --elements Li Fe P O --nelements 4 \
  --ratios 1 1 1 4 --ratio-mode ratio
```

`X` keeps an element mandatory while leaving its fraction unconstrained:

```bash
genim synth --elements Li Fe P O --nelements 4 \
  --ratios 1 X 1 X --ratio-mode ratio
```

## Checkpoints and reproducibility

Large datasets and weights remain outside Git. Publish them as immutable release
assets and record SHA256 values. Format-v2 checkpoints contain:

- `format_version`;
- strict `model_config`, vocabulary, tokenizer config, and weights;
- GenIM version, creation time, seed, training steps;
- source token-dataset SHA256 and source-dataset provenance;
- dataset and symmetry statistics.

Use `genim.checkpoints.download_checkpoint(...)` for atomic downloads with
optional checksum enforcement. Details are in
[Checkpoint format](docs/CHECKPOINT_FORMAT.md).

```bash
genim checkpoint-info --ckpt checkpoints/mp_train_fullsg_60.pt \
  --expected-sha256 3777449fe396522c0173aaa699c70c99b4d28e26b436200545f08f86ff28173c
```

The historical intermetallic checkpoint remains available from
[GitHub Releases](https://github.com/XYG-Research/GenIM/releases), but its
training domain must be reported when it is used. Its verified filenames, sizes,
URLs, and hashes are recorded in
[the v0.1.0 resource manifest](resources/release-v0.1.0.json).

## Optional scientific screening

```bash
genim score --conf conf.yml --cif-dir output/cif
genim surface-screen --input-dir output/cif --out-dir output/surfaces
```

`score` uses the configured MLIP and a consistent reference pool to estimate
energy above hull. These values inherit model/reference uncertainty and should
not be mixed with DFT values as though they shared one energy scale.

## Development

```bash
python -m pytest -q
```

Architecture and compatibility boundaries are documented in
[Architecture](docs/ARCHITECTURE.md). Multi-model scientific and licensing
boundaries are documented in [Matra integration](docs/MATRA_INTEGRATION.md).

## References

The representation and workflow are inspired by:

1. [doi:10.1038/s41524-025-01881-2](https://doi.org/10.1038/s41524-025-01881-2)
2. [doi:10.1038/s41524-025-01940-8](https://doi.org/10.1038/s41524-025-01940-8)

## License

[BSD 3-Clause](LICENSE)
