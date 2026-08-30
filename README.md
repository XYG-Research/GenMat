# GenMat: generative materials discovery

**English** | [简体中文](README.zh-CN.md)

GenMat is a Python package and command-line toolkit for building, generating,
validating, ranking, benchmarking, and screening periodic crystal structures.
Version 0.6 establishes the canonical `genmat` distribution and import
namespace, a versioned model catalog, and model-aware Python, CLI, HTTP, and
Studio integration. Version 0.5 added controlled, composition-preserving population mutation and
exact Hall-token space-group conditioning. Version 0.4 added evidence-aware
Matra/Alexandria observables, chemistry-aware
seed geometry, and deterministic scientific triage. Version 0.3 added auditable
multi-model proposal backends. Version 0.2 made the chemistry
domain explicit and general-purpose: oxides, nitrides,
halides, carbides, semiconductors, elemental solids, and intermetallics can use
the same Hall/Wyckoff generation pipeline.

The canonical distribution and import namespace are now `genmat`, the primary
class is `GenMat`, and the commands are `genmat` and `genmat-api`. The historical
`genim` namespace, `GenIM` class, `GenIMBackend`, and `genim`/`genim-api` commands
remain thin compatibility interfaces so published workflows and checkpoints stay
reproducible.

When upgrading an environment that installed the old `genim` distribution,
remove it before installing `genmat`; installing both distributions would make
them co-own the same compatibility package files:

```bash
python -m pip uninstall genim
python -m pip install --upgrade genmat
```

The representation combines space-group symmetry, Wyckoff sites, discretized
lattice/coordinate tokens, and a causal Transformer. Generated candidates are
decoded to ASE `Atoms`, checked geometrically and crystallographically, deduplicated,
and optionally relaxed/scored with an ML interatomic potential.

## What changed in 0.6

- `GenMat` and `GenMatBackend` now own the implementation; `GenIM` and
  `GenIMBackend` are compatibility subclasses rather than the primary classes.
- The published package is `genmat` and contains both `genmat` and legacy
  `genim` imports, with matching canonical and compatibility command entry points.
- `ModelRegistry` provides versioned model IDs, aliases, provenance, capability
  declarations, license gates, offline mode, content-addressed caching, and
  SHA256/size verification where an immutable checksum is published.
- `genmat models list|info|pull`, `GET /v1/models`, and the `model` generation
  request field make GenMat, Matra, Alexandria, and OMat24 assets discoverable
  without hard-coded paths. See [Model access](docs/MODELS.md).
- `GENMAT_*` is the canonical environment-variable family. Existing `GENIM_*`
  variables remain supported at lower precedence.

## What changed in 0.5

- `n` is the requested final population size. `mutation_fraction` divides it
  into independently generated direct parents and validated mutants, while
  always retaining at least one direct model result.
- Mutants preserve the complete composition and parent lineage. Exact
  space-group requests use symmetry-conservative cell mutation plus projected
  Wyckoff-orbit motion and compatible whole-orbit swaps; every local mutant is
  independently rechecked with spglib and rejected on mismatch.
- Geometry-dependent parent observables, including remote relaxation energy,
  are deliberately removed from mutants. Relax and rescore mutants before
  comparing their energies.
- `SamplingConfig.fixed_spacegroup` resolves the requested group to a Hall
  setting and forces that Hall token in the autoregressive sequence. Vocabulary
  coverage enables conditioning but does not prove that the checkpoint learned
  that structural family well.
- The historical GenIM checkpoint generator was not a simple random generator:
  it learned Hall/Wyckoff/lattice/site token sequences. The former Studio edge
  fallback was a separate algorithmic geometry seed and did not represent a
  learned model over all 230 space groups; it is now explicit-only.

## What changed in 0.4

- Matra `EHULL`/`EHULL_DISC` sequence blocks are preserved as auditable
  conditioning targets or model emissions, never silently promoted to computed
  thermodynamic evidence.
- `AlexandriaMatraBackend` integrates the public Matra generation/relaxation
  endpoint and preserves its reported energy as `relaxed_energy_per_atom` with
  source and evidence metadata. It is not relabelled as DFT, formation energy,
  or energy above hull because the public response does not expose that basis.
- `ScientificObservable` distinguishes conditioning targets, model emissions,
  postprocessed estimates, and independently calculated values. Optional
  `ScientificEvaluator` stages can add MLIP, hull, DFT, phonon, or other evidence.
- The non-ML fallback now constructs cells and sites from covalent radii,
  packing bounds, periodic minimum-image distances, and farthest-point sampling.
- Every selected candidate receives a transparent schema-v3 triage rank based
  on software validity, geometry, constraint evidence, backend consistency, and
  observable evidence. Rank is not a stability or synthesizability claim.

## What changed in 0.3

- `GenerationBackend`, `GenerationConstraints`, `GeneratedCandidate`, and
  `EnsembleGenerator` define a model-neutral generation and provenance contract.
  The original 0.3 names remain compatibility aliases.
- `GenMatBackend` and the optional `MatraBackend` run through the same ASE
  conversion, validation, condition audit, and cross-model deduplication path.
- `AlgorithmicSeedBackend` and `GeneratorService` provide a reproducible,
  composition-exact default when no compatible checkpoint is configured,
  without presenting the result as a learned-model prediction.
- The optional HTTP API exposes `/v1/health`, `/v1/capabilities`, and
  `/v1/generate` with the same candidate records used by Python.
- Matra checkpoints are loaded with PyTorch weights-only mode, SHA256
  verification, schema checks, and exact reconstructed-model weight matching.
- `genmat generate-ensemble` produces selected CIFs, a complete `candidates.jsonl`
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
- `GenMat.from_checkpoint(...)` provides a stable Python API with batched
  grammar-constrained sampling.
- New checkpoints have a versioned schema, SHA256 verification, training/data
  provenance, and strict weight/config compatibility checks. Legacy v1
  checkpoints remain loadable.
- New models default to `periodic8` element descriptors: atomic number,
  covalent radius, mass, period, group, and broad chemical-class indicators.
  Legacy feature checkpoints retain their original four-dimensional projection.
- `ValidationReport` exposes decision metrics, while `genmat benchmark` reports
  validity, uniqueness, formula/element coverage, and space-group coverage.
- CI and a tracked regression suite cover chemistry policies, legacy checkpoint
  loading, batch sampling, generation provenance, and the existing workflows.

## Scientific scope

GenMat proposes symmetry-consistent candidates; it does **not** prove that a
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
it explicitly from an approved source; GenMat does not vendor Matra code or
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
GenMat deliberately avoids an `inorganic` heuristic because that label cannot be
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
genmat examples-make --out data/examples.jsonl
genmat preprocess --in data/examples.jsonl --out data/examples.tokens.pt \
  --seed-all-elements --seed-all-hall --chemistry any
genmat train --data data/examples.tokens.pt --out checkpoints/example.pt \
  --steps 200 --element-emb features --element-feature-set periodic8
genmat generate --ckpt checkpoints/example.pt --n 10 --out-dir output/cif \
  --chemistry any --nelements-min 1
genmat validate --cif-dir output/cif
genmat benchmark --cif-dir output/cif --out output/cif/benchmark.json
```

## Materials Project training data

Set `MP_API_KEY` (or `PMG_MAPI_KEY`), or place the key in the gitignored
`.mp_api_key` file.

```bash
genmat mp-download --chemistry any --max-atoms 100 --eah-max 0.5 \
  --nelements-min 1 --nelements-max 5 --limit 50000 --out data/mp.jsonl
genmat preprocess --in data/mp.jsonl --out data/mp.tokens.pt \
  --seed-all-elements --seed-all-hall --chemistry any
genmat inspect --in data/mp.jsonl --wyckoff
genmat inspect --in data/mp.tokens.pt
genmat train --data data/mp.tokens.pt --out checkpoints/mp_general.pt \
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
uses a configured local checkpoint first, can use the public Matra/Alexandria
generation-and-relaxation service when explicitly enabled, and otherwise
returns an explicitly labelled algorithmic starting geometry:

```bash
genmat serve --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"formula":"LiFePO4","spacegroup_number":62,"n":4,"seed":7}'
```

Use `GENMAT_MATRA_CHECKPOINT`, `GENMAT_MATRA_SHA256`,
`GENMAT_MODEL_CHECKPOINT`, and `GENMAT_MODEL_SHA256` to configure checkpoint
backends. Set `GENMAT_ENABLE_ALEXANDRIA=1` to opt into the public remote backend.
The same `GENIM_*` names remain lower-priority compatibility aliases.
In a source checkout, the service can discover the verified sibling
`matra-v02-med.ckpt`. `backend="auto"` prefers local checkpoints, then the
enabled remote service, then the geometry seed; explicit unavailable checkpoint
requests fail instead of silently falling back.
See [Public generation API](docs/PUBLIC_API.md).

## Model access

```bash
genmat models list
genmat models info matra/genoa-mpas-med@0.2
genmat models pull matra/genoa-mpas-med@0.2 --accept-license
genmat generate-ensemble --model matra/genoa-mpas-med@0.2 \
  --accept-model-license --elements Na Cl --stoichiometry 1 1 \
  --n-per-backend 8 --out-dir output/matra-nacl
```

Python callers use the same catalog and cache:

```python
from genmat import ModelRegistry

models = ModelRegistry.default()
spec = models.info("matra-v02-med")
backend = models.load_backend(spec, accept_license=True, device="auto")
```

Catalog metadata does not claim that a model is scientifically suitable for an
arbitrary chemistry. Inspect its training-domain evidence and model card before
interpreting results. See [Model access and provenance](docs/MODELS.md).

## Python API

```python
from genmat import ChemistryPolicy, GenMat, SamplingConfig

model = GenMat.from_checkpoint(
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

## Ensemble GenMat + Matra generation

The hybrid interface treats models as independent proposal sources. It does not
average or concatenate incompatible state dictionaries.

```bash
genmat matra-checkpoint-info \
  --ckpt ../matra-genoa-preview/checkpoints/matra-v02-med.ckpt \
  --expected-sha256 4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa

genmat generate-ensemble \
  --genmat-ckpt checkpoints/mp_general.pt \
  --matra-ckpt ../matra-genoa-preview/checkpoints/matra-v02-med.ckpt \
  --matra-sha256 4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa \
  --elements Na Cl --stoichiometry 1 1 --stability stable \
  --n-per-backend 64 --seed 7 --out-dir output/hybrid-nacl
```

Equivalent Python API:

```python
from genmat import (
    EnsembleGenerator, GenerationConstraints, GenerationSettings, GenMatBackend,
    MatraBackend, write_ensemble_run,
)

backends = [
    GenMatBackend.from_checkpoint("checkpoints/mp_general.pt"),
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

`genmat synth` reads `conf.yml` and supports exact ratio or atomic-percentage
constraints:

```bash
genmat synth --elements Na Cl --nelements 2 --ratios 1 1
genmat synth --elements Fe O --nelements 2 --ratios 2 3
genmat synth --elements Li Fe P O --nelements 4 \
  --ratios 1 1 1 4 --ratio-mode ratio
```

`X` keeps an element mandatory while leaving its fraction unconstrained:

```bash
genmat synth --elements Li Fe P O --nelements 4 \
  --ratios 1 X 1 X --ratio-mode ratio
```

## Checkpoints and reproducibility

Large datasets and weights remain outside Git. Publish them as immutable release
assets and record SHA256 values. Format-v2 checkpoints contain:

- `format_version`;
- strict `model_config`, vocabulary, tokenizer config, and weights;
- GenMat version (legacy files may use the `genim_version` key), creation time,
  seed, and training steps;
- source token-dataset SHA256 and source-dataset provenance;
- dataset and symmetry statistics.

Use `genmat.checkpoints.download_checkpoint(...)` for atomic downloads with
optional checksum enforcement. Details are in
[Checkpoint format](docs/CHECKPOINT_FORMAT.md).

```bash
genmat checkpoint-info --ckpt checkpoints/mp_train_fullsg_60.pt \
  --expected-sha256 3777449fe396522c0173aaa699c70c99b4d28e26b436200545f08f86ff28173c
```

The historical intermetallic checkpoint remains available from
[GitHub Releases](https://github.com/XYG-Research/GenIM/releases), but its
training domain must be reported when it is used. Its verified filenames, sizes,
URLs, and hashes are recorded in
[the v0.1.0 resource manifest](resources/release-v0.1.0.json).

## Optional scientific screening

```bash
genmat score --conf conf.yml --cif-dir output/cif
genmat surface-screen --input-dir output/cif --out-dir output/surfaces
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
