# GenMat model access and provenance

GenMat 0.6 separates the software package from the models it can operate. A
model catalog record makes an asset discoverable and loadable; it does **not**
claim that the model is accurate for every composition, space group, property,
or scientific question.

## Built-in catalog

| Stable model ID | Role | Backend | Integrity and scope note |
|---|---|---|---|
| `genmat/mp-fullsg-legacy@0.1.0` | structure generation | GenMat token model | SHA256 pinned; historical training domain, not a universal-chemistry model |
| `matra/genoa-mpas@0.2` | structure generation | Matra Genoa | SHA256 and size pinned; separate upstream non-commercial research terms |
| `matra/genoa-mpas-med@0.2` | structure generation | Matra Genoa | SHA256 and size pinned; separate upstream non-commercial research terms |
| `matra/genoa-mp@0.2` | structure generation | Matra Genoa | SHA256 and size pinned; separate upstream non-commercial research terms |
| `alexandria/gen-crystal@remote` | generation and remote relaxation | Alexandria/Matra service | remote service; upstream model/calculator can change and is not checkpoint-pinned |
| `fairchem/esen-30m-oam@omat24` | relaxation and energy | OMat24/eSEN | Hugging Face commit, LFS SHA256, and byte size pinned; gated upstream terms |

The machine-readable source of truth is packaged as
`genmat.data/model-catalog-v1.json`. The `genim` import namespace and historical
model aliases remain available for compatibility during the 0.x migration
window.

## Command line

```bash
genmat models list
genmat models list --provider matra --json
genmat models info matra-v02-med
genmat models pull matra/genoa-mpas-med@0.2 --accept-license
```

`pull` writes to a deterministic cache location, verifies the mandatory size
and SHA256 value, and only then atomically publishes the file. Hugging Face
entries must also pin an immutable 40-character commit. `--offline`
forbids network access. Models with separate upstream terms require the explicit
`--accept-license` acknowledgement; GenMat does not grant or broaden those terms.

Named models can be used directly in an ensemble:

```bash
genmat generate-ensemble \
  --model genmat/mp-fullsg-legacy@0.1.0 \
  --model matra/genoa-mpas-med@0.2 \
  --accept-model-license \
  --elements Na Cl --stoichiometry 1 1 \
  --n-per-backend 16 --out-dir output/nacl-ensemble
```

The canonical checkpoint flags are `--genmat-ckpt` and `--genmat-sha256`;
`--genim-ckpt` and `--genim-sha256` remain aliases for old scripts.

## Python

```python
from genmat import ModelRegistry

registry = ModelRegistry.default()
for model in registry.list(task="generation"):
    print(model.id, model.backend, model.capabilities)

spec = registry.info("matra-v02-med")
path = registry.resolve(spec, accept_license=True)
backend = registry.load_backend(spec, accept_license=True, device="auto")
```

The principal calls are:

- `list(provider=..., task=..., backend=...)`
- `info(model_id_or_alias)`
- `resolve(..., download=True, offline=False, accept_license=False)`
- `pull(...)`
- `load_backend(...)`

All local model references are validated against the packaged catalog. Arbitrary
URLs supplied by an HTTP client are not accepted as model identifiers.

## HTTP and Studio

- `GET /v1/models` returns catalog records, server-side license readiness, and
  missing optional runtime dependencies without loading model weights.
- `GET /v1/models/{model-id}` returns one immutable catalog record.
- `GET /v1/capabilities` includes configured backends and catalog models.
- `POST /v1/generate` accepts a stable model ID or alias in the `model` field.

Example:

```json
{
  "formula": "LiFePO4",
  "model": "genmat/mp-fullsg-legacy@0.1.0",
  "backend": "auto",
  "spacegroup_number": 62,
  "n": 8,
  "mutation_fraction": 0.25,
  "seed": 7
}
```

GenMat Studio consumes `/v1/capabilities`, displays only available generation
models, and sends the same `model` field. Relaxation-only assets such as OMat24
are therefore not presented as structure generators.

## Configuration and compatibility

Canonical variables take precedence over their legacy equivalents:

| Canonical | Compatibility alias | Purpose |
|---|---|---|
| `GENMAT_MODEL_CACHE` | `GENIM_MODEL_CACHE` | model cache root |
| `GENMAT_CHECKPOINT_DIR` | `GENIM_CHECKPOINT_DIR` | alternate cache root |
| `GENMAT_MODEL_PATH` | `GENIM_MODEL_PATH` | additional local asset search roots |
| `GENMAT_OFFLINE` | `GENIM_OFFLINE` | prohibit network model resolution |
| `GENMAT_ACCEPT_MODEL_LICENSES` | `GENIM_ACCEPT_MODEL_LICENSES` | administrator acceptance by model ID/provider |
| `GENMAT_MODEL` | `GENIM_MODEL` | default service model |
| `GENMAT_MODEL_CHECKPOINT` | `GENIM_MODEL_CHECKPOINT` | direct local GenMat checkpoint |

The default cache is `%LOCALAPPDATA%/GenMat/models` on Windows when available,
`$XDG_CACHE_HOME/genmat/models` on configured Unix systems, and otherwise
`~/.cache/genmat/models`.

## Adding later models

A future model should be added without changing user code:

1. assign an immutable ID of the form `provider/name@version`;
2. record provider, backend adapter, task, format, filename, source, license,
   optional dependencies, and capability modes;
3. publish byte size and SHA256 for every downloadable asset, and pin a full
   commit for Hugging Face sources;
4. add a backend adapter only when the model grammar or runtime differs;
5. add unit tests for alias resolution, offline behavior, integrity failure,
   license gating, and backend dispatch;
6. publish a model card describing training corpus, exclusions, deduplication,
   chemistry and symmetry coverage, held-out tests, and known failure modes.

For scientific releases, also pin the software commit, catalog version, model
asset hash, inference settings, random seed, and post-generation validation or
relaxation protocol. A permissive `ChemistryPolicy(mode="any")` expands what the
software accepts; it does not expand what a checkpoint learned.
