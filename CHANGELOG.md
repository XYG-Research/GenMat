# Changelog

All notable user-visible changes are recorded here. The project follows
semantic versioning for public APIs and serialized schemas.

## 0.4.0

- Added schema-version-3 scientific observables that distinguish conditioning
  targets, generative-model emissions, postprocessed estimates, and calculated
  evidence.
- Matra sequence property blocks such as `EHULL` and `EHULL_DISC` are now
  preserved with model/checkpoint provenance without being promoted to
  independent stability calculations.
- Added an optional rate-limited `AlexandriaMatraBackend` for the public Matra
  generation-and-relaxation pipeline. Its upstream `energy` value is reported
  conservatively because the public response does not name the calculator or
  reference zero.
- Added the `ScientificEvaluator` plugin contract, independent hull-evidence
  reconciliation, transparent multi-component candidate ranking, and rank
  records in ensemble reports.
- Replaced the grid-jitter fallback with covalent-radius, packing-aware
  farthest-point geometry construction and expanded validation metrics for
  coordination, connectivity, and covalent packing.
- Corrected the service's omitted-seed behavior so the documented default seed
  is actually `7`.

- Added canonical `GenerationBackend`, `GenerationConstraints`,
  `GenerationSettings`, and `GeneratedCandidate` names with GenIM 0.3 aliases.
- Added machine-readable backend constraint application modes.
- Added per-candidate three-state constraint assessments with method and
  evidence level.
- Added `selected` as the scientifically clearer pipeline state while retaining
  `accepted` as a compatibility alias.
- Added schema-version-3 ensemble reports with backend capability records.
- Added canonical `genim generate-ensemble` command; `hybrid-generate` remains
  an alias.
- Added the always-available `AlgorithmicSeedBackend`, `GeneratorService`, and
  optional FastAPI endpoints with reproducible schema-v3 defaults.
- Split ensemble and service CLI handlers out of the legacy CLI orchestration
  module and added `genim serve` / `genim-api` entry points.
- Restricted pytest discovery to the public test suite.

## 0.3.0

- Added auditable GenIM and optional Matra generation backends.
- Added safe Matra checkpoint inspection and cross-backend deduplication.
- Added complete candidate JSONL provenance and ensemble reports.

## 0.2.0

- Generalized chemistry policy beyond intermetallic systems.
- Added the reusable checkpoint-backed Python API and versioned checkpoints.
