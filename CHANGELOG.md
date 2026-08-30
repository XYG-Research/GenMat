# Changelog

All notable user-visible changes are recorded here. The project follows
semantic versioning for public APIs and serialized schemas.

## Unreleased

## 0.6.0

- Made `genmat` the canonical distribution/import namespace and made `GenMat`
  and `GenMatBackend` the implementation-owning classes. The `genim` namespace,
  `GenIM`/`GenIMBackend`, legacy commands, checkpoint keys, and backend wire ID
  remain compatibility-stable.
- Added a packaged, versioned `ModelRegistry` covering the historical GenMat
  checkpoint, three Matra Genoa checkpoints, Alexandria remote generation, and
  OMat24 eSEN relaxation. The registry supports aliases, offline mode, explicit
  license acknowledgement, deterministic caching, and published size/SHA256
  verification.
- Require size/SHA256 metadata for every downloadable catalog asset and an
  immutable commit for Hugging Face sources; the OMat24 eSEN entry pins both its
  upstream commit and LFS digest.
- Added `genmat models list|info|pull`, catalog-backed `--model` ensemble input,
  `/v1/models`, the HTTP generation `model` field, and `GENMAT_*` configuration
  with lower-priority `GENIM_*` aliases.
- Updated Studio to discover runnable generation models from capabilities and to
  preserve non-renderable schema-v3 candidate records as audit evidence instead
  of discarding an otherwise valid mixed response.
- Preserved pre-0.6 positional API construction, propagated environment-driven
  offline mode, and prevented non-generation or explicitly named models from
  being loaded or silently replaced by another backend.

## 0.5.0

- Added fixed-size population generation with configurable direct-parent and
  composition-preserving mutant fractions.
- Mutants use crystal-system-compatible cell strain, site-stabilizer-projected
  Wyckoff-orbit motion, and compatible whole-orbit occupancy swaps. Every mutant
  is validated, constraint-audited, deduplicated, and linked to its parent.
- Parent energies and other geometry-dependent observables are invalidated on
  mutation instead of being copied to a new geometry.
- `GenIM.sample()` and `GenIMBackend` now condition exact space-group requests
  by forcing the corresponding Hall token, then independently verify decoded
  symmetry with spglib.
- Alexandria population requests now make independent upstream generation and
  relaxation calls for direct parents; `n` means final returned population,
  not merely an upstream internal pool size.
- GenIM Studio defaults to the real Matra/Alexandria generation-and-relaxation
  path and no longer disguises a remote failure as an algorithmic seed result.
  Geometry seeds remain available only as an explicit backend choice.

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
