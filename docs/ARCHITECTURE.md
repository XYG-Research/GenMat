# GenMat architecture

```text
Materials Project / JSONL / examples
              |
              v
  ChemistryPolicy + preprocessing
              |
              v
 Hall/Wyckoff token dataset + provenance
              |
              v
 Transformer training -> versioned GenMat checkpoint
              |
              +----------------------+----------------------+----------------+
              |                      |                      |                |
              v                      v                      v                v
   AlgorithmicSeedBackend      GenMatBackend         MatraBackend   AlexandriaMatraBackend
              |                      |                      |                |
              +---------------- GenerationBackend -------------------------+
                         |
                         v
            GeneratedCandidate + provenance
                         |
                         v
       direct-parent planning + composition-preserving mutation
                         |
                         v
            ASE conversion + shared validation
                         |
                         v
 observables/evaluators + constraint evidence + cross-backend deduplication
                         |
                         v
          transparent scientific triage ranking
                         |
              +----------+-----------+
              |          |           |
              v          v           v
       HTTP/schema v3  benchmark JSON  MLIP/hull  surface screening
```

## Stable boundaries

- `chem.py`: composition-domain policy independent of data source and model.
- `preprocess.py`: structure-to-token conversion and token-dataset provenance.
- `model.py`: model architecture, including legacy/new element feature contracts.
- `checkpoints.py`: safe loading, schema compatibility, checksums, and downloads.
- `models.py` + packaged catalog: stable model IDs, provenance, license gates,
  offline/cache policy, asset integrity, and backend dispatch.
- `generate.py`: grammar-constrained scalar and batch token sampling.
- `api.py`: reusable model/result/provenance contract.
- `backends/base.py`: backend-neutral constraints, generation settings,
  capability declarations, candidate records, and three-state evidence.
- `backends/genim.py`: canonical `GenMatBackend` plus the `GenIMBackend`
  compatibility subclass; the historical wire ID remains `genim`.
- `backends/matra.py`: optional, safely loaded Matra inference adapter.
- `backends/alexandria.py`: optional public Matra/Alexandria generation and
  relaxation adapter with conservative energy semantics.
- `backends/seed.py`: always-available, composition-exact starting-geometry
  construction without a learned-model claim.
- `backends/mutation.py`: deterministic population planning, parent-linked
  composition-preserving mutation, observable invalidation, and independent
  validation/constraint re-audit.
- `backends/ranking.py`: deterministic, component-wise scientific triage rank;
  this is a scheduling aid, not a stability classifier.
- `backends/ensemble.py`: shared cross-backend validation, deduplication,
  optional scientific evaluators, source-aware summaries, CIF and JSONL audit output.
- `service.py`: framework-neutral request parsing, backend discovery/selection,
  default settings, and schema-version-3 response assembly.
- `server.py`: optional FastAPI transport (`/v1/health`,
  `/v1/capabilities`, `/v1/models`, `/v1/generate`).
- `commands/`: composable CLI registrations and handlers extracted from the
  legacy orchestration hub.
- `validate.py`: fast structural decisions plus auditable metrics.
- `benchmark.py`: deterministic aggregate evaluation.
- `score.py`, `hull.py`, `surface_screen.py`: optional higher-cost scientific screens.

## Compatibility rule

Chemistry policy and model training domain are separate concepts. The policy
controls what is allowed at input/output boundaries; the checkpoint determines
what distribution the model has learned. Code must not infer model competence
from a permissive policy.

External checkpoint schemas are never coerced into the GenMat checkpoint schema.
Each backend owns its safe loader and exposes a common proposal record only
after decoding. Transformer state dictionaries are not merged across token
grammars.

`selected` is a software-pipeline state: structurally valid,
constraint-not-disproved, and unique in the current run. It is not a stability
claim. `accepted` remains a GenIM 0.3 compatibility alias. Every requested
constraint has one of `satisfied`, `violated`, or `not_evaluated`; the last must
never be reported as satisfied.

Backends declare *how* they apply a constraint (`construction`, `conditioning`,
`sampling_filter`, `post_filter`, or `unsupported`). This capability statement
is deliberately separate from per-candidate evidence. Matra conditioning and a
GenMat token mask are generation mechanisms; decoded-structure checks, MLIP,
convex-hull, DFT, and phonons provide progressively stronger evidence.

Population mutation happens before cross-backend deduplication. It never copies
energies or other geometry-dependent observables from a parent. When an exact
space group is requested, local mutation is accepted only if the post-mutation
spglib check still matches; otherwise the requested population may report a
shortfall rather than silently weaken the constraint.

`ScientificObservable` separates four roles: `conditioning_target`,
`model_emission`, `postprocessed_estimate`, and `calculated`. Only an
independently validated calculated `energy_above_hull` can satisfy or violate a
stability/hull constraint. Remote energies with undisclosed calculators or
reference zeros remain postprocessed estimates even when numerically precise.

The HTTP service defaults to `backend="auto"`. It prefers a configured Matra
or GenMat checkpoint, then an explicitly enabled Alexandria remote backend, and
otherwise uses `AlgorithmicSeedBackend`. An explicit
request for an unavailable checkpoint backend fails rather than being silently
relabelled as a model prediction. The service layer is the canonical integration
boundary for the visual application. A catalog `model` reference selects a
pinned adapter without accepting arbitrary URLs; the older `GenIM` objects
remain available for compatibility.
