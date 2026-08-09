# Architecture

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
 Transformer training -> versioned GenIM checkpoint
              |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
   AlgorithmicSeedBackend      GenIMBackend          MatraBackend (optional)
              |                      |                      |
              +----------- GenerationBackend -------------+
                         |
                         v
            GeneratedCandidate + provenance
                         |
                         v
            ASE conversion + shared validation
                         |
                         v
  constraint evidence + cross-backend deduplication
                         |
              +----------+-----------+
              |          |           |
              v          v           v
       HTTP/schema v2  benchmark JSON  MLIP/hull  surface screening
```

## Stable boundaries

- `chem.py`: composition-domain policy independent of data source and model.
- `preprocess.py`: structure-to-token conversion and token-dataset provenance.
- `model.py`: model architecture, including legacy/new element feature contracts.
- `checkpoints.py`: safe loading, schema compatibility, checksums, and downloads.
- `generate.py`: grammar-constrained scalar and batch token sampling.
- `api.py`: reusable model/result/provenance contract.
- `backends/base.py`: backend-neutral constraints, generation settings,
  capability declarations, candidate records, and three-state evidence.
- `backends/genim.py`: adapter from the existing public GenIM API.
- `backends/matra.py`: optional, safely loaded Matra inference adapter.
- `backends/seed.py`: always-available, composition-exact starting-geometry
  construction without a learned-model claim.
- `backends/ensemble.py`: shared cross-backend validation, deduplication,
  source-aware summaries, CIF and JSONL audit output.
- `service.py`: framework-neutral request parsing, backend discovery/selection,
  default settings, and schema-version-2 response assembly.
- `server.py`: optional FastAPI transport (`/v1/health`,
  `/v1/capabilities`, `/v1/generate`).
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

External checkpoint schemas are never coerced into the GenIM checkpoint schema.
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
GenIM token mask are generation mechanisms; decoded-structure checks, MLIP,
convex-hull, DFT, and phonons provide progressively stronger evidence.

The HTTP service defaults to `backend="auto"`. It prefers a configured Matra
or GenIM checkpoint and otherwise uses `AlgorithmicSeedBackend`. An explicit
request for an unavailable checkpoint backend fails rather than being silently
relabelled as a model prediction. The service layer is the canonical integration
boundary for the visual application; the older checkpoint-specific `api.py`
objects remain available for GenIM 0.3 compatibility.
