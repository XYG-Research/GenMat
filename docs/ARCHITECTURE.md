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
              +----------------------+
              |                      |
              v                      v
        GenIMBackend           MatraBackend (optional)
              |                      |
              +---- ProposalBackend -+
                         |
                         v
            ProposedStructure + provenance
                         |
                         v
            ASE conversion + shared validation
                         |
                         v
       condition audit + cross-backend deduplication
                         |
              +----------+-----------+
              |          |           |
              v          v           v
       benchmark JSON  MLIP/hull  surface screening
```

## Stable boundaries

- `chem.py`: composition-domain policy independent of data source and model.
- `preprocess.py`: structure-to-token conversion and token-dataset provenance.
- `model.py`: model architecture, including legacy/new element feature contracts.
- `checkpoints.py`: safe loading, schema compatibility, checksums, and downloads.
- `generate.py`: grammar-constrained scalar and batch token sampling.
- `api.py`: reusable model/result/provenance contract.
- `backends/base.py`: backend-neutral conditions, proposal records, capability
  and condition-compliance contracts.
- `backends/genim.py`: adapter from the existing public GenIM API.
- `backends/matra.py`: optional, safely loaded Matra inference adapter.
- `backends/ensemble.py`: shared cross-backend validation, deduplication,
  source-aware summaries, CIF and JSONL audit output.
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

`accepted` is a software-pipeline state: structurally valid,
condition-not-disproved, and unique in the current run. A `null` condition
check means the condition has not been scientifically evaluated and must not be
reported as satisfied.
