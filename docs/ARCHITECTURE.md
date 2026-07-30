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
 Transformer training -> versioned checkpoint
              |
              v
 batched constrained sampling -> token decode -> ASE Atoms
              |
              v
 geometry/symmetry validation -> chemistry policy -> deduplication
              |
              +--> benchmark JSON
              +--> MLIP relaxation / hull scoring
              +--> surface screening
```

## Stable boundaries

- `chem.py`: composition-domain policy independent of data source and model.
- `preprocess.py`: structure-to-token conversion and token-dataset provenance.
- `model.py`: model architecture, including legacy/new element feature contracts.
- `checkpoints.py`: safe loading, schema compatibility, checksums, and downloads.
- `generate.py`: grammar-constrained scalar and batch token sampling.
- `api.py`: reusable model/result/provenance contract.
- `validate.py`: fast structural decisions plus auditable metrics.
- `benchmark.py`: deterministic aggregate evaluation.
- `score.py`, `hull.py`, `surface_screen.py`: optional higher-cost scientific screens.

## Compatibility rule

Chemistry policy and model training domain are separate concepts. The policy
controls what is allowed at input/output boundaries; the checkpoint determines
what distribution the model has learned. Code must not infer model competence
from a permissive policy.
