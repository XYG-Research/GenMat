# Contributing to GenMat

GenMat welcomes contributions across crystal representations, generation
backends, validation, benchmarks, documentation, and visualization.

## Development setup

```powershell
python -m pip install -e ".[hull,test]"
python -m pytest -q
python -m compileall -q src/genmat src/genim
```

Run commands from the repository root. Pytest is intentionally restricted to
the public `tests/` directory so local datasets and adjacent research projects
cannot alter release acceptance.

## Scientific contribution requirements

- Separate model conditioning from independent verification.
- Represent unevaluated constraints as `not_evaluated`, never as a pass.
- Declare each backend constraint mode with `BackendCapabilities`.
- Record checkpoint SHA256, sampling settings, seed, validation parameters,
  software versions, and data/reference-set provenance.
- State the achieved evidence level from `docs/SCIENTIFIC_SCOPE.md`.
- Include negative and boundary tests, not only successful examples.
- Do not call a candidate stable, novel, or synthesizable without the evidence
  required for that claim.

## Compatibility and naming

Use the canonical public names documented in `docs/PUBLIC_API.md`.
Compatibility aliases from GenIM 0.3 may be maintained, but new public fields
must have unambiguous scientific semantics and a versioned serialized schema.

## External models and data

Do not commit model weights, private data, credentials, or license-restricted
third-party source. Matra code and checkpoints remain under their own
non-commercial research license and are not covered by GenMat's BSD license.
New checkpoint manifests must include immutable URLs, byte sizes, SHA256, model
configuration, intended use, training-domain summary, and known limitations.

## Pull requests

Keep changes focused, explain the scientific or software rationale, list the
commands used for validation, and identify any behavior or schema changes.
