# GenIM acceptance protocol

This document defines release acceptance for the general-chemistry package. It
does not preserve historical benchmark numbers because those numbers describe a
specific intermetallic checkpoint rather than the software's universal scope.

## Required software checks

```powershell
python -m pip install -e ".[hull,test]"
python -m pytest -q
python -m compileall -q src/genim
genim --help
genim hybrid-generate --help
```

The suite must pass on every supported Python version in CI. No secrets, model
weights, generated output, or Materials Project data may be committed.

## Offline pipeline

```powershell
genim examples-make --out data/examples.jsonl
genim preprocess --in data/examples.jsonl --out data/examples.tokens.pt `
  --seed-all-elements --seed-all-hall --chemistry any
genim train --data data/examples.tokens.pt --out checkpoints/example.pt `
  --steps 5 --element-emb features --element-feature-set periodic8
genim generate --ckpt checkpoints/example.pt --n 2 --out-dir output/smoke `
  --chemistry any --nelements-min 1
genim benchmark --cif-dir output/smoke --out output/smoke/benchmark.json
```

This is a software smoke test, not a model-quality benchmark.

## Multi-backend acceptance

Unit tests must cover condition translation, optional Matra imports, safe
checkpoint inspection, pymatgen-to-ASE conversion, backend consistency flags,
cross-backend deduplication, source-aware summaries, and non-overwriting output.

When the local research-licensed Matra package and checkpoint are available:

```powershell
python -m pip install -e ..\matra-genoa-preview
genim matra-checkpoint-info `
  --ckpt ..\matra-genoa-preview\checkpoints\matra-v02-med.ckpt `
  --expected-sha256 4e511528c4665be006e451f0e473381b3d02c286981608c7db0b743dcea0e4fa
```

The real-checkpoint smoke test must reconstruct the model without missing or
unexpected weights and generate at least one candidate through the shared
validation path. Generated smoke output remains gitignored.

## Checkpoint acceptance

For each published checkpoint, archive:

- asset name, byte size, SHA256, and format version;
- GenIM commit and package version;
- full model/tokenizer configuration;
- training seed and steps;
- input token-dataset SHA256 and raw-data provenance;
- chemistry domain, data filters, split/dedup policy;
- benchmark JSON on held-out composition and prototype splits.

Load the checkpoint with `expected_sha256` before benchmarking. Legacy format-v1
checkpoints must be labeled with their original training domain.

## Scientific acceptance

At minimum report decoded validity, uniqueness, formulas, elements, space-group
coverage, and every rejection reason. MLIP or hull claims must name the potential,
potential checkpoint, relaxation settings, reference construction, and energy
units. DFT/phonon evidence is required before describing a generated candidate as
first-principles stable.

For multi-model comparisons, hold the requested conditions and random seeds
fixed, and report both equal-sample and equal-wall-time results. Report each
backend separately before the union. Composition/prototype-held-out test
structures must be deduplicated against every backend's training corpus where
that corpus is available.
