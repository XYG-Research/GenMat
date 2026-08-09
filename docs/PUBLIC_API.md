# Public generation API

GenIM's canonical multi-backend vocabulary is intentionally model-neutral:

| Concept | Public type | Meaning |
|---|---|---|
| requested scientific intent | `GenerationConstraints` | composition, symmetry, or energy-related targets |
| run controls | `GenerationSettings` | sample count, seed, temperature, decoding and validation options |
| backend contract | `GenerationBackend` | a checkpoint-backed or explicitly algorithmic candidate generator |
| backend declaration | `BackendCapabilities` | how each requested constraint is applied |
| one result | `GeneratedCandidate` | decoded structure, provenance, validation, evidence and selection state |
| multi-model run | `EnsembleGenerator` | validation and deduplication across independent backends |

The older GenIM 0.3 names `GenerationCondition`, `ProposalConfig`,
`ProposalBackend`, and `ProposedStructure` are aliases. They remain importable,
but new integrations should use the names above.

## Capability is not evidence

`BackendCapabilities` records one application mode per constraint:

- `conditioning`: the requested value is included in a model prompt;
- `construction`: the requested value is built directly into an algorithmic structure;
- `sampling_filter`: invalid tokens or choices are removed during sampling;
- `post_filter`: results are filtered after generation;
- `unsupported`: the backend does not apply the request.

This declaration does not say that the output satisfies the request. Every
decoded candidate separately records a `ConstraintAssessment` with status
`satisfied`, `violated`, or `not_evaluated`, plus the evaluation method and the
scientific evidence level. In particular, a Matra stability prompt remains
`not_evaluated` until an energy model and compatible convex-hull reference set
have actually been run.

## Selection semantics

`candidate.selected` means only that the structure passed the configured
software validation, no evaluated constraint was violated, and it is unique in
the current ensemble run. `candidate.accepted` is a compatibility alias. Neither
property means synthesizable or thermodynamically, mechanically, or dynamically
stable.

`GeneratedCandidate.to_record()` emits both the canonical fields and legacy
compatibility fields. New consumers should read `selected`, `selection_reason`,
`constraint_status`, `constraint_assessments`, and `backend_capabilities` from
the schema-version-2 report.

## Service facade and HTTP API

`GeneratorService` is the canonical application boundary. It accepts a plain
mapping, normalizes formula or element/stoichiometry input, selects configured
backends, and returns schema version 2. Request defaults are:

```json
{
  "formula": "SiO2",
  "backend": "auto",
  "n": 4,
  "temperature": 0.8,
  "seed": 7
}
```

`auto` prefers an available Matra or GenIM checkpoint and otherwise selects the
non-ML `AlgorithmicSeedBackend`. Explicit `matra` or `genim` requests fail when
that backend is unavailable. Install `genim[api]` and run `genim serve` (or
`genim-api`) for `/v1/health`, `/v1/capabilities`, `/v1/generate`, and OpenAPI
documentation at `/docs`.
