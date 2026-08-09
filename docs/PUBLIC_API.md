# Public generation API

GenIM's canonical multi-backend vocabulary is intentionally model-neutral:

| Concept | Public type | Meaning |
|---|---|---|
| requested scientific intent | `GenerationConstraints` | composition, symmetry, or energy-related targets |
| run controls | `GenerationSettings` | sample count, seed, temperature, decoding and validation options |
| backend contract | `GenerationBackend` | a checkpoint-backed or explicitly algorithmic candidate generator |
| backend declaration | `BackendCapabilities` | how each requested constraint is applied |
| one result | `GeneratedCandidate` | decoded structure, provenance, validation, evidence and selection state |
| one property | `ScientificObservable` | value, units, scientific role, method, source, uncertainty and validation state |
| optional evidence stage | `ScientificEvaluator` | independent relaxation, energy, hull, DFT, phonon, or domain-specific evidence |
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
`constraint_status`, `constraint_assessments`, `scientific_observables`,
`ranking`, and `backend_capabilities` from the schema-version-3 report.

Each observable has a role: `conditioning_target`, `model_emission`,
`postprocessed_estimate`, or `calculated`. Only independently validated
observables may upgrade a stability or hull constraint from `not_evaluated`.
The schema therefore preserves useful Matra energy information without
asserting an unsupported thermodynamic interpretation.

## Service facade and HTTP API

`GeneratorService` is the canonical application boundary. It accepts a plain
mapping, normalizes formula or element/stoichiometry input, selects configured
backends, and returns schema version 3. Request defaults are:

```json
{
  "formula": "SiO2",
  "backend": "auto",
  "n": 4,
  "temperature": 0.8,
  "seed": 7
}
```

`auto` prefers an available local Matra or GenIM checkpoint, then an Alexandria
remote backend when `GENIM_ENABLE_ALEXANDRIA=1`, and otherwise selects the
non-ML `AlgorithmicSeedBackend`. Explicit unavailable checkpoint requests fail.
Install `genim[api]` and run `genim serve` (or
`genim-api`) for `/v1/health`, `/v1/capabilities`, `/v1/generate`, and OpenAPI
documentation at `/docs`.
