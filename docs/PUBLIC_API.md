# GenMat public generation API

GenMat's canonical multi-backend vocabulary is intentionally model-neutral:

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
  "model": null,
  "n": 4,
  "temperature": 0.8,
  "seed": 7,
  "mutation_fraction": 0.0,
  "preserve_spacegroup": true
}
```

`n` is the final per-backend population target. With a nonzero
`mutation_fraction`, GenMat first requests `n - round(n * mutation_fraction)`
direct parents (at least one), then attempts to fill the remainder with
composition-preserving mutants. `mutation_strain`, `mutation_displacement`, and
`mutation_attempts` control the bounded mutation search. A mutant is retained
only after shared structural validation, constraint audit, and uniqueness
checks. Any observable tied to the parent geometry is invalidated.

The same controls are available from `genmat generate-ensemble` as
`--n-per-backend`, `--spacegroup`, `--mutation-fraction`,
`--mutation-strain`, `--mutation-displacement`, and `--mutation-attempts`.

`spacegroup_number` accepts an integer in 1..230. The checkpoint-backed GenMat
adapter converts it to a Hall token and forces it during sampling; local
outputs and mutants are still independently checked with spglib. A Hall token
being present in a universally seeded vocabulary does not establish that the
checkpoint learned that space group from adequate training examples.

`model` accepts a stable ID or alias from `ModelRegistry`; arbitrary client URLs
are not accepted. The response records both the requested reference and resolved
immutable model ID under `model_selection`.

`auto` prefers an available local Matra or GenMat checkpoint, then an Alexandria
remote backend when `GENMAT_ENABLE_ALEXANDRIA=1`, and otherwise selects the
non-ML `AlgorithmicSeedBackend`. Explicit unavailable checkpoint requests fail.
Install `genmat[api]` and run `genmat serve` (or
`genmat-api`) for `/v1/health`, `/v1/capabilities`, `/v1/models`,
`/v1/generate`, and OpenAPI
documentation at `/docs`.

`GET /v1/models` is discovery metadata, not a scientific endorsement. Models
whose separate license has not been acknowledged or whose optional runtime is
missing are returned with `available=false` and an explicit availability reason;
Studio does not offer them as runnable generation choices.
