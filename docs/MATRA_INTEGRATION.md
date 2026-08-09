# Matra integration

GenIM 0.3 treats Matra Genoa as an optional generation backend. The integration
combines Matra's condition-aware Wyckoff representation with GenIM's shared ASE
validation, provenance, cross-model deduplication, MLIP/hull screening, and
benchmark reporting.

## License boundary

Matra's repository license permits non-commercial research, education,
evaluation, and personal use. It does not permit commercial use, sale,
sublicensing, paid services, or support of a revenue-generating workflow
without prior written permission.

GenIM therefore does not vendor Matra source code or checkpoint weights. Users
must install Matra separately from an approved source and preserve its license,
copyright, attribution, and modification notices. GenIM's BSD license does not
replace or broaden the Matra license.

## Security and checkpoint contract

`MatraBackend.from_checkpoint(...)`:

1. computes and optionally verifies SHA256;
2. loads the checkpoint with `torch.load(..., weights_only=True)`;
3. validates the vocabulary, configuration, tensor payload, and required keys;
4. constructs `MatraGenoa(pretrained=False)` from primitive configuration data;
5. loads the state dictionary and rejects any missing or unexpected weights.

This bypasses Matra's historical convenience path that loads pickle-compatible
objects with `weights_only=False`. Matra checkpoints are not relabeled as GenIM
format-v2 checkpoints because their vocabularies, token semantics, output
heads, and positional lengths differ.

## Conditions and evidence

Matra prompts can apply:

- discrete stability (`EHULL_DISC`);
- exact element set and amount (`ELMS`, `AMT`);
- stoichiometry (`STOICH`);
- space group (`SPACEGROUP`);
- Matra checkpoint-specific Wyckoff indices (`WYCKOFF`);
- continuous hull target (`EHULL`).

Element set, stoichiometry, and decoded space group are independently checked
after conversion to ASE. Stability, continuous hull targets, and Matra Wyckoff
indices cannot all be independently proven from the decoded ASE structure
alone; their checks remain `null` unless a later evidence stage evaluates
them. A Matra prompt being applied is not the same as its scientific target
being verified.

GenIM currently applies exact element-set conditioning through its vocabulary
mask. Other Matra-specific conditions are recorded as unsupported for the
GenIM backend and are still evaluated after decoding where possible. Nothing
is silently reported as enforced.

## Output contract

`genim generate-ensemble` (legacy alias: `hybrid-generate`) writes:

- selected, unique CIF files;
- `candidates.jsonl`, containing every valid or rejected proposal, raw sequence,
  checkpoint SHA256, condition checks, validation metrics, errors, and duplicate
  linkage;
- `ensemble-report.json`, with total, valid, unique, selected, unknown-constraint,
  and rejection counts for each backend.

`selected` is intentionally weaker than "stable": it means structurally valid,
constraint-not-disproved, and unique in that run. `accepted` is retained only
as a compatibility field for GenIM 0.3 readers.

Every requested constraint also receives an auditable three-state assessment:
`satisfied`, `violated`, or `not_evaluated`, including the method and evidence
level. The backend capability record separately identifies whether a request
was applied by conditioning, a sampling filter, a post-filter, or not supported.

## Scientific comparison protocol

Compare at least:

1. GenIM alone;
2. each Matra checkpoint alone;
3. the deduplicated union;
4. any later cross-conditioned or distilled model.

Use both equal-sample and equal-wall-time budgets. Report raw decoding yield,
geometric validity, condition compliance/unknown rate, uniqueness, novelty
against training data, composition and space-group coverage, MLIP relaxation
success, energy-above-hull distribution, latency, and peak memory.

Because Matra-MP/MPAS and GenIM may share Materials Project structures, use
composition- and prototype-held-out evaluation and deduplicate the evaluation
set against all known training corpora. Do not claim improvement from a union
until this leakage control is in place.

## Experimental transfer learning

Matra-MPAS-Med and the historical GenIM checkpoint both use six 256-dimensional
layers with eight attention heads, but their token semantics and output heads
remain incompatible. Layer-wise warm-starting may be studied as an ablation,
but direct state-dictionary averaging or concatenation is invalid.

The safer transfer route is structure-level distillation:

1. generate Matra candidates with recorded checkpoint and prompt;
2. validate, deduplicate, relax, and score them;
3. keep provenance and teacher-source labels;
4. exclude held-out compositions and prototypes;
5. retrain a GenIM-format model and compare it against the unaugmented baseline.
