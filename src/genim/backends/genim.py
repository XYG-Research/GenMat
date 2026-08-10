from __future__ import annotations

from pathlib import Path

import torch

from ..api import GenIM, SamplingConfig
from ..chem import ChemistryPolicy
from .base import (
    BackendCapabilities,
    ConstraintApplication,
    GeneratedCandidate,
    GenerationConstraints,
    GenerationSettings,
    evaluate_constraints,
)


def _slug(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    return "-".join(part for part in text.split("-") if part) or "model"


class GenIMBackend:
    """Expose a regular GenIM checkpoint through the proposal-backend contract."""

    backend_name = "genim"
    capabilities = BackendCapabilities(
        {
            "elements": ConstraintApplication.SAMPLING_FILTER,
            "spacegroup_number": ConstraintApplication.CONDITIONING,
        },
        notes=(
            "Element symbols are restricted by a vocabulary mask. Exact space group "
            "requests force the corresponding Hall token before decoding and are then "
            "independently checked with spglib. Other constraints are evaluated after decoding."
        ),
    )

    def __init__(self, generator: GenIM, *, model_name: str | None = None):
        self.generator = generator
        self.model_name = model_name or generator.checkpoint.path.name
        self.checkpoint_sha256 = generator.checkpoint.sha256

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "auto",
        expected_sha256: str | None = None,
    ) -> "GenIMBackend":
        generator = GenIM.from_checkpoint(path, device=device, expected_sha256=expected_sha256)
        return cls(generator, model_name=Path(path).name)

    def propose(
        self,
        *,
        condition: GenerationConstraints,
        config: GenerationSettings,
    ) -> list[GeneratedCandidate]:
        elements = list(condition.elements)
        if elements and not condition.exact_elements:
            raise ValueError("GenIM element-token restriction supports exact element sets, not required subsets")
        if elements:
            chemistry = ChemistryPolicy(
                mode="any",
                allowed_elements=elements,
                min_elements=len(elements),
                max_elements=len(elements),
            )
        else:
            chemistry = ChemistryPolicy(mode="any")

        generated = self.generator.sample(
            config=SamplingConfig(
                n=int(config.n),
                batch_size=int(config.batch_size),
                max_sites=int(config.max_sites),
                min_sites=int(config.min_sites),
                temperature=float(config.temperature),
                top_k=int(config.top_k),
                seed=config.seed,
                fixed_spacegroup=condition.spacegroup_number,
            ),
            chemistry=chemistry,
            restrict_elements=elements or None,
            validate=True,
            validation_options=config.validation_kwargs(),
        )
        requested = set(condition.requested_fields())
        applied = self.capabilities.applied_fields(requested)
        unsupported = self.capabilities.unsupported_fields(requested)
        prefix = f"genim-{_slug(self.model_name)}"
        proposals: list[GeneratedCandidate] = []
        for index, result in enumerate(generated):
            assessments = evaluate_constraints(result.atoms, result.validation, condition)
            proposals.append(
                GeneratedCandidate(
                    candidate_id=f"{prefix}-{index:05d}",
                    backend=self.backend_name,
                    model_name=self.model_name,
                    checkpoint_sha256=self.checkpoint_sha256,
                    atoms=result.atoms,
                    validation=result.validation,
                    condition=condition.as_dict(),
                    condition_checks={
                        name: assessment.value for name, assessment in assessments.items()
                    },
                    applied_constraints=applied,
                    unsupported_constraints=unsupported,
                    sampling=config.as_dict(),
                    constraint_assessments=assessments,
                    backend_metrics={
                        "token_count": len(result.tokens),
                        "chemistry_policy": dict(result.chemistry_policy),
                        "forced_hall_number": result.sampling.get("forced_hall_number"),
                        "hall_resolution_source": result.sampling.get("hall_resolution_source"),
                    },
                    raw_sequence=" ".join(result.tokens),
                    error=result.error,
                )
            )
        return proposals


__all__ = ["GenIMBackend"]
