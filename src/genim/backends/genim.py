from __future__ import annotations

from pathlib import Path

import torch

from ..api import GenIM, SamplingConfig
from ..chem import ChemistryPolicy
from .base import (
    GenerationCondition,
    ProposalConfig,
    ProposedStructure,
    evaluate_condition,
)


def _slug(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    return "-".join(part for part in text.split("-") if part) or "model"


class GenIMBackend:
    """Expose a regular GenIM checkpoint through the proposal-backend contract."""

    backend_name = "genim"
    capabilities = frozenset({"elements"})

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
        condition: GenerationCondition,
        config: ProposalConfig,
    ) -> list[ProposedStructure]:
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
            ),
            chemistry=chemistry,
            restrict_elements=elements or None,
            validate=True,
            validation_options=config.validation_kwargs(),
        )
        requested = set(condition.requested_fields())
        applied = sorted(requested.intersection(self.capabilities))
        unsupported = sorted(requested.difference(self.capabilities))
        prefix = f"genim-{_slug(self.model_name)}"
        proposals: list[ProposedStructure] = []
        for index, result in enumerate(generated):
            proposals.append(
                ProposedStructure(
                    candidate_id=f"{prefix}-{index:05d}",
                    backend=self.backend_name,
                    model_name=self.model_name,
                    checkpoint_sha256=self.checkpoint_sha256,
                    atoms=result.atoms,
                    validation=result.validation,
                    condition=condition.as_dict(),
                    condition_checks=evaluate_condition(result.atoms, result.validation, condition),
                    applied_constraints=applied,
                    unsupported_constraints=unsupported,
                    sampling=config.as_dict(),
                    backend_metrics={
                        "token_count": len(result.tokens),
                        "chemistry_policy": dict(result.chemistry_policy),
                    },
                    raw_sequence=" ".join(result.tokens),
                    error=result.error,
                )
            )
        return proposals


__all__ = ["GenIMBackend"]
