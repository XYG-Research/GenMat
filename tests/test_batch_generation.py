from __future__ import annotations

import torch

from genim.generate import sample_sequences


class _CountingModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, input_ids: torch.Tensor, *, key_padding_mask=None) -> torch.Tensor:
        self.calls += 1
        self.batch_sizes.append(int(input_ids.shape[0]))
        batch, length = input_ids.shape
        logits = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
        return logits.expand(batch, length, self.vocab_size)


def test_batch_sampler_uses_one_model_call_per_position() -> None:
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "HALL_1",
        "LEN_0",
        "ANG_0",
        "E_O",
        "E_Na",
        "W_a",
        "COORD_0",
    ]
    vocab = {token: index for index, token in enumerate(tokens)}
    model = _CountingModel(len(tokens))
    sequences = sample_sequences(
        model,
        n=4,
        vocab=vocab,
        max_len=14,
        temperature=1.0,
        top_k=1,
        max_sites=1,
        min_sites=1,
        restrict_elements=["O", "Na"],
        device=torch.device("cpu"),
    )
    assert len(sequences) == 4
    assert all(sequence[-1] == vocab["<EOS>"] for sequence in sequences)
    assert model.calls == 13
    assert model.batch_sizes == [4] * 13
