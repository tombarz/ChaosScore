from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models.scfoundation_conditioned_mgp import (
    CellTypeConditionedMaskedGenePredictor,
    MaskedAttentionPooler,
    ScFoundationEncoderBackbone,
)


class MaskedAttentionPoolerTests(unittest.TestCase):
    def test_pooler_ignores_padded_tokens(self) -> None:
        torch.manual_seed(0)
        pooler = MaskedAttentionPooler(hidden_dim=4)
        encoded_tokens = torch.randn(2, 3, 4)
        padding_mask = torch.tensor(
            [
                [False, False, True],
                [False, True, True],
            ]
        )

        baseline = pooler(encoded_tokens, padding_mask)
        changed = encoded_tokens.clone()
        changed[padding_mask] = 1000.0
        updated = pooler(changed, padding_mask)

        self.assertEqual(tuple(baseline.shape), (2, 4))
        torch.testing.assert_close(updated, baseline)

    def test_pooler_returns_finite_zeros_for_all_padding_rows(self) -> None:
        pooler = MaskedAttentionPooler(hidden_dim=4)
        encoded_tokens = torch.randn(2, 3, 4)
        padding_mask = torch.tensor(
            [
                [True, True, True],
                [False, True, True],
            ]
        )

        pooled = pooler(encoded_tokens, padding_mask)

        self.assertTrue(torch.isfinite(pooled).all().item())
        torch.testing.assert_close(pooled[0], torch.zeros(4))

    def test_pooler_parameters_receive_gradients(self) -> None:
        torch.manual_seed(0)
        pooler = MaskedAttentionPooler(hidden_dim=4)
        encoded_tokens = torch.randn(2, 3, 4, requires_grad=True)
        padding_mask = torch.tensor(
            [
                [False, False, True],
                [False, True, True],
            ]
        )

        loss = pooler(encoded_tokens, padding_mask).sum()
        loss.backward()

        gradients = [parameter.grad for parameter in pooler.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(any(bool((gradient != 0).any().item()) for gradient in gradients if gradient is not None))

    def test_pooler_exposes_normalized_non_padding_weights(self) -> None:
        pooler = MaskedAttentionPooler(hidden_dim=4)
        encoded_tokens = torch.randn(2, 3, 4)
        padding_mask = torch.tensor(
            [
                [False, False, True],
                [True, True, True],
            ]
        )

        weights = pooler.compute_attention_weights(encoded_tokens, padding_mask)

        self.assertEqual(tuple(weights.shape), (2, 3))
        self.assertAlmostEqual(float(weights[0].sum().detach()), 1.0, places=6)
        self.assertEqual(float(weights[0, 2].detach()), 0.0)
        torch.testing.assert_close(weights[1], torch.zeros(3))


class AttentionPoolingDetailsTests(unittest.TestCase):
    def test_backbone_returns_attention_weights_with_matching_gene_ids(self) -> None:
        class TokenEmbedding(nn.Module):
            def forward(self, x: torch.Tensor, output_weight: int = 0) -> torch.Tensor:
                return torch.cat([x, x + 1.0], dim=-1)

        class IdentityEncoder(nn.Module):
            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
                return x

        def gather_data(data: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
            widths = labels.sum(dim=1).tolist()
            max_width = int(max(widths))
            gathered = data.new_full((data.shape[0], max_width), pad_token_id)
            padding = torch.ones((data.shape[0], max_width), dtype=torch.bool, device=data.device)
            for row_idx, width in enumerate(widths):
                width = int(width)
                if width == 0:
                    continue
                values = data[row_idx, labels[row_idx]]
                gathered[row_idx, :width] = values
                padding[row_idx, :width] = False
            return gathered, padding

        backbone = ScFoundationEncoderBackbone.__new__(ScFoundationEncoderBackbone)
        nn.Module.__init__(backbone)
        backbone.pooling = "attention"
        backbone.pad_token_id = 99
        backbone._gather_data = gather_data
        backbone.token_emb = TokenEmbedding()
        backbone.pos_emb = nn.Embedding(100, 2)
        backbone.encoder = IdentityEncoder()
        backbone.attention_pooler = MaskedAttentionPooler(hidden_dim=2)

        x_masked = torch.tensor(
            [
                [0.0, 2.0, 0.0, 4.0],
                [3.0, 0.0, 0.0, 0.0],
            ]
        )

        details = backbone.attention_pooling_details(x_masked)

        self.assertEqual(tuple(details.pooled.shape), (2, 2))
        self.assertEqual(tuple(details.attention_weights.shape), (2, 2))
        self.assertEqual(details.gene_ids.tolist(), [[1, 3], [0, 99]])
        self.assertEqual(details.padding_mask.tolist(), [[False, False], [False, True]])
        self.assertAlmostEqual(float(details.attention_weights[0].sum().detach()), 1.0, places=6)
        self.assertAlmostEqual(float(details.attention_weights[1].sum().detach()), 1.0, places=6)
        self.assertEqual(float(details.attention_weights[1, 1].detach()), 0.0)

    def test_attention_details_rejects_non_attention_pooling(self) -> None:
        backbone = ScFoundationEncoderBackbone.__new__(ScFoundationEncoderBackbone)
        nn.Module.__init__(backbone)
        backbone.pooling = "max"
        backbone.attention_pooler = None

        with self.assertRaisesRegex(ValueError, "pooling='attention'"):
            backbone.attention_pooling_details(torch.zeros(1, 3))


class AttentionPoolingPredictorShapeTests(unittest.TestCase):
    def test_attention_pooling_uses_single_hidden_width_cell_embedding(self) -> None:
        class DummyBackbone(nn.Module):
            hidden_dim = 6
            pooling = "attention"

            def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
                return x_masked.new_zeros((x_masked.shape[0], self.hidden_dim))

        model = CellTypeConditionedMaskedGenePredictor(
            backbone=DummyBackbone(),  # type: ignore[arg-type]
            num_cell_types=3,
            num_genes=5,
            d_type=2,
            d_depth=3,
            d_gene=4,
            hidden_dim=7,
            use_depth_covariate=True,
        )

        self.assertEqual(model.prediction_head[0].in_features, 6 + 2 + 3 + 4)

        predictions = model(
            x_masked=torch.zeros(2, 5),
            masked_gene_ids=torch.tensor([[0, 1], [2, 5]]),
            masked_positions_valid=torch.tensor([[True, True], [True, False]]),
            cell_type_ids=torch.tensor([0, 2]),
            depth_features=torch.tensor([1.0, 2.0]),
        )

        self.assertEqual(tuple(predictions.shape), (2, 2))
        self.assertEqual(float(predictions[1, 1].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
