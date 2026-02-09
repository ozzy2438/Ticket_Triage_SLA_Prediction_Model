"""
Unit tests for the MultiTaskTicketModel architecture.
"""

import pytest
import torch
import numpy as np
from src.models.multi_task import MultiTaskTicketModel


class TestMultiTaskModel:
    """Tests for model architecture and forward pass."""

    def test_model_creation(self, sample_model):
        """Model should initialize with correct architecture."""
        assert sample_model is not None
        assert isinstance(sample_model, torch.nn.Module)

    def test_forward_output_shapes(self, sample_model):
        """Forward pass should return correct output shapes."""
        batch_size = 4
        x = torch.randn(batch_size, 772)

        class_logits, sla_pred = sample_model(x)

        assert class_logits.shape == (batch_size, 20), f"Expected (4, 20), got {class_logits.shape}"
        assert sla_pred.shape == (batch_size,), f"Expected (4,), got {sla_pred.shape}"

    def test_single_sample_forward(self, sample_model):
        """Model should handle single sample input in eval mode (production use case)."""
        sample_model.eval()
        x = torch.randn(1, 772)

        with torch.no_grad():
            class_logits, sla_pred = sample_model(x)

        assert class_logits.shape == (1, 20)
        # Single sample squeeze gives scalar
        assert sla_pred.dim() == 0 or sla_pred.shape == (1,)

    def test_softmax_probabilities(self, sample_model):
        """Classification output should produce valid probabilities after softmax."""
        x = torch.randn(2, 772)
        class_logits, _ = sample_model(x)
        probs = torch.softmax(class_logits, dim=1)

        # Probabilities should sum to 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_custom_hidden_dims(self):
        """Model should accept custom hidden dimensions."""
        model = MultiTaskTicketModel(
            input_dim=100,
            num_classes=5,
            hidden_dims=[64, 32, 16],
            dropout=0.1,
        )
        x = torch.randn(2, 100)
        class_logits, sla_pred = model(x)

        assert class_logits.shape == (2, 5)

    def test_parameter_count(self, sample_model):
        """Model should have trainable parameters."""
        total_params = sum(p.numel() for p in sample_model.parameters())
        trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable

    def test_eval_mode(self, sample_model):
        """Model should work in eval mode (no dropout, frozen batchnorm)."""
        sample_model.eval()
        x = torch.randn(2, 772)

        with torch.no_grad():
            out1 = sample_model(x)
            out2 = sample_model(x)

        # In eval mode, same input should give same output
        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])
