"""
Multi-Task Neural Network for Ticket Triage

Architecture:
    Shared backbone (3 layers) -> Classification head (agency routing)
                                -> Regression head (SLA prediction)

Trained on NYC 311 dataset (227K+ tickets).
Input: DistilBERT embeddings (768) + temporal features (4) = 772 dims.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class MultiTaskTicketModel(nn.Module):
    """
    Multi-task neural network for ticket routing + SLA prediction.

    Args:
        input_dim: Input feature dimension (default 772: BERT 768 + temporal 4)
        num_classes: Number of routing categories
        hidden_dims: Hidden layer dimensions for shared backbone
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Shared backbone
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head (agency routing)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Regression head (SLA prediction)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_layers(x)
        class_logits = self.classification_head(shared_features)
        regression_pred = self.regression_head(shared_features).squeeze()
        return class_logits, regression_pred
