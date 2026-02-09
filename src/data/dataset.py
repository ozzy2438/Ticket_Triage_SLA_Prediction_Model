"""
PyTorch Dataset for multi-task ticket triage training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class TicketTriageDataset(Dataset):
    """
    Custom Dataset for multi-task learning.

    Args:
        X: Feature matrix (BERT embeddings + temporal features)
        y_class: Classification labels (agency indices)
        y_reg: Regression targets (SLA hours, scaled)
    """

    def __init__(self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y_class = torch.LongTensor(y_class)
        self.y_reg = torch.FloatTensor(y_reg)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_class[idx], self.y_reg[idx]
