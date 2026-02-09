"""
Feature extraction and preprocessing for ticket triage inference.

Pipeline:
    Raw text -> DistilBERT [CLS] embedding (768-dim)
    Temporal features (hour, day_of_week, is_weekend, month) -> 4-dim
    Concatenated -> 772-dim feature vector
"""

import torch
import numpy as np
from typing import Optional


def get_bert_embedding(
    text: str,
    tokenizer,
    bert_model,
    device: torch.device,
    max_length: int = 512,
) -> np.ndarray:
    """
    Generate DistilBERT [CLS] embedding for a single text.

    Args:
        text: Input complaint text
        tokenizer: DistilBERT tokenizer
        bert_model: DistilBERT model (in eval mode)
        device: torch device (cpu/cuda)
        max_length: Maximum token length

    Returns:
        768-dimensional numpy array
    """
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

    return embedding


def prepare_features(
    text: str,
    hour: int,
    day_of_week: int,
    is_weekend: int,
    month: int,
    tokenizer,
    bert_model,
    device: torch.device,
) -> np.ndarray:
    """
    Build the full 772-dim feature vector for model inference.

    Args:
        text: Complaint text
        hour: Hour of ticket creation (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        is_weekend: Weekend flag (0 or 1)
        month: Month (1-12)
        tokenizer: DistilBERT tokenizer
        bert_model: DistilBERT model
        device: torch device

    Returns:
        772-dimensional numpy array (768 BERT + 4 temporal)
    """
    text_emb = get_bert_embedding(text, tokenizer, bert_model, device)

    temporal_features = np.array(
        [hour, day_of_week, is_weekend, month], dtype=np.float32
    )

    return np.concatenate([text_emb, temporal_features])
