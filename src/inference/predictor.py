"""
Production inference pipeline for ticket triage.

Handles model loading, feature extraction, prediction, and HITL decisions.
"""

import torch
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.multi_task import MultiTaskTicketModel
from src.data.preprocessing import get_bert_embedding, prepare_features

logger = logging.getLogger(__name__)


class TicketPredictor:
    """
    Production-ready predictor that wraps model loading, inference, and HITL logic.

    Args:
        config: Application configuration dict with model/hitl settings
    """

    def __init__(self, config: dict):
        self.config = config
        self.model: Optional[MultiTaskTicketModel] = None
        self.tokenizer = None
        self.bert_model = None
        self.label_encoder = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False

    def load(self) -> bool:
        """Load all model artifacts. Returns True on success."""
        model_cfg = self.config["model"]
        checkpoint_dirs = self.config.get("checkpoint_dirs", ["models", "checkpoints"])

        checkpoint_path = None
        for dir_name in checkpoint_dirs:
            path = Path(dir_name)
            if path.exists() and (path / model_cfg["artifact_name"]).exists():
                checkpoint_path = path
                logger.info(f"Found model artifacts at: {path}")
                break

        if checkpoint_path is None:
            logger.warning("Model artifacts not found. Running in demo mode.")
            return False

        try:
            from transformers import DistilBertTokenizer, DistilBertModel

            logger.info("Loading DistilBERT...")
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.bert_model.to(self.device)
            self.bert_model.eval()

            with open(checkpoint_path / "label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"Loaded label encoder: {len(self.label_encoder.classes_)} classes")

            with open(checkpoint_path / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded StandardScaler")

            num_classes = len(self.label_encoder.classes_)
            self.model = MultiTaskTicketModel(
                input_dim=model_cfg["input_dim"],
                num_classes=num_classes,
                hidden_dims=model_cfg["hidden_dims"],
                dropout=model_cfg["dropout"],
            )

            checkpoint = torch.load(
                checkpoint_path / model_cfg["artifact_name"],
                map_location=self.device,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self.model_loaded = True
            logger.info(f"Model loaded successfully (device={self.device})")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False

    def predict(
        self,
        complaint_text: str,
        hour: int = 12,
        day_of_week: int = 3,
        is_weekend: int = 0,
        month: int = 7,
    ) -> Dict:
        """
        Run inference on a single ticket.

        Returns dict with: predicted_agency, confidence, sla_hours,
        action, escalation_reason, confidence_category, top_3_predictions
        """
        features = prepare_features(
            text=complaint_text,
            hour=hour,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            month=month,
            tokenizer=self.tokenizer,
            bert_model=self.bert_model,
            device=self.device,
        )
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            class_logits, sla_scaled = self.model(features_tensor)
            class_probs = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
            predicted_idx = int(np.argmax(class_probs))
            confidence = float(class_probs[predicted_idx])
            predicted_agency = self.label_encoder.inverse_transform([predicted_idx])[0]

            sla_hours = self.scaler.inverse_transform(
                sla_scaled.cpu().numpy().reshape(-1, 1)
            )[0][0]
            sla_hours = max(0.1, float(sla_hours))

        # Top 3 predictions
        top_indices = np.argsort(class_probs)[-3:][::-1]
        top_3 = [
            {
                "agency": self.label_encoder.inverse_transform([idx])[0],
                "probability": round(float(class_probs[idx]), 4),
            }
            for idx in top_indices
        ]

        # HITL decision
        action, reason, category = self._hitl_decision(confidence)

        return {
            "predicted_agency": predicted_agency,
            "confidence": round(confidence, 4),
            "sla_prediction_hours": round(sla_hours, 2),
            "action": action,
            "escalation_reason": reason,
            "confidence_category": category,
            "top_3_predictions": top_3,
        }

    def _hitl_decision(self, confidence: float) -> Tuple[str, str, str]:
        """Human-in-the-Loop decision based on confidence thresholds."""
        thresholds = self.config["hitl"]

        if confidence >= thresholds["auto_route_threshold"]:
            return "AUTO_ROUTE", "High confidence - auto-approved", "HIGH"
        elif confidence >= thresholds["human_review_threshold"]:
            return "HUMAN_REVIEW", "Medium confidence - human review recommended", "MEDIUM"
        else:
            return "ESCALATE", "Low confidence - escalate to supervisor", "LOW"
