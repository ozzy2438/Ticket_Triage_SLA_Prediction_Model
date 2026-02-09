"""
Unit tests for the SQLite persistence layer.
"""

import pytest
from src.db.persistence import DatabaseManager


class TestDatabaseManager:
    """Tests for prediction and feedback persistence."""

    def test_db_initialization(self, tmp_db):
        """Database should initialize with empty tables."""
        stats = tmp_db.get_stats()
        assert stats["total_processed"] == 0
        assert stats["total_feedback"] == 0

    def test_save_prediction(self, tmp_db):
        """Should persist a prediction record."""
        tmp_db.save_prediction(
            ticket_id="TEST-001",
            predicted_agency="Department of Health",
            confidence=0.92,
            sla_hours=18.5,
            action="AUTO_ROUTE",
            processing_time_ms=45.2,
        )

        stats = tmp_db.get_stats()
        assert stats["total_processed"] == 1
        assert stats["auto_routed"] == 1

    def test_save_multiple_predictions(self, tmp_db):
        """Should handle multiple predictions with different actions."""
        tmp_db.save_prediction("T-1", "Agency A", 0.9, 10.0, "AUTO_ROUTE", 30.0)
        tmp_db.save_prediction("T-2", "Agency B", 0.75, 15.0, "HUMAN_REVIEW", 35.0)
        tmp_db.save_prediction("T-3", "Agency C", 0.5, 20.0, "ESCALATE", 40.0)

        stats = tmp_db.get_stats()
        assert stats["total_processed"] == 3
        assert stats["auto_routed"] == 1
        assert stats["human_reviewed"] == 1
        assert stats["escalated"] == 1

    def test_save_feedback(self, tmp_db):
        """Should persist feedback and track corrections."""
        tmp_db.save_feedback(
            ticket_id="TEST-001",
            model_prediction="Agency A",
            human_decision="Agency A",
            was_corrected=False,
            operator_id="OP-1",
        )

        stats = tmp_db.get_stats()
        assert stats["total_feedback"] == 1
        assert stats["correct_predictions"] == 1

    def test_feedback_correction(self, tmp_db):
        """Should correctly track corrected predictions."""
        tmp_db.save_feedback("T-1", "Agency A", "Agency A", False)  # correct
        tmp_db.save_feedback("T-2", "Agency A", "Agency B", True)   # corrected

        stats = tmp_db.get_stats()
        assert stats["total_feedback"] == 2
        assert stats["correct_predictions"] == 1

    def test_recent_confidences(self, tmp_db):
        """Should return recent confidence scores for drift detection."""
        for i in range(5):
            tmp_db.save_prediction(f"T-{i}", "Agency", 0.8 + i * 0.02, 10.0, "AUTO_ROUTE", 30.0)

        recent = tmp_db.get_recent_confidences(limit=3)
        assert len(recent) == 3
        # Should be most recent first
        assert recent[0] >= recent[-1]

    def test_prediction_count(self, tmp_db):
        """Should return correct total count."""
        assert tmp_db.get_prediction_count() == 0

        tmp_db.save_prediction("T-1", "Agency", 0.9, 10.0, "AUTO_ROUTE", 30.0)
        assert tmp_db.get_prediction_count() == 1

    def test_avg_metrics(self, tmp_db):
        """Should calculate correct averages."""
        tmp_db.save_prediction("T-1", "A", 0.80, 10.0, "AUTO_ROUTE", 30.0)
        tmp_db.save_prediction("T-2", "B", 0.90, 20.0, "AUTO_ROUTE", 50.0)

        stats = tmp_db.get_stats()
        assert stats["avg_confidence"] == pytest.approx(0.85, abs=0.01)
        assert stats["avg_sla_hours"] == pytest.approx(15.0, abs=0.01)
