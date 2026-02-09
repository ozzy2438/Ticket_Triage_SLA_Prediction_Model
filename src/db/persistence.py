"""
SQLite persistence layer for feedback and statistics.

Survives API restarts - no more in-memory-only state.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages persistent storage for predictions and feedback.

    Args:
        db_path: Path to SQLite database file
    """

    def __init__(self, db_path: str = "data/ticket_triage.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    predicted_agency TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sla_hours REAL NOT NULL,
                    action TEXT NOT NULL,
                    processing_time_ms REAL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    model_prediction TEXT NOT NULL,
                    human_decision TEXT NOT NULL,
                    was_corrected INTEGER NOT NULL,
                    actual_resolution_hours REAL,
                    correction_reason TEXT,
                    operator_id TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_ticket
                ON predictions(ticket_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_ticket
                ON feedback(ticket_id)
            """)
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def save_prediction(
        self,
        ticket_id: str,
        predicted_agency: str,
        confidence: float,
        sla_hours: float,
        action: str,
        processing_time_ms: float,
    ):
        """Persist a prediction record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO predictions
                   (ticket_id, predicted_agency, confidence, sla_hours, action, processing_time_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ticket_id, predicted_agency, confidence, sla_hours, action, processing_time_ms,
                 datetime.now().isoformat()),
            )
            conn.commit()

    def save_feedback(
        self,
        ticket_id: str,
        model_prediction: str,
        human_decision: str,
        was_corrected: bool,
        actual_resolution_hours: Optional[float] = None,
        correction_reason: Optional[str] = None,
        operator_id: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Persist a feedback record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO feedback
                   (ticket_id, model_prediction, human_decision, was_corrected,
                    actual_resolution_hours, correction_reason, operator_id, notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticket_id, model_prediction, human_decision, int(was_corrected),
                 actual_resolution_hours, correction_reason, operator_id, notes,
                 datetime.now().isoformat()),
            )
            conn.commit()

    def get_stats(self) -> Dict:
        """Aggregate statistics from persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Prediction stats
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN action='AUTO_ROUTE' THEN 1 ELSE 0 END) as auto_routed,
                    SUM(CASE WHEN action='HUMAN_REVIEW' THEN 1 ELSE 0 END) as human_reviewed,
                    SUM(CASE WHEN action='ESCALATE' THEN 1 ELSE 0 END) as escalated,
                    AVG(confidence) as avg_confidence,
                    AVG(sla_hours) as avg_sla_hours,
                    AVG(processing_time_ms) as avg_processing_time
                FROM predictions
            """).fetchone()

            total = row["total"] or 0
            auto_routed = row["auto_routed"] or 0
            human_reviewed = row["human_reviewed"] or 0
            escalated = row["escalated"] or 0

            # Feedback stats
            fb = conn.execute("""
                SELECT
                    COUNT(*) as total_feedback,
                    SUM(CASE WHEN was_corrected=0 THEN 1 ELSE 0 END) as correct
                FROM feedback
            """).fetchone()

            total_feedback = fb["total_feedback"] or 0
            correct = fb["correct"] or 0

        return {
            "total_processed": total,
            "auto_routed": auto_routed,
            "human_reviewed": human_reviewed,
            "escalated": escalated,
            "avg_confidence": round(row["avg_confidence"] or 0, 4),
            "avg_sla_hours": round(row["avg_sla_hours"] or 0, 2),
            "avg_processing_time_ms": round(row["avg_processing_time"] or 0, 2),
            "total_feedback": total_feedback,
            "correct_predictions": correct,
        }

    def get_recent_confidences(self, limit: int = 100) -> List[float]:
        """Get recent confidence scores for drift detection."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT confidence FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r[0] for r in rows]

    def get_prediction_count(self) -> int:
        """Quick count of total predictions."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return row[0] if row else 0
