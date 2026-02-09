"""
=============================================================================
PRODUCTION-READY TICKET TRIAGE API
=============================================================================
AI-Powered Customer Support Automation System

Key Features:
- Multi-task ML: Automatic routing + SLA prediction
- Human-in-the-loop: Confidence-based escalation
- Real-time monitoring: Performance tracking & drift detection
- Batch processing: Handle high volumes efficiently
- Persistent storage: SQLite-backed stats and feedback
- RESTful API: Easy integration with existing systems

Use Case: Restaurant/Customer Support Automation (HungerRush-ready)
=============================================================================
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.multi_task import MultiTaskTicketModel
from src.inference.predictor import TicketPredictor
from src.config.settings import load_config
from src.db.persistence import DatabaseManager

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class TicketRequest(BaseModel):
    """Ticket prediction request."""
    ticket_id: str = Field(..., description="Unique ticket identifier", example="TKT-20250205-001")
    complaint_text: str = Field(
        ..., min_length=10, max_length=5000,
        description="Ticket description/complaint text",
        example="Order delayed for 2 hours, customer is waiting",
    )
    hour: int = Field(12, ge=0, le=23, description="Hour ticket was created (0-23)")
    day_of_week: int = Field(3, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_weekend: int = Field(0, ge=0, le=1, description="Is weekend (0=No, 1=Yes)")
    month: int = Field(7, ge=1, le=12, description="Month (1-12)")
    borough: str = Field("MANHATTAN", description="Borough/Location", example="MANHATTAN")
    priority: Optional[str] = Field(None, description="Priority level (if known)")
    customer_id: Optional[str] = Field(None, description="Customer ID (if applicable)")

    @validator("is_weekend", pre=True, always=True)
    def auto_calculate_weekend(cls, v, values):
        if v is None and "day_of_week" in values:
            return 1 if values["day_of_week"] >= 5 else 0
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TKT-20250205-001",
                "complaint_text": "Food delivery is extremely late, customer called 3 times",
                "hour": 19, "day_of_week": 5, "is_weekend": 1,
                "month": 2, "borough": "BROOKLYN",
                "priority": "high", "customer_id": "CUST-12345",
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response with full details."""
    ticket_id: str
    predicted_agency: str
    confidence: float = Field(..., ge=0, le=1)
    sla_prediction_hours: float = Field(..., ge=0)
    action: str = Field(..., description="AUTO_ROUTE | HUMAN_REVIEW | ESCALATE")
    escalation_reason: str
    top_3_predictions: List[Dict[str, Any]]
    confidence_category: str = Field(..., description="HIGH | MEDIUM | LOW")
    processed_at: str
    processing_time_ms: float
    model_version: str = "1.0.0"


class BatchPredictionRequest(BaseModel):
    """Batch prediction request (up to 100 tickets)."""
    tickets: List[TicketRequest]

    @validator("tickets")
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 tickets per batch")
        if len(v) == 0:
            raise ValueError("At least 1 ticket required")
        return v


class BatchPredictionResponse(BaseModel):
    total_processed: int
    successful: int
    failed: int
    total_time_ms: float
    predictions: List[PredictionResponse]
    errors: List[Dict[str, str]] = []


class FeedbackRequest(BaseModel):
    """Human operator feedback for active learning."""
    ticket_id: str
    model_prediction: str
    human_decision: str
    actual_resolution_hours: Optional[float] = None
    correction_reason: Optional[str] = None
    operator_id: Optional[str] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    ticket_id: str
    was_corrected: bool
    model_accuracy_impact: str
    recorded_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_version: str
    uptime_seconds: float
    total_predictions: int
    avg_processing_time_ms: float
    last_prediction_at: Optional[str]


class StatsResponse(BaseModel):
    total_processed: int
    auto_routed: int
    human_reviewed: int
    escalated: int
    automation_rate: str
    avg_confidence: float
    avg_sla_hours: float
    total_feedback: int
    correct_predictions: int
    accuracy_rate: str
    estimated_time_saved_hours: float
    estimated_cost_saved: float
    period: str = "all_time"
    last_updated: str


# =============================================================================
# APP INITIALIZATION
# =============================================================================

config = load_config()
predictor = TicketPredictor(config)
db = DatabaseManager(config.get("database", {}).get("path", "data/ticket_triage.db"))
start_time = datetime.now()
last_prediction_at: Optional[str] = None

app = FastAPI(
    title="AI-Powered Ticket Triage System",
    description="""
    **Production-Ready Customer Support Automation**

    ## Business Problem
    Manual ticket routing causes delays, inconsistencies, and SLA breaches.
    This system automates 75%+ of routing decisions while maintaining quality.

    ## Key Features

    ### 1. Intelligent Routing (Multi-class Classification)
    - Automatically assigns tickets to correct department/agent
    - 90%+ accuracy on test data
    - Handles 20+ categories

    ### 2. SLA Prediction (Regression)
    - Predicts resolution time in hours
    - Enables proactive resource allocation

    ### 3. Human-in-the-Loop (HITL)
    - High confidence (>85%): Auto-route
    - Medium confidence (70-85%): Human review suggested
    - Low confidence (<70%): Escalate to supervisor

    ### 4. Persistent Storage
    - All predictions and feedback stored in SQLite
    - Survives API restarts
    - Ready for migration to PostgreSQL

    ## Endpoints
    - **POST /predict**: Single ticket prediction
    - **POST /predict/batch**: Batch processing (up to 100 tickets)
    - **POST /feedback**: Record operator corrections
    - **GET /stats**: Performance metrics & ROI
    - **GET /health**: System status
    - **GET /monitoring/drift**: Data/model drift detection
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "ML Team", "email": "ml@company.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# STARTUP & SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    global start_time
    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("Ticket Triage API - Starting...")
    logger.info("=" * 60)

    model_loaded = predictor.load()

    if model_loaded:
        logger.info("Production mode: Real model loaded")
        logger.info(f"  Device: {predictor.device}")
        logger.info(f"  Classes: {len(predictor.label_encoder.classes_)}")
    else:
        logger.info("Demo mode: Model not loaded")
        logger.info("  Train model via notebooks/04_evaluation.ipynb")

    logger.info("=" * 60)
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Health Check: http://localhost:8000/health")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    stats = db.get_stats()
    logger.info("API shutting down...")
    logger.info(f"Total predictions: {stats['total_processed']}")
    logger.info(f"Auto-routed: {stats['auto_routed']}")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "AI-Powered Ticket Triage System",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_loaded": predictor.model_loaded,
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    System health check endpoint.

    Use Case: Monitoring tools (Prometheus, Datadog) poll this endpoint
    to ensure the service is operational.
    """
    uptime = (datetime.now() - start_time).total_seconds()
    stats = db.get_stats()

    if not predictor.model_loaded:
        health_status = "degraded"
    elif stats["avg_processing_time_ms"] > 200:
        health_status = "degraded"
    else:
        health_status = "healthy"

    return HealthResponse(
        status=health_status,
        model_loaded=predictor.model_loaded,
        device=str(predictor.device),
        model_version=config["model"]["version"],
        uptime_seconds=round(uptime, 2),
        total_predictions=stats["total_processed"],
        avg_processing_time_ms=stats["avg_processing_time_ms"],
        last_prediction_at=last_prediction_at,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_ticket(request: TicketRequest, background_tasks: BackgroundTasks):
    """
    **Single ticket prediction** - Main endpoint for real-time routing.

    ## Workflow
    1. Feature Extraction: Convert text to BERT embeddings + temporal features
    2. Model Inference: Multi-task prediction (routing + SLA)
    3. HITL Decision: Auto-route, review, or escalate based on confidence
    4. Persist: Save prediction to database

    ## Business Impact
    - High confidence (>85%): Auto-route -> saves ~15 min/ticket
    - Medium confidence (70-85%): Human review -> saves ~10 min/ticket
    - Low confidence (<70%): Escalate -> ensures quality
    """
    global last_prediction_at

    if not predictor.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. API running in demo mode.",
        )

    start_time_ms = time.time() * 1000

    try:
        result = predictor.predict(
            complaint_text=request.complaint_text,
            hour=request.hour,
            day_of_week=request.day_of_week,
            is_weekend=request.is_weekend,
            month=request.month,
        )

        processing_time = (time.time() * 1000) - start_time_ms
        last_prediction_at = datetime.now().isoformat()

        # Persist prediction in background
        background_tasks.add_task(
            db.save_prediction,
            ticket_id=request.ticket_id,
            predicted_agency=result["predicted_agency"],
            confidence=result["confidence"],
            sla_hours=result["sla_prediction_hours"],
            action=result["action"],
            processing_time_ms=round(processing_time, 2),
        )

        return PredictionResponse(
            ticket_id=request.ticket_id,
            predicted_agency=result["predicted_agency"],
            confidence=result["confidence"],
            sla_prediction_hours=result["sla_prediction_hours"],
            action=result["action"],
            escalation_reason=result["escalation_reason"],
            top_3_predictions=result["top_3_predictions"],
            confidence_category=result["confidence_category"],
            processed_at=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2),
            model_version=config["model"]["version"],
        )

    except Exception as e:
        logger.error(f"Prediction error for {request.ticket_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    **Batch prediction** - Process up to 100 tickets at once.

    Use Cases:
    - Process backlog of tickets overnight
    - Bulk import from external systems
    - Historical data reclassification
    """
    if not predictor.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    start_time_batch = time.time()
    predictions = []
    errors = []
    successful = 0

    for ticket in request.tickets:
        try:
            result = await predict_ticket(ticket, BackgroundTasks())
            predictions.append(result)
            successful += 1
        except Exception as e:
            logger.error(f"Batch prediction error for {ticket.ticket_id}: {e}")
            errors.append({"ticket_id": ticket.ticket_id, "error": str(e)})

    total_time_ms = (time.time() - start_time_batch) * 1000

    return BatchPredictionResponse(
        total_processed=len(request.tickets),
        successful=successful,
        failed=len(errors),
        total_time_ms=round(total_time_ms, 2),
        predictions=predictions,
        errors=errors,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def record_feedback(request: FeedbackRequest):
    """
    **Record operator feedback** - Active learning & monitoring.

    Purpose:
    1. Track accuracy: Compare model vs. human decisions
    2. Identify patterns: Find systematic errors
    3. Enable retraining: Collect labeled data for model updates
    4. Measure ROI: Calculate cost savings and automation rate
    """
    was_corrected = request.model_prediction != request.human_decision

    db.save_feedback(
        ticket_id=request.ticket_id,
        model_prediction=request.model_prediction,
        human_decision=request.human_decision,
        was_corrected=was_corrected,
        actual_resolution_hours=request.actual_resolution_hours,
        correction_reason=request.correction_reason,
        operator_id=request.operator_id,
        notes=request.notes,
    )

    stats = db.get_stats()
    total_fb = stats["total_feedback"]
    correct = stats["correct_predictions"]
    accuracy = correct / total_fb if total_fb > 0 else 0

    if accuracy > 0.9:
        impact = "maintained"
    elif accuracy > 0.85:
        impact = "improved"
    else:
        impact = "degraded"

    logger.info(
        f"Feedback recorded: {request.ticket_id} | "
        f"Corrected: {was_corrected} | Accuracy: {accuracy:.2%}"
    )

    return FeedbackResponse(
        status="recorded",
        ticket_id=request.ticket_id,
        was_corrected=was_corrected,
        model_accuracy_impact=impact,
        recorded_at=datetime.now().isoformat(),
    )


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """
    **Performance statistics** - Business metrics & ROI.

    Key Metrics:
    - Automation Rate: % of tickets auto-routed (target >75%)
    - Accuracy Rate: % of correct predictions from feedback
    - Cost Savings: Based on time saved per ticket ($30/hour)

    Data persists across API restarts via SQLite.
    """
    stats = db.get_stats()
    total = stats["total_processed"]

    automation_rate = (stats["auto_routed"] / total * 100) if total > 0 else 0

    total_fb = stats["total_feedback"]
    accuracy_rate = (stats["correct_predictions"] / total_fb * 100) if total_fb > 0 else 0

    business = config.get("business", {})
    hourly_rate = business.get("hourly_rate", 30)
    auto_save = business.get("auto_route_time_saved_min", 15)
    review_save = business.get("human_review_time_saved_min", 10)

    time_saved_hours = (stats["auto_routed"] * auto_save + stats["human_reviewed"] * review_save) / 60
    cost_saved = time_saved_hours * hourly_rate

    return StatsResponse(
        total_processed=total,
        auto_routed=stats["auto_routed"],
        human_reviewed=stats["human_reviewed"],
        escalated=stats["escalated"],
        automation_rate=f"{automation_rate:.1f}%",
        avg_confidence=stats["avg_confidence"],
        avg_sla_hours=stats["avg_sla_hours"],
        total_feedback=total_fb,
        correct_predictions=stats["correct_predictions"],
        accuracy_rate=f"{accuracy_rate:.1f}%",
        estimated_time_saved_hours=round(time_saved_hours, 2),
        estimated_cost_saved=round(cost_saved, 2),
        period="all_time",
        last_updated=datetime.now().isoformat(),
    )


@app.get("/monitoring/drift", tags=["Monitoring"])
async def get_drift_report():
    """
    **Drift detection report** - Data & model monitoring.

    Checks three types of drift:
    1. Data Drift: Average confidence dropping below threshold
    2. Prediction Drift: Escalation rate exceeding threshold
    3. Performance Drift: Accuracy falling below threshold (from feedback)

    Thresholds are configurable via configs/app_config.yaml.
    """
    stats = db.get_stats()
    mon_cfg = config.get("monitoring", {})

    # Data drift
    recent = db.get_recent_confidences(limit=mon_cfg.get("processing_time_window", 100))
    avg_recent_confidence = sum(recent) / len(recent) if recent else 0.85

    conf_threshold = mon_cfg.get("confidence_drift_threshold", 0.75)
    data_drift_detected = avg_recent_confidence < conf_threshold
    data_drift_severity = (
        "high" if avg_recent_confidence < conf_threshold - 0.05
        else "medium" if avg_recent_confidence < conf_threshold
        else "normal"
    )

    # Prediction drift
    total = stats["total_processed"]
    escalation_rate = stats["escalated"] / total if total > 0 else 0
    esc_threshold = mon_cfg.get("escalation_rate_threshold", 0.30)
    prediction_drift_detected = escalation_rate > esc_threshold
    prediction_drift_severity = (
        "high" if escalation_rate > esc_threshold + 0.10
        else "medium" if escalation_rate > esc_threshold
        else "normal"
    )

    # Performance drift
    total_fb = stats["total_feedback"]
    accuracy = stats["correct_predictions"] / total_fb if total_fb > 10 else 1.0
    acc_threshold = mon_cfg.get("accuracy_drift_threshold", 0.85)
    performance_drift_detected = accuracy < acc_threshold
    performance_drift_severity = (
        "high" if accuracy < acc_threshold - 0.05
        else "medium" if accuracy < acc_threshold
        else "normal"
    )

    drift_count = sum([data_drift_detected, prediction_drift_detected, performance_drift_detected])
    overall_health = "HEALTHY" if drift_count == 0 else ("WARNING" if drift_count == 1 else "CRITICAL")

    return {
        "timestamp": datetime.now().isoformat(),
        "overall_health": overall_health,
        "data_drift": {
            "detected": data_drift_detected,
            "severity": data_drift_severity,
            "avg_confidence": round(avg_recent_confidence, 4),
            "threshold": conf_threshold,
            "recommendation": "Consider model retraining" if data_drift_detected else "No action needed",
        },
        "prediction_drift": {
            "detected": prediction_drift_detected,
            "severity": prediction_drift_severity,
            "escalation_rate": f"{escalation_rate:.2%}",
            "threshold": f"{esc_threshold:.0%}",
            "recommendation": "Investigate high escalation rate" if prediction_drift_detected else "No action needed",
        },
        "performance_drift": {
            "detected": performance_drift_detected,
            "severity": performance_drift_severity,
            "accuracy": f"{accuracy:.2%}",
            "threshold": f"{acc_threshold:.0%}",
            "recommendation": "Model retraining recommended" if performance_drift_detected else "No action needed",
        },
        "metrics_summary": {
            "total_predictions": total,
            "total_feedback": total_fb,
            "avg_confidence": round(avg_recent_confidence, 4),
            "accuracy": f"{accuracy:.2%}",
            "escalation_rate": f"{escalation_rate:.2%}",
        },
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("  AI-POWERED TICKET TRIAGE SYSTEM")
    print("=" * 70)
    print("   Swagger UI:    http://localhost:8000/docs")
    print("   ReDoc:         http://localhost:8000/redoc")
    print("   Health Check:  http://localhost:8000/health")
    print("   Statistics:    http://localhost:8000/stats")
    print("   Drift Monitor: http://localhost:8000/monitoring/drift")
    print("=" * 70)

    api_cfg = config.get("api", {})
    uvicorn.run(
        "main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=True,
        log_level="info",
    )
