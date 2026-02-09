"""
Configuration management - loads settings from YAML files.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class AppConfig:
    """Typed configuration container."""
    model: Dict[str, Any] = field(default_factory=dict)
    api: Dict[str, Any] = field(default_factory=dict)
    hitl: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    database: Dict[str, Any] = field(default_factory=dict)
    checkpoint_dirs: List[str] = field(default_factory=lambda: ["models", "checkpoints"])


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.

    Falls back to defaults if file not found.
    """
    default_paths = [
        Path("configs/app_config.yaml"),
        Path("../configs/app_config.yaml"),
    ]

    config_file = None
    if config_path:
        config_file = Path(config_path)
    else:
        for p in default_paths:
            if p.exists():
                config_file = p
                break

    if config_file and config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config

    # Defaults matching the trained model
    return {
        "model": {
            "input_dim": 772,
            "hidden_dims": [512, 256, 128],
            "num_classes": 20,
            "dropout": 0.3,
            "artifact_name": "ticket_triage_model.pth",
            "version": "1.0.0",
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "cors_origins": ["*"],
            "max_batch_size": 100,
        },
        "hitl": {
            "auto_route_threshold": 0.85,
            "human_review_threshold": 0.70,
        },
        "monitoring": {
            "confidence_drift_threshold": 0.75,
            "escalation_rate_threshold": 0.30,
            "accuracy_drift_threshold": 0.85,
            "processing_time_window": 1000,
        },
        "database": {
            "path": "data/ticket_triage.db",
        },
        "checkpoint_dirs": ["models", "checkpoints"],
    }
