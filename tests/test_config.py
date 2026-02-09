"""
Unit tests for configuration management.
"""

import pytest
from src.config.settings import load_config


class TestConfig:
    """Tests for YAML config loading and defaults."""

    def test_load_config_from_file(self):
        """Should load config from YAML file."""
        config = load_config("configs/app_config.yaml")
        assert "model" in config
        assert "hitl" in config
        assert "api" in config

    def test_load_config_defaults(self):
        """Should fall back to defaults if file not found."""
        config = load_config("/nonexistent/path.yaml")
        assert config["model"]["input_dim"] == 772
        assert config["model"]["hidden_dims"] == [512, 256, 128]

    def test_hitl_thresholds(self):
        """HITL thresholds should be within valid range."""
        config = load_config()
        auto = config["hitl"]["auto_route_threshold"]
        review = config["hitl"]["human_review_threshold"]

        assert 0 < review < auto <= 1.0
        assert auto == 0.85
        assert review == 0.70

    def test_model_config_matches_trained(self):
        """Config should match the trained model architecture."""
        config = load_config()
        assert config["model"]["input_dim"] == 772  # BERT 768 + temporal 4
        assert config["model"]["dropout"] == 0.3
        assert len(config["model"]["hidden_dims"]) == 3

    def test_checkpoint_dirs_present(self):
        """Config should specify checkpoint search paths."""
        config = load_config()
        assert "checkpoint_dirs" in config
        assert len(config["checkpoint_dirs"]) > 0
