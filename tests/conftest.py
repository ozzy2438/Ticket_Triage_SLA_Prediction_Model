"""
Shared test fixtures for the ticket triage test suite.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_config
from src.db.persistence import DatabaseManager
from src.models.multi_task import MultiTaskTicketModel


@pytest.fixture
def config():
    """Load default configuration."""
    return load_config()


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "test.db")
    return DatabaseManager(db_path)


@pytest.fixture
def sample_model():
    """Create an untrained model instance for structure tests."""
    return MultiTaskTicketModel(
        input_dim=772,
        num_classes=20,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
    )
