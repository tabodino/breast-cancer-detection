"""
Pytest configuration for the Breast Cancer Detection project.

This file automatically sets up the PYTHONPATH so that imports
from the src directory work correctly in the tests.
"""

import pytest
import sys
from pathlib import Path
from loguru import logger


@pytest.fixture(autouse=True, scope="session")
def setup_loguru():
    # Remove existing handlers (no double logs)
    logger.remove()
    # Add stderr handler (for pytest output), compatible with caplog/capture
    # Level DEBUG to capture all logs during tests
    logger.add(sys.stderr, level="DEBUG")


# Add project root directory on PYTHONPATH
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
