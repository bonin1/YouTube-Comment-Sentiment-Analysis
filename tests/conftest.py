"""
Test configuration for pytest

This file contains pytest configuration and fixtures for the test suite.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Configure pytest
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark network tests
        if "network" in item.name.lower() or "scraper" in item.name.lower():
            item.add_marker(pytest.mark.network)
        
        # Mark integration tests
        if "integration" in item.name.lower() or "full_pipeline" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "performance" in item.name.lower() or "batch" in item.name.lower():
            item.add_marker(pytest.mark.slow)
