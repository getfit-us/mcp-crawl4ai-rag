"""Pytest configuration for PostgreSQL tests."""

import pytest
import asyncio
import os
from typing import Generator


def pytest_configure(config):
    """Configure pytest for async tests."""
    config.addinivalue_line("markers", "asyncio: mark test as async")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def pytest_collection_modifyitems(config, items):
    """Automatically mark async test functions."""
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# Ensure test environment variables are set
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("DATABASE_URL", "postgresql://postgres:password@localhost:5432/test_crawl4ai")
    # Set required API keys for testing (avoid validation errors)
    os.environ.setdefault("LLM_MODEL_API_KEY", "test-key")
    os.environ.setdefault("POSTGRES_USER", "postgres")
    os.environ.setdefault("POSTGRES_PASSWORD", "password") 