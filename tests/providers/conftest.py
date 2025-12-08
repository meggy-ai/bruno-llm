"""Pytest configuration for provider tests."""

import pytest


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx response."""
    from unittest.mock import Mock
    
    def _create(status_code=200, json_data=None):
        response = Mock()
        response.status_code = status_code
        if json_data:
            response.json.return_value = json_data
        return response
    
    return _create
