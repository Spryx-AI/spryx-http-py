"""Test SpryxException handling in raise_for_status."""

import json

import pytest

from spryx_http.exceptions import (
    ClientError,
    ServerError,
    SpryxException,
    raise_for_status,
)


class MockResponse:
    """Mock httpx.Response for testing."""

    def __init__(self, status_code: int, json_data: dict | None = None, text: str | None = None):
        self.status_code = status_code
        self._json_data = json_data
        self._text = text or json.dumps(json_data) if json_data else ""

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON data")
        return self._json_data

    @property
    def text(self):
        return self._text


def test_raise_for_status_with_spryx_error():
    """Test that SpryxException is raised when response contains Spryx error format."""
    response = MockResponse(
        status_code=400,
        json_data={
            "error": {
                "message": "Validation failed",
                "code": "VALIDATION_ERROR",
                "status_code": 400,
                "details": {"field": "email", "reason": "invalid format"},
            }
        },
    )

    with pytest.raises(SpryxException) as exc_info:
        raise_for_status(response)

    assert exc_info.value.message == "Validation failed"
    assert exc_info.value.code == "VALIDATION_ERROR"
    assert exc_info.value.status_code == 400
    assert exc_info.value.details == {"field": "email", "reason": "invalid format"}


def test_raise_for_status_with_spryx_error_no_details():
    """Test SpryxException is raised even when details are missing."""
    response = MockResponse(
        status_code=500,
        json_data={
            "error": {
                "message": "Internal server error",
                "code": "INTERNAL_ERROR",
                "status_code": 500,
            }
        },
    )

    with pytest.raises(SpryxException) as exc_info:
        raise_for_status(response)

    assert exc_info.value.message == "Internal server error"
    assert exc_info.value.code == "INTERNAL_ERROR"
    assert exc_info.value.status_code == 500
    assert exc_info.value.details == {}


def test_raise_for_status_with_non_spryx_error():
    """Test that default HttpError is raised for non-Spryx error formats."""
    response = MockResponse(
        status_code=404,
        json_data={"detail": "Resource not found"},
    )

    with pytest.raises(ClientError) as exc_info:
        raise_for_status(response)

    assert "HTTP Error 404: Resource not found" in str(exc_info.value)
    assert exc_info.value.status_code == 404


def test_raise_for_status_with_invalid_json():
    """Test that default error handling works when JSON parsing fails."""
    response = MockResponse(
        status_code=500,
        text="Internal Server Error",
    )
    response._json_data = None  # Force json() to raise ValueError

    with pytest.raises(ServerError) as exc_info:
        raise_for_status(response)

    assert "HTTP Error 500" in str(exc_info.value)
    assert exc_info.value.status_code == 500


def test_raise_for_status_with_incomplete_spryx_error():
    """Test that default error is raised when Spryx error format is incomplete."""
    response = MockResponse(
        status_code=400,
        json_data={
            "error": {
                "message": "Bad request",
                # Missing 'code' and 'status_code'
            }
        },
    )

    with pytest.raises(ClientError) as exc_info:
        raise_for_status(response)

    assert "HTTP Error 400" in str(exc_info.value)


def test_raise_for_status_success():
    """Test that no exception is raised for successful responses."""
    response = MockResponse(
        status_code=200,
        json_data={"data": {"id": 1, "name": "Test"}},
    )

    # Should not raise any exception
    raise_for_status(response)


def test_raise_for_status_redirect():
    """Test that no exception is raised for redirect responses."""
    response = MockResponse(
        status_code=302,
        json_data={},
    )

    # Should not raise any exception
    raise_for_status(response)
