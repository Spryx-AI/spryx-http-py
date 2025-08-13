"""
Comprehensive unit tests for the _process_response_data method.
"""

from unittest.mock import Mock

import httpx
import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from spryx_http.auth_strategies import ClientCredentialsAuthStrategy
from spryx_http.base import SpryxClientBase
from spryx_http.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    RateLimitError,
    ServerError,
)


class ResponseTestModel(BaseModel):
    """Test model for validation."""

    id: int
    name: str
    email: str

class ResponseTestModelAdmin(BaseModel):
    """Test model for validation."""

    id: int
    name: str
    email: str
    admin: bool

class TestProcessResponseData:
    """Test suite for _process_response_data method usability and functionality."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test_client",
            client_secret="test_secret",
            token_url="https://auth.test.com/token"
        )
        return SpryxClientBase(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

    @pytest.fixture
    def mock_response(self):
        """Create a mock httpx.Response."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.headers = {"content-type": "application/json"}
        return response

    def test_process_json_response_without_cast_to(self, client, mock_response):
        """Test processing JSON response without model casting."""
        test_data = {"id": 1, "name": "Test", "email": "test@example.com"}
        mock_response.json.return_value = test_data

        result = client._process_response_data(mock_response, cast_to=None)

        assert result == test_data
        mock_response.json.assert_called_once()

    def test_process_json_response_with_cast_to_model(self, client, mock_response):
        """Test processing JSON response with model casting."""
        test_data = {"id": 1, "name": "Test", "email": "test@example.com"}
        mock_response.json.return_value = test_data

        result = client._process_response_data(mock_response, cast_to=ResponseTestModel)

        assert isinstance(result, ResponseTestModel)
        assert result.id == 1
        assert result.name == "Test"
        assert result.email == "test@example.com"

    def test_process_json_response_with_data_wrapper_with_cast_to(self, client, mock_response):
        """Test processing JSON response with standardized 'data' wrapper."""
        test_data = {"id": 1, "name": "Test", "email": "test@example.com"}
        response_data = {"data": test_data, "status": "success"}
        mock_response.json.return_value = response_data

        # Now returns the model response extracting 'data'
        result = client._process_response_data(mock_response, cast_to=ResponseTestModel)

        assert isinstance(result, ResponseTestModel)
        assert result.id == 1
        assert result.email == "test@example.com"


    def test_process_json_response_with_data_wrapper_without_cast_to(self, client, mock_response):
        """Test processing JSON response with standardized 'data' wrapper."""
        test_data = {"id": 1, "name": "Test", "email": "test@example.com"}
        response_data = {"data": test_data, "status": "success"}
        mock_response.json.return_value = response_data

        # Now returns the raw response without extracting 'data'
        result = client._process_response_data(mock_response, cast_to=None)

        assert result == response_data
        assert result['data']['id'] == 1
        assert result['data']['name'] == 'Test'

    def test_process_json_response_with_cast_to_type_adapter(self, client, mock_response):
        """Test processing JSON response with list model casting."""
        test_data = [
            {"id": 1, "name": "Test1", "email": "test1@example.com"},
            {"id": 2, "name": "Test2", "email": "test2@example.com", "admin": True},
        ]
        mock_response.json.return_value = test_data

        result = client._process_response_data(mock_response, cast_to=TypeAdapter(list[ResponseTestModel | ResponseTestModelAdmin]))

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], ResponseTestModel) and (result[1], ResponseTestModelAdmin)
        assert result[0].name == "Test1"
        assert result[1].admin == True

    def test_process_json_response_with_list_cast_to(self, client, mock_response):
        """Test processing JSON response with list model casting."""
        test_data = [
            {"id": 1, "name": "Test1", "email": "test1@example.com"},
            {"id": 2, "name": "Test2", "email": "test2@example.com"},
        ]
        mock_response.json.return_value = test_data

        result = client._process_response_data(mock_response, cast_to=list[ResponseTestModel])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, ResponseTestModel) for item in result)
        assert result[0].id == 1
        assert result[1].id == 2

    def test_process_json_response_with_list_data_wrapper(self, client, mock_response):
        """Test processing JSON response with list in 'data' wrapper."""
        test_data = [
            {"id": 1, "name": "Test1", "email": "test1@example.com"},
            {"id": 2, "name": "Test2", "email": "test2@example.com"},
        ]
        response_data = {"data": test_data, "status": "success"}
        mock_response.json.return_value = response_data

        # Now returns the raw response without extracting 'data'
        result = client._process_response_data(mock_response, cast_to=None)

        assert result == response_data
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 2

    def test_process_response_with_non_json_content_type(self, client):
        """Test processing response with non-JSON content type."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.headers = {"content-type": "text/plain"}

        result = client._process_response_data(response, cast_to=None)

        assert result is None

    def test_process_response_with_no_content_type(self, client):
        """Test processing response with no content type header."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.headers = None

        result = client._process_response_data(response, cast_to=None)

        assert result is None

    def test_process_response_with_invalid_json(self, client, mock_response):
        """Test processing response with invalid JSON raises ServerError."""
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ServerError):
            client._process_response_data(mock_response, cast_to=None)

    def test_process_response_raises_authentication_error_401(self, client, mock_response):
        """Test that 401 status code raises AuthenticationError."""
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Unauthorized"}

        with pytest.raises(AuthenticationError):
            client._process_response_data(mock_response, cast_to=None)

    def test_process_response_raises_authorization_error_403(self, client, mock_response):
        """Test that 403 status code raises AuthorizationError."""
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": "Forbidden"}

        with pytest.raises(AuthorizationError):
            client._process_response_data(mock_response, cast_to=None)

    def test_process_response_raises_not_found_error_404(self, client, mock_response):
        """Test that 404 status code raises NotFoundError."""
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Not Found"}

        with pytest.raises(NotFoundError):
            client._process_response_data(mock_response, cast_to=None)

    def test_process_response_raises_conflict_error_409(self, client, mock_response):
        """Test that 409 status code raises ConflictError."""
        mock_response.status_code = 409
        mock_response.json.return_value = {"error": "Conflict"}

        with pytest.raises(ConflictError):
            client._process_response_data(mock_response, cast_to=None)

    def test_process_response_raises_rate_limit_error_429(self, client, mock_response):
        """Test that 429 status code raises RateLimitError."""
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "Too Many Requests"}

        with pytest.raises(RateLimitError):
            client._process_response_data(mock_response, cast_to=None)

    def test_process_response_raises_bad_request_error_4xx(self, client, mock_response):
        """Test that other 4xx status codes raise BadRequestError."""
        mock_response.status_code = 422
        mock_response.json.return_value = {"error": "Unprocessable Entity"}

        with pytest.raises(BadRequestError):
            client._process_response_data(mock_response, cast_to=None)

    def test_process_response_raises_server_error_5xx(self, client, mock_response):
        """Test that 5xx status codes raise ServerError."""
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal Server Error"}

        with pytest.raises(ServerError):
            client._process_response_data(mock_response, cast_to=None)

    def test_process_response_with_invalid_model_data(self, client, mock_response):
        """Test processing response with data that doesn't match model schema."""
        invalid_data = {"invalid_field": "value"}
        mock_response.json.return_value = invalid_data

        with pytest.raises(ValidationError):  # Specific Pydantic validation error
            client._process_response_data(mock_response, cast_to=ResponseTestModel)

    def test_process_response_preserves_response_json_structure(self, client, mock_response):
        """Test that the method preserves the original response JSON structure when not casting."""
        complex_data = {
            "data": {"id": 1, "name": "Test"},
            "metadata": {"total": 1, "page": 1},
            "links": {"self": "/api/test"},
        }
        mock_response.json.return_value = complex_data

        result = client._process_response_data(mock_response, cast_to=None)

        assert result == complex_data

    def test_process_response_handles_empty_response(self, client, mock_response):
        """Test processing empty JSON response."""
        mock_response.json.return_value = {}

        result = client._process_response_data(mock_response, cast_to=None)

        assert result == {}

    def test_process_response_error_with_none_response_json(self, client):
        """Test error handling when response_json is None."""
        response = Mock(spec=httpx.Response)
        response.status_code = 500
        response.headers = {"content-type": "text/plain"}

        with pytest.raises(ServerError) as exc_info:
            client._process_response_data(response, cast_to=None)

        assert exc_info.value.response_json is None

    def test_usability_real_world_scenario(self, client, mock_response):
        """Test a real-world scenario demonstrating the method's usability."""
        # Simulate a typical API response for user data
        api_response = {
            "data": {"id": 123, "name": "John Doe", "email": "john@example.com"},
            "status": "success",
            "timestamp": "2023-01-01T00:00:00Z",
        }
        mock_response.json.return_value = api_response

        # Test without model casting (returns raw JSON)
        raw_result = client._process_response_data(mock_response, cast_to=None)
        assert raw_result == api_response

        # Reset mock for second call
        mock_response.json.return_value = api_response

        # Test with model casting - now user needs to extract data manually
        # This demonstrates the new behavior where data extraction is explicit
        user_data = api_response["data"]
        model_result = ResponseTestModel.model_validate(user_data)
        assert isinstance(model_result, ResponseTestModel)
        assert model_result.id == 123
        assert model_result.name == "John Doe"
        assert model_result.email == "john@example.com"


