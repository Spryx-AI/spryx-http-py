"""
Updated tests for SpryxAsyncClient with new auth strategy architecture.
"""

import time
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from spryx_http.async_client import SpryxAsyncClient
from spryx_http.auth_strategies import ClientCredentialsAuthStrategy
from spryx_http.settings import HttpClientSettings


class UserModel(BaseModel):
    """Model for user data in tests."""
    id: int
    name: str
    email: str


@pytest.fixture
def auth_strategy():
    """Standard auth strategy."""
    return ClientCredentialsAuthStrategy(
        client_id="test_client_id",
        client_secret="test_client_secret",
        token_url="https://auth.test.com/token"
    )


class TestSpryxAsyncClientBasic:
    """Basic functionality tests."""

    def test_client_initialization(self, auth_strategy):
        """Test client initialization with auth strategy."""
        client = SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

        assert client._base_url == "https://api.test.com"
        assert client._auth_strategy == auth_strategy

    def test_client_with_settings(self, auth_strategy):
        """Test client initialization with custom settings."""
        settings = HttpClientSettings()
        client = SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy,
            settings=settings
        )

        assert client.settings == settings

    @pytest.mark.asyncio
    async def test_authentication_flow(self, auth_strategy):
        """Test basic authentication flow."""
        client = SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

        # Mock the request method
        with patch.object(client, 'request') as mock_request:
            # Mock auth response
            auth_response = Mock()
            auth_response.raise_for_status.return_value = None
            auth_response.json.return_value = {
                "access_token": "test_token",
                "refresh_token": "test_refresh",
                "expires_in": 3600
            }

            mock_request.return_value = auth_response

            # Call authenticate
            await client.authenticate()

            # Verify auth request was made
            mock_request.assert_called_once_with(
                "POST",
                "https://auth.test.com/token",
                json={
                    "grant_type": "client_credentials",
                    "client_id": "test_client_id",
                    "client_secret": "test_client_secret"
                }
            )

            # Verify token was stored
            assert auth_strategy._access_token == "test_token"
            assert auth_strategy._refresh_token == "test_refresh"

    @pytest.mark.asyncio
    async def test_api_request_flow(self, auth_strategy):
        """Test making an API request."""
        # Pre-authenticate
        auth_strategy._access_token = "test_token"
        auth_strategy._token_expires_at = int(time.time()) + 3600

        client = SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

        with patch.object(client, 'request') as mock_request:
            # Mock API response
            api_response = Mock()
            api_response.status_code = 200
            api_response.headers = {"content-type": "application/json"}
            api_response.json.return_value = {"id": 1, "name": "test", "email": "test@example.com"}

            mock_request.return_value = api_response

            # Make API call
            result = await client.get("/users/1")

            # Verify request was made with auth headers
            mock_request.assert_called_once_with(
                "GET",
                "users/1",
                headers={"Authorization": "Bearer test_token"},
                params=None,
                json=None
            )

            # Verify result
            assert result == {"id": 1, "name": "test", "email": "test@example.com"}

    @pytest.mark.asyncio
    async def test_token_refresh_on_401(self, auth_strategy):
        """Test token refresh when receiving 401."""
        # Pre-authenticate with expired token
        auth_strategy._access_token = "expired_token"
        auth_strategy._refresh_token = "refresh_token"
        auth_strategy._token_expires_at = int(time.time()) - 100

        client = SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

        with patch.object(client, 'request') as mock_request:
            # First call returns 401
            unauthorized_response = Mock()
            unauthorized_response.status_code = 401

            # Refresh call returns new token
            refresh_response = Mock()
            refresh_response.raise_for_status.return_value = None
            refresh_response.json.return_value = {
                "access_token": "new_token",
                "refresh_token": "new_refresh",
                "expires_in": 3600
            }

            # Retry call succeeds
            success_response = Mock()
            success_response.status_code = 200
            success_response.headers = {"content-type": "application/json"}
            success_response.json.return_value = {"success": True}

            mock_request.side_effect = [
                unauthorized_response,  # Initial request fails
                refresh_response,       # Refresh succeeds
                success_response        # Retry succeeds
            ]

            # Make API call
            result = await client.get("/users")

            # Verify refresh was called
            assert mock_request.call_count == 3

            # Verify new token was stored
            assert auth_strategy._access_token == "new_token"
            assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_http_methods(self, auth_strategy):
        """Test all HTTP methods."""
        # Pre-authenticate
        auth_strategy._access_token = "test_token"
        auth_strategy._token_expires_at = int(time.time()) + 3600

        client = SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

        with patch.object(client, 'request') as mock_request:
            # Mock successful response
            success_response = Mock()
            success_response.status_code = 200
            success_response.headers = {"content-type": "application/json"}
            success_response.json.return_value = {"success": True}
            mock_request.return_value = success_response

            # Test all HTTP methods
            await client.get("/test")
            await client.post("/test", json={"data": "test"})
            await client.put("/test", json={"data": "test"})
            await client.patch("/test", json={"data": "test"})
            await client.delete("/test")

            # Verify all calls were made
            assert mock_request.call_count == 5

    @pytest.mark.asyncio
    async def test_model_casting(self, auth_strategy):
        """Test response casting to Pydantic models."""
        # Pre-authenticate
        auth_strategy._access_token = "test_token"
        auth_strategy._token_expires_at = int(time.time()) + 3600

        client = SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

        with patch.object(client, 'request') as mock_request:
            # Mock API response
            api_response = Mock()
            api_response.status_code = 200
            api_response.headers = {"content-type": "application/json"}
            api_response.json.return_value = {"id": 1, "name": "John", "email": "john@example.com"}

            mock_request.return_value = api_response

            # Make API call with model casting
            result = await client.get("/users/1", cast_to=UserModel)

            # Verify result is a UserModel instance
            assert isinstance(result, UserModel)
            assert result.id == 1
            assert result.name == "John"
            assert result.email == "john@example.com"
