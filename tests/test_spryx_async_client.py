"""
Comprehensive unit tests for SpryxAsyncClient.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from spryx_http.async_client import SpryxAsyncClient
from spryx_http.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    RateLimitError,
    ServerError,
)
from spryx_http.settings import HttpClientSettings
from spryx_http.types import OAuthTokenResponse


class UserModel(BaseModel):
    """Model for user data in tests."""

    id: int
    name: str
    email: str


class ProductModel(BaseModel):
    """Model for product data in tests."""

    id: int
    title: str
    price: float


@pytest.fixture
def mock_oauth_response():
    """Mock OAuth token response."""
    return OAuthTokenResponse(
        access_token="test_access_token",
        token_type="Bearer",
        expires_in=3600,
        refresh_token="test_refresh_token",
        scope="read write",
    )


@pytest.fixture
def client_config():
    """Standard client configuration."""
    return {
        "base_url": "https://api.test.com",
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "token_url": "https://auth.test.com/token",
    }


class TestSpryxAsyncClientInitialization:
    """Test client initialization scenarios."""

    def test_init_with_all_parameters(self, client_config):
        """Test initialization with all required parameters."""
        settings = HttpClientSettings()
        client = SpryxAsyncClient(
            base_url=client_config["base_url"],
            client_id=client_config["client_id"],
            client_secret=client_config["client_secret"],
            token_url=client_config["token_url"],
            settings=settings,
        )

        assert client._base_url == client_config["base_url"]
        assert client._client_id == client_config["client_id"]
        assert client._client_secret == client_config["client_secret"]
        assert client._token_url == client_config["token_url"]
        assert client.settings == settings

    def test_init_without_base_url(self, client_config):
        """Test initialization without base_url (for full URL requests)."""
        client = SpryxAsyncClient(
            base_url=None,
            client_id=client_config["client_id"],
            client_secret=client_config["client_secret"],
            token_url=client_config["token_url"],
        )

        assert client._base_url is None
        assert client._client_id == client_config["client_id"]

    def test_init_with_custom_settings(self, client_config):
        """Test initialization with custom HTTP settings."""
        custom_settings = HttpClientSettings(timeout_s=30.0, retries=5)
        client = SpryxAsyncClient(**client_config, settings=custom_settings)

        assert client.settings.timeout_s == 30.0
        assert client.settings.retries == 5


class TestSpryxAsyncClientAuthentication:
    """Test OAuth 2.0 authentication flows."""

    @pytest.mark.asyncio
    async def test_authenticate_client_credentials_success(self, client_config, mock_oauth_response):
        """Test successful client credentials authentication."""
        client = SpryxAsyncClient(**client_config)

        with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_oauth_response

            token = await client.authenticate_client_credentials()

            assert token == "test_access_token"
            assert client._access_token == "test_access_token"
            assert client._refresh_token == "test_refresh_token"
            assert client._token_expires_at is not None

            mock_post.assert_called_once_with(
                client_config["token_url"],
                json={
                    "grant_type": "client_credentials",
                    "client_id": client_config["client_id"],
                    "client_secret": client_config["client_secret"],
                },
                cast_to=OAuthTokenResponse,
            )

    @pytest.mark.asyncio
    async def test_authenticate_client_credentials_missing_client_id(self, client_config):
        """Test authentication fails with missing client_id."""
        config = client_config.copy()
        config["client_id"] = None
        client = SpryxAsyncClient(**config)

        with pytest.raises(ValueError, match="client_id is required"):
            await client.authenticate_client_credentials()

    @pytest.mark.asyncio
    async def test_authenticate_client_credentials_missing_client_secret(self, client_config):
        """Test authentication fails with missing client_secret."""
        config = client_config.copy()
        config["client_secret"] = None
        client = SpryxAsyncClient(**config)

        with pytest.raises(ValueError, match="client_secret is required"):
            await client.authenticate_client_credentials()

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self, client_config, mock_oauth_response):
        """Test successful token refresh."""
        client = SpryxAsyncClient(**client_config)
        client._refresh_token = "existing_refresh_token"

        with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_oauth_response

            token = await client.refresh_access_token()

            assert token == "test_access_token"
            mock_post.assert_called_once_with(
                client_config["token_url"],
                json={"grant_type": "refresh_token", "refresh_token": "existing_refresh_token"},
                cast_to=OAuthTokenResponse,
            )

    @pytest.mark.asyncio
    async def test_refresh_access_token_no_refresh_token(self, client_config, mock_oauth_response):
        """Test token refresh falls back to client credentials when no refresh token."""
        client = SpryxAsyncClient(**client_config)
        client._refresh_token = None

        with patch.object(client, "authenticate_client_credentials", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = "new_access_token"

            token = await client.refresh_access_token()

            assert token == "new_access_token"
            mock_auth.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_access_token_fails_fallback_to_client_credentials(self, client_config, mock_oauth_response):
        """Test token refresh falls back to client credentials on failure."""
        client = SpryxAsyncClient(**client_config)
        client._refresh_token = "invalid_refresh_token"

        with (
            patch.object(client, "post", new_callable=AsyncMock) as mock_post,
            patch.object(client, "authenticate_client_credentials", new_callable=AsyncMock) as mock_auth,
        ):
            mock_post.side_effect = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=Mock())
            mock_auth.return_value = "new_access_token"

            token = await client.refresh_access_token()

            assert token == "new_access_token"
            mock_auth.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_token_when_no_token(self, client_config):
        """Test _get_token when no token exists."""
        client = SpryxAsyncClient(**client_config)

        with patch.object(client, "refresh_access_token", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.return_value = "new_token"
            client._access_token = "new_token"  # Simulate token being set

            token = await client._get_token()

            assert token == "new_token"
            mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_token_when_token_expired(self, client_config):
        """Test _get_token when token is expired."""
        client = SpryxAsyncClient(**client_config)
        client._access_token = "expired_token"
        client._token_expires_at = int(time.time()) - 100  # Expired token

        with patch.object(client, "refresh_access_token", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.return_value = "refreshed_token"
            client._access_token = "refreshed_token"  # Simulate token being refreshed

            token = await client._get_token()

            assert token == "refreshed_token"
            mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_token_fails_completely(self, client_config):
        """Test _get_token when authentication completely fails."""
        client = SpryxAsyncClient(**client_config)

        with (
            patch.object(client, "refresh_access_token", new_callable=AsyncMock) as mock_refresh,
            patch.object(client, "authenticate_client_credentials", new_callable=AsyncMock) as mock_auth,
        ):
            mock_refresh.side_effect = Exception("Refresh failed")
            mock_auth.side_effect = Exception("Auth failed")

            with pytest.raises(Exception, match="Failed to obtain a valid authentication token"):
                await client._get_token()


class TestSpryxAsyncClientHTTPMethods:
    """Test HTTP method implementations."""

    @pytest.fixture
    def authenticated_client(self, client_config):
        """Create an authenticated client for testing."""
        client = SpryxAsyncClient(**client_config)
        client._access_token = "valid_token"
        client._token_expires_at = int(time.time()) + 3600  # Valid for 1 hour
        return client

    @pytest.mark.asyncio
    async def test_get_with_model_casting(self, authenticated_client):
        """Test GET request with Pydantic model casting."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"id": 1, "name": "John", "email": "john@test.com"}

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await authenticated_client.get("/users/1", cast_to=UserModel)

            assert isinstance(result, UserModel)
            assert result.id == 1
            assert result.name == "John"
            assert result.email == "john@test.com"

            mock_request.assert_called_once_with(
                "GET", "users/1", headers={"Authorization": "Bearer valid_token"}, params=None, json=None
            )

    @pytest.mark.asyncio
    async def test_get_with_list_casting(self, authenticated_client):
        """Test GET request with list model casting."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = [
            {"id": 1, "name": "John", "email": "john@test.com"},
            {"id": 2, "name": "Jane", "email": "jane@test.com"},
        ]

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await authenticated_client.get("/users", cast_to=list[UserModel])

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(user, UserModel) for user in result)
            assert result[0].name == "John"
            assert result[1].name == "Jane"

    @pytest.mark.asyncio
    async def test_get_without_casting(self, authenticated_client):
        """Test GET request without model casting (returns raw JSON)."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        raw_data = {"id": 1, "name": "John", "custom_field": "value"}
        mock_response.json.return_value = raw_data

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await authenticated_client.get("/users/1")

            assert result == raw_data

    @pytest.mark.asyncio
    async def test_post_with_json_data(self, authenticated_client):
        """Test POST request with JSON data."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"id": 3, "name": "New User", "email": "new@test.com"}

        post_data = {"name": "New User", "email": "new@test.com"}

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await authenticated_client.post("/users", json=post_data, cast_to=UserModel)

            assert isinstance(result, UserModel)
            assert result.id == 3
            assert result.name == "New User"

            mock_request.assert_called_once_with(
                "POST", "users", headers={"Authorization": "Bearer valid_token"}, params=None, json=post_data
            )

    @pytest.mark.asyncio
    async def test_put_request(self, authenticated_client):
        """Test PUT request."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"id": 1, "name": "Updated User", "email": "updated@test.com"}

        update_data = {"name": "Updated User", "email": "updated@test.com"}

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await authenticated_client.put("/users/1", json=update_data, cast_to=UserModel)

            assert isinstance(result, UserModel)
            assert result.name == "Updated User"

    @pytest.mark.asyncio
    async def test_patch_request(self, authenticated_client):
        """Test PATCH request."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"id": 1, "name": "Patched User", "email": "john@test.com"}

        patch_data = {"name": "Patched User"}

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await authenticated_client.patch("/users/1", json=patch_data, cast_to=UserModel)

            assert isinstance(result, UserModel)
            assert result.name == "Patched User"

    @pytest.mark.asyncio
    async def test_delete_request(self, authenticated_client):
        """Test DELETE request."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"message": "User deleted successfully"}

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await authenticated_client.delete("/users/1")

            assert result == {"message": "User deleted successfully"}

    @pytest.mark.asyncio
    async def test_request_with_query_parameters(self, authenticated_client):
        """Test request with query parameters."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = [{"id": 1, "name": "John", "email": "john@test.com"}]

        params = {"page": 1, "limit": 10, "search": "john"}

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            await authenticated_client.get("/users", params=params)

            mock_request.assert_called_once_with(
                "GET", "users", headers={"Authorization": "Bearer valid_token"}, params=params, json=None
            )


class TestSpryxAsyncClientErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def authenticated_client(self, client_config):
        """Create an authenticated client for testing."""
        client = SpryxAsyncClient(**client_config)
        client._access_token = "valid_token"
        client._token_expires_at = int(time.time()) + 3600
        return client

    @pytest.mark.asyncio
    async def test_401_error_triggers_token_refresh(self, authenticated_client):
        """Test that 401 error triggers token refresh and retry."""
        # First response: 401 Unauthorized
        first_response = Mock(spec=httpx.Response)
        first_response.status_code = 401

        # Second response: 200 OK (after token refresh)
        second_response = Mock(spec=httpx.Response)
        second_response.status_code = 200
        second_response.headers = {"content-type": "application/json"}
        second_response.json.return_value = {"id": 1, "name": "John", "email": "john@test.com"}

        with (
            patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request,
            patch.object(authenticated_client, "refresh_access_token", new_callable=AsyncMock) as mock_refresh,
        ):
            mock_request.side_effect = [first_response, second_response]
            mock_refresh.return_value = "new_token"
            authenticated_client._access_token = "new_token"  # Simulate token refresh

            result = await authenticated_client.get("/users/1", cast_to=UserModel)

            assert isinstance(result, UserModel)
            assert mock_request.call_count == 2
            mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_status_errors_are_handled(self, authenticated_client):
        """Test that HTTP status errors are properly converted to custom exceptions."""
        test_cases = [
            (400, BadRequestError),
            (401, AuthenticationError),
            (403, AuthorizationError),
            (404, NotFoundError),
            (409, ConflictError),
            (429, RateLimitError),
            (500, ServerError),
        ]

        for status_code, expected_exception in test_cases:
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = status_code
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"error": f"HTTP {status_code} error"}

            with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response

                with pytest.raises(expected_exception):
                    await authenticated_client.get("/test")

    @pytest.mark.asyncio
    async def test_validation_error_on_invalid_response_data(self, authenticated_client):
        """Test ValidationError when response data doesn't match model schema."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"invalid_field": "value"}  # Missing required fields

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(ValidationError):
                await authenticated_client.get("/users/1", cast_to=UserModel)


class TestSpryxAsyncClientEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_request_without_base_url_with_full_url(self, client_config):
        """Test making requests without base_url using full URLs."""
        config = client_config.copy()
        config["base_url"] = None
        client = SpryxAsyncClient(**config)
        client._access_token = "valid_token"
        client._token_expires_at = int(time.time()) + 3600

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"message": "success"}

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.get("https://api.external.com/users")

            assert result == {"message": "success"}
            mock_request.assert_called_once_with(
                "GET",
                "https://api.external.com/users",
                headers={"Authorization": "Bearer valid_token"},
                params=None,
                json=None,
            )

    @pytest.mark.asyncio
    async def test_request_without_base_url_with_relative_path_fails(self, client_config):
        """Test that relative paths fail when no base_url is provided."""
        config = client_config.copy()
        config["base_url"] = None
        client = SpryxAsyncClient(**config)
        client._access_token = "valid_token"
        client._token_expires_at = int(time.time()) + 3600

        with pytest.raises(ValueError, match="Either base_url must be provided"):
            await client.get("/users")

    @pytest.mark.asyncio
    async def test_unsupported_protocol_error_handling(self, authenticated_client):
        """Test handling of UnsupportedProtocol errors."""
        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = httpx.UnsupportedProtocol("Invalid URL")

            with pytest.raises(ValueError, match="Either base_url must be provided"):
                await authenticated_client.get("/users")

    @pytest.mark.asyncio
    async def test_custom_headers_are_preserved(self, authenticated_client):
        """Test that custom headers are preserved along with auth headers."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"message": "success"}

        custom_headers = {"X-Custom-Header": "custom-value", "Content-Type": "application/json"}

        with patch.object(authenticated_client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            await authenticated_client.get("/users", headers=custom_headers)

            expected_headers = {
                "Authorization": "Bearer valid_token",
                "X-Custom-Header": "custom-value",
                "Content-Type": "application/json",
            }

            mock_request.assert_called_once_with("GET", "users", headers=expected_headers, params=None, json=None)

    @pytest.mark.asyncio
    async def test_token_expiration_check(self, client_config):
        """Test token expiration checking logic."""
        client = SpryxAsyncClient(**client_config)

        # Test with no token
        assert client.is_token_expired is True

        # Test with expired token
        client._access_token = "token"
        client._token_expires_at = int(time.time()) - 100  # Expired
        assert client.is_token_expired is True

        # Test with valid token (with buffer)
        client._token_expires_at = int(time.time()) + 100  # Valid with buffer
        assert client.is_token_expired is False
