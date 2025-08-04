"""Tests for authentication strategies."""

import time
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from spryx_http.auth_strategies import ApiKeyAuthStrategy, ClientCredentialsAuthStrategy


class TestClientCredentialsAuthStrategy:
    """Test ClientCredentialsAuthStrategy implementation."""

    def test_init(self):
        """Test strategy initialization."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        assert strategy.client_id == "test-id"
        assert strategy.client_secret == "test-secret"
        assert strategy.token_url == "https://auth.example.com/token"
        assert strategy.type == "client_credentials"
        assert strategy._access_token is None
        assert strategy._refresh_token is None
        assert strategy._token_expires_at is None

    def test_supports_refresh(self):
        """Test that ClientCredentials supports refresh."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        assert strategy.supports_refresh() is True

    def test_needs_refresh_no_token(self):
        """Test needs_refresh when no token exists."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        assert strategy.needs_refresh() is True

    def test_needs_refresh_expired_token(self):
        """Test needs_refresh when token is expired."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        # Set expired token
        strategy._access_token = "expired-token"
        strategy._token_expires_at = int(time.time()) - 100  # expired 100 seconds ago

        assert strategy.needs_refresh() is True

    def test_needs_refresh_valid_token(self):
        """Test needs_refresh when token is still valid."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        # Set valid token (expires in 1 hour)
        strategy._access_token = "valid-token"
        strategy._token_expires_at = int(time.time()) + 3600

        assert strategy.needs_refresh() is False

    @pytest.mark.asyncio
    async def test_authenticate_async_success(self):
        """Test successful async authentication."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        # Mock client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
        }
        mock_client.request.return_value = mock_response

        # Call authenticate
        result = await strategy.authenticate_async(mock_client)

        # Verify request was made correctly
        mock_client.request.assert_called_once_with(
            "POST",
            "https://auth.example.com/token",
            json={"grant_type": "client_credentials", "client_id": "test-id", "client_secret": "test-secret"},
        )

        # Verify response
        assert result == {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
        }

    def test_authenticate_sync_success(self):
        """Test successful sync authentication."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        # Mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
        }
        mock_client.request.return_value = mock_response

        # Call authenticate
        result = strategy.authenticate_sync(mock_client)

        # Verify request was made correctly
        mock_client.request.assert_called_once_with(
            "POST",
            "https://auth.example.com/token",
            json={"grant_type": "client_credentials", "client_id": "test-id", "client_secret": "test-secret"},
        )

        # Verify response
        assert result == {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
        }

    @pytest.mark.asyncio
    async def test_refresh_async_success(self):
        """Test successful async token refresh."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        # Set refresh token
        strategy._refresh_token = "test-refresh-token"

        # Mock client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
        }
        mock_client.request.return_value = mock_response

        # Call refresh
        result = await strategy.refresh_async(mock_client)

        # Verify request was made correctly
        mock_client.request.assert_called_once_with(
            "POST",
            "https://auth.example.com/token",
            json={"grant_type": "refresh_token", "refresh_token": "test-refresh-token"},
        )

        # Verify response
        assert result == {"access_token": "new-access-token", "refresh_token": "new-refresh-token", "expires_in": 3600}

    @pytest.mark.asyncio
    async def test_refresh_async_no_refresh_token(self):
        """Test async refresh when no refresh token exists."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        mock_client = AsyncMock()

        # Call refresh without refresh token
        result = await strategy.refresh_async(mock_client)

        # Should return None
        assert result is None
        mock_client.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_async_failure(self):
        """Test async refresh failure."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        # Set refresh token
        strategy._refresh_token = "test-refresh-token"

        # Mock client to raise exception
        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=Mock())

        # Call refresh
        result = await strategy.refresh_async(mock_client)

        # Should return None on failure
        assert result is None

    def test_store_token_response(self):
        """Test storing token response."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        token_response = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
        }

        # Store token response
        strategy.store_token_response(token_response)

        # Verify tokens were stored
        assert strategy._access_token == "test-access-token"
        assert strategy._refresh_token == "test-refresh-token"
        assert strategy._token_expires_at is not None
        assert strategy._token_expires_at > int(time.time())

    def test_get_auth_headers_success(self):
        """Test getting auth headers with valid token."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        strategy._access_token = "test-access-token"

        headers = strategy.get_auth_headers()

        assert headers == {"Authorization": "Bearer test-access-token"}

    def test_get_auth_headers_no_token(self):
        """Test getting auth headers without token."""
        strategy = ClientCredentialsAuthStrategy(
            client_id="test-id", client_secret="test-secret", token_url="https://auth.example.com/token"
        )

        with pytest.raises(ValueError, match="No access token available"):
            strategy.get_auth_headers()


class TestApiKeyAuthStrategy:
    """Test ApiKeyAuthStrategy implementation."""

    def test_init(self):
        """Test strategy initialization."""
        strategy = ApiKeyAuthStrategy(api_key="test-api-key")

        assert strategy.api_key == "test-api-key"
        assert strategy.type == "api_key"

    def test_supports_refresh(self):
        """Test that ApiKey doesn't support refresh."""
        strategy = ApiKeyAuthStrategy(api_key="test-api-key")

        assert strategy.supports_refresh() is False

    def test_needs_refresh(self):
        """Test needs_refresh for API key (always false)."""
        strategy = ApiKeyAuthStrategy(api_key="test-api-key")

        assert strategy.needs_refresh() is False


    @pytest.mark.asyncio
    async def test_authenticate_async_success(self):
        """Test async authentication returns API key directly."""
        strategy = ApiKeyAuthStrategy(api_key="test-api-key")

        # Mock client (should not be used)
        mock_client = AsyncMock()

        # Call authenticate
        result = await strategy.authenticate_async(mock_client)

        # Verify no request was made
        mock_client.request.assert_not_called()

        # Verify response contains the API key as access token
        assert result == {"access_token": "test-api-key"}

    def test_authenticate_sync_success(self):
        """Test sync authentication returns API key directly."""
        strategy = ApiKeyAuthStrategy(api_key="test-api-key")

        # Mock client (should not be used)
        mock_client = Mock()

        # Call authenticate
        result = strategy.authenticate_sync(mock_client)

        # Verify no request was made
        mock_client.request.assert_not_called()

        # Verify response contains the API key as access token
        assert result == {"access_token": "test-api-key"}

    def test_store_token_response(self):
        """Test storing token response (no-op for API key)."""
        strategy = ApiKeyAuthStrategy(api_key="test-api-key")

        token_response = {"access_token": "test-access-token", "expires_in": 3600}

        # Store token response (should be a no-op)
        strategy.store_token_response(token_response)

        # Verify nothing was stored (API key is used directly)
        # No internal state should be modified

    def test_get_auth_headers_success(self):
        """Test getting auth headers uses API key directly."""
        strategy = ApiKeyAuthStrategy(api_key="test-api-key")

        headers = strategy.get_auth_headers()

        assert headers == {"Authorization": "Bearer test-api-key"}

