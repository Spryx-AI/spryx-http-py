"""Integration tests for SpryxAsyncClient and SpryxSyncClient with auth strategies."""

from unittest.mock import Mock, patch

import pytest

from spryx_http import SpryxAsyncClient, SpryxSyncClient
from spryx_http.auth_strategies import ApiKeyAuthStrategy, ClientCredentialsAuthStrategy


class TestSpryxAsyncClientIntegration:
    """Test SpryxAsyncClient integration with auth strategies."""

    @pytest.mark.asyncio
    async def test_client_credentials_flow(self):
        """Test complete flow with ClientCredentials strategy."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test-id",
            client_secret="test-secret",
            token_url="https://auth.example.com/token"
        )

        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Mock the httpx request method
        with patch.object(client, 'request') as mock_request:
            # Mock auth response
            auth_response = Mock()
            auth_response.raise_for_status.return_value = None
            auth_response.json.return_value = {
                "access_token": "test-access-token",
                "refresh_token": "test-refresh-token",
                "expires_in": 3600
            }

            # Mock API response
            api_response = Mock()
            api_response.status_code = 200
            api_response.headers = {"content-type": "application/json"}
            api_response.json.return_value = {"data": [{"id": 1, "name": "test"}]}

            # Configure mock to return auth response first, then API response
            mock_request.side_effect = [auth_response, api_response]

            # Make API call
            result = await client.get("/users")

            # Verify authentication call was made
            mock_request.assert_any_call(
                "POST",
                "https://auth.example.com/token",
                json={
                    "grant_type": "client_credentials",
                    "client_id": "test-id",
                    "client_secret": "test-secret"
                }
            )

            # Verify API call was made with auth headers
            mock_request.assert_any_call(
                "GET",
                "users",
                headers={"Authorization": "Bearer test-access-token"},
                params=None,
                json=None
            )

            # Verify result
            assert result == {"data": [{"id": 1, "name": "test"}]}

    @pytest.mark.asyncio
    async def test_api_key_flow(self):
        """Test complete flow with ApiKey strategy."""
        auth_strategy = ApiKeyAuthStrategy(
            api_key="test-api-key"
        )

        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Mock the httpx request method
        with patch.object(client, 'request') as mock_request:
            # Mock API response only (no auth needed)
            api_response = Mock()
            api_response.status_code = 200
            api_response.headers = {"content-type": "application/json"}
            api_response.json.return_value = {"data": [{"id": 1, "name": "test"}]}

            mock_request.return_value = api_response

            # Make API call
            result = await client.get("/users")

            # Verify only one call was made (API call, no auth call)
            assert mock_request.call_count == 1

            # Verify API call was made with API key as Bearer token
            mock_request.assert_called_with(
                "GET",
                "users",
                headers={"Authorization": "Bearer test-api-key"},
                params=None,
                json=None
            )

            # Verify result
            assert result == {"data": [{"id": 1, "name": "test"}]}

    @pytest.mark.asyncio
    async def test_token_refresh_on_401(self):
        """Test token refresh when receiving 401 response."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test-id",
            client_secret="test-secret",
            token_url="https://auth.example.com/token"
        )

        # Pre-authenticate to set initial token
        auth_strategy._access_token = "expired-token"
        auth_strategy._refresh_token = "test-refresh-token"

        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Mock the httpx request method
        with patch.object(client, 'request') as mock_request:
            # Mock 401 response first
            unauthorized_response = Mock()
            unauthorized_response.status_code = 401

            # Mock refresh response
            refresh_response = Mock()
            refresh_response.raise_for_status.return_value = None
            refresh_response.json.return_value = {
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600
            }

            # Mock successful API response after refresh
            api_response = Mock()
            api_response.status_code = 200
            api_response.headers = {"content-type": "application/json"}
            api_response.json.return_value = {"data": [{"id": 1, "name": "test"}]}

            # Configure mock responses
            mock_request.side_effect = [
                unauthorized_response,  # First API call returns 401
                refresh_response,       # Refresh token call
                api_response           # Retry API call succeeds
            ]

            # Make API call
            result = await client.get("/users")

            # Verify refresh call was made
            mock_request.assert_any_call(
                "POST",
                "https://auth.example.com/token",
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": "test-refresh-token"
                }
            )

            # Verify result
            assert result == {"data": [{"id": 1, "name": "test"}]}

    @pytest.mark.asyncio
    async def test_api_key_no_refresh_needed(self):
        """Test that API Key strategy doesn't attempt refresh."""
        auth_strategy = ApiKeyAuthStrategy(
            api_key="test-api-key"
        )

        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Mock the httpx request method
        with patch.object(client, 'request') as mock_request:
            # Mock API response
            api_response = Mock()
            api_response.status_code = 200
            api_response.headers = {"content-type": "application/json"}
            api_response.json.return_value = {"data": [{"id": 1, "name": "test"}]}

            mock_request.return_value = api_response

            # Make API call
            result = await client.get("/users")

            # Verify only one call was made (no auth call)
            assert mock_request.call_count == 1

            # Verify API call was made with API key as Bearer token
            mock_request.assert_called_with(
                "GET",
                "users",
                headers={"Authorization": "Bearer test-api-key"},
                params=None,
                json=None
            )

            # Verify result
            assert result == {"data": [{"id": 1, "name": "test"}]}


class TestSpryxSyncClientIntegration:
    """Test SpryxSyncClient integration with auth strategies."""

    def test_client_credentials_flow(self):
        """Test complete flow with ClientCredentials strategy."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test-id",
            client_secret="test-secret",
            token_url="https://auth.example.com/token"
        )

        client = SpryxSyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Mock the httpx request method
        with patch.object(client, 'request') as mock_request:
            # Mock auth response
            auth_response = Mock()
            auth_response.raise_for_status.return_value = None
            auth_response.json.return_value = {
                "access_token": "test-access-token",
                "refresh_token": "test-refresh-token",
                "expires_in": 3600
            }

            # Mock API response
            api_response = Mock()
            api_response.status_code = 200
            api_response.headers = {"content-type": "application/json"}
            api_response.json.return_value = {"data": [{"id": 1, "name": "test"}]}

            # Configure mock to return auth response first, then API response
            mock_request.side_effect = [auth_response, api_response]

            # Make API call
            result = client.get("/users")

            # Verify authentication call was made
            mock_request.assert_any_call(
                "POST",
                "https://auth.example.com/token",
                json={
                    "grant_type": "client_credentials",
                    "client_id": "test-id",
                    "client_secret": "test-secret"
                }
            )

            # Verify API call was made with auth headers
            mock_request.assert_any_call(
                "GET",
                "users",
                headers={"Authorization": "Bearer test-access-token"},
                params=None,
                json=None
            )

            # Verify result
            assert result == {"data": [{"id": 1, "name": "test"}]}

    def test_api_key_flow(self):
        """Test complete flow with ApiKey strategy."""
        auth_strategy = ApiKeyAuthStrategy(
            api_key="test-api-key"
        )

        client = SpryxSyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Mock the httpx request method
        with patch.object(client, 'request') as mock_request:
            # Mock API response only (no auth needed)
            api_response = Mock()
            api_response.status_code = 200
            api_response.headers = {"content-type": "application/json"}
            api_response.json.return_value = {"data": [{"id": 1, "name": "test"}]}

            mock_request.return_value = api_response

            # Make API call
            result = client.get("/users")

            # Verify only one call was made (API call, no auth call)
            assert mock_request.call_count == 1

            # Verify API call was made with API key as Bearer token
            mock_request.assert_called_with(
                "GET",
                "users",
                headers={"Authorization": "Bearer test-api-key"},
                params=None,
                json=None
            )

            # Verify result
            assert result == {"data": [{"id": 1, "name": "test"}]}

    def test_token_refresh_on_401(self):
        """Test token refresh when receiving 401 response."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test-id",
            client_secret="test-secret",
            token_url="https://auth.example.com/token"
        )

        # Pre-authenticate to set initial token
        auth_strategy._access_token = "expired-token"
        auth_strategy._refresh_token = "test-refresh-token"

        client = SpryxSyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Mock the httpx request method
        with patch.object(client, 'request') as mock_request:
            # Mock 401 response first
            unauthorized_response = Mock()
            unauthorized_response.status_code = 401

            # Mock refresh response
            refresh_response = Mock()
            refresh_response.raise_for_status.return_value = None
            refresh_response.json.return_value = {
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600
            }

            # Mock successful API response after refresh
            api_response = Mock()
            api_response.status_code = 200
            api_response.headers = {"content-type": "application/json"}
            api_response.json.return_value = {"data": [{"id": 1, "name": "test"}]}

            # Configure mock responses
            mock_request.side_effect = [
                unauthorized_response,  # First API call returns 401
                refresh_response,       # Refresh token call
                api_response           # Retry API call succeeds
            ]

            # Make API call
            result = client.get("/users")

            # Verify refresh call was made
            mock_request.assert_any_call(
                "POST",
                "https://auth.example.com/token",
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": "test-refresh-token"
                }
            )

            # Verify result
            assert result == {"data": [{"id": 1, "name": "test"}]}


class TestAuthStrategyValidation:
    """Test validation and error handling of auth strategies."""

    @pytest.mark.asyncio
    async def test_invalid_auth_strategy_type(self):
        """Test client behavior with invalid auth strategy."""
        # This should be caught by type checking, but test runtime behavior
        with pytest.raises(AttributeError):
            client = SpryxAsyncClient(
                base_url="https://api.example.com",
                auth_strategy="invalid-strategy"  # type: ignore
            )
            await client.get("/test")

    def test_strategy_immutability(self):
        """Test that strategy state is properly isolated between instances."""
        strategy1 = ClientCredentialsAuthStrategy(
            client_id="test-id-1",
            client_secret="test-secret-1",
            token_url="https://auth1.example.com/token"
        )

        strategy2 = ClientCredentialsAuthStrategy(
            client_id="test-id-2",
            client_secret="test-secret-2",
            token_url="https://auth2.example.com/token"
        )

        # Set tokens on first strategy
        strategy1._access_token = "token1"
        strategy1._refresh_token = "refresh1"

        # Second strategy should be unaffected
        assert strategy2._access_token is None
        assert strategy2._refresh_token is None
