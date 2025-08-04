"""
Basic tests for spryx_http package.
"""

import os

import pytest

from spryx_http import ClientCredentialsAuthStrategy, SpryxAsyncClient, SpryxSyncClient


def test_package_structure():
    """Test that the package files exist."""
    assert os.path.isdir("spryx_http")
    assert os.path.isfile("spryx_http/__init__.py")


def test_package_modules():
    """Test that the core modules exist."""
    module_files = [
        "base.py",
        "exceptions.py",
        "retry.py",
        "settings.py",
    ]
    for module in module_files:
        assert os.path.isfile(f"spryx_http/{module}"), f"Module file {module} should exist"


class TestSpryxAsyncClient:
    @pytest.mark.asyncio
    async def test_async_client_initialization(self):
        """Test that SpryxAsyncClient can be initialized properly."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token"
        )

        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        assert client._base_url == "https://api.example.com"
        assert client._auth_strategy == auth_strategy
        assert client._auth_strategy.client_id == "test_client_id"
        assert client._auth_strategy.client_secret == "test_client_secret"
        assert client._auth_strategy.token_url == "https://auth.example.com/token"

    @pytest.mark.asyncio
    async def test_async_token_expiration_check(self):
        """Test token expiration validation."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token"
        )

        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Initially token should need refresh (None)
        assert client._auth_strategy.needs_refresh() is True

        # Set a token with future expiration
        import time

        client._auth_strategy._access_token = "test_token"
        client._auth_strategy._token_expires_at = int(time.time()) + 3600  # 1 hour from now

        assert client._auth_strategy.needs_refresh() is False


class TestSpryxSyncClient:
    def test_sync_client_initialization(self):
        """Test that SpryxSyncClient can be initialized properly."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token"
        )

        client = SpryxSyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        assert client._base_url == "https://api.example.com"
        assert client._auth_strategy == auth_strategy
        assert client._auth_strategy.client_id == "test_client_id"
        assert client._auth_strategy.client_secret == "test_client_secret"
        assert client._auth_strategy.token_url == "https://auth.example.com/token"

    def test_sync_token_expiration_check(self):
        """Test token expiration validation."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token"
        )

        client = SpryxSyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Initially token should need refresh (None)
        assert client._auth_strategy.needs_refresh() is True

        # Set a token with future expiration
        import time

        client._auth_strategy._access_token = "test_token"
        client._auth_strategy._token_expires_at = int(time.time()) + 3600  # 1 hour from now

        assert client._auth_strategy.needs_refresh() is False

    def test_sync_client_inheritance(self):
        """Test that SpryxSyncClient has all expected methods."""
        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token"
        )

        client = SpryxSyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Test that sync client has all HTTP methods
        assert hasattr(client, "get")
        assert hasattr(client, "post")
        assert hasattr(client, "put")
        assert hasattr(client, "patch")
        assert hasattr(client, "delete")

        # Test that methods are not async
        import inspect

        assert not inspect.iscoroutinefunction(client.get)
        assert not inspect.iscoroutinefunction(client.post)
        assert not inspect.iscoroutinefunction(client.put)
        assert not inspect.iscoroutinefunction(client.patch)
        assert not inspect.iscoroutinefunction(client.delete)


class TestSharedFunctionality:
    """Test shared functionality between async and sync clients."""

    def test_both_clients_share_base_functionality(self):
        """Test that both clients inherit from the same base."""
        from spryx_http.base import SpryxClientBase

        auth_strategy = ClientCredentialsAuthStrategy(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token"
        )

        async_client = SpryxAsyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        sync_client = SpryxSyncClient(
            base_url="https://api.example.com",
            auth_strategy=auth_strategy
        )

        # Both should inherit from SpryxClientBase
        assert isinstance(async_client, SpryxClientBase)
        assert isinstance(sync_client, SpryxClientBase)

        # Both should have same properties through auth strategy
        assert hasattr(async_client._auth_strategy, "needs_refresh")
        assert hasattr(sync_client._auth_strategy, "needs_refresh")

        # Both should have same data processing methods
        assert hasattr(async_client, "_extract_data_from_response")
        assert hasattr(sync_client, "_extract_data_from_response")
        assert hasattr(async_client, "_parse_model_data")
        assert hasattr(sync_client, "_parse_model_data")
