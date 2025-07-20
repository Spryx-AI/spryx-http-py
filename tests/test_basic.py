"""
Basic tests for spryx_http package.
"""

import os

import pytest

from spryx_http import SpryxAsyncClient, SpryxSyncClient


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
        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token",
            scope="read write",
        )

        assert client._base_url == "https://api.example.com"
        assert client._client_id == "test_client_id"
        assert client._client_secret == "test_client_secret"
        assert client._token_url == "https://auth.example.com/token"
        assert client._scope == "read write"

    @pytest.mark.asyncio
    async def test_async_token_expiration_check(self):
        """Test token expiration validation."""
        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token",
        )

        # Initially token should be expired (None)
        assert client.is_token_expired is True

        # Set a token with future expiration
        import time

        client._access_token = "test_token"
        client._token_expires_at = int(time.time()) + 3600  # 1 hour from now

        assert client.is_token_expired is False


class TestSpryxSyncClient:
    def test_sync_client_initialization(self):
        """Test that SpryxSyncClient can be initialized properly."""
        client = SpryxSyncClient(
            base_url="https://api.example.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token",
        )

        assert client._base_url == "https://api.example.com"
        assert client._client_id == "test_client_id"
        assert client._client_secret == "test_client_secret"
        assert client._token_url == "https://auth.example.com/token"

    def test_sync_token_expiration_check(self):
        """Test token expiration validation."""
        client = SpryxSyncClient(
            base_url="https://api.example.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token",
        )

        # Initially token should be expired (None)
        assert client.is_token_expired is True

        # Set a token with future expiration
        import time

        client._access_token = "test_token"
        client._token_expires_at = int(time.time()) + 3600  # 1 hour from now

        assert client.is_token_expired is False

    def test_sync_client_inheritance(self):
        """Test that SpryxSyncClient has all expected methods."""
        client = SpryxSyncClient(
            base_url="https://api.example.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token",
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

        async_client = SpryxAsyncClient(
            base_url="https://api.example.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token",
        )

        sync_client = SpryxSyncClient(
            base_url="https://api.example.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_url="https://auth.example.com/token",
        )

        # Both should inherit from SpryxClientBase
        assert isinstance(async_client, SpryxClientBase)
        assert isinstance(sync_client, SpryxClientBase)

        # Both should have same properties
        assert hasattr(async_client, "is_token_expired")
        assert hasattr(sync_client, "is_token_expired")

        # Both should have same data processing methods
        assert hasattr(async_client, "_extract_data_from_response")
        assert hasattr(sync_client, "_extract_data_from_response")
        assert hasattr(async_client, "_parse_model_data")
        assert hasattr(sync_client, "_parse_model_data")
