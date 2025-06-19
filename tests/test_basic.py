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
        "auth.py",
        "pagination.py",
        "exceptions.py",
        "retry.py",
        "settings.py",
    ]
    for module in module_files:
        assert os.path.isfile(
            f"spryx_http/{module}"
        ), f"Module file {module} should exist"


class TestSpryxAsyncClient:
    @pytest.mark.asyncio
    async def test_async_client_initialization(self):
        """Test that SpryxAsyncClient can be initialized properly."""
        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            application_id="test_app",
            application_secret="test_secret",
            iam_base_url="https://iam.example.com",
        )

        assert client._base_url == "https://api.example.com"
        assert client._application_id == "test_app"
        assert client._application_secret == "test_secret"
        assert client._iam_base_url == "https://iam.example.com"

    @pytest.mark.asyncio
    async def test_async_token_expiration_check(self):
        """Test token expiration validation."""
        client = SpryxAsyncClient(
            base_url="https://api.example.com",
            application_id="test_app",
            application_secret="test_secret",
            iam_base_url="https://iam.example.com",
        )

        # Initially token should be expired (None)
        assert client.is_token_expired is True

        # Set a token with future expiration
        import time

        client._token = "test_token"
        client._token_expires_at = int(time.time()) + 3600  # 1 hour from now

        assert client.is_token_expired is False


class TestSpryxSyncClient:
    def test_sync_client_initialization(self):
        """Test that SpryxSyncClient can be initialized properly."""
        client = SpryxSyncClient(
            base_url="https://api.example.com",
            application_id="test_app",
            application_secret="test_secret",
            iam_base_url="https://iam.example.com",
        )

        assert client._base_url == "https://api.example.com"
        assert client._application_id == "test_app"
        assert client._application_secret == "test_secret"
        assert client._iam_base_url == "https://iam.example.com"

    def test_sync_token_expiration_check(self):
        """Test token expiration validation."""
        client = SpryxSyncClient(
            base_url="https://api.example.com",
            application_id="test_app",
            application_secret="test_secret",
            iam_base_url="https://iam.example.com",
        )

        # Initially token should be expired (None)
        assert client.is_token_expired is True

        # Set a token with future expiration
        import time

        client._token = "test_token"
        client._token_expires_at = int(time.time()) + 3600  # 1 hour from now

        assert client.is_token_expired is False

    def test_sync_client_inheritance(self):
        """Test that SpryxSyncClient has all expected methods."""
        client = SpryxSyncClient(
            base_url="https://api.example.com",
            application_id="test_app",
            application_secret="test_secret",
            iam_base_url="https://iam.example.com",
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
            application_id="test_app",
            application_secret="test_secret",
            iam_base_url="https://iam.example.com",
        )

        sync_client = SpryxSyncClient(
            base_url="https://api.example.com",
            application_id="test_app",
            application_secret="test_secret",
            iam_base_url="https://iam.example.com",
        )

        # Both should inherit from SpryxClientBase
        assert isinstance(async_client, SpryxClientBase)
        assert isinstance(sync_client, SpryxClientBase)

        # Both should have same properties
        assert hasattr(async_client, "is_token_expired")
        assert hasattr(sync_client, "is_token_expired")
        assert hasattr(async_client, "is_refresh_token_expired")
        assert hasattr(sync_client, "is_refresh_token_expired")

        # Both should have same data processing methods
        assert hasattr(async_client, "_extract_data_from_response")
        assert hasattr(sync_client, "_extract_data_from_response")
        assert hasattr(async_client, "_parse_model_data")
        assert hasattr(sync_client, "_parse_model_data")
