from unittest.mock import MagicMock, patch

import pytest

from spryx_http.async_client import SpryxAsyncClient
from spryx_http.sync_client import SpryxSyncClient


def test_sync_client_with_none_base_url():
    """Test that SpryxSyncClient can be initialized with None base_url."""
    client = SpryxSyncClient(
        base_url=None,
        client_id="test_client_id",
        client_secret="test_client_secret",
        token_url="https://auth.example.com/token",
    )
    # httpx converts None to an empty URL
    assert str(client._base_url) == ""

    # Mock the _get_token method to avoid authentication
    with patch.object(client, "_get_token", return_value="test_token"):
        # Test that a ValueError is raised when path is not a full URL
        with pytest.raises(ValueError, match="Either base_url must be provided"):
            client._make_request("GET", "api/endpoint")

        # Test that a full URL works
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": {"test": "value"}}
            mock_request.return_value = mock_response

            result = client._make_request("GET", "https://example.com/api/endpoint")

            # Verify request was called with the full URL
            mock_request.assert_called_once()
            # Check the URL as positional argument
            assert mock_request.call_args.args[1] == "https://example.com/api/endpoint"

            # Verify result
            assert result == {"test": "value"}


@pytest.mark.asyncio
async def test_async_client_with_none_base_url():
    """Test that SpryxAsyncClient can be initialized with None base_url."""
    client = SpryxAsyncClient(
        base_url=None,
        client_id="test_client_id",
        client_secret="test_client_secret",
        token_url="https://auth.example.com/token",
    )
    # httpx converts None to an empty URL
    assert str(client._base_url) == ""

    # Mock the _get_token method to avoid authentication
    with patch.object(client, "_get_token", return_value="test_token"):
        # Test that a ValueError is raised when path is not a full URL
        with pytest.raises(ValueError, match="Either base_url must be provided"):
            await client._make_request("GET", "api/endpoint")

        # Test that a full URL works
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": {"test": "value"}}
            mock_request.return_value = mock_response

            result = await client._make_request("GET", "https://example.com/api/endpoint")

            # Verify request was called with the full URL
            mock_request.assert_called_once()
            # Check the URL as positional argument
            assert mock_request.call_args.args[1] == "https://example.com/api/endpoint"

            # Verify result
            assert result == {"test": "value"}
