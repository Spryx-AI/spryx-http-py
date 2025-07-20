"""Tests for the _remove_not_given method functionality."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from spryx_core import NotGiven

from spryx_http.async_client import SpryxAsyncClient
from spryx_http.sync_client import SpryxSyncClient


class TestRemoveNotGiven:
    """Test the _remove_not_given method behavior."""

    @pytest.fixture
    def client_config(self):
        """Basic client configuration."""
        return {
            "base_url": "https://api.example.com",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "token_url": "https://auth.example.com/token",
        }

    def test_remove_not_given_none_input(self, client_config):
        """Test _remove_not_given with None input."""
        client = SpryxAsyncClient(**client_config)
        result = client._remove_not_given(None)
        assert result is None

    def test_remove_not_given_empty_dict(self, client_config):
        """Test _remove_not_given with empty dictionary."""
        client = SpryxAsyncClient(**client_config)
        result = client._remove_not_given({})
        assert result == {}

    def test_remove_not_given_no_not_given_values(self, client_config):
        """Test _remove_not_given with dictionary containing no NotGiven values."""
        client = SpryxAsyncClient(**client_config)
        input_dict = {"key1": "value1", "key2": 123, "key3": ["item1", "item2"], "key4": {"nested": "dict"}}
        result = client._remove_not_given(input_dict)
        assert result == input_dict

    def test_remove_not_given_only_not_given_values(self, client_config):
        """Test _remove_not_given with dictionary containing only NotGiven values."""
        client = SpryxAsyncClient(**client_config)
        input_dict = {"key1": NotGiven, "key2": NotGiven, "key3": NotGiven}
        result = client._remove_not_given(input_dict)
        assert result == {}

    def test_remove_not_given_mixed_values(self, client_config):
        """Test _remove_not_given with dictionary containing mixed values."""
        client = SpryxAsyncClient(**client_config)
        input_dict = {
            "keep1": "value1",
            "remove1": NotGiven,
            "keep2": 123,
            "remove2": NotGiven,
            "keep3": {"nested": "dict"},
            "remove3": NotGiven,
        }
        expected = {"keep1": "value1", "keep2": 123, "keep3": {"nested": "dict"}}
        result = client._remove_not_given(input_dict)
        assert result == expected

    def test_remove_not_given_preserves_none_values(self, client_config):
        """Test _remove_not_given preserves None values (only removes NotGiven)."""
        client = SpryxAsyncClient(**client_config)
        input_dict = {"none_value": None, "not_given_value": NotGiven, "empty_string": "", "zero": 0, "false": False}
        expected = {"none_value": None, "empty_string": "", "zero": 0, "false": False}
        result = client._remove_not_given(input_dict)
        assert result == expected


class TestRemoveNotGivenIntegration:
    """Integration tests for _remove_not_given in HTTP requests."""

    @pytest.fixture
    def client_config(self):
        """Basic client configuration."""
        return {
            "base_url": "https://api.example.com",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "token_url": "https://auth.example.com/token",
        }

    @pytest.mark.asyncio
    async def test_async_get_request_with_not_given_headers(self, client_config):
        """Test async GET request filters NotGiven values from headers."""
        client = SpryxAsyncClient(**client_config)

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-type": "application/json"}
            mock_request.return_value = mock_response

            # Mock authentication
            with patch.object(client, "_get_token", new_callable=AsyncMock) as mock_get_token:
                mock_get_token.return_value = "test_token"

                # Make request with NotGiven values in headers
                await client.get(
                    "/test", headers={"Custom-Header": "value", "Remove-This": NotGiven, "Keep-This": "another_value"}
                )

                # Verify request was called with filtered headers
                mock_request.assert_called_once()
                call_kwargs = mock_request.call_args[1]

                # Headers should not contain NotGiven values
                expected_headers = {
                    "Authorization": "Bearer test_token",
                    "Custom-Header": "value",
                    "Keep-This": "another_value",
                }
                assert call_kwargs["headers"] == expected_headers

    @pytest.mark.asyncio
    async def test_async_post_request_with_not_given_params_and_json(self, client_config):
        """Test async POST request filters NotGiven values from params and json."""
        client = SpryxAsyncClient(**client_config)

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 123}
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-type": "application/json"}
            mock_request.return_value = mock_response

            # Mock authentication
            with patch.object(client, "_get_token", new_callable=AsyncMock) as mock_get_token:
                mock_get_token.return_value = "test_token"

                # Make request with NotGiven values in params and json
                await client.post(
                    "/test",
                    params={"keep_param": "value1", "remove_param": NotGiven, "another_param": "value2"},
                    json={"field1": "data1", "field2": NotGiven, "field3": {"nested": "data"}},
                )

                # Verify request was called with filtered params and json
                mock_request.assert_called_once()
                call_kwargs = mock_request.call_args[1]

                # Params should not contain NotGiven values
                expected_params = {"keep_param": "value1", "another_param": "value2"}
                assert call_kwargs["params"] == expected_params

                # JSON should not contain NotGiven values
                expected_json = {"field1": "data1", "field3": {"nested": "data"}}
                assert call_kwargs["json"] == expected_json

    @pytest.mark.asyncio
    async def test_async_request_with_all_not_given_optional_params(self, client_config):
        """Test async request when all optional params are NotGiven."""
        client = SpryxAsyncClient(**client_config)

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-type": "application/json"}
            mock_request.return_value = mock_response

            # Mock authentication
            with patch.object(client, "_get_token", new_callable=AsyncMock) as mock_get_token:
                mock_get_token.return_value = "test_token"

                # Make request with all optional params as NotGiven
                await client.get(
                    "/test",
                    headers={"Remove-All": NotGiven, "Also-Remove": NotGiven},
                    params={"remove_this": NotGiven, "and_this": NotGiven},
                )

                # Verify request was called with only auth headers and no params
                mock_request.assert_called_once()
                call_kwargs = mock_request.call_args[1]

                # Only auth header should remain
                expected_headers = {"Authorization": "Bearer test_token"}
                assert call_kwargs["headers"] == expected_headers

                # Params should be empty dict
                assert call_kwargs["params"] == {}

    @pytest.mark.asyncio
    async def test_async_request_preserves_none_values_in_json(self, client_config):
        """Test that None values are preserved in JSON while NotGiven are removed."""
        client = SpryxAsyncClient(**client_config)

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-type": "application/json"}
            mock_request.return_value = mock_response

            # Mock authentication
            with patch.object(client, "_get_token", new_callable=AsyncMock) as mock_get_token:
                mock_get_token.return_value = "test_token"

                # Make request with None and NotGiven values
                await client.post(
                    "/test",
                    json={
                        "explicit_none": None,
                        "remove_this": NotGiven,
                        "empty_string": "",
                        "zero": 0,
                        "false_value": False,
                    },
                )

                # Verify None and other falsy values are preserved, NotGiven removed
                mock_request.assert_called_once()
                call_kwargs = mock_request.call_args[1]

                expected_json = {"explicit_none": None, "empty_string": "", "zero": 0, "false_value": False}
                assert call_kwargs["json"] == expected_json

    def test_sync_client_also_supports_remove_not_given(self, client_config):
        """Test that sync client also has _remove_not_given functionality."""
        client = SpryxSyncClient(**client_config)

        # Test the method exists and works
        input_dict = {"keep": "value", "remove": NotGiven}
        result = client._remove_not_given(input_dict)
        expected = {"keep": "value"}
        assert result == expected


class TestRemoveNotGivenEdgeCases:
    """Test edge cases for _remove_not_given functionality."""

    @pytest.fixture
    def client_config(self):
        """Basic client configuration."""
        return {
            "base_url": "https://api.example.com",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "token_url": "https://auth.example.com/token",
        }

    def test_remove_not_given_with_nested_dict_containing_not_given(self, client_config):
        """Test _remove_not_given does not recurse into nested dictionaries."""
        client = SpryxAsyncClient(**client_config)
        input_dict = {
            "keep": "value",
            "nested": {
                "inner_keep": "inner_value",
                "inner_remove": NotGiven,  # This should NOT be removed (no recursion)
            },
            "remove": NotGiven,
        }
        expected = {
            "keep": "value",
            "nested": {
                "inner_keep": "inner_value",
                "inner_remove": NotGiven,  # Still present because no recursion
            },
        }
        result = client._remove_not_given(input_dict)
        assert result == expected

    def test_remove_not_given_with_list_containing_not_given(self, client_config):
        """Test _remove_not_given does not filter NotGiven from lists."""
        client = SpryxAsyncClient(**client_config)
        input_dict = {"keep": "value", "list_with_not_given": ["item1", NotGiven, "item3"], "remove": NotGiven}
        expected = {
            "keep": "value",
            "list_with_not_given": ["item1", NotGiven, "item3"],  # List items unchanged
        }
        result = client._remove_not_given(input_dict)
        assert result == expected

    def test_remove_not_given_preserves_original_dict(self, client_config):
        """Test _remove_not_given does not modify the original dictionary."""
        client = SpryxAsyncClient(**client_config)
        original_dict = {"keep": "value", "remove": NotGiven}
        original_copy = original_dict.copy()

        result = client._remove_not_given(original_dict)

        # Original should be unchanged
        assert original_dict == original_copy
        # Result should be filtered
        assert result == {"keep": "value"}

    @pytest.mark.asyncio
    async def test_async_request_handles_none_parameters_correctly(self, client_config):
        """Test that None parameters are handled correctly (not filtered)."""
        client = SpryxAsyncClient(**client_config)

        with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-type": "application/json"}
            mock_request.return_value = mock_response

            # Mock authentication
            with patch.object(client, "_get_token", new_callable=AsyncMock) as mock_get_token:
                mock_get_token.return_value = "test_token"

                # Test with None headers, params, and json (should not be filtered)
                await client.post("/test", headers=None, params=None, json=None)

                # Verify None values were passed through (not filtered)
                mock_request.assert_called_once()
                call_kwargs = mock_request.call_args[1]

                # None should be preserved, only NotGiven should be filtered
                assert call_kwargs["headers"] == {"Authorization": "Bearer test_token"}
                assert call_kwargs["params"] is None
                assert call_kwargs["json"] is None

    @pytest.mark.asyncio
    async def test_multiple_http_methods_use_remove_not_given(self, client_config):
        """Test that all HTTP methods use _remove_not_given consistently."""
        client = SpryxAsyncClient(**client_config)

        methods_to_test = [
            ("get", client.get),
            ("post", client.post),
            ("put", client.put),
            ("patch", client.patch),
            ("delete", client.delete),
        ]

        for method_name, method_func in methods_to_test:
            with patch.object(client, "request", new_callable=AsyncMock) as mock_request:
                mock_response = Mock(spec=httpx.Response)
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True}
                mock_response.raise_for_status.return_value = None
                mock_response.headers = {"content-type": "application/json"}
                mock_request.return_value = mock_response

                # Mock authentication
                with patch.object(client, "_get_token", new_callable=AsyncMock) as mock_get_token:
                    mock_get_token.return_value = "test_token"

                    # Make request with NotGiven values
                    await method_func(
                        "/test",
                        headers={"Keep-Header": "value", "Remove-Header": NotGiven},
                        params={"keep_param": "value", "remove_param": NotGiven},
                    )

                    # Verify NotGiven values were filtered for all methods
                    mock_request.assert_called_once()
                    call_kwargs = mock_request.call_args[1]

                    expected_headers = {"Authorization": "Bearer test_token", "Keep-Header": "value"}
                    expected_params = {"keep_param": "value"}

                    assert call_kwargs["headers"] == expected_headers, f"Failed for method {method_name}"
                    assert call_kwargs["params"] == expected_params, f"Failed for method {method_name}"

    def test_sync_client_request_uses_remove_not_given(self, client_config):
        """Test that sync client also uses _remove_not_given in requests."""
        client = SpryxSyncClient(**client_config)

        with patch.object(client, "request") as mock_request:
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-type": "application/json"}
            mock_request.return_value = mock_response

            # Mock authentication
            with patch.object(client, "_get_token") as mock_get_token:
                mock_get_token.return_value = "test_token"

                # Make request with NotGiven values
                client.post(
                    "/test",
                    headers={"Keep-Header": "value", "Remove-Header": NotGiven},
                    params={"keep_param": "value", "remove_param": NotGiven},
                    json={"field1": "data1", "field2": NotGiven, "field3": None},
                )

                # Verify request was called with filtered values
                mock_request.assert_called_once()
                call_kwargs = mock_request.call_args[1]

                # Headers should not contain NotGiven values
                expected_headers = {"Authorization": "Bearer test_token", "Keep-Header": "value"}
                assert call_kwargs["headers"] == expected_headers

                # Params should not contain NotGiven values
                expected_params = {"keep_param": "value"}
                assert call_kwargs["params"] == expected_params

                # JSON should not contain NotGiven values but preserve None
                expected_json = {"field1": "data1", "field3": None}
                assert call_kwargs["json"] == expected_json
