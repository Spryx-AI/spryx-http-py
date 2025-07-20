"""
Comprehensive unit tests for the retry system.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from spryx_http.retry import AsyncRetryTransport, SyncRetryTransport, build_retry_transport
from spryx_http.settings import HttpClientSettings


class TestAsyncRetryTransport:
    """Test async retry transport functionality."""

    @pytest.fixture
    def mock_transport(self):
        """Create a mock async transport."""
        transport = Mock(spec=httpx.AsyncBaseTransport)
        transport.handle_async_request = AsyncMock()
        return transport

    @pytest.fixture
    def retry_transport(self, mock_transport):
        """Create retry transport with mock underlying transport."""
        return AsyncRetryTransport(
            transport=mock_transport,
            max_retries=3,
            backoff_factor=0.01,  # Very small for faster tests (was 0.1)
            jitter=False,  # Disable for predictable tests
        )

    @pytest.fixture
    def mock_request(self):
        """Create a mock HTTP request."""
        request = Mock(spec=httpx.Request)
        request.method = "GET"
        request.url = httpx.URL("https://api.test.com/users")
        return request

    @pytest.mark.asyncio
    async def test_successful_request_no_retry(self, retry_transport, mock_transport, mock_request):
        """Test successful request doesn't trigger retry."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_transport.handle_async_request.return_value = mock_response

        result = await retry_transport.handle_async_request(mock_request)

        assert result == mock_response
        assert mock_transport.handle_async_request.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_status_codes(self, retry_transport, mock_transport, mock_request):
        """Test retry on specific status codes (429, 502, 503, 504)."""
        retryable_codes = [429, 502, 503, 504]

        for status_code in retryable_codes:
            # Reset mock
            mock_transport.handle_async_request.reset_mock()

            # First two calls return error, third succeeds
            error_response = Mock(spec=httpx.Response)
            error_response.status_code = status_code
            success_response = Mock(spec=httpx.Response)
            success_response.status_code = 200

            mock_transport.handle_async_request.side_effect = [error_response, error_response, success_response]

            result = await retry_transport.handle_async_request(mock_request)

            assert result == success_response
            assert mock_transport.handle_async_request.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_status_codes(self, retry_transport, mock_transport, mock_request):
        """Test no retry on non-retryable status codes."""
        non_retryable_codes = [400, 401, 403, 404, 422]

        for status_code in non_retryable_codes:
            # Reset mock
            mock_transport.handle_async_request.reset_mock()

            error_response = Mock(spec=httpx.Response)
            error_response.status_code = status_code
            mock_transport.handle_async_request.return_value = error_response

            result = await retry_transport.handle_async_request(mock_request)

            assert result == error_response
            assert mock_transport.handle_async_request.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_only_idempotent_methods(self, mock_transport):
        """Test retry only happens for idempotent HTTP methods."""
        retry_transport = AsyncRetryTransport(
            transport=mock_transport,
            max_retries=2,
            backoff_factor=0.01,  # Very small for faster tests (was 0.1)
            jitter=False,
        )

        # Test idempotent methods (should retry)
        idempotent_methods = ["GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"]

        for method in idempotent_methods:
            mock_transport.handle_async_request.reset_mock()

            request = Mock(spec=httpx.Request)
            request.method = method
            request.url = httpx.URL("https://api.test.com/test")

            error_response = Mock(spec=httpx.Response)
            error_response.status_code = 503
            success_response = Mock(spec=httpx.Response)
            success_response.status_code = 200

            mock_transport.handle_async_request.side_effect = [error_response, success_response]

            result = await retry_transport.handle_async_request(request)

            assert result == success_response
            assert mock_transport.handle_async_request.call_count == 2

        # Test non-idempotent methods (should not retry)
        non_idempotent_methods = ["POST", "PATCH"]

        for method in non_idempotent_methods:
            mock_transport.handle_async_request.reset_mock()

            request = Mock(spec=httpx.Request)
            request.method = method
            request.url = httpx.URL("https://api.test.com/test")

            error_response = Mock(spec=httpx.Response)
            error_response.status_code = 503
            # Reset side_effect to single return value
            mock_transport.handle_async_request.side_effect = None
            mock_transport.handle_async_request.return_value = error_response

            result = await retry_transport.handle_async_request(request)

            assert result == error_response
            assert mock_transport.handle_async_request.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_network_errors(self, retry_transport, mock_transport, mock_request):
        """Test retry on network errors."""
        network_errors = [
            httpx.ConnectError("Connection failed"),
            httpx.ReadError("Read timeout"),
            httpx.WriteError("Write failed"),
        ]

        for error in network_errors:
            mock_transport.handle_async_request.reset_mock()

            success_response = Mock(spec=httpx.Response)
            success_response.status_code = 200

            # First call raises error, second succeeds
            mock_transport.handle_async_request.side_effect = [error, success_response]

            result = await retry_transport.handle_async_request(mock_request)

            assert result == success_response
            assert mock_transport.handle_async_request.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_status_code(self, retry_transport, mock_transport, mock_request):
        """Test that after max retries, the last response is returned."""
        error_response = Mock(spec=httpx.Response)
        error_response.status_code = 503
        mock_transport.handle_async_request.return_value = error_response

        result = await retry_transport.handle_async_request(mock_request)

        assert result == error_response
        # Should try 3 times (max_retries = 3)
        assert mock_transport.handle_async_request.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_network_error(self, retry_transport, mock_transport, mock_request):
        """Test that after max retries with network error, exception is raised."""
        network_error = httpx.ConnectError("Connection failed")
        mock_transport.handle_async_request.side_effect = network_error

        with pytest.raises(httpx.ConnectError):
            await retry_transport.handle_async_request(mock_request)

        # Should try 3 times (max_retries = 3)
        assert mock_transport.handle_async_request.call_count == 3

    @pytest.mark.asyncio
    async def test_backoff_calculation(self, mock_transport):
        """Test exponential backoff calculation."""
        retry_transport = AsyncRetryTransport(transport=mock_transport, max_retries=3, backoff_factor=1.0, jitter=False)

        # Test backoff calculation for different retry numbers
        assert retry_transport._calculate_backoff(1) == 1.0  # 1.0 * (2^0) = 1.0
        assert retry_transport._calculate_backoff(2) == 2.0  # 1.0 * (2^1) = 2.0
        assert retry_transport._calculate_backoff(3) == 4.0  # 1.0 * (2^2) = 4.0

    @pytest.mark.asyncio
    async def test_backoff_with_jitter(self, mock_transport):
        """Test backoff calculation with jitter."""
        retry_transport = AsyncRetryTransport(transport=mock_transport, max_retries=3, backoff_factor=1.0, jitter=True)

        # With jitter, backoff should be between base_delay * 0.5 and base_delay * 1.5
        for retry_num in [1, 2, 3]:
            base_delay = 1.0 * (2 ** (retry_num - 1))
            backoff = retry_transport._calculate_backoff(retry_num)
            assert base_delay * 0.5 <= backoff <= base_delay * 1.5

    @pytest.mark.asyncio
    async def test_backoff_timing(self, mock_transport, mock_request):
        """Test that backoff timing is actually applied."""
        retry_transport = AsyncRetryTransport(
            transport=mock_transport,
            max_retries=2,
            backoff_factor=0.01,  # Very small delay for test speed (was 0.1)
            jitter=False,
        )

        error_response = Mock(spec=httpx.Response)
        error_response.status_code = 503
        success_response = Mock(spec=httpx.Response)
        success_response.status_code = 200

        mock_transport.handle_async_request.side_effect = [error_response, success_response]

        start_time = time.time()
        result = await retry_transport.handle_async_request(mock_request)
        end_time = time.time()

        assert result == success_response
        # Should have waited at least 0.01 seconds (backoff_factor * 2^0)
        assert end_time - start_time >= 0.01


class TestSyncRetryTransport:
    """Test sync retry transport functionality."""

    @pytest.fixture
    def mock_transport(self):
        """Create a mock sync transport."""
        transport = Mock(spec=httpx.BaseTransport)
        return transport

    @pytest.fixture
    def retry_transport(self, mock_transport):
        """Create sync retry transport with mock underlying transport."""
        return SyncRetryTransport(
            transport=mock_transport,
            max_retries=2,
            backoff_factor=0.01,  # Very small for faster tests (was 0.1)
            jitter=False,
        )

    @pytest.fixture
    def mock_request(self):
        """Create a mock HTTP request."""
        request = Mock(spec=httpx.Request)
        request.method = "GET"
        request.url = httpx.URL("https://api.test.com/users")
        return request

    def test_sync_retry_on_status_code(self, retry_transport, mock_transport, mock_request):
        """Test sync retry on retryable status codes."""
        error_response = Mock(spec=httpx.Response)
        error_response.status_code = 503
        success_response = Mock(spec=httpx.Response)
        success_response.status_code = 200

        mock_transport.handle_request.side_effect = [error_response, success_response]

        result = retry_transport.handle_request(mock_request)

        assert result == success_response
        assert mock_transport.handle_request.call_count == 2

    def test_sync_no_retry_non_idempotent(self, retry_transport, mock_transport):
        """Test sync transport doesn't retry non-idempotent methods."""
        request = Mock(spec=httpx.Request)
        request.method = "POST"
        request.url = httpx.URL("https://api.test.com/users")

        error_response = Mock(spec=httpx.Response)
        error_response.status_code = 503
        mock_transport.handle_request.return_value = error_response

        result = retry_transport.handle_request(request)

        assert result == error_response
        assert mock_transport.handle_request.call_count == 1


class TestBuildRetryTransport:
    """Test the build_retry_transport factory function."""

    def test_build_async_transport_default(self):
        """Test building async retry transport with default settings."""
        transport = build_retry_transport(is_async=True)

        assert isinstance(transport, AsyncRetryTransport)
        assert transport.max_retries == 3  # Default from settings
        assert transport.backoff_factor == 0.5  # Default from settings

    def test_build_sync_transport_default(self):
        """Test building sync retry transport with default settings."""
        transport = build_retry_transport(is_async=False)

        assert isinstance(transport, SyncRetryTransport)
        assert transport.max_retries == 3  # Default from settings
        assert transport.backoff_factor == 0.5  # Default from settings

    def test_build_transport_custom_settings(self):
        """Test building transport with custom settings."""
        import os

        with patch.dict(os.environ, {"HTTP_RETRIES": "5", "HTTP_BACKOFF_FACTOR": "1.0"}):
            custom_settings = HttpClientSettings()
            transport = build_retry_transport(settings=custom_settings, is_async=True)

            assert isinstance(transport, AsyncRetryTransport)
            assert transport.max_retries == 5
            assert transport.backoff_factor == 1.0

    def test_build_transport_with_base_transport(self):
        """Test building transport with custom base transport."""
        base_transport = Mock(spec=httpx.AsyncBaseTransport)
        transport = build_retry_transport(transport=base_transport, is_async=True)

        assert isinstance(transport, AsyncRetryTransport)
        assert transport.transport == base_transport

    def test_build_transport_environment_settings(self):
        """Test building transport with environment variable settings."""
        import os

        with patch.dict(os.environ, {"HTTP_RETRIES": "7", "HTTP_BACKOFF_FACTOR": "2.0"}):
            transport = build_retry_transport(is_async=True)

            assert isinstance(transport, AsyncRetryTransport)
            assert transport.max_retries == 7
            assert transport.backoff_factor == 2.0


class TestRetryIntegration:
    """Integration tests for retry functionality."""

    def test_retry_status_codes_configuration(self):
        """Test custom retry status codes configuration."""
        custom_status_codes = {429, 500, 502}
        transport = AsyncRetryTransport(max_retries=2, status_codes=custom_status_codes)

        assert transport.status_codes == custom_status_codes

    def test_retry_methods_configuration(self):
        """Test custom retry methods configuration."""
        custom_methods = {"GET", "POST"}  # Including POST for testing
        transport = AsyncRetryTransport(max_retries=2, methods=custom_methods)

        assert transport.methods == custom_methods
