"""HTTP retry transport with exponential backoff for httpx clients.

This module provides automatic retry functionality for HTTP requests using exponential
backoff with jitter. It wraps httpx transports to add resilience against transient
failures commonly encountered in distributed systems.

Key Features:
- Exponential backoff with randomized jitter
- Configurable retry limits and backoff factors
- Smart retry logic (only idempotent methods and specific status codes)
- Support for both async and sync httpx clients
- Comprehensive logging of retry attempts

Retry Strategy:
- Status Codes: Only retries on 429 (rate limit) and 5xx server errors (502, 503, 504)
- HTTP Methods: Only retries idempotent methods (GET, HEAD, PUT, DELETE, OPTIONS, TRACE)
- Network Errors: Retries on connection, read, and write errors
- Backoff Formula: delay = backoff_factor * (2 ** (retry_number - 1)) * jitter_multiplier

Configuration:
Set environment variables to customize retry behavior:
- HTTP_RETRIES: Maximum retry attempts (default: 3)
- HTTP_BACKOFF_FACTOR: Base backoff multiplier (default: 0.5)

Example:
    # Basic usage with default settings
    transport = build_retry_transport(is_async=True)
    client = httpx.AsyncClient(transport=transport)

    # Custom retry configuration
    from spryx_http.settings import HttpClientSettings
    settings = HttpClientSettings(retries=5, backoff_factor=1.0)
    transport = build_retry_transport(settings=settings, is_async=False)
    client = httpx.Client(transport=transport)
"""

import asyncio
import random
import time

import httpx

from .logger import logger
from .settings import HttpClientSettings, get_http_settings


class AsyncRetryTransport(httpx.AsyncBaseTransport):
    """Custom async transport with retry logic and exponential backoff.

    This transport wraps another transport and adds retry logic with
    exponential backoff for handling transient failures.
    """

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        status_codes: set[int] | None = None,
        methods: set[str] | None = None,
        jitter: bool = True,
    ):
        """Initialize async retry transport.

        Args:
            transport: The underlying transport to use. If not provided, a
                default transport will be created.
            max_retries: Maximum number of retries before giving up.
            backoff_factor: Backoff factor to apply between attempts.
                {backoff_factor} * (2 ** (retry - 1))
            status_codes: HTTP status codes that should trigger a retry.
                Default is [429, 502, 503, 504].
            methods: HTTP methods that should be retried.
                Default is ["GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"].
            jitter: Whether to add a small random delay to avoid thundering herd.
        """
        self.transport = transport or httpx.AsyncHTTPTransport()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        # Default to common retryable status codes
        self.status_codes = status_codes or {429, 502, 503, 504}
        # Default to idempotent methods
        self.methods = methods or {"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"}
        self.jitter = jitter

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Process the request with retry logic.

        Args:
            request: The HTTP request to send.

        Returns:
            httpx.Response: The HTTP response.

        Raises:
            httpx.RequestError: If the request fails after all retries.
        """
        method = request.method.upper()
        retries_left = self.max_retries

        # Don't retry if the method shouldn't be retried
        if method not in self.methods:
            return await self.transport.handle_async_request(request)

        last_exception: Exception | None = None

        while retries_left > 0:
            try:
                response = await self.transport.handle_async_request(request)

                # If status code is not in the retry list, return the response
                if response.status_code not in self.status_codes:
                    return response

                # If this is the last retry, return the response regardless
                if retries_left == 1:
                    return response

                # Retry is needed, calculate backoff and retry
                retry_number = self.max_retries - retries_left + 1
                wait_time = self._calculate_backoff(retry_number)

                logger.debug(
                    f"Retrying {method} request to {request.url} due to status code {response.status_code} "
                    f"(retry {retry_number}, waiting {wait_time}s)"
                )

                # Wait before retrying
                await asyncio.sleep(wait_time)
                retries_left -= 1

            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError) as exc:
                # Network errors that are often temporary and retryable
                if retries_left == 1:
                    # Last retry, raise the exception
                    raise

                retry_number = self.max_retries - retries_left + 1
                wait_time = self._calculate_backoff(retry_number)

                logger.debug(
                    f"Retrying {method} request to {request.url} due to error: {exc} "
                    f"(retry {retry_number}, waiting {wait_time}s)"
                )

                # Wait before retrying
                await asyncio.sleep(wait_time)
                retries_left -= 1
                last_exception = exc

        # If we've exhausted all retries due to exceptions
        if last_exception is not None:
            raise last_exception

        # This should not happen, but just in case
        raise httpx.TransportError("Exhausted all retries")

    def _calculate_backoff(self, retry_number: int) -> float:
        """Calculate the backoff time for a retry.

        Args:
            retry_number: The current retry attempt number.

        Returns:
            float: The time to wait in seconds.
        """
        backoff = self.backoff_factor * (2 ** (retry_number - 1))

        if self.jitter:
            # Add a small jitter to avoid thundering herds
            backoff = backoff * (0.5 + random.random())

        return backoff


class SyncRetryTransport(httpx.BaseTransport):
    """Custom sync transport with retry logic and exponential backoff.

    This transport wraps another transport and adds retry logic with
    exponential backoff for handling transient failures.
    """

    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        *,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        status_codes: set[int] | None = None,
        methods: set[str] | None = None,
        jitter: bool = True,
    ):
        """Initialize sync retry transport.

        Args:
            transport: The underlying transport to use. If not provided, a
                default transport will be created.
            max_retries: Maximum number of retries before giving up.
            backoff_factor: Backoff factor to apply between attempts.
                {backoff_factor} * (2 ** (retry - 1))
            status_codes: HTTP status codes that should trigger a retry.
                Default is [429, 502, 503, 504].
            methods: HTTP methods that should be retried.
                Default is ["GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"].
            jitter: Whether to add a small random delay to avoid thundering herd.
        """
        self.transport = transport or httpx.HTTPTransport()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        # Default to common retryable status codes
        self.status_codes = status_codes or {429, 502, 503, 504}
        # Default to idempotent methods
        self.methods = methods or {"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"}
        self.jitter = jitter

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Process the request with retry logic.

        Args:
            request: The HTTP request to send.

        Returns:
            httpx.Response: The HTTP response.

        Raises:
            httpx.RequestError: If the request fails after all retries.
        """
        method = request.method.upper()
        retries_left = self.max_retries

        # Don't retry if the method shouldn't be retried
        if method not in self.methods:
            return self.transport.handle_request(request)

        last_exception: Exception | None = None

        while retries_left > 0:
            try:
                response = self.transport.handle_request(request)

                # If status code is not in the retry list, return the response
                if response.status_code not in self.status_codes:
                    return response

                # If this is the last retry, return the response regardless
                if retries_left == 1:
                    return response

                # Retry is needed, calculate backoff and retry
                retry_number = self.max_retries - retries_left + 1
                wait_time = self._calculate_backoff(retry_number)

                logger.debug(
                    f"Retrying {method} request to {request.url} due to status code {response.status_code} "
                    f"(retry {retry_number}, waiting {wait_time}s)"
                )

                # Wait before retrying
                time.sleep(wait_time)
                retries_left -= 1

            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError) as exc:
                # Network errors that are often temporary and retryable
                if retries_left == 1:
                    # Last retry, raise the exception
                    raise

                retry_number = self.max_retries - retries_left + 1
                wait_time = self._calculate_backoff(retry_number)

                logger.debug(
                    f"Retrying {method} request to {request.url} due to error: {exc} "
                    f"(retry {retry_number}, waiting {wait_time}s)"
                )

                # Wait before retrying
                time.sleep(wait_time)
                retries_left -= 1
                last_exception = exc

        # If we've exhausted all retries due to exceptions
        if last_exception is not None:
            raise last_exception

        # This should not happen, but just in case
        raise httpx.TransportError("Exhausted all retries")

    def _calculate_backoff(self, retry_number: int) -> float:
        """Calculate the backoff time for a retry.

        Args:
            retry_number: The current retry attempt number.

        Returns:
            float: The time to wait in seconds.
        """
        backoff = self.backoff_factor * (2 ** (retry_number - 1))

        if self.jitter:
            # Add a small jitter to avoid thundering herds
            backoff = backoff * (0.5 + random.random())

        return backoff


# Backward compatibility alias
RetryTransport = AsyncRetryTransport


def build_retry_transport(
    transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    settings: HttpClientSettings | None = None,
    *,
    is_async: bool = True,
) -> AsyncRetryTransport | SyncRetryTransport:
    """Build a retry transport for httpx client with exponential backoff.

    Creates a transport wrapper that automatically retries failed requests using
    exponential backoff with jitter. This is the main entry point for configuring
    retry behavior in Spryx HTTP clients.

    Retry Strategy:
    - Only retries on specific status codes: 429 (rate limit), 502/503/504 (server errors)
    - Only retries idempotent HTTP methods: GET, HEAD, PUT, DELETE, OPTIONS, TRACE
    - Uses exponential backoff: delay = backoff_factor * (2 ** (retry_number - 1))
    - Adds random jitter to prevent thundering herd effect
    - Retries on network errors: connection, read, write failures

    Configuration via Environment Variables:
    - HTTP_RETRIES: Maximum retry attempts (default: 3)
    - HTTP_BACKOFF_FACTOR: Base backoff multiplier (default: 0.5)

    Example Usage:
        # Async client with custom transport
        transport = build_retry_transport(is_async=True)
        client = httpx.AsyncClient(transport=transport)

        # Sync client with default settings
        transport = build_retry_transport(is_async=False)
        client = httpx.Client(transport=transport)

    Args:
        transport: Base transport to wrap with retry logic.
                  If None, creates a default HTTP transport.
        settings: HTTP client settings containing retry configuration.
                 If None, loads settings from environment variables.
        is_async: Whether to build an async or sync transport.
                 True for AsyncRetryTransport, False for SyncRetryTransport.

    Returns:
        Union[AsyncRetryTransport, SyncRetryTransport]: Configured retry transport
        that wraps the base transport with exponential backoff retry logic.
    """
    settings = settings or get_http_settings()

    if is_async:
        async_transport = transport if isinstance(transport, httpx.AsyncBaseTransport) else None
        return AsyncRetryTransport(
            transport=async_transport,
            max_retries=settings.retries,
            backoff_factor=settings.backoff_factor,
        )
    else:
        sync_transport = transport if isinstance(transport, httpx.BaseTransport) else None
        return SyncRetryTransport(
            transport=sync_transport,
            max_retries=settings.retries,
            backoff_factor=settings.backoff_factor,
        )
