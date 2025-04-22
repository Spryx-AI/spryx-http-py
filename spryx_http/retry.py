import asyncio
import random
from typing import Optional, Set

import httpx
import logfire

from spryx_http.settings import HttpClientSettings, get_http_settings


class RetryTransport(httpx.AsyncBaseTransport):
    """Custom transport with retry logic and exponential backoff.

    This transport wraps another transport and adds retry logic with
    exponential backoff for handling transient failures.
    """

    def __init__(
        self,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        *,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        status_codes: Optional[Set[int]] = None,
        methods: Optional[Set[str]] = None,
        jitter: bool = True,
    ):
        """Initialize retry transport.

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

        last_exception: Optional[Exception] = None

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

                logfire.debug(
                    f"Retrying {method} request to {request.url} due to status code {response.status_code}",
                    retry_number=retry_number,
                    status_code=response.status_code,
                    backoff_seconds=wait_time,
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

                logfire.debug(
                    f"Retrying {method} request to {request.url} due to error",
                    retry_number=retry_number,
                    error=str(exc),
                    backoff_seconds=wait_time,
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


def build_retry_transport(
    transport: Optional[httpx.AsyncBaseTransport] = None,
    settings: Optional[HttpClientSettings] = None,
) -> RetryTransport:
    """Build a retry transport for httpx client.

    Args:
        transport: Base transport to wrap with retry logic.
        settings: HTTP client settings.

    Returns:
        RetryTransport: Configured retry transport.
    """
    settings = settings or get_http_settings()

    return RetryTransport(
        transport=transport,
        max_retries=settings.retries,
        backoff_factor=settings.backoff_factor,
    )
