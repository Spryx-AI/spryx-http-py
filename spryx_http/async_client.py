"""Asynchronous HTTP client implementation for Spryx services.

This module provides the SpryxAsyncClient class, an async HTTP client built on top of httpx
with OAuth 2.0 M2M authentication, retry logic, and Pydantic model parsing support.
"""

from typing import Any, TypeVar, overload

import httpx
from pydantic import BaseModel, TypeAdapter

from spryx_http.auth_strategies import AuthStrategy, ClientCredentialsAuthStrategy

from .base import ResponseJson, SpryxClientBase
from .logger import logger
from .settings import HttpClientSettings

T = TypeVar("T", bound=BaseModel)


class SpryxAsyncClient(SpryxClientBase, httpx.AsyncClient):
    """Spryx HTTP async client with retry and auth capabilities.

    Extends httpx.AsyncClient with:
    - OAuth 2.0 M2M authentication with refresh token support
    - Retry with exponential backoff
    - Pydantic model response parsing
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        auth_strategy: AuthStrategy,
        settings: HttpClientSettings | None = None,
        **kwargs,
    ):
        """Initialize the Spryx HTTP async client.

        Args:
            base_url: Base URL for all API requests. Can be None.
            client_id: OAuth 2.0 client ID for M2M authentication.
            client_secret: OAuth 2.0 client secret for M2M authentication.
            token_url: OAuth 2.0 token endpoint URL.
            settings: HTTP client settings.
            **kwargs: Additional arguments to pass to httpx.AsyncClient.
        """
        # Initialize base class
        SpryxClientBase.__init__(
            self,
            base_url=base_url,
            auth_strategy=auth_strategy,
            settings=settings,
            **kwargs,
        )

        # Initialize httpx.AsyncClient with async transport
        transport_kwargs = self._get_transport_kwargs(**self._httpx_kwargs)
        # Pass empty string instead of None to httpx.AsyncClient
        httpx_base_url = "" if self._base_url is None else self._base_url
        httpx.AsyncClient.__init__(self, base_url=httpx_base_url, **transport_kwargs)

    async def authenticate(self) -> None:
        """Authenticate using the configured authentication strategy.

        Uses the authentication strategy provided during initialization
        to authenticate and obtain access tokens.

        Raises:
            httpx.HTTPStatusError: If the token request fails.
        """
        logger.debug(f"Authenticating using {type(self._auth_strategy).__name__}")

        # Delegate authentication to the strategy
        token_data = await self._auth_strategy.authenticate_async(self)
        self._auth_strategy.store_token_response(token_data)

        logger.debug(f"Successfully authenticated using {type(self._auth_strategy).__name__}")

    async def refresh_access_token(self) -> None:
        """Refresh the access token if the strategy supports it.

        This method attempts to use the refresh token to get a new access token
        without requiring full re-authentication. If the strategy doesn't support
        refresh or the refresh fails, it falls back to full authentication.

        Raises:
            ValueError: If unable to refresh or authenticate.
        """
        logger.debug("Attempting to refresh access token")

        if not self._auth_strategy.supports_refresh():
            logger.debug("Strategy doesn't support refresh, doing full authentication")
            await self.authenticate()
            return

        # Try to refresh with ClientCredentials strategy
        if isinstance(self._auth_strategy, ClientCredentialsAuthStrategy):
            token_data = await self._auth_strategy.refresh_async(self)
            if token_data:
                self._auth_strategy.store_token_response(token_data)
                logger.debug("Successfully refreshed access token")
                return

        # Refresh failed, fall back to full authentication
        logger.debug("Refresh failed, falling back to full authentication")
        await self.authenticate()

    async def _ensure_authenticated(self) -> None:
        """Ensure the client is authenticated with a valid token.

        This method handles token lifecycle management, including:
        - Initial authentication if no token exists
        - Token refresh if access token has expired
        - Fallback to full re-authentication if refresh fails

        Raises:
            Exception: If unable to obtain a valid token.
        """
        if self._auth_strategy.needs_refresh():
            logger.debug("Token needs refresh or doesn't exist")
            try:
                await self.refresh_access_token()
            except Exception:
                logger.debug("Refresh failed, doing full authentication", exc_info=True)
                await self.authenticate()

    async def request(
        self,
        method: str,
        url: str | httpx.URL,
        *,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> httpx.Response:
        """Send an HTTP request.

        Args:
            method: HTTP method.
            url: Request URL.
            headers: Request headers.
            **kwargs: Additional arguments to pass to the base request method.

        Returns:
            httpx.Response: The HTTP response.
        """
        # Initialize headers if None
        headers = headers or {}

        return await super().request(method, url, headers=headers, **kwargs)

    @overload
    async def _make_request(
        self,
        method: str,
        path: str,
        *,
        cast_to: type[T],
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> T: ...

    @overload
    async def _make_request(
        self,
        method: str,
        path: str,
        *,
        cast_to: type[list[T]],
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> list[T]: ...

    @overload
    async def _make_request(
        self,
        method: str,
        path: str,
        *,
        cast_to: TypeAdapter,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> Any: ...

    @overload
    async def _make_request(
        self,
        method: str,
        path: str,
        *,
        cast_to: None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> ResponseJson: ...

    async def _make_request(
        self,
        method: str,
        path: str,
        *,
        cast_to: type[T] | type[list[T]] | TypeAdapter | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> T | list[T] | Any | ResponseJson:
        """Core request method to handle HTTP requests with optional Pydantic model parsing.

        Args:
            method: HTTP method.
            path: Request path to be appended to base_url or a full URL if base_url is None.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            params: Optional query parameters.
            json: Optional JSON data for the request body.
            headers: Optional request headers.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            T | ResponseJson: Pydantic model instance or raw JSON data.
        """
        # Check if path is a full URL when base_url is None
        if self._base_url is None and not path.startswith(("http://", "https://")):
            raise ValueError("Either base_url must be provided during initialization or path must be a full URL")

        # Handle path to prevent double slashes if it's not a full URL
        if not path.startswith(("http://", "https://")):
            path = path.lstrip("/")

        # Ensure authenticated
        await self._ensure_authenticated()

        # Handle headers
        request_headers = headers or {}
        request_headers.update(self._auth_strategy.get_auth_headers())

        # Make the request
        try:
            response = await self.request(
                method,
                path,
                headers=self._remove_not_given(request_headers),
                params=self._remove_not_given(params),
                json=self._remove_not_given(json),
                **kwargs,
            )
        except httpx.UnsupportedProtocol as e:
            raise ValueError("Either base_url must be provided during initialization or path must be a full URL") from e

        # Handle authentication failures
        if response.status_code == 401:
            # Token might be expired, try to refresh and retry once
            await self.refresh_access_token()
            request_headers.update(self._auth_strategy.get_auth_headers())
            response = await self.request(method, path, headers=request_headers, params=params, json=json, **kwargs)

        # Process the response
        return self._process_response_data(response, cast_to)

    # HTTP method overloads for proper type inference
    @overload
    async def get(
        self,
        path: str,
        *,
        cast_to: type[T],
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T: ...

    @overload
    async def get(
        self,
        path: str,
        *,
        cast_to: type[list[T]],
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[T]: ...

    @overload
    async def get(
        self,
        path: str,
        *,
        cast_to: TypeAdapter,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any: ...

    @overload
    async def get(
        self,
        path: str,
        *,
        cast_to: None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson: ...

    async def get(
        self,
        path: str,
        *,
        cast_to: type[T] | type[list[T]] | TypeAdapter | None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | Any | ResponseJson:
        """Send a GET request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            T | ResponseJson: Pydantic model instance or raw JSON data.
        """
        return await self._make_request("GET", path, cast_to=cast_to, params=params, **kwargs)

    @overload
    async def post(
        self,
        path: str,
        *,
        cast_to: type[T],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T: ...

    @overload
    async def post(
        self,
        path: str,
        *,
        cast_to: type[list[T]],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[T]: ...

    @overload
    async def post(
        self,
        path: str,
        *,
        cast_to: TypeAdapter,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any: ...

    @overload
    async def post(
        self,
        path: str,
        *,
        cast_to: None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson: ...

    async def post(
        self,
        path: str,
        *,
        cast_to: type[T] | type[list[T]] | TypeAdapter | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | Any | ResponseJson:
        """Send a POST request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            T | ResponseJson: Pydantic model instance or raw JSON data.
        """
        return await self._make_request("POST", path, cast_to=cast_to, json=json, **kwargs)

    @overload
    async def put(
        self,
        path: str,
        *,
        cast_to: type[T],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T: ...

    @overload
    async def put(
        self,
        path: str,
        *,
        cast_to: type[list[T]],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[T]: ...

    @overload
    async def put(
        self,
        path: str,
        *,
        cast_to: TypeAdapter,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any: ...

    @overload
    async def put(
        self,
        path: str,
        *,
        cast_to: None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson: ...

    async def put(
        self,
        path: str,
        *,
        cast_to: type[T] | type[list[T]] | TypeAdapter | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | Any | ResponseJson:
        """Send a PUT request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            T | ResponseJson: Pydantic model instance or raw JSON data.
        """
        return await self._make_request("PUT", path, cast_to=cast_to, json=json, **kwargs)

    @overload
    async def patch(
        self,
        path: str,
        *,
        cast_to: type[T],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T: ...

    @overload
    async def patch(
        self,
        path: str,
        *,
        cast_to: type[list[T]],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[T]: ...

    @overload
    async def patch(
        self,
        path: str,
        *,
        cast_to: TypeAdapter,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any: ...

    @overload
    async def patch(
        self,
        path: str,
        *,
        cast_to: None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson: ...

    async def patch(
        self,
        path: str,
        *,
        cast_to: type[T] | type[list[T]] | TypeAdapter | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | Any | ResponseJson:
        """Send a PATCH request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            T | ResponseJson: Pydantic model instance or raw JSON data.
        """
        return await self._make_request("PATCH", path, cast_to=cast_to, json=json, **kwargs)

    @overload
    async def delete(
        self,
        path: str,
        *,
        cast_to: type[T],
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T: ...

    @overload
    async def delete(
        self,
        path: str,
        *,
        cast_to: type[list[T]],
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[T]: ...

    @overload
    async def delete(
        self,
        path: str,
        *,
        cast_to: TypeAdapter,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any: ...

    @overload
    async def delete(
        self,
        path: str,
        *,
        cast_to: None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson: ...

    async def delete(
        self,
        path: str,
        *,
        cast_to: type[T] | type[list[T]] | TypeAdapter | None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | Any | ResponseJson:
        """Send a DELETE request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            T | ResponseJson: Pydantic model instance or raw JSON data.
        """
        return await self._make_request("DELETE", path, cast_to=cast_to, params=params, **kwargs)
