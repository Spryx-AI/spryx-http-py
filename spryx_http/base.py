import time
from collections.abc import Mapping
from typing import (
    Any,
    TypeVar,
    overload,
)

import httpx
import jwt
import logfire
from pydantic import BaseModel

from spryx_http.exceptions import raise_for_status
from spryx_http.retry import build_retry_transport
from spryx_http.settings import HttpClientSettings, get_http_settings

T = TypeVar("T", bound=BaseModel)
ResponseJson = Mapping[str, Any]


class SpryxClientBase:
    """Base class for Spryx HTTP clients with common functionality.

    Contains shared functionality between async and sync clients:
    - OAuth 2.0 M2M authentication with refresh token support
    - Token management and validation
    - Response data processing
    - Settings management
    """

    _access_token: str | None = None
    _token_expires_at: int | None = None
    _refresh_token: str | None = None

    def __init__(
        self,
        *,
        base_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_url: str,
        scope: str | None = None,
        settings: HttpClientSettings | None = None,
        **kwargs,
    ):
        """Initialize the base Spryx HTTP client.

        Args:
            base_url: Base URL for all API requests. Can be None.
            client_id: OAuth 2.0 client ID for M2M authentication.
            client_secret: OAuth 2.0 client secret for M2M authentication.
            token_url: OAuth 2.0 token endpoint URL.
            scope: Optional OAuth 2.0 scope for the access token.
            settings: HTTP client settings.
            **kwargs: Additional arguments to pass to httpx client.
        """
        self._base_url = base_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._scope = scope
        self.settings = settings or get_http_settings()

        # Configure timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.settings.timeout_s

        self._httpx_kwargs = kwargs

    def _get_transport_kwargs(self, **kwargs):
        """Get transport configuration for the client.

        This method should be overridden by subclasses to provide
        the appropriate transport configuration.
        """
        # Configure retry transport if not provided
        if "transport" not in kwargs:
            kwargs["transport"] = build_retry_transport(
                settings=self.settings, is_async=True
            )
        return kwargs

    @property
    def is_token_expired(self) -> bool:
        """Check if the access token is expired.

        Returns:
            bool: True if the token is expired or not set, False otherwise.
        """
        if self._access_token is None or self._token_expires_at is None:
            return True

        # Add 30 seconds buffer to account for request time
        current_time = int(time.time()) + 30
        return current_time >= self._token_expires_at

    def _parse_jwt_expiration(self, token: str) -> int:
        """Parse JWT token to extract expiration time.

        Args:
            token: The JWT token to parse.

        Returns:
            int: Unix timestamp of token expiration.
        """
        try:
            # Decode JWT without verification (we just need the exp claim)
            decoded = jwt.decode(token, options={"verify_signature": False})
            return int(decoded.get("exp", 0))
        except Exception:
            # If we can't parse the JWT, assume it expires in 15 minutes
            return int(time.time()) + 900

    def _extract_data_from_response(self, response_data: dict[str, Any]) -> Any:
        """Extract data from standardized API response.

        In our standardized API response, the actual entity is always under a 'data' key.

        Args:
            response_data: The response data dictionary.

        Returns:
            Any: The extracted data.
        """
        if "data" in response_data:
            return response_data["data"]
        return response_data

    def _parse_model_data(self, model_cls: type[T], data: Any) -> T | list[T]:
        """Parse data into a Pydantic model or list of models.

        Args:
            model_cls: The Pydantic model class to parse into.
            data: The data to parse.

        Returns:
            Union[T, List[T]]: Parsed model instance(s).
        """
        if isinstance(data, list):
            return [model_cls.model_validate(item) for item in data]
        return model_cls.model_validate(data)

    def _process_response_data(
        self, response: httpx.Response, cast_to: type[T] | None = None
    ) -> T | list[T] | ResponseJson:
        """Process the response by validating status and converting to model.

        Args:
            response: The HTTP response.
            cast_to: Optional Pydantic model class to parse response into.
                     If None, returns the raw JSON data.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        # Raise exception for error status codes
        raise_for_status(response)

        # Parse JSON response
        try:
            json_data = response.json()
        except ValueError as e:
            raise ValueError(f"Failed to parse JSON response: {response.text}") from e

        # Extract data from standard response format
        data = self._extract_data_from_response(json_data)

        # If cast_to is provided, parse into model, otherwise return the raw data
        if cast_to is not None:
            return self._parse_model_data(cast_to, data)
        return data


class SpryxAsyncClient(SpryxClientBase, httpx.AsyncClient):
    """Spryx HTTP async client with retry, tracing, and auth capabilities.

    Extends httpx.AsyncClient with:
    - OAuth 2.0 M2M authentication with refresh token support
    - Retry with exponential backoff
    - Structured logging with Logfire
    - Correlation ID propagation
    - Pydantic model response parsing
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_url: str,
        scope: str | None = None,
        settings: HttpClientSettings | None = None,
        **kwargs,
    ):
        """Initialize the Spryx HTTP async client.

        Args:
            base_url: Base URL for all API requests. Can be None.
            client_id: OAuth 2.0 client ID for M2M authentication.
            client_secret: OAuth 2.0 client secret for M2M authentication.
            token_url: OAuth 2.0 token endpoint URL.
            scope: Optional OAuth 2.0 scope for the access token.
            settings: HTTP client settings.
            **kwargs: Additional arguments to pass to httpx.AsyncClient.
        """
        # Initialize base class
        SpryxClientBase.__init__(
            self,
            base_url=base_url,
            client_id=client_id,
            client_secret=client_secret,
            token_url=token_url,
            scope=scope,
            settings=settings,
            **kwargs,
        )

        # Initialize httpx.AsyncClient with async transport
        transport_kwargs = self._get_transport_kwargs(**self._httpx_kwargs)
        # Pass empty string instead of None to httpx.AsyncClient
        httpx_base_url = "" if self._base_url is None else self._base_url
        httpx.AsyncClient.__init__(self, base_url=httpx_base_url, **transport_kwargs)

    async def authenticate_client_credentials(self) -> str:
        """Authenticate using OAuth 2.0 Client Credentials flow.

        Uses the client_id and client_secret provided during initialization
        to authenticate with the OAuth 2.0 token endpoint and obtain access and refresh tokens.

        Returns:
            str: The access token.

        Raises:
            ValueError: If client_id or client_secret is not provided.
            httpx.HTTPStatusError: If the token request fails.
        """
        if self._client_id is None:
            raise ValueError("client_id is required for OAuth 2.0 authentication")

        if self._client_secret is None:
            raise ValueError("client_secret is required for OAuth 2.0 authentication")

        payload = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }

        if self._scope:
            payload["scope"] = self._scope

        response = await self.request("POST", self._token_url, data=payload)
        response.raise_for_status()

        token_data = response.json()

        self._access_token = token_data["access_token"]
        self._refresh_token = token_data.get("refresh_token")

        # Parse JWT to get expiration time
        if self._access_token:
            self._token_expires_at = self._parse_jwt_expiration(self._access_token)

        if self._access_token is None:
            raise ValueError("Failed to obtain access token")
        return self._access_token

    async def refresh_access_token(self) -> str:
        """Refresh the access token using the refresh token.

        This method attempts to use the refresh token to get a new access token
        without requiring full re-authentication. If the refresh token has expired
        or the refresh fails, it falls back to full client credentials authentication.

        Returns:
            str: The new access token.

        Raises:
            ValueError: If no refresh token is available and client credentials fail.
        """
        if self._refresh_token is None:
            # No refresh token available, do full authentication
            return await self.authenticate_client_credentials()

        try:
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            }

            response = await self.request("POST", self._token_url, data=payload)
            response.raise_for_status()

            token_data = response.json()

            self._access_token = token_data["access_token"]
            # Some implementations return a new refresh token
            if "refresh_token" in token_data:
                self._refresh_token = token_data["refresh_token"]

            # Parse JWT to get expiration time
            if self._access_token:
                self._token_expires_at = self._parse_jwt_expiration(self._access_token)

            if self._access_token is None:
                raise ValueError("Failed to obtain access token")
            return self._access_token

        except (httpx.HTTPStatusError, httpx.RequestError, ValueError, KeyError):
            # Refresh failed, fall back to full authentication
            logfire.debug("Refresh token failed, falling back to client credentials authentication")
            return await self.authenticate_client_credentials()

    async def _get_token(self) -> str:
        """Get a valid authentication token.

        This method handles token lifecycle management, including:
        - Initial authentication if no token exists
        - Token refresh if access token has expired
        - Fallback to full re-authentication if refresh fails

        Returns:
            str: A valid authentication token.

        Raises:
            Exception: If unable to obtain a valid token.
        """
        if self._access_token is None or self.is_token_expired:
            if self._refresh_token is not None:
                # Try to refresh first
                try:
                    await self.refresh_access_token()
                except Exception:
                    # If refresh fails, do full authentication
                    await self.authenticate_client_credentials()
            else:
                # No refresh token, do full authentication
                await self.authenticate_client_credentials()

        if self._access_token is None:
            raise Exception("Failed to obtain a valid authentication token")

        return self._access_token

    async def request(
        self,
        method: str,
        url: str | httpx.URL,
        *,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> httpx.Response:
        """Send an HTTP request with added functionality.

        Extends the base request method with:
        - Structured logging

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

        # Log the request with Logfire
        logfire.debug(
            "HTTP request",
            http_method=method,
            url=str(url),
        )

        try:
            response = await super().request(method, url, headers=headers, **kwargs)

            # Log the response with Logfire
            logfire.debug(
                "HTTP response",
                status_code=response.status_code,
                url=str(url),
            )

            return response
        except httpx.RequestError as e:
            # Log the error with Logfire
            logfire.error(
                "HTTP request error",
                error=str(e),
                url=str(url),
                _exc_info=True,
            )
            raise

    async def _make_request(
        self,
        method: str,
        path: str,
        *,
        cast_to: type[T] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
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
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        # Check if path is a full URL when base_url is None
        if self._base_url is None and not path.startswith(("http://", "https://")):
            raise ValueError(
                "Either base_url must be provided during initialization or path must be a full URL"
            )

        # Handle path to prevent double slashes if it's not a full URL
        if not path.startswith(("http://", "https://")):
            path = path.lstrip("/")

        # Get authentication token
        token = await self._get_token()

        # Handle headers
        request_headers = headers or {}
        request_headers.update({"Authorization": f"Bearer {token}"})

        # Make the request
        try:
            response = await self.request(
                method, path, headers=request_headers, params=params, json=json, **kwargs
            )
        except httpx.UnsupportedProtocol as e:
            raise ValueError(
                "Either base_url must be provided during initialization or path must be a full URL"
            ) from e

        # Handle authentication failures
        if response.status_code == 401:
            # Token might be expired, try to refresh and retry once
            await self.refresh_access_token()
            request_headers.update({"Authorization": f"Bearer {self._access_token}"})
            response = await self.request(
                method, path, headers=request_headers, params=params, json=json, **kwargs
            )

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
    ) -> T | list[T]:
        ...

    @overload
    async def get(
        self,
        path: str,
        *,
        cast_to: None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    async def get(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a GET request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return await self._make_request(
            "GET", path, cast_to=cast_to, params=params, **kwargs
        )

    @overload
    async def post(
        self,
        path: str,
        *,
        cast_to: type[T],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T]:
        ...

    @overload
    async def post(
        self,
        path: str,
        *,
        cast_to: None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    async def post(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a POST request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return await self._make_request(
            "POST", path, cast_to=cast_to, json=json, **kwargs
        )

    @overload
    async def put(
        self,
        path: str,
        *,
        cast_to: type[T],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T]:
        ...

    @overload
    async def put(
        self,
        path: str,
        *,
        cast_to: None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    async def put(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a PUT request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return await self._make_request(
            "PUT", path, cast_to=cast_to, json=json, **kwargs
        )

    @overload
    async def patch(
        self,
        path: str,
        *,
        cast_to: type[T],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T]:
        ...

    @overload
    async def patch(
        self,
        path: str,
        *,
        cast_to: None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    async def patch(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a PATCH request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return await self._make_request(
            "PATCH", path, cast_to=cast_to, json=json, **kwargs
        )

    @overload
    async def delete(
        self,
        path: str,
        *,
        cast_to: type[T],
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T]:
        ...

    @overload
    async def delete(
        self,
        path: str,
        *,
        cast_to: None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    async def delete(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a DELETE request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return await self._make_request(
            "DELETE", path, cast_to=cast_to, params=params, **kwargs
        )


class SpryxSyncClient(SpryxClientBase, httpx.Client):
    """Spryx HTTP synchronous client with retry, tracing, and auth capabilities.

    Extends httpx.Client with:
    - OAuth 2.0 M2M authentication with refresh token support
    - Retry with exponential backoff
    - Structured logging with Logfire
    - Correlation ID propagation
    - Pydantic model response parsing
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_url: str,
        scope: str | None = None,
        settings: HttpClientSettings | None = None,
        **kwargs,
    ):
        """Initialize the Spryx HTTP sync client.

        Args:
            base_url: Base URL for all API requests. Can be None.
            client_id: OAuth 2.0 client ID for M2M authentication.
            client_secret: OAuth 2.0 client secret for M2M authentication.
            token_url: OAuth 2.0 token endpoint URL.
            scope: Optional OAuth 2.0 scope for the access token.
            settings: HTTP client settings.
            **kwargs: Additional arguments to pass to httpx.Client.
        """
        # Initialize base class
        SpryxClientBase.__init__(
            self,
            base_url=base_url,
            client_id=client_id,
            client_secret=client_secret,
            token_url=token_url,
            scope=scope,
            settings=settings,
            **kwargs,
        )

        # Initialize httpx.Client with sync transport
        transport_kwargs = self._get_sync_transport_kwargs(**self._httpx_kwargs)
        # Pass empty string instead of None to httpx.Client
        httpx_base_url = "" if self._base_url is None else self._base_url
        httpx.Client.__init__(self, base_url=httpx_base_url, **transport_kwargs)

    def _get_sync_transport_kwargs(self, **kwargs):
        """Get sync transport configuration for the client."""
        # Configure retry transport if not provided
        if "transport" not in kwargs:
            kwargs["transport"] = build_retry_transport(
                settings=self.settings, is_async=False
            )
        return kwargs

    def authenticate_client_credentials(self) -> str:
        """Authenticate using OAuth 2.0 Client Credentials flow.

        Uses the client_id and client_secret provided during initialization
        to authenticate with the OAuth 2.0 token endpoint and obtain access and refresh tokens.

        Returns:
            str: The access token.

        Raises:
            ValueError: If client_id or client_secret is not provided.
            httpx.HTTPStatusError: If the token request fails.
        """
        if self._client_id is None:
            raise ValueError("client_id is required for OAuth 2.0 authentication")

        if self._client_secret is None:
            raise ValueError("client_secret is required for OAuth 2.0 authentication")

        payload = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }

        if self._scope:
            payload["scope"] = self._scope

        response = self.request("POST", self._token_url, data=payload)
        response.raise_for_status()

        token_data = response.json()

        self._access_token = token_data["access_token"]
        self._refresh_token = token_data.get("refresh_token")

        # Parse JWT to get expiration time
        if self._access_token:
            self._token_expires_at = self._parse_jwt_expiration(self._access_token)

        if self._access_token is None:
            raise ValueError("Failed to obtain access token")
        return self._access_token

    def refresh_access_token(self) -> str:
        """Refresh the access token using the refresh token.

        This method attempts to use the refresh token to get a new access token
        without requiring full re-authentication. If the refresh token has expired
        or the refresh fails, it falls back to full client credentials authentication.

        Returns:
            str: The new access token.

        Raises:
            ValueError: If no refresh token is available and client credentials fail.
        """
        if self._refresh_token is None:
            # No refresh token available, do full authentication
            return self.authenticate_client_credentials()

        try:
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            }

            response = self.request("POST", self._token_url, data=payload)
            response.raise_for_status()

            token_data = response.json()

            self._access_token = token_data["access_token"]
            # Some implementations return a new refresh token
            if "refresh_token" in token_data:
                self._refresh_token = token_data["refresh_token"]

            # Parse JWT to get expiration time
            if self._access_token:
                self._token_expires_at = self._parse_jwt_expiration(self._access_token)

            if self._access_token is None:
                raise ValueError("Failed to obtain access token")
            return self._access_token

        except (httpx.HTTPStatusError, httpx.RequestError, ValueError, KeyError):
            # Refresh failed, fall back to full authentication
            logfire.debug("Refresh token failed, falling back to client credentials authentication")
            return self.authenticate_client_credentials()

    def _get_token(self) -> str:
        """Get a valid authentication token.

        This method handles token lifecycle management, including:
        - Initial authentication if no token exists
        - Token refresh if access token has expired
        - Fallback to full re-authentication if refresh fails

        Returns:
            str: A valid authentication token.

        Raises:
            Exception: If unable to obtain a valid token.
        """
        if self._access_token is None or self.is_token_expired:
            if self._refresh_token is not None:
                # Try to refresh first
                try:
                    self.refresh_access_token()
                except Exception:
                    # If refresh fails, do full authentication
                    self.authenticate_client_credentials()
            else:
                # No refresh token, do full authentication
                self.authenticate_client_credentials()

        if self._access_token is None:
            raise Exception("Failed to obtain a valid authentication token")

        return self._access_token

    def request(
        self,
        method: str,
        url: str | httpx.URL,
        *,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> httpx.Response:
        """Send an HTTP request with added functionality.

        Extends the base request method with:
        - Structured logging

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

        # Log the request with Logfire
        logfire.debug(
            "HTTP request",
            http_method=method,
            url=str(url),
        )

        try:
            response = super().request(method, url, headers=headers, **kwargs)

            # Log the response with Logfire
            logfire.debug(
                "HTTP response",
                status_code=response.status_code,
                url=str(url),
            )

            return response
        except httpx.RequestError as e:
            # Log the error with Logfire
            logfire.error(
                "HTTP request error",
                error=str(e),
                url=str(url),
                _exc_info=True,
            )
            raise

    def _make_request(
        self,
        method: str,
        path: str,
        *,
        cast_to: type[T] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
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
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        # Check if path is a full URL when base_url is None
        if self._base_url is None and not path.startswith(("http://", "https://")):
            raise ValueError(
                "Either base_url must be provided during initialization or path must be a full URL"
            )

        # Handle path to prevent double slashes if it's not a full URL
        if not path.startswith(("http://", "https://")):
            path = path.lstrip("/")

        # Get authentication token
        token = self._get_token()

        # Handle headers
        request_headers = headers or {}
        request_headers.update({"Authorization": f"Bearer {token}"})

        # Make the request
        try:
            response = self.request(
                method, path, headers=request_headers, params=params, json=json, **kwargs
            )
        except httpx.UnsupportedProtocol as e:
            raise ValueError(
                "Either base_url must be provided during initialization or path must be a full URL"
            ) from e

        # Handle authentication failures
        if response.status_code == 401:
            # Token might be expired, try to refresh and retry once
            self.refresh_access_token()
            request_headers.update({"Authorization": f"Bearer {self._access_token}"})
            response = self.request(
                method, path, headers=request_headers, params=params, json=json, **kwargs
            )

        # Process the response
        return self._process_response_data(response, cast_to)

    # HTTP method overloads for proper type inference
    @overload
    def get(
        self,
        path: str,
        *,
        cast_to: type[T],
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T]:
        ...

    @overload
    def get(
        self,
        path: str,
        *,
        cast_to: None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    def get(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a GET request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return self._make_request("GET", path, cast_to=cast_to, params=params, **kwargs)

    @overload
    def post(
        self,
        path: str,
        *,
        cast_to: type[T],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T]:
        ...

    @overload
    def post(
        self,
        path: str,
        *,
        cast_to: None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    def post(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a POST request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return self._make_request("POST", path, cast_to=cast_to, json=json, **kwargs)

    @overload
    def put(
        self,
        path: str,
        *,
        cast_to: type[T],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T]:
        ...

    @overload
    def put(
        self,
        path: str,
        *,
        cast_to: None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    def put(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a PUT request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return self._make_request("PUT", path, cast_to=cast_to, json=json, **kwargs)

    @overload
    def patch(
        self,
        path: str,
        *,
        cast_to: type[T],
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T]:
        ...

    @overload
    def patch(
        self,
        path: str,
        *,
        cast_to: None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    def patch(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a PATCH request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return self._make_request("PATCH", path, cast_to=cast_to, json=json, **kwargs)

    @overload
    def delete(
        self,
        path: str,
        *,
        cast_to: type[T],
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T]:
        ...

    @overload
    def delete(
        self,
        path: str,
        *,
        cast_to: None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> ResponseJson:
        ...

    def delete(
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> T | list[T] | ResponseJson:
        """Send a DELETE request.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Optional Pydantic model class to parse response into.
                    If None, returns the raw JSON data.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T], ResponseJson]: Pydantic model instance(s) or raw JSON data.
        """
        return self._make_request(
            "DELETE", path, cast_to=cast_to, params=params, **kwargs
        )

