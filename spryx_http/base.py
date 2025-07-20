import time
from typing import Any, TypeVar, cast

import httpx
from pydantic import BaseModel
from spryx_core import NotGiven

from spryx_http.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    RateLimitError,
    ResponseJson,
    ServerError,
)
from spryx_http.retry import build_retry_transport
from spryx_http.settings import HttpClientSettings, get_http_settings
from spryx_http.types import OAuthTokenResponse

T = TypeVar("T", bound=BaseModel)


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
        settings: HttpClientSettings | None = None,
        **kwargs,
    ):
        """Initialize the base Spryx HTTP client.

        Args:
            base_url: Base URL for all API requests. Can be None.
            client_id: OAuth 2.0 client ID for M2M authentication.
            client_secret: OAuth 2.0 client secret for M2M authentication.
            token_url: OAuth 2.0 token endpoint URL.
            settings: HTTP client settings.
            **kwargs: Additional arguments to pass to httpx client.
        """
        self._base_url = base_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
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
            kwargs["transport"] = build_retry_transport(settings=self.settings, is_async=True)
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

    def _store_token_data(self, oauth_token_response: OAuthTokenResponse) -> str:
        """Store token data and parse expiration time.

        This method is responsible for:
        - Validating the OAuth token response format
        - Extracting and storing the access token
        - Extracting and storing the refresh token (if present)
        - Calculating expiration timestamp from expires_in

        Args:
            token_data: OAuth token response from server, either as dict or parsed model.
                       Must contain 'access_token' and optionally 'refresh_token'.

        Returns:
            str: The access token.

        Raises:
            ValueError: If access_token is missing or validation fails.
            ValidationError: If token_data doesn't match expected OAuth format.
        """
        # Store tokens
        self._access_token = oauth_token_response.access_token
        self._refresh_token = oauth_token_response.refresh_token

        # Calculate expiration time from expires_in
        self._token_expires_at = int(time.time()) + oauth_token_response.expires_in

        # Validate we have a valid access token
        if not self._access_token:
            raise ValueError("Failed to obtain access token")

        return self._access_token

    def _extract_data_from_response(self, response_json: ResponseJson | None) -> Any:
        """Extract data from standardized API response.

        In our standardized API response, the actual entity is always under a 'data' key.

        Args:
            response_data: The response data dictionary.

        Returns:
            Any: The extracted data.
        """
        if response_json is not None and "data" in response_json:
            return response_json["data"]
        return response_json

    def _parse_model_data(self, model_cls: type[T] | type[list[T]], data: Any) -> T | list[T]:
        """Parse data into a Pydantic model or list of models.

        Args:
            model_cls: The Pydantic model class or list type to parse into.
            data: The data to parse.

        Returns:
            T | list[T]: Parsed model instance or list of instances.
        """
        # Check if it's a list type by string representation (works for typing generics)
        if str(model_cls).startswith("typing.Generic") or "list[" in str(model_cls):
            # Extract the inner type from list[T] using typing inspection
            import typing

            if hasattr(typing, "get_args") and hasattr(typing, "get_origin"):
                origin = typing.get_origin(model_cls)
                if origin is list:
                    inner_type = typing.get_args(model_cls)[0]
                    return self._parse_model_list_data(inner_type, data)
            raise ValueError("Could not extract inner type from list type")

        # Regular single model parsing
        if isinstance(model_cls, type) and issubclass(model_cls, BaseModel):
            return model_cls.model_validate(data)
        else:
            raise ValueError("Invalid model class")

    def _parse_model_list_data(self, model_cls: type[T], data: Any) -> list[T]:
        """Parse data into a list of Pydantic models.

        Args:
            model_cls: The Pydantic model class to parse into.
            data: The data to parse (must be a list).

        Returns:
            list[T]: List of parsed model instances.
        """
        if not isinstance(data, list):
            raise ValueError("Expected list data for list parsing")
        return [model_cls.model_validate(item) for item in data]

    def _process_response_data(
        self, response: httpx.Response, cast_to: type[T] | type[list[T]] | None = None
    ) -> T | list[T] | ResponseJson:
        """Process the response by validating status and converting to model.

        Args:
            response: The HTTP response.
            cast_to: Optional Pydantic model class to parse response into.
                     If None, returns the raw JSON data.

        Returns:
            T | ResponseJson: Pydantic model instance or raw JSON data.
        """
        response_json = None
        content_type = response.headers.get("content-type") if response.headers is not None else None
        if content_type is not None and "application/json" in content_type:
            try:
                response_json = response.json()
            except ValueError as e:
                raise ServerError(response, None) from e

        self._maybe_raise_error_by_status_code(response, response_json)

        # Extract data from standard response format
        data = self._extract_data_from_response(response_json)

        # If cast_to is provided, parse into model, otherwise return the raw data
        if cast_to is not None:
            return self._parse_model_data(cast_to, data)

        return cast(ResponseJson, response_json)

    def _maybe_raise_error_by_status_code(self, response: httpx.Response, response_json: ResponseJson | None) -> None:
        """Raise appropriate HTTP error based on status code.

        Args:
            response: The HTTP response object
            response_json: Parsed JSON response or None

        Raises:
            HttpError: Appropriate exception based on status code
        """
        status_code = response.status_code

        if status_code >= 400 and status_code < 500:
            if status_code == 401:
                raise AuthenticationError(response=response, response_json=response_json)
            elif status_code == 403:
                raise AuthorizationError(response=response, response_json=response_json)
            elif status_code == 404:
                raise NotFoundError(response=response, response_json=response_json)
            elif status_code == 409:
                raise ConflictError(response=response, response_json=response_json)
            elif status_code == 429:
                raise RateLimitError(response=response, response_json=response_json)

            raise BadRequestError(response=response, response_json=response_json)
        elif status_code >= 500 and status_code < 600:
            raise ServerError(response=response, response_json=response_json)

    def _remove_not_given(self, kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
        if kwargs is None:
            return None
        return {k: v for k, v in kwargs.items() if v != NotGiven}
