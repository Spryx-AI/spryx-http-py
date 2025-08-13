from typing import Any, TypeVar, cast

import httpx
from pydantic import BaseModel, TypeAdapter
from spryx_core import NotGiven

from spryx_http.auth_strategies import AuthStrategy
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

T = TypeVar("T", bound=BaseModel)


class SpryxClientBase:
    """Base class for Spryx HTTP clients with common functionality.

    Contains shared functionality between async and sync clients:
    - Authentication strategy pattern support
    - Response data processing
    - Settings management
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        auth_strategy: AuthStrategy,
        settings: HttpClientSettings | None = None,
        **kwargs,
    ):
        """Initialize the base Spryx HTTP client.

        Args:
            base_url: Base URL for all API requests. Can be None.
            auth_strategy: Authentication strategy to use (ClientCredentials or ApiKey).
            settings: HTTP client settings.
            **kwargs: Additional arguments to pass to httpx client.
        """
        self._base_url = base_url
        self._auth_strategy = auth_strategy
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

    def _extract_data_from_response(self, response_json: ResponseJson | None) -> Any:
        """Extract data from API response.

        Returns the response data as-is without extracting nested fields.

        Args:
            response_data: The response data dictionary.

        Returns:
            Any: The response data.
        """
        if response_json is None:
            return None

        if isinstance(response_json, dict):
            return response_json.get("data", response_json)

        return response_json


    def _parse_model_data(self, model_cls: type[T] | type[list[T]] | TypeAdapter, data: Any) -> T | list[T] | Any:
        """Parse data into a Pydantic model, list of models, or using TypeAdapter.

        Args:
            model_cls: The Pydantic model class, list type, or TypeAdapter to parse with.
            data: The data to parse.

        Returns:
            T | list[T] | Any: Parsed data.
        """
        # Handle TypeAdapter
        if isinstance(model_cls, TypeAdapter):
            return model_cls.validate_python(data)
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
        self, response: httpx.Response, cast_to: type[T] | type[list[T]] | TypeAdapter | None = None
    ) -> T | list[T] | Any | ResponseJson:
        """Process the response by validating status and converting to model.

        Args:
            response: The HTTP response.
            cast_to: Optional Pydantic model class, list of models, or TypeAdapter to parse response into.
                     If None, returns the raw JSON data.

        Returns:
            T | list[T] | Any | ResponseJson: Parsed data or raw JSON data.
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
