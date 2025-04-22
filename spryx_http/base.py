from typing import Any, Dict, List, Optional, Type, TypeVar, Union, overload

import httpx
import logfire
from pydantic import BaseModel

from spryx_http.auth import AuthStrategy, NoAuth
from spryx_http.exceptions import raise_for_status
from spryx_http.retry import build_retry_transport
from spryx_http.settings import HttpClientSettings, get_http_settings

T = TypeVar("T", bound=BaseModel)


class SpryxAsyncClient(httpx.AsyncClient):
    """Spryx HTTP async client with retry, tracing, and auth capabilities.

    Extends httpx.AsyncClient with:
    - Retry with exponential backoff
    - Authentication via pluggable strategies
    - Structured logging with Logfire
    - Correlation ID propagation
    - Pydantic model response parsing
    """

    def __init__(
        self,
        *,
        auth_strategy: Optional[AuthStrategy] = None,
        settings: Optional[HttpClientSettings] = None,
        **kwargs,
    ):
        """Initialize the client.

        Args:
            auth_strategy: Authentication strategy to use.
            settings: HTTP client settings.
            **kwargs: Additional arguments to pass to httpx.AsyncClient.
        """
        self.auth_strategy = auth_strategy or NoAuth()
        self.settings = settings or get_http_settings()

        # Configure timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.settings.timeout_s

        # Configure retry transport if not provided
        if "transport" not in kwargs:
            kwargs["transport"] = build_retry_transport(settings=self.settings)

        super().__init__(**kwargs)

    async def request(
        self,
        method: str,
        url: Union[str, httpx.URL],
        *,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Send an HTTP request with added functionality.

        Extends the base request method with:
        - Adding authentication headers
        - Adding correlation ID
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

        # Add authentication headers
        auth_headers = self.auth_strategy.headers()
        headers.update(auth_headers)

        # Add correlation ID header if available
        correlation_id = logfire.get_context().get("correlation_id")
        if correlation_id:
            headers["x-correlation-id"] = correlation_id

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

    def _extract_data_from_response(self, response_data: Dict[str, Any]) -> Any:
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

    def _parse_model_data(self, model_cls: Type[T], data: Any) -> Union[T, List[T]]:
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

    async def _make_request(
        self,
        method: str,
        url: str,
        *,
        model_cls: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], T, List[T]]:
        """Core request method to handle HTTP requests with optional Pydantic model parsing.

        This method handles all HTTP requests and decides whether to parse the response
        as a Pydantic model based on whether model_cls is provided.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE).
            url: Request URL.
            model_cls: Optional Pydantic model class to parse response into.
            params: Optional query parameters.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[Dict[str, Any], T, List[T]]:
                - Dict if no model_cls is provided
                - Pydantic model instance or list of instances if model_cls is provided

        Raises:
            HttpError: If the response status code is 4xx or 5xx.
            ValueError: If the response cannot be parsed.
        """
        # Make the request using the request method to ensure auth headers are added
        response = await self.request(method, url, params=params, json=json, **kwargs)

        # Raise exception for error status codes
        raise_for_status(response)

        # Parse JSON response
        json_data = response.json()

        # If no model class is provided, return the raw JSON data
        if model_cls is None:
            return json_data

        # Extract data from standard response format and parse into model
        data = self._extract_data_from_response(json_data)
        return self._parse_model_data(model_cls, data)

    @overload
    async def get(
        self,
        url: str,
        *,
        model: Type[T],
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]: ...

    @overload
    async def get(
        self, url: str, *, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]: ...

    async def get(
        self,
        url: str,
        *,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], T, List[T]]:
        """Send a GET request and optionally parse the response into a Pydantic model.

        Args:
            url: Request URL.
            model: Optional Pydantic model class to parse response into.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[Dict[str, Any], T, List[T]]:
                - Dict if no model is provided
                - Pydantic model instance or list of instances if model is provided
        """
        return await self._make_request(
            "GET", url, model_cls=model, params=params, **kwargs
        )

    @overload
    async def post(
        self,
        url: str,
        *,
        model: Type[T],
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]: ...

    @overload
    async def post(
        self, url: str, *, json: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]: ...

    async def post(
        self,
        url: str,
        *,
        model: Optional[Type[T]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], T, List[T]]:
        """Send a POST request and optionally parse the response into a Pydantic model.

        Args:
            url: Request URL.
            model: Optional Pydantic model class to parse response into.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[Dict[str, Any], T, List[T]]:
                - Dict if no model is provided
                - Pydantic model instance or list of instances if model is provided
        """
        return await self._make_request(
            "POST", url, model_cls=model, json=json, **kwargs
        )

    @overload
    async def put(
        self,
        url: str,
        *,
        model: Type[T],
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]: ...

    @overload
    async def put(
        self, url: str, *, json: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]: ...

    async def put(
        self,
        url: str,
        *,
        model: Optional[Type[T]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], T, List[T]]:
        """Send a PUT request and optionally parse the response into a Pydantic model.

        Args:
            url: Request URL.
            model: Optional Pydantic model class to parse response into.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[Dict[str, Any], T, List[T]]:
                - Dict if no model is provided
                - Pydantic model instance or list of instances if model is provided
        """
        return await self._make_request(
            "PUT", url, model_cls=model, json=json, **kwargs
        )

    @overload
    async def patch(
        self,
        url: str,
        *,
        model: Type[T],
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]: ...

    @overload
    async def patch(
        self, url: str, *, json: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]: ...

    async def patch(
        self,
        url: str,
        *,
        model: Optional[Type[T]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], T, List[T]]:
        """Send a PATCH request and optionally parse the response into a Pydantic model.

        Args:
            url: Request URL.
            model: Optional Pydantic model class to parse response into.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[Dict[str, Any], T, List[T]]:
                - Dict if no model is provided
                - Pydantic model instance or list of instances if model is provided
        """
        return await self._make_request(
            "PATCH", url, model_cls=model, json=json, **kwargs
        )

    @overload
    async def delete(
        self,
        url: str,
        *,
        model: Type[T],
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]: ...

    @overload
    async def delete(
        self, url: str, *, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]: ...

    async def delete(
        self,
        url: str,
        *,
        model: Optional[Type[T]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], T, List[T]]:
        """Send a DELETE request and optionally parse the response into a Pydantic model.

        Args:
            url: Request URL.
            model: Optional Pydantic model class to parse response into.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[Dict[str, Any], T, List[T]]:
                - Dict if no model is provided
                - Pydantic model instance or list of instances if model is provided
        """
        return await self._make_request(
            "DELETE", url, model_cls=model, params=params, **kwargs
        )
