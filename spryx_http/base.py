import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

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

    _token: Optional[str] = None
    _token_expires_at: Optional[int] = None
    _refresh_token: Optional[str] = None
    _refresh_token_expires_at: Optional[int] = None

    def __init__(
        self,
        *,
        base_url: str,
        application_id: Optional[str] = None,
        application_secret: Optional[str] = None,
        auth_strategy: Optional[AuthStrategy] = None,
        settings: Optional[HttpClientSettings] = None,
        **kwargs,
    ):
        """Initialize the Spryx HTTP async client.

        Args:
            base_url: Base URL for all API requests.
            application_id: Application ID for authentication.
            application_secret: Application secret for authentication.
            auth_strategy: Authentication strategy to use.
            settings: HTTP client settings.
            **kwargs: Additional arguments to pass to httpx.AsyncClient.
        """
        self._base_url = base_url

        self._application_id = application_id
        self._application_secret = application_secret

        self.auth_strategy = auth_strategy or NoAuth()
        self.settings = settings or get_http_settings()

        # Configure timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.settings.timeout_s

        # Configure retry transport if not provided
        if "transport" not in kwargs:
            kwargs["transport"] = build_retry_transport(settings=self.settings)

        self._token_expires_at = None

        super().__init__(**kwargs)

    @property
    def is_token_expired(self) -> bool:
        """Check if the access token is expired.

        Returns:
            bool: True if the token is expired or not set, False otherwise.
        """
        if self._token is None or self._token_expires_at is None:
            return True

        current_time = int(time.time())
        return current_time >= self._token_expires_at

    @property
    def is_refresh_token_expired(self) -> bool:
        """Check if the refresh token is expired.

        Returns:
            bool: True if the refresh token is expired or not set, False otherwise.
        """
        if self._refresh_token is None or self._refresh_token_expires_at is None:
            return True

        current_time = int(time.time())
        return current_time >= self._refresh_token_expires_at

    async def authenticate_application(self) -> None:
        """Authenticate the application with credentials provided in the constructor.

        Uses the application_id and application_secret provided during initialization
        to authenticate with the API and obtain access and refresh tokens.

        Raises:
            ValueError: If application_id or application_secret is not provided.
        """
        if self._application_id is None:
            raise ValueError("application_id is required")

        if self._application_secret is None:
            raise ValueError("application_secret is required")

        payload = {
            "application_id": self._application_id,
            "application_secret": self._application_secret,
        }
        response = await self.request(
            "POST", f"{self._base_url}/v1/auth/application", json=payload
        )
        json_response = response.json()
        data = json_response.get("data", {})

        self._token_expires_at = data.get("exp")
        self._token = json_response.get("access_token")
        self._refresh_token = json_response.get("refresh_token")
        self._refresh_token_expires_at = data.get("refresh_token_exp")

    async def _generate_new_token(self):
        """Generate a new access token using the refresh token.

        This method is called automatically when the access token expires
        but the refresh token is still valid.

        Raises:
            ValueError: If refresh token is not available.
        """
        if self._refresh_token is None:
            raise ValueError(
                "Refresh token is not available. Please authenticate with authenticate_application() first."
            )

        payload = {"refresh_token": self._refresh_token}

        response = await self.request(
            "POST",
            url=f"{self._base_url}/v1/auth/application/refresh-token",
            json=payload,
        )

        json_response = response.json()
        data = json_response.get("data")

        self._token_expires_at = data.get("exp")
        self._token = json_response.get("access_token")

    async def _get_token(self) -> str:
        """Get a valid authentication token.

        This method handles token lifecycle management, including:
        - Initial authentication if no token exists
        - Re-authentication if refresh token has expired
        - Token refresh if access token has expired but refresh token is valid

        Returns:
            str: A valid authentication token.

        Raises:
            Exception: If unable to obtain a valid token.
        """
        if self._token is None:
            await self.authenticate_application()
            if self._token is None:
                raise Exception(
                    "Failed to obtain a valid authentication token. Authentication did not provide a token."
                )

        if self.is_refresh_token_expired:
            await self.authenticate_application()
            if self._token is None:
                raise Exception(
                    "Failed to obtain a valid authentication token. Re-authentication did not provide a token."
                )
            return self._token

        if self.is_token_expired:
            await self._generate_new_token()
            if self._token is None:
                raise Exception(
                    "Failed to obtain a valid authentication token. Token refresh did not provide a token."
                )

        # At this point, we've done all we can to get a valid token
        # If it's still None, raise an exception
        if self._token is None:
            raise Exception(
                "Failed to obtain a valid authentication token. Check your credentials and try again."
            )

        return self._token

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
        path: str,
        *,
        cast_to: Type[T],
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]:
        """Core request method to handle HTTP requests with Pydantic model parsing.

        This method handles all HTTP requests and parses the response
        into the provided Pydantic model.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE).
            path: Request path to be appended to base_url.
            cast_to: Pydantic model class to parse response into.
            params: Optional query parameters.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T]]: Pydantic model instance or list of instances

        Raises:
            HttpError: If the response status code is 4xx or 5xx.
            ValueError: If the response cannot be parsed.
        """
        url = f"{self.base_url}/{path}"

        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        # Make the request using the request method to ensure auth headers are added
        response = await self.request(
            method, url, params=params, headers=headers, json=json, **kwargs
        )

        # Raise exception for error status codes
        raise_for_status(response)

        # Parse JSON response
        json_data = response.json()

        # Extract data from standard response format and parse into model
        data = self._extract_data_from_response(json_data)
        return self._parse_model_data(cast_to, data)

    async def get(
        self,
        path: str,
        *,
        cast_to: Type[T],
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]:
        """Send a GET request and parse the response into a Pydantic model.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Pydantic model class to parse response into.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T]]: Pydantic model instance or list of instances
        """
        return await self._make_request(
            "GET", path, cast_to=cast_to, params=params, **kwargs
        )

    async def post(
        self,
        path: str,
        *,
        cast_to: Type[T],
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]:
        """Send a POST request and parse the response into a Pydantic model.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Pydantic model class to parse response into.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T]]: Pydantic model instance or list of instances
        """
        return await self._make_request(
            "POST", path, cast_to=cast_to, json=json, **kwargs
        )

    async def put(
        self,
        path: str,
        *,
        cast_to: Type[T],
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]:
        """Send a PUT request and parse the response into a Pydantic model.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Pydantic model class to parse response into.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T]]: Pydantic model instance or list of instances
        """
        return await self._make_request(
            "PUT", path, cast_to=cast_to, json=json, **kwargs
        )

    async def patch(
        self,
        path: str,
        *,
        cast_to: Type[T],
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]:
        """Send a PATCH request and parse the response into a Pydantic model.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Pydantic model class to parse response into.
            json: Optional JSON data for the request body.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T]]: Pydantic model instance or list of instances
        """
        return await self._make_request(
            "PATCH", path, cast_to=cast_to, json=json, **kwargs
        )

    async def delete(
        self,
        path: str,
        *,
        cast_to: Type[T],
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[T, List[T]]:
        """Send a DELETE request and parse the response into a Pydantic model.

        Args:
            path: Request path to be appended to base_url.
            cast_to: Pydantic model class to parse response into.
            params: Optional query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Union[T, List[T]]: Pydantic model instance or list of instances
        """
        return await self._make_request(
            "DELETE", path, cast_to=cast_to, params=params, **kwargs
        )
