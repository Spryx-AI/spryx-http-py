"""Base exceptions for the Spryx application.

All custom exceptions across slices should inherit from SpryxException
to ensure consistent error handling and HTTP status code mapping.
"""

from typing import Any, TypeAlias

import httpx

ExcDetails: TypeAlias = dict[str, Any] | None


class SpryxException(Exception):
    """Base exception class for all Spryx application exceptions.

    Provides a consistent interface for error handling with:
    - message: Human-readable error description
    - code: Unique identifier for programmatic error handling
    - details: Optional additional context information
    - status_code: HTTP status code for API responses
    """

    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 400,
        details: ExcDetails | None = None,
    ):
        """Initialize the Spryx exception.

        Args:
            message: Human-readable error message
            code: Unique error code for identification
            status_code: HTTP status code (default: 400)
            details: Optional additional error context
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"code='{self.code}', "
            f"status_code={self.status_code}, "
            f"details={self.details})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "message": self.message,
            "code": self.code,
            "details": self.details,
            "status_code": self.status_code,
        }


class HttpError(Exception):
    """Base class for HTTP client errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: httpx.Response | None = None,
    ):
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class ClientError(HttpError):
    """Error for 4xx status codes."""

    pass


class ServerError(HttpError):
    """Error for 5xx status codes."""

    pass


class RateLimitError(ClientError):
    """Error for 429 status code (Too Many Requests)."""

    pass


class AuthenticationError(ClientError):
    """Error for 401 status code (Unauthorized)."""

    pass


class ForbiddenError(ClientError):
    """Error for 403 status code (Forbidden)."""

    pass


class NotFoundError(ClientError):
    """Error for 404 status code (Not Found)."""

    pass


# Mapping of status codes to exception classes
STATUS_CODE_TO_EXCEPTION: dict[int, type[HttpError]] = {
    400: ClientError,
    401: AuthenticationError,
    403: ForbiddenError,
    404: NotFoundError,
    429: RateLimitError,
    500: ServerError,
    502: ServerError,
    503: ServerError,
    504: ServerError,
}


def raise_for_status(response: httpx.Response) -> None:
    """Raise an exception if the response status code is 4xx or 5xx.

    Args:
        response: The HTTP response object.

    Raises:
        SpryxException: If the response contains a Spryx-formatted error.
        HttpError: An exception corresponding to the response status code.
    """
    if 400 <= response.status_code < 600:
        # Try to parse the response as JSON and check for Spryx error format
        try:
            response_json = response.json()
            if isinstance(response_json, dict) and "error" in response_json:
                error_data = response_json["error"]

                # Check if it's a Spryx-formatted error
                if isinstance(error_data, dict) and all(
                    key in error_data for key in ["message", "code", "status_code"]
                ):
                    # Create and raise SpryxException
                    raise SpryxException(
                        message=error_data["message"],
                        code=error_data["code"],
                        status_code=error_data["status_code"],
                        details=error_data.get("details"),
                    )
        except (ValueError, KeyError):
            # If JSON parsing fails or required keys are missing, fall back to default handling
            pass

        # Default error handling for non-Spryx errors
        error_cls = STATUS_CODE_TO_EXCEPTION.get(
            response.status_code,
            ClientError if response.status_code < 500 else ServerError,
        )

        # Try to extract error details from response body
        error_message = f"HTTP Error {response.status_code}"
        try:
            response_json = response.json()
            if isinstance(response_json, dict):
                error_detail = response_json.get("detail") or response_json.get("message")
                if error_detail:
                    error_message = f"{error_message}: {error_detail}"
        except Exception:
            pass

        raise error_cls(message=error_message, status_code=response.status_code, response=response)
