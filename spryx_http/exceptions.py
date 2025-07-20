"""Base exceptions for the Spryx application.

All custom exceptions across slices should inherit from SpryxException
to ensure consistent error handling and HTTP status code mapping.
"""

from collections.abc import Mapping
from typing import Any

import httpx

ResponseJson = Mapping[str, Any]


class SpryxRequestException(Exception):
    def __init__(self, response: httpx.Response, response_json: ResponseJson | None) -> None:
        super().__init__(response_json)

        self.response = response
        self.response_json = response_json

        self.message = self.extract_from_json("message", "No message")
        self.code = self.extract_from_json("code", None)
        self.details = self.extract_from_json("details", None)
        self.status_code = response.status_code

    def extract_from_json(self, key: str, alt: str | None = None) -> str | None:
        if self.response_json is None:
            return alt

        return self.response_json.get(key, alt)

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

    def __str__(self) -> str:
        exception = f"(message={self.message}"

        if self.code is not None:
            exception += f", code={self.code}"

        if self.status_code is not None:
            exception += f", status_code={self.status_code}"

        if self.response_json is not None:
            for key, value in self.response_json.items():
                if key not in ["message", "code", "status_code", "details"]:
                    exception += f", {key}={value}"

        return exception + ")"


class BadRequestError(SpryxRequestException):
    """Error for 4xx status codes."""

    pass


class ServerError(SpryxRequestException):
    """Error for 5xx status codes."""

    pass


class RateLimitError(BadRequestError):
    """Error for 429 status code (Too Many Requests)."""

    pass


class AuthenticationError(BadRequestError):
    """Error for 401 status code (Unauthorized)."""

    pass


class AuthorizationError(BadRequestError):
    """Error for 403 status code (Forbidden)."""

    pass


class NotFoundError(BadRequestError):
    """Error for 404 status code (Not Found)."""

    pass


class ConflictError(BadRequestError):
    """Error for 409 status code (Conflict)."""

    pass
