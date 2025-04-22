from typing import Dict, Optional, Type

import httpx


class HttpError(Exception):
    """Base class for HTTP client errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[httpx.Response] = None,
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
STATUS_CODE_TO_EXCEPTION: Dict[int, Type[HttpError]] = {
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
        HttpError: An exception corresponding to the response status code.
    """
    if 400 <= response.status_code < 600:
        error_cls = STATUS_CODE_TO_EXCEPTION.get(
            response.status_code,
            ClientError if response.status_code < 500 else ServerError,
        )

        # Try to extract error details from response body
        error_message = f"HTTP Error {response.status_code}"
        try:
            response_json = response.json()
            if isinstance(response_json, dict):
                error_detail = response_json.get("detail") or response_json.get(
                    "message"
                )
                if error_detail:
                    error_message = f"{error_message}: {error_detail}"
        except Exception:
            pass

        raise error_cls(
            message=error_message, status_code=response.status_code, response=response
        )
