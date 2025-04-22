__version__ = "0.1.1"

from spryx_http.auth import AuthStrategy, HmacAuth, JwtAuth, NoAuth
from spryx_http.base import SpryxAsyncClient, T
from spryx_http.exceptions import (
    AuthenticationError,
    ClientError,
    ForbiddenError,
    HttpError,
    NotFoundError,
    RateLimitError,
    ServerError,
    raise_for_status,
)
from spryx_http.pagination import AsyncPaginator

__all__ = [
    "SpryxAsyncClient",
    "T",  # TypeVar for generic type hints
    "AuthStrategy",
    "JwtAuth",
    "HmacAuth",
    "NoAuth",
    "HttpError",
    "ClientError",
    "ServerError",
    "RateLimitError",
    "AuthenticationError",
    "ForbiddenError",
    "NotFoundError",
    "raise_for_status",
    "AsyncPaginator",
]
