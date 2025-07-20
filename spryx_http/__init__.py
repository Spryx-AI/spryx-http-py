__version__ = "0.1.1"

from spryx_http.base import SpryxAsyncClient, SpryxSyncClient
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

__all__ = [
    "SpryxAsyncClient",
    "SpryxSyncClient",
    "HttpError",
    "ClientError",
    "ServerError",
    "RateLimitError",
    "AuthenticationError",
    "ForbiddenError",
    "NotFoundError",
    "raise_for_status",
]
