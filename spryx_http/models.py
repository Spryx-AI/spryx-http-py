from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ResponseV1(BaseModel, Generic[T]):
    """Base response model for all API v1 endpoints."""

    data: T | list[T]
    message: str = "success"
    metadata: dict[str, Any] | None = None
