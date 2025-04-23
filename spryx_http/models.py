from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ResponseV1(BaseModel, Generic[T]):
    """Base response model for all API v1 endpoints."""

    data: T | List[T]
    message: str = "success"
    metadata: Optional[Dict[str, Any]] = None
