"""Data models and type definitions for Spryx HTTP client.

This module contains Pydantic models and type definitions used throughout
the Spryx HTTP client library for data validation and type safety.
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseV1(BaseModel, Generic[T]):
    """Base response model for all API v1 endpoints."""

    data: T
    message: str = "success"
    metadata: dict[str, Any] | None = None


class OAuthTokenResponse(BaseModel):
    """OAuth 2.0 token response model.

    Represents the standardized response from an OAuth 2.0 token endpoint
    for both client_credentials and refresh_token grant types.

    Reference: RFC 6749 - Section 5.1 (Successful Response)
    """

    access_token: str = Field(description="The access token issued by the authorization server")
    token_type: str = Field(default="Bearer", description="The type of the token (typically 'Bearer')")
    expires_in: int | None = Field(default=None, description="The lifetime in seconds of the access token")
    refresh_token: str | None = Field(default=None, description="The refresh token (if issued)")
    scope: str | None = Field(default=None, description="The scope of the access token (if different from requested)")
