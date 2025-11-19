import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import httpx
import pydantic

from spryx_http.types import OAuthTokenResponse


class AbstractAuthStrategy(ABC):
    """Abstract base class for authentication strategies."""

    @abstractmethod
    async def authenticate_async(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Authenticate and return token data asynchronously."""
        pass

    @abstractmethod
    def authenticate_sync(self, client: httpx.Client) -> dict[str, Any]:
        """Authenticate and return token data synchronously."""
        pass

    @abstractmethod
    def get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for requests."""
        pass

    @abstractmethod
    def needs_refresh(self) -> bool:
        """Check if the token needs to be refreshed."""
        pass

    @abstractmethod
    def supports_refresh(self) -> bool:
        """Check if this strategy supports token refresh."""
        pass

    @abstractmethod
    def store_token_response(self, response: dict[str, Any]) -> None:
        """Store the token response data."""
        pass


@dataclass
class ClientCredentialsAuthStrategy(AbstractAuthStrategy):
    client_id: str
    client_secret: str
    token_url: str
    type: Literal["client_credentials"] = "client_credentials"

    # Internal token storage
    _access_token: str | None = field(default=None, init=False, repr=False)
    _refresh_token: str | None = field(default=None, init=False, repr=False)
    _token_expires_at: int | None = field(default=None, init=False, repr=False)

    async def authenticate_async(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Authenticate using OAuth 2.0 Client Credentials flow."""
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        response = await client.request("POST", self.token_url, json=payload)
        response.raise_for_status()
        return response.json()

    def authenticate_sync(self, client: httpx.Client) -> dict[str, Any]:
        """Authenticate using OAuth 2.0 Client Credentials flow."""
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        response = client.request("POST", self.token_url, json=payload)
        response.raise_for_status()
        return response.json()

    async def refresh_async(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        """Refresh the access token using the refresh token."""
        if not self._refresh_token:
            return None

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }

        try:
            response = await client.request("POST", self.token_url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError):
            return None

    def refresh_sync(self, client: httpx.Client) -> dict[str, Any] | None:
        """Refresh the access token using the refresh token."""
        if not self._refresh_token:
            return None

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }

        try:
            response = client.request("POST", self.token_url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError):
            return None

    def get_auth_headers(self) -> dict[str, str]:
        """Get OAuth Bearer token headers."""
        if not self._access_token:
            raise ValueError("No access token available. Call authenticate first.")
        return {"Authorization": f"Bearer {self._access_token}"}

    def needs_refresh(self) -> bool:
        """Check if the token needs to be refreshed."""
        if self._access_token is None or self._token_expires_at is None:
            return True

        # Add 30 seconds buffer to account for request time
        current_time = int(time.time()) + 30
        return current_time >= self._token_expires_at

    def supports_refresh(self) -> bool:
        """OAuth 2.0 supports token refresh."""
        return True

    def store_token_response(self, response: dict[str, Any]) -> None:
        """Store OAuth token response."""
        token_data = OAuthTokenResponse.model_validate(response)
        self._access_token = token_data.access_token
        self._refresh_token = token_data.refresh_token
        self._token_expires_at = int(time.time()) + token_data.expires_in if token_data.expires_in else None


@dataclass
class ApiKeyAuthStrategy(AbstractAuthStrategy):
    api_key: str
    type: Literal["api_key"] = "api_key"

    async def authenticate_async(self, _client: httpx.AsyncClient) -> dict[str, Any]:
        """API Key authentication - no actual authentication needed."""
        # API key is used directly, no token endpoint needed
        return {"access_token": self.api_key}

    def authenticate_sync(self, _client: httpx.Client) -> dict[str, Any]:
        """API Key authentication - no actual authentication needed."""
        # API key is used directly, no token endpoint needed
        return {"access_token": self.api_key}

    def get_auth_headers(self) -> dict[str, str]:
        """Get API Key auth headers."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def needs_refresh(self) -> bool:
        """API Keys don't expire and don't need refresh."""
        return False

    def supports_refresh(self) -> bool:
        """API Keys don't support refresh."""
        return False

    def store_token_response(self, _response: dict[str, Any]) -> None:
        """Store API Key token response - no-op since API key is used directly."""
        # API key is used directly, no need to store anything
        pass


AuthStrategy = Annotated[ClientCredentialsAuthStrategy | ApiKeyAuthStrategy, pydantic.Field(discriminator="type")]
