from abc import ABC, abstractmethod
from typing import Callable, Dict


class AuthStrategy(ABC):
    """Base class for authentication strategies."""

    @abstractmethod
    def headers(self) -> Dict[str, str]:
        """Return headers required for authentication.

        Returns:
            Dict[str, str]: Headers to include in the request.
        """
        pass


class JwtAuth(AuthStrategy):
    """JWT token authentication strategy.

    Adds an 'Authorization: Bearer {token}' header to requests.
    """

    def __init__(self, token_provider: Callable[[], str]):
        """Initialize with a token provider function.

        Args:
            token_provider: A callable that returns the JWT token string.
        """
        self.token_provider = token_provider

    def headers(self) -> Dict[str, str]:
        """Return headers with JWT token.

        Returns:
            Dict[str, str]: Headers with JWT token.
        """
        token = self.token_provider()
        return {"Authorization": f"Bearer {token}"}


class HmacAuth(AuthStrategy):
    """HMAC authentication strategy (placeholder).

    Will compute HMAC signatures for requests.
    """

    def __init__(self, api_key: str, api_secret: str):
        """Initialize with API credentials.

        Args:
            api_key: The API key for the HMAC authentication.
            api_secret: The API secret for the HMAC authentication.
        """
        self.api_key = api_key
        self.api_secret = api_secret

    def headers(self) -> Dict[str, str]:
        """Return headers with HMAC authentication.

        Note: This is a placeholder implementation. The actual HMAC
        implementation will depend on the specific requirements.

        Returns:
            Dict[str, str]: Headers with HMAC authentication.
        """
        # Placeholder for future implementation
        return {
            "X-Api-Key": self.api_key,
            # Real implementation would compute HMAC signature
        }


class NoAuth(AuthStrategy):
    """No authentication strategy."""

    def headers(self) -> Dict[str, str]:
        """Return empty headers.

        Returns:
            Dict[str, str]: Empty headers dictionary.
        """
        return {}
