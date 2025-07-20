from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HttpClientSettings(BaseSettings):
    """HTTP client configuration settings.

    These settings can be configured via environment variables:
    - HTTP_TIMEOUT_S: Request timeout in seconds
    - HTTP_RETRIES: Maximum number of retry attempts
    - HTTP_BACKOFF_FACTOR: Backoff factor for retry delay calculation
    """

    model_config = SettingsConfigDict(env_prefix="HTTP_", case_sensitive=False)

    timeout_s: float = Field(default=30.0, validation_alias="HTTP_TIMEOUT_S")
    retries: int = Field(default=3, validation_alias="HTTP_RETRIES")
    backoff_factor: float = Field(default=0.5, validation_alias="HTTP_BACKOFF_FACTOR")


def get_http_settings() -> HttpClientSettings:
    """Get HTTP client settings from environment variables."""
    return HttpClientSettings()
