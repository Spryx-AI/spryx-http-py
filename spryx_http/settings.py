from pydantic import Field
from pydantic_settings import BaseSettings


class HttpClientSettings(BaseSettings):
    """HTTP client configuration settings.

    These settings can be configured via environment variables:
    - HTTP_TIMEOUT_S: Request timeout in seconds
    - HTTP_RETRIES: Maximum number of retry attempts
    - HTTP_BACKOFF_FACTOR: Backoff factor for retry delay calculation
    - HTTP_TRACE_ENABLED: Enable distributed tracing
    """

    timeout_s: float = Field(default=30.0, env="HTTP_TIMEOUT_S")
    retries: int = Field(default=3, env="HTTP_RETRIES")
    backoff_factor: float = Field(default=0.5, env="HTTP_BACKOFF_FACTOR")
    trace_enabled: bool = Field(default=True, env="HTTP_TRACE_ENABLED")

    class Config:
        env_prefix = "HTTP_"
        case_sensitive = False


def get_http_settings() -> HttpClientSettings:
    """Get HTTP client settings from environment variables."""
    return HttpClientSettings()
