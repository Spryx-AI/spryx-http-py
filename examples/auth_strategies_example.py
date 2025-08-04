"""
Example of using different authentication strategies with Spryx HTTP clients.
"""

import asyncio

from spryx_http import SpryxAsyncClient, SpryxSyncClient
from spryx_http.auth_strategies import ApiKeyAuthStrategy, ClientCredentialsAuthStrategy


# Example 1: Using ClientCredentials Strategy (OAuth 2.0)
async def client_credentials_example():
    # Create the authentication strategy
    auth_strategy = ClientCredentialsAuthStrategy(
        client_id="your-client-id", client_secret="your-client-secret", token_url="https://auth.example.com/oauth/token"
    )

    # Create the client with the strategy
    async with SpryxAsyncClient(base_url="https://api.example.com", auth_strategy=auth_strategy) as client:
        # The client will automatically authenticate and refresh tokens as needed
        response = await client.get("/users")
        print("Users:", response)


# Example 2: Using API Key Strategy
async def api_key_example():
    # Create the authentication strategy
    auth_strategy = ApiKeyAuthStrategy(api_key="your-api-key", token_url="https://auth.example.com/api/token")

    # Create the client with the strategy
    async with SpryxAsyncClient(base_url="https://api.example.com", auth_strategy=auth_strategy) as client:
        # The client will authenticate once and reuse the token
        # API keys don't expire, so no refresh is needed
        response = await client.get("/products")
        print("Products:", response)


# Example 3: Sync client with ClientCredentials
def sync_client_example():
    auth_strategy = ClientCredentialsAuthStrategy(
        client_id="your-client-id", client_secret="your-client-secret", token_url="https://auth.example.com/oauth/token"
    )

    with SpryxSyncClient(base_url="https://api.example.com", auth_strategy=auth_strategy) as client:
        response = client.get("/orders")
        print("Orders:", response)


# Example 4: Using different strategies based on environment
def get_auth_strategy():
    """Factory function to get the appropriate auth strategy."""
    import os

    if os.getenv("USE_API_KEY"):
        return ApiKeyAuthStrategy(api_key=os.getenv("API_KEY", ""), token_url=os.getenv("TOKEN_URL", ""))
    else:
        return ClientCredentialsAuthStrategy(
            client_id=os.getenv("CLIENT_ID", ""),
            client_secret=os.getenv("CLIENT_SECRET", ""),
            token_url=os.getenv("TOKEN_URL", ""),
        )


async def dynamic_strategy_example():
    auth_strategy = get_auth_strategy()

    async with SpryxAsyncClient(base_url="https://api.example.com", auth_strategy=auth_strategy) as client:
        # The client works the same way regardless of the strategy
        response = await client.get("/data")
        print("Data:", response)


if __name__ == "__main__":
    # Run async examples
    asyncio.run(client_credentials_example())
    asyncio.run(api_key_example())
    asyncio.run(dynamic_strategy_example())

    # Run sync example
    sync_client_example()
