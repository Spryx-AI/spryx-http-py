# Spryx HTTP Client

A robust HTTP client library for Python with built-in retry logic, authentication, and structured logging.

## Features

- **Async and Sync Support**: Both asynchronous (`SpryxAsyncClient`) and synchronous (`SpryxSyncClient`) clients
- **Retry with Exponential Backoff**: Automatic retry of failed requests with configurable backoff
- **Authentication Management**: Pluggable authentication strategies with automatic token refresh
- **Structured Logging**: Integration with Logfire for detailed request/response logging
- **Pydantic Model Support**: Automatic parsing of responses into Pydantic models
- **Type Safe**: Full type hints and generic support

## Installation

```bash
pip install spryx-http
```

## Quick Start

### Async Client

```python
import asyncio
from spryx_http import SpryxAsyncClient
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str

async def main():
    # Initialize the async client
    client = SpryxAsyncClient(
        base_url="https://api.example.com",
        application_id="your_app_id",
        application_secret="your_app_secret",
        iam_base_url="https://iam.example.com"
    )
    
    # Make authenticated requests
    async with client:
        # GET request with model parsing
        user = await client.get("/users/1", cast_to=User)
        print(f"User: {user.name} ({user.email})")
        
        # POST request
        new_user_data = {"name": "John Doe", "email": "john@example.com"}
        created_user = await client.post("/users", json=new_user_data, cast_to=User)
        
        # Raw JSON response (without model parsing)
        raw_data = await client.get("/users/1")
        print(raw_data)

    # You can also initialize the client without a base_url
    # and use full URLs in your requests
    client_without_base_url = SpryxAsyncClient(
        application_id="your_app_id",
        application_secret="your_app_secret",
        iam_base_url="https://iam.example.com"
    )
    
    async with client_without_base_url:
        # Use full URLs in your requests
        user = await client_without_base_url.get(
            "https://api.example.com/users/1", 
            cast_to=User
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Sync Client

```python
from spryx_http import SpryxSyncClient
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str

def main():
    # Initialize the sync client
    client = SpryxSyncClient(
        base_url="https://api.example.com",
        application_id="your_app_id",
        application_secret="your_app_secret",
        iam_base_url="https://iam.example.com"
    )
    
    # Make authenticated requests
    with client:
        # GET request with model parsing
        user = client.get("/users/1", cast_to=User)
        print(f"User: {user.name} ({user.email})")
        
        # POST request
        new_user_data = {"name": "Jane Doe", "email": "jane@example.com"}
        created_user = client.post("/users", json=new_user_data, cast_to=User)
        
        # Raw JSON response (without model parsing)
        raw_data = client.get("/users/1")
        print(raw_data)

if __name__ == "__main__":
    main()
```

## API Reference

### Common Methods (Available in both clients)

Both `SpryxAsyncClient` and `SpryxSyncClient` provide the same HTTP methods:

- `get(path, *, cast_to=None, params=None, **kwargs)`
- `post(path, *, cast_to=None, json=None, **kwargs)`
- `put(path, *, cast_to=None, json=None, **kwargs)`
- `patch(path, *, cast_to=None, json=None, **kwargs)`
- `delete(path, *, cast_to=None, params=None, **kwargs)`

### Parameters

- `path`: Request path to be appended to base_url, or a full URL if base_url is None
- `cast_to`: Optional Pydantic model class to parse response into
- `params`: Optional query parameters (for GET/DELETE)
- `json`: Optional JSON data for request body (for POST/PUT/PATCH)
- `**kwargs`: Additional arguments passed to the underlying httpx request

### Client Initialization

Both clients can be initialized with or without a base_url:

```python
# With base_url
client = SpryxAsyncClient(
    base_url="https://api.example.com",
    # ... other parameters
)

# Without base_url (requires using full URLs in requests)
client = SpryxAsyncClient(
    # ... other parameters
)
# Then use full URLs in requests:
await client.get("https://api.example.com/users/1")
```

### Authentication

Both clients support automatic authentication management:

- **Application Authentication**: Uses `application_id` and `application_secret`
- **Token Refresh**: Automatically refreshes expired tokens
- **Retry on Auth Failure**: Retries requests after token refresh

### Configuration

```python
from spryx_http.settings import HttpClientSettings

settings = HttpClientSettings(
    timeout_s=30.0,
    retries=3,
    backoff_factor=0.5
)

client = SpryxAsyncClient(
    base_url="https://api.example.com",
    settings=settings,
    # ... other parameters
)
```

## Error Handling

The clients raise appropriate HTTP exceptions:

```python
from spryx_http.exceptions import (
    HttpError,
    BadRequestError, 
    ServerError,
    RateLimitError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError
)

try:
    user = await client.get("/users/1", cast_to=User)
except NotFoundError:
    print("User not found")
except AuthenticationError:
    print("Authentication failed")
except RateLimitError:
    print("Rate limit exceeded")
```

## Architecture

The library uses a shared base class (`SpryxClientBase`) for common functionality:

- **Token Management**: Shared token validation and refresh logic
- **Response Processing**: Common data extraction and model parsing
- **Settings Management**: Shared configuration handling

The async and sync clients inherit from this base and their respective httpx client classes, providing the same API with appropriate sync/async behavior.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 