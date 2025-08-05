"""Tests for TypeAdapter support in cast_to parameter."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Optional
from pydantic import BaseModel, TypeAdapter, Field

from spryx_http import SpryxAsyncClient, SpryxSyncClient
from spryx_http.auth_strategies import ApiKeyAuthStrategy


class Address(BaseModel):
    """Address model."""
    street: str
    city: str
    country: str
    postal_code: Optional[str] = None


class Person(BaseModel):
    """Person model with nested Address."""
    id: int
    name: str
    age: int
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    address: Optional[Address] = None


# Create TypeAdapters for various use cases
dict_adapter = TypeAdapter(dict[str, str])
list_str_adapter = TypeAdapter(list[str])
optional_int_adapter = TypeAdapter(Optional[int])
person_list_adapter = TypeAdapter(list[Person])
complex_dict_adapter = TypeAdapter(dict[str, list[int]])


class TestTypeAdapterAsync:
    """Test TypeAdapter support in SpryxAsyncClient."""

    @pytest.fixture
    def auth_strategy(self):
        """Create API key auth strategy."""
        return ApiKeyAuthStrategy(api_key="test-api-key")

    @pytest.fixture
    def client(self, auth_strategy):
        """Create async client."""
        return SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

    @pytest.mark.asyncio
    async def test_type_adapter_dict(self, client):
        """Test TypeAdapter with dict type."""
        with patch.object(client, 'request') as mock_request:
            # Mock response
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {"key1": "value1", "key2": "value2"}
            mock_request.return_value = response

            # Make request with TypeAdapter
            result = await client.get("/test", cast_to=dict_adapter)

            # Verify result
            assert isinstance(result, dict)
            assert result == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_type_adapter_list_str(self, client):
        """Test TypeAdapter with list[str] type."""
        with patch.object(client, 'request') as mock_request:
            # Mock response
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = ["item1", "item2", "item3"]
            mock_request.return_value = response

            # Make request with TypeAdapter
            result = await client.get("/test", cast_to=list_str_adapter)

            # Verify result
            assert isinstance(result, list)
            assert all(isinstance(item, str) for item in result)
            assert result == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_type_adapter_optional_int(self, client):
        """Test TypeAdapter with Optional[int] type."""
        with patch.object(client, 'request') as mock_request:
            # Test with value
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = 42
            mock_request.return_value = response

            result = await client.get("/test", cast_to=optional_int_adapter)
            assert result == 42

            # Test with None
            response.json.return_value = None
            result = await client.get("/test", cast_to=optional_int_adapter)
            assert result is None

    @pytest.mark.asyncio
    async def test_type_adapter_person_list(self, client):
        """Test TypeAdapter with list of Pydantic models."""
        with patch.object(client, 'request') as mock_request:
            # Mock response
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = [
                {
                    "id": 1,
                    "name": "John Doe",
                    "age": 30,
                    "email": "john@example.com",
                    "address": {
                        "street": "123 Main St",
                        "city": "New York",
                        "country": "USA",
                        "postal_code": "10001"
                    }
                },
                {
                    "id": 2,
                    "name": "Jane Smith",
                    "age": 25,
                    "email": "jane@example.com"
                }
            ]
            mock_request.return_value = response

            # Make request with TypeAdapter
            result = await client.get("/people", cast_to=person_list_adapter)

            # Verify result
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(person, Person) for person in result)
            assert result[0].name == "John Doe"
            assert result[0].address.city == "New York"
            assert result[1].name == "Jane Smith"
            assert result[1].address is None

    @pytest.mark.asyncio
    async def test_type_adapter_complex_dict(self, client):
        """Test TypeAdapter with complex dict type."""
        with patch.object(client, 'request') as mock_request:
            # Mock response
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {
                "group1": [1, 2, 3],
                "group2": [4, 5, 6],
                "group3": []
            }
            mock_request.return_value = response

            # Make request with TypeAdapter
            result = await client.get("/groups", cast_to=complex_dict_adapter)

            # Verify result
            assert isinstance(result, dict)
            assert all(isinstance(v, list) for v in result.values())
            assert all(isinstance(i, int) for v in result.values() for i in v)
            assert result["group1"] == [1, 2, 3]
            assert result["group2"] == [4, 5, 6]
            assert result["group3"] == []

    @pytest.mark.asyncio
    async def test_type_adapter_validation_error(self, client):
        """Test TypeAdapter validation error handling."""
        with patch.object(client, 'request') as mock_request:
            # Mock response with invalid data
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {
                "id": "not-an-int",  # Invalid: should be int
                "name": "John Doe",
                "age": 30,
                "email": "invalid-email"  # Invalid email format
            }
            mock_request.return_value = response

            # Create adapter for Person model
            person_adapter = TypeAdapter(Person)

            # Make request and expect validation error
            with pytest.raises(ValueError):  # Pydantic will raise validation error
                await client.get("/person", cast_to=person_adapter)


class TestTypeAdapterSync:
    """Test TypeAdapter support in SpryxSyncClient."""

    @pytest.fixture
    def auth_strategy(self):
        """Create API key auth strategy."""
        return ApiKeyAuthStrategy(api_key="test-api-key")

    @pytest.fixture
    def client(self, auth_strategy):
        """Create sync client."""
        return SpryxSyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

    def test_type_adapter_dict(self, client):
        """Test TypeAdapter with dict type."""
        with patch.object(client, 'request') as mock_request:
            # Mock response
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {"key1": "value1", "key2": "value2"}
            mock_request.return_value = response

            # Make request with TypeAdapter
            result = client.get("/test", cast_to=dict_adapter)

            # Verify result
            assert isinstance(result, dict)
            assert result == {"key1": "value1", "key2": "value2"}

    def test_type_adapter_person_list(self, client):
        """Test TypeAdapter with list of Pydantic models."""
        with patch.object(client, 'request') as mock_request:
            # Mock response
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = [
                {
                    "id": 1,
                    "name": "John Doe",
                    "age": 30,
                    "email": "john@example.com"
                }
            ]
            mock_request.return_value = response

            # Make request with TypeAdapter
            result = client.get("/people", cast_to=person_list_adapter)

            # Verify result
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], Person)
            assert result[0].name == "John Doe"

    def test_type_adapter_with_post(self, client):
        """Test TypeAdapter with POST request."""
        with patch.object(client, 'request') as mock_request:
            # Mock response
            response = Mock()
            response.status_code = 201
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {
                "id": 3,
                "name": "New Person",
                "age": 28,
                "email": "new@example.com"
            }
            mock_request.return_value = response

            # Create adapter for Person model
            person_adapter = TypeAdapter(Person)

            # Make POST request with TypeAdapter
            result = client.post(
                "/people",
                json={"name": "New Person", "age": 28, "email": "new@example.com"},
                cast_to=person_adapter
            )

            # Verify result
            assert isinstance(result, Person)
            assert result.id == 3
            assert result.name == "New Person"


class TestTypeAdapterEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def auth_strategy(self):
        """Create API key auth strategy."""
        return ApiKeyAuthStrategy(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_type_adapter_with_union_types(self, auth_strategy):
        """Test TypeAdapter with Union types."""
        from typing import Union
        
        # Create adapter for Union type
        union_adapter = TypeAdapter(Union[str, int, None])
        
        client = SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

        with patch.object(client, 'request', new_callable=AsyncMock) as mock_request:
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            
            # Configure mock_request to return response
            mock_request.return_value = response
            
            # Test with string
            response.json.return_value = "hello"
            result = await client.get("/test", cast_to=union_adapter)
            assert result == "hello"
            
            # Test with int
            response.json.return_value = 42
            result = await client.get("/test", cast_to=union_adapter)
            assert result == 42
            
            # Test with None
            response.json.return_value = None
            result = await client.get("/test", cast_to=union_adapter)
            assert result is None

    @pytest.mark.asyncio
    async def test_type_adapter_with_custom_validators(self, auth_strategy):
        """Test TypeAdapter with custom validators."""
        from pydantic import field_validator
        
        class ValidatedModel(BaseModel):
            value: int
            
            @field_validator('value')
            def value_must_be_positive(cls, v):
                if v <= 0:
                    raise ValueError('value must be positive')
                return v
        
        validated_adapter = TypeAdapter(ValidatedModel)
        
        client = SpryxAsyncClient(
            base_url="https://api.test.com",
            auth_strategy=auth_strategy
        )

        with patch.object(client, 'request', new_callable=AsyncMock) as mock_request:
            response = Mock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            
            # Configure mock_request to return response
            mock_request.return_value = response
            
            # Test with valid value
            response.json.return_value = {"value": 10}
            result = await client.get("/test", cast_to=validated_adapter)
            assert result.value == 10
            
            # Test with invalid value
            response.json.return_value = {"value": -5}
            with pytest.raises(ValueError):
                await client.get("/test", cast_to=validated_adapter)