"""Pagination utilities for HTTP clients.

This module provides utilities for working with paginated API responses
in an asynchronous way.
"""

from collections import deque
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from spryx_http.base import SpryxAsyncClient

T = TypeVar("T")


class AsyncPaginator(AsyncIterator[Dict[str, Any]]):
    """Asynchronous paginator for iterating over paginated API responses.

    This class provides an easy way to iterate over paginated API responses
    using an `async for` loop. It uses a client's get_json method to fetch
    paginated data, and delegates the parsing of the response and building
    of query parameters to caller-provided functions.

    Example:
        ```python
        paginator = AsyncPaginator(
            client=http_client,
            url="/v1/messages",
            page_parser=lambda res: (res["data"]["items"], res["data"].get("next_cursor")),
            params_builder=lambda cursor: {"cursor": cursor} if cursor else {}
        )

        async for item in paginator:
            process_item(item)
        ```

    Attributes:
        client: The HTTP client to use for requests.
        url: The URL to request.
        page_parser: Function to extract items and next cursor from the response.
        params_builder: Function to build query parameters from a cursor.
    """

    def __init__(
        self,
        client: SpryxAsyncClient,
        url: str,
        *,
        page_parser: Callable[
            [Dict[str, Any]], Tuple[List[Dict[str, Any]], Optional[str]]
        ],
        params_builder: Callable[[Optional[str]], Dict[str, Any]],
    ) -> None:
        """Initialize the paginator.

        Args:
            client: The HTTP client to use for requests.
            url: The URL to request.
            page_parser: Function to extract items and next cursor from the response.
                It should return a tuple of (items, next_cursor) where items is a list
                of items and next_cursor is the cursor to use for the next page, or None
                if there are no more pages.
            params_builder: Function to build query parameters from a cursor.
                It should return a dictionary of query parameters to use for the request.
        """
        self.client = client
        self.url = url
        self.page_parser = page_parser
        self.params_builder = params_builder

        # Buffer for items from the current page
        self._buffer: Deque[Dict[str, Any]] = deque()

        # Current cursor
        self._cursor: Optional[str] = None

        # Flag to track if we've fetched all pages
        self._exhausted = False

    async def __anext__(self) -> Dict[str, Any]:
        """Get the next item from the paginator.

        Returns:
            The next item from the paginator.

        Raises:
            StopAsyncIteration: When there are no more items.
        """
        # If the buffer is empty, try to fetch the next page
        if not self._buffer and not self._exhausted:
            await self._fetch_next_page()

        # If the buffer is still empty after attempting to fetch, we're done
        if not self._buffer:
            raise StopAsyncIteration

        # Return the next item from the buffer
        return self._buffer.popleft()

    async def _fetch_next_page(self) -> None:
        """Fetch the next page of results and add them to the buffer."""
        # Build the query parameters for the next page
        params = self.params_builder(self._cursor)

        # Fetch the next page
        response = await self.client.get_json(self.url, params=params)

        # Parse the response to get items and the next cursor
        items, next_cursor = self.page_parser(response)

        # Update the buffer with new items
        self._buffer.extend(items)

        # Update the cursor
        self._cursor = next_cursor

        # If there's no next cursor or the page is empty, mark as exhausted
        if next_cursor is None or not items:
            self._exhausted = True
