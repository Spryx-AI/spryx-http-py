from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spryx_http import SpryxAsyncClient, SpryxSyncClient


class AResource:
    def __init__(self, client: "SpryxAsyncClient"):
        self._client = client


class Resource:
    def __init__(self, client: "SpryxSyncClient"):
        self._client = client
