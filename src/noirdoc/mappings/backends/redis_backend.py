from __future__ import annotations

from collections.abc import Awaitable
from typing import cast

from redis.asyncio import Redis

# redis-py types its async command methods as ``Awaitable[Any] | Any`` (a union
# shared with the sync client), so awaiting them yields ``Any`` and ``ping()``
# is even flagged as possibly-not-awaitable. We narrow each result to the
# concrete type we know the async client returns at runtime.


class RedisMappingBackend:
    """Redis implementation of the MappingBackend protocol."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        await self._redis.set(key, value, ex=ttl_seconds)

    async def get(self, key: str) -> bytes | None:
        return cast("bytes | None", await self._redis.get(key))

    async def delete(self, key: str) -> bool:
        deleted = cast(int, await self._redis.delete(key))
        return deleted > 0

    async def get_ttl(self, key: str) -> int | None:
        ttl = cast(int, await self._redis.ttl(key))
        return ttl if ttl >= 0 else None

    async def ping(self) -> bool:
        try:
            return bool(await cast("Awaitable[object]", self._redis.ping()))
        except Exception:
            return False
