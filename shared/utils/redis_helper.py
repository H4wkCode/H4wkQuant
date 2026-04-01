"""
H4wkQuant - Redis Helper
Sentinel-aware Redis connection helper with auto-reconnect
"""
import asyncio
import os
import redis.asyncio as redis
from typing import Optional
from loguru import logger


async def get_redis_client(
    redis_url: Optional[str] = None,
    sentinel_hosts: Optional[str] = None,
    sentinel_master: str = "h4wkmaster",
    sentinel_password: Optional[str] = None,
    decode_responses: bool = True,
) -> redis.Redis:
    sentinel_hosts = sentinel_hosts or os.environ.get("REDIS_SENTINEL_HOSTS")
    sentinel_master = os.environ.get("REDIS_SENTINEL_MASTER", sentinel_master)
    sentinel_password = sentinel_password or os.environ.get("REDIS_SENTINEL_PASSWORD")

    if sentinel_hosts:
        return await _connect_sentinel(
            sentinel_hosts, sentinel_master, sentinel_password, decode_responses
        )

    url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/1")
    client = redis.from_url(
        url,
        decode_responses=decode_responses,
        retry_on_error=[redis.ConnectionError, redis.TimeoutError],
        health_check_interval=15,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )
    await client.ping()
    logger.info(f"Redis connected: {_mask_url(url)}")
    return client


async def _connect_sentinel(
    hosts_str: str, master_name: str, password: Optional[str], decode_responses: bool,
) -> redis.Redis:
    sentinels = []
    for entry in hosts_str.split(","):
        entry = entry.strip()
        if ":" in entry:
            host, port = entry.rsplit(":", 1)
            sentinels.append((host, int(port)))
        else:
            sentinels.append((entry, 26379))

    sentinel = redis.Sentinel(
        sentinels,
        sentinel_kwargs={"password": password} if password else {},
        password=password,
        decode_responses=decode_responses,
    )

    client = sentinel.master_for(master_name)
    await client.ping()
    master_info = await sentinel.discover_master(master_name)
    logger.info(f"Redis Sentinel master: {master_info[0]}:{master_info[1]}")
    return client


async def resilient_subscribe(redis_client: redis.Redis, channel: str, handler, service_name: str = ""):
    """Subscribe to a Redis channel with automatic reconnect on failure (Bug #3 fix)"""
    while True:
        try:
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(channel)
            logger.info(f"[{service_name}] Subscribed to {channel}")

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                await handler(message)

        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            logger.error(f"[{service_name}] Redis connection lost on {channel}: {e}")
            logger.info(f"[{service_name}] Reconnecting in 3 seconds...")
            await asyncio.sleep(3)
        except Exception as e:
            logger.error(f"[{service_name}] Unexpected pubsub error on {channel}: {e}")
            await asyncio.sleep(3)


def _mask_url(url: str) -> str:
    if "@" in url:
        prefix, rest = url.split("@", 1)
        if ":" in prefix:
            scheme_part = prefix.rsplit(":", 1)[0]
            return f"{scheme_part}:***@{rest}"
    return url
