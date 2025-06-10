import asyncio
from typing import *


def get_first[K, V](d: Dict[K, V], keys: List[K], default: Optional[V] = None) -> V:
    for key in keys:
        if key in d:
            return d[key]
    return default


async def limited_gather[T](tasks: List[Awaitable[T]], limit: int) -> List[T]:
    semaphore = asyncio.Semaphore(limit)

    async def wrapped_task(task: Awaitable[T]) -> Awaitable[T]:
        async with semaphore:
            return await task

    return await asyncio.gather(*[wrapped_task(task) for task in tasks])
