import asyncio
from typing import *

T = TypeVar("T")


async def lock_iter(it: Iterator[T], lock: asyncio.Lock) -> AsyncGenerator[T, None]:
    while True:
        async with lock:
            try:
                i = next(it)
            except StopIteration:
                return
        yield i


async def lock_aiter(it: AsyncIterator[T], lock: asyncio.Lock) -> AsyncGenerator[T, None]:
    while True:
        async with lock:
            try:
                i = await anext(it)
            except StopAsyncIteration:
                return
        yield i


async def assign_tasks(tasks: Iterable[T], worker_jobs: List[Callable[[AsyncIterator[T]], Awaitable[None]]]) -> None:
    lock = asyncio.Lock()
    it = iter(tasks)
    return await asyncio.gather(*[j(lock_iter(it, lock)) for j in worker_jobs])
