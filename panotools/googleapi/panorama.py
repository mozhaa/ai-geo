import math
import asyncio
import itertools
from typing import List, Tuple

import aiohttp
import numpy as np
from PIL import Image

from .calls import get_tile


def concat_grid(arrays: List[List[np.ndarray]]) -> np.ndarray:
    return np.concatenate([np.concatenate(row, axis=1) for row in arrays], axis=0)


async def get_hires_tile(
    session: aiohttp.ClientSession,
    panoid: str,
    x: int,
    y: int,
    w: int,
    h: int,
    zoom: int,
) -> np.ndarray:
    tasks = [get_tile(session, panoid, x + dx, y + dy, zoom) for dy in range(h) for dx in range(w)]
    tiles = await asyncio.gather(*tasks)
    grid = list(itertools.batched(tiles, w))
    return concat_grid(grid)


def get_dimenstions(size: Tuple[int, int], tile_size: Tuple[int, int]) -> Tuple[int, int]:
    return math.ceil(size[1] / tile_size[1]), math.ceil(size[0] / tile_size[0])


async def get_pano(
    session: aiohttp.ClientSession,
    panoid: str,
    sizes: List[Tuple[int, int]],
    tile_size: Tuple[float, float],
    zoom: int,
) -> Image.Image:
    size = sizes[zoom]
    w, h = get_dimenstions(size, tile_size)
    pano = await get_hires_tile(session, panoid, 0, 0, w, h, zoom)
    return Image.fromarray(pano[: size[0], : 2 * size[0], ...])
