import math
from contextlib import nullcontext
from typing import List, Optional, Tuple

import aiohttp
import numpy as np
from PIL import Image

from .calls import get_tile


def concat_grid(arrays: List[List[np.ndarray]]) -> np.ndarray:
    return np.concatenate([np.concatenate(row, axis=1) for row in arrays], axis=0)


async def get_hires_tile(
    panoid: str,
    x: int,
    y: int,
    w: int,
    h: int,
    zoom: int,
    session: Optional[aiohttp.ClientSession] = None,
) -> np.ndarray:
    if session is None:
        session_ctx = session = aiohttp.ClientSession()
    else:
        session_ctx = nullcontext()
    async with session_ctx:
        grid = []
        for dy in range(h):
            row = []
            for dx in range(w):
                row.append(await get_tile(panoid, x + dx, y + dy, zoom, session=session))
            grid.append(row)
        return concat_grid(grid)


def get_dimenstions(size: Tuple[int, int], tile_size: Tuple[int, int]) -> Tuple[int, int]:
    return math.ceil(size[1] / tile_size[1]), math.ceil(size[0] / tile_size[0])


async def get_pano(
    panoid: str,
    sizes: List[Tuple[int, int]],
    tile_size: Tuple[float, float],
    zoom: int,
    session: Optional[aiohttp.ClientSession] = None,
) -> Image.Image:
    size = sizes[zoom]
    w, h = get_dimenstions(size, tile_size)
    pano = await get_hires_tile(panoid, 0, 0, w, h, zoom, session=session)
    return Image.fromarray(pano[: size[0], : 2 * size[0], ...])
