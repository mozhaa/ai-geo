import asyncio
import io
import traceback
from typing import *

import aiohttp
import orjson
from PIL import Image

from gpano.utils import safe_index


async def single_image_search(
    session: aiohttp.ClientSession, lat: float, lng: float, radius: float = 100, n_retries: int = 3
) -> Any:
    url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/SingleImageSearch"
    headers = {"x-user-agent": "grpc-web-javascript/0.1", "content-type": "application/json+protobuf"}
    body = (
        '[["apiv3", null, null, null, "US", null, null, null, null, null, [[false]]], '
        + f'[[null, null, {lat}, {lng}], {radius}], [null, ["en", "GB"], null, null, null, null, null, '
        + "null, [2], null, [[[2, true, 2], [3, true, 2], [10, true, 2]]]], [[1, 2, 3, 4, 8, 6]]]"
    )

    latest_error_message = ""
    for _ in range(n_retries):
        try:
            async with session.post(url=url, headers=headers, data=body.encode("utf-8")) as response:
                text = await response.text()
                if response.status in [400, 404]:
                    raise RuntimeError(f"single_image_search returned {response.status}. message: {text}")

                if response.ok:
                    data = orjson.loads(text)
                    if len(data) == 2:
                        if data[1] in [
                            "Internal error encountered.",
                            "The service is currently unavailable.",
                            "Unrecoverable data loss or corruption.",
                        ]:
                            latest_error_message = data[1]
                            continue
                    if len(data) == 1 and len(data[0]) >= 3 and data[0][2] == "Search returned no images.":
                        raise RuntimeError(f"single_image_search failed with message: {data[0][2]}")

                    result = {}

                    result["panoid"] = safe_index(data, [1, 1, 1])
                    if result["panoid"] is None or len(result["panoid"]) >= 36:
                        alt_images = safe_index(data, [1, 5, 0, 3, 0])
                        if alt_images is None:
                            raise RuntimeError("no panoid found in metadata")
                        for alt_image in alt_images:
                            panoid = safe_index(alt_image, [0, 1])
                            if panoid is not None and len(panoid) < 36:
                                result["panoid"] = panoid
                                break
                        else:
                            raise RuntimeError("no panoid found in metadata")

                    sizes = safe_index(data, [1, 2, 3, 0], raise_on_error=True)
                    if sizes is not None:
                        result["sizes"] = list(map(lambda x: (x[0][0], x[0][1]), sizes))

                    result["tile_size"] = safe_index(data, [1, 2, 3, 1], raise_on_error=True)
                    result["country_code"] = safe_index(data, [1, 5, 0, 1, 4])

                    description_node = safe_index(data, [1, 3])
                    subdivision = safe_index(description_node, [2, 1, 0]) or safe_index(description_node, [2, 0, 0])
                    if subdivision is not None:
                        result["subdivision"] = subdivision.split(",")[-1].strip()

                    result["lat"] = safe_index(data, [1, 5, 0, 1, 0, 2]) or lat
                    result["lng"] = safe_index(data, [1, 5, 0, 1, 0, 3]) or lng

                    return result
                else:
                    latest_error_message = text
        except (aiohttp.ClientConnectionError, asyncio.exceptions.TimeoutError):
            latest_error_message = traceback.format_exc()

    raise RuntimeError(f"single_image_search failed after {n_retries} retries. error: {latest_error_message}")


async def get_metadata(session: aiohttp.ClientSession, panoid: str, n_retries: int = 3) -> Any:
    url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata"
    body = (
        f'[["apiv3",null,null,null,"US",null,null,null,null,null,[[0]]],["en","US"],[[[2,"{panoid}"]]],[[1,2,3,4,8,6]]]'
    )
    headers = {"x-user-agent": "grpc-web-javascript/0.1", "content-type": "application/json+protobuf"}

    latest_error_message = ""
    for _ in range(n_retries):
        try:
            async with session.post(url=url, headers=headers, data=body.encode("utf-8")) as response:
                text = await response.text()
                if response.status in [400, 404]:
                    raise RuntimeError(f"get_metadata returned {response.status}. message: {text}")

                if response.ok:
                    data = orjson.loads(text)
                    result = {}

                    result["panoid"] = panoid

                    sizes = safe_index(data, [1, 0, 2, 3, 0], raise_on_error=True)
                    if sizes is not None:
                        result["sizes"] = list(map(lambda x: (x[0][0], x[0][1]), sizes))

                    result["tile_size"] = safe_index(data, [1, 0, 2, 3, 1], raise_on_error=True)
                    result["country_code"] = safe_index(data, [1, 0, 5, 0, 1, 4])

                    description_node = safe_index(data, [1, 0, 3])
                    subdivision = safe_index(description_node, [2, 1, 0]) or safe_index(description_node, [2, 0, 0])
                    if subdivision is not None:
                        result["subdivision"] = subdivision.split(",")[-1].strip()

                    result["lat"] = safe_index(data, [1, 0, 5, 0, 1, 0, 2], raise_on_error=True)
                    result["lng"] = safe_index(data, [1, 0, 5, 0, 1, 0, 3], raise_on_error=True)

                    return result
                else:
                    latest_error_message = text
        except (aiohttp.ClientConnectionError, asyncio.exceptions.TimeoutError):
            latest_error_message = traceback.format_exc()

    raise RuntimeError(f"get_metadata failed after {n_retries} retries. error: {latest_error_message}")


MEDIA_TYPE_TO_EXTENSION = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/webp": "webp",
    "image/tiff": "tiff",
    "image/avif": "avif",
    "image/svg+xml": "svg",
}


async def get_tile(
    session: aiohttp.ClientSession, panoid: str, x: int, y: int, zoom: int, n_retries: int = 3
) -> Image.Image:
    url = (
        "https://streetviewpixels-pa.googleapis.com"
        + f"/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
    )
    headers = {
        "accept": "image/jpeg,image/png,image/*;q=0.9,*/*;q=0.8",
        "referer": "https://www.google.com/",
    }

    latest_error_message = ""
    for _ in range(n_retries):
        try:
            async with session.get(url=url, headers=headers) as response:
                if response.status in [400, 404]:
                    raise RuntimeError(f"get_tile returned {response.status}. message: {await response.text()}")

                if response.ok:
                    ext = MEDIA_TYPE_TO_EXTENSION[response.headers["Content-Type"]]
                    return Image.open(io.BytesIO(await response.content.read()), formats=[ext])
                else:
                    latest_error_message = await response.text()
        except (aiohttp.ClientConnectionError, asyncio.exceptions.TimeoutError):
            latest_error_message = traceback.format_exc()

    raise RuntimeError(f"get_tile failed after {n_retries} retries. error: {latest_error_message}")
