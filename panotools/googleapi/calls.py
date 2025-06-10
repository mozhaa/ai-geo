import io
from typing import *

import aiohttp
import orjson
from PIL import Image

from panotools.utils import country_codes_to_index


async def single_image_search(session: aiohttp.ClientSession, lat: float, lng: float, radius: float = 100) -> Any:
    url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/SingleImageSearch"
    headers = {"x-user-agent": "grpc-web-javascript/0.1", "content-type": "application/json+protobuf"}
    body = (
        '[["apiv3", null, null, null, "US", null, null, null, null, null, [[false]]], '
        + f'[[null, null, {lat}, {lng}], {radius}], [null, ["en", "GB"], null, null, null, null, null, '
        + "null, [2], null, [[[2, true, 2], [3, true, 2], [10, true, 2]]]], [[1, 2, 3, 4, 8, 6]]]"
    )

    async with session.post(url=url, headers=headers, data=body.encode("utf-8")) as response:
        text = await response.text()
        if not response.ok:
            raise RuntimeError(text)

        data = orjson.loads(text)
        if len(data) == 2:
            if data[1] in [
                "Internal error encountered.",
                "The service is currently unavailable.",
                "Unrecoverable data loss or corruption.",
            ]:
                raise RuntimeError(data[1])
        if len(data) == 1 and len(data[0]) >= 3 and data[0][2] == "Search returned no images.":
            raise ValueError(data[0][2])

        result = {}

        if len(data[1][1][1]) < 36:
            result["panoid"] = data[1][1][1]
        else:
            for alt_image in data[1][5][0][3][0]:
                panoid = alt_image[0][1]
                if len(panoid) < 36:
                    result["panoid"] = panoid
                    break
            else:
                raise RuntimeError("no panoid found in metadata")

        result["sizes"] = list(map(lambda x: (x[0][0], x[0][1]), data[1][2][3][0]))
        result["tile_size"] = data[1][2][3][1]
        result["country_code"] = data[1][5][0][1][4]

        if result["country_code"] not in country_codes_to_index:
            raise RuntimeError("no country code found in metadata")

        description_node = data[1][3]
        subdivision = None

        if len(description_node) >= 3 and description_node[2] is not None:
            if len(description_node[2]) >= 2:
                subdivision = description_node[2][1][0]
            elif len(description_node[2]) >= 1:
                subdivision = description_node[2][0][0]

        if subdivision:
            result["subdivision"] = subdivision.split(",")[-1].strip()

        result["lat"] = data[1][5][0][1][0][2]
        result["lng"] = data[1][5][0][1][0][3]

        return result


async def get_metadata(session: aiohttp.ClientSession, panoid: str) -> Any:
    url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata"
    body = (
        f'[["apiv3",null,null,null,"US",null,null,null,null,null,[[0]]],["en","US"],[[[2,"{panoid}"]]],[[1,2,3,4,8,6]]]'
    )
    headers = {"x-user-agent": "grpc-web-javascript/0.1", "content-type": "application/json+protobuf"}

    async with session.post(url=url, headers=headers, data=body.encode("utf-8")) as response:
        text = await response.text()
        if not response.ok:
            raise RuntimeError(text)

        data = orjson.loads(text)

        result = {}

        result["panoid"] = panoid
        result["sizes"] = list(map(lambda x: (x[0][0], x[0][1]), data[1][0][2][3][0]))
        result["tile_size"] = data[1][0][2][3][1]
        result["country_code"] = data[1][0][5][0][1][4]

        if result["country_code"] not in country_codes_to_index:
            raise RuntimeError("no country code found in metadata")

        description_node = data[1][0][3]
        subdivision = None

        if len(description_node) >= 3 and description_node[2] is not None:
            if len(description_node[2]) >= 2:
                subdivision = description_node[2][1][0]
            elif len(description_node[2]) >= 1:
                subdivision = description_node[2][0][0]

        if subdivision:
            result["subdivision"] = subdivision.split(",")[-1].strip()

        result["lat"] = data[1][0][5][0][1][0][2]
        result["lng"] = data[1][0][5][0][1][0][3]

        return result


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


async def get_tile(session: aiohttp.ClientSession, panoid: str, x: int, y: int, zoom: int) -> Image.Image:
    url = (
        "https://streetviewpixels-pa.googleapis.com"
        + f"/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
    )
    headers = {
        "accept": "image/jpeg,image/png,image/*;q=0.9,*/*;q=0.8",
        "referer": "https://www.google.com/",
    }

    async with session.get(url=url, headers=headers) as response:
        if response.ok:
            ext = MEDIA_TYPE_TO_EXTENSION[response.headers["Content-Type"]]
            return Image.open(io.BytesIO(await response.content.read()), formats=[ext])
        else:
            text = await response.text()
            raise RuntimeError(text)
