import argparse
import asyncio
import itertools
import traceback
from pathlib import Path
from typing import *

import aiohttp
import orjson
from tqdm import tqdm

from panotools.googleapi import get_metadata, get_pano, single_image_search
from panotools.utils import get_first, limited_gather


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input JSON with locations")
    parser.add_argument("-d", "--directory", type=str, default="panoramas", help="panoramas directory name")
    parser.add_argument("-z", "--zoom", type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=3, help="panorama zoom level")
    parser.add_argument("--coro-limit", type=int, default=128, help="max number of coroutines for locations")
    parser.add_argument("--conn-limit", type=int, default=64, help="max number of simultaneous TCP connections")
    parser.add_argument(
        "--conn-limit-per-host",
        type=int,
        default=16,
        help="max number of simultaneous TCP connections per host",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="output JSON file")
    parser.epilog = "every location must have either panoid or lat+lng for obtaining panoid"
    return parser.parse_args()


async def process_location(
    location: Any, images_dir: Path, zoom: int, session: aiohttp.ClientSession, pbar: tqdm
) -> bool:
    try:
        # load location metadata
        if "metadata" not in location:
            lat = get_first(location, ["lat", "latitude"])
            lng = get_first(location, ["lng", "lon", "longitude"])
            panoid = get_first(location, ["panoId", "panoid"])

            if (lat is None or lng is None) and panoid is None:
                return

            if panoid is not None:
                location["metadata"] = await get_metadata(session, panoid)
            else:
                location["metadata"] = await single_image_search(session, lat, lng)

        # load panorama
        panoid = location["metadata"]["panoid"]
        path = images_dir / panoid[0] / panoid[1] / f"{panoid}.jpg"
        if "panorama" not in location or not Path(images_dir / location["panorama"]).exists():
            if not path.exists():
                pano = await get_pano(
                    session,
                    location["metadata"]["panoid"],
                    location["metadata"]["sizes"],
                    location["metadata"]["tile_size"],
                    zoom,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                pano.save(path)

            location["panorama"] = str(path)

        return True
    except Exception:
        tqdm.write(f"[warning]: skipped location due to error: {traceback.format_exc()}")
        return False
    finally:
        pbar.update(1)


async def main(args: argparse.Namespace) -> None:
    images_dir = Path(args.directory)
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        data = orjson.loads(f.read())
    if not isinstance(data, list):
        if "customCoordinates" not in data:
            raise ValueError("unknown format of JSON file")
        locations = data["customCoordinates"]
    else:
        locations = data

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=args.conn_limit, limit_per_host=args.conn_limit_per_host)
    ) as session:
        with tqdm(total=len(locations)) as pbar:
            locations_selectors = await limited_gather(
                [process_location(location, images_dir, args.zoom, session, pbar) for location in locations],
                args.coro_limit,
            )

    locations = itertools.compress(locations, locations_selectors)

    with open(args.output, "wb") as f:
        f.write(orjson.dumps(data))


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
