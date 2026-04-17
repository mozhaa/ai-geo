import argparse
import asyncio
import itertools
import traceback
from pathlib import Path
from typing import Any, Optional

import aiohttp
import orjson
from tqdm import tqdm

from aigeo.google import get_metadata, get_pano, single_image_search
from aigeo.utils import get_first


async def process_location(
    location: Any, storage_dir: Path, images_dir: str, zoom: int, session: aiohttp.ClientSession
) -> Optional[bool]:
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
        metadata = location["metadata"]
        panoid = metadata["panoid"]
        rel_path = Path(images_dir) / panoid[0] / panoid[1] / f"{panoid}.jpg"
        abs_path = storage_dir / rel_path
        if "panorama" not in location or not (storage_dir / location["panorama"]).exists():
            if not abs_path.exists():
                pano = await get_pano(
                    session,
                    metadata["panoid"],
                    metadata["sizes"],
                    metadata["tile_size"],
                    zoom,
                )
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                pano.save(abs_path)

            location["panorama"] = str(rel_path.as_posix())

        return True
    except Exception:
        tqdm.write(f"[warning]: skipped location due to error: {traceback.format_exc()}")
        return False


async def load_panoramas(args: argparse.Namespace) -> None:
    storage_dir = Path(args.output_dir)

    with open(args.infile, "r", encoding="utf-8") as f:
        locations = orjson.loads(f.read())
    if not isinstance(locations, list):
        if "customCoordinates" not in locations:
            raise ValueError("unknown format of JSON file")
        locations = locations["customCoordinates"]

    selectors: list[Optional[bool]] = [True for _ in locations]
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=args.conn_limit, limit_per_host=args.conn_limit_per_host)
        ) as session:
            batches = itertools.batched(locations, args.batch_size)
            for i, loc_batch in enumerate(tqdm(batches, total=len(locations) // args.batch_size)):
                tasks = [process_location(loc, storage_dir, args.images_dir, args.zoom, session) for loc in loc_batch]
                from_, to_ = i * args.batch_size, (i + 1) * args.batch_size
                selectors[from_:to_] = await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
        tqdm.write("interrupted, saving to JSON...")
    finally:
        locations = list(itertools.compress(locations, selectors))
        with open(str(storage_dir / args.json_filename), "wb") as f:
            f.write(orjson.dumps(locations))


def main(args: argparse.Namespace) -> None:
    asyncio.run(load_panoramas(args))
