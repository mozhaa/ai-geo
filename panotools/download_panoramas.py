import argparse
import asyncio
from pathlib import Path
from typing import *

import aiohttp
import orjson
from tqdm import tqdm

from panotools.googleapi import get_metadata, get_pano, single_image_search
from panotools.utils import assign_tasks, get_first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input JSON with locations")
    parser.add_argument("-d", "--directory", type=str, default="panoramas", help="panoramas directory name")
    parser.add_argument("-n", "--n-workers", type=int, default=8, help="max number of parallel sessions")
    parser.add_argument("-z", "--zoom", type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=3, help="panorama zoom level")
    parser.add_argument("-o", "--output", type=str, required=True, help="output JSON file")
    parser.epilog = "every location must have either panoid or lat+lng for obtaining panoid"
    return parser.parse_args()


class Worker:
    def __init__(self, images_dir: Path, zoom: int) -> None:
        self.images_dir = images_dir
        self.zoom = zoom
        self.locations = []

    async def load_metadata(self, location: Any, session: aiohttp.ClientSession) -> Any:
        if "metadata" in location:
            return location["metadata"]

        lat = get_first(location, ["lat", "latitude"])
        lng = get_first(location, ["lng", "lon", "longitude"])
        panoid = get_first(location, ["panoId", "panoid"])

        if (lat is None or lng is None) and panoid is None:
            return

        if panoid is not None:
            return await get_metadata(panoid, session=session)
        else:
            return await single_image_search(lat, lng, session=session)

    def get_path_by_panoid(self, panoid: str) -> Path:
        return self.images_dir / panoid[0] / panoid[1] / f"{panoid}.jpg"

    async def load_panorama(self, location: Any, session: aiohttp.ClientSession) -> str:
        path = self.get_path_by_panoid(location["metadata"]["panoid"])
        if "panorama" in location and Path(self.images_dir / location["panorama"]).exists():
            return location["panorama"]
        if path.exists():
            return str(path)

        pano = await get_pano(
            location["metadata"]["panoid"],
            location["metadata"]["sizes"],
            location["metadata"]["tile_size"],
            self.zoom,
            session=session,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pano.save(path)
        return str(path)

    async def process_locations(self, locations: AsyncIterator[Any]) -> None:
        async with aiohttp.ClientSession() as session:
            async for location in locations:
                try:
                    location["metadata"] = await self.load_metadata(location, session)
                    location["panorama"] = await self.load_panorama(location, session)
                    self.locations.append(location)
                except Exception as e:
                    print(f"[warning]: skipped location due to error: {str(e)}")
                    continue


async def main(args: argparse.Namespace) -> None:
    images_dir = Path(args.directory)
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        in_locations = orjson.loads(f.read())
    if not isinstance(in_locations, list):
        if "customCoordinates" not in in_locations:
            raise ValueError("unknown format of JSON file")
        in_locations = in_locations["customCoordinates"]

    workers = [Worker(images_dir, args.zoom) for _ in range(args.n_workers)]
    await assign_tasks(tqdm(in_locations), [w.process_locations for w in workers])
    out_locations = sum([w.locations for w in workers], [])

    with open(args.output, "wb") as f:
        f.write(orjson.dumps({"customCoordinates": out_locations}))


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
