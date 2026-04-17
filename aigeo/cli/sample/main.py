import argparse
from pathlib import Path

import orjson
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm import tqdm

from aigeo.transforms import PanoConverter
from aigeo.utils import batchedby


def main(args: argparse.Namespace) -> None:
    sample_dir = Path(args.output)
    sample_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = Path(args.input).parent
    output_json = sample_dir / args.json_filename

    with open(args.input, "r", encoding="utf-8") as f:
        locations = orjson.loads(f.read())
    if not isinstance(locations, list):
        if "customCoordinates" not in locations:
            raise ValueError("unknown format of JSON file")
        locations = locations["customCoordinates"]

    for location in locations:
        if "panorama" not in location:
            raise RuntimeError("found location without panorama in input JSON")
        if not (storage_dir / location["panorama"]).exists():
            raise RuntimeError("found location with invalid panorama in input JSON (no such file)")

    if args.append:
        with open(output_json, "rb") as f:
            out_locations = orjson.loads(f.read())
    else:
        out_locations = []

    if args.count is None:
        indices = list(range(len(locations)))
    else:
        if args.count >= len(locations):
            raise ValueError("--count should not be bigger than number of locations")
        indices = torch.randperm(len(locations))[: args.count].tolist()

    converter = PanoConverter(
        size=args.size,
        phi=args.phi / 180 * torch.pi,
        theta=args.theta / 180 * torch.pi,
        fov=args.fov / 180 * torch.pi,
        batch_size=args.batch_size,
        device=args.device,
    )

    opened_panoramas = map(
        lambda i: (
            i,
            pil_to_tensor(Image.open(storage_dir / locations[i]["panorama"])).float(),
        ),
        tqdm(indices),
    )
    batches = batchedby(opened_panoramas, key=lambda x: x[1].shape, n=args.batch_size)

    images_counter = len(out_locations)

    try:
        for batch in batches:
            indices_batch, images = zip(*batch)
            converted_images = converter.convert(torch.stack(images))
            converted_images = map(lambda t: to_pil_image(t.to(torch.uint8)), converted_images)

            for i, converted_image in zip(indices_batch, converted_images):
                fn = Path(args.images_dir) / f"{images_counter}.jpg"
                images_counter += 1

                (sample_dir / fn).parent.mkdir(parents=True, exist_ok=True)
                converted_image.save(sample_dir / fn)

                metadata = locations[i]["metadata"]
                out_locations.append(
                    {
                        "lat": metadata["lat"],
                        "lng": metadata["lng"],
                        "image": str(fn.as_posix()),
                    }
                )
    except KeyboardInterrupt:
        tqdm.write("interrupted, saving to JSON...")
    finally:
        with open(output_json, "wb") as f:
            f.write(orjson.dumps(out_locations))
