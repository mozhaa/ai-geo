import argparse
from pathlib import Path

import orjson
import torch
from PIL import Image
from tqdm import tqdm

from gpano.utils import batchedby


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sample dataset from database")
    parser.add_argument("input", type=str, help="input JSON with locations")
    parser.add_argument("-s", "--size", type=int, default=512, help="size of generated images")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size for converting")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="torch device for converting")
    parser.add_argument("-c", "--count", type=int, default=None, help="number of samples")
    parser.add_argument("--phi", type=float, default=0, help="horizontal angle of camera ([-1, 1], 0=forward)")
    parser.add_argument("--theta", type=float, default=0, help="vertical angle of camera ([-1, 1], 0=forward)")
    parser.add_argument("--fov", type=float, default=0.5, help="FOV of camera ([0, 1], 1/2=pi/2)")
    parser.add_argument("--json-filename", type=str, default="sample.json", help="name of output JSON")
    parser.add_argument("--images-dir", type=str, default="images", help="name of images directory")
    parser.add_argument("--append", action="store_true", help="append locations to output instead of overwriting")
    parser.add_argument("-o", "--output", type=str, required=True, help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sample_dir = Path(args.output)
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

    from torchvision.transforms.functional import pil_to_tensor, to_pil_image

    from gpano.transforms import PanoConverter

    converter = PanoConverter(
        args.size,
        args.phi * torch.pi,
        args.theta * torch.pi / 2,
        fov=args.fov * torch.pi,
        batch_size=args.batch_size,
        device=args.device,
    )

    opened_panoramas = map(
        lambda i: (i, pil_to_tensor(Image.open(storage_dir / locations[i]["panorama"])).float()), tqdm(indices)
    )
    batches = batchedby(opened_panoramas, key=lambda x: x[1].shape, n=args.batch_size)

    images_counter = len(out_locations)

    try:
        for batch in batches:
            indices_batch, images = zip(*batch)
            converted_images = converter.convert(torch.stack(images))
            converted_images = map(lambda t: to_pil_image(t.to(torch.uint8)), converted_images)

            for i, converted_image in zip(indices_batch, converted_images):
                fn = Path(args.images_dir) / str(images_counter // 100) / f"{images_counter}.jpg"
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


if __name__ == "__main__":
    main()
