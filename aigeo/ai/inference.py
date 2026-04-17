import argparse
import asyncio
import re
from pathlib import Path
from urllib.parse import unquote

import aiohttp
from PIL import Image

from aigeo.google import get_metadata, get_pano


class InferenceModel:
    def __init__(self, model, ckpt_path: str | Path, converter, zoom: int = 2, device: str = "cpu") -> None:
        import torch
        from torchvision.transforms import v2

        self.device = device

        self.model = model
        state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
        state_dict = {re.sub("^model\\.", "", k): v for k, v in state_dict.items()}
        self.model.to(device)
        self.model.eval()

        self.converter = converter
        self.zoom = zoom
        self.transform = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.session = aiohttp.ClientSession()

        self.panoid_re = re.compile("panoid=([^&]+)(&|$)")
        self.url1_re = re.compile(re.escape("google.com/maps/"))
        self.url2_re = re.compile(re.escape("maps.app.goo.gl/"))

    def parse_panoid_from_url(self, url: str) -> str | None:
        m = self.panoid_re.search(unquote(url))
        if m is not None:
            return m.group(1)

    async def get_panoid_from_url(self, url: str) -> str | None:
        if self.url1_re.search(url) is not None:
            return self.parse_panoid_from_url(url)
        if self.url2_re.search(url) is not None:
            async with self.session.get(url) as response:
                return self.parse_panoid_from_url(str(response.url))

    async def predict_from_url(self, url: str):
        panoid = await self.get_panoid_from_url(url)
        if panoid is None:
            raise ValueError(f"invalid Google Street View url: {url}")

        return await self.predict_from_panoid(panoid)

    async def predict_from_panoid(self, panoid: str):
        metadata = await get_metadata(self.session, panoid)
        pano = await get_pano(self.session, panoid, metadata["sizes"], metadata["tile_size"], self.zoom)

        return self.predict_from_pano(pano)

    def predict_from_pano(self, pano: Image):
        import torch
        from torchvision.transforms.functional import pil_to_tensor
        pano.show()
        t = pil_to_tensor(pano).float()
        print(torch.stack([t] * self.converter.batch_size).shape)
        images = self.converter.convert(torch.stack([t] * self.converter.batch_size))
        preds = self.predict_from_tensor(images)
        return preds.mean(dim=0)

    def predict_from_image(self, image: Image):
        from torchvision.transforms.functional import pil_to_tensor

        return self.predict_from_tensor(pil_to_tensor(image).float().unsqueeze(0))

    def predict_from_tensor(self, t):
        from torchvision.transforms.functional import to_pil_image
        import torch

        to_pil_image(t[0].to(torch.uint8)).show()
        return self.model.forward(self.transform(t).to(self.device))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=["efficientnet_b0"], help="model architecture")
    parser.add_argument("ckpt_path", type=str, help="path to model checkpoint")
    parser.add_argument("-z", "--zoom", choices=[0, 1, 2, 3, 4, 5, 6], default=3, type=int, help="panorama zoom")
    parser.add_argument("-s", "--size", type=int, help="size of image samples to make")
    parser.add_argument("--phi", nargs="+", type=float, help="horizontal angles to make samples with ([-1, 1])")
    parser.add_argument("--theta", nargs="+", type=float, help="vertical angles to make samples with ([-1, 1])")
    parser.add_argument("--fov", nargs="+", type=float, help="fov of camera to make samples with ([0, 1])")
    parser.add_argument("--converter-device", type=str, default="cpu", help="torch device for PanoConverter")
    parser.add_argument("--device", type=str, default="cpu", help="torch device for model")
    return parser.parse_args()


async def main_loop(args: argparse.Namespace) -> None:
    from aigeo.transforms import PanoConverter

    if args.model == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        import torch.nn as nn

        model = efficientnet_b0()
        last_hidden_dim = next(model.classifier.parameters()).shape[1]
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_hidden_dim, 2),
        )
    else:
        raise ValueError(f"unknown model architecture: {args.model}")

    converter = PanoConverter(args.size, args.phi, args.theta, args.fov, device=args.converter_device)
    inference_model = InferenceModel(model, args.ckpt_path, converter, args.zoom, args.device)

    while True:
        url = input("street view url: ")
        pred = await inference_model.predict_from_url(url)
        lat, lng = pred[0], pred[1]
        print(f"{lat},{lng}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_loop(args))


if __name__ == "__main__":
    main()
