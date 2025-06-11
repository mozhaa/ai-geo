from typing import *

import torch
from torch.nn import functional as F


class PanoConverter:
    def __init__(
        self, size: int, phi: float, theta: float, fov: float = torch.pi / 2, batch_size: int = 1, device: Any = "cpu"
    ) -> None:
        self.size = size
        self.phi = phi
        self.theta = theta
        self.fov = fov
        self.batch_size = batch_size
        self.device = device

        self._calculate_support_vectors()
        self._prepare_base_mapping()

    def _calculate_support_vectors(self) -> None:
        self.v = torch.tensor(
            [
                torch.cos(self.phi) * torch.cos(self.theta),
                torch.sin(self.phi) * torch.cos(self.theta),
                -torch.sin(self.theta),
            ]
        )
        s = torch.tan(self.fov / 2)

        up = torch.tensor([0, 0, 1])
        if torch.allclose(self.v, up):
            up = torch.tensor([1, 0, 0])

        self.dj = -torch.cross(self.v, up)
        self.di = torch.cross(self.v, self.dj)

        self.dj *= s / torch.linalg.norm(self.dj)
        self.di *= s / torch.linalg.norm(self.di)

    def _cube_to_3d(self, i: float, j: float) -> torch.ndarray:
        return self.v + self.di * (2 * i - 1) + self.dj * (2 * j - 1)

    def _cube_to_pano(self, i: float, j: float) -> Tuple[float, float]:
        x, y, z = self._cube_to_3d(i, j)
        phi = torch.atan2(y, x)
        rsin = torch.hypot(x, y)
        theta = torch.atan2(z, rsin)
        ox = 2 * theta / torch.pi
        oy = phi / torch.pi
        return ox, oy

    @torch.no_grad()
    def _prepare_base_mapping(self) -> None:
        self.base_mapping = torch.zeros((self.batch_size, self.size, self.size, 2), dtype=torch.float)
        for i in range(self.size):
            for j in range(self.size):
                xmap, ymap = self._cube_to_pano((i + 0.5) / self.size, (j + 0.5) / self.size)
                self.base_mapping[:, i, j, 0] = ymap
                self.base_mapping[:, i, j, 1] = xmap
        self.base_mapping = self.base_mapping.to(self.device)

    @torch.no_grad()
    def convert(self, pano_batch: torch.Tensor) -> torch.Tensor:
        if len(pano_batch.shape) != 4:
            raise TypeError("expected pano shape to be (N, C, H, W)")

        n_batches = pano_batch.shape[0]
        if n_batches > self.batch_size:
            raise TypeError("too many batches")

        return F.grid_sample(pano_batch.to(self.device), self.base_mapping[:n_batches, ...], align_corners=True)
