from typing import *

import numpy as np
import torch
from torch.nn import functional as F


def get_cube_center(phi: float, theta: float) -> np.ndarray:
    return np.array([np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), -np.sin(theta)])


def calculate_support_vectors(phi: float, theta: float, fov: float) -> Tuple[np.ndarray, np.ndarray]:
    v = get_cube_center(phi, theta)
    s = np.tan(fov / 2)

    up = np.array([0, 0, 1])
    if np.allclose(v, up):
        up = np.array([1, 0, 0])

    dj = -np.cross(v, up)
    di = np.cross(v, dj)

    dj *= s / np.linalg.norm(dj)
    di *= s / np.linalg.norm(di)

    return di, dj


def cube_to_3d(i: float, j: float, di: np.ndarray, dj: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v + di * (2 * i - 1) + dj * (2 * j - 1)


def cube_to_pano(i: float, j: float, di: np.ndarray, dj: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    x, y, z = cube_to_3d(i, j, di, dj, v)
    phi = np.atan2(y, x)
    rsin = np.hypot(x, y)
    theta = np.atan2(z, rsin)
    ox = 2 * theta / np.pi
    oy = phi / np.pi
    return ox, oy


def prepare_base_mapping(size: int, phi: float, theta: float, fov: float, batch_size: int, device: str) -> torch.Tensor:
    mapping = torch.zeros((batch_size, size, size, 2), dtype=torch.float, device=device)
    di, dj = calculate_support_vectors(phi, theta, fov)
    v = get_cube_center(phi, theta)
    for i in range(size):
        for j in range(size):
            xmap, ymap = cube_to_pano((i + 0.5) / size, (j + 0.5) / size, di, dj, v)
            mapping[:, i, j, 0] = ymap
            mapping[:, i, j, 1] = xmap
    return mapping


class PanoConverter:
    def __init__(
        self,
        size: int,
        phi: float,
        theta: float,
        fov: float,
        batch_size: int,
        device: Any = "cpu",
    ) -> None:
        self.device = device
        self.batch_size = batch_size

        with torch.no_grad():
            self.base_mapping = prepare_base_mapping(size, phi, theta, fov, batch_size, device)

    @torch.no_grad()
    def convert(self, pano_batch: torch.Tensor) -> torch.Tensor:
        if len(pano_batch.shape) != 4:
            raise TypeError("expected pano shape to be (N, C, H, W)")

        n_batches = pano_batch.shape[0]
        if n_batches > self.batch_size:
            raise TypeError("too many batches")

        return F.grid_sample(pano_batch.to(self.device), self.base_mapping[:n_batches, ...], align_corners=True)
