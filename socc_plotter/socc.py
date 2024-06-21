from typing import Optional, Tuple

import numpy as np

from .occupancy_grid import uniform_density_colorwise
from .transforms import estimate_intrinsics


def get_socc(
    disparity: np.ndarray,
    semantics: np.ndarray,
    scale: Tuple[float, float, float] = (1, 1, -1),
    intrinsics: Optional[np.ndarray] = None,
    subsample: int = 30,
    mask: Optional[np.ndarray] = None,
):
    """
    Takes depth and semantics as input and produces 3D semantic occupancy
    """

    HEIGHT, WIDTH = disparity.shape

    if intrinsics is None:
        intrinsics = estimate_intrinsics(70, 70, HEIGHT, WIDTH)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    focal_length = (fx + fy) / 2.0
    print("focal_length", focal_length)
    baseline = 1.0
    points = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

    if mask is None:
        # default to bottom half of image
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[HEIGHT // 2 :, :] = 1
        # mask[:, WIDTH//2:] = 1

    disparity[~mask] = -1.0
    depth = focal_length * baseline * np.reciprocal(disparity)

    U, V = np.ix_(
        np.arange(HEIGHT), np.arange(WIDTH)
    )  # pylint: disable=unbalanced-tuple-unpacking
    Z = depth.copy()

    X = (V - cx) * Z / fx
    Y = (U - cy) * Z / fy

    points[:, :, 0] = X * scale[0]
    points[:, :, 1] = Y * scale[1]
    points[:, :, 2] = Z * scale[2]

    colors = semantics / 255.0

    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    # subsample
    points = points[::subsample, :]
    colors = colors[::subsample, :]

    points = points.clip(-80, 80)

    points, colors = uniform_density_colorwise(points, colors, 0.15, 1)

    # x, y, z = z, -x, -y
    points[:, 0], points[:, 1], points[:, 2] = (
        points[:, 2].copy(),
        -points[:, 0].copy(),
        -points[:, 1].copy(),
    )

    return (points, colors)
