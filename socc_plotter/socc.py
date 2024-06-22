from typing import Dict, List, Optional, Tuple

import numpy as np

from .colormap import create_cityscapes_label_colormap
from .occupancy_grid import uniform_density_colorwise
from .transforms import estimate_intrinsics


def get_socc(
    disparity: np.ndarray,
    semantics: np.ndarray,
    scale: Tuple[float, float, float] = (1, 1, -1),
    intrinsics: Optional[np.ndarray] = None,
    subsample: int = 30,
    mask: Optional[np.ndarray] = None,
    fov_x: float = 70,  # degrees
    fov_y: float = 70,  # degrees
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes depth and semantics as input and produces 3D semantic occupancy
    """

    HEIGHT, WIDTH = disparity.shape

    if intrinsics is None:
        intrinsics = estimate_intrinsics(fov_x, fov_y, HEIGHT, WIDTH)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    focal_length = (fx + fy) / 2.0
    baseline = 1.5
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

    points, colors = uniform_density_colorwise(points, colors, 0.2, 1)

    points[:, 0], points[:, 1], points[:, 2] = (
        -points[:, 0].copy(),
        points[:, 1].copy(),
        points[:, 2].copy(),
    )

    return (points, colors)


def get_multicam_socc(
    sensors: List[str],
    frame_data: Dict,
    calibration_data: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    all_points_l = []
    all_colors_l = []
    colormap = create_cityscapes_label_colormap()

    for sensor in sensors:
        depth = frame_data[sensor]["depth"]
        semantics = frame_data[sensor]["semantics"]
        semantics_rgb = semantic_to_rgb(semantics, colormap)

        socc = get_socc(depth, semantics_rgb)

        points = socc[0]
        ones_column = np.ones((points.shape[0], 1))
        points = np.hstack((points, ones_column))
        points_rot = points.copy()
        # points_rot = points @ calibration_data[sensor]
        points_rot = (calibration_data[sensor] @ points.T).T
        # points_rot = (np.linalg.inv(calibration_data[sensor]) @ points.T).T
        # points_rot = points @ np.linalg.inv(calibration_data[sensor])
        points_rot = points_rot[:, :3]

        all_points_l.append(points_rot)
        all_colors_l.append(socc[1])

    all_points: np.ndarray = np.concatenate(all_points_l, axis=0)
    all_colors: np.ndarray = np.concatenate(all_colors_l, axis=0)

    invalid_points = (
        np.isnan(all_points[:, 0])
        | np.isnan(all_points[:, 1])
        | np.isnan(all_points[:, 2])
    ) | (
        np.isinf(all_points[:, 0])
        | np.isinf(all_points[:, 1])
        | np.isinf(all_points[:, 2])
    )

    all_points = all_points[~invalid_points]
    all_colors = all_colors[~invalid_points]

    frame_socc = uniform_density_colorwise(all_points, all_colors, 0.2, 1)

    return frame_socc


def semantic_to_rgb(
    pred_semantic_map: np.ndarray,
    palette: np.ndarray = create_cityscapes_label_colormap(),
) -> np.ndarray:
    # Convert segmentation map to color map
    color_map = palette[pred_semantic_map]

    return color_map
