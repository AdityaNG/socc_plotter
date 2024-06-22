from typing import Tuple

import numpy as np


def uniform_density(
    point_cloud: np.ndarray, voxel_size: float, occupancy_threshold: int = 1
) -> np.ndarray:
    """
    Convert a point cloud to uniform density using an occupancy grid approach.

    Parameters:
        point_cloud (numpy.ndarray): Nx3 array of points.
        voxel_size (float): The size of each voxel.
        occupancy_threshold (int): Minimum number of points to consider a voxel
        as occupied.

    Returns:
        uniform_point_cloud (numpy.ndarray): Mx3 array of uniformly sampled
        points.
    """

    if point_cloud.size == 0:
        return point_cloud

    # Determine the min and max coordinates
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)

    # Create grid dimensions
    grid_dims = (
        np.ceil((max_coords - min_coords) / voxel_size).astype(int) + 1
    )  # Adding 1 to include max point
    grid_dims = np.clip(grid_dims, 1, float("inf")).astype(
        int
    )  # Ensuring no zero dimension

    # Initialize occupancy grid
    occupancy_grid = np.zeros(grid_dims, dtype=int)

    # Translate points to voxel grid
    translated_points = (point_cloud - min_coords) / voxel_size
    voxel_indices = np.floor(translated_points).astype(int)

    if np.any(voxel_indices < 0):
        raise ValueError("Voxel indices must be non-negative.")

    # Clamp the voxel indices to the valid range
    voxel_indices = np.minimum(voxel_indices, grid_dims - 1)

    # Mark occupied voxels
    for idx in voxel_indices:
        occupancy_grid[tuple(idx)] += 1

    # Get occupied voxels
    occupied_voxels = np.argwhere(occupancy_grid >= occupancy_threshold)

    if occupied_voxels.size == 0:
        print(
            f"No voxels occupied with threshold {occupancy_threshold}."
            f"Check voxel size and point cloud distribution."
        )

    # Create new uniform point cloud
    uniform_points = []
    for voxel in occupied_voxels:
        voxel_center = (voxel + 0.5) * voxel_size + min_coords
        uniform_points.append(voxel_center)

    return np.array(uniform_points)


def uniform_density_colorwise(
    point_cloud: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    occupancy_threshold: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a point cloud to uniform density using an occupancy grid approach.

    Parameters:
        point_cloud (numpy.ndarray): Nx3 array of points.
        colors (numpy.ndarray): Nx3 array of colors.
        voxel_size (float): The size of each voxel.
        occupancy_threshold (int): Minimum number of points to consider a voxel
        as occupied.

    Returns:
        uniform_point_cloud (numpy.ndarray): Mx3 array of uniformly sampled
        points.
        uniform_colors (numpy.ndarray): Mx3 array of colors corresponding to
        the sampled points.
    """

    if point_cloud.size == 0:
        return point_cloud, colors

    # Determine the min and max coordinates
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)

    # Create grid dimensions
    grid_dims = (
        np.ceil((max_coords - min_coords) / voxel_size).astype(int) + 1
    )  # Adding 1 to include max point

    grid_dims = np.clip(grid_dims, 1, float("inf")).astype(int)

    # Initialize occupancy grid and color accumulation
    occupancy_grid = np.zeros(grid_dims, dtype=int)
    # Dict[int, Tuple(float, float, float)]
    color_grid = {}  # type: ignore

    # Translate points to voxel grid
    translated_points = (point_cloud - min_coords) / voxel_size
    voxel_indices = np.floor(translated_points).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, grid_dims - 1)

    # Mark occupied voxels and accumulate colors
    for point, color, idx in zip(point_cloud, colors, voxel_indices):
        idx_tuple = tuple(idx)
        occupancy_grid[idx_tuple] += 1
        if idx_tuple not in color_grid:
            color_grid[idx_tuple] = []
        color_grid[idx_tuple].append(color)

    # Get occupied voxels
    occupied_voxels = np.argwhere(occupancy_grid >= occupancy_threshold)

    # Create new uniform point cloud and corresponding colors
    uniform_points = []
    uniform_colors = []

    for voxel in occupied_voxels:
        voxel_tuple = tuple(voxel)
        voxel_center = (voxel + 0.5) * voxel_size + min_coords
        uniform_points.append(voxel_center)

        # Determine the color of the voxel by majority vote
        if voxel_tuple in color_grid:
            voxel_colors = np.array(color_grid[voxel_tuple])
            avg_color = np.mean(
                voxel_colors, axis=0
            )  # Using mean for color accumulation
            uniform_colors.append(avg_color)

    return np.array(uniform_points), np.array(uniform_colors)
