import numpy as np
import pytest
from socc_plotter.occupancy_grid import (
    uniform_density,
    uniform_density_colorwise,
)


def test_uniform_density_basic():
    point_cloud = np.array(
        [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [1, 1, 1], [1, 1, 1.1]]
    )
    voxel_size = 0.5
    result = uniform_density(point_cloud, voxel_size, occupancy_threshold=1)

    # There should be 2 occupied voxels
    assert len(result) == 2
    # The centers of these voxels should be approximately [(0.25, 0.25, 0.25), (1.25, 1.25, 1.25)]
    np.testing.assert_array_almost_equal(
        result[0], [0.25, 0.25, 0.25], decimal=1
    )
    np.testing.assert_array_almost_equal(
        result[1], [1.25, 1.25, 1.25], decimal=1
    )


def test_uniform_density_large_point_cloud():
    np.random.seed(0)  # For reproducibility
    point_cloud = np.random.uniform(-10, 10, (100000, 3))
    voxel_size = 2.0
    result = uniform_density(point_cloud, voxel_size, occupancy_threshold=3)

    # Ensure that we have a reasonable number of uniform points
    assert result.shape[0] > 0
    assert result.shape[1] == 3  # Should be a Nx3 array


def test_uniform_density_colorwise_basic():
    point_cloud = np.array(
        [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [1, 1, 1], [1, 1, 1.1]]
    )
    colors = np.array(
        [[255, 0, 0], [255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]]
    )
    voxel_size = 0.5
    uniform_points, uniform_colors = uniform_density_colorwise(
        point_cloud, colors, voxel_size, occupancy_threshold=1
    )

    # There should be 2 occupied voxels
    assert len(uniform_points) == 2
    # The centers of these voxels should be approximately [(0.25, 0.25, 0.25), (1.25, 1.25, 1.25)]
    np.testing.assert_array_almost_equal(
        uniform_points[0], [0.25, 0.25, 0.25], decimal=1
    )
    np.testing.assert_array_almost_equal(
        uniform_points[1], [1.25, 1.25, 1.25], decimal=1
    )

    # Colors should be the mean of the voxel colors
    np.testing.assert_array_almost_equal(
        uniform_colors[0], [255, 0, 0], decimal=0
    )
    np.testing.assert_array_almost_equal(
        uniform_colors[1], [0, 255, 0], decimal=0
    )


def test_uniform_density_colorwise_complex_colors():
    point_cloud = np.array(
        [
            [0, 0, 0],
            [0, 0, 0.1],
            [0, 0, 0.2],
            [1, 1, 1],
            [1, 1, 1.1],
            [1, 1, 1.2],
        ]
    )
    colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
        ]
    )
    voxel_size = 0.5
    uniform_points, uniform_colors = uniform_density_colorwise(
        point_cloud, colors, voxel_size, occupancy_threshold=1
    )

    # There should be 3 occupied voxels
    assert len(uniform_points) == 2

    # The colors should be the average color in each voxel
    expected_colors = [[85.0, 85.0, 85.0], [170.0, 170.0, 170.0]]
    uniform_colors = uniform_colors.tolist()
    expected_colors = set(
        [
            f"{str(round(float(i[0]),1)), str(round(float(i[1]),1)), str(round(float(i[2]),1))}"
            for i in expected_colors
        ]
    )
    uniform_colors = set(
        [
            f"{str(round(float(i[0]),1)), str(round(float(i[1]),1)), str(round(float(i[2]),1))}"
            for i in uniform_colors
        ]
    )

    assert expected_colors == uniform_colors


def test_uniform_density_edge_case_no_points():
    point_cloud = np.array([]).reshape(0, 3)
    voxel_size = 1.0
    result = uniform_density(point_cloud, voxel_size, occupancy_threshold=1)
    assert result.size == 0


def test_uniform_density_colorwise_edge_case_no_points():
    point_cloud = np.array([]).reshape(0, 3)
    colors = np.array([]).reshape(0, 3)
    voxel_size = 1.0
    uniform_points, uniform_colors = uniform_density_colorwise(
        point_cloud, colors, voxel_size, occupancy_threshold=1
    )
    assert uniform_points.size == 0
    assert uniform_colors.size == 0


def test_uniform_density_edge_case_single_point():
    point_cloud = np.array([[1, 1, 1]])
    voxel_size = 1.0
    result = uniform_density(point_cloud, voxel_size, occupancy_threshold=1)
    assert result.shape[0] == 1
    np.testing.assert_array_equal(result[0], [1.5, 1.5, 1.5])


def test_uniform_density_colorwise_edge_case_single_point():
    point_cloud = np.array([[1, 1, 1]])
    colors = np.array([[255, 0, 0]])
    voxel_size = 1.0
    uniform_points, uniform_colors = uniform_density_colorwise(
        point_cloud, colors, voxel_size, occupancy_threshold=1
    )
    assert uniform_points.shape[0] == 1
    np.testing.assert_array_equal(uniform_points[0], [1.5, 1.5, 1.5])
    np.testing.assert_array_equal(uniform_colors[0], [255, 0, 0])
