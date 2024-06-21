import pytest
import numpy as np
from pyquaternion import Quaternion
from socc_plotter.transforms import (
    quart_to_transformation_matrix,
    intrinsic_matrix_array,
    estimate_intrinsics,
    create_transformation_matrix,
)


# Test cases for quart_to_transformation_matrix
def test_quart_to_transformation_matrix_identity():
    quaternion = (1, 0, 0, 0)  # Identity quaternion
    translation = (0, 0, 0)
    expected_matrix = np.eye(4)
    result = quart_to_transformation_matrix(quaternion, translation)
    assert np.allclose(
        result, expected_matrix
    ), "Identity quaternion should produce identity matrix"


def test_quart_to_transformation_matrix_translation():
    quaternion = (1, 0, 0, 0)
    translation = (1, 2, 3)
    expected_matrix = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    )
    result = quart_to_transformation_matrix(quaternion, translation)
    assert np.allclose(
        result, expected_matrix
    ), "Translation should be correctly applied"


def test_quart_to_transformation_matrix_rotation():
    quaternion = (0.7071068, 0.7071068, 0, 0)  # 90 degrees around x-axis
    translation = (0, 0, 0)
    expected_matrix = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    result = quart_to_transformation_matrix(quaternion, translation)
    assert np.allclose(
        result, expected_matrix, atol=1e-6
    ), "90 degrees rotation around x-axis should be correct"


# Test cases for intrinsic_matrix_array
def test_intrinsic_matrix_array():
    f_x, f_y, c_x, c_y = 1000, 1000, 500, 500
    expected_matrix = np.array(
        [[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float16
    )
    result = intrinsic_matrix_array(f_x, f_y, c_x, c_y)
    assert np.allclose(
        result, expected_matrix
    ), "Intrinsic matrix should match expected values"


# Test cases for estimate_intrinsics
def test_estimate_intrinsics():
    fov_x, fov_y = 90, 60  # degrees
    width, height = 800, 600
    c_x, c_y = width / 2.0, height / 2.0
    f_x = c_x / np.tan(np.deg2rad(fov_x) / 2.0)
    f_y = c_y / np.tan(np.deg2rad(fov_y) / 2.0)
    expected_matrix = np.array(
        [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=np.float16
    )
    result = estimate_intrinsics(fov_x, fov_y, height, width)
    assert np.allclose(
        result, expected_matrix, atol=1e-3
    ), "Intrinsic matrix should be estimated correctly"


# Test cases for create_transformation_matrix
def test_create_transformation_matrix_identity():
    pos = (0, 0, 0)
    rot = (1, 0, 0, 0)  # Identity quaternion
    expected_matrix = np.eye(4)
    result = create_transformation_matrix(pos, rot)
    assert np.allclose(
        result, expected_matrix
    ), "Identity quaternion and zero position should produce identity matrix"


def test_create_transformation_matrix_translation():
    pos = (1, 2, 3)
    rot = (1, 0, 0, 0)
    expected_matrix = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    )
    result = create_transformation_matrix(pos, rot)
    assert np.allclose(
        result, expected_matrix
    ), "Translation should be correctly applied"


def test_create_transformation_matrix_rotation():
    pos = (0, 0, 0)
    rot = (0.7071068, 0.7071068, 0, 0)  # 90 degrees around x-axis
    expected_matrix = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    result = create_transformation_matrix(pos, rot)
    assert np.allclose(
        result, expected_matrix, atol=1e-6
    ), "90 degrees rotation around x-axis should be correct"
