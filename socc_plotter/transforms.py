from typing import Tuple

import numpy as np
from pyquaternion import Quaternion


def quaternion_to_transformation_matrix(
    quaternion: Tuple[float, float, float, float],
    translation: Tuple[float, float, float],
) -> np.ndarray:
    # Convert the quaternion to a rotation matrix
    quaternion_obj = Quaternion(quaternion)
    rotation_matrix = quaternion_obj.rotation_matrix

    # Convert the rotation matrix and the translation into a 4x4 matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    return transformation_matrix


def transformation_matrix_to_quaternion(
    transformation_matrix: np.ndarray,
) -> Tuple[float, float, float, float]:
    # Extract the rotation matrix from the transformation matrix
    rotation_matrix = transformation_matrix[:3, :3]

    # Convert the rotation matrix to a quaternion
    quaternion = Quaternion(matrix=rotation_matrix)
    return tuple(quaternion.elements)


# def quaternion_to_transformation_matrix(
#     quaternion: Tuple[float, float, float, float],
#     translation: Tuple[float, float, float],
# ) -> np.ndarray:
#     # Create a Quaternion object from the list
#     q = Quaternion(quaternion)

#     # Convert quaternion to a 3x3 rotation matrix
#     R = q.rotation_matrix

#     # Create the 4x4 transformation matrix
#     T = np.eye(4)
#     T[0:3, 0:3] = R
#     T[0:3, 3] = translation

#     return T


def intrinsic_matrix_array(
    f_x: float, f_y: float, c_x: float, c_y: float
) -> np.ndarray:
    intrinsic_matrix = np.array(
        [
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1],
        ],
        dtype=np.float16,
    )
    return intrinsic_matrix


def estimate_intrinsics(
    fov_x: float,  # degrees
    fov_y: float,  # degrees
    height: int,  # pixels
    width: int,  # pixels
) -> np.ndarray:
    """
    The intrinsic matrix can be extimated from the FOV and image dimensions

    :param fov_x: FOV on x axis in degrees
    :type fov_x: float
    :param fov_y: FOV on y axis in degrees
    :type fov_y: float
    :param height: Height in pixels
    :type height: focal
    :param width: Width in pixels
    :type width: int
    :returns: (3,3) intrinsic matrix
    """
    fov_x = np.deg2rad(fov_x)
    fov_y = np.deg2rad(fov_y)
    c_x = width / 2.0
    c_y = height / 2.0
    f_x = c_x / np.tan(fov_x / 2.0)
    f_y = c_y / np.tan(fov_y / 2.0)

    intrinsic_matrix = np.array(
        [
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1],
        ],
        dtype=np.float16,
    )

    return intrinsic_matrix


def create_transformation_matrix(
    pos: Tuple[float, float, float], rot: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Creates a 4x4 transformation matrix from position and rotation.

    :param pos: List or array of three position coordinates [x, y, z].
    :param rot: List or array of four quaternion components [qw, qx, qy, qz].
    :return: A 4x4 transformation matrix.
    """
    # Convert rotation to a Quaternion object
    quat = Quaternion(rot)

    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)

    # Set the rotation part (3x3 matrix)
    transformation_matrix[:3, :3] = quat.rotation_matrix

    # Set the translation part
    transformation_matrix[:3, 3] = pos

    return transformation_matrix
