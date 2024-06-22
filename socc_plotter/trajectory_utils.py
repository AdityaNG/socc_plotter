import math
from typing import Dict

import numpy as np


def create_voxel_meshes(
    points: np.ndarray, size: float, color=(0, 0, 1, 1)
) -> Dict:
    num_points = points.shape[0]
    color = np.array(color, dtype=np.float32)
    if len(color.shape) == 1:
        color = np.tile(color, (num_points, 1))

    N = 8
    M = 12

    # Create mesh data arrays
    vertices = np.zeros((N * num_points, 3), dtype=np.float32)
    faces = np.zeros((M * num_points, 3), dtype=np.uint8)
    face_colors = np.ones((N * num_points, 4), dtype=np.float32)

    # Generate mesh data for each point
    for i in range(num_points):
        point = points[i]
        # Calculate the indices for the current point

        start_idx = N * i
        end_idx = start_idx + N
        start_face = M * i

        # Create a rectangular mesh around the point
        vertices[start_idx:end_idx, :] = np.array(
            [
                [point[0] - size, point[1] - size, point[2] - size],
                [point[0] + size, point[1] - size, point[2] - size],
                [point[0] + size, point[1] + size, point[2] - size],
                [point[0] - size, point[1] + size, point[2] - size],
                [point[0] - size, point[1] - size, point[2] + size],
                [point[0] + size, point[1] - size, point[2] + size],
                [point[0] + size, point[1] + size, point[2] + size],
                [point[0] - size, point[1] + size, point[2] + size],
            ]
        )

        # Define the face indices for the current point
        faces[start_face : start_face + M, :] = np.array(  # noqa
            [
                [start_idx + 0, start_idx + 1, start_idx + 2],
                [start_idx + 2, start_idx + 3, start_idx + 0],
                [start_idx + 4, start_idx + 5, start_idx + 6],
                [start_idx + 6, start_idx + 7, start_idx + 4],
                [start_idx + 0, start_idx + 1, start_idx + 5],
                [start_idx + 5, start_idx + 4, start_idx + 0],
                [start_idx + 1, start_idx + 2, start_idx + 6],
                [start_idx + 6, start_idx + 5, start_idx + 1],
                [start_idx + 2, start_idx + 3, start_idx + 7],
                [start_idx + 7, start_idx + 6, start_idx + 2],
                [start_idx + 3, start_idx + 0, start_idx + 4],
                [start_idx + 4, start_idx + 7, start_idx + 3],
            ]
        )

        # Assign the color to the current point
        face_colors[start_idx : start_idx + N, :] = np.tile(  # noqa
            color[i, :], (N, 1)
        )

    return {"vertexes": vertices, "faces": faces, "faceColors": face_colors}


def create_meshes(points, width, color=(0, 0, 1, 1)):
    """
    points: (N, 3)
    width: float
    color: (4, ) or (N, 4)

    given a trajectory as a set of points, create a mesh for each point
    such that the trajectory is a flat ribbon of the given width
    """
    num_points = points.shape[0]
    # Create mesh data arrays
    vertices = np.zeros((4 * num_points, 3), dtype=np.float32)
    faces = np.zeros((2 * num_points, 3), dtype=np.uint32)
    face_colors = np.ones((4 * num_points, 4), dtype=np.float32)

    # Generate mesh data for each point
    for i in range(1, points.shape[0]):
        point = points[i]
        point_p = points[i - 1]
        # Calculate the indices for the current point
        N = 4
        start_idx = N * i
        end_idx = start_idx + N
        start_face = 2 * i

        # Create a rectangular mesh around the point
        vertices[start_idx:end_idx, :] = np.array(
            [
                [point_p[0] - width / 2, point_p[1], point_p[2]],
                [point_p[0] + width / 2, point_p[1], point_p[2]],
                [point[0] + width / 2, point[1], point[2]],
                [point[0] - width / 2, point[1], point[2]],
            ]
        )

        # Define the face indices for the current point
        faces[start_face, :] = np.array(
            [
                start_idx + 0,
                start_idx + 1,
                start_idx + 2,
            ]
        )
        faces[start_face + 1, :] = np.array(
            [
                start_idx + 0,
                start_idx + 2,
                start_idx + 3,
            ]
        )

        # Assign the color to the current point
        face_colors[start_idx + 0, :] = color
        face_colors[start_idx + 1, :] = color
        face_colors[start_idx + 2, :] = color
        face_colors[start_idx + 3, :] = color

    return {"vertexes": vertices, "faces": faces, "faceColors": face_colors}


def generate_3D_frame(
    final_points, final_colors, vertexes=[], faces=[], faceColors=[]
):
    meshes = {
        "vertexes": np.array(vertexes),
        "faces": np.array(faces),
        "faceColors": np.array(faceColors),
    }
    return {
        "POINTS": final_points,
        "POINTS_COLOR": final_colors,
        "MESHES": meshes,
    }


def rotate_points(points, angles):
    """
    Rotate the set of points by the given euler angles.

    points: numpy array of shape (N, 3)
        The array containing N points with 3D coordinates (x, y, z).
    a, b, c: float
        The euler angles in degrees

    Returns:
    numpy array of shape (N, 3)
        The rotated points.
    """
    a, b, c = angles

    # Convert the angles from degrees to radians
    a = math.radians(a)
    b = math.radians(b)
    c = math.radians(c)

    # Create the rotation matrices
    rotation_matrix_a = np.array(
        [
            [1, 0, 0],
            [0, math.cos(a), -math.sin(a)],
            [0, math.sin(a), math.cos(a)],
        ]
    )
    rotation_matrix_b = np.array(
        [
            [math.cos(b), 0, math.sin(b)],
            [0, 1, 0],
            [-math.sin(b), 0, math.cos(b)],
        ]
    )
    rotation_matrix_c = np.array(
        [
            [math.cos(c), -math.sin(c), 0],
            [math.sin(c), math.cos(c), 0],
            [0, 0, 1],
        ]
    )

    # Rotate the points using the rotation matrices
    rotated_points = np.dot(points, rotation_matrix_a.T)
    rotated_points = np.dot(rotated_points, rotation_matrix_b.T)
    rotated_points = np.dot(rotated_points, rotation_matrix_c.T)

    return rotated_points


def trajectory_to_3D(trajectory, correction_angle=0.0):
    trajectory_3D = np.zeros((trajectory.shape[0], 3))
    x, y, z = (
        trajectory[:, 0].copy(),
        trajectory[:, 1].copy(),
        # trajectory[:, 2].copy(),
        np.zeros_like(trajectory[:, 0]),
    )
    trajectory_3D[:, 0] = -x
    trajectory_3D[:, 1] = -y
    trajectory_3D[:, 2] = z

    if correction_angle != 0.0:
        trajectory_3D = rotate_points(trajectory_3D, correction_angle)
    return trajectory_3D


def random_subset(points, percentage):
    num_points = points.shape[0]
    num_subset = int(num_points * percentage / 100)
    indices = np.random.choice(num_points, num_subset, replace=False)
    return points[indices]
