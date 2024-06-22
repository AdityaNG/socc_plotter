"""CLI interface for socc_plotter project.

Demo to run on NuScenes Mini
"""

import os
import pickle
from typing import Callable, Dict, List

import cv2
import numpy as np
import torch
from general_navigation.models.model_utils import plot_steering_traj
from nuscenes.nuscenes import NuScenes
from PIL import Image
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)

from .colormap import create_cityscapes_label_colormap
from .socc import get_multicam_socc, get_socc, semantic_to_rgb
from .transforms import create_transformation_matrix


def get_future_vehicle_trajectory(
    nusc: NuScenes, current_sample_token: str, future_horizon: int = 6
) -> np.ndarray:
    """
    Extracts the future vehicle trajectory starting from a given sample token.

    :param nusc: NuScenes object.
    :param current_sample_token: Token of the current sample to start from
    :param future_horizon: Number of future time steps to consider.
    :return: A dictionary with vehicle trajectories.
    """
    vehicle_trajectory = []

    sample_token = current_sample_token

    for _ in range(future_horizon):
        if not sample_token:
            break

        sample = nusc.get("sample", sample_token)
        cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose_token = cam_front_data["ego_pose_token"]

        ego_pose = nusc.get("ego_pose", ego_pose_token)
        pos = ego_pose["translation"]
        rot = ego_pose["rotation"]

        vehicle_pose = create_transformation_matrix(pos, rot)

        vehicle_trajectory.append(vehicle_pose)

        sample_token = sample["next"]  # Move to the next sample in the scene.

    vehicle_trajectory_np = np.array(vehicle_trajectory)

    origin = vehicle_trajectory_np[0]
    vehicle_trajectory_np = np.linalg.inv(origin) @ vehicle_trajectory_np

    vehicle_trajectory_xy = vehicle_trajectory_np[:, [1, 0], 3]

    return vehicle_trajectory_xy


def depth_to_rgb(depth_map):
    depth_min, depth_max = np.min(depth_map), np.max(depth_map)
    if depth_max > depth_min:  # Avoid division by zero
        normalized_depth_map = (
            (depth_map - depth_min) / (depth_max - depth_min) * 255.0
        )
    else:
        normalized_depth_map = np.zeros_like(depth_map)

    # Convert to uint8
    normalized_depth_map_uint8 = np.uint8(normalized_depth_map)

    # Apply JET colormap
    color_mapped_image = cv2.applyColorMap(
        normalized_depth_map_uint8, cv2.COLORMAP_JET
    )

    return color_mapped_image


def infer_semantics_and_depth(
    nusc: NuScenes,
    sensors: List,
    sample: Dict,
    current_sample_token: str,
    dataroot: str,
    depth_estimator: AutoImageProcessor,
    depth_transform: Callable,
    device: torch.device,
    calibration_data: Dict,
    model_mask2former: Mask2FormerForUniversalSegmentation,
    image_processor: AutoImageProcessor,
    cacheroot: str = "data/socc",
) -> Dict:
    frame_data = dict()
    cache_pkl_path = os.path.join(cacheroot, current_sample_token + ".pkl")
    if not os.path.exists(cache_pkl_path):
        colormap = create_cityscapes_label_colormap()

        for sensor in sensors:
            # PIL image
            cam_data = nusc.get("sample_data", sample["data"][sensor])

            rgb_img_pil = Image.open(
                os.path.join(dataroot, cam_data["filename"])
            )
            rgb_img_np = np.array(rgb_img_pil)
            rgb_img_np = cv2.cvtColor(rgb_img_np, cv2.COLOR_RGB2BGR)

            # Depth
            # depth_outputs = depth_estimator(rgb_img_pil)
            # depth = depth_outputs.depth
            # depth_np = depth_estimator.infer_pil(rgb_img_pil)  # as numpy
            depth_input_batch = depth_transform(
                images=rgb_img_pil, return_tensors="pt"
            )
            depth_input_batch["pixel_values"] = depth_input_batch[
                "pixel_values"
            ].to(device=device)
            depth_prediction = depth_estimator(
                **depth_input_batch
            ).predicted_depth

            depth_prediction = torch.nn.functional.interpolate(
                depth_prediction.unsqueeze(1),
                size=rgb_img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # Semantics
            semantics_inputs = image_processor(
                rgb_img_pil, return_tensors="pt"
            )
            semantics_inputs["pixel_values"] = semantics_inputs[
                "pixel_values"
            ].to(device=device)
            semantics_outputs = model_mask2former(**semantics_inputs)

            # (H, W)
            pred_semantic_map = (
                image_processor.post_process_semantic_segmentation(
                    semantics_outputs, target_sizes=[rgb_img_pil.size[::-1]]
                )[0]
            )

            depth_np = depth_prediction.cpu().numpy()
            pred_semantic_map_np = pred_semantic_map.cpu().numpy()

            pred_semantic_map_np_rgb = semantic_to_rgb(
                pred_semantic_map_np, colormap
            )

            socc = get_socc(depth_np, pred_semantic_map_np_rgb)

            depth_np_rgb = depth_to_rgb(depth_np)

            frame_data[sensor] = {
                "rgb": rgb_img_np,
                "depth": depth_np,
                "depth_rgb": depth_np_rgb,
                "semantics": pred_semantic_map_np,
                "semantics_rgb": pred_semantic_map_np_rgb,
                "socc": socc,
            }

        frame_data["socc"] = get_multicam_socc(
            sensors,
            frame_data,
            calibration_data,
        )  # type: ignore
        # Save frame_data
        with open(cache_pkl_path, "wb") as handle:
            pickle.dump(frame_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load frame_data
        with open(cache_pkl_path, "rb") as handle:
            frame_data = pickle.load(handle)

    return frame_data


def overlay_semantics(cam_data: Dict) -> np.ndarray:
    semantics = cam_data["semantics_rgb"].astype(np.uint8)
    semantics = cv2.cvtColor(semantics, cv2.COLOR_BGR2RGB)

    cam_frame = cv2.addWeighted(
        cam_data["rgb"].astype(np.uint8), 1.0, semantics, 0.2, 0.0
    )

    return cam_frame


def get_2D_visual(frame_data: Dict, trajectory: np.ndarray):
    # Get the dimensions of the images (assuming all are the same size)
    image_height, image_width, _ = frame_data["CAM_FRONT"]["rgb"].shape

    # Define the layout dimensions based on the input image size
    # Each row will use the height of the image, and we'll double the height
    # for the top and bottom cameras
    row_height = image_height
    double_row_height = 2 * image_height
    column_width = image_width
    # blank_width = image_width

    # Calculate total layout dimensions
    layout_height = 6 * row_height  # Total height for 6 rows
    layout_width = 2 * column_width  # Total width for 2 columns

    # Create a blank layout image
    layout_image = np.zeros((layout_height, layout_width, 3), dtype=np.uint8)

    front_frame = overlay_semantics(frame_data["CAM_FRONT"])

    front_frame = plot_steering_traj(
        front_frame,
        trajectory,
        # color=(255, 255, 255),
        color=(0, 255, 0),
        offsets=(0, -1.4, 1.5),
        # offsets=(0, 0.2, 0.0),
        track=True,
        track_width=1.4,
        fov_x=70,
        fov_y=70,
    )

    # Place CAM_FRONT at the top spanning two rows
    layout_image[:double_row_height, :] = cv2.resize(
        front_frame,
        (layout_width, double_row_height),
    )

    # Place CAM_FRONT_LEFT and CAM_FRONT_RIGHT in the third row
    layout_image[
        double_row_height : double_row_height + row_height, :column_width
    ] = overlay_semantics(frame_data["CAM_FRONT_LEFT"])
    layout_image[
        double_row_height : double_row_height + row_height, column_width:
    ] = overlay_semantics(frame_data["CAM_FRONT_RIGHT"])

    # Place CAM_BACK_LEFT and CAM_BACK_RIGHT in the fourth row
    layout_image[
        double_row_height + row_height : double_row_height + 2 * row_height,
        :column_width,
    ] = overlay_semantics(frame_data["CAM_BACK_LEFT"])
    layout_image[
        double_row_height + row_height : double_row_height + 2 * row_height,
        column_width:,
    ] = overlay_semantics(frame_data["CAM_BACK_RIGHT"])

    # Place CAM_BACK at the bottom spanning two rows
    layout_image[double_row_height + 2 * row_height :, :] = cv2.resize(
        overlay_semantics(frame_data["CAM_BACK"]),
        (layout_width, double_row_height),
    )

    return layout_image
