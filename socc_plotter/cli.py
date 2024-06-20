"""CLI interface for socc_plotter project.

Demo to run on NuScenes Mini
"""

import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Mask2FormerForUniversalSegmentation,
)

from .colormap import create_ade20k_label_colormap, socc_label_colormap


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


def semantic_to_rgb(pred_semantic_map, palette=create_ade20k_label_colormap()):
    # Convert segmentation map to color map
    color_map = palette[pred_semantic_map]

    return color_map


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


def get_socc(
    disparity: np.ndarray,
    semantics: np.ndarray,
    scale: Tuple[float, float, float] = (1, 1, -1),
    intrinsics: Optional[np.ndarray] = None,
    subsample: int = 10,
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
    baseline = 2.0
    points = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

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

    points = points.reshape(HEIGHT * WIDTH, 3)
    colors = colors.reshape(HEIGHT * WIDTH, 3)

    # subsample
    points = points[::subsample, :]
    colors = colors[::subsample, :]

    points = points[:, [0, 2, 1]]

    return (points, colors)


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
    print("current_sample_token", current_sample_token)
    cache_pkl_path = os.path.join(cacheroot, current_sample_token + ".pkl")
    if not os.path.exists(cache_pkl_path):
        colormap = socc_label_colormap()

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

            print("rgb_img_np", rgb_img_np.shape, rgb_img_np.dtype)
            print("depth_np", depth_np.shape, depth_np.dtype)
            print(
                "pred_semantic_map_np",
                pred_semantic_map_np_rgb.shape,
                pred_semantic_map_np_rgb.dtype,
            )

            frame_data[sensor] = {
                "rgb": rgb_img_np,
                "depth": depth_np,
                "depth_rgb": depth_np_rgb,
                "semantics": pred_semantic_map_np,
                "semantics_rgb": pred_semantic_map_np_rgb,
                "socc": socc,
            }

        all_points_l = []
        all_colors_l = []
        # for sensor in list(frame_data.keys())[:1]:
        for sensor in frame_data:
            # depth = frame_data[sensor]["depth"]
            # semantics = frame_data[sensor]["semantics"]
            # semantics_rgb = semantic_to_rgb(semantics, colormap)
            socc = frame_data[sensor]["socc"]

            points = socc[0]
            ones_column = np.ones((points.shape[0], 1))
            points = np.hstack((points, ones_column))
            # points_rot = (
            #     calibration_data[sensor] @ points.T
            # ).T
            points_rot = points @ calibration_data[sensor]
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

        frame_data["socc"] = (all_points, all_colors)  # type: ignore
        # Save frame_data
        with open(cache_pkl_path, "wb") as handle:
            pickle.dump(frame_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load frame_data
        with open(cache_pkl_path, "rb") as handle:
            frame_data = pickle.load(handle)

    return frame_data


def get_2D_visual(frame_data: Dict):
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

    # Place CAM_FRONT at the top spanning two rows
    layout_image[:double_row_height, :] = cv2.resize(
        frame_data["CAM_FRONT"]["depth_rgb"], (layout_width, double_row_height)
    )

    # Place CAM_FRONT_LEFT and CAM_FRONT_RIGHT in the third row
    layout_image[
        double_row_height : double_row_height + row_height, :column_width
    ] = frame_data["CAM_FRONT_LEFT"]["rgb"]
    layout_image[
        double_row_height : double_row_height + row_height, column_width:
    ] = frame_data["CAM_FRONT_RIGHT"]["rgb"]

    # Place CAM_BACK_LEFT and CAM_BACK_RIGHT in the fourth row
    layout_image[
        double_row_height + row_height : double_row_height + 2 * row_height,
        :column_width,
    ] = frame_data["CAM_BACK_LEFT"]["rgb"]
    layout_image[
        double_row_height + row_height : double_row_height + 2 * row_height,
        column_width:,
    ] = frame_data["CAM_BACK_RIGHT"]["rgb"]

    # Place CAM_BACK at the bottom spanning two rows
    layout_image[double_row_height + 2 * row_height :, :] = cv2.resize(
        frame_data["CAM_BACK"]["rgb"], (layout_width, double_row_height)
    )

    return layout_image


def transformation_matrix(
    quaternion: List[float], translation: Tuple[float, float, float]
) -> np.ndarray:
    # Create a Quaternion object from the list
    q = Quaternion(quaternion)

    # Convert quaternion to a 3x3 rotation matrix
    R = q.rotation_matrix

    # Create the 4x4 transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = translation

    return T


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


global nusc
global current_sample_token
global depth_estimator
global model_mask2former
global image_processor


nusc: Optional[NuScenes] = None
current_sample_token: Optional[str] = None
depth_estimator: Optional[AutoModelForDepthEstimation] = None
model_mask2former: Optional[Mask2FormerForUniversalSegmentation] = None
image_processor: Optional[AutoImageProcessor] = None


@torch.no_grad()
def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m socc_plotter` and `$ socc_plotter `.

    Runs Semantic Occupancy plotting on the NuScenes dataset
    """
    from socc_plotter.plotter import Plotter

    global nusc
    global current_sample_token
    global depth_estimator
    global model_mask2former, image_processor

    #########################################################
    # Dataset
    dataroot = os.path.abspath("./data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    scene = nusc.scene[0]

    sensors = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]

    calibration_data = {}

    for sensor, calibration in zip(nusc.sensor, nusc.calibrated_sensor):
        channel = sensor["channel"]
        if channel in sensors:
            translation = calibration["translation"]
            rotation = calibration["rotation"]
            print("rotation", rotation)
            calibration_data[channel] = transformation_matrix(
                rotation,
                translation,
            )
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    #########################################################
    # Depth
    # DepthAnything
    depth_transform = AutoImageProcessor.from_pretrained(
        "LiheYoung/depth-anything-large-hf",
        device=device,
    )
    depth_estimator = AutoModelForDepthEstimation.from_pretrained(
        "LiheYoung/depth-anything-large-hf",
        device_map=device,
    )
    #########################################################
    # Semantics
    model_mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-ade-semantic",
        device_map=device,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-ade-semantic",
        device=device,
    )
    #########################################################

    current_sample_token = cast(str, scene["first_sample_token"])

    def loop(plot: Plotter):
        global nusc
        global current_sample_token
        global depth_estimator
        global model_mask2former, image_processor

        try:
            sample = nusc.get("sample", current_sample_token)
        except KeyError:
            current_sample_token = cast(str, scene["first_sample_token"])
            sample = nusc.get("sample", current_sample_token)

        frame_data = infer_semantics_and_depth(
            nusc,
            sensors,
            sample,
            cast(str, current_sample_token),
            dataroot,
            depth_estimator,
            depth_transform,
            device,
            calibration_data,
            model_mask2former,
            image_processor,
        )

        img = get_2D_visual(frame_data)
        plot.set_2D_visual(img)

        # socc = frame_data["socc"]
        # all_points, all_colors = socc

        all_points_l = []
        all_colors_l = []
        colormap = socc_label_colormap()

        # for sensor in list(frame_data.keys())[:1]:
        for sensor in sensors[:1]:
            depth = frame_data[sensor]["depth"]
            semantics = frame_data[sensor]["semantics"]
            semantics_rgb = semantic_to_rgb(semantics, colormap)

            socc = get_socc(depth, semantics_rgb)

            points = socc[0]
            ones_column = np.ones((points.shape[0], 1))
            points = np.hstack((points, ones_column))
            # points_rot = (
            #     calibration_data[sensor] @ points.T
            # ).T
            points_rot = points @ calibration_data[sensor]
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

        plot.set_3D_visual(all_points, all_colors)

        print("all_points", all_points.shape)

        current_sample_token = sample["next"]
        if current_sample_token is None:
            current_sample_token = cast(str, scene["first_sample_token"])

        plot.sleep(0.3)
        # plot.sleep(0.01)

    plotter = Plotter(
        callback=loop,
    )
    plotter.start()


if __name__ == "__main__":
    main()
