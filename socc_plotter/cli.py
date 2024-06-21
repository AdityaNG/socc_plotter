"""CLI interface for socc_plotter project.

Demo to run on NuScenes Mini
"""

import os
from typing import Dict, Optional, Tuple, cast

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Mask2FormerForUniversalSegmentation,
)

from .colormap import create_cityscapes_label_colormap
from .socc import get_socc
from .socc_helper import (
    get_2D_visual,
    get_future_vehicle_trajectory,
    infer_semantics_and_depth,
    semantic_to_rgb,
)
from .transforms import quart_to_transformation_matrix

global nusc
global frame_socc
global frame_data
global trajectory
global scene_index
global current_sample_token
# global depth_estimator
# global model_mask2former
# global image_processor


nusc: Optional[NuScenes] = None
frame_socc: Optional[Tuple[np.ndarray, np.ndarray]] = None
frame_data: Optional[Dict] = None
trajectory: Optional[np.ndarray] = None
scene_index: int = 0
current_sample_token: Optional[str] = None
# depth_estimator: Optional[AutoModelForDepthEstimation] = None
# model_mask2former: Optional[Mask2FormerForUniversalSegmentation] = None
# image_processor: Optional[AutoImageProcessor] = None


@torch.no_grad()
def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m socc_plotter` and `$ socc_plotter `.

    Runs Semantic Occupancy plotting on the NuScenes dataset
    """
    from socc_plotter.plotter import Plotter

    global nusc
    global frame_socc
    global frame_data
    global scene_index
    global current_sample_token
    # global depth_estimator
    # global model_mask2former, image_processor

    #########################################################
    # Dataset
    dataroot = os.path.abspath("./data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    scene_index = 0
    scene = nusc.scene[scene_index]

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
            print(f"rotation[{sensor}]", rotation)
            calibration_data[channel] = quart_to_transformation_matrix(
                rotation,
                translation,
            )

    current_sample_token = cast(str, scene["first_sample_token"])

    def loop():
        global nusc
        global frame_socc
        global frame_data
        global trajectory
        global scene_index
        global current_sample_token

        # global depth_estimator
        # global model_mask2former, image_processor

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
        model_mask2former = (
            Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-large-cityscapes-semantic",
                device_map=device,
            )
        )
        image_processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-large-cityscapes-semantic",
            device=device,
        )
        #########################################################

        while True:
            try:
                sample = nusc.get("sample", current_sample_token)
            except KeyError:
                scene_index += 1
                if scene_index >= len(nusc.scene):
                    exit()
                scene = nusc.scene[scene_index]
                current_sample_token = cast(str, scene["first_sample_token"])
                sample = nusc.get("sample", current_sample_token)

            print("current_sample_token", current_sample_token)

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

            trajectory = get_future_vehicle_trajectory(
                nusc, cast(str, current_sample_token)
            )

            all_points_l = []
            all_colors_l = []
            colormap = create_cityscapes_label_colormap()

            # for sensor in list(frame_data.keys())[:1]:
            for sensor in sensors[:1]:
                # for sensor in [sensors[i] for i in [0, 1, 5]]:
                # for sensor in [
                #     "CAM_FRONT",
                #     "CAM_FRONT_RIGHT",
                #     "CAM_FRONT_LEFT",
                # ]:
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
                # points_rot = points @ calibration_data[sensor]
                points_rot = points @ np.linalg.inv(calibration_data[sensor])
                # points_rot = (
                #     np.linalg.inv(calibration_data[sensor]) @ points.T
                # ).T
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

            frame_socc = (all_points, all_colors)

            current_sample_token = sample["next"]
            torch.cuda.empty_cache()

    loop()

    def ui_loop(plot: Plotter):
        global frame_socc
        global frame_data
        global trajectory

        if frame_data is None:
            return

        if trajectory is None:
            return

        if frame_socc is None:
            return

        img = get_2D_visual(frame_data, trajectory)
        plot.set_2D_visual(img)

        all_points, all_colors = frame_socc

        all_points, all_colors = all_points.copy(), all_colors.copy()

        plot.set_3D_visual(all_points, all_colors)
        plot.set_3D_trajectory(trajectory, 1.4)

        # plot.sleep(0.3)

    plotter = Plotter(
        ui_callback=ui_loop,
        compute_callback=loop,
    )
    plotter.start()


if __name__ == "__main__":
    main()
