"""CLI interface for socc_plotter project.

Demo to run on NuScenes Mini
"""

import os
from typing import Dict, Optional, Tuple, cast

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Mask2FormerForUniversalSegmentation,
)

from .socc import get_multicam_socc
from .socc_helper import (
    get_2D_visual,
    get_future_vehicle_trajectory,
    infer_semantics_and_depth,
)
from .transforms import quaternion_to_transformation_matrix

global nusc
global frame_socc
global frame_data
global trajectory
global scene_index
global current_sample_token


nusc: Optional[NuScenes] = None
frame_socc: Optional[Tuple[np.ndarray, np.ndarray]] = None
frame_data: Optional[Dict] = None
trajectory: Optional[np.ndarray] = None
scene_index: int = 0
current_sample_token: Optional[str] = None


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

    current_sample_token = cast(str, scene["first_sample_token"])

    precompute = False

    def loop():
        global nusc
        global frame_socc
        global frame_data
        global trajectory
        global scene_index
        global current_sample_token

        recompute_socc = False

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print("device", device)
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
        print("Done loading models")

        frame_index = 0

        while True:
            try:
                sample = nusc.get("sample", current_sample_token)
            except KeyError:
                scene_index += 1
                if scene_index >= len(nusc.scene):
                    break
                scene = nusc.scene[scene_index]
                current_sample_token = cast(str, scene["first_sample_token"])
                sample = nusc.get("sample", current_sample_token)

            calibration_data = {}

            for sensor in sensors:
                # PIL image
                cam_data = nusc.get("sample_data", sample["data"][sensor])
                calibrated_sensor_token = cam_data["calibrated_sensor_token"]
                calibration = nusc.get(
                    "calibrated_sensor", calibrated_sensor_token
                )
                translation = calibration["translation"]
                rotation = calibration["rotation"]

                calibration_data[sensor] = quaternion_to_transformation_matrix(
                    rotation,
                    translation,
                )

            cacheroot: str = "data/socc"
            cache_pkl_path = os.path.join(
                cacheroot, current_sample_token + ".pkl"
            )

            if precompute and os.path.exists(cache_pkl_path):
                print("Skipping", current_sample_token)
                current_sample_token = sample["next"]
                continue

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
                cacheroot,
            )

            trajectory = get_future_vehicle_trajectory(
                nusc, cast(str, current_sample_token)
            )

            if recompute_socc:

                frame_socc = get_multicam_socc(
                    sensors,
                    frame_data,
                    calibration_data,
                )
            else:
                frame_socc = frame_data["socc"]

            frame_data["index"] = frame_index

            frame_index += 1

            current_sample_token = sample["next"]
            torch.cuda.empty_cache()
            import time

            time.sleep(0.5)

    if precompute:
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

        if "index" not in frame_data:
            return

        img = get_2D_visual(frame_data, trajectory)
        plot.set_2D_visual(img)

        all_points, all_colors = frame_socc

        all_points, all_colors = all_points.copy(), all_colors.copy()

        plot.set_3D_visual(all_points, all_colors)
        plot.set_3D_trajectory(trajectory, 1.4)

        frame_3d = plot.get_3d_frame()
        frame_3d = cv2.resize(frame_3d, img.shape[:2][::-1])

        vis = np.hstack((img, frame_3d))

        vis = cv2.resize(vis, (0, 0), fx=0.25, fy=0.25)

        cv2.imwrite(
            "data/demo_video/" + f"{frame_data['index']}".zfill(10) + ".png",
            vis,
        )

        # plot.sleep(0.3)

    plotter = Plotter(
        ui_callback=ui_loop,
        compute_callback=loop,
    )
    plotter.start()


if __name__ == "__main__":
    main()
