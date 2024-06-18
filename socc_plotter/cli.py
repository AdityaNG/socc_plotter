"""CLI interface for socc_plotter project.

Demo to run on NuScenes Mini
"""

import os

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)

from .colormap import create_ade20k_label_colormap


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


def semantic_to_rgb(pred_semantic_map):
    # ADE20k color palette
    ade20k_palette = create_ade20k_label_colormap()

    # Convert segmentation map to color map
    color_map = ade20k_palette[pred_semantic_map]

    return color_map


@torch.no_grad()
def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m socc_plotter` and `$ socc_plotter `.

    Runs Semantic Occupancy plotting on the NuScenes dataset
    """

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

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    #########################################################
    # Depth
    # Zoe_K
    depth_estimator = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    depth_transform = midas_transforms.dpt_transform
    depth_estimator = depth_estimator.to(device=device)
    #########################################################
    # Semantics
    model_mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-ade-semantic"
    )
    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-small-ade-semantic"
    )
    #########################################################

    current_sample_token = scene["first_sample_token"]

    def loop():
        global current_sample_token
        global depth_estimator
        global model_mask2former, image_processor

        sample = nusc.get("sample", current_sample_token)

        frame_data = dict()

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
            depth_input_batch = depth_transform(rgb_img_np).to(device)
            depth_prediction = depth_estimator(depth_input_batch)

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
            semantics_outputs = model_mask2former(**semantics_inputs)

            # (H, W)
            pred_semantic_map = (
                image_processor.post_process_semantic_segmentation(
                    semantics_outputs, target_sizes=[rgb_img_pil.size[::-1]]
                )[0]
            )

            pred_semantic_map_np = pred_semantic_map.cpu().numpy()
            depth_np = depth_prediction.cpu().numpy()

            pred_semantic_map_np = semantic_to_rgb(pred_semantic_map_np)
            depth_np = depth_to_rgb(depth_np)

            print("rgb_img_np", rgb_img_np.shape, rgb_img_np.dtype)
            print("depth_np", depth_np.shape, depth_np.dtype)
            print(
                "pred_semantic_map_np",
                pred_semantic_map_np.shape,
                pred_semantic_map_np.dtype,
            )

            frame_data[sensor] = {
                "rgb": rgb_img_np,
                "depth": depth_np,
                "semantics": pred_semantic_map_np,
            }

        vis = np.hstack(
            (
                frame_data["CAM_FRONT"]["rgb"],
                frame_data["CAM_FRONT"]["depth"],
                frame_data["CAM_FRONT"]["semantics"],
            )
        )
        vis = cv2.resize(vis, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow("ui", vis)

        key = cv2.waitKey(1)

        if key == ord("q"):
            exit(0)

        current_sample_token = sample["next"]
        if current_sample_token is None:
            current_sample_token = scene["first_sample_token"]

    while True:
        loop()


if __name__ == "__main__":
    main()
