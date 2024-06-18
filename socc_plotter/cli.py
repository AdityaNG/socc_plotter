"""CLI interface for socc_plotter project.

Demo to run on NuScenes Mini
"""
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, pipeline
from nuscenes.nuscenes import NuScenes

@torch.no_grad()
def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m socc_plotter` and `$ socc_plotter `.

    Runs Semantic Occupancy plotting on the NuScenes dataset
    """

    #########################################################
    # Dataset
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
    scene = nusc.scene[0]

    first_sample_token = scene['first_sample_token']

    sample = nusc.get('sample', first_sample_token)

    sensors = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT',
    ]
    #########################################################
    # Depth
    # Zoe_K
    depth_estimator = pipeline(task="depth-estimation", model="Intel/zoedepth-kitti")
    #########################################################
    # Semantics
    model_mask2former = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
    image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
    #########################################################

    while sample['next'] is not None:

        frame_data = dict()

        for sensor in sensors:
            # PIL image
            cam_data = nusc.get('sample_data', sample['data'][sensor])

            # Depth
            depth_outputs = depth_estimator(cam_data)
            depth = depth_outputs.depth

            # Semantics
            semantics_inputs = image_processor(cam_data, return_tensors="pt")
            semantics_outputs = model_mask2former(**semantics_inputs)

            # (H, W)
            pred_semantic_map = image_processor.post_process_semantic_segmentation(
                semantics_outputs, target_sizes=[cam_data.size[::-1]]
            )[0]


            frame_data[sensor] = {
                'rgb': cam_data,
                'depth': depth,
                'semantics': pred_semantic_map,
            }


        sample = nusc.get('sample', sample['next'])

if __name__ == "__main__":
    main()
