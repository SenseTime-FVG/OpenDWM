import argparse
import dwm.common
import json
import os
import copy
import torch
import torchvision


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script is designed to convert the nuscenes "
        "dataset into independent data packets, suitable for data "
        "loading in preview.py.")
    parser.add_argument(
        "--reference-frame-count", type=int, required=True,
        help="Save the nums of reference frame.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save data packets.")

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        **dwm.common.instantiate_config(config["validation_dataloader"]))
    print("The validation dataset is loaded with {} items.".format(
        len(validation_dataset)))
    
    sensor_channels = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT"
    ]

    for batch in validation_dataloader:
        scene_name = batch["scene"]["name"][0]
        output_path = os.path.join(args.output_path, scene_name)
        os.makedirs(output_path, exist_ok=True)
        timestamp = 0

        json_path = os.path.join(output_path, "data.json")
        with open(json_path, "w") as json_file: pass

        data_package = dict()
        frame_data = dict()
        for frame in range(batch["vae_images"].shape[1]):
            frame_data["camera_infos"] = dict()
            for sensor_channel in sensor_channels:
                frame_data["camera_infos"][sensor_channel] = dict()

            for view in range(len(sensor_channels)):    
                frame_data["camera_infos"][sensor_channels[view]]["extrin"] = \
                    batch["camera_transforms"][0, frame, view].tolist()
                frame_data["camera_infos"][sensor_channels[view]]["intrin"] = \
                    batch["camera_intrinsics"][0, frame, view].tolist()
                frame_data["camera_infos"][sensor_channels[view]]["image_description"] = \
                    batch["clip_text"][0][frame][view]
                
                if frame % batch["vae_images"].shape[1] < args.reference_frame_count:
                    image_output_path = os.path.join(
                        output_path, sensor_channels[view], "rgb", f"{timestamp}.png")
                    os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                    torchvision.transforms.functional.to_pil_image(
                        batch["vae_images"][0, frame, view]).save(image_output_path)
                    frame_data["camera_infos"][sensor_channels[view]]["rgb"] = \
                        os.path.relpath(image_output_path, output_path)
                else:
                    frame_data["camera_infos"][sensor_channels[view]]["rgb"] = None

                _3dbox_output_path = os.path.join(
                    output_path, sensor_channels[view], "3dbox", f"{timestamp}.png")
                os.makedirs(os.path.dirname(_3dbox_output_path), exist_ok=True)
                torchvision.transforms.functional.to_pil_image(
                    batch["3dbox_images"][0, frame, view]).save(_3dbox_output_path)
                frame_data["camera_infos"][sensor_channels[view]]["3dbox"] = \
                    os.path.relpath(_3dbox_output_path, output_path)

                hdmap_output_path = os.path.join(
                    output_path, sensor_channels[view], "hdmap", f"{timestamp}.png")
                os.makedirs(os.path.dirname(hdmap_output_path), exist_ok=True)
                torchvision.transforms.functional.to_pil_image(
                    batch["hdmap_images"][0, frame, view]).save(hdmap_output_path)
                frame_data["camera_infos"][sensor_channels[view]]["hdmap"] = \
                    os.path.relpath(hdmap_output_path, output_path)

            frame_data["timestamp"] = timestamp 
            timestamp += 1/int(batch["fps"])
            timestamp = round(timestamp, 4)
            frame_data["ego_pose"] = batch["ego_transforms"][0, frame, 0].tolist()
            data_package[frame] = copy.deepcopy(frame_data)

        with open(json_path, "a") as json_file:
            json.dump(data_package, json_file, indent=4)
            json_file.write("\n")
        