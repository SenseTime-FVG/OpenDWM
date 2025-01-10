import argparse
import dwm.common
from PIL import Image
import json
import pickle
import tqdm


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True)
    parser.add_argument(
        "-c", "--config-path", type=str, required=True)
    parser.add_argument(
        "-f", "--range-from", type=int, default=0)
    parser.add_argument(
        "-t", "--range-to", type=int, default=-1)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    channels = [
        "left_front_camera",
        "center_camera_fov120",
        "right_front_camera",
        "right_rear_camera",
        "rear_camera",
        "left_rear_camera"
    ]

    with open(args.input_path, "rb") as f:
        meta = pickle.load(f)

    with open(args.config_path, "r", encoding="utf-8") as f:
        fs = dwm.common.create_instance_from_config(json.load(f))

    range_to = len(meta["infos"]) if args.range_to == -1 else args.range_to
    print(
        "Total {}, range {} - {}".format(
            len(meta["infos"]), args.range_from, range_to))
    for i in tqdm.tqdm(meta["infos"][args.range_from:range_to]):
        with fs.open(i["token"][70:], "r", encoding="utf-8") as f:
            sample = json.load(f)

        for camera_name in channels:
            j = sample["cams"][camera_name]
            try:
                with fs.open(j["data_path"][5:], "rb") as f:
                    image = Image.open(f)
                    image.load()
            except:
                print(
                    "{} {} {} broken".format(
                        i["scene_token"], i["token"], camera_name))
