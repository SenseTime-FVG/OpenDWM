import argparse
import av
import dwm.common
import fractions
import json
import numpy as np
import os


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to make OpenDV meta file on given videos.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The path of the config file.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the meta file in JSON format.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    fs = dwm.common.create_instance_from_config(config["fs"])
    file_paths = fs.ls("", detail=False)

    rs = np.random.RandomState(config["seed"])
    score_for_validation = rs.permutation(len(file_paths))

    meta_list = []
    for i, file_path in enumerate(file_paths):
        is_validation = score_for_validation[i] < config["validation_count"]
        meta_item = {
            "id": i + 1,
            "videoid": os.path.splitext(file_path)[0],
            "split": "Val" if is_validation else "Train",
            "start_discard": 0,
            "end_discard": 0
        }
        with fs.open(file_path) as f:
            with av.open(f) as container:
                s = container.streams.video[0]
                if s.duration is None:
                    length = container.duration // 1000000
                else:
                    length = int(s.duration * s.time_base)

                meta_item["length"] = length

        meta_list.append(meta_item)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2)
