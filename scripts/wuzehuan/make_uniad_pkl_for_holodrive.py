import argparse
import json
import pickle


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to make index file for the ZIP file.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The path (local file system) of file system config file to read "
        "the ZIP file.")
    parser.add_argument(
        "-ot", "--output-train-pkl-path", type=str, required=True,
        help="The output path to save the pkl file of train split.")
    parser.add_argument(
        "-ov", "--output-val-pkl-path", type=str, required=True,
        help="The output path to save the pkl file of validation split.")
    return parser


def get_scenes_name(samples, filter_datatime="20231222", scenes_to_exclude=[]):
    scene_dict = {}
    for i, sample in enumerate(samples):
        if "".join(sample["scene_token"].split("_", 3)[:3]) < filter_datatime:
            continue

        if sample["scene_token"] in scenes_to_exclude:
            continue

        if sample["scene_token"] not in scene_dict:
            scene_dict[sample["scene_token"]] = [i]
        else:
            scene_dict[sample["scene_token"]].append(i)

    scene_names = list(scene_dict.keys())

    return scene_names


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    info_list = [
        j for i in config["pkl_list"]
        for j in pickle.load(open(i, "rb"))["infos"]
    ]
    scene_names = get_scenes_name(
        info_list, scenes_to_exclude=config["scenes_to_exclude"])

    if "train_blob_list" in config:
        train_scene_indices = set([
            j for i in config["train_blob_list"]
            for j in range(
                i * config["scene_count_per_blob"],
                min((i + 1) * config["scene_count_per_blob"], len(scene_names)))
        ])
    else:
        train_scene_indices = None

    val_scene_indices = set([
        j for i in config["validation_blob_list"]
        for j in range(
            i * config["scene_count_per_blob"],
            min((i + 1) * config["scene_count_per_blob"], len(scene_names)))
    ])
    scenes_marked_blur = set(config["scenes_marked_blur"])

    train_scenes = set([
        scene_name for i, scene_name in enumerate(scene_names)
        if (train_scene_indices is None and i not in val_scene_indices) or \
            (train_scene_indices is not None and i in train_scene_indices)
    ])
    val_scenes = set([
        scene_name for i, scene_name in enumerate(scene_names)
        if i in val_scene_indices
    ])

    with open(args.output_train_pkl_path, "wb") as f:
        train_infos = [
            i for i in info_list
            if i["scene_token"] in train_scenes and
            # in case of non-standard scene token
            i["token"].split("/")[-2] not in scenes_marked_blur
        ]
        pickle.dump({"infos": train_infos}, f)
        print("Count of train split: {}".format(len(train_infos)))

    with open(args.output_val_pkl_path, "wb") as f:
        val_infos = [
            i for i in info_list if i["scene_token"] in val_scenes and
            # in case of non-standard scene token
            i["token"].split("/")[-2] not in scenes_marked_blur
        ]
        pickle.dump({"infos": val_infos}, f)
        print("Count of validation split: {}".format(len(val_infos)))
