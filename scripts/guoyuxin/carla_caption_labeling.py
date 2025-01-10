import argparse
import base64
import dwm.common
import json
import re
import requests
import tqdm
from dwm.datasets.preview import PreviewDataset
import os
import io
from PIL import Image


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to auto labels the caption with VQA APIs.")
    parser.add_argument(
        "-c", "--config-path",
        type=str, default="/mnt/storage/user/guoyuxin/DWM/scripts/guoyuxin/caption_auto_labelling_carla.json",
        help="The dataset config")
    parser.add_argument(
        "-e", "--endpoint-url",
        type=str, default="http://103.237.29.236:10030/service",
        help="The URL of the VQA API.")
    parser.add_argument("-d", required=True
                        help="仿真carla数据目录下data_notext.json的路径")
    parser.add_argument("-f", "--range-from", type=int, default=0)
    parser.add_argument("-t", "--range-to", type=int, default=-1)

    return parser


def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    json_answer_pattern = re.compile("```json\n(?P<content>(.|\n)*)```")
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    data_jsonl = args.d
    data_dir = "/".join(data_jsonl.split("/")[:-1])

    data_list = []
    with open(data_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    sensor_channels = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT"
    ]

    dataset = PreviewDataset(
        json_file=data_jsonl,
        sequence_length=1,
        fps_stride_tuples=[(0, 1)],
        sensor_channels=sensor_channels,
        enable_camera_transforms=True
    )

    range_to = len(dataset) if args.range_to == -1 else args.range_to
    print(
        "Dataset count: {}, processing range {} - {}".format(
            len(dataset), args.range_from, range_to))

    result = {}

    caption = {}

    for i in tqdm.tqdm(range(args.range_from, range_to)):

        sample = dataset[i]
        images = sample["images"]

        if i % 5 == 0:
            prompt_cache = ["" for _ in range(len(sensor_channels))]

        sample = data_list[i]

        for idx, img in enumerate(images[0]):

            if i % 5 == 0:

                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                image_byte = img_byte_arr.getvalue()
                image_data = base64.b64encode(image_byte).decode("utf-8")

                answers = {}
                for k in config["prompts"]:

                    response = requests.post(
                        args.endpoint_url,
                        data={
                            "question": k["question"],
                            "image": image_data,
                        }).json()

                    if k["key"] is None:
                        json_answer = re\
                            .search(json_answer_pattern, response["answer"])\
                            .group("content")
                        for rk, rv in json.loads(json_answer).items():
                            answers[rk] = rv
                    else:
                        answers[k["key"]] = response["answer"]

                # image_description
                desc = []
                for k, v in answers.items():
                    desc.append(v)

                description = ".".join(desc)

                prompt_cache[idx] = description

            sample["camera_infos"][sensor_channels[idx]
                                   ]["image_description"] = prompt_cache[idx]

        with open(os.path.join(data_dir, "data.json"), "a") as f:
            json.dump(sample, f)
            f.write('\n')
