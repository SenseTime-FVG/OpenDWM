import argparse
import base64
import dwm.common
import dwm.datasets.common
import dwm.fs.czip
import io
import json
import pickle
from PIL import Image
import re
import requests
import torch
import tqdm
import urllib3


class ImageRawDataset(torch.utils.data.Dataset):

    def __init__(
        self, indices: dict, fs: dwm.fs.czip.CombinedZipFileSystem,
        sensor_channels: list, range_from: int, range_to: int
    ):
        self.fs = fs
        items = [
            {
                "token": "{}image_undistort/{}/{}".format(scene_k, sensor_channel, image_name),
                "path": "{}/gac_baidu_segments_batch_{}/{}image_undistort/{}/{}".format(
                    bucket_k[5:], batch_k, scene_k, sensor_channel, image_name)
            }
            for bucket_k, bucket_v in indices.items()
            for batch_k, batch_v in bucket_v.items()
            for scene_k, scene_v in batch_v.items()
            for sensor_channel in sensor_channels
            for i_id, image_name in enumerate(scene_v["full_imgs"])
            if i_id % 5 == 0
        ]

        range_to = len(items) if range_to == -1 else range_to
        print(
            "Dataset count: {}, processing range {} - {}".format(
                len(items), range_from, range_to))

        self.items = items[range_from:range_to]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        try:
            data = self.fs.cat_file(item["path"])
        except:
            print("{} fails to download".format(item["path"]))
            image = Image.new("RGB", (640, 360))
            with io.BytesIO() as f:
                image.save(f, format="JPEG")
                data = f.getvalue()

        return {
            "token": item["token"],
            "image_data": data
        }


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to auto labels the caption with VQA APIs.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The dataset config.")
    parser.add_argument(
        "-e", "--endpoint-url", type=str,
        default="http://112.111.7.64:10067/cabin_intern/service",
        help="The URL of the VQA API.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the dict of image ID and labelled "
        "caption.")
    parser.add_argument("-f", "--range-from", type=int, default=0)
    parser.add_argument("-t", "--range-to", type=int, default=-1)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    json_answer_pattern = re.compile("```json\n(?P<content>(.|\n)*)```")
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(config["indices_file"], "rb") as f:
        indices = pickle.load(f)

    fs = dwm.common.create_instance_from_config(config["fs"])
    dataset = ImageRawDataset(
        indices, fs, config["sensor_channels"], args.range_from, args.range_to)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        num_workers=0,
        collate_fn=dwm.datasets.common.CollateFnIgnoring(
            ["token", "image_data"]))

    session = requests.Session()
    retries = urllib3.util.Retry(total=32, allowed_methods={"POST"})
    http_adapter = requests.adapters.HTTPAdapter(max_retries=retries)
    session.mount("http://", http_adapter)

    result = {}
    for item in tqdm.tqdm(dataloader):

        token = item["token"][0]
        image_data = item["image_data"][0]
        answers = {}
        for k in config["prompts"]:
            response = session.post(
                args.endpoint_url,
                data={
                    "question": k["question"],
                    "image": base64.b64encode(image_data).decode("utf-8"),
                }).json()

            if k["key"] is None:
                json_answer_search = re.search(
                    json_answer_pattern, response["answer"])
                if json_answer_search is None:
                    answers["time"] = answers["weather"] = \
                        answers["environment"] = answers["objects"] = ""
                    print("Warning: no JSON answer of {}".format(token))
                else:
                    try:
                        json_answer = json_answer_search.group("content")
                        for rk, rv in json.loads(json_answer).items():
                            answers[rk] = rv
                    except:
                        answers["time"] = answers["weather"] = \
                            answers["environment"] = answers["objects"] = ""
                        print("Warning: wrong JSON format of {}".format(token))

            else:
                answers[k["key"]] = response["answer"]

        result[token] = answers

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, **config["dump_options"])
