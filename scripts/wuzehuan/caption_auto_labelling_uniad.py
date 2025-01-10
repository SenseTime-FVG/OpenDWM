import argparse
import base64
import dwm.common
import dwm.datasets.common
import fsspec
import json
import pickle
import re
import requests
import torch
import tqdm
import urllib3.util


class MultiviewRawDataset(torch.utils.data.Dataset):

    def __init__(
        self, pkl_path: str, fs: fsspec.AbstractFileSystem,
        stride: int, range_from: int, range_to: int,
        sensor_channels: list = ["center_camera_fov120"]
    ):
        self.fs = fs
        self.sensor_channels = sensor_channels
        with open(pkl_path, "rb") as f:
            raw_indices = pickle.load(f)

        indices = {"infos": []}
        current_scene = None
        stride_flag = 0
        for i in raw_indices["infos"]:
            if i["scene_token"] != current_scene:
                current_scene = i["scene_token"]
                stride_flag = 0

            if stride_flag % stride == 0:
                indices["infos"].append(i)

            stride_flag += 1

        del raw_indices

        range_to = len(indices["infos"]) if range_to == -1 else range_to
        print(
            "Dataset count: {}, processing range {} - {}".format(
                len(indices["infos"]), range_from, range_to))

        self.infos = indices["infos"][range_from:range_to]

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index: int):
        item = self.infos[index]

        with open(item["token"], "r", encoding="utf-8") as f:
            sample = json.load(f)

        token_list = []
        image_data_list = []
        for j in self.sensor_channels:
            sample_data = sample["cams"][j]
            token_list.append("{}${}".format(item["token"], j))
            image_data_list.append(fs.cat_file(sample_data["data_path"][5:]))

        return {
            "token_list": token_list,
            "image_data_list": image_data_list
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

    fs = dwm.common.create_instance_from_config(config["fs"])
    dataset = MultiviewRawDataset(
        config["pkl"], fs, config["stride"], args.range_from, args.range_to,
        config["sensor_channels"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1, prefetch_factor=3,
        collate_fn=dwm.datasets.common.CollateFnIgnoring(
            ["token_list", "image_data_list"]))

    session = requests.Session()
    retries = urllib3.util.Retry(total=32, allowed_methods={"POST"})
    http_adapter = requests.adapters.HTTPAdapter(max_retries=retries)
    session.mount("http://", http_adapter)

    result = {}
    for item in tqdm.tqdm(dataloader):

        for token, image_data in zip(
            item["token_list"][0], item["image_data_list"][0]
        ):
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
                        json_answer = json_answer_search.group("content")
                        for rk, rv in json.loads(json_answer).items():
                            answers[rk] = rv
                else:
                    answers[k["key"]] = response["answer"]

            result[token] = answers

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, **config["dump_options"])
