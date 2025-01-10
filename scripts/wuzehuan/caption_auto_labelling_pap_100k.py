import argparse
import av
import base64
import fractions
import fsspec
import dwm.common
import dwm.datasets.common
import io
import json
import re
import requests
import torch
import tqdm
import urllib3


class ImageBytesFromVideoDataset(torch.utils.data.Dataset):

    @staticmethod
    def read_image_with_pts_from_video(
        fs: fsspec.AbstractFileSystem, item: dict,
    ):
        video_path = "{}/camera/{}.mp4".format(item["scene"], item["camera"])
        pts_ms = item["time"]

        frames = []
        tb_ms = fractions.Fraction(1, 1000)
        with fs.open(video_path) as f:
            with av.open(f) as container:
                stream = container.streams.video[0]
                pts_stream = int(pts_ms * tb_ms / stream.time_base)
                container.seek(pts_stream, stream=stream)
                for i in container.decode(stream):

                    # rescale the pts to the unit of ms
                    i.pts = int(i.pts * stream.time_base / tb_ms)

                    if i.pts < pts_ms:
                        continue

                    frames.append(i)
                    if i.pts > pts_ms:
                        break

                stream.codec_context.close()
                stream = None

        pts_list = [i.pts for i in frames]
        actual_pts_index = dwm.datasets.common.find_nearest(pts_list, pts_ms)
        actual_image = frames[actual_pts_index].to_image()
        actual_pts = pts_list[actual_pts_index]
        return actual_image, actual_pts

    def __init__(
        self, scene_indices_path: str, fs: fsspec.AbstractFileSystem,
        range_from: int, range_to: int, interval: int = 500
    ):
        with open(scene_indices_path, "r", encoding="utf-8") as f:
            self.scene_indices = json.load(f)

        self.fs = fs

        self.scene_info_dict = {}
        for i in self.scene_indices:
            scene_token = i["raw_data"]
            scene_root = scene_token.replace("s3://", "")
            scene_info_path = "{}/info.json".format(scene_root)
            with fs.open(scene_info_path, "r", encoding="utf-8") as f:
                self.scene_info_dict[scene_root] = json.load(f)

        items = [
            {
                "scene": scene_root,
                "camera": camera_name,
                "time": i
            }
            for scene_root, scene_info in self.scene_info_dict.items()
            for camera_name, camera_info in scene_info["cameras"].items()
            for i in range(
                0, camera_info["time_range"][1] - camera_info["time_range"][0],
                interval
            )
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
        image, pts = ImageBytesFromVideoDataset.read_image_with_pts_from_video(
            self.fs, item)
        with io.BytesIO() as f:
            image.save(f, format="JPEG", quality=95)
            bytes = f.getvalue()

        return {
            "token": "{}|{}|{}".format(item["scene"], item["camera"], pts),
            "image_bytes": bytes
        }


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to auto labels the caption with VQA APIs.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The dataset config.")
    parser.add_argument(
        "-e", "--endpoint-url", type=str,
        default="http://10.119.30.71:6550/service",
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

    dataset = ImageBytesFromVideoDataset(
        config["scene_indices_path"],
        dwm.common.create_instance_from_config(config["fs"]), args.range_from,
        args.range_to)
    collate_fn = dwm.datasets.common.CollateFnIgnoring(["token", "image_bytes"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1, prefetch_factor=3,
        collate_fn=collate_fn)

    session = requests.Session()
    retries = urllib3.util.Retry(total=32, allowed_methods={"POST"})
    http_adapter = requests.adapters.HTTPAdapter(max_retries=retries)
    session.mount("http://", http_adapter)

    result = {}
    for i_id, item in enumerate(tqdm.tqdm(dataloader)):

        token = item["token"][0]
        image_bytes = item["image_bytes"][0]

        answers = {}
        for k in config["prompts"]:
            response = session.post(
                args.endpoint_url,
                data={
                    "question": k["question"],
                    "image": base64.b64encode(image_bytes).decode("utf-8"),
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
