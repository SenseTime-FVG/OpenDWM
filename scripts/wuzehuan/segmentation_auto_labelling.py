import argparse
import contextlib
import dwm.common
import dwm.datasets.nuscenes
import dwm.fs.czip
import einops
import io
import json
from PIL import Image
import torch
import torchvision
import tqdm
import transformers
import zipfile


class ImageRawDataset(torch.utils.data.Dataset):

    def __init__(
        self, fs: dwm.fs.czip.CombinedZipFileSystem, range_from: int,
        range_to: int, format_list: list = [".jpeg", ".jpg", ".png"]
    ):
        self.fs = fs
        items = [
            i for i in fs._belongs_to.keys()
            if any([i.endswith(j) for j in format_list])
        ]

        range_to = len(items) if range_to == -1 else range_to
        print(
            "Dataset count: {}, processing range {} - {}".format(
                len(items), range_from, range_to))

        self.items = items[range_from:range_to]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        data = self.fs.cat_file(self.items[index])
        return {
            "token": self.items[index],
            "image_data": data
        }


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to auto labels the semantic segmentation with "
        "SegFormer.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The dataset config.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the dict of image ID and labelled "
        "caption.")
    parser.add_argument("-f", "--range-from", type=int, default=0)
    parser.add_argument("-t", "--range-to", type=int, default=-1)
    return parser


def get_autocast_context(config):
    if "autocast" in config:
        return torch.autocast(**config["autocast"])
    else:
        return contextlib.nullcontext()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    feature_extractor = transformers.SegformerFeatureExtractor.from_pretrained(
        config["pretrained_model_name_or_path"])
    model = transformers.SegformerForSemanticSegmentation.from_pretrained(
        config["pretrained_model_name_or_path"])
    if "device" in config:
        model.to(torch.device(config["device"]))

    fs = dwm.common.create_instance_from_config(config["fs"])
    dataset = ImageRawDataset(
        fs, args.range_from, args.range_to,
        config.get("format_list", [".jpeg", ".jpg", ".png"]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, # num_workers=1, prefetch_factor=3,
        num_workers=0,
        collate_fn=dwm.datasets.common.CollateFnIgnoring(
            ["token", "image_data"]))

    result = {}
    with zipfile.ZipFile(
        args.output_path, "w", compression=zipfile.ZIP_STORED
    ) as zf:
        for item in tqdm.tqdm(dataloader):

            token = item["token"][0]
            image_data = item["image_data"][0]
            with io.BytesIO(image_data) as f:
                image = Image.open(f)
                image.load()

            inputs = feature_extractor(images=image, return_tensors="pt")
            if "device" in config:
                inputs.data["pixel_values"] = \
                    inputs.data["pixel_values"].to(
                        torch.device(config["device"]))

            with get_autocast_context(config):
                outputs = model(**inputs)
                masks = torch.nn.functional.pad(
                    torch.nn.functional.sigmoid(outputs.logits),
                    (0, 0, 0, 0, 0, 5))  # make 24 channels

            mask_image_tensor = einops.rearrange(
                masks, "b (gh gw c) h w -> c (b gh h) (gw w)", gw=4, c=3)
            mask_image = torchvision.transforms.functional.to_pil_image(
                mask_image_tensor)
            with zf.open("{}.png".format(token), "w") as f:
                mask_image.save(f, format="png")
