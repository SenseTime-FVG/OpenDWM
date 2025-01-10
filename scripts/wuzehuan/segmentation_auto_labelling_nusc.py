import argparse
import contextlib
import dwm.common
import dwm.datasets.nuscenes
import einops
import json
from PIL import Image
import torch
import torchvision
import tqdm
import transformers
import zipfile


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

    dataset = dwm.common.create_instance_from_config(config["dataset"])
    range_to = len(dataset) if args.range_to == -1 else args.range_to
    print(
        "Dataset count: {}, processing range {} - {}".format(
            len(dataset), args.range_from, range_to))

    result = {}
    with zipfile.ZipFile(
        args.output_path, "w", compression=zipfile.ZIP_STORED
    ) as zf:
        for i in tqdm.tqdm(range(args.range_from, range_to)):
            item = dataset.items[i]
            flatten_segments = [k for j in item["segment"] for k in j]
            for j in flatten_segments:
                sample_data = dwm.datasets.nuscenes.MotionDataset.query(
                    dataset.tables, dataset.indices, "sample_data", j)
                with dataset.fs.open(sample_data["filename"]) as f:
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
                with zf.open(
                    "{}.png".format(sample_data["filename"]), "w"
                ) as f:
                    mask_image.save(f, format="png")
