import argparse
from PIL import Image
import tqdm
import zipfile


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    image_formats = [".jpg", ".jpeg", ".png"]
    with zipfile.ZipFile(args.input_path) as zf:
        for i in tqdm.tqdm(zf.namelist()):
            if any([i.endswith(j) for j in image_formats]):
                try:
                    with zf.open(i) as f:
                        image = Image.open(f)
                        image.load()
                except:
                    print("{} broken".format(i))
