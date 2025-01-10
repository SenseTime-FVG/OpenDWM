import argparse
import json
import tqdm
import zipfile


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to pack caption JSON as ZIP.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of input JSON file.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path of output ZIP file.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        caption_dict = json.load(f)

    with zipfile.ZipFile(
        args.output_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for key, value in tqdm.tqdm(caption_dict.items()):
            zf.writestr(
                "caption/{}.json".format(key),
                json.dumps(value, indent=2).encode("utf-8"))
