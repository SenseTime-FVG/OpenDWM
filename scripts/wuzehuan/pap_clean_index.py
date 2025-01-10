import argparse
import json


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to clean the pap index file.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of input index file.")
    parser.add_argument(
        "-ig", "--ignore-list-path", type=str, required=True,
        help="The path of the ignore list.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path of output index file.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        input_index = json.load(f)

    with open(args.ignore_list_path, "r", encoding="utf-8") as f:
        ignore = set(json.load(f))

    output_index = [
        i for i in input_index
        if i["raw_data"] not in ignore
    ]
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_index, f, indent=2)
