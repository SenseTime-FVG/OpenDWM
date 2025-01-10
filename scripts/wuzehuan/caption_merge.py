import argparse
import json


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to auto labels the caption with VQA APIs.")
    parser.add_argument(
        "-i", "--input-prefix", type=str, required=True,
        help="The prefix of the input path before part numbers.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the merged dict file.")
    parser.add_argument("-f", "--range-from", type=int, default=0)
    parser.add_argument("-t", "--range-to", type=int)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    result = {}
    for i in range(args.range_from, args.range_to):
        with open(
            "{}_{}.json".format(args.input_prefix, i), "r", encoding="utf-8"
        ) as f:
            result.update(json.load(f))

    print("Merged {} items.".format(len(result)))
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
