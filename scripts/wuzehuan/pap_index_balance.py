import argparse
import json


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to balance the pap dataset.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The input index path.")
    parser.add_argument(
        "-o1", "--output-path-1", type=str, required=True,
        help="The output path to save the balanced indices.")
    parser.add_argument(
        "-o2", "--output-path-2", type=str, required=True,
        help="The output path to save the remaining indices.")
    parser.add_argument(
        "-c", "--count", default=100, type=int,
        help="The selected item count for each tag.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        indices = json.load(f)

    tag_stat = {}
    output_1_indices = []
    output_2_indices = []
    for i in indices:
        if i["scene_tag"] not in tag_stat:
            tag_stat[i["scene_tag"]] = 0

        if tag_stat[i["scene_tag"]] < args.count:
            output_1_indices.append(i)
        else:
            output_2_indices.append(i)

        tag_stat[i["scene_tag"]] += 1

    print("{} tag types".format(len(tag_stat.keys())))
    print("\n".join(["{}: {}".format(i[0], i[1]) for i in tag_stat.items()]))
    print("Input count {}, output 1 count {}, output 2 count {}.".format(
        len(indices), len(output_1_indices), len(output_2_indices)
    ))

    with open(args.output_path_1, "w", encoding="utf-8") as f:
        json.dump(output_1_indices, f, indent=2)

    with open(args.output_path_2, "w", encoding="utf-8") as f:
        json.dump(output_2_indices, f, indent=2)
