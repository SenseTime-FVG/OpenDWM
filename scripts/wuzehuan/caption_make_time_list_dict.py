import argparse
import json


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to make time list for caption dict for fast "
        "seek.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of input caption file.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the time list file.")
    parser.add_argument(
        "-t", "--type", default="pap", type=str, choices=["pap", "opendv"],
        help="The output path to save the time list file.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        caption_dict = json.load(f)

    caption_time_list_dict = {}
    for i in caption_dict.keys():
        if args.type == "pap":
            scene, view, time = i.split("|")
            key = "{}|{}".format(scene, view)
        elif args.type == "opendv":
            file, ext, time = i.split(".")
            key = "{}.{}".format(file, ext)

        if key not in caption_time_list_dict:
            caption_time_list_dict[key] = []

        caption_time_list_dict[key].append(int(time))

    caption_time_list_dict = {
        k: sorted(v)
        for k, v in caption_time_list_dict.items()
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(caption_time_list_dict, f)
