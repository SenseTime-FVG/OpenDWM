import argparse
import dwm.common
import dwm.datasets.pap_100k
import dwm.fs.czip
import json


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to debug dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset = dwm.common.create_instance_from_config(config["dataset"])
    print(len(dataset))
    a = dataset[0]
    print(a.keys())
