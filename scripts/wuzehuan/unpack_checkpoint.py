import argparse
import torch


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to unpack the checkpoint.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of the input checkpoint.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save the unpacked checkpoint file.")
    parser.add_argument("-p", "--prefix", type=str, default="ctsd.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    state = torch.load(args.input_path, map_location="cpu")
    state = {k.replace(args.prefix, ""): v for k, v in state.items()}

    torch.save(state, args.output_path)
