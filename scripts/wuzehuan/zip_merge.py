import argparse
import zipfile


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to merge ZIP files.")
    parser.add_argument(
        "-i", "--input-prefix", type=str, required=True,
        help="The prefix of the input path before part numbers.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the merged ZIP file.")
    parser.add_argument("-f", "--range-from", type=int, default=0)
    parser.add_argument("-t", "--range-to", type=int)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with zipfile.ZipFile(
        args.output_path, "w", compression=zipfile.ZIP_STORED
    ) as zfo:
        for i in range(args.range_from, args.range_to):
            with zipfile.ZipFile(
                "{}_{}.zip".format(args.input_prefix, i)
            ) as zfi:
                for j in zfi.namelist():
                    with zfo.open(j, "w") as f:
                        f.write(zfi.read(j))

        print("Merged {} items.".format(len(zfo.namelist())))
