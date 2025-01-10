import argparse
import dwm.datasets.nuscenes
import dwm.fs.czip
import fsspec.implementations.local
import json


def create_parser():
    parser = argparse.ArgumentParser(
        description="Convert the nuscenes caption to version 2, format of "
        "(scene|sensor|time).")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of the input caption file.")
    parser.add_argument(
        "-m", "--meta-path", type=str, required=True,
        help="The path of the dataset meta.")
    parser.add_argument(
        "-n", "--dataset-name", type=str, required=True,
        choices=["v1.0-mini", "v1.0-trainval"],
        help="The dataset name.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save the converted caption file.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    table_names = [
        "calibrated_sensor", "sample", "sample_data", "scene", "sensor"
    ]
    index_names = [
        "calibrated_sensor.token", "sample.token", "sample_data.sample_token",
        "sample_data.token", "scene.token", "sensor.token"
    ]
    fs = dwm.fs.czip.CombinedZipFileSystem(
        fsspec.implementations.local.LocalFileSystem(), [args.meta_path])
    tables, indices = dwm.datasets.nuscenes.MotionDataset.load_tables(
        fs, args.dataset_name, table_names, index_names)

    with open(args.input_path, "r", encoding="utf-8") as f:
        input_caption = json.load(f)

    output_caption = {}
    for k, v in input_caption.items():
        sample_data = dwm.datasets.nuscenes.MotionDataset.query(
            tables, indices, "sample_data", k)

        sample = dwm.datasets.nuscenes.MotionDataset.query(
            tables, indices, "sample", sample_data["sample_token"])
        scene = dwm.datasets.nuscenes.MotionDataset.query(
            tables, indices, "scene", sample["scene_token"])

        calibrated_sensor = dwm.datasets.nuscenes.MotionDataset.query(
            tables, indices, "calibrated_sensor",
            sample_data["calibrated_sensor_token"])
        sensor = dwm.datasets.nuscenes.MotionDataset.query(
            tables, indices, "sensor", calibrated_sensor["sensor_token"])

        new_key = "{}|{}|{}".format(
            scene["token"], sensor["channel"], sample_data["timestamp"])
        output_caption[new_key] = v

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_caption, f, indent=2)
