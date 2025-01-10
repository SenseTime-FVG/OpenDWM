import argparse
import av
import bisect
import dwm.common
import fractions
import io
import json
import os
from PIL import Image


def get_sorted_table(tables: dict, index_name: str):
    table_name, column_name = index_name.split(".")
    sorted_table = sorted(tables[table_name], key=lambda i: i[column_name])
    index_column = [i[column_name] for i in sorted_table]
    return index_column, sorted_table


def load_tables(
    reader, dataset_name: str, table_names: list, index_names: list
):
    tables = dict([
        (i, json.loads(
            reader.read("{}/{}.json".format(dataset_name, i)).decode()))
        for i in table_names])
    indices = dict([
        (i, get_sorted_table(tables, i))
        for i in index_names])
    return tables, indices


def query(
    indices: dict, table_name: str, key: str, column_name: str = "token"
):
    index_column, sorted_table = \
        indices["{}.{}".format(table_name, column_name)]
    i = bisect.bisect_left(index_column, key)
    return sorted_table[i]


def query_range(
    indices: dict, table_name: str, key: str, column_name: str = "token"
):
    index_column, sorted_table = \
        indices["{}.{}".format(table_name, column_name)]
    i0 = bisect.bisect_left(index_column, key)
    i1 = bisect.bisect_right(index_column, key)
    return sorted_table[i0:i1] if i1 > i0 else None


def get_scene_samples(indices, scene):
    result = []
    i = scene["first_sample_token"]
    while i != "":
        sample = query(indices, "sample", i)
        result.append(sample)
        i = sample["next"]

    return result


def is_frontal_camera(indices, sample_data):
    calibrated_sensor = query(
        indices, "calibrated_sensor", sample_data["calibrated_sensor_token"])
    sensor = query(indices, "sensor", calibrated_sensor["sensor_token"])
    return sensor["modality"] == "camera" and sensor["channel"] == "CAM_FRONT"


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config of the dataset reader.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save the video clips.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    table_names = [
        "calibrated_sensor", "sample", "sample_data", "scene", "sensor"
    ]
    index_names = [
        "calibrated_sensor.token", "sample.token",
        "sample_data.sample_token", "sample_data.token",
        "scene.token", "sensor.token"
    ]

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    reader = dwm.common.create_instance_from_config(config["reader"])
    reader2 = dwm.common.create_instance_from_config(config["reader2"])
    tables, indices = load_tables(
        reader, config["dataset_name"], table_names, index_names)
    scene_frontal_frames = [
        (i, sorted([
            k
            for j in get_scene_samples(indices, i)
            for k in query_range(
                indices, "sample_data", j["token"], column_name="sample_token")
            if is_frontal_camera(indices, k)
        ], key=lambda x: x["timestamp"]))
        for i in tables["scene"]
    ]

    os.makedirs(args.output_path, exist_ok=True)
    for scene, sample_data_list in scene_frontal_frames:
        path = os.path.join(args.output_path, "{}.mp4".format(scene["name"]))
        is_frame_size_set = False
        with av.open(path, mode="w") as container:
            stream = container.add_stream("mpeg4", 12)
            stream.time_base = fractions.Fraction(1, 20)
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "16", "b": "10M"}
            stream.codec_context.bit_rate_tolerance = 5000000
            for i in sample_data_list:
                image = Image.open(io.BytesIO(reader.read(i["filename"])))
                condition_image = Image.open(
                    io.BytesIO(reader2.read(i["filename"]))) \
                    if i["filename"] in reader2.items \
                    else Image.new("RGB", image.size)

                if not is_frame_size_set:
                    stream.width = condition_image.width
                    stream.height = condition_image.height
                    is_frame_size_set = True

                frame = av.VideoFrame.from_image(condition_image)
                frame.pts = (
                    i["timestamp"] - sample_data_list[0]["timestamp"] + 25000) // 50000
                for p in stream.encode(frame):
                    container.mux(p)

            for p in stream.encode():
                container.mux(p)
