import argparse
import dwm.common
import json
import numpy as np
import pyarrow.parquet
import tqdm
import zipfile


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to repack the camera or LiDAR data of Waymo "
        "perception 2 from parquet (256 items per page) to the uncompressed "
        "ZIP blob.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The file system config.")
    parser.add_argument(
        "-s", "--split", default="training", type=str,
        choices=["testing", "testing_location", "training", "validation"],
        help="The file system config.")
    parser.add_argument(
        "-m", "--modality", type=str, required=True,
        choices=["camera_image", "lidar"],
        help="The file system config.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output ZIP file path.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    fs = dwm.common.create_instance_from_config(config)
    file_list = [
        i for i in fs.ls(
            "{}/{}".format(args.split, args.modality), detail=False)
        if i.endswith(".parquet")
    ]
    if args.modality == "camera_image":
        columns = [
            "index", "key.camera_name", "[CameraImageComponent].image"
        ]

        with zipfile.ZipFile(
            args.output_path, "w", compression=zipfile.ZIP_STORED
        ) as fo:
            for i in tqdm.tqdm(file_list):
                with fs.open(i) as f:
                    table = pyarrow.parquet.read_table(
                        f, columns=columns, use_threads=False)
                    table_dict = table.to_pydict()
                    table_columns = [table_dict[j] for j in columns]
                    for index, key, data in zip(*table_columns):
                        fo.writestr("{}.{}.jpg".format(index, key), data)

    elif args.modality == "lidar":
        lcp_info_columns = [
            "index",
            "key.laser_name",
            "[LiDARCameraProjectionComponent].range_image_return1.shape",
            "[LiDARCameraProjectionComponent].range_image_return2.shape"
        ]
        lcp_data_column = [
            "[LiDARCameraProjectionComponent].range_image_return1.values",
            "[LiDARCameraProjectionComponent].range_image_return2.values"
        ]

        lp_info_columns = [
            "index",
            "key.laser_name",
            "[LiDARPoseComponent].range_image_return1.shape"
        ]
        lp_data_column = [
            "[LiDARPoseComponent].range_image_return1.values"
        ]

        lc_info_columns = [
            "index",
            "key.laser_name",
            "[LiDARComponent].range_image_return1.shape",
            "[LiDARComponent].range_image_return2.shape"
        ]
        lc_data_columns = [
            "[LiDARComponent].range_image_return1.values",
            "[LiDARComponent].range_image_return2.values"
        ]

        with zipfile.ZipFile(
            args.output_path, "w", compression=zipfile.ZIP_STORED
        ) as fo:
            for i in tqdm.tqdm(file_list):
                lcp_path = i.replace("lidar/", "lidar_camera_projection/")
                with fs.open(lcp_path) as fi:
                    table = pyarrow.parquet.read_table(fi, use_threads=False)
                    table_dict = table.select(lcp_info_columns).to_pydict()
                    for j in range(table.num_rows):
                        index, name, shape1, shape2 = \
                            [table_dict[k][j] for k in lcp_info_columns]
                        values1, values2 = [
                            table[k][j].values.to_numpy()
                            for k in lcp_data_column]

                        values1 = values1.reshape(shape1)
                        with fo.open("lcp.{}.{}.1.npz".format(index, name), "w") as f:
                            np.savez_compressed(f, values1)

                        values2 = values2.reshape(shape2)
                        with fo.open("lcp.{}.{}.2.npz".format(index, name), "w") as f:
                            np.savez_compressed(f, values2)

                lp_path = i.replace("lidar/", "lidar_pose/")
                with fs.open(lp_path) as fi:
                    table = pyarrow.parquet.read_table(fi, use_threads=False)
                    table_dict = table.select(lp_info_columns).to_pydict()
                    for j in range(table.num_rows):
                        index, name, shape1 = \
                            [table_dict[k][j] for k in lp_info_columns]
                        values1, = [
                            table[k][j].values.to_numpy()
                            for k in lp_data_column]

                        values1 = values1.reshape(shape1)
                        with fo.open("lp.{}.{}.1.npz".format(index, name), "w") as f:
                            np.savez_compressed(f, values1)

                with fs.open(i) as fi:
                    table = pyarrow.parquet.read_table(fi, use_threads=False)
                    table_dict = table.select(lc_info_columns).to_pydict()
                    for j in range(table.num_rows):
                        index, name, shape1, shape2 = [
                            table_dict[k][j] for k in lc_info_columns]
                        values1, values2 = [
                            table[k][j].values.to_numpy()
                            for k in lc_data_columns]

                        values1 = values1.reshape(shape1)
                        with fo.open("l.{}.{}.1.npz".format(index, name), "w") as f:
                            np.savez_compressed(f, values1)

                        values2 = values2.reshape(shape2)
                        with fo.open("l.{}.{}.2.npz".format(index, name), "w") as f:
                            np.savez_compressed(f, values2)
