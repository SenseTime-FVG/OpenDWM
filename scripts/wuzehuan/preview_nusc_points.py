import argparse
import numpy as np
from PIL import Image
import transforms3d


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to preview nuScenes point file.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The input path of videos.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the merged dict file.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.input_path, "rb") as f:
        point_data = np.frombuffer(f.read(), dtype=np.float32)

    point_data = point_data.reshape((-1, 5))[:, :3]
    bev_from_ego = np.array([
        [6.4, 0, 0, 320],
        [0, -6.4, 0, 320],
        [0, 0, -6.4, 0],
        [0, 0, 0, 1]
    ])
    tilt = np.eye(4)
    tilt[:3, :3] = transforms3d.euler.euler2mat(-np.pi / 2, 0, 0)

    point_data = bev_from_ego @ np\
        .concatenate([point_data, np.ones((point_data.shape[0], 1))], axis=-1)\
        .transpose()
    point_data = point_data[:2].round().astype(np.int32)
    point_mask = np.all([
        point_data[0] >= 0, point_data[0] < 640,
        point_data[1] >= 0, point_data[1] < 640
    ], axis=0)
    point_data = point_data[:, point_mask]
    pixel_indices = point_data[0] + point_data[1] * 640

    bev_image_plain_data = np.zeros((640 * 640,), dtype=np.uint8)
    bev_image_plain_data[pixel_indices] = 255
    bev_image_data = bev_image_plain_data.reshape((640, 640))
    Image.fromarray(bev_image_data).save(args.output_path)
