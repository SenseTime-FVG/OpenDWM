import argparse
import json
import math
import numpy as np
import transforms3d


def create_parser():
    parser = argparse.ArgumentParser(
        description="Make Carla camera parameters from intrinsic matrices and "
        "transform matrices.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of input camera parameter file.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path of output Carla parameter file.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    rear_ego_to_center_ego = [-1.5, 0, 0]
    lh_from_rh = rh_from_lh = np.diag([1, -1, 1, 1])

    # z fontal camera (OpenCV style) is x-right, y-down, z-front.
    # x fontal camera (Y flipped Carla style) is x-front, y-left, z-up.
    z_frontal_camera_from_x_frontal_camera = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ], np.float32)

    with open(args.input_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    result = {}
    for k, v in config.items():
        carla_transform = lh_from_rh @ np.array(v["transform"]) @ \
            z_frontal_camera_from_x_frontal_camera @ rh_from_lh
        euler_rotation = transforms3d.euler.mat2euler(
            carla_transform[:3, :3], "syzx")
        result[k] = {
            "attributes": {
                "fov": str(
                    math.degrees(
                        math.atan(v["intrinsic"][0][2] / v["intrinsic"][0][0])
                        + math.atan(
                            (v["image_size"][0] - v["intrinsic"][0][2]) /
                            v["intrinsic"][0][0]))
                ),
                "role_name": k
            },
            "spawn_transform": {
                "location": [
                    (carla_transform[i][3] + rear_ego_to_center_ego[i]).item()
                    for i in range(3)
                ],
                "rotation": [
                    math.degrees(-euler_rotation[0]),
                    math.degrees(euler_rotation[1]),
                    math.degrees(-euler_rotation[2])
                ]
            }
        }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
