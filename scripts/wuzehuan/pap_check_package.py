import argparse
import dwm.common
import json
import tqdm


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to clean the pap index file.")
    parser.add_argument(
        "-i", "--index-path", type=str, required=True,
        help="The path of index file.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    fs = dwm.common.create_instance_from_config({
        "_class_name": "dwm.fs.czip.CombinedZipFileSystem",
        "fs": {
            "_class_name": "dwm.fs.s3fs.ForkableS3FileSystem",
            "endpoint_url": "http://aoss-internal-v2.st-sh-01.sensecoreapi-oss.cn",
            "aws_access_key_id": "4853705CDE93446E8D902F70291C2C92",
            "aws_secret_access_key": ""
        },
        "paths": [
            "users/wuzehuan/data/pap/pap_100k_b1_0.zip",
            "users/wuzehuan/data/pap/pap_100k_b1_1.zip",
            "users/wuzehuan/data/pap/pap_100k_b1_2.zip",
            "users/wuzehuan/data/pap/pap_100k_b1_3.zip"
        ]
    })

    with open(args.index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    checklist = [
        "camera/left_front_camera.mp4",
        "camera/center_camera_fov30.mp4",
        "camera/right_front_camera.mp4",
        "camera/right_rear_camera.mp4",
        "camera/rear_camera.mp4",
        "camera/left_rear_camera.mp4",
        "camera/center_camera_fov120.mp4",
        "camera/left_camera_fov195.mp4",
        "camera/front_camera_fov195.mp4",
        "camera/right_camera_fov195.mp4",
        "camera/rear_camera_fov195.mp4",
        "info.json"
    ]
    for i in tqdm.tqdm(index):
        prefix = i["raw_data"].replace("s3://", "")
        for j in checklist:
            path = "{}/{}".format(prefix, j)
            assert fs.exists(path), path
