import argparse
import av
import av.video.reformatter
import dwm.common
import fractions
import json
import numpy as np
import os
from PIL import Image
import pickle
import tqdm


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to convert pap pvb dataset videos for better "
        "performance.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config path.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the merged dict file.")
    parser.add_argument("-f", "--range-from", type=int, default=0)
    parser.add_argument("-t", "--range-to", type=int, default=-1)
    return parser


def guess_rate(timestamps, frame_count = 24):
    assert frame_count > 0
    elapsed_time = timestamps[frame_count] - timestamps[0]
    result = (frame_count * 1000000000 + elapsed_time // 2) // elapsed_time
    return result


def try_calculate_target_frame_size(
    frame_size, target_frame_size, out_frame_size
):
    if len(out_frame_size) == 0:
        assert target_frame_size[0] > 0 or target_frame_size[1] > 0
        if target_frame_size[0] <= 0:
            out_frame_size.append(
                (frame_size[0] * target_frame_size[1] + frame_size[1] // 2) //
                frame_size[1] // 2 * 2)
            out_frame_size.append(target_frame_size[1] // 2 * 2)
        elif target_frame_size[1] <= 0:
            out_frame_size.append(target_frame_size[0] // 2 * 2)
            out_frame_size.append(
                (target_frame_size[0] * frame_size[1] + frame_size[0] // 2) //
                frame_size[0] // 2 * 2)
        return True
    else:
        return False


def transcode_pap_pvb(
    fs, frame_files, timestamps, c_out, s_out, config, camera_info
):
    time_range = camera_info["time_range"]
    frame_size = []
    for frame_id, frame_path in enumerate(frame_files):
        with fs.open(frame_path) as f:
            image = Image.open(f)
            image.load()

        if "origin_frame_size" not in camera_info:
            camera_info["origin_frame_size"] = [image.width, image.height]

        if try_calculate_target_frame_size(
            (image.width, image.height), config["target_frame_size"],
            frame_size
        ):
            camera_info["frame_size"] = frame_size
            s_out.width = frame_size[0]
            s_out.height = frame_size[1]

        target_frame = av.VideoFrame.from_image(
            image.resize(tuple(frame_size), Image.Resampling.BICUBIC))
        target_frame.pts = (timestamps[frame_id] + 500000) // 1000000 - \
            time_range[0]
        target_frame.time_base = s_out.codec_context.time_base
        for p in s_out.encode(target_frame):
            c_out.mux(p)

    for p in s_out.encode():
        c_out.mux(p)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # global settings
    av.logging.set_level(av.logging.ERROR)

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    fs = dwm.common.create_instance_from_config(config["fs"])
    with open(config["index_file"], "rb") as f:
        indices = pickle.load(f)

    range_to = len(indices) if args.range_to == -1 else args.range_to
    print(
        "Dataset count: {}, processing range {} - {}".format(
            len(indices), args.range_from, range_to))

    reformatter = av.video.reformatter.VideoReformatter()
    indices = indices[args.range_from:range_to]
    for scene, scene_root, _ in tqdm.tqdm(indices):
        if scene in config["ignored_scene_list"]:
            continue

        scene_info = {
            "cameras": {}
        }
        try:
            for j in config["camera_list"]:
                calib_path = "{}/calib/{}.json".format(scene_root, j)
                with fs.open(calib_path, "r", encoding="utf-8") as f:
                    calibration = json.load(f)
                    intrinsic_data = [l for k in calibration["P"] for l in k[:3]]
                    extrinsic_inv = np.array(calibration["T"] + [[0, 0, 0, 1]])
                    extrinsic = np.linalg.inv(extrinsic_inv)
                    extrinsic_data = extrinsic.flatten().tolist()

                camera_folder = j + ("#s3" if j.endswith("fov195") else "#s2")
                frame_files = fs.ls(
                    "{}/cameras/{}".format(scene_root, camera_folder),
                    detail=False)
                if len(frame_files) == 0:
                    camera_folder = j
                    frame_files = fs.ls(
                        "{}/cameras/{}".format(scene_root, camera_folder),
                        detail=False)

                frame_names = [os.path.basename(k) for k in frame_files]
                timestamps = [
                    int(os.path.splitext(k)[0].replace(".", ""))
                    for k in frame_names
                ]

                time_range = [
                    (timestamps[0] + 500000) // 1000000,
                    (timestamps[-1] + 500000) // 1000000 + 1
                ]
                fps = guess_rate(timestamps)
                camera_info = {
                    "time_range": time_range,
                    "fps": fps,
                    "intrinsic": intrinsic_data,
                    "extrinsic": extrinsic_data
                }

                output_video_path = os.path.join(
                    args.output_path, scene_root, "camera/{}.mp4".format(j))
                output_time_base = fractions.Fraction(1, 1000)
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                with av.open(output_video_path, mode="w") as c_out:
                    s_out = c_out.add_stream("libx265", fps)
                    s_out.pix_fmt = "yuv420p"
                    s_out.options = config["encoding_options"]
                    s_out.codec_context.time_base = output_time_base
                    transcode_pap_pvb(
                        fs, frame_files, timestamps, c_out, s_out, config,
                        camera_info)

                scene_info["cameras"][j] = camera_info

            output_scene_info_path = os.path.join(
                args.output_path, scene_root, "info.json")
            with open(output_scene_info_path, "w", encoding="utf-8") as f:
                json.dump(scene_info, f, **config.get("dump_options", {}))

        except Exception as e:
            print("{} - {} fails\n{}".format(scene_root, j, e))
