import argparse
import av
import av.video.reformatter
import dwm.common
import fractions
import json
import os
import tqdm


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to convert pap dataset videos for better "
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


def guess_rate(video_info, frame_count: int = 24):
    assert frame_count > 0
    elapsed_time = video_info[frame_count][1] - video_info[0][1]
    result = (frame_count * 1000000000 + elapsed_time // 2) // elapsed_time
    return result


def try_calculate_target_frame_size(
    frame_size, target_frame_size, out_frame_size
):
    if len(out_frame_size) == 0:
        assert target_frame_size[0] > 0 or target_frame_size[1] > 0
        if target_frame_size[0] <= 0:
            out_frame_size.append(
                ((frame_size[0] * target_frame_size[1] + frame_size[1] // 2) //
                 frame_size[1] + 1) // 2 * 2)
            out_frame_size.append(target_frame_size[1])
        elif target_frame_size[1] <= 0:
            out_frame_size.append(target_frame_size[0])
            out_frame_size.append(
                ((target_frame_size[0] * frame_size[1] + frame_size[0] // 2) //
                 frame_size[0] + 1) // 2 * 2)
        return True
    else:
        return False


def transcode_pap(
    c_in, c_out, s_out, reformatter, config, video_info, camera_info
):
    time_range = camera_info["time_range"]
    frame_size = []
    for frame_id, frame in enumerate(c_in.decode(c_in.streams.video[0])):
        if frame_id >= len(video_info):
            break

        frame_info = video_info[frame_id]
        if "origin_frame_size" not in camera_info:
            camera_info["origin_frame_size"] = [frame.width, frame.height]

        if try_calculate_target_frame_size(
            (frame.width, frame.height), config["target_frame_size"],
            frame_size
        ):
            camera_info["frame_size"] = frame_size
            s_out.width = frame_size[0]
            s_out.height = frame_size[1]

        target_frame = reformatter.reformat(
            frame, width=frame_size[0], height=frame_size[1],
            interpolation=av.video.reformatter.Interpolation.BICUBIC)
        target_frame.pts = (frame_info[1] + 500000) // 1000000 - time_range[0]
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
    with open(config["index_file"], "r", encoding="utf-8") as f:
        indices = json.load(f)

    range_to = len(indices) if args.range_to == -1 else args.range_to
    print(
        "Dataset count: {}, processing range {} - {}".format(
            len(indices), args.range_from, range_to))

    reformatter = av.video.reformatter.VideoReformatter()
    indices = indices[args.range_from:range_to]
    for i in tqdm.tqdm(indices):
        scene_token = i["raw_data"]
        if scene_token in config["ignored_scene_list"]:
            continue

        try:
            scene_root = i["raw_data"].replace("s3://", "")
            scene_info = {
                "cameras": {}
            }
            for j in config["camera_list"]:
                video_info_path = "{}/camera/{}.txt".format(scene_root, j)
                with fs.open(video_info_path, "r", encoding="utf-8") as f:
                    video_info = [
                        [int(number_text) for number_text in line.split(",")]
                        for line in f.read().split("\n") if line != ""
                    ]

                intrinsic_path = "{}/calib/{}/{}-intrinsic.json"\
                    .format(scene_root, j, j)
                with fs.open(intrinsic_path, "r", encoding="utf-8") as f:
                    intrinsic = json.load(f)
                    intrinsic_param = intrinsic[
                        "{}-intrinsic".format(j)
                        if "{}-intrinsic".format(j) in intrinsic else "value0"
                    ]["param"]
                    intrinsic_matrix = intrinsic_param[
                        "cam_K_new" if "cam_K_new" in intrinsic_param else "cam_K"
                    ]
                    intrinsic_data = [
                        l for k in intrinsic_matrix["data"]
                        for l in k
                    ]

                extrinsic_path = "{}/calib/{}/{}-to-car_center-extrinsic.json"\
                    .format(scene_root, j, j)
                with fs.open(extrinsic_path, "r", encoding="utf-8") as f:
                    extrinsic = json.load(f)
                    extrinsic_name = "{}-to-car_center-extrinsic".format(j)
                    extrinsic_param = extrinsic[extrinsic_name]["param"]
                    extrinsic_data = [
                        l for k in extrinsic_param["sensor_calib"]["data"]
                        for l in k
                    ]

                time_range = [
                    (video_info[0][1] + 500000) // 1000000,
                    (video_info[-1][1] + 500000) // 1000000 + 1
                ]
                fps = guess_rate(video_info)
                camera_info = {
                    "time_range": time_range,
                    "fps": fps,
                    "intrinsic": intrinsic_data,
                    "extrinsic": extrinsic_data
                }

                input_video_path = "{}/camera/{}.hevc".format(scene_root, j)
                output_video_path = os.path.join(
                    args.output_path, scene_root, "camera/{}.mp4".format(j))
                output_time_base = fractions.Fraction(1, 1000)
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                with av.open(output_video_path, mode="w") as c_out:
                    with fs.open(input_video_path) as f:
                        s_out = c_out.add_stream("libx265", fps)
                        s_out.pix_fmt = "yuv420p"
                        s_out.options = config["encoding_options"]
                        s_out.codec_context.time_base = output_time_base
                        with av.open(f) as c_in:
                            transcode_pap(
                                c_in, c_out, s_out, reformatter, config,
                                video_info, camera_info)

                scene_info["cameras"][j] = camera_info

            output_scene_info_path = os.path.join(
                args.output_path, scene_root, "info.json")
            with open(output_scene_info_path, "w", encoding="utf-8") as f:
                json.dump(scene_info, f, **config.get("dump_options", {}))

        except Exception as e:
            print("{} - {} fails\n{}".format(scene_root, j, e))
