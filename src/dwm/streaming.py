import argparse
import av
import dwm.common
import einops
import json
import numpy as np
from PIL import Image
import queue
import time
import torch


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-l", "--log-path", type=str, required=True,
        help="The path to save log files.")
    parser.add_argument(
        "-s", "--streaming-path", type=str, required=True,
        help="The path to upload the video stream.")
    parser.add_argument(
        "-f", "--format", default="rtsp", type=str,
        help="The streaming format.")
    parser.add_argument(
        "--fps", default=2, type=int,
        help="The streaming FPS.")
    parser.add_argument(
        "-vcodec", "--video-encoder", default="libx264", type=str,
        help="The video encoder type.")
    parser.add_argument(
        "--pix-fmt", default="yuv420p", type=str,
        help="The pixel format.")
    return parser


def merge_multiview_images(pipeline_frame, data_condition=None):
    image_data = np.concatenate([np.asarray(i) for i in pipeline_frame], 1)
    if data_condition is not None:
        _3dbox_data = torch.nn.functional.interpolate(
            einops.rearrange(
                data_condition["3dbox_images"],
                "b t v c h w -> b c h (t v w)"),
            image_data.shape[:2]
        )[0].permute(1, 2, 0).numpy()
        hdmap_data = torch.nn.functional.interpolate(
            einops.rearrange(
                data_condition["hdmap_images"],
                "b t v c h w -> b c h (t v w)"),
            image_data.shape[:2]
        )[0].permute(1, 2, 0).numpy()
        condition_data = np.maximum(_3dbox_data, hdmap_data)
        condition_ahpla = np.max(condition_data, -1, keepdims=True) * 0.6
        image_data = (
            condition_data * 255 * condition_ahpla +
            image_data * (1 - condition_ahpla)
        ).astype(np.uint8)

    return Image.fromarray(image_data)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # setup the global state
    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    # load the pipeline including the models
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=args.log_path, config=config,
        device=torch.device(config["device"]))
    print("The pipeline is loaded.")

    data_adapter = dwm.common.create_instance_from_config(
        config["data_adapter"])

    size = pipeline.inference_config["preview_image_size"]
    latent_shape = (
        1, pipeline.inference_config["sequence_length_per_iteration"],
        len(data_adapter.sensor_channels), pipeline.vae.config.latent_channels,
        config["latent_size"][0], config["latent_size"][1]
    )
    pipeline.reset_streaming(latent_shape, "pil")

    streaming_state = {}
    data_queue = queue.Queue()
    with av.open(
        args.streaming_path, mode="w", format=args.format,
        container_options=config.get("container_options", {})
    ) as container:
        stream = container.add_stream(args.video_encoder, args.fps)
        stream.pix_fmt = args.pix_fmt
        stream.options = config.get("stream_options", {})
        while True:
            data = data_adapter.query_data()
            data_queue.put_nowait(data)
            pipeline.send_frame_condition(data)
            pipeline_frame = pipeline.receive_frame()
            if pipeline_frame is None:
                continue

            matched_data = data_queue.get_nowait()
            image = merge_multiview_images(
                pipeline_frame,
                (
                    matched_data
                    if config.get("preview_condition", False)
                    else None
                ))
            if not streaming_state.get("is_frame_size_set", False):
                stream.width = image.width
                stream.height = image.height
                streaming_state["is_frame_size_set"] = True

            while (
                "expected_time" in streaming_state and
                time.time() < streaming_state["expected_time"]
            ):
                time.sleep(0.01)

            frame = av.VideoFrame.from_image(image)
            for p in stream.encode(frame):
                container.mux(p)

            streaming_state["expected_time"] = (
                time.time()
                if "expected_time" not in streaming_state
                else streaming_state["expected_time"]
            ) + 1 / args.fps
            print("{:.1f}".format(streaming_state["expected_time"]))
