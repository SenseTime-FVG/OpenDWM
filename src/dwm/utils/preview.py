import av
import torch
import torchvision


def make_ctsd_preview_tensor(output_images, batch, inference_config):

    # The output image sequece length may be shorter than the input due to the
    # autoregressive inference, so use the output sequence length to clip batch
    # data.
    batch_size, _, view_count = batch["vae_images"].shape[:3]
    output_images = output_images\
        .cpu().unflatten(0, (batch_size, -1, view_count))
    sequence_length = output_images.shape[1]

    collected_images = [batch["vae_images"][:, :sequence_length]]
    if "3dbox_images" in batch:
        collected_images.append(
            batch["3dbox_images"][:, :sequence_length])

    if "hdmap_images" in batch:
        collected_images.append(
            batch["hdmap_images"][:, :sequence_length])

    collected_images.append(output_images)

    stacked_images = torch.stack(collected_images)
    resized_images = torch.nn.functional.interpolate(
        stacked_images.flatten(0, 3),
        tuple(inference_config["preview_image_size"][::-1])
    )
    resized_images = resized_images.view(
        *stacked_images.shape[:4], -1, *resized_images.shape[-2:])
    if sequence_length == 1:
        # image preview with shape [C, B * T * S * H, V * W]
        preview_tensor = resized_images.permute(4, 1, 2, 0, 5, 3, 6)\
            .flatten(-2).flatten(1, 4)
    else:
        # video preview with shape [T, C, B * S * H, V * W]
        preview_tensor = resized_images.permute(2, 4, 1, 0, 5, 3, 6)\
            .flatten(-2).flatten(2, 4)

    return preview_tensor


def make_lidar_preview_tensor(
    ground_truth_volumn, generated_volumn, batch, inference_config
):
    collected_images = [
        ground_truth_volumn.amax(-3, keepdim=True).repeat_interleave(3, -3)
        .cpu()
    ]
    if "3dbox_bev_images_denorm" in batch:
        collected_images.append(batch["3dbox_bev_images_denorm"])

    if "hdmap_bev_images_denorm" in batch:
        collected_images.append(batch["hdmap_bev_images_denorm"])

    if isinstance(generated_volumn, list):
        for gv in generated_volumn:
            collected_images.append(
                gv.amax(-3, keepdim=True).repeat_interleave(3, -3).cpu())
    else:
        collected_images.append(
            generated_volumn.amax(-3, keepdim=True).repeat_interleave(3, -3).cpu())

    # assume all BEV images have the same size
    stacked_images = torch.stack(collected_images)
    if ground_truth_volumn.shape[1] == 1:
        # BEV image preview with shape [C, B * T * H, S * W]
        preview_tensor = stacked_images.permute(3, 1, 2, 4, 0, 5).flatten(-2)\
            .flatten(1, 3)
    else:
        # BEV video preview with shape [T, C, B * H, S * W]
        preview_tensor = stacked_images.permute(2, 3, 1, 4, 0, 5).flatten(-2)\
            .flatten(2, 3)

    return preview_tensor


def save_tensor_to_video(
    path: str, video_encoder: str, fps, tensor_list, pix_fmt: str = "yuv420p",
    stream_options: dict = {"crf": "16"}
):
    tensor_shape = tensor_list[0].shape
    with av.open(path, mode="w") as container:
        stream = container.add_stream(video_encoder, int(fps))
        stream.width = tensor_shape[-1]
        stream.height = tensor_shape[-2]
        stream.pix_fmt = pix_fmt
        stream.options = stream_options
        for i in tensor_list:
            frame = av.VideoFrame.from_image(
                torchvision.transforms.functional.to_pil_image(i))
            for p in stream.encode(frame):
                container.mux(p)

        for p in stream.encode():
            container.mux(p)
