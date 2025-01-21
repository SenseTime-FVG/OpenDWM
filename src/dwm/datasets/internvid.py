import av
import dwm.common
import dwm.datasets.common
import fsspec
import json
import os
from PIL import Image
import torch
import random


from datasets import load_dataset


class InternVidFLT(torch.utils.data.Dataset):
    """The motion dataset of OpenDV-Youtube
    (https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv).

    Args:
        fs (fsspec.AbstractFileSystem): The file system for the dataset
            content files.
        meta_path (str): The meta file acquired following the OpenDV dataset
            guide (https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv#meta-data-preparation)
        sequence_length (int): The frame count of each video clips extracted
            from the dataset, also the "T" of the video tensor shape
            [T, C, H, W]. When mini_batch is set K, the video tensor is
            returned in the shape of [T, K, C, H, W].
        fps_stride_tuples (list): The list of tuples in the form of
            (FPS, stride). The stride is the begin time in second between 2
            adjacent video clips.
        split (str): The dataset split different purpose of training and
            validation. Should be one of "Train", "Val".
        mini_batch (int or None): If enable the sub-dimension between the
            sequence and item content. Useful to align with the multi-view
            datasets.
        shuffle_seed (int or None): If is set a number, the dataset order is
            shuffled by this seed. If is set None, the dataset keeps the origin
            order.
        take_video_count (int or None): If is set a number, the dataset only
            use partial videos in the listed files [:take_video_count].
        ignore_list (list): The videos with IDs in the list are not read.
        enable_fake_camera_transforms (bool): The flag to return the fake 4x4
            transform matrix to the world from a frontal camera (by the key
            "camera_transforms"), the 3x3 camera intrinsic transform matrix
            (by the key "camera_intrinsics"), and the image size (tuple of
            (width, height), by the key "image_size").
        enable_fake_3dbox_images (bool): The flag to return empty images by the
            key "3dbox_images".
        enable_fake_hdmap_images (bool): The flag to return empty images by the
            key "hdmap_images".
        stub_key_data_dict (dict): The dict of stub key and data, to align with
            other datasets with keys and data missing in this dataset.
    """

    @staticmethod
    def time_to_second(time_str):
        # Split the time string into hours, minutes, seconds and milliseconds
        parts = time_str.split(':')
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])

        total_milliseconds = hours * 3600 + minutes * 60 + seconds
        return total_milliseconds

    @staticmethod
    def get_empty_images(image_or_list):
        if isinstance(image_or_list, Image.Image):
            return Image.new("RGB", image_or_list.size)
        elif isinstance(image_or_list, list):
            return [InternVidFLT.get_empty_images(i) for i in image_or_list]
        else:
            raise Exception("Unexpected input type to get empty images.")

    def __init__(
        self, fs: fsspec.AbstractFileSystem, meta_path: str,
        sequence_length: int, fps_stride_tuples: list, split="train",
        mini_batch=None, shuffle_seed=42, enable_fake_camera_transforms: bool = False,
        enable_fake_3dbox_images: bool = False,
        enable_fake_hdmap_images: bool = False,
        stub_key_data_dict=None, aes_th=4.5, vila_mode=True,
        max_samples=None,
    ):
        self.fs = fs
        self.sequence_length = sequence_length
        self.split = split
        self.enable_fake_camera_transforms = enable_fake_camera_transforms
        self.enable_fake_3dbox_images = enable_fake_3dbox_images
        self.enable_fake_hdmap_images = enable_fake_hdmap_images
        self.stub_key_data_dict = stub_key_data_dict
        if mini_batch is not None:
            assert len(fps_stride_tuples) == 1, \
                "mini batch only support single FPS"

        meta_dict = load_dataset("json", data_files=meta_path)
        self.meta_list = meta_dict[split]
        self.vila_mode = vila_mode

        items = []
        for idx, meta in enumerate(self.meta_list):
            if meta["Aesthetic_Score"] <= aes_th:
                continue
            delta = InternVidFLT.time_to_second(meta["End_timestamp"]) - \
                InternVidFLT.time_to_second(meta["Start_timestamp"])
            if self.vila_mode:
                for fps, stride in fps_stride_tuples:
                    if delta > self.sequence_length / fps:
                        items.append((idx, 0, fps))
                continue

            for fps, stride in fps_stride_tuples:
                t = 0

                # incorrect here to minus the start offset, but being
                # workaround of some bad tailing videos like XxJjyO-RQY4
                s = float(delta - self.sequence_length / fps)
                while t <= s:
                    items.append((idx, t, fps))
                    t += stride

        if shuffle_seed is not None:
            local_random = random.Random(shuffle_seed)
            local_random.shuffle(items)

        if mini_batch is not None:
            items = [
                items[i*mini_batch:(i+1)*mini_batch]
                for i in range(len(items) // mini_batch)
            ]

        if max_samples is not None:
            items = items[:max_samples]
        self.items = dwm.common.SerializedReadonlyList(items)

    def __len__(self):
        return len(self.items)

    def read_item(self, index=None, index2=None):
        frames = []
        count, max_tries = 0, 10
        mid, t, fps = self.items[index] if index2 is None else self.items[index][index2]
        while count < max_tries:
            try:
                meta = self.meta_list[mid]
                file_name = "_".join([meta["YoutubeID"], meta["Start_timestamp"], meta["End_timestamp"]]) + ".mp4"
                with self.fs.open(file_name) as f:
                    with av.open(f) as container:
                        stream = container.streams.video[0]
                        time_base = stream.time_base
                        first_pts = int((t - 0.5 / fps) / time_base)
                        last_pts = int(
                            (t + (self.sequence_length + 0.5) / fps) / time_base)
                        container.seek(first_pts, stream=stream)
                        for i in container.decode(stream):
                            if i.pts < first_pts:
                                continue

                            elif i.pts > last_pts:
                                break

                            frames.append(i)

                        stream = None
                        break
            except:
                count += 1
                if index2 is not None:
                    mid, t, fps = self.items[(index+count)%len(self.items)][index2]
                else:
                    mid, t, fps = self.items[(index+count)%len(self.items)]

        pts_list = [i.pts for i in frames]
        try:
            expected_ptss = [
                int((t + i / fps) / time_base)
                for i in range(self.sequence_length)
            ]
            candidates = [
                frames[dwm.datasets.common.find_nearest(pts_list, i)]
                for i in expected_ptss
            ]

            pts = [
                int(1000 * (i.pts - candidates[0].pts) * time_base + 0.5)
                for i in candidates
            ]
            images = [i.to_image() for i in candidates]

            result = {
                # this PIL Image item should be converted to tensor before data
                # loader collation
                "images": images,
                "pts": torch.tensor(pts, dtype=torch.float32),
                "fps": torch.tensor(fps, dtype=torch.float32)
            }
            result["video_description"] = [meta["Caption"]]
            delta = InternVidFLT.time_to_second(meta["End_timestamp"]) - \
                InternVidFLT.time_to_second(meta["Start_timestamp"])
            result["text_duration"] = int(delta*10)
        except Exception as e:
            print(
                "Data item WARNING: Name {}, time {}, FPS {}, frame count {}, "
                "PTS: {}, message: {}".format(
                    file_name, t, fps, len(frames), pts_list,
                    "None" if e is None else e))
            result = {
                "images": [
                    Image.new("RGB", (1280, 720), (128, 128, 128))
                    for i in range(self.sequence_length)
                ],
                "pts": torch.zeros((self.sequence_length), dtype=torch.float32),
                "fps": torch.tensor(fps, dtype=torch.float32),
            }

            result["video_description"] = [""]
            result["text_duration"] = 0

        if self.enable_fake_camera_transforms:
            result["camera_transforms"] = torch.tensor([[
                [0, 0, 1, 1.7],
                [-1, 0, 0, 0],
                [0, -1, 0, 1.5],
                [0, 0, 0, 1]
            ]], dtype=torch.float32).repeat(self.sequence_length, 1, 1)
            result["camera_intrinsics"] = torch.stack([
                torch.tensor([
                    [0.5 * (i.width + i.height), 0, 0.5 * i.width],
                    [0, 0.5 * (i.width + i.height), 0.5 * i.height],
                    [0, 0, 1]
                ], dtype=torch.float32)
                for i in result["images"]
            ])
            result["image_size"] = torch.stack([
                torch.tensor([i.width, i.height], dtype=torch.long)
                for i in result["images"]
            ])

        return result

    def __getitem__(self, index: int):
        item  = self.items[index]

        if isinstance(item, list):
            list_results = [self.read_item(index=index, index2=cid) for cid in range(len(item))]
            result = {
                "images": list(map(list, zip(*[i["images"] for i in list_results]))),
                "pts": torch.tensor(list(map(list, zip(*[i["pts"] for i in list_results])))),
                "fps": list_results[0]["fps"]
            }
            result["video_description"] = list(
                map(list, zip(*[i["video_description"] for i in list_results])))

            if self.enable_fake_camera_transforms:
                camera_transform_keys = [
                    "camera_transforms", "camera_intrinsics", "image_size"
                ]
                for i in camera_transform_keys:
                    result[i] = torch.stack([j[i] for j in list_results], 1)
        else:
            result = self.read_item(index=index)

        if self.enable_fake_3dbox_images:
            result["3dbox_images"] = InternVidFLT.get_empty_images(
                result["images"])

        if self.enable_fake_hdmap_images:
            result["hdmap_images"] = InternVidFLT.get_empty_images(
                result["images"])

        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)
        assert "video_description" in result and "image_description" not in result, result.keys()

        return result