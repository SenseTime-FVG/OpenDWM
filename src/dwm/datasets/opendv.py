import av
import dwm.common
import dwm.datasets.common
import fsspec
import json
import numpy as np
import os
from PIL import Image
import torch
import random


class MotionDataset(torch.utils.data.Dataset):
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
        enable_pts (bool): The flag to return the PTS per frame by the key
            "pts". Default is True.
        enable_fake_camera_transforms (bool): The flag to return the fake 4x4
            transform matrix to the world from a frontal camera (by the key
            "camera_transforms"), the 3x3 camera intrinsic transform matrix
            (by the key "camera_intrinsics"), and the image size (tuple of
            (width, height), by the key "image_size").
        enable_fake_3dbox_images (bool): The flag to return empty images by the
            key "3dbox_images".
        enable_fake_hdmap_images (bool): The flag to return empty images by the
            key "hdmap_images".
        fake_condition_image_color (int or tuple): The color argument for the
            PIL.Image.new() to create the fake condition images.
        image_description_settings (dict or None): If is set a dict, the image
            description is enabled and text is returned by the key
            "image_description". The "path" in the setting is for the content
            JSON file. The "candidates_times_path" in the setting is for the
            file to seek the nearest labelled time points. Please refers to
            dwm.datasets.common.make_image_description_string() for other
            settings.
        stub_key_data_dict (dict): The dict of stub key and data, to align with
            other datasets with keys and data missing in this dataset.
    """

    @staticmethod
    def get_image_description(
        image_description: dict, file_path: str, time_list_dict: dict,
        time: float
    ):
        time_list = time_list_dict[file_path]
        time_ms = int(time * 1000)
        i = dwm.datasets.common.find_nearest(time_list, time_ms)
        nearest_time = time_list[i]
        return image_description["{}.{:.0f}".format(file_path, nearest_time)]

    @staticmethod
    def get_empty_images(image_or_list, color=0):
        if isinstance(image_or_list, Image.Image):
            return Image.new(
                "RGB", image_or_list.size,
                tuple(color) if isinstance(color, list) else color)
        elif isinstance(image_or_list, list):
            return [
                MotionDataset.get_empty_images(i, color) for i in image_or_list
            ]
        else:
            raise Exception("Unexpected input type to get empty images.")

    def __init__(
        self, fs: fsspec.AbstractFileSystem, meta_path: str,
        sequence_length: int, fps_stride_tuples: list, split=None,
        mini_batch=None, shuffle_seed=42, take_video_count=None,
        ignore_list: list = [], enable_pts: bool = True,
        enable_fake_camera_transforms: bool = False,
        enable_fake_3dbox_images: bool = False,
        enable_fake_hdmap_images: bool = False, fake_condition_image_color=0,
        image_description_settings=None, stub_key_data_dict=None
    ):
        self.fs = fs
        self.sequence_length = sequence_length
        self.split = split
        self.enable_pts = enable_pts
        self.enable_fake_camera_transforms = enable_fake_camera_transforms
        self.enable_fake_3dbox_images = enable_fake_3dbox_images
        self.enable_fake_hdmap_images = enable_fake_hdmap_images
        self.fake_condition_image_color = fake_condition_image_color
        self.image_description_settings = image_description_settings
        self.stub_key_data_dict = stub_key_data_dict
        if mini_batch is not None:
            assert len(fps_stride_tuples) == 1, \
                "mini batch only support single FPS"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = {
                i["videoid"]: i
                for i in json.load(f)
                if split is None or i["split"] == split and
                i["videoid"] not in ignore_list
            }

        file_paths = self.fs.ls("", detail=False)
        if take_video_count is not None:
            file_paths = file_paths[:take_video_count]

        if self.image_description_settings is not None:
            with open(
                self.image_description_settings["path"], "r", encoding="utf-8"
            ) as f:
                self.image_descriptions = json.load(f)

            self.image_desc_rs = np.random.RandomState(
                image_description_settings["seed"]
                if "seed" in image_description_settings else None)

            if "candidates_times_path" in self.image_description_settings:
                with open(
                    self.image_description_settings["candidates_times_path"],
                    "r", encoding="utf-8"
                ) as f:
                    candidates_times = json.load(f)
            else:
                candidates_times = {}
                for file_path in file_paths:
                    candidates_times[file_path] = [
                        int(key.split(".")[-1])
                        for key in self.image_descriptions.keys()
                        if key.startswith(file_path)
                    ]

            self.candidates_times = dwm.common.SerializedReadonlyDict(
                candidates_times)

        items = []
        for file_path in file_paths:
            video_id = os.path.splitext(file_path)[0]
            if video_id not in meta_dict:
                continue

            meta = meta_dict[video_id]
            for fps, stride in fps_stride_tuples:
                t = float(meta["start_discard"])

                # incorrect here to minus the start offset, but being
                # workaround of some bad tailing videos like XxJjyO-RQY4
                s = float(
                    meta["length"] - meta["end_discard"] -
                    meta["start_discard"] - self.sequence_length / fps)
                while t <= s:
                    items.append((file_path, t, fps))
                    t += stride

        if shuffle_seed is not None:
            local_random = random.Random(shuffle_seed)
            local_random.shuffle(items)

        if mini_batch is not None:
            items = [
                items[i*mini_batch:(i+1)*mini_batch]
                for i in range(len(items) // mini_batch)
            ]

        self.items = dwm.common.SerializedReadonlyList(items)

    def __len__(self):
        return len(self.items)

    def read_item(self, file_path, t, fps):
        frames = []
        with self.fs.open(file_path) as f:
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
                "fps": torch.tensor(fps, dtype=torch.float32)
            }

            if self.enable_pts:
                result["pts"] = torch.tensor(pts, dtype=torch.float32)

            if self.image_description_settings is not None:
                image_caption = [
                    MotionDataset.get_image_description(
                        self.image_descriptions, file_path,
                        self.candidates_times, t + i / fps)
                    for i in range(self.sequence_length)
                ]
                result["image_description"] = [
                    dwm.datasets.common.make_image_description_string(
                        i, self.image_description_settings, self.image_desc_rs)
                    for i in image_caption
                ]

        except Exception as e:
            print(
                "Data item WARNING: Name {}, time {}, FPS {}, frame count {}, "
                "PTS: {}, message: {}".format(
                    file_path, t, fps, len(frames), pts_list,
                    "None" if e is None else e))
            result = {
                "images": [
                    Image.new("RGB", (1280, 720), (128, 128, 128))
                    for i in range(self.sequence_length)
                ],
                "fps": torch.tensor(fps, dtype=torch.float32)
            }

            if self.enable_pts:
                result["pts"] = torch.zeros(
                    (self.sequence_length), dtype=torch.float32)

            if self.image_description_settings is not None:
                result["image_description"] = [
                    "" for _ in range(self.sequence_length)
                ]

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
        item = self.items[index]

        if isinstance(item, list):
            list_results = [self.read_item(*i) for i in item]
            result = {
                "images": list(map(list, zip(*[i["images"] for i in list_results]))),
                "fps": list_results[0]["fps"]
            }
            if self.enable_pts:
                result["pts"] = torch.tensor(
                    list(map(list, zip(*[i["pts"] for i in list_results]))))

            if self.image_description_settings is not None:
                result["image_description"] = list(
                    map(list, zip(*[i["image_description"] for i in list_results])))

            if self.enable_fake_camera_transforms:
                camera_transform_keys = [
                    "camera_transforms", "camera_intrinsics", "image_size"
                ]
                for i in camera_transform_keys:
                    result[i] = torch.stack([j[i] for j in list_results], 1)
        else:
            result = self.read_item(*item)

        if self.enable_fake_3dbox_images:
            result["3dbox_images"] = MotionDataset.get_empty_images(
                result["images"], self.fake_condition_image_color)

        if self.enable_fake_hdmap_images:
            result["hdmap_images"] = MotionDataset.get_empty_images(
                result["images"], self.fake_condition_image_color)

        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)

        return result
