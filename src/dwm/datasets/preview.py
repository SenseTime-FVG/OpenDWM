import os
import json
import dwm.common
import dwm.datasets.common

from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class PreviewDataset(Dataset):
    """
        Load the package, the structure of the package is as follows:
        |- sample
            |- view1
                |- 3dbox (optional)
                |- hdmap (optional)
                |- rgb (optional)
            |- view2
           ...
            |- data.json 

        Args:
            json_file: The json contains information for each view of every frame 
                    (camera intrinsic and extrinsic parameters, image description, 
                    image condition paths, and timestamps). It is worth noting that 
                    all information, except for the image description, can be None
            sequence_length: The frame count of the temporal sequence.
            fps_stride_tuples: The list of tuples in the form of
                    (FPS, stride). If the FPS > 0, stride is the begin time in second
                    between 2 adjacent video clips, else the stride is the index count
                    of the beginning between 2 adjacent video clips.
            sensor_channels: The string list of required views, example:
                    "LIDAR_TOP", "CAM_FRONT", "CAM_BACK", "CAM_BACK_LEFT",
                    "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", following
                    the nuScenes sensor name.
            enable_camera_transforms: Whether to output the transformation matrices 
                    related to the view and ego.
            use_hdmap: Whether to use HD map conditions.
            use_3dbox: Whether to use 3dbox conditions.
            drop_vehicle_color: Whether to drop vehicle color.
            stub_key_data_dict (dict or None): The dict of stub key and data, to
                    align with other datasets with keys and data missing in this
                    dataset. Please refer to dwm.datasets.common.add_stub_key_data()
                    for details.
        """

    color = [
        "red", "green", "blue", "black", "yellow", "brown", "white",
        "purple", "grey", "beige", "maroon", "orange", "cream", "UPS",
        "silver", "tan", "copper-colored", "dark-colored", "dark"
    ]
    vehicle_name = [
        "SUV", "SUVs", "bus", "buses", "car", "cars", "truck",
        "trucks", "van", "vehicle", "sedan", "Volkswagen", "pickup",
        "taxi", "Mercedes-Benz", "minivan", "RV", "limousine", "trolley",
        "shuttle", "tram", "semi-truck", "motorbike"
    ]

    def enumerate_segments(
        self, sample_list: list, sample_indices: list, sequence_length, fps, stride
    ):
        # Handle case when there aren't enough samples
        if len(sample_indices) < sequence_length:
            return

        if fps == 0:
            # Adjusted to prevent index overflow
            for t in range(0, len(sample_indices) - sequence_length + 1, max(1, stride)):
                yield [
                    sample_indices[t + i]
                    for i in range(sequence_length)
                ]
        else:
            def enumerate_begin_time(
                sample_list: list, sample_indices: list, sequence_duration,
                stride
            ):
                s = sample_list[sample_indices[-1]]["timestamp"] - \
                    sequence_duration
                t = sample_list[sample_indices[0]]["timestamp"]
                while t <= s:
                    yield t
                    t += stride

            timestamp_list = [
                int(sample_list[i]["timestamp"]*1000) / 1000 for i in sample_indices
            ]
            for t in enumerate_begin_time(
                sample_list, sample_indices, sequence_length / fps, stride
            ):
                # find the indices of the first frame matching the given
                # timestamp
                yield [
                    sample_indices[
                        dwm.datasets.common.find_nearest(
                            timestamp_list, (t + i / fps))
                    ]
                    for i in range(sequence_length)
                ]

    @staticmethod
    def drop_vehicle_color_func(text):
        words = text.split(" ")

        new_words = []
        for i, word in enumerate(words):
            if (word in PreviewDataset.vehicle_name or
                word.rstrip('.,') in PreviewDataset.vehicle_name) \
                    and i > 0 and words[i - 1] in PreviewDataset.color:
                new_words.pop()
            else:
                new_words.append(word)
        return " ".join(new_words)

    def __init__(
        self,
        json_file: str,
        sequence_length: int,
        fps_stride_tuples: list,
        sensor_channels: list,
        enable_camera_transforms: bool,
        use_hdmap: bool = True,
        use_3dbox: bool = True,
        use_3dbox_bev: bool = False,
        use_hdmap_bev: bool = False,
        drop_vehicle_color: bool = False,
        stub_key_data_dict: dict = None
    ):

        with open(json_file, 'r') as f:
            data_package = json.load(f)
        sample_list = [data_package[str(key)] for key in sorted(
            map(int, data_package.keys()))]

        self.prefix = os.path.dirname(json_file)

        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples

        self.sensor_channels = sensor_channels
        self.enable_camera_transforms = enable_camera_transforms
        self.use_hdmap = use_hdmap
        self.use_3dbox = use_3dbox
        self.use_3dbox_bev = use_3dbox_bev
        self.use_hdmap_bev = use_hdmap_bev
        self.drop_vehicle_color = drop_vehicle_color
        self.hws = dict()
        self.lidar_hws = dict()
        self.stub_key_data_dict = {} if stub_key_data_dict is None \
            else stub_key_data_dict

        self.items = [
            {"segment": segment, "fps": fps}
            for fps, stride in self.fps_stride_tuples
            for segment in self.enumerate_segments(
                sample_list, range(
                    len(sample_list)), self.sequence_length, fps, stride
            )
        ]

        self.sample_infos = sample_list

    def __len__(self):
        return len(self.items)

    def get_camera_element(self, sample, data_type="rgb"):
        element = []

        for s in self.sensor_channels:

            camera_info = sample["camera_infos"][s]

            if data_type == "image_description":
                text = camera_info[data_type]

                if self.drop_vehicle_color:
                    text = self.drop_vehicle_color_func(text)
                element.append(text)

            elif data_type == "rgb":
                if camera_info.get(data_type, None) is not None:
                    file_path = os.path.join(
                        self.prefix, camera_info[data_type])
                    image = Image.open(file_path)
                    image.load()
                else:
                    image = Image.new("RGB", (448, 256))

                self.hws[s] = [image.size[1], image.size[0]]
                element.append(image.convert('RGB'))

            else:
                if camera_info.get(data_type, None) is not None:
                    file_path = os.path.join(
                        self.prefix, camera_info[data_type])
                    image = Image.open(file_path)
                    image.load()
                else:
                    image = Image.new("RGB", (448, 256))

                element.append(image.convert('RGB'))

        return element

    def get_lidar_element(self, sample, data_type="rgb"):
        element = []
        if not "LIDAR_TOP" in self.sensor_channels:
            return element
        lidar_info = sample["lidar_infos"]["LIDAR_TOP"]
        if data_type == "3dbox_bev" or data_type == "hdmap_bev":
            if lidar_info.get(data_type, None) is not None:
                file_path = os.path.join(self.prefix, lidar_info[data_type])
                image = Image.open(file_path)
                image.load()
            else:
                image = Image.new("RGB", (640, 640))

            self.lidar_hws["LIDAR_TOP"] = [image.size[1], image.size[0]]
            element = image.convert('RGB')
        elif data_type == "lidar":
            # b is important -> binary
            with open(os.path.join(self.prefix, lidar_info["lidar"]), mode='rb') as file:
                fileContent = file.read()
            point_data = np.frombuffer(fileContent, dtype=np.float32)
            element = torch.tensor(point_data.reshape((-1, 5))[:, :3])
        elif data_type == "lidar_transforms":
            element = torch.tensor(lidar_info["lidar_transforms"]).unsqueeze(0)

        return element

    def __getitem__(self, idx):

        item = self.items[idx]
        segment = [self.sample_infos[i] for i in item["segment"]]

        result = dict()
        result["fps"] = torch.tensor(item["fps"], dtype=torch.float32)
        result["pts"] = torch.tensor(
            [[(int(i["timestamp"]))] * len(self.sensor_channels)
             for i in segment], dtype=torch.float32)
        if any(["CAM" in i for i in self.sensor_channels]):
            result["images"] = [
                self.get_camera_element(i, "rgb") for i in segment
            ]
            result["image_description"] = [
                self.get_camera_element(i, "image_description")
                for i in segment
            ]
        if "LIDAR_TOP" in self.sensor_channels:
            result["lidar_points"] = [
                self.get_lidar_element(i, "lidar") for i in segment]
            result["lidar_transforms"] = torch.stack(
                [self.get_lidar_element(i, "lidar_transforms") for i in segment])

        if self.use_hdmap:
            result["hdmap_images"] = [
                self.get_camera_element(i, "hdmap") for i in segment]

        if self.use_3dbox:
            result["3dbox_images"] = [
                self.get_camera_element(i, "3dbox") for i in segment]

        if self.use_3dbox_bev:
            result["3dbox_bev_images"] = [
                self.get_lidar_element(i, "3dbox_bev") for i in segment]

        if self.use_hdmap_bev:
            result["hdmap_bev_images"] = [
                self.get_lidar_element(i, "hdmap_bev") for i in segment]

        if self.enable_camera_transforms:
            if "images" in result:
                result["camera_transforms"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            i["camera_infos"][s]["extrin"] if
                            i["camera_infos"][s].get("extrin") is not None
                            else torch.eye(4).tolist(),
                            dtype=torch.float32
                        )
                        for s in self.sensor_channels
                    ])
                    for i in segment
                ])

                result["camera_intrinsics"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            i["camera_infos"][s]["intrin"] if
                            i["camera_infos"][s].get("intrin") is not None
                            else torch.eye(3).tolist(),
                            dtype=torch.float32
                        )
                        for s in self.sensor_channels
                    ])
                    for i in segment
                ])

                result["ego_transforms"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            i["ego_pose"] if i["ego_pose"] is not None
                            else torch.eye(4).tolist(),
                            dtype=torch.float32
                        )
                        for s in self.sensor_channels
                    ])
                    for i in segment
                ])

                result["image_size"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            [self.hws[s][0], self.hws[s][1]], dtype=torch.long)
                        for s in self.sensor_channels
                    ])
                    for i in segment
                ])

        for key, data in self.stub_key_data_dict.items():
            if key not in result.keys():
                if data[0] == "tensor":
                    shape, value = data[1:]
                    result[key] = value * torch.ones(shape)
                else:
                    result[key] = data[1]

        return result
