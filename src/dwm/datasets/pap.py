import io
import json
import numpy as np
import torch

import aoss_client.client

import dwm.common
import dwm.datasets.common
import dwm.fs.s3fs
from PIL import Image

# PYTHONPATH includes ${workspaceFolder}/externals/senseauto
from pap_utils import *
from calib_helper import CameraIntrinsic

# TODO ALL SOLID?
LANE_TYPE = {
    "SOLID_LANE": "SOLID",
    "DASHED_LANE": "DASHED",
    "LEFT_DASHED_RIGHT_SOLID": "DASHED",
    "LEFT_SOLID_RIGHT_DASHED": "DASHED",
    "DOUBLE_SOLID": "SOLID",
    "DOUBLE_DASHED": "DASHED",
    "FISHBONE_SOLID": "SOLID",
    "FISHBONE_DASHED": "DASHED",
    "THICK_SOLID": "SOLID",
    "THICK_DASHED": "DASHED",
    "VARIABLE_LANE": "SOLID",
    "OTHER_LANE_TYPE": "SOLID",
    "CURB": "SOLID",
    "FENCE": "SOLID",
    "WALL": "SOLID",
    "DITCH": "SOLID",
    "BILATERAL": "SOLID",
    "SINGLE": "SOLID",
    "GUIDE": "SOLID",
    "VIRTUAL": "SOLID",
    "VIRTUAL_LANE": "DASHED",
    "ROAD_EDGE": "SOLID",
    "FLUSH_CURB": "SOLID",
    "TRAFFIC_BARRIER": "SOLID"
}

# "sensor_channels": [
#     "center_camera_fov30",
#     "center_camera_fov120",
#     "left_front_camera",
#     "left_rear_camera",
#     "rear_camera",
#     "right_front_camera",
#     "right_rear_camera",
#     "front_camera_fov195",
#     "left_camera_fov195",
#     "rear_camera_fov195",
#     "right_camera_fov195"
# ],


class MotionDataset(torch.utils.data.Dataset):

    @staticmethod
    def get_sample_img_id(sample: dict):
        return sample["camera_infos"]["center_camera_fov30"]["image_id"]

    @staticmethod
    def enumerate_segments(
        sample_list: list, sample_indices: list, sequence_length: int, fps, stride
    ):
        # enumerate segments
        if fps == 0:
            # frames are extracted by the index.
            for t in range(0, len(sample_indices), max(1, stride)):
                if t + sequence_length <= len(sample_indices):
                    yield [
                        sample_indices[t + i] for i in range(sequence_length)
                    ]
        else:

            def enumerate_begin_time(
                sample_list: list, sample_indices: list, sequence_duration,
                stride
            ):
                s = int(sample_list[sample_indices[-1]]["timestamp"]) / 1000000000 - \
                    sequence_duration
                t = int(sample_list[sample_indices[0]]
                        ["timestamp"]) / 1000000000
                while t <= s:
                    yield t
                    t += stride

            timestamp_list = [
                int(sample_list[i]["timestamp"]) for i in sample_indices
            ]
            for t in enumerate_begin_time(
                sample_list, sample_indices, sequence_length / fps, stride
            ):
                # find the indices of the first frame matching the given
                # timestamp
                yield [
                    sample_indices[
                        dwm.datasets.common.find_nearest(
                            timestamp_list, (t + i / fps) * 1000000000)
                    ]
                    for i in range(sequence_length)
                ]

    def get_images(
        self,
        segment,
    ):
        camera_infos = segment["camera_infos"]

        images = []
        for s in self.sensor_channels:
            camera_info = camera_infos[s]
            image_path = camera_infos[s]["filename"].replace("auto-oss:", "")

            with self.reader.open(image_path) as f:
                image = Image.open(f)
                image.load()

            self.hws[s] = [image.size[1], image.size[0]]
            # rgb_gt_np.append(
            #     self.image_processor.preprocess(
            #         self.vae_transform(
            #             img
            #         ),
            #         height=self.image_size[0],
            #         width=self.image_size[1]
            #     )[0]
            # )
            images.append(image)

        return images

    def get_hdmap_images(
        self,
        segment,
        image_condition_type="lane"
    ):
        ann_file = segment['gt_ann_info_path'].replace("auto-oss:", "")

        with self.reader.open(ann_file) as f:
            ann_info = json.load(f)
            camera_infos = ann_info['camera_infos']

        if image_condition_type == "lane":
            image_condition_infos = self.generate_maps(
                ann_info,
                thickness=3
            )
        elif image_condition_type == "road_marker":
            # TODO
            pass

        return image_condition_infos

    def generate_maps(
        self,
        ann_info,
        thickness=2
    ):
        cams = {}
        cams_list = []
        camera_infos = ann_info["camera_infos"]

        all_lane = []
        if 'anno_infos' in ann_info:
            all_lane = ann_info["anno_infos"]

        for s in self.sensor_channels:
            hw = self.hws[s]
            cams[s] = Image.new('RGB', (hw[1], hw[0]), (0, 0, 0))

        for idx, lane in enumerate(all_lane):
            lane_type = lane["style"]
            if lane['color'] in self.hdmap_image_settings.keys():
                color = self.hdmap_image_settings[lane['color']]
            else:
                color = self.hdmap_image_settings["DEFAULT_COLOR"]

            color = tuple(color)
            cur_pts = lane["geo"]
            if not len(cur_pts):
                continue

            cur_pts = pts_transfer(np.array(cur_pts))
            for s in self.sensor_channels:
                h, w = self.hws[s]
                calib = camera_infos[s]["calibration_info"]
                if s not in [
                    'front_camera_fov195',
                    'rear_camera_fov195',
                    'left_camera_fov195',
                    'right_camera_fov195'
                ]:
                    intrin = np.array(calib["intrin"])
                    extrin = np.array(calib["extrin"])
                    img_pts = project_to_camera(
                        cur_pts, intrin, np.linalg.inv(extrin))

                else:
                    cur_pts = lane["geo"]
                    cur_pts = linestrip_to_dense(cur_pts, 0.2)
                    if not len(cur_pts):
                        continue
                    cam_dist = calib["cam_dist"]
                    if len(cam_dist) != 2:
                        if "fov195" in s:
                            img_dist_type = "oriK"
                        else:
                            img_dist_type = "oriK"
                            cam_dist = np.zeros((4, 1))
                    else:
                        img_dist_type = "scaramuzza"
                    config = {
                        "0": {
                            "sensor_name": s,
                            "param": {
                                "img_dist_type": img_dist_type,
                                "cam_K": {
                                    "data": calib["intrin"],
                                },
                                "cam_dist": {
                                    "data": cam_dist
                                },
                                "cam_K_new": {
                                    "data": calib["intrin"],
                                },
                                "img_dist_h": h,
                                "img_dist_w": w,
                            }
                        }
                    }
                    camera_intrinsic_new = CameraIntrinsic(config)
                    cur_pts = np.array(cur_pts)
                    cur_pts = np.hstack(
                        (cur_pts, np.ones((cur_pts.shape[0], 1)))).T
                    camera_points = np.linalg.inv(calib["extrin"]).dot(cur_pts)
                    img_pts = camera_intrinsic_new.projectPoints(
                        camera_points[:3, :].T,
                    )
                    cd_1 = np.logical_and(
                        img_pts[:, 0] >= 0, img_pts[:, 0] < w)
                    cd_2 = np.logical_and(
                        img_pts[:, 1] >= 0, img_pts[:, 1] < h)
                    img_pts = img_pts[np.logical_and(cd_1, cd_2)]
                    # cam_model = ScaramuzzaCameraModel(calib)
                    # img_pts = cam_model.space_to_image(cur_pts)
                    if img_pts.shape[0] == 0:
                        continue

                if img_pts.shape[0] > 0:
                    if LANE_TYPE[lane_type] == "SOLID":
                        if "195" in s:
                            draw_pts(cams[s], img_pts,
                                     color=color, thickness=thickness)
                        elif s == "rear_camera":
                            draw_pts(cams[s], img_pts, color=color,
                                     thickness=thickness * 2)
                        else:
                            draw_pts(cams[s], img_pts, color=color,
                                     thickness=thickness * 3)
                    else:
                        if cams[s] is None:
                            import ipdb
                            ipdb.set_trace()
                        if "195" in s:
                            draw_dashed_pts(
                                cams[s], img_pts, color=color, thickness=thickness, dash=(10, 10))
                        elif s == "rear_camera":
                            draw_dashed_pts(
                                cams[s], img_pts, color=color, thickness=thickness * 2, dash=(20, 20))
                        else:
                            draw_dashed_pts(
                                cams[s], img_pts, color=color, thickness=thickness * 3, dash=(30, 30))

        for s in self.sensor_channels:
            cams_list.append(cams[s])

        return cams_list

    def __init__(
        self,
        fs: dwm.fs.s3fs.ForkableS3FileSystem,
        is_val: bool,
        sequence_length: int,
        fps_stride_tuples: list,
        sensor_channels: list,
        pap_caption_path: str,
        enable_camera_transforms: bool = False,
        enable_ego_transforms: bool = False,
        hdmap_image_settings: dict = None,
        image_description_settings: dict = None,
        stub_key_data_dict: dict = None
    ):
        self.reader = fs

        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples

        self.sensor_channels = sensor_channels

        self.enable_camera_transforms = enable_camera_transforms
        self.image_description_settings = image_description_settings
        self.hdmap_image_settings = hdmap_image_settings

        self.stub_key_data_dict = {} if stub_key_data_dict is None \
            else stub_key_data_dict

        self.hws = dict()

        sample_list = []
        with open(pap_caption_path, 'r') as f:
            for line in f.readlines():
                dic = json.loads(line)
                sample_list.append(dic)

        # TODO HARDCODE 8-2
        end_idx = int(len(sample_list)*0.8)
        if is_val:
            sample_list = sample_list[end_idx:]
        else:
            sample_list = sample_list[:end_idx]

        self.scene_sample_indices = dict()

        scene_idx = 0
        for sample_idx in range(0, len(sample_list)):
            if str(scene_idx) not in self.scene_sample_indices:
                self.scene_sample_indices[str(scene_idx)] = []

            self.scene_sample_indices[str(scene_idx)].append(sample_idx)

            if sample_idx == len(sample_list) - 1:
                break

            if MotionDataset.get_sample_img_id(sample_list[sample_idx+1]) - \
               MotionDataset.get_sample_img_id(sample_list[sample_idx]) != 1:
                scene_idx += 1

        self.items = dwm.common.SerializedReadonlyList([
            {"segment": segment, "fps": fps, "scene_idx": scene_idx}
            for scene_idx, sample_indices in self.scene_sample_indices.items()
            for fps, stride in self.fps_stride_tuples
            for segment in MotionDataset.enumerate_segments(
                sample_list, sample_indices, self.sequence_length, fps, stride
            )
        ])

        self.sample_infos = dwm.common.SerializedReadonlyList(sample_list)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):

        item = self.items[index]
        segment = [self.sample_infos[i] for i in item["segment"]]

        result = dict()
        result["fps"] = torch.tensor(item["fps"], dtype=torch.float32)
        result["pts"] = torch.tensor([
            [
                (int(i["timestamp"]) - int(segment[0]["timestamp"])) // 1000000
            ] * len(self.sensor_channels)
            for i in segment
        ], dtype=torch.float32)

        # rgb
        images = [self.get_images(i) for i in segment]
        if len(images) > 0:
            result["images"] = images  # [sequence_length, view_count]

        # text
        if self.image_description_settings is not None:
            result["image_description"] = [
                [i["caption"][s]["general"]
                    for s in self.sensor_channels
                 ]
                for i in segment
            ]

        if self.hdmap_image_settings is not None:
            result["hdmap_images"] = [
                self.get_hdmap_images(i) for i in segment]

        # extrin & intrin
        if self.enable_camera_transforms:
            if "images" in result:
                result["camera_transforms"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            i["camera_infos"][s]["calibration_info"]["extrin"],
                            dtype=torch.float32
                        )
                        for s in self.sensor_channels
                    ])
                    for i in segment
                ])

                result["camera_intrinsics"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            i["camera_infos"][s]["calibration_info"]["intrin"],
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


if __name__ == "__main__":

    fs = {
        'endpoint_url': 'http://auto-internal.st-sh-01.sensecoreapi-oss.cn',
        'aws_access_key_id': 'O5B34FE51MMZRZGEY1Z5',
        'aws_secret_access_key': ''
    }

    dataset = MotionDataset(4, [(2, 1)], fs,
        image_description_settings={"test": 1},
        hdmap_image_settings={
        "WHITE": [255, 255, 255],
        "YELLOW": [255, 255, 0],
        "RED": [255, 0, 0],
        "BLUE": [0, 0, 255],
        "DEFAULT_COLOR": [0, 255, 0]
    },
        sensor_channels=[
            "center_camera_fov120",
            "left_front_camera",
            "left_rear_camera",
            "rear_camera",
            "right_front_camera",
            "right_rear_camera"
    ],
        pap_caption_path="/mnt/storage/user/guoyuxin/DWM/scripts/guoyuxin/mdc/caption.json",
        enable_camera_transforms=True,
    )
