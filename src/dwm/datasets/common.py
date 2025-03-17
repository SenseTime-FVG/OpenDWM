import bisect
import math
import numpy as np
from PIL import ImageDraw
import torch
import transforms3d
import random
from torchvision.transforms import Compose, Resize, ToTensor


class Copy():
    def __call__(self, a):
        return a


class FilterPoints():
    def __init__(self, min_distance: float = 0, max_distance: float = 1000.0):
        self.min_distance = min_distance
        self.max_distance = max_distance

    def __call__(self, a):
        distances = a[:, :3].norm(dim=-1)
        mask = torch.logical_and(
            distances >= self.min_distance, distances < self.max_distance)

        return a[mask]


class TakePoints():
    def __init__(self, max_count: int = 32768):
        self.max_count = max_count

    def __call__(self, a):
        if a.shape[0] > self.max_count:
            indices = torch.randperm(a.shape[0])[:self.max_count]
            a = a[indices]

        return a


class DatasetAdapter(torch.utils.data.Dataset):
    def apply_transform(transform, a, stack: bool = True):
        if isinstance(a, list):
            result = [
                DatasetAdapter.apply_transform(transform, i, stack) for i in a
            ]
            if stack:
                result = torch.stack(result)

            return result
        else:
            return transform(a)

    def __init__(
        self, base_dataset: torch.utils.data.Dataset, transform_list: list,
        pop_list=None
    ):
        self.base_dataset = base_dataset
        self.transform_list = transform_list
        self.pop_list = pop_list

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):

        if isinstance(index, int):
            item = self.base_dataset[index]
            for i in self.transform_list:
                if i.get("is_dynamic_transform", False):
                    item = i["transform"](item)
                else:
                    item[i["new_key"]] = DatasetAdapter.apply_transform(
                        i["transform"], item[i["old_key"]],
                        i["stack"] if "stack" in i else True)

            if self.pop_list is not None:
                for i in self.pop_list:
                    if i in item:
                        item.pop(i)

        elif isinstance(index, str):
            idx, num_frame, height, width = [
                int(val) for val in index.split("-")]
            item = self.base_dataset[idx]

            start_f = random.randint(0, len(item['images']) - num_frame)

            for k, v in item.items():
                if k != 'fps' and k != 'crossview_mask':
                    v = v[start_f:start_f+num_frame]

                item[k] = v

            for i in self.transform_list:

                if i['old_key'] in ['images', '3dbox_images', 'hdmap_images']:
                    i['transform'] = Compose([
                        Resize(size=[height, width]),
                        ToTensor()
                    ])

                if getattr(i["transform"], 'is_temporal_transform', False):
                    item[i["new_key"]] = DatasetAdapter.apply_temporal_transform(
                        i["transform"], item[i["old_key"]])
                else:
                    item[i["new_key"]] = DatasetAdapter.apply_transform(
                        i["transform"], item[i["old_key"]],
                        i["stack"] if "stack" in i else True)

            if self.pop_list is not None:
                for i in self.pop_list:
                    if i in item:
                        item.pop(i)

        return item


class ConcatMotionDataset(torch.utils.data.Dataset):
    """Concatenate multiple datasets with given ratio. It is implemented for
    the training recipe in Vista(https://arxiv.org/abs/2405.17398).

    Args:
        datasets: a list of datasets.
        ratios: a list of ratios for each dataset.
    """

    def __init__(self, datasets: list, ratios: list):
        self.datasets = datasets
        self.full_size = math.ceil(
            max([
                len(dataset) / ratio
                for dataset, ratio in zip(datasets, ratios)
            ]))
        self.ranges = torch.cumsum(
            torch.tensor([int(ratio * self.full_size) for ratio in ratios]),
            dim=0)

    def __len__(self):
        return self.full_size

    def __getitem__(self, index):
        for i, range in enumerate(self.ranges):
            if index < range:
                return self.datasets[i][index % len(self.datasets[i])]

        raise Exception(f"invalid index {index}")


class CollateFnIgnoring():
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, item_list: list):
        ignored = [
            (key, [item.pop(key) for item in item_list])
            for key in self.keys
        ]
        result = torch.utils.data.default_collate(item_list)
        for key, value in ignored:
            result[key] = value

        return result


def find_nearest(list: list, value, return_item=False):
    i = bisect.bisect_left(list, value)
    if i == 0:
        pass
    elif i >= len(list):
        i = len(list) - 1
    else:
        diff_0 = value - list[i - 1]
        diff_1 = list[i] - value
        if i > 0 and diff_0 <= diff_1:
            i -= 1

    return list[i] if return_item else i


def find_sample_data_of_nearest_time(
    sample_data_list: list, timestamp_list: list, timestamp
):
    # deprecated, use find_nearest() instead
    i = bisect.bisect_left(timestamp_list, timestamp)
    if i == 0:
        pass
    elif i >= len(timestamp_list):
        i = len(timestamp_list) - 1
    else:
        t0 = timestamp - timestamp_list[i - 1]
        t1 = timestamp_list[i] - timestamp
        if i > 0 and t0 <= t1:
            i -= 1

    return i if sample_data_list is None else sample_data_list[i]


def get_transform(rotation: list, translation: list, output_type: str = "np"):
    result = np.eye(4)
    result[:3, :3] = transforms3d.quaternions.quat2mat(rotation)
    result[:3, 3] = np.array(translation)
    if output_type == "np":
        return result
    elif output_type == "pt":
        return torch.tensor(result, dtype=torch.float32)
    else:
        raise Exception("Unknown output type of the get_transform()")


def make_intrinsic_matrix(fx_fy: list, cx_cy: list, output_type: str = "np"):
    result = np.diag(fx_fy + [1])
    result[:2, 2] = np.array(cx_cy)
    if output_type == "np":
        return result
    elif output_type == "pt":
        return torch.tensor(result, dtype=torch.float32)
    else:
        raise Exception("Unknown output type of the make_intrinsic_matrix()")


def project_line(
    a: np.array, b: np.array, near_z: float = 0.05, far_z: float = 512.0
):
    if (a[2] < near_z and b[2] < near_z) or (a[2] > far_z and b[2] > far_z):
        return None

    ca = a
    cb = b
    if a[2] >= near_z and b[2] < near_z:
        r = (near_z - b[2]) / (a[2] - b[2])
        cb = a * r + b * (1 - r)
    elif a[2] < near_z and b[2] >= near_z:
        r = (b[2] - near_z) / (b[2] - a[2])
        ca = a * r + b * (1 - r)

    if a[2] > far_z and b[2] <= far_z:
        r = (far_z - b[2]) / (a[2] - b[2])
        ca = a * r + b * (1 - r)
    elif a[2] <= far_z and b[2] > far_z:
        r = (b[2] - far_z) / (b[2] - a[2])
        cb = a * r + b * (1 - r)

    pa = ca[:2] / ca[2]
    pb = cb[:2] / cb[2]
    return (pa[0], pa[1], pb[0], pb[1])


def draw_edges_to_image(
    draw: ImageDraw.ImageDraw, points: np.array, edge_indices: list,
    pen_color: tuple, pen_width: int
):
    for a, b in edge_indices:
        xy = project_line(points[:, a], points[:, b])
        if xy is not None:
            draw.line(xy, fill=pen_color, width=pen_width)


def draw_3dbox_image(
    draw: ImageDraw.ImageDraw, view_transform: np.array,
    list_annotation_func, get_world_transform_func, get_annotation_label,
    pen_width: int, color_table: dict, corner_templates: list,
    edge_indices: list
):
    corner_templates_np = np.array(corner_templates).transpose()
    for sa in list_annotation_func():
        sa_label = get_annotation_label(sa)
        if sa_label in color_table:
            pen_color = tuple(color_table[sa_label])
            world_transform = get_world_transform_func(sa)
            p = view_transform @ world_transform @ corner_templates_np
            draw_edges_to_image(draw, p, edge_indices, pen_color, pen_width)


def align_image_description_crossview(caption_list: list, settings: dict):
    if "align_keys" in settings:
        for k in settings["align_keys"]:
            value_count = {}
            for i in caption_list:
                if i[k] not in value_count:
                    value_count[i[k]] = 0

                value_count[i[k]] += 1

            dominated_value = max(value_count, key=value_count.get)
            for i in caption_list:
                i[k] = dominated_value

    return caption_list


def make_image_description_string(
    caption_dict: dict, settings: dict, random_state: np.random.RandomState
):
    """Make the image description string from the caption dict with given
    settings.

    Args:
        caption_dict (dict): The caption dict contains textual descriptions of
            various categories such as time, environment, and more.
        settings (dict): The dict of settings to decide how to compose the
            final image descrption string.
            * selected_keys (list if exist): The value in the caption dict is
                used when its key in the list of selected keys.
            * reorder_keys (bool if exist): If set to True, the elements used
                to compose text descriptions in caption_dict will be shuffled.
            * drop_rates (dict if exist): The entries in the dict are the
                probabilities of the corresponding key elements in the
                caption_dict being dropped.
        random_state (np.random.RandomState): The random state for reproducible
            randomness.
    """
    default_image_description_keys = [
        "time", "weather", "environment", "objects", "image_description"
    ]
    selected_keys = settings.get(
        "selected_keys", default_image_description_keys)

    if "reorder_keys" in settings and settings["reorder_keys"]:
        new_order = random_state.permutation(len(selected_keys))
        selected_keys = [selected_keys[i] for i in new_order]

    if "drop_rates" in settings:
        drop = {
            k: random_state.rand() <= v
            for k, v in settings["drop_rates"].items()
        }
        selected_keys = [
            i for i in selected_keys
            if i not in drop or not drop[i]
        ]

    result = ". ".join([caption_dict[j] for j in selected_keys])
    return result


def add_stub_key_data(stub_key_data_dict, result: dict):
    """Add the stub key and data into the result dict.

    Args:
        stub_key_data_dict (dict or None): If set, the items are used to create
            stub item for the result dict. The value of this dict should be
            tuple. If the first item of the value tuple is "tensor", a tensor
            filled with the 3rd item in the shape of 2nd item is created as
            the stub data. Otherwise the 2nd item of the value tuple is
            deserialized as the stub data.
        result (dict): The result dict to insert created stub items.
    """

    if stub_key_data_dict is None:
        return

    for key, data in stub_key_data_dict.items():
        if key not in result.keys():
            if data[0] == "tensor":
                shape, value = data[1:]
                result[key] = value * torch.ones(shape)
            else:
                result[key] = data[1]
