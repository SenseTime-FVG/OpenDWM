import av
import bisect
import json
from PIL import Image
import torch

# The walkaround for video rotation meta (waiting for side data support on pyav
# streams)
# Note: Linux should apt install the mediainfo package for the shared library
# files.
import pymediainfo


class InfoExtension:
    def __init__(self, reader):
        self.reader = reader

    def query_and_interpolate(
        self, video_path: str, timestamps: list,
        table_names: list = ["locations"]
    ):
        path = video_path.replace("videos", "info/100k")\
            .replace(".mov", ".json")
        info = json.loads(self.reader.read(path).decode())

        indices = {
            i: list([j["timestamp"] for j in info[i]]) for i in table_names
        }
        result = []
        for i in timestamps:
            r = {}
            t = info["startTime"] + i
            for j in table_names:
                table = info[j]
                index = bisect.bisect_left(indices[j], t)
                if index == len(indices[j]):
                    r[j] = None
                elif index == 0:
                    r[j] = table[0] if t == table[0]["timestamp"] else None
                else:
                    item = table[index]
                    item_1 = table[index - 1]
                    alpha = (t - item_1["timestamp"]) / \
                        (item["timestamp"] - item_1["timestamp"])
                    r[j] = {
                        key: alpha * v + (1 - alpha) * item_1[key]
                        for key, v in item.items() if key != "timestamp"
                    }

            result.append(r)

        return result


class MotionDataset(torch.utils.data.Dataset):
    def find_frame_of_nearest_time(frames: list, pts_list: list, pts: int):
        i = bisect.bisect_left(pts_list, pts)
        if i >= len(frames):
            return frames[-1]

        if i > 0:
            t0 = pts - pts_list[i - 1]
            t1 = pts_list[i] - pts
            if t0 <= t1:
                i -= 1

        return frames[i]

    def __init__(
        self, reader, sequence_length: int, fps_stride_tuples: list,
        info_extension: InfoExtension = None,
        ignore_list: list = ["bdd100k/videos/val/c4742900-81aa45ae.mov"],
        stub_key_data_dict: dict = {}
    ):
        self.reader = reader
        self.sequence_length = sequence_length

        # for the ego speed
        self.info_extension = info_extension
        self.stub_key_data_dict = stub_key_data_dict
        if info_extension is not None:
            self.stub_key_data_dict["ego_speed"] = \
                ("tensor", (sequence_length,), -1000)

        self.items = []
        for i in self.reader.namelist():
            if i in ignore_list:
                continue

            f = self.reader.get_io_object(i)
            with av.open(f) as container:
                stream = container.streams.video[0]
                for fps, stride in fps_stride_tuples:
                    sequence_duration = sequence_length / fps
                    t = float(stream.start_time * stream.time_base)
                    s = float(
                        stream.duration * stream.time_base - sequence_duration)
                    while t <= s:
                        self.items.append((i, t, fps))
                        t += stride

            f.close()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        name, t, fps = self.items[index]
        with self.reader.get_io_object(name) as f:

            # get video rotation meta
            info = pymediainfo.MediaInfo.parse(f)
            f.seek(0)

            # decode frames
            frames = []
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

                stream.codec_context.close()

        pts_list = [i.pts for i in frames]
        try:
            expected_ptss = [
                int((t + i / fps) / time_base)
                for i in range(self.sequence_length)
            ]
            candidates = [
                MotionDataset.find_frame_of_nearest_time(
                    frames, pts_list, i)
                for i in expected_ptss
            ]

            # the flags to rotate back
            pil_rotations = {
                90: Image.Transpose.ROTATE_270,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_90
            }
            pts = [
                int(1000 * (i.pts - candidates[0].pts) * time_base + 0.5)
                for i in candidates
            ]
            frame_rotation = int(float(info.video_tracks[0].rotation))
            images = [
                i.to_image().transpose(pil_rotations[frame_rotation])
                if frame_rotation != 0 else i.to_image()
                for i in candidates
            ]
            result = {
                # this PIL Image item should be converted to tensor before data
                # loader collation
                "images": images,
                "pts": torch.tensor(pts),
                "fps": fps
            }

            # extension part
            if self.info_extension is not None:
                info_t = self.info_extension.query_and_interpolate(
                    name,
                    [
                        i + float(1000 * candidates[0].pts * time_base)
                        for i in pts
                    ])
                if all([
                    i is not None and "locations" in i and
                        i["locations"] is not None
                    for i in info_t
                ]):
                    result["ego_speed"] = torch.tensor(
                        [i["locations"]["speed"] for i in info_t])

        except Exception as e:
            print("Data item WARNING: Name {}, time {}, FPS {}, frame count {}, PTS: {}".format(
                name, t, fps, len(frames), pts_list))
            result = {
                "images": [
                    Image.new("RGB", (1280, 720), (128, 128, 128))
                    for i in range(self.sequence_length)
                ],
                "pts": torch.zeros((self.sequence_length), dtype=torch.long),
                "fps": fps
            }

        # add stub values for heterogeneous dataset merging
        for key, data in self.stub_key_data_dict.items():
            if key not in result.keys():
                if data[0] == "tensor":
                    shape, value = data[1:]
                    result[key] = value * torch.ones(shape)
                else:
                    result[key] = data[1]

        return result
