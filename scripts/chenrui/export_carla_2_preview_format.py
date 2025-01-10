import json
import os
import shutil
import copy


json_file = "/mnt/storage/user/chenrui4/datasets/carla/20241205143543862_Pedestrian_town04/data.json"
package_name = "carla_town04_package"
output_path = f"/mnt/storage/user/chenrui4/datasets/carla/{package_name}"
os.makedirs(output_path, exist_ok=True)
output_json_path = os.path.join(output_path, "data.json")
with open(output_json_path, "w") as f: pass

start_time = 0
frames_count = 227
reference_frame_count = 0

sample_list = []
with open(json_file, 'r') as f:
    for line in f:
        sample = json.loads(line)
        sample_list.append(sample)
prefix = "/".join(json_file.split("/")[:-2])

data_package = dict()
frame_data = dict()
timestamp = 0
sensor_channels = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT"
]

first_frame_index = -1
for frame in range(frames_count):
    if sample_list[frame]["timestamp"] < start_time:
        continue
    elif first_frame_index == -1:
        first_frame_index = frame

    frame_data["camera_infos"] = dict()
    for sensor_channel in sensor_channels:
        frame_data["camera_infos"][sensor_channel] = dict()

    for view in range(len(sensor_channels)):    
        frame_data["camera_infos"][sensor_channels[view]]["extrin"] = \
            sample_list[frame]["camera_infos"][sensor_channels[view]]["extrin"]
        frame_data["camera_infos"][sensor_channels[view]]["intrin"] = \
            sample_list[frame]["camera_infos"][sensor_channels[view]]["intrin"]
        frame_data["camera_infos"][sensor_channels[view]]["image_description"] = \
            sample_list[frame]["camera_infos"][sensor_channels[view]]["image_description"]
        
        if frame < reference_frame_count:
            image_output_path = os.path.join(
                output_path, sensor_channels[view], "rgb", f"{timestamp}.png")
            os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
            shutil.copy(sample_list[frame]["camera_infos"][
                sensor_channels[view]]["rgb"].replace("data", prefix),
                image_output_path)
            frame_data["camera_infos"][sensor_channels[view]]["rgb"] = \
                os.path.relpath(image_output_path, output_path)
        else:
            frame_data["camera_infos"][sensor_channels[view]]["rgb"] = None

        _3dbox_output_path = os.path.join(
            output_path, sensor_channels[view], "3dbox", f"{timestamp}.png")
        os.makedirs(os.path.dirname(_3dbox_output_path), exist_ok=True)
        shutil.copy(sample_list[frame]["camera_infos"][
                sensor_channels[view]]["3dbox"].replace("data", prefix),
                _3dbox_output_path)
        frame_data["camera_infos"][sensor_channels[view]]["3dbox"] = \
            os.path.relpath(_3dbox_output_path, output_path)

        hdmap_output_path = os.path.join(
            output_path, sensor_channels[view], "hdmap", f"{timestamp}.png")
        os.makedirs(os.path.dirname(hdmap_output_path), exist_ok=True)
        shutil.copy(sample_list[frame]["camera_infos"][
                sensor_channels[view]]["hdmap"].replace("data", prefix),
                hdmap_output_path)
        frame_data["camera_infos"][sensor_channels[view]]["hdmap"] = \
            os.path.relpath(hdmap_output_path, output_path)

    frame_data["timestamp"] = timestamp 
    timestamp += 1/10
    timestamp = round(timestamp, 4)
    frame_data["ego_pose"] = sample_list[frame]["ego_pose"]
    data_package[frame-first_frame_index] = copy.deepcopy(frame_data)

with open(output_json_path, "a") as json_file:
    json.dump(data_package, json_file, indent=4)
    json_file.write("\n")


