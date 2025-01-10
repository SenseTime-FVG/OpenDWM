import json
import re
import zipfile


dataset_name = "nuscenes"
sensor_channels = \
    [   
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT"
    ]
color = ["red", "green", "blue", "black", "yellow", "brown", "white", 
        "purple", "grey", "beige", "maroon", "orange", "cream", "UPS",
        "silver", "tan", "copper-colored", "dark-colored", "dark"]
vehicle_name = ["SUV", "SUVs", "bus", "buses", "car", "cars", "truck", 
                "trucks", "van", "vehicle", "sedan", "Volkswagen", "pickup", 
                "taxi", "Mercedes-Benz", "minivan", "RV", "limousine", "trolley",
                "shuttle", "tram", "semi-truck", "motorbike"]

color_word = set()
vehicle_world = set()

if dataset_name == "carla":
    # carla
    json_file = "/mnt/afs/user/chenrui4/datasets/carla/20241105143335963_Pedestrian_town04/data.json"
    sample_list = []
    with open(json_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            sample_list.append(sample)
    for sample in sample_list:
        for s in sensor_channels:
            image_description = sample["camera_infos"][s]["image_description"]
            image_description = re.sub(r"[.,]", "", image_description)
            words = image_description.split(" ")
    for i, word in enumerate(words):
            if word in color and i+1 < len(words) and words[i + 1] not in vehicle_name:
                vehicle_world.add(words[i + 1])
    for i, word in enumerate(words):
        if word in vehicle_name and i > 0 and words[i - 1] not in color:
                color_word.add(words[i - 1])
else:
    if dataset_name == "nuscenes":
        path = "/cache/aoss.cn-sh-01.sensecoreapi-oss.cn/users/wuzehuan/workspaces/worldmodels/data/nuscenes_v1.0-trainval_caption_v2_val.json"
        with open(path, 'r') as file:
            data = json.load(file)
 
    elif dataset_name == "waymo":
        zip_path = "/cache/aoss.cn-sh-01.sensecoreapi-oss.cn/users/wuzehuan/workspaces/worldmodels/data/waymo_caption_val.zip"
        data = {}
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for file_name in file_list:
                # Check if the file is a JSON file
                if file_name.endswith('.json'):
                    with zip_ref.open(file_name) as file:
                        json_content = json.load(file)
                        data[file_name] = json_content
    elif dataset_name == "waymo":
        path = "/cache/aoss.cn-sh-01.sensecoreapi-oss.cn/users/wuzehuan/workspaces/worldmodels/data/av2_sensor_caption_v2_val.json"
        with open(path, 'r') as file:
            data = json.load(file)

    for d in data.values():
        image_description = d["image_description"]
        image_description = re.sub(r"[.,]", "", image_description)
        words = image_description.split(" ")
        for i, word in enumerate(words):
            if word in color and i+1 < len(words) and words[i + 1] not in vehicle_name:
                vehicle_world.add(words[i + 1])
        for i, word in enumerate(words):
            if word in vehicle_name and i > 0 and words[i - 1] not in color:
                color_word.add(words[i - 1])

print(vehicle_world)
print(color_word)
