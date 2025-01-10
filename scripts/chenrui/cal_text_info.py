import json
import matplotlib.pyplot as plt
import zipfile
import random

dataset_name = "opendv"

# path = "/cache/aoss.cn-sh-01.sensecoreapi-oss.cn/users/wuzehuan/workspaces/worldmodels/data/nuscenes_v1.0-trainval_caption_v2_train.json"
path = "/cache/aoss.cn-sh-01.sensecoreapi-oss.cn/users/wuzehuan/workspaces/worldmodels/data/opendv_caption.json"
with open(path, 'r') as file:
    data = json.load(file)

# zip_path = "/cache/aoss.cn-sh-01.sensecoreapi-oss.cn/users/wuzehuan/workspaces/worldmodels/data/waymo_caption_train.zip"
# data = {}
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     file_list = zip_ref.namelist()
#     for file_name in file_list:
#         # Check if the file is a JSON file
#         if file_name.endswith('.json'):
#             with zip_ref.open(file_name) as file:
#                 json_content = json.load(file)
#                 data[file_name] = json_content

info = {"time": {}, "weather": {}, "environment": {}}
key_col = ["time", "weather", "environment"]
class_nums = [4, 4, 10]
for key, value in data.items():
    for k in key_col:
        v = value[k].split(",")[0]
        if k == "weather" and v == "clear":
            v = "clear sky"
        if v in info[k].keys():
            info[k][v] += 1
        else:
            info[k][v] = 0 

for class_num, k in zip(class_nums, key_col):
    data = info[k]
    sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
    
    top_15 = dict(list(sorted_data.items())[:class_num])
    other_count = sum(list(sorted_data.values())[class_num:])
    if other_count > 0:
        top_15["Other"] = other_count

    sorted_top_15 = dict(sorted(top_15.items(), key=lambda x: x[1], reverse=True))
    labels = list(sorted_top_15.keys())
    if "Other" in labels:
        index = labels.index("Other")
        labels = labels[:index] + labels[index+1:] + [labels[index]]
    sizes = list(sorted_top_15.values())
    if "Other" in labels:
        sizes = sizes[:index] + sizes[index+1:] + [sizes[index]]

    cmap = plt.cm.get_cmap('Set3')
    colors = [cmap(i) for i in range(12)]
    cmap = plt.cm.get_cmap('Set2')
    colors += [cmap(i) for i in range(5)]
    colors = colors[:len(labels)]

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: ('%.1f%%' % pct) if pct > 3 else '', 
        colors=colors,
        pctdistance = 0.5,
        labeldistance = 1.0,
        counterclock=False,
        startangle = 90,
        rotatelabels = True,
        radius=0.7

    )
    # plt.title(f"Statistics on {k} conditions")
    
    plt.savefig(f"/mnt/afs/user/chenrui4/DWM/scripts/chenrui/{dataset_name}_{k}.svg")

