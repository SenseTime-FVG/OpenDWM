import argparse
import aoss_client.client 
import json
import pickle
import time
import zipfile


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-f", "--range-from", type=int, required=True)
    parser.add_argument(
        "-t", "--range-to", type=int, required=True)
    return parser


def get_scenes_name(samples, filter_datatime="20231222", scenes_remove_set={}):
    scene_dict = {}
    for i, sample in enumerate(samples):
        if ''.join(sample['scene_token'].split('_', 3)[:3]) < filter_datatime:
            continue

        if sample['scene_token'] in scenes_remove_set:
            continue

        if sample['scene_token'] not in scene_dict:
            scene_dict[sample['scene_token']] = [i]
        else:
            scene_dict[sample['scene_token']].append(i)

    scene_names = list(scene_dict.keys())
    scene_names_indices = [scene_dict[i] for i in scene_names]

    return scene_names, scene_names_indices  

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    channels = [
        "left_front_camera", "center_camera_fov120", "right_front_camera",
        "right_rear_camera", "rear_camera", "left_rear_camera"
    ]
    scenes_remove_set = {
        "2024_03_27_14_44_04", "2024_03_27_13_26_05",
        "2024_01_03_11_51_24_AutoCollect_12", "2024_03_23_10_12_27",
        "2024_01_24_14_19_37_AutoCollect_5",
        "2024_01_24_14_19_37_AutoCollect_9",
        "2024_03_27_13_24_05", "2024_03_27_14_36_04",
        "2024_01_26_16_20_00_AutoCollect_1", "2024_03_27_13_28_05",
        "2024_03_27_14_42_04", "2024_03_27_14_38_04", "2024_03_27_10_59_35",
        "2024_03_27_13_20_05", "2024_03_27_14_32_04", "2024_03_27_14_34_04",
        "2024_01_03_11_51_24_AutoCollect_11"
    }
    pkl_list = [
        "/mnt/afs/user/fanjianan/workdir/reserved_data/all_600w_data_0530/pkl/05301738_real_0450_backup_E_qachecked.pkl", 
        "/mnt/afs/user/fanjianan/workdir/reserved_data/all_600w_data_0530/pkl/05301745_real_0530_backup_new300_qachecked.pkl"
    ]
    info_list = [
        j for i in pkl_list
        for j in pickle.load(open(i, 'rb'))["infos"]
    ]
    scene_names, scene_names_indices = get_scenes_name(
        info_list, scenes_remove_set=scenes_remove_set)

    client = aoss_client.client.Client('/mnt/afs/user/wuzehuan/aoss.conf')
    blob_size = 40
    blob_count = (len(scene_names) + blob_size - 1) // blob_size
    print("Blob count {}, blob size {}".format(blob_count, blob_size))
    for blob_id in range(args.range_from, args.range_to):
        part_indices = scene_names_indices[blob_id*blob_size:min((blob_id+1)*blob_size, len(scene_names_indices))]
        upload_path = f"s3://users/wuzehuan/data/uniad/0530/blob_{blob_id:03d}.zip"

        with zipfile.ZipFile("uniad_blob_{}.zip".format(args.range_from), "w", compression=zipfile.ZIP_STORED) as zf:
            for i_id, i in enumerate(part_indices):
                t0 = time.time()
                for j_id, j in enumerate(i):
                    sample_item = info_list[j]
                    with open(sample_item["token"], "r") as f:
                        sample = json.load(f)

                    cameras = sample["cams"]
                    for channel in channels:
                        data = client.get(cameras[channel]["data_path"])
                        assert data is not None, "{} get None".format(cameras[channel]["data_path"])
                        assert len(data) > 0, "{} is empty".format(cameras[channel]["data_path"])
                        zf.writestr(
                            cameras[channel]["data_path"].replace("s3://", ""),
                            data)

                    if sample_item["lidar_path"].endswith("pkl"):
                        s3_path = sample_item["lidar_path"]
                        data = client.get(s3_path)
                        assert data is not None, "{} get None".format(s3_path)
                        assert len(data) > 0, "{} is empty".format(s3_path)
                        zf.writestr(s3_path.replace("s3://", ""), data)
                    else:
                        s3_path = sample["ceph_root"][:sample["ceph_root"].rfind("/")].replace(
                            "meta_json", sample_item["lidar_path"])
                        data = client.get(s3_path)
                        assert data is not None, "{} get None".format(s3_path)
                        assert len(data) > 0, "{} is empty".format(s3_path)
                        zf.writestr(s3_path.replace("s3://", ""), data)

                    if (j_id + 1) % 200 == 0:
                        print("Blob {}, scene {}, {} frames are packed.".format(blob_id, i_id, j_id + 1))

                print("Blob {}, scene {} is packed in {:.1f} s".format(blob_id, i_id, time.time() - t0))

        t0 = time.time()
        client.upload_file(upload_path, "uniad_blob_{}.zip".format(args.range_from))
        print("Blob {} is uploaded in {:.1f}".format(blob_id, time.time() - t0))
