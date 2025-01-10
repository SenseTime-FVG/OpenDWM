import os
import filecmp
import shutil

def sync_folders(root, a_dir, b_dir):
    comparison = filecmp.dircmp(a_dir, b_dir)
    
    only_in_a = comparison.left_only
    print(len(only_in_a))

    for item in only_in_a:
        src_path = os.path.join(root, item)
        dest_path = os.path.join(b_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)  
            print(f"files are done: {src_path} -> {dest_path}")

folder_name = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]

root = '/mnt/storage/user/chenrui4/datasets/nuscenes/v1.0-trainval/samples'
a_root = '/mnt/storage/user/chenrui4/projects/BEVFormer/data/nuscenes/samples'
b_root = '/mnt/storage/user/chenrui4/Tasks/stage2_sd3_crossview_temporal_ti_bm_12hz_nusc_argo_waymo_export/samples'

for name in folder_name:
    sync_folders(os.path.join(root, name), os.path.join(a_root, name), os.path.join(b_root, name))


