
# Datasets

Currently we support 4 datasets: nuScenes, Waymo Perception, Argoverse 2 Sensor, OpenDV.

## nuScenes

1. Download the [nuScenes](https://www.nuscenes.org/download) dataset files to `{NUSCENES_TGZ_ROOT}` on your file system. After the dataset is downloaded, there will be some `*.tgz` files under path `{NUSCENES_TGZ_ROOT}`.

2. Since the TGZ format does not support random access to content, we recommend converting these files to ZIP format using the following command lines:

```
mkdir -p {NUSCENES_ZIP_ROOT}
python src/dwm/tools/tar2zip.py -i {NUSCENES_TGZ_ROOT}/v1.0-trainval_meta.tgz -o {NUSCENES_ZIP_ROOT}/v1.0-trainval_meta.zip
python src/dwm/tools/tar2zip.py -i {NUSCENES_TGZ_ROOT}/v1.0-trainval01_blobs.tgz -o {NUSCENES_ZIP_ROOT}/v1.0-trainval01_blobs.zip
python src/dwm/tools/tar2zip.py -i {NUSCENES_TGZ_ROOT}/v1.0-trainval02_blobs.tgz -o {NUSCENES_ZIP_ROOT}/v1.0-trainval02_blobs.zip
...
python src/dwm/tools/tar2zip.py -i {NUSCENES_TGZ_ROOT}/v1.0-trainval10_blobs.tgz -o {NUSCENES_ZIP_ROOT}/v1.0-trainval10_blobs.zip
```

3. Now the `{NUSCENES_ZIP_ROOT}` is ready to update the nuScenes file system of your config file, for [example](../configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_nusc_a.json#L12).

4. Prepare the HD map data.

    1. Download the `nuScenes-map-expansion-v1.3.zip` file from the [nuScenes](https://www.nuscenes.org/download) to `{NUSCENES_ZIP_ROOT}`.

    2. Add the file into the [config](../configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_nusc_a.json#L27), so the dataset can load the map data.

5. *Optional*. When the 3D box conditions are used for training, the 12hz metadata is recommended.

    1. Download 12 Hz nuScenes meta from [Corner Case Scene Generation](https://coda-dataset.github.io/w-coda2024/track2/). After the metadata is downloaded, there will be `interp_12Hz.tar` file.

    2. Extract and repack the 12 Hz metadata to `interp_12Hz_trainval.zip`, then update the [FS](../configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_nusc_a.json#L15) and [dataset name](../configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_nusc_a.json#L206) in the config.

```
python -m tarfile -e interp_12Hz.tar
cd data/nuscenes
python -m zipfile -c ../../interp_12Hz_trainval.zip interp_12Hz_trainval/
cd ../..
rm -rf data/
```

6. *Alternative solution for 5*. In the case of a broken download link, you can also regenerate 12 Hz annotations according to the instructions of [ASAP](https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md) from the origin nuScenes dataset.

7. Download the annotation of text prompt and update the config following the section [text description for images](#text-description-for-images)

## Waymo

There are two versions of the Waymo Perception dataset. This project chooses version 1 (>= 1.4.2) because only this version provides HD map annotation, while version 2 does not provide HD map annotation.

1. *Optional*. The Waymo Perception 1.x requires protobuffer, if you try to avoid installing waymo_open_dataset and its dependencies, you need to compile the proto files. Install the [proto buffer compiler](https://github.com/protocolbuffers/protobuf/releases/tag/v25.4), then run following commands to compile proto files. After compilation, `import waymo_open_dataset.dataset_pb2` works by adding `externals/waymo-open-dataset/src` to the environmant variable `PYTHONPATH`.

```
cd externals/waymo-open-dataset/src
protoc --proto_path=. --python_out=. waymo_open_dataset/*.proto
protoc --proto_path=. --python_out=. waymo_open_dataset/protos/*.proto
```

2. Download the [Waymo Perception](https://waymo.com/open/download) dataset (>= 1.4.2 for the annotation of HD map) to `{WAYMO_ROOT}`. After the dataset is downloaded, there will be some `*.tfrecord` files under the path `{WAYMO_ROOT}/training` and `{WAYMO_ROOT}/validation`.

3. Then make information JSON files to support inner-scene random access, by

```
PYTHONPATH=src python src/dwm/tools/dataset_make_info_json.py -dt waymo -i {WAYMO_ROOT}/training -o {WAYMO_ROOT}/training.info.json
PYTHONPATH=src python src/dwm/tools/dataset_make_info_json.py -dt waymo -i {WAYMO_ROOT}/validation -o {WAYMO_ROOT}/validation.info.json
```

4. Now the `{WAYMO_ROOT}` and its information JSON files are ready to update the Waymo dataset of your config file, for [example](../configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_waymo.json#L182).

5. Download the annotation of text prompt and update the config following the section [text description for images](#text-description-for-images)

## Argoverse

1. Download the [Argoverse 2 Sensor](https://www.argoverse.org/av2.html#download-link) dataset files to `{ARGOVERSE_ROOT}` on your file system. After the dataset is downloaded, there will be some `*.tar` files under path `{ARGOVERSE_ROOT}`.

2. Then make information JSON files to accelerate the loading speed, by:

```
PYTHONPATH=src python src/dwm/tools/dataset_make_info_json.py -dt argoverse -i {ARGOVERSE_ROOT} -o {ARGOVERSE_ROOT}
```

3. Now the `{ARGOVERSE_ROOT}` is ready to update the Argoverse file system of your config file, for [example](../configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_argo.json#L184).

4. Download the annotation of text prompt and update the config following the section [text description for images](#text-description-for-images)

## OpenDV

1. Download the [OpenDV](https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv) dataset video files to `{OPENDV_ORIGIN_ROOT}` on your file system, and the meta file to `{OPENDV_JSON_META_PATH}` prepared as JSON format. After the dataset is downloaded, there will be about 2K video files under the path `{OPENDV_ORIGIN_ROOT}` in the format of `.mp4` and `.webp`.

2. *Optional.* It is recommended to transcode the original video files for better read and seek performance during training, by:

```
apt update && apt install -y ffmpeg
python src/dwm/tools/transcode_video.py -c src/dwm/tools/transcode_video.json -i {OPENDV_ORIGIN_ROOT} -o {OPENDV_ROOT}
```

3. Now the `{OPENDV_ORIGIN_ROOT}` (or `{OPENDV_ROOT}`) is ready to update the OpenDV [file system config](../configs/ctsd/multi_datasets/ctsd_21_tirda_nwao.json#L31), and `{OPENDV_JSON_META_PATH}` to update the [dataset config](../configs/ctsd/multi_datasets/ctsd_21_tirda_nwao.json#L409).

## KITTI360

Register an account and download the [KITTI360](https://www.cvlibs.net/datasets/kitti-360/download.php) dataset. We only require the LiDAR data from KITTI360, so you only need to download the Raw Velodyne Scans, 3D Bounding Boxes, and Vehicle Poses. The scenes `2013_05_28_drive_0000_sync` and `2013_05_28_drive_0002_sync` are used for validation, while all other scenes are used as training data. You may keep the files in their original zip format, as our code processes them automatically.

## Text description for images

We made the image captions for both nuScenes, Waymo, Argoverse, OpenDV datasets by [DriveMLM](https://arxiv.org/abs/2312.09245) model. The caption files are available here.

| Dataset | Downloads |
| :-: | :-: |
| nuScenes | [mini](https://huggingface.co/datasets/wzhgba/opendwm-data/resolve/main/nuscenes_v1.0-mini_caption_v2.zip?download=true), [trainval](https://huggingface.co/datasets/wzhgba/opendwm-data/resolve/main/nuscenes_v1.0-trainval_caption_v2.zip?download=true) |
| Waymo | [trainval](https://huggingface.co/datasets/wzhgba/opendwm-data/resolve/main/waymo_caption_v2.zip?download=true) |
| Argoverse | [trainval](https://huggingface.co/datasets/wzhgba/opendwm-data/resolve/main/av2_sensor_caption_v2.zip?download=true) |
| OpenDV | [all](https://huggingface.co/datasets/wzhgba/opendwm-data/resolve/main/opendv_caption.zip?download=true) |

1. Download the packages above and unzip them.

2. You will get some JSON files such as `nuscenes_v1.0-trainval_caption_v2_train.json` (text of image captions), `nuscenes_v1.0-trainval_caption_v2_times_train.json` (indicates the moments of frames selected for image caption annotations in each scenario)

3. Update [paths](../configs/ctsd/multi_datasets/ctsd_35_tirda_bm_nwao.json#L315) to those files in your config. Please notice that the paths in the "image_description_settings" of all the datasets should be updated to your local downloaded and extracted files.
