# Open Driving World Models (OpenDWM)

[![Youtube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/j9RRj-xzOA4) [<img src=https://img.shields.io/badge/%E4%B8%AD%E6%96%87%E7%AE%80%E4%BB%8B-blue>](README_intro_zh.md) 

https://github.com/user-attachments/assets/649d3b81-3b1f-44f9-9f51-4d1ed7756476

Welcome to the OpenDWM project! This is an open-source initiative, focusing on autonomous driving video generation. Our mission is to provide a high-quality, controllable tool for generating autonomous driving videos using the latest technology. We aim to build a codebase that is both user-friendly and highly reusable, and hope to continuously improve the project through the collective wisdom of the community.

The driving world models generate multi-view images or videos of autonomous driving scenes based on text and road environment layout conditions. Whether it's the environment, weather conditions, vehicle type, or driving path, you can adjust them according to your needs.

The highlights are as follows:

1. **Significant improvement in the environmental diversity.** Through the use of multiple datasets, the model's generalization ability has been enhanced like never before. Take the example of a generation task controlled by layout conditions, such as a snowy city street or a lakeside highway with distant snow mountains, these scenarios are impossible tasks for generative models trained with a single dataset.

2. **Greatly improved generation quality.** Support for popular model architectures (SD 2.1, 3.5) enables more convenient utilization of the advanced pre-training generation capabilities within the community. Various training techniques, including multitasking and self-supervision, allow the model to utilize the information in autonomous driving video data more effectively.

3. **Convenient evaluation.** Evaluation follows the popular framework `torchmetrics`, which is easy to configure, develop, and integrate into the pipeline. Public configurations (such as FID, FVD on the nuScenes validation set) are provided to align other research works.

Furthermore, our code modules are designed with high reusability in mind, for easy application in other projects.

Currently, the project has implemented the following papers:

> [UniMLVG: Unified Framework for Multi-view Long Video Generation with Comprehensive Control Capabilities for Autonomous Driving](https://sensetime-fvg.github.io/UniMLVG)<br>
> Rui Chen<sup>1,2</sup>, Zehuan Wu<sup>2</sup>, Yichen Liu<sup>2</sup>, Yuxin Guo<sup>2</sup>, Jingcheng Ni<sup>2</sup>, Haifeng Xia<sup>1</sup>, Siyu Xia<sup>1</sup><br>
> <sup>1</sup>Southeast University <sup>2</sup>SenseTime Research

> [MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction](https://sensetime-fvg.github.io/MaskGWM)<br>
> Jingcheng Ni, Yuxin Guo, Yichen Liu, Rui Chen, Lewei Lu, Zehuan Wu<br>
> SenseTime Research

## News

* [2025/3/17] Experimental release the [Interactive Generation with Carla](docs/InteractiveGeneration.md)
* [2025/3/7] Release the [LiDAR Generation](#lidar-models)
* [2025/3/4] Release the [CTSD 3.5 with layout condition](#video-models)
* [2025/2/7] Release the [UniMLVG](#video-models)

## Setup

Hardware requirement:

* Training and testing multi-view image generation or short video (<= 6 frames per iteration) generation requires 32GB GPU memory (e.g. V100)
* Training and testing multi-view long video (6 ~ 40 frames per iteration) generation requires 80GB GPU memory (e.g. A100, H100)

Software requirement:

* git (>= 2.25)
* python (>= 3.9)

Install the [PyTorch](https://pytorch.org/) >= 2.5:

```
python -m pip install torch==2.5.1 torchvision==0.20.1
```

Clone the repository, then install the dependencies.

```
cd OpenDWM
git submodule update --init --recursive
python -m pip install -r requirements.txt
```

## Models

### Video Models

Our cross-view temporal SD (CTSD) pipeline support loading the pretrained SD 2.1, 3.0, 3.5, or the checkpoints we trained on the autonomous driving datasets.

| Base model | Text conditioned <br/> driving generation | Text and layout (box, map) <br/> conditioned driving generation |
| :-: | :-: | :-: |
| [SD 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | [Config](configs/ctsd/multi_datasets/ctsd_21_tirda_nwao.json), [Download](http://103.237.29.236:10030/ctsd_21_tirda_nwao_30k.pth) | [Config](configs/ctsd/multi_datasets/ctsd_21_tirda_bm_nwa.json), [Download](http://103.237.29.236:10030/ctsd_21_tirda_bm_nwa_30k.pth) |
| [SD 3.0](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | | [UniMLVG Config](configs/ctsd/unimlvg/ctsd_unimlvg_stage3_tirda_bm_nwa.json), [Download](http://103.237.29.236:10030/ctsd_unimlvg_tirda_bm_nwa_60k.pth) |
| [SD 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | [Config](configs/ctsd/multi_datasets/ctsd_35_tirda_nwao.json), [Download](http://103.237.29.236:10030/ctsd_35_tirda_nwao_20k.pth) | [Config](configs/ctsd/multi_datasets/ctsd_35_tirda_bm_nwao.json), [Download](http://103.237.29.236:10030/ctsd_35_tirda_bm_nwao_40k.pth) |

### LiDAR Models

You can download our pre-trained tokenzier and generation model in the following link.

| Model Architecture | Configs | Checkpoint Download |
| :-: | :-: | :-: |
| VQVAE | [Config](configs/lidar/lidar_vqvae_nwa.json) | [checkpoint](http://103.237.29.236:10030/lidar_vqvae_nwa_60k.pth), [blank code ](http://103.237.29.236:10030/lidar_vqvae_nwa_60k_blank_code.pkl) |
| MaskGIT | [Config](configs/lidar/lidar_maskgit_layout_ns.json)| [checkpoint](http://103.237.29.236:10030/lidar_maskgit_nusc_150k.pth) |
| Temporal MaskGIT |  |  |
## Examples

### T2I, T2V generation with CTSD pipeline

Download base model (for VAE, text encoders, scheduler config) and driving generation model checkpoint, and edit the [path](examples/ctsd_35_6views_image_generation.json#L102) and [prompts](examples/ctsd_35_6views_image_generation.json#L221) in the JSON config, then run this command.

```bash
PYTHONPATH=src python examples/ctsd_generation_example.py -c examples/ctsd_35_6views_image_generation.json -o output/ctsd_35_6views_image_generation
```

### Layout conditioned T2V generation with CTSD pipeline

1. Download base model (for VAE, text encoders, scheduler config) and driving generation model checkpoint, and edit the [path](examples/ctsd_35_6views_video_generation_with_layout.json#L156) in the JSON config.
2. Download layout resource package ([nuscenes_scene-0627_package.zip](http://103.237.29.236:10030/nuscenes_scene-0627_package.zip), or [carla_town04_package](http://103.237.29.236:10030/carla_town04_package.zip)) and unzip to the `{RESOURCE_PATH}`. Then edit the meta [path](examples/ctsd_35_6views_video_generation_with_layout.json#L162) as `{RESOURCE_PATH}/data.json` in the JSON config.
3. Run this command to generate the video.

```bash
PYTHONPATH=src python src/dwm/preview.py -c examples/ctsd_35_6views_video_generation_with_layout.json -o output/ctsd_35_6views_video_generation_with_layout
```

### Layout conditioned LiDAR generation with MaskGIT pipeline

1. Download LiDAR VQVAE and LiDAR MaskGIT generation model checkpoint.
2. Prepare the dataset ( [nuscenes_scene-0627_lidar_package.zip](http://103.237.29.236:10030/nuscenes_scene-0627_lidar_package.zip) ).
3. Modify the values of `json_file`, `vq_point_cloud_ckpt_path`, `vq_blank_code_path` and `model_ckpt_path` to the paths of your dataset and checkpoints in the json file `examples/lidar_maskgit_preview.json` .
4. Run the following command to visualize the LiDAR of the validation set and save the generated point cloud as `.bin` file.

```bash
PYTHONPATH=src python src/dwm/preview.py -c examples/lidar_maskgit_preview.json -o output/test
```

## Train

Preparation:

1. Download the base models.
2. Download and process [datasets](docs/Datasets.md).
3. Edit the configuration file (mainly the path of the model and dataset under the user environment).

Once the config file is updated with the correct model and data information, launch training by:

```
PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd python src/dwm/train.py -c {YOUR_CONFIG} -o output/{YOUR_WORKSPACE}
```

Or distributed training by:

```
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd python -m torch.distributed.run --nnodes $WORLD_SIZE --nproc-per-node 8 --node-rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT src/dwm/train.py -c {YOUR_CONFIG} -o output/{YOUR_WORKSPACE}
```

Then you can check the preview under `output/{YOUR_WORKSPACE}/preview`, and get the checkpoint files from `output/{YOUR_WORKSPACE}/checkpoints`.

Some training tasks require multi stages (for the configurations with names of `train_warmup.json` and `train.json`), you should fill the path of the saved checkpoint from the previous stage into the following stage (for [example](configs/ctsd/single_dataset/ctsd_21_tirda_bm_nusc_a.json#L200)), then launch the training of this following stage.

## Evaluation

We have integrated the functions of FID and FVD metric evaluation in the pipeline, which involves filling in the [validation set](configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_nusc_a.json#L394) ([source](configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_nusc_a.json#L398), [sampling interval](configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_nusc_a.json#L408)) and [evaluation parameters](configs/ctsd/single_dataset/ctsd_21_crossview_tirda_bm_nusc_a.json#L192) (for example, the [number of frames](configs/ctsd/single_dataset/ctsd_21_tirda_bm_nusc_a.json#L212) of each video segment to be measured in FVD) in the configuration file.

The specific call method is as follows.

```
PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd python src/dwm/evaluate.py -c {YOUR_CONFIG} -o output/{YOUR_WORKSPACE}
```

Or distributed evaluation by `torch.distributed.run`, similar to the distributed training.

## Development

### Folder structure

* `configs` The config files for data and pipeline with different arguments.
* `examples` The inference code and configurations.
* `externals` The dependency projects.
* `src/dwm` The shared components of this project.
  * `datasets` implements `torch.utils.data.Dataset` for our training pipeline by reading multi-view, LiDAR and temporal data, with optional text, 3D box, HD map, pose, camera parameters as conditions.
  * `fs` provides flexible access methods following `fsspec` to the data stored in ZIP blobs, or in the S3 compatible storage services.
  * `metrics` implements `torchmetrics` compatible classes for quantitative evaluation.
  * `models` implements generation models and their building blocks.
  * `pipelines` implements the training logic for different models.
  * `tools` provides dataset and file processing scripts for faster initialization and reading.

Introduction about the [file system](src/dwm/fs/README.md), and [dataset](src/dwm/datasets/README.md).
