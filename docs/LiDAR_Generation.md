# Layout-Condition LiDAR Generation with Masked Generative Transformer


## Introduction

We propose a pipeline for generating LiDAR data conditioned on layout information using the Mask Generative Image Transformer (MaskGIT) [[1]](#1). Our approach builds upon the models introduced in Copilot4D [[5]](#5) and UltraLiDAR [[4]](#4).

## Method

We first train a Vector Quantized Variational AutoEncoder (VQ-VAE) to tokenize LiDAR data into a 2D latent space. Then,
the LiDAR MaskGIT model incorporates the layout information such as High-definition maps (HDmaps) and 3D object
bounding boxes (3Dboxes) and guides the generation of LiDAR in the latent space.

### LiDAR VQ-VAE

We build our LiDAR VQ-VAE following the approach of UltraLiDAR [[4]](#4) and Copilot4D [[5]](#5). The point cloud $\mathbf{x}$ is fed into a voxelizer $V$ and converted into a Birds-Eye-View (BEV) image. Additionally, we adopt an auxiliary depth rendering branch during decoding. Specifically, given a latent representation $\mathbf z$ of a LiDAR point cloud, the regular decoder transforms the latent back into the original voxel shape, and binary cross entropy loss is used to optimize the network. Furthermore, a depth render network $D$ decodes the latent into a 3D feature voxel, which is used to render the depth of each point. For any point $p$ in $\mathbf x$, we sample a ray from the LiDAR center to $p$ and use the method from [[3]](#3)  to calculate the depth value in this direction. For further details, please refer to [[5]](#5).


<p id="fig-main">
    <img src="https://github.com/user-attachments/assets/cbb2cab3-b819-4f70-baa9-a53cfcd693e9" alt>
</p>


### LiDAR MaskGIT

Our LiDAR MaskGIT is designed to generate sequences of LiDAR point clouds conditioned on layouts. Specifically, given the LiDAR point clouds from the first $k$ frames, our model can predict subsequent LiDAR point clouds at following timestamps guided by layout conditions such as 3D bounding boxes (3D boxes) and high-definition maps (HD maps). Additionally, we extend our model to directly generate LiDAR data without reference frames.

We follow the Copilot4D framework to build our LiDAR MaskGIT, as illustrated in Figure [1](#fig-main). The input to LiDAR MaskGIT is the masked quantized latent, where masked regions are filled with mask tokens, and the model outputs the probability distribution of VQ-VAE codebook IDs at each position. The architecture employs a spatio-temporal transformer that interleaves spatial and temporal attention. We also adopt free space suppression to encourage more diverse generation results. Please refer to~\cite{copilot4d} for additional details. Furthermore, our model can generate LiDAR data based on layout conditions, which we describe below.

#### Layout Conditioning
Since LiDAR point clouds are encoded into 2D BEV space, both 3D boxes and HD maps are also projected into BEV images. Similar to [[2]](#2), each instance is mapped into the color space. These two conditional images are concatenated and processed through a lightweight image adapter, generating multi-level features that are added to the latent representation at the corresponding transformer layers.


## Experiment
We conduct our experiments on nuScenes [[6]](#6) and KITTI360 [[7]](#7) datasets and report the quantitative results in the following table.

<table id="tab-quant_results" >
  <caption style="caption-side:bottom"></caption>
  <tr>
    <th>Dataset</th>
    <th>IoU</th>
    <th>CD</th>
    <th>MMD</th>
    <th>JSD</th>
  </tr>
  <tr>
    <td>nuScenes</td>
    <td>0.055</td>
    <td>4.438</td>
    <td>-----</td>
    <td>-----</td>
  </tr>
  <tr>
    <td>KITTI360</td>
    <td>0.045</td>
    <td>5.838</td>
    <td>0.00461</td>
    <td>0.471</td>
  </tr>
  <tr>
    <td>nuScenes Temporal</td>
    <td>0.126</td>
    <td>3.487</td>
    <td>-----</td>
    <td>-----</td>
  </tr>
  <tr>
    <td>KITTI360 Temporal</td>
    <td>0.117</td>
    <td>3.347</td>
    <td>0.00411</td>
    <td>0.313</td>
  </tr>
</table>

## Visualization
In this section, we provide some qualitative results of our method. First, you need to install `open3d` in the environment. The visualization code is provided in `src/dwm/utils/lidar_visualizer.py`. You can run the following bash script to generate visualization results.
```
python src/dwm/utils/lidar_visualizer.py \
--data_type nuscenes \
--lidar_root /path/to/generated/lidar \
--data_root /path/to/nuscenes/json \
--output_path /path/to/output/folder \
```
### Single Frame Generation
#### NuScenes
<table>
  <caption style="caption-side:bottom"></caption>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9e775f65-e35d-4169-91a3-17a5a92b36eb"></td>
    <td><img src="https://github.com/user-attachments/assets/9da1436e-0dab-4377-90b6-2fde4f9b06cf"></td>
    <td><img src="https://github.com/user-attachments/assets/b70c6cac-486b-4fc6-b0dd-e92bb74abcf3"></td>
    <td><img src="https://github.com/user-attachments/assets/75ba14ee-d408-445f-a7ef-adac573e7684"></td>
  </tr>
</table>

#### KITTI360
<table>
  <caption style="caption-side:bottom"></caption>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/14e87063-7ba1-47b5-aa53-bd9b387fdcdc"></td>
    <td><img src="https://github.com/user-attachments/assets/02014996-9ff2-4dfc-8ad4-559203087a50"></td>
    <td><img src="https://github.com/user-attachments/assets/19c1d76c-b25c-4ffd-9837-26ecaf7c942c"></td>
    <td><img src="https://github.com/user-attachments/assets/3e621d8d-824e-4b1e-a879-777a70018a25"></td>
  </tr>
</table>


### Autoregressive Generation
#### NuScenes
<table>
  <caption style="caption-side:bottom"></caption>
  <tr>
    <td><center><img src="https://github.com/user-attachments/assets/2d6ae389-c5f5-4da4-837f-d7c580ad1294">Reference Frame</center></td>
    <td><center><img src="https://github.com/user-attachments/assets/fec86625-9be2-4f49-94a7-ff5f8e0648ba">Frame=2</center></td>
    <td><center><img src="https://github.com/user-attachments/assets/89f37a7f-b2a3-4d29-ac54-29e2f80282e0">Frame=4</center></td>
    <td><center><img src="https://github.com/user-attachments/assets/516bcdf0-34e0-4f30-80d6-d9ff44369282">Frame=6</center></td>
  </tr>
</table>

#### KITTI360
<table>
  <caption style="caption-side:bottom"></caption>
  <tr>
    <td><center><img src="https://github.com/user-attachments/assets/9a23b4e2-7687-419d-8486-ebb95abca5dd">Reference Frame</center></td>
    <td><center><img src="https://github.com/user-attachments/assets/760e3c14-eb47-4a6e-b071-47fc9b8d6ba4">Frame=2</center></td>
    <td><center><img src="https://github.com/user-attachments/assets/2c4026c1-55c2-43fc-aaff-1d4be77eb682">Frame=4</center></td>
    <td><center><img src="https://github.com/user-attachments/assets/8b2712c2-bc51-45f5-9f05-42c5d5f92a82">Frame=6</center></td>
  </tr>
</table>

## References

<a id="1">[1]</a>  Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315–11325, 2022.

<a id="2">[2]</a>  Rui Chen, Zehuan Wu, Yichen Liu, Yuxin Guo, Jingcheng Ni, Haifeng Xia, and Siyu Xia. Unimlvg: Unified framework for multi-view long video generation with comprehensive control capabilities for autonomous driving. arXiv preprint arXiv:2412.04842, 2024.

<a id="3">[3]</a>  Cheng Sun, Min Sun, and Hwann-Tzong Chen. Improved direct voxel grid optimization for radiance fields reconstruction. arXiv preprint arXiv:2206.05085, 2022.

<a id="4">[4]</a>  Yuwen Xiong, Wei-Chiu Ma, Jingkang Wang, and Raquel Urtasun. Ultralidar: Learning compact representations for lidar completion and generation. arXiv preprint arXiv:2311.01448, 2023.

<a id="5">[5]</a>  Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, and Raquel Urtasun. Copilot4d: Learning unsupervised world models for autonomous driving via discrete diffusion. arXiv preprint arXiv:2311.01017, 2023.

<a id="6">[6]</a> Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan,
and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 11621–11631, 2020.

<a id="7">[7]</a> Yiyi Liao, Jun Xie, and Andreas Geiger. Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 45(3):3292–3310, 2022.
