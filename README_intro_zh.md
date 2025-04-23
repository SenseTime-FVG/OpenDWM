# Open Driving World Models (OpenDWM)

[[English README](README.md)]

https://github.com/user-attachments/assets/649d3b81-3b1f-44f9-9f51-4d1ed7756476

[视频链接](https://youtu.be/j9RRj-xzOA4)

欢迎来到 OpenDWM 项目！这是一个专注于自动驾驶视频生成的开源项目。我们的使命是提供一个高质量、可控的、使用最新技术的自动驾驶视频生成工具。我们的目标是构建一个既用户友好，又高度可复用的代码库，并希望通过聚集社区智慧，不断改进。

驾驶世界模型根据文本和道路环境布局条件，生成自动驾驶场景的多视角图像或视频。无论是环境、天气条件、车辆类型，还是驾驶路径，你都可以根据需求来调整。

亮点如下：

1. **透明且可复现的训练。** 我们提供完整的训练代码和配置，让大家可以根据需要进行实验复现、在自有数据上微调、定制开发功能。

2. **环境多样性的显著改进。** 通过对多个数据集的使用，模型的泛化能力得到前所未有的提升。以布局条件控制生成任务为例，下雪的城市街道，远处有雪山的湖边高速路，这些场景对于仅使用单一数据集训练的生成模型都是不可能的任务。

3. **大幅提升生成质量。** 对于流行模型架构（SD 2.1, 3.5）的支持，可以更便捷地利用社区内先进的预训练生成能力。包括多任务、自监督在内的多种训练技巧，让模型更有效地利用视频数据里的信息。

4. **方便测评。** 测评遵循流行框架 `torchmetrics`，易于配置、开发、并集成到已有管线。一些公开配置（例如在 nuScenes 验证集上的 FID, FVD）用于和其他研究工作对齐。

此外，我们设计的代码模块考虑到了相当程度的可复用性，以便于在其他项目中应用。

截止现在，本项目实现了以下论文中的技巧：

> [UniMLVG: Unified Framework for Multi-view Long Video Generation with Comprehensive Control Capabilities for Autonomous Driving](https://sensetime-fvg.github.io/UniMLVG)<br>
> Rui Chen<sup>1,2</sup>, Zehuan Wu<sup>2</sup>, Yichen Liu<sup>2</sup>, Yuxin Guo<sup>2</sup>, Jingcheng Ni<sup>2</sup>, Haifeng Xia<sup>1</sup>, Siyu Xia<sup>1</sup><br>
> <sup>1</sup>Southeast University <sup>2</sup>SenseTime Research

> [MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction](https://sensetime-fvg.github.io/MaskGWM)<br>
> Jingcheng Ni, Yuxin Guo, Yichen Liu, Rui Chen, Lewei Lu, Zehuan Wu<br>
> SenseTime Research

## 设置和运行

请参考 [README](README.md#setup) 中的步骤。
