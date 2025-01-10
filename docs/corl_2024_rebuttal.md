# CORL 2024 rebuttal

## Rebuttal

We would like to express our sincere gratitude for the time and effort you have put into reviewing our manuscript. We are grateful to you for recognizing the strengths of our paper, including:

* Interesting idea (Reviewer KA2L, 3dp1, EEWE)
* Novel idea, important problem (Reviewer KA2L, EEWE)
* Good writing and illustration, impressive experiment results (Reviewer KA2L, EEWE)

And also appreciate your suggestions and constructive criticisms which have contributed to improving the quality of our paper. We have carefully considered all your comments and made necessary modifications to address them.

In this rebuttal, we have included some new files in the newly uploaded zip file. Here, we briefly summarize the content of these files to facilitate quick reference.
* Main.pdf, which contain image-type qualitative results.
* video_demos, which is a directory contain video demos.

## General answer

### General answer to the evaluation of consistency (to R.KA2L-Q4, R.EEWE-Q1)

Thanks to the reviewer KA2L's suggestion of perceptual metric, and reviewer EEWE's suggestion from the aspect of detection consistency. The perceptual metric takes both objects and environment into account. The detection consistency focus more on the conditioned objects. Both metrics are meaningful and we will study on this as our future work.

Here we make an initial attempt at evaluating the detection consistency. We get the LiDAR detection result by a pre-trained VoxelNeXt, and camera detection result by BEVFormer, then calcuate mAP with LiDAR detection result (as GT) and camera detection result.

| Camera generation | LiDAR generation | detection consistency mAP |
| - | - | - |
| HoloDrive | HoloDrive | **TODO** |
| HoloDrive 2D branch (our reproduction of DWM) | HoloDrive 3D branch (our reproduction of single-frame Copilot4D) | **TODO** |

## To Reviewer KA2L

**Q1: The motivation for using a MaskGIT-based LiDAR generation branch, with a U-Net style architecture, is somewhat unclear. Are there any ablations (or preliminary results) showing the importance of multi-scale features? Is there any advantage over diffusion-based or autoregressive generation?**

Our decision to utilize this architecture is inspired by an existing approach, Copilot4D, which has already demonstrated its effectiveness in LiDAR generation. The established metric values from this pre-existing approach further aid us in evaluating the quality of our generated LiDAR results.

**Q2: The training procedure for the final model is complex, requiring 2 forward passes with and without condition inputs, as well as well-tuned dropouts of input components, in order to gain benefits with the proposed modules. Could the authors include a supplementary section with details on training time and whether they have any issue with training stability?**

The training procedure is designed for the inference procedure. Each inference step consumes the cross-modality features collected from the previous step. For the two passes in training, the 1st pass is used to generate features (same as the 1st inference step), and the 2nd pass uses the features output by the 1st pass. These features, after the modality transformation, are used as conditions (similar to the inference steps other than the 1st inference step).

So when the timestep <= 1/50, we calculate the loss for both passes. When the timestep > 1/50, we only calculate the loss for the 2nd pass. However, the 1st pass still updates the gradient through backpropagation.

The training strategy is designed based on the inference process, not for the sake of training stability.

**Q3: The evaluation of LiDAR generation conducted in the paper relies on deterministic metrics, unlike image generation which uses Frechet metrics between distributions. However, this seems to be common for the field, and is not a weakness exclusively of this work. An FID-style metric for LiDAR would be interesting.**

Your suggestion to implement a perceptual metric for LiDAR generation is valuable however beyond the scope of our work, and we will consider it for future work. In this study, we adhered to existing LiDAR generation metrics primarily for the comparison with exist LiDAR generation methods.

**Q4: While this work introduces an architecture capable of a new task, the quantitative benchmarking is limited to existing settings. If possible, a new benchmark with suitable metrics for evaluating cross-modal consistency would be good as an additional contribution. For example, one could run camera-only and LiDAR-only perception models on the generated output, and evaluate the consistency of these predictions. However, I understand that this may be challenging, and could be fine to mention as a direction for future work.**

Thanks to the suggestion of perceptual metric, we acknowledge the importance of metrics to evaluate consistency and take a lot of inspiration from your advice. We will append this metric on our future work.

**Reply to mentioned minor issues**

Thank you for your thorough review. We will address and correct these typographical errors.

## To Reviewer 3dp1

**Q1: What is the key technical novelty of this work?**

The principal technical innovation of this work is the implementation of appropriate modules between heterogeneous generative models, which enables the joint generation of both 2D and 3D content. Furthermore, we initially discovered that joint generation improves the results compared to each individual modality generation.

**Q2: What is the computational cost of the training and inference stages?**

Here we make a table to show the computation cost of training

| Stage | Pre-trained parameters | GPU | Total batch size | Steps | Time |
| - | - | - | - | - | - |
| 1. image generation | SD 2.1 | 16 x V100 (32G) | 768 | 18000 | 19 hours |
| 2. LiDAR generation | - | 16 x V100 (32G) | 256 | 20000 | 25 hours |
| 3. Joint generation | BEVFormer | 64 x V100 (32G) | 64 | 43200 | 41 hours |
| 4. Temporal | - | 8 x A800 (80G) | 16 | 30000 | 48 hours |

and inference (each generated sample contains 6 camera views and 1 point cloud, by 50 steps latent denoising and token prediction, with CFG on).

| Generation task | V100 | A800 |
| - | - | - |
| Single frame joint generation | 45 sec | 25 sec |
| Multi-frame joint generation | untested | 100 sec |

Each step consumes the cross-modality features from the previous step except for the 1st step, and no extra forward pass within an inference step.

Other baseline methods did not provide specific inference time costs, and the optimization of inference speed is currently not in our scope.

**Q3: What is the maximum length of the video that the generative model can produce?**

In a single scheduler step, 8 frames (including 4 reference frames) are generated. Our model is capable of making autoregressive predictions, and we have found almost no collapse occurring during the 8 autoregressive steps with 36 frames.

**Q4: The authors should provide video demonstrations.**

For the video demonstration, we have selected some of the generated videos from the Section E of the supplementary file.

**Q5: How does the model perform on other datasets such as Padaset and Waymo?**

To verify generalization on other datasets, we provide qualitative comparison results on the Argoverse dataset because its data processing method is similar to Nuscenes and is easier to integrate into our pipeline, and the Padaset or Waymo datasets were not processed here due to time constraints. We compare the results of our HoloDrive and baseline (Conditional UltraLiDAR), the visualization results are shown in Section B of the supplementary file. The results show that 2D-3D joint generation can improve the quality of point cloud generation, especially the details of the scene.

**Q6: The authors are suggested to add a pseudo algorithm to illustrate the overall framework.**

We appreciate the reviewer's suggestion to include a pseudo algorithm to better illustrate our framework. We agree that it would aid in understanding the overall process. We plan to incorporate this in our revised manuscript. Thank you for bringing this to our attention.

**Q7: The qualitative comparison is significantly lacking.**

We acknowledge that this could be improved. In response to your comment, we have selected some of the generated results by both our approach and the single modality baseline approach for qualitative comparison. Please refer to the Section C and D of the supplementary file.

**Q8: Only one dataset is used.**

We chose to evaluate our method using the nuScenes dataset based on the selection of existing baseline methods, e.g. Drive-WM [9], Copilot4D [11], MagicDrive [25]. Additionally, unlike other datasets, the nuScenes dataset directly provides scene descriptions required for the T2I methods.

We are working on labelling text descriptions for other datasets and may update the result in the revised version.

## To Reviewer EEWE

**Q1: Is there a quantitative evaluation on the consistency?**

Thank you for your suggestion regarding the consistency metric. But due to time constraints, our metrics are limited to previous works. We agree with the importance of quantitative evaluation and will introduce this kind of metric on our future work.

**Q2: Please explain the "text and layout conditions" you feed as input to the model at inference time.**

The conditions for the single-frame generation:

* For the camera
  * Text: scene descriptions from the dataset.
  * Layout: image-space projected 3D box and HD map images.

* For the LiDAR
  * Layout: BEV-space projected 3D box and HD map images.

The reason behind this is to make fair comparison with results on Table.1, e.g. MagicDrive [25], Drive-WM [9]

For video generation, there are 2 cases:

* Initial video generation from the condition
  * The same conditions as single-frame generation

* Autoregressive generation on the reference frames
  * Text only for the camera
  * 4 reference frames for both camera and LiDAR

Because we do not anticipate that the video generation results strictly follow the layout condition.

**Q3: Page 5 Joint training & inference: I understand you use two forward passes to acquire the interaction features. By saying “all results are used to calculate the gradient” do you mean the same loss function is applied to both passes and final loss is sum of them?**

Yes, for the timestep <= 1/50. In this case the same loss is applied to both passes and the final loss is sum of them. For the timestep > 1/50, we just calculate the loss of the 2nd pass, however the 1st pass still receives the gradient back propagated from the 2nd pass.

This is because only the 1st inference step doesn't take cross-modality features as input (just like the 1st training pass), while all other inference steps do take the cross-modality features from the previous inference step (similar to the 2nd training pass).

**Q4: For a single inference iteration, how many times the unidirectional information exchange between the camera and lidar sub-pipelines happen?**

In our work, each inference iteration utilizes the cross-modality features from the previous inference iteration for 1 times. For the total `R` steps of denoising, we conduct `R - 1` information exchanges. Please note that the 1st step does not involve any information exchange.

**Q5: Minor issue: at page 6 line 226, is length 8 in frames or seconds?**

The term 'length' in this context refers to frames. Given that our model operates at 2 FPS, the corresponding time is 4 seconds.

**Q6: In the supplementary you gave editing examples where you replaced/removed cars with trucks, how do instruct your model to perform this task?**

The editing is implemented by controlling the 2D condition input, i.e., the projected 3D box condition.

* For replacement, we can edit the category of the projected 3D box of the selected target, for example, changing the car to a truck.
* For removal, we can remove the projected 3D box of the selected target.

Modified projected 3D boxes affect the generation of multi-view images, and thus the generation of corresponding point clouds.

**Q7: Discussion of failure cases**

We have included some examples of failure cases in the supplementary file (Section A).

**Q8: In figure 2 part b, the top half pipeline represents a denoising processing where variable _z_ is the latent, but the bottom half pipeline drawn in a similar fashion confuses me.**

We apologize if the variable naming has caused confusion. We will distinguish these processes more clearly in future iterations of the figure.

**Q9: The explanation of temporal model training is a little confusing. Perhaps explain why and how it differs from that of the single frame joint training.**

The temporal model start from pretrained single frame model with joint training, then, we add temporal blocks and only fine-tune temporal parameters for 30k steps. This is the common practice for training video generation model like Drive-WM [9], ADriver-I [39].

We introduce some differences as following:

1. Structure of temporal block: we folow the stucture used in Drive-WM [9], which reshape the input of attention layers from `(B T) (H W) C` to `(B H W) T C`.
2. Input of conditions: as we target at predicting the future, during training, we input 4 ground-truth 2D frames and 3D points as past observations for predicting. Layout conditions are also given with a dropout probability 0.3. Druing inference, we drop all layout conditions as our model cannot predictive this in autoregressive generation.

**Q10: In the qualitative results (i.e. camera lidar images), it’s better to mark the orientation of the ego and the position of the camera.**

We will certainly incorporate these markings in the revised version of our paper to enhance the visual representation and understanding of our results. Meanwhile, we apologize as we have identified an issue in the point cloud visualization, which resulted in some of the results being inconsistent with the image in terms of left-right direction. We will rectify this error in the revised version of our work.
