# Env Setup

To run HoloDrive, there are several packages that are not mentioned in the readme file.
- These packages can be run simply by `pip install`
```
torch-scatter
einops
pytorch-fid
open3d
pyemd
accelerate

```
- For chamferdist, go to `external/chamferdist` and run `python setup.py install`
- For mmXX, please install the following verisons, which supports torch2.x, includes all the functions in bevformer and do not have conflicts.
```
mmcv-full==1.7.2
mmdet==2.26.0
mmengine==0.10.4
```
- For diffuser, run `pip install -U diffuser` to install the newset version.
- If you run your code in the server in Mainland, please download the required ckpt before you run the code. Otherwise, it may encounter ''Connection Failed'' error.
    - Download [fid ckpt](https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth) and put it in `~/.cache/torch/hub/checkpoints`
    - Download [SD2.1 ckpt](https://huggingface.co/stabilityai/stable-diffusion-2-1) and change the value of ''pretrained_model_name_or_path'' to your SD2.1 path in the config file. Alternatively, you can change your huggingface endpoints.


waymo-open-dataset-tf-2-12-0