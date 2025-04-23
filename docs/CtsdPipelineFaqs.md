# CTSD pipeline FAQs

The CTSD is short for cross-view temporal stable diffusion. This pipeline is an extension of the pre-trained Stable Diffusion model for autonomous driving multi-view tasks and video sequence tasks.

Here are some frequently asked questions.

* [Single GPU training](#single-gpu-training)
* [Remove conditions](#remove-conditions)

## Single GPU training

The default training configurations use the [Hybrid FSDP](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html#how-to-use-devicemesh-with-hsdp) to reduce the GPU memory usage for distributed training, which does not reduce memory usage on single GPU systems.

To reduce memory on single GPU system, we should edit the config to enable quantization and CUDA AMP.

1. Set `"quantization_config"` in the `"text_encoder_load_args"` for those models implementing the [HfQuantizer](https://huggingface.co/docs/transformers/v4.48.2/en/main_classes/quantization#transformers.quantizers.HfQuantizer) (make sure the [`bitsandbytes`](https://huggingface.co/docs/bitsandbytes/main/en/index) package is installed for the following config).

```JSON
{
    "pipeline": {
        "common_config": {
            "text_encoder_load_args": {
                "variant": "fp16",
                "torch_dtype": {
                    "_class_name": "get_class",
                    "class_name": "torch.float16"
                },
                "quantization_config": {
                    "_class_name": "diffusers.quantizers.quantization_config.BitsAndBytesConfig",
                    "load_in_4bit": true,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": {
                        "_class_name": "get_class",
                        "class_name": "torch.float16"
                    }
                }
            }
        }
    }
}
```

2. Switch to quantized optimizer.

```JSON
{
    "optimizer": {
        "_class_name": "bitsandbytes.optim.Adam8bit",
        "lr": 5e-5
    }
}
```

3. Enable the CUDA AMP instead of FSDP mixed precision.

```JSON
{
    "pipeline": {
        "common_config": {
            "autocast": {
                "device_type": "cuda"
            }
        }
    }
}
```

4. Remove all `"device_mesh"` related config items.

## Remove conditions

Remove layout conditions (3D boxes, HD map):

* Model: in the case of only removing one condition, adjust the input channel number of `"pipeline.model.condition_image_adapter_config.in_channels"` to 3; If both 3dbox and hdmap are removed, the `"pipeline.model.condition_image_adapter_config"` section should be completely deleted.

* Dataset: remove `"_3dbox_image_settings"`, `"hdmap_image_settings"`

* Dataset adapter: remove `torchvision.transforms` items of `"3dbox_images"`, `"hdmap_images"`.

Dataset related settings, note that modifications to the training set and validation set must be consistent.

Remove components of text prompt:

* Refer to the [text condition processing function](../src/dwm/datasets/common.py#L316), in the dataset settings `"image_description_settings"`, add `"selected_keys": ["time", "weather", "environment"]` to exclude other text fields.
