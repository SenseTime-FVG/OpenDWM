# Configurations

The configuration files are in the JSON format. They include settings for the models, datasets, pipelines, or any arguments for the program.

## Introduction

In our code, we mainly use JSON objects in three ways:

1. As a dictionary
2. As a function's parameter list
3. As a constructor and parameter for objects

### As a dictionary

The most common way for the config, for example:

```JSON
{
    "guidance_scale": 4,
    "inference_steps": 40,
    "preview_image_size": [
        448,
        252
    ]
}
```

The pipeline finds the corresponding value variable in the dictionary through the key, which determines the behavior at runtime.

### As a function's parameter list

The content of a JSON object is passed into a function, for example:

```JSON
{
    "num_workers": 3,
    "prefetch_factor": 3,
    "persistent_workers": true
}
```

The PyTorch data loader will accept all the arguments by

```Python
data_loader = torch.utils.data.DataLoader(
    dataset, **deserialized_json_object)
```

In this case, you can fill in the required parameters according to the reference documentation of the function (such as the [data loader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) here).

### As a constructor and parameter for objects

The JSON object declares the name of the object to be created, as well as the parameters, for example:

```JSON
{
    "_class_name": "torch.optim.AdamW",
    "lr": 6e-5,
    "betas": [
        0.9,
        0.975
    ]
}
```

The "_class_name" is in the format of `{name_space}.{class_or_function_name}`, and other key-value pairs are used as parameters for the class constructor (e.g. [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) here) or the function.

In the code, this type of object is parsed with `dwm.common.create_instance_from_config()` function.

With this design, the configuration, framework, and components are **loosely coupled**. For example, user can easily switch to a third-party optimizer "bitsandbytes.optim.Adam8bit" without editing the code. Developers can provide any component class (e.g. dataset, data transforms) without having to register to a specific framework.

## Development

### Name convention

The configs in this folder are mainly about the pipelines and consumed by the `src/dwm/train.py`. So they are named in the format of `{pipeline_name}_{model_config}_{condition_config}_{data_config}.json`.

* Pipeline name: the python script name in the `src/dwm/pipelines`.
* Model config: the most discriminative model arguments, such as `spatial`, `crossview`, `temporal` for the SD models.
* Condition config: the additional input for the model, such as `ts` for the "text description per scene", `ti` for the "text description per image", `b` for the box condition, `m` for the map condition.
* Data config: `mini` for the debug purpose. Combination of `nuscenes`, `argoverse`, `waymo`, `opendv` (or their initial letters), for the data components.
