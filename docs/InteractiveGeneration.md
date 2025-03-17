# Interactive Generation

4x accelerated playing speed:

https://github.com/user-attachments/assets/933b84d3-496a-41bd-b6ab-3022a0137062

We've implemented the interactive generation with Carla (0.9.15). The main components:

* Server-side [Carla](https://carla.org/) maintains the simulation state.
* Server-side [simulation script](../src/dwm/utils/carla_simulation.py) configs the environment, ego car, sensors, and traffic manager.
* Server-side [streaming generation](../src/dwm/streaming.py) reads condition data from the Carla, and write generated frames to video streaming server.
* Server-side [video streaming server](https://github.com/bluenviron/mediamtx) to publish the video streaming for client video player.
* Client-side [Carla control](../src/dwm/utils/carla_control.py) to control the ego car in the simulation world with kayboard.
* Client-side [video player](https://ffmpeg.org/) to receive the generated result.

The dataflow is:

1. Carla control
2. Carla (configured by the simulation script)
3. Streaming generation
4. Video streaming server
5. Video player

## Requirement

The server requires:

1. GPU (nVidia A100 is recommended)
2. network accessibility.
3. Python in 3.9 or 3.10
4. Carla == 0.9.15
5. mediamtx

The client requires:
1. Windows or Ubuntu (The supported platforms for the Carla Python API).
2. Python in 3.9 or 3.10
3. ffmpeg

## Models

The interactive generative model is trained from scratch on autonomous driving data after the specification reduction (model size, view count, resolution) of CTSD 3.5, in order to reduce the overhead of model inference.

| Base Model | Temporal Training Style | Prediction Style | Configs | Checkpoint Download |
| :-: | :-: | :-: | :-: | :-: |
| [SD 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | [Diffusion forcing transformer](https://arxiv.org/abs/2502.06764) | [FIFO diffusion](https://arxiv.org/abs/2405.11473) | [Config](../configs/experimental/multi_datasets/ctsd_35_xs_p6_tirda_bm_nwao.json) | [Checkpoint](http://103.237.29.236:10030/ctsd_35_xs_p6_tirda_bm_nwao_60k.pth) |

## Inference

### Server-side Setup

1. Download the base model (for VAE and text encoders) and model checkpoint, then edit the [config](../configs/experimental/streaming/ctsd_35_xs_p6_tirda_bm_nwao_streaming.json#L168).

2. Launch the video streaming server following the [official guide](https://github.com/bluenviron/mediamtx?tab=readme-ov-file#installation).

3. Launch the Carla: `{CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen -quality-level=Low`

4. Configure the Carla by editing the [config template](../configs/experimental/simulation/carla_simulation_town10_nusc_3views.json) and run: `PYTHONPATH=src python src/dwm/utils/carla_simulation.py -c configs/experimental/simulation/carla_simulation_town10_nusc_3views.json --client-timeout 150`

5. Edit the generation config template (e.g. [Carla endpoint](../configs/experimental/streaming/ctsd_35_xs_p6_tirda_bm_nwao_streaming.json#L7), [video streaming options](../configs/experimental/streaming/ctsd_35_xs_p6_tirda_bm_nwao_streaming.json#L268)) and run: `PYTHONPATH=src python src/dwm/streaming.py -c configs/experimental/streaming/ctsd_35_xs_p6_tirda_bm_nwao_streaming.json -l output/ctsd_35_xs_p6_tirda_bm_nwao_streaming -s rtsp://{VIDEO_STREAMING_ENDPOINT}/live --fps 2`

### Client-side Setup

1. Launch the video player after the server-side streaming begin: `ffplay -fflags nobuffer -rtsp_transport tcp rtsp://{VIDEO_STREAMING_ENDPOINT}/live`

2. Launch the Carla control after the server-side streaming begin: `python src\dwm\utils\carla_control.py --host {CARLA_SERVER_ADDRESS} -p {CARLA_SERVER_PORT}`

## Known issues

1. Generation speed.
2. Latency due to the denoising queue.
