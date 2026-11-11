# Img2Img Example

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="../../assets/img2img1.gif" width=80%>
</p>

<p align="center">
  <img src="../../assets/img2img2.gif" width=80%>
</p>


This example, based on this [MPJEG server](https://github.com/radames/Real-Time-Latent-Consistency-Model/), runs image-to-image with a live webcam feed or screen capture on a web browser.

## Usage
You need Node.js 18+ and Python 3.10 to run this example.
Please make sure you've installed all dependencies according to the [installation instructions](../../README.md#installation).

### 1. Download the models (recommended)

Run this once from the repository root:

```bash
python scripts/download_models.py --preset realtime-img2img
```

It stores `stabilityai/sd-turbo` and `madebyollin/taesd` under `models/`
(`models/sd-turbo`, `models/taesd`, about 2.6 GB with fp16 weights). The demo
automatically uses those local directories, so later starts do not touch the
network. Add `--precision fp32` if you need full-precision weights, or
`--local-files-only` to the demo to guarantee offline operation.

If Hugging Face is unreachable, configure a proxy or mirror first, for example:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
# or
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. Build the frontend and start the server

```bash
cd frontend
npm i
npm run build
cd ..
pip install -r requirements.txt
python main.py
```

or

```bash
chmod +x start.sh
./start.sh
```

then open `http://0.0.0.0:7860` in your browser.
(*If `http://0.0.0:7860` does not work well, try `http://localhost:7860`)

### Device selection

The demo no longer hard-codes MPS or CUDA. It picks the best available backend
automatically:

| Hardware       | Device | Dtype   | Acceleration              |
| -------------- | ------ | ------- | ------------------------- |
| NVIDIA GPU     | cuda   | float16 | xformers (if installed)   |
| Apple Silicon  | mps    | float16 | none                      |
| CPU only       | cpu    | float32 | none                      |

You can override the choice explicitly:

```bash
python main.py --device cuda          # or cuda:1
python main.py --device mps
python main.py --device cpu --dtype float32
python main.py --acceleration none
python main.py --acceleration tensorrt --engine-dir engines   # NVIDIA only
```

On Apple Silicon, if an MPS operator is not implemented by PyTorch, enable the
CPU fallback for that operator:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py --device mps
```

The same settings can be provided through environment variables:
`DEVICE`, `DTYPE`, `ACCELERATION`, `MODEL_ID`, `VAE_ID`.

Use `--model /path/to/model` or `--vae /path/to/taesd` to load custom weights.
TensorRT, xformers and StableFast are CUDA-only; the demo skips them on
MPS/CPU instead of printing a traceback.

### Effect strength

The default `t_index_list=[35, 45]` is intentionally subtle: it only applies
two late, low-noise steps, so the webcam image (and therefore the original
clothing) is preserved. If the prompt seems to have no effect, this is why.

Use `--t-index` to change it. The **last** index controls how much the current
frame is changed; lower means stronger. Examples, from subtle to strong:

```bash
python main.py --t-index 35,45        # default: subtle
python main.py --t-index 10,20,30,40  # stronger, 4 steps
python main.py --t-index 0,10,20,30   # much stronger, 4 steps
```

More steps = lower FPS. On a 2080 Ti, 3 steps such as
`--t-index 10,20,30` is a reasonable compromise. If you lower
`--num-inference-steps`, keep the largest `t-index` below it.

> SD-Turbo is a photorealistic model. Strong denoising can make the result
> stylized, but for a real anime look use an anime fine-tuned SD1.5 model with
> LCM-LoRA (the `demo/realtime-txt2img` setup) instead of SD-Turbo.

### Making the effect obvious

The default prompt is the single-person "turn into a JK high school girl"
transformation (white sailor uniform, black pleated skirt). The two-person
prompts below are only optional interaction-scene examples.

A static portrait is the hardest case to judge: with the default subtle
`t_index`, the output intentionally looks almost like the input. For a demo
where the change is clearly visible, use a **moving, full-body scene** and a
stronger denoising setting:

1. Play a short video with people interacting in a window or browser tab
   (handshake, high-five, dance, passing an object, interview).
2. Start the demo, then click the **Share** button on the video panel and pick
   that window/tab (browser screen capture).
3. Use a scene-level prompt that changes both clothing and background, not
   just a clothing word. For example:
   - `two people shaking hands, wearing colorful Japanese anime school uniforms, cherry blossom background, anime illustration, vibrant colors, detailed line art, masterpiece`
   - `two dancers in neon cyberpunk kimonos, glowing city background, anime style, dynamic pose, vibrant colors, detailed line art`
   - `two friends high-fiving, wearing Japanese streetwear, anime style, summer festival background, detailed illustration`
4. Start with `--t-index 0,10,20,30`; if it flickers too much, back off to
   `--t-index 10,20,30,40`.

Tips: full body and visible hands work better than a tight face shot; a
consistent background and even lighting help; avoid fast cuts. Screen capture
requires HTTPS or `localhost`, so use the SSH tunnel if the demo is remote.

> For a true 2D anime look, SD-Turbo is still a photorealistic base model.
> The better model choice is an anime fine-tuned SD1.5 checkpoint with
> LCM-LoRA (the setup used by `demo/realtime-txt2img`).

### Reducing clothing flicker

At stronger settings the model re-interprets the outfit on every processed
frame, which looks like flickering. This demo has no temporal model, but these
switches help:

```bash
python main.py \
  --t-index 25,30,40 \
  --no-add-noise \
  --similar-image-filter --similar-image-filter-threshold 0.99 \
  --similar-image-filter-max-skip 3 \
  --smoothing 0.15
```

- `--similar-image-filter` reuses the previous output when the input frame is
  nearly identical. Lower threshold = more skipping (more stable, more lag);
  0.97~0.99 is a reasonable range; `--similar-image-filter-max-skip` prevents
  freezing forever.
- `--smoothing 0.15` blends 15% of the previous output; higher values reduce
  flicker more but add ghosting/blur (0.3 already looks dreamy).
- `--no-add-noise` removes the fixed noise from the stream buffer (the vid2vid
  example uses this).
- Make the prompt more specific, e.g. `both wearing identical white sailor
  uniforms with black pleated skirts`, instead of `colorful` / `various`.
- If the scene is static, the filter can hold the output until motion; if it
  freezes too much, raise the threshold toward 1 or increase max-skip.

For production-grade temporal consistency you need a video model
(StreamV2V / AnimateDiff / ControlNet), not per-frame SD-Turbo img2img.

### Running with Docker

```bash
docker build -t img2img .
docker run -ti -e ENGINE_DIR=/data -e HF_HOME=/data -v ~/.cache/huggingface:/data  -p 7860:7860 --gpus all img2img
```

Where `ENGINE_DIR` and `HF_HOME` set a local cache directory, making it faster to restart the docker container.
