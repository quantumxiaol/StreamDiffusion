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

To move this demo (with the downloaded models) to an NVIDIA machine, see
[`DEPLOY-CUDA.md`](../../DEPLOY-CUDA.md). Do not copy the macOS `.venv`.

### Running with Docker

```bash
docker build -t img2img .
docker run -ti -e ENGINE_DIR=/data -e HF_HOME=/data -v ~/.cache/huggingface:/data  -p 7860:7860 --gpus all img2img
```

Where `ENGINE_DIR` and `HF_HOME` set a local cache directory, making it faster to restart the docker container.
