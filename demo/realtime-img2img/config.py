from typing import NamedTuple
import argparse
import os


class Args(NamedTuple):
    host: str
    port: int
    reload: bool
    mode: str
    max_queue_size: int
    timeout: float
    safety_checker: bool
    taesd: bool
    ssl_certfile: str
    ssl_keyfile: str
    debug: bool
    acceleration: str
    engine_dir: str
    device: str
    dtype: str
    model: str
    vae: str
    local_files_only: bool
    t_index: str
    num_inference_steps: int
    do_add_noise: bool
    smoothing: float
    similar_image_filter: bool
    similar_image_filter_threshold: float
    similar_image_filter_max_skip_frame: int

    def pretty_print(self):
        print("\n")
        for field, value in self._asdict().items():
            print(f"{field}: {value}")
        print("\n")


MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 0))
TIMEOUT = float(os.environ.get("TIMEOUT", 0))
SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None) == "True"
USE_TAESD = os.environ.get("USE_TAESD", "True") == "True"
ENGINE_DIR = os.environ.get("ENGINE_DIR", "engines")
ACCELERATION = os.environ.get("ACCELERATION", "auto")

# "auto" resolves to CUDA -> MPS -> CPU in utils/device.py.
DEVICE = os.environ.get("DEVICE", "auto")
DTYPE = os.environ.get("DTYPE", "auto")
# Optional explicit paths/ids. When unset, img2img.py uses models/<name> if it
# has been downloaded, otherwise it falls back to the Hugging Face model id.
MODEL = os.environ.get("MODEL_ID") or os.environ.get("MODEL")
VAE = os.environ.get("VAE_ID") or os.environ.get("VAE")
# Comma separated scheduler indices; None keeps the demo default [35, 45].
T_INDEX = os.environ.get("T_INDEX")
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", 50))
DO_ADD_NOISE = os.environ.get("DO_ADD_NOISE", "True") == "True"
SMOOTHING = float(os.environ.get("SMOOTHING", 0.0))
SIMILAR_IMAGE_FILTER = os.environ.get("SIMILAR_IMAGE_FILTER", "False") == "True"
SIMILAR_IMAGE_FILTER_THRESHOLD = float(
    os.environ.get("SIMILAR_IMAGE_FILTER_THRESHOLD", 0.98)
)
SIMILAR_IMAGE_FILTER_MAX_SKIP = int(
    os.environ.get("SIMILAR_IMAGE_FILTER_MAX_SKIP", 10)
)

default_host = os.getenv("HOST", "0.0.0.0")
default_port = int(os.getenv("PORT", "7860"))
default_mode = os.getenv("MODE", "default")

parser = argparse.ArgumentParser(description="Run the app")
parser.add_argument("--host", type=str, default=default_host, help="Host address")
parser.add_argument("--port", type=int, default=default_port, help="Port number")
parser.add_argument("--reload", action="store_true", help="Reload code on change")
parser.add_argument(
    "--mode", type=str, default=default_mode, help="App Inferece Mode: txt2img, img2img"
)
parser.add_argument(
    "--max-queue-size",
    dest="max_queue_size",
    type=int,
    default=MAX_QUEUE_SIZE,
    help="Max Queue Size",
)
parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Timeout")
parser.add_argument(
    "--safety-checker",
    dest="safety_checker",
    action="store_true",
    default=SAFETY_CHECKER,
    help="Safety Checker",
)
parser.add_argument(
    "--taesd",
    dest="taesd",
    action="store_true",
    help="Use Tiny Autoencoder",
)
parser.add_argument(
    "--no-taesd",
    dest="taesd",
    action="store_false",
    help="Use Tiny Autoencoder",
)
parser.add_argument(
    "--ssl-certfile",
    dest="ssl_certfile",
    type=str,
    default=None,
    help="SSL certfile",
)
parser.add_argument(
    "--ssl-keyfile",
    dest="ssl_keyfile",
    type=str,
    default=None,
    help="SSL keyfile",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Debug",
)
parser.add_argument(
    "--acceleration",
    type=str,
    default=ACCELERATION,
    choices=["auto", "none", "xformers", "sfast", "tensorrt"],
    help=(
        "Acceleration backend. 'auto' enables xformers on CUDA when installed "
        "and disables acceleration on MPS/CPU."
    ),
)
parser.add_argument(
    "--engine-dir",
    dest="engine_dir",
    type=str,
    default=ENGINE_DIR,
    help="Engine Dir",
)
parser.add_argument(
    "--device",
    type=str,
    default=DEVICE,
    help=(
        "Device: 'auto' (default), 'cuda', 'cuda:0', 'mps' or 'cpu'. "
        "An unavailable device falls back to auto-detection."
    ),
)
parser.add_argument(
    "--dtype",
    type=str,
    default=DTYPE,
    help=(
        "Inference dtype: 'auto' (default), 'float16', 'bfloat16' or 'float32'. "
        "float16 is used on CUDA/MPS, float32 on CPU."
    ),
)
parser.add_argument(
    "--model",
    dest="model",
    type=str,
    default=MODEL,
    help=(
        "Model id or local path. Defaults to models/sd-turbo when it exists, "
        "otherwise stabilityai/sd-turbo."
    ),
)
parser.add_argument(
    "--vae",
    dest="vae",
    type=str,
    default=VAE,
    help=(
        "Tiny VAE id or local path. Defaults to models/taesd when it exists, "
        "otherwise madebyollin/taesd."
    ),
)
parser.add_argument(
    "--t-index",
    dest="t_index",
    type=str,
    default=T_INDEX,
    help=(
        "Comma separated scheduler indices. '35,45' (default) is subtle and "
        "keeps the webcam image; '10,20,30,40' is stronger; '0,10,20,30' is "
        "much stronger. The LAST index controls how much the current frame is "
        "changed (lower = more change)."
    ),
)
parser.add_argument(
    "--num-inference-steps",
    dest="num_inference_steps",
    type=int,
    default=NUM_INFERENCE_STEPS,
    help="Scheduler steps used to resolve --t-index (default: 50).",
)
parser.add_argument(
    "--do-add-noise",
    dest="do_add_noise",
    action="store_true",
    default=DO_ADD_NOISE,
    help="Add fixed noise to the stream buffer (default: on).",
)
parser.add_argument(
    "--no-add-noise",
    dest="do_add_noise",
    action="store_false",
    help="Disable stream-buffer noise; often reduces clothing flicker.",
)
parser.add_argument(
    "--smoothing",
    type=float,
    default=SMOOTHING,
    help=(
        "Blend each output with the previous frame, 0..0.9 (default: 0). "
        "Try 0.1-0.2 to reduce flicker; higher values add motion ghosting."
    ),
)
parser.add_argument(
    "--similar-image-filter",
    dest="similar_image_filter",
    action="store_true",
    default=SIMILAR_IMAGE_FILTER,
    help=(
        "Skip inference for nearly identical input frames and reuse the "
        "previous output; useful for mostly static camera scenes."
    ),
)
parser.add_argument(
    "--similar-image-filter-threshold",
    dest="similar_image_filter_threshold",
    type=float,
    default=SIMILAR_IMAGE_FILTER_THRESHOLD,
    help="Cosine similarity threshold for --similar-image-filter (default: 0.98).",
)
parser.add_argument(
    "--similar-image-filter-max-skip",
    dest="similar_image_filter_max_skip_frame",
    type=int,
    default=SIMILAR_IMAGE_FILTER_MAX_SKIP,
    help="Maximum consecutive skipped frames (default: 10).",
)
parser.add_argument(
    "--local-files-only",
    dest="local_files_only",
    action="store_true",
    default=False,
    help="Never download from Hugging Face (run the download script first).",
)
parser.set_defaults(taesd=USE_TAESD)
config = Args(**vars(parser.parse_args()))
config.pretty_print()
