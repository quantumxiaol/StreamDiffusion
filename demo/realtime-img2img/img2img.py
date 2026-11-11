import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math
from pathlib import Path

base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def resolve_model_path(args: Args) -> str:
    """Prefer a model downloaded by scripts/download_models.py."""

    if args.model:
        return args.model
    local_path = MODEL_DIR / "sd-turbo"
    if (local_path / "model_index.json").is_file() and any(
        local_path.glob("**/*.safetensors")
    ):
        return str(local_path)
    return base_model


def resolve_vae_path(args: Args) -> str:
    """Prefer a local TinyVAE, otherwise fall back to the Hub model id."""

    if args.vae:
        return args.vae
    local_path = MODEL_DIR / "taesd"
    if (local_path / "config.json").is_file() and any(
        local_path.glob("*.safetensors")
    ):
        return str(local_path)
    return taesd_model


def parse_t_index(value: str, num_inference_steps: int) -> list[int]:
    """Parse --t-index and validate it against the scheduler length."""

    try:
        indices = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise SystemExit(
            f"Invalid --t-index {value!r}; expected comma separated integers, "
            "e.g. '35,45' or '0,16,32,45'."
        ) from exc
    if not indices:
        raise SystemExit("--t-index must contain at least one index, e.g. '35,45'.")
    invalid = [index for index in indices if index < 0 or index >= num_inference_steps]
    if invalid:
        raise SystemExit(
            f"--t-index {invalid} is outside [0, {num_inference_steps - 1}]. "
            "Increase --num-inference-steps or use smaller indices."
        )
    return indices


default_prompt = "Anime style portrait of a single person transformed into a Japanese JK high school girl, wearing a white sailor uniform with a black pleated skirt, detailed anime illustration, clean line art, cinematic lighting, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        params = self.InputParams()
        model_id_or_path = resolve_model_path(args)
        vae_id = resolve_vae_path(args) if args.taesd else None
        t_index_list = (
            parse_t_index(args.t_index, args.num_inference_steps)
            if args.t_index
            else [35, 45]
        )
        print(f"Model: {model_id_or_path}")
        print(
            f"Denoising: t_index={t_index_list}, "
            f"num_inference_steps={args.num_inference_steps}"
        )
        print(
            f"Stability: do_add_noise={args.do_add_noise}, "
            f"smoothing={args.smoothing}, "
            f"similar_image_filter={args.similar_image_filter}"
        )
        if vae_id is not None:
            print(f"Tiny VAE: {vae_id}")

        self.stream = StreamDiffusionWrapper(
            model_id_or_path=model_id_or_path,
            use_tiny_vae=args.taesd,
            device=device,
            dtype=torch_dtype,
            t_index_list=t_index_list,
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=vae_id,
            acceleration=args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",
            use_safety_checker=args.safety_checker,
            do_add_noise=args.do_add_noise,
            enable_similar_image_filter=args.similar_image_filter,
            similar_image_filter_threshold=args.similar_image_filter_threshold,
            similar_image_filter_max_skip_frame=args.similar_image_filter_max_skip_frame,
            engine_dir=args.engine_dir,
            local_files_only=args.local_files_only,
        )

        self.smoothing = max(0.0, min(0.9, args.smoothing))
        self.prev_output: Image.Image | None = None
        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=1.2,
        )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        image_tensor = self.stream.preprocess_image(params.image)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        if self.smoothing > 0 and self.prev_output is not None:
            output_image = Image.blend(
                self.prev_output,
                output_image,
                alpha=1.0 - self.smoothing,
            )
        self.prev_output = output_image
        return output_image
