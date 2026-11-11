#!/usr/bin/env python3
"""Download the model weights used by the StreamDiffusion demos.

The demo used to download weights from Hugging Face on every run (and failed
if the machine was offline).  This script stores them under ``models/`` so the
demo can load them from disk and run with ``--local-files-only``.

Examples
--------
# SD-Turbo + TinyVAE for demo/realtime-img2img (~2.6 GB with fp16 weights)
python scripts/download_models.py --preset realtime-img2img

# Full precision weights (~5.2 GB)
python scripts/download_models.py --preset realtime-img2img --precision fp32

# Behind a proxy / using a mirror (set before running)
HTTPS_PROXY=http://127.0.0.1:7890 python scripts/download_models.py --preset realtime-img2img
HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py --preset realtime-img2img
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    dest_name: str
    kind: str = "diffusers"  # "diffusers" | "taesd" | "generic"


PRESETS = {
    "realtime-img2img": [
        ModelSpec("stabilityai/sd-turbo", "sd-turbo", "diffusers"),
        ModelSpec("madebyollin/taesd", "taesd", "taesd"),
    ],
}

# Never download duplicate / legacy formats. The safetensors files are enough
# for diffusers and are much safer to load.
IGNORE_PATTERNS = [
    "*.bin",
    "*.ckpt",
    "*.pth",
    "*.onnx",
    "*.msgpack",
    "*.h5",
    "*.ot",
]

_DIFFUSERS_COMMON = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/config.json",
    "unet/config.json",
    "vae/config.json",
]

_DIFFUSERS_FP16 = [
    "text_encoder/*.fp16.safetensors",
    "unet/*.fp16.safetensors",
    "vae/*.fp16.safetensors",
]

_DIFFUSERS_FP32 = [
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]

_HINTS = """
Hints:
  * Check that https://huggingface.co is reachable from this machine.
  * Behind a proxy:  export HTTPS_PROXY=http://127.0.0.1:7890
  * Behind the GFW:  export HF_ENDPOINT=https://hf-mirror.com
  * Re-run the same command; already downloaded files are reused.
  * You can also copy a model directory into models/ manually and use
    --model /path/to/model with the demo.
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model weights for the StreamDiffusion demos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Download every model used by a demo (recommended).",
    )
    parser.add_argument(
        "--repo",
        nargs="+",
        metavar="REPO_ID",
        help=(
            "Extra Hugging Face repo(s) to download, e.g. "
            "stabilityai/sd-turbo. The last path component is used as the "
            "destination directory. Downloads the full repo."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help=(
            "Weight precision for diffusers models (default: fp16). "
            "fp16 is what the MPS/CUDA demos use and needs roughly half the disk space."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision / branch / commit (default: main).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token for gated/private repos (or set HF_TOKEN).",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "Hugging Face endpoint, e.g. https://hf-mirror.com. Defaults to "
            "HF_ENDPOINT when set, otherwise https://huggingface.co."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers (default: 8).",
    )
    parser.add_argument(
        "--xet",
        action="store_true",
        help=(
            "Use Hugging Face's Xet transfer backend. It can be much faster, "
            "but may stall behind some proxies; disabled by default."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when a local copy already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be downloaded.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Show the available presets and exit.",
    )
    return parser.parse_args(argv)


def build_specs(args: argparse.Namespace) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    if args.preset:
        specs.extend(PRESETS[args.preset])
    for repo_id in args.repo or []:
        repo_id = repo_id.strip().rstrip("/")
        if not repo_id:
            continue
        specs.append(ModelSpec(repo_id, repo_id.split("/")[-1], "generic"))

    # De-duplicate repos while keeping the order stable.
    seen = set()
    unique = []
    for spec in specs:
        if spec.repo_id not in seen:
            seen.add(spec.repo_id)
            unique.append(spec)
    return unique


def get_patterns(spec: ModelSpec, precision: str) -> Tuple[Optional[List[str]], List[str]]:
    if spec.kind == "diffusers":
        weights = _DIFFUSERS_FP16 if precision == "fp16" else _DIFFUSERS_FP32
        return _DIFFUSERS_COMMON + weights, IGNORE_PATTERNS
    if spec.kind == "taesd":
        return ["config.json", "*.safetensors"], IGNORE_PATTERNS
    # Unknown repos: let the Hub decide, but still skip legacy formats.
    return None, IGNORE_PATTERNS


def verify_download(spec: ModelSpec, destination: Path, precision: str) -> List[str]:
    missing: List[str] = []
    if spec.kind == "diffusers":
        for name in (
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "unet/config.json",
            "vae/config.json",
        ):
            if not (destination / name).is_file():
                missing.append(name)
        pattern = "**/*.fp16.safetensors" if precision == "fp16" else "**/*.safetensors"
        if not any(destination.glob(pattern)):
            missing.append(pattern)
    elif spec.kind == "taesd":
        if not (destination / "config.json").is_file():
            missing.append("config.json")
        if not any(destination.glob("*.safetensors")):
            missing.append("*.safetensors")
    else:
        if not any(destination.rglob("*.safetensors")):
            missing.append("*.safetensors")
    return missing


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def describe_endpoint(explicit: Optional[str]) -> str:
    if explicit:
        return explicit.rstrip("/")
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_presets:
        for name, specs in PRESETS.items():
            print(f"{name}:")
            for spec in specs:
                print(f"  - {spec.repo_id} -> models/{spec.dest_name}")
        return 0

    specs = build_specs(args)
    if not specs:
        print(
            "Nothing to download. Pass --preset realtime-img2img (or --repo REPO_ID).",
            file=sys.stderr,
        )
        return 2

    # Must be set before huggingface_hub is imported. Xet is opt-in because it
    # can hang behind proxies/transparent firewalls while the normal HTTP
    # downloader keeps working.
    if args.xet:
        os.environ.pop("HF_HUB_DISABLE_XET", None)
    else:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    output_dir: Path = args.output_dir.expanduser().resolve()
    endpoint = describe_endpoint(args.endpoint)
    if args.endpoint:
        # Must be set before huggingface_hub is imported for the first time.
        os.environ["HF_ENDPOINT"] = endpoint

    print("StreamDiffusion model downloader")
    print(f"  destination : {output_dir}")
    print(f"  precision   : {args.precision}")
    print(f"  endpoint    : {endpoint}")
    print()

    for spec in specs:
        allow_patterns, ignore_patterns = get_patterns(spec, args.precision)
        destination = output_dir / spec.dest_name
        print(f"[{spec.repo_id}] -> {destination}")
        if args.dry_run:
            if allow_patterns is None:
                print("  files: all (except legacy formats)")
            else:
                print("  files:")
                for pattern in allow_patterns:
                    print(f"    - {pattern}")
            print()
            continue

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(
                "huggingface_hub is not installed. Install the demo requirements first:\n"
                "    pip install -r demo/realtime-img2img/requirements.txt",
                file=sys.stderr,
            )
            return 1

        destination.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=spec.repo_id,
                revision=args.revision,
                local_dir=str(destination),
                token=args.token or os.environ.get("HF_TOKEN"),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=args.max_workers,
                force_download=args.force,
            )
        except KeyboardInterrupt:
            print("\nCancelled.", file=sys.stderr)
            return 130
        except Exception as exc:  # noqa: BLE001 - show a friendly message
            print(f"\n[error] Failed to download {spec.repo_id}: {exc}", file=sys.stderr)
            print(_HINTS, file=sys.stderr)
            return 1

        missing = verify_download(spec, destination, args.precision)
        size_gb = directory_size(destination) / 1_000_000_000
        if missing:
            print(
                f"  [warning] download finished but these files are missing: "
                f"{', '.join(missing)}",
                file=sys.stderr,
            )
            print(_HINTS, file=sys.stderr)
            return 1
        print(f"  done ({size_gb:.2f} GB)")
        print()

    if args.dry_run:
        print("Dry run complete; nothing was downloaded.")
        return 0

    print("All models are ready.")
    print("Run the demo with:")
    print("    cd demo/realtime-img2img")
    print("    python main.py            # auto device: cuda > mps > cpu")
    print("    python main.py --local-files-only   # fully offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
