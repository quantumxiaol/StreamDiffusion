# 在 NVIDIA 机器上部署 `demo/realtime-img2img`（以 2080 Ti 为例）

结论：**可以跑**。2080 Ti 有 11 GB 显存，512x512、fp16、2-step 的 SD-Turbo 足够。
但 **macOS 的 `.venv` 不能复制过去**（它是 arm64 的，而且里面写死了 `/Users/...`
路径），必须在目标机重新建环境。模型文件可以直接复用，不需要重新下载。

---

## 0. 目标机前提

| 项目 | 要求 |
| --- | --- |
| 系统 | x86_64 Linux（推荐 Ubuntu 22.04）或 Windows 10/11 |
| Python | 3.10（与仓库 `.python-version` 一致） |
| GPU | NVIDIA 2080 Ti（sm_75），驱动正常 |
| 显存 | 11 GB 足够；建议关闭其他占显存的程序 |
| 磁盘 | 模型 2.6 GB；pip 精简方式建议 6 GB+，uv 完整项目建议 10 GB+ |
| 内存 | 建议 16 GB 以上 |
| Node.js | 只有需要重新构建前端时才要 18+；打包时带上 `frontend/public` 就不需要 |

先在目标机确认驱动：

```bash
nvidia-smi
```

如果你准备用 `uv`（推荐），先看 `nvidia-smi` 右上角的 `CUDA Version`：
`uv.lock` 锁定的是 PyPI `torch==2.9.0`，自带 CUDA 12.8 运行库，所以需要
驱动 >= 570、`CUDA Version >= 12.8`。低于这个版本就走下面的 pip + cu121 方式。

下面的表是 **pip 方式**的 PyTorch 选择：

| 驱动版本 | 推荐安装 |
| --- | --- |
| >= 560 | `torch==2.9.0` + `torchvision==0.24.0`，CUDA 12.6 |
| 525 ~ 559 | `torch==2.5.1` + `torchvision==0.20.1`，CUDA 12.1 |
| >= 450 | `torch==2.5.1` + `torchvision==0.20.1`，CUDA 11.8 |
| 更老 | 先升级 NVIDIA 驱动，再按上面选 |

驱动太老时，PyTorch 会出现 `CUDA driver version is insufficient for CUDA runtime version`，
这不是代码问题，升级驱动即可。

---

## 1. 在 Mac 上打包（不要包含 `.venv`）

直接 `tar` 即可，不需要先 `rsync` 复制一份（那样会多占 2.6 GB）。在仓库的
**上一级目录**执行（`/Volumes/ZHITAI2T/dev`）：

```bash
cd /Volumes/ZHITAI2T/dev

tar -czf StreamDiffusion-cuda.tar.gz \
  --exclude='StreamDiffusion/.venv' \
  --exclude='StreamDiffusion/.git' \
  --exclude='*/__pycache__' \
  --exclude='StreamDiffusion/demo/realtime-img2img/frontend/node_modules' \
  --exclude='StreamDiffusion/engines' \
  --exclude='.DS_Store' \
  StreamDiffusion
```

> macOS 自带的 `tar`（bsdtar）支持上面的 `--exclude` 写法（已用同级目录结构验证）。
> `--exclude` 必须放在最后的文件列表 `StreamDiffusion` 之前。

这样会保留：

- `models/sd-turbo`、`models/taesd`（约 2.6 GB）
- `demo/realtime-img2img/frontend/public`（已构建好的前端，180 KB）
- 所有源码和 `requirements-cuda.txt`

不会带上：

- `.venv`（macOS arm64，目标机不能用）
- `.git`、`node_modules`、`__pycache__`、`engines`

打包后检查一下（下面两条命令应该分别输出“模型存在”和“没有 venv”）：

```bash
tar -tzf StreamDiffusion-cuda.tar.gz | grep 'models/sd-turbo/model_index.json'
tar -tzf StreamDiffusion-cuda.tar.gz | grep -E '^StreamDiffusion/(\.venv|\.git|.*node_modules)' | head
```

第一条应该能看到 `StreamDiffusion/models/sd-turbo/model_index.json`，第二条应该没有输出。
压缩包里的顶层目录仍然是 `StreamDiffusion/`。

---

## 2. 目标机解压并检查

```bash
tar -xzf StreamDiffusion-cuda.tar.gz
cd StreamDiffusion

nvidia-smi
python3.10 --version
ls models/sd-turbo/model_index.json models/taesd/config.json
ls demo/realtime-img2img/frontend/public/index.html
```

模型已经在本地的，后面运行时加 `--local-files-only`，就完全不会访问 Hugging Face。

> 如果你是 **git pull** 部署（而不是打包 `models/`），模型不会出现在仓库里
> （`models/sd-turbo`、`models/taesd` 已被 `.gitignore` 排除）。在目标机执行：
>
> ```bash
> python scripts/download_models.py --preset realtime-img2img
> ```
>
> 或者从 Mac 上把 `models/` 目录直接 `scp`/`rsync` 过去。
> 没有模型时 demo 会回退到 Hugging Face 在线下载，但加 `--local-files-only`
> 会直接报错。

---

## 3. 安装环境（二选一）

### 方式 A：直接用 uv（推荐，CUDA Version >= 12.8 / 驱动 >= 570）

`pyproject.toml` + `uv.lock` 已经锁好整套环境。lock 里是 PyPI 的
`torch==2.9.0`，Linux/Windows 轮子自带 CUDA 12.8 运行库，所以要求驱动
>= 570（`nvidia-smi` 右上角显示 12.8 或更高）。

```bash
# 安装 uv（目标机还没有的话）
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd StreamDiffusion

# 目标机没有 Python 3.10 时，让 uv 装一个
uv python install 3.10

# 按 uv.lock 创建 .venv，并安装全部依赖
uv sync --locked
```

验证 CUDA：

```bash
uv run python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

看到 `True` 和 2080 Ti 之后，直接跳到第 4 节（前端）和第 5 节（启动）。
启动时用 `uv run`，它会自动使用项目里的 `.venv`：

```bash
cd demo/realtime-img2img
uv run python main.py \
  --device cuda --dtype float16 --acceleration none --local-files-only
```

注意：

- 打包时必须保留 `uv.lock`。它在 `.gitignore` 里，但 tar 会带上；如果手动删掉，
  `uv sync --locked` 会报错。
- `uv sync` 安装的是完整项目（含 PyQt6、torchaudio 等），比
  `requirements-cuda.txt` 重，但版本和 lock 完全一致。
- `uv run` 会检查环境；已经 `uv sync` 过之后，也可以用
  `source .venv/bin/activate` 然后直接 `python main.py`。

### 方式 B：驱动较老，用 pip + CUDA 12.1 / 11.8

如果 `nvidia-smi` 的 CUDA Version 低于 12.8（例如 12.2、12.4），
方式 A 的 torch 2.9（CUDA 12.8 轮子）可能无法初始化。按下面的
pip 方式显式安装 cu121/cu118 的 torch 2.5.1。

#### 3B.1 创建全新的 Python 环境

Linux：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
```

Windows（PowerShell / cmd）：

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip wheel
```

> 不要 `source` 从 Mac 带过来的 `.venv`。目标机的 `.venv` 必须是新建的。

---

#### 3B.2 先安装 CUDA 版 PyTorch（关键）

**必须最先装 PyTorch**，否则后面的 `pip install` 可能把它替换成 CPU 版。

驱动 >= 560：

```bash
pip install torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

驱动 525 ~ 559：

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

驱动 >= 450 但较老：

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

装完必须验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

必须看到 `available: True` 和 `RTX 2080 Ti` 之类的输出。
如果是 `False`，不要继续往下跑，先解决驱动/PyTorch 轮子问题。

---

#### 3B.3 安装 demo 运行依赖

```bash
pip install -r requirements-cuda.txt
```

这个文件是本次在 Mac 上验证过的版本集合（diffusers 0.35.2 / transformers 4.57.3
等），已经包含 FastAPI、uvicorn、Pillow、safetensors 等 demo 需要的包。

> 不建议在新 CUDA 机器上直接 `pip install -r demo/realtime-img2img/requirements.txt`：
> 那个文件还是 2023 年的固定版本（diffusers 0.24 / transformers 4.35），
> 容易和现在的 torch 冲突。`requirements-cuda.txt` 是给部署用的。

如果还要跑仓库里的 examples、LoRA、TensorRT，再考虑安装完整项目：

```bash
pip install -e .
```

> 注意：`pyproject.toml` 要求 `torchvision>=0.24.0` / `torchaudio>=2.9.0`，
> 这会按 torch 2.9 的依赖去解析。如果你用的是 torch 2.5.1+cu121，先不要执行
> `pip install -e .`，否则可能把 CUDA 的 torch 换掉。只跑网页 demo 时完全不需要它。

可选：安装 xformers。xformers 对 torch 版本非常敏感，**不要为了装它让 pip 改动 torch**：

```bash
pip install xformers
```

如果 pip 提示要升级/降级 torch，直接中止，改用 `--acceleration none` 即可。
xformers 不是运行 demo 的必需品。

---

## 4. 前端

打包时已经带了 `demo/realtime-img2img/frontend/public`，直接启动即可。

如果你把它删了，或者改了前端源码，再构建：

```bash
cd demo/realtime-img2img/frontend
npm ci
npm run build
cd ../..
```

---

## 5. 启动 demo

```bash
cd demo/realtime-img2img

python main.py \
  --device cuda \
  --dtype float16 \
  --acceleration none \
  --local-files-only \
  --host 0.0.0.0 \
  --port 7860
```

如果你用的是方式 A（uv），把 `python main.py` 换成 `uv run python main.py`
即可；也可以在仓库根目录执行
`uv run python demo/realtime-img2img/main.py ...`。

第一次建议先用 `--acceleration none`，确认整条链路通了，再尝试加速：

```bash
# CUDA 上自动选 xformers（装了才启用）
python main.py --device cuda --acceleration auto --local-files-only

# 或显式
python main.py --device cuda --acceleration xformers --local-files-only
python main.py --device cuda:1 --acceleration none --local-files-only
```

启动成功的日志里应该能看到：

```text
Using device: cuda (dtype=float16, acceleration=none)
Model: .../models/sd-turbo
Tiny VAE: .../models/taesd
Uvicorn running on http://0.0.0.0:7860
```

浏览器打开：

```text
http://<目标机IP>:7860
```

如果是无显示器的远程服务器，在本机做 SSH 端口转发：

```bash
ssh -L 7860:127.0.0.1:7860 user@<目标机IP>
```

然后在本机浏览器打开 `http://127.0.0.1:7860`。

> 摄像头是浏览器（客户端）采集的，不是服务器采集的，所以远程访问也能用摄像头；
> 屏幕共享同理。
>
> **注意浏览器的安全限制**：`getUserMedia` / `getDisplayMedia` 只在“安全上下文”
> 可用——`localhost` / `127.0.0.1` 或 HTTPS。直接用
> `http://<目标机IP>:7860` 打开时，Chrome/Edge/Safari 可能拒绝摄像头。
> 推荐用上面的 SSH 隧道（本机就是 localhost），或者用
> `--ssl-certfile` / `--ssl-keyfile` 配 HTTPS 后访问
> `https://<目标机IP>:7860`。

---

## 6. 2080 Ti 上的性能/加速建议

| 方式 | 说明 |
| --- | --- |
| `--acceleration none` | 最稳，先跑通 |
| `--acceleration xformers` | 推荐；需要装与 torch 匹配的 xformers |
| `--acceleration tensorrt` | 可选，最复杂；需要 TensorRT/polygraphy/onnx，并在**目标机**上重新编译 engines，不要作为第一步 |

2080 Ti 是 Turing（sm_75），上面这些 CUDA 轮子和 fp16 都支持。512x512、2-step
SD-Turbo + TinyVAE 的显存占用不高，11 GB 足够；如果同时开浏览器、其他 CUDA 程序，
注意留出余量。

如果显存不够，优先：

- 关闭其他占 GPU 的程序
- 保持 `--taesd`（默认开启，TinyVAE 更省显存）
- 不要用 `--dtype float32`（更慢、更占显存，只用于排查）

---

## 7. 常见问题

**`torch.cuda.is_available()` 是 False**

说明装成了 CPU 版，或者驱动不支持。用方式 A 时确认
`nvidia-smi` 的 CUDA Version >= 12.8；用方式 B 时重新按 3B.2 的
对应 CUDA `--index-url` 安装，
并检查 `nvidia-smi`。

**报 `no kernel image is available for execution on the device`**

通常是 PyTorch/CUDA 轮子和驱动不匹配，或装到了错误平台的包。卸载重装对应 CUDA 版本。

**xformers 报错**

xformers 对 torch 版本非常敏感。直接改用 `--acceleration none`，不影响 demo 功能。

**浏览器打开是 404**

确认 `demo/realtime-img2img/frontend/public/index.html` 存在；不存在就按第 4 步构建。
当前代码已经改成按 `main.py` 所在目录找前端，所以从仓库根目录运行也可以。

**端口被占用**

换端口：`--port 7861`。

**首帧是纯色/黑图**

当前代码已修复（用第一帧自身的 latent 初始化 streaming buffer）。如果目标机上是旧代码，
更新 `src/streamdiffusion/pipeline.py` 即可。

**模型缺失或想重新下载**

```bash
python scripts/download_models.py --preset realtime-img2img
```

国内网络如果直连 Hugging Face 失败：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
# 或按你所在网络选择可用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 8. Windows 补充

- 用 `.venv\Scripts\activate` 激活环境。
- `start.sh` 是 bash 脚本，Windows 下不要用；直接执行 `python main.py ...`。
- 用 uv 时，Windows 安装命令是
  `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`，之后
  `uv sync --locked`、`uv run python main.py ...` 与 Linux 相同。
- PyTorch 安装命令与 Linux 相同（把 `--index-url` 换成对应 CUDA 版本即可）。
- 防火墙需要放行 7860 端口，或只在本机通过 `http://127.0.0.1:7860` 访问。
