#!/usr/bin/env python3
"""
setup_vllm.py — vllm GPU server setup, flash-attn, and model management
Requires Linux or WSL2 on Windows.  Pure GPU inference (no CPU mode).

Usage:
    python setup_vllm.py                     # interactive menu
    python setup_vllm.py --install           # install vllm + flash-attn in venv
    python setup_vllm.py --install --no-flash-attn
    python setup_vllm.py --download          # download HF model (interactive)
    python setup_vllm.py --server            # start vllm OpenAI-compatible server
    python setup_vllm.py --all               # install + download + server
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLATFORM = platform.system()          # "Windows", "Linux", "Darwin"
SEP      = "=" * 62
ARCH     = platform.machine().lower()

# Virtual environment  (Linux/macOS only — same as install.py)
USE_VENV    = PLATFORM in ("Linux", "Darwin")
VENV_DIR    = Path.home() / "gemma4-env"
VENV_PYTHON = (
    VENV_DIR / "Scripts" / "python.exe" if PLATFORM == "Windows"
    else VENV_DIR / "bin" / "python"
)

# Models directory for downloaded HF checkpoints
MODELS_DIR = Path.home() / "gemma4-models"

# Hugging Face model IDs (full precision — for vllm)
HF_MODELS = [
    ("gemma-4-E2B-it   (Effective 2.3B MoE, ~4 GB GPU)",
     "google/gemma-4-E2B-it",    "google--gemma-4-E2B-it"),
    ("gemma-4-E4B-it   (Effective 4.5B MoE, ~8 GB GPU)",
     "google/gemma-4-E4B-it",    "google--gemma-4-E4B-it"),
    ("gemma-4-26B-A4B-it  (26B params, 4B active, ~20 GB GPU)",
     "google/gemma-4-26B-A4B-it","google--gemma-4-26B-A4B-it"),
    ("gemma-4-31B-it   (31B dense, ~64 GB GPU for BF16)",
     "google/gemma-4-31B-it",    "google--gemma-4-31B-it"),
]

# vllm server defaults
DEFAULT_PORT   = 8000
DEFAULT_DTYPE  = "bfloat16"
DEFAULT_HOST   = "0.0.0.0"

# ---------------------------------------------------------------------------
# Helpers (same style as install.py / setup_llamacpp.py)
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print(f"\n{SEP}")
    print(f" {title}")
    print(SEP)


def step(n: int, total: int, msg: str) -> None:
    print(f"\n[Step {n}/{total}] {msg}...")


def ok(msg: str) -> None:
    print(f"[OK]      {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def error(msg: str) -> None:
    print(f"[ERROR]   {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO]    {msg}")


def ask(prompt: str, default: bool = False) -> bool:
    hint = "[y/N]" if not default else "[Y/n]"
    try:
        answer = input(f"  {prompt} {hint}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return (answer in ("y", "yes")) if answer else default


def run_cmd(cmd: list, check: bool = True, capture: bool = False,
            env_extra: dict = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    kwargs = dict(check=check, env=env)
    if capture:
        kwargs.update(capture_output=True, text=True)
    return subprocess.run(cmd, **kwargs)


def setup_venv() -> Path:
    """Create ~/gemma4-env on Linux/macOS if needed. Returns Python path."""
    if not USE_VENV:
        return Path(sys.executable)
    if not VENV_PYTHON.is_file():
        info(f"Creating virtual environment at {VENV_DIR} ...")
        r = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)])
        if r.returncode != 0:
            error("Failed to create virtual environment.")
            if PLATFORM == "Linux":
                error("Try:  sudo apt install python3-venv python3-full")
            sys.exit(1)
        ok(f"Venv created: {VENV_DIR}")
    else:
        ok(f"Venv ready:   {VENV_DIR}")
    return VENV_PYTHON


def pip(*packages: str, python: Path = None, extra_index: str = None,
        no_build_isolation: bool = False, env_extra: dict = None) -> bool:
    py = str(python or (VENV_PYTHON if USE_VENV else Path(sys.executable)))
    cmd = [py, "-m", "pip", "install", "--upgrade"] + list(packages)
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(cmd, env=env).returncode == 0


def detect_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi"):
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False
        )
        if r.returncode == 0 and r.stdout.strip():
            info(f"GPU: {r.stdout.strip().splitlines()[0]}")
            return True
    if shutil.which("nvcc"):
        r = subprocess.run(["nvcc", "--version"], capture_output=True,
                           text=True, check=False)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "release" in line.lower():
                    info(f"CUDA: {line.strip()}")
                    break
            return True
    return False


def print_activate_hint() -> None:
    if not USE_VENV:
        return
    print()
    print("  -------------------------------------------------------")
    print("  Activate your environment before running scripts:")
    print(f"    source {VENV_DIR}/bin/activate")
    print(f"  Or call directly:  {VENV_PYTHON} your_script.py")
    print("  -------------------------------------------------------")


# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------

def require_linux() -> None:
    """vllm is not supported on Windows (native).  WSL2 is required."""
    if PLATFORM == "Windows":
        error("vllm is not supported on native Windows.")
        error("Please use WSL2 (Ubuntu) and run this script there.")
        error("  https://learn.microsoft.com/windows/wsl/install")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Install vllm (+ flash-attn)
# ---------------------------------------------------------------------------

def install_vllm(flash_attn: bool = True) -> None:
    banner("vllm — GPU Inference Server Installer")
    require_linux()

    step(1, 5, "Checking Python version")
    ver = sys.version_info
    if ver < (3, 9):
        error(f"Python 3.9+ required. Found {ver.major}.{ver.minor}.")
        sys.exit(1)
    ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")

    step(2, 5, "Detecting NVIDIA GPU")
    if not detect_nvidia_gpu():
        error("No NVIDIA GPU detected. vllm requires a CUDA-capable GPU.")
        error("Check that nvidia-smi is available and your driver is installed.")
        sys.exit(1)
    ok("NVIDIA GPU ready.")

    step(3, 5, "Setting up virtual environment")
    setup_venv()
    pip("pip", "setuptools", "wheel")

    step(4, 5, "Installing PyTorch + vllm")
    torch_installed = False

    info("Trying PyTorch CUDA 12.1 ...")
    if pip("torch", "torchvision", "torchaudio",
           extra_index="https://download.pytorch.org/whl/cu121"):
        torch_installed = True
        ok("PyTorch (CUDA 12.1) installed.")
    else:
        info("Trying PyTorch CUDA 11.8 ...")
        if pip("torch", "torchvision", "torchaudio",
               extra_index="https://download.pytorch.org/whl/cu118"):
            torch_installed = True
            ok("PyTorch (CUDA 11.8) installed.")

    if not torch_installed:
        warn("PyTorch installation may have failed — continuing with vllm anyway.")

    info("Installing vllm ...")
    if not pip("vllm>=0.5.3"):
        error("vllm installation failed.")
        error("Check CUDA driver version (19xx+ recommended for CUDA 12.1).")
        sys.exit(1)
    ok("vllm installed.")

    step(5, 5, "Installing extras")
    pip("huggingface_hub", "hf_transfer", "transformers", "accelerate",
        "sentencepiece", "protobuf", "openai")
    ok("Helper packages installed.")

    if flash_attn:
        info("Installing flash-attn (requires a C++ compiler; may take 5–15 min) ...")
        if pip("flash-attn", no_build_isolation=True):
            ok("flash-attn installed (faster MHA for Gemma models).")
        else:
            warn("flash-attn failed — vllm will still work without it.")
            warn("You can retry manually:  pip install flash-attn --no-build-isolation")

    # Verify vllm import
    py = str(VENV_PYTHON if USE_VENV else Path(sys.executable))
    r = subprocess.run([py, "-c",
        "import vllm; print('vllm', vllm.__version__, 'OK')"],
        capture_output=True, text=True, check=False)
    if r.returncode == 0:
        ok(r.stdout.strip())
    else:
        error("vllm import verification failed.")
        if r.stderr:
            print(r.stderr[:800])
        sys.exit(1)

    print(f"\n{SEP}")
    print(f" vllm installation complete!  FlashAttn={flash_attn}")
    print(SEP)
    print_activate_hint()
    print("\n  Next steps:")
    print("  • python setup_vllm.py --download  (download a model)")
    print("  • python setup_vllm.py --server    (start API server)")


# ---------------------------------------------------------------------------
# 2. Download HF model for vllm
# ---------------------------------------------------------------------------

def download_hf_model() -> None:
    banner("Gemma 4 — Download HuggingFace Checkpoint")

    print()
    print("  These are BF16 transformer checkpoints used by vllm.")
    print("  Requires a Hugging Face account and `gemma` access request.")
    print("  https://huggingface.co/google/gemma-4-E2B-it")
    print()
    for i, (label, model_id, _) in enumerate(HF_MODELS, 1):
        print(f"  [{i}] {label}")
        print(f"       {model_id}")
        print()
    print("  [0] Cancel")
    print()

    try:
        raw = input("  Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if raw == "0" or not raw:
        return
    if not raw.isdigit() or not (1 <= int(raw) <= len(HF_MODELS)):
        error("Invalid choice.")
        sys.exit(1)

    _, model_id, subdir = HF_MODELS[int(raw) - 1]

    # Download destination
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    default_dir = str(MODELS_DIR / subdir)
    print()
    try:
        raw_dir = input(f"  Save to directory [{default_dir}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw_dir = ""
    local_dir = Path(raw_dir) if raw_dir else Path(default_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print()
        warn("HF_TOKEN not set.  For gated models (Gemma) you need to authenticate.")
        print("  Option A: export HF_TOKEN=hf_xxx  (in your shell)")
        print("  Option B: huggingface-cli login     (interactive)")
        if ask("Run 'huggingface-cli login' now?", False):
            subprocess.run(["huggingface-cli", "login"], check=False)

    info(f"Downloading {model_id} → {local_dir}")
    print("  (This is a large model; please be patient.)\n")

    # Enable fast transfer
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    py = str(VENV_PYTHON if USE_VENV else Path(sys.executable))
    cmd = [py, "-c",
           f"""
import os; os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='{model_id}', local_dir='{local_dir}',
    ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*'])
print('Downloaded to:', path)
"""]
    r = subprocess.run(cmd, env=env, check=False)
    if r.returncode != 0:
        error("Download failed.")
        error(f"Try manually:  huggingface-cli download {model_id} --local-dir {local_dir}")
        sys.exit(1)

    ok(f"Model saved to: {local_dir}")
    print(f"\n{SEP}")
    print(f" Download complete.  Model ID: {model_id}")
    print(SEP)
    print(f"\n  Start the server with:")
    print(f"    python setup_vllm.py --server")
    print(f"  Or pass the local path directly:")
    print(f"    python -m vllm.entrypoints.openai.api_server \\")
    print(f"        --model {local_dir} --port {DEFAULT_PORT}")


# ---------------------------------------------------------------------------
# 3. Start vllm OpenAI-compatible API server
# ---------------------------------------------------------------------------

def start_server() -> None:
    banner("vllm — Start OpenAI-Compatible API Server")
    require_linux()

    # Pick a model
    print()
    print("  Choose the model to serve:")
    print()
    for i, (label, model_id, subdir) in enumerate(HF_MODELS, 1):
        local_dir = MODELS_DIR / subdir
        status = "  (cached)" if local_dir.exists() else ""
        print(f"  [{i}] {label}{status}")
        print(f"       HF: {model_id}")
        print()
    print("  [5] Custom model path or HF model ID")
    print("  [0] Cancel")
    print()

    try:
        raw = input("  Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if raw == "0" or not raw:
        return

    if raw == "5":
        try:
            model_arg = input("  Enter local path or HF model ID: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
    elif raw.isdigit() and 1 <= int(raw) <= len(HF_MODELS):
        _, model_id, subdir = HF_MODELS[int(raw) - 1]
        local_dir = MODELS_DIR / subdir
        model_arg = str(local_dir) if local_dir.exists() else model_id
    else:
        error("Invalid choice.")
        sys.exit(1)

    # Server config
    print()
    try:
        port_raw = input(f"  Port [{DEFAULT_PORT}]: ").strip()
        port = int(port_raw) if port_raw.isdigit() else DEFAULT_PORT
    except (EOFError, KeyboardInterrupt):
        port = DEFAULT_PORT

    try:
        host_raw = input(f"  Host [{DEFAULT_HOST}]: ").strip()
        host = host_raw if host_raw else DEFAULT_HOST
    except (EOFError, KeyboardInterrupt):
        host = DEFAULT_HOST

    try:
        dtype_raw = input(f"  Data type (bfloat16/float16/auto) [{DEFAULT_DTYPE}]: ").strip()
        dtype = dtype_raw if dtype_raw else DEFAULT_DTYPE
    except (EOFError, KeyboardInterrupt):
        dtype = DEFAULT_DTYPE

    gpu_mem = 0.9
    try:
        gm_raw = input(f"  GPU memory utilisation (0.0–1.0) [0.9]: ").strip()
        gpu_mem = float(gm_raw) if gm_raw else 0.9
    except (EOFError, KeyboardInterrupt):
        gpu_mem = 0.9

    tensor_parallel = 1
    if ask("Enable tensor parallelism (multi-GPU)?", False):
        try:
            tp_raw = input("  Number of GPUs: ").strip()
            tensor_parallel = int(tp_raw) if tp_raw.isdigit() else 1
        except (EOFError, KeyboardInterrupt):
            pass

    trust_code = ask("Trust remote code (required for some models)?", True)

    # Build command
    py = str(VENV_PYTHON if USE_VENV else Path(sys.executable))
    cmd = [
        py, "-m", "vllm.entrypoints.openai.api_server",
        "--model",           model_arg,
        "--host",            host,
        "--port",            str(port),
        "--dtype",           dtype,
        "--gpu-memory-utilization", str(gpu_mem),
        "--tensor-parallel-size",   str(tensor_parallel),
    ]
    if trust_code:
        cmd.append("--trust-remote-code")

    print(f"\n{SEP}")
    print(" vllm server command:")
    print(f"\n  {' '.join(cmd)}\n")
    print(f" Endpoint: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/v1")
    print(f" OpenAI-compatible: /v1/chat/completions  /v1/models")
    print(SEP)
    print()
    print("  Press Ctrl+C to stop the server.")
    print()

    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n\n  Server stopped.")
    except subprocess.CalledProcessError as exc:
        error(f"Server exited with code {exc.returncode}.")
        sys.exit(exc.returncode)


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def interactive_menu() -> None:
    banner("Gemma 4 — vllm Setup")
    print(f"  Platform  : {PLATFORM} ({ARCH})")
    print(f"  Python    : {sys.version.split()[0]}")
    if USE_VENV:
        status = "exists" if VENV_PYTHON.is_file() else "will be created"
        print(f"  Venv      : {VENV_DIR} ({status})")
    print(f"  Models    : {MODELS_DIR}")
    print()

    if PLATFORM == "Windows":
        warn("vllm is NOT supported on native Windows.")
        warn("Use WSL2 (Ubuntu) and re-run this script inside WSL.")
        print()

    print("  [1] Install vllm + flash-attn (GPU, Linux / WSL2)")
    print("  [2] Install vllm only (no flash-attn)")
    print("  [3] Download Gemma 4 HuggingFace model")
    print("  [4] Start vllm OpenAI API server")
    print("  [5] Everything: install (with flash-attn) + download + server")
    print("  [0] Exit")
    print()
    try:
        choice = input("  Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if choice == "1":
        install_vllm(flash_attn=True)
    elif choice == "2":
        install_vllm(flash_attn=False)
    elif choice == "3":
        download_hf_model()
    elif choice == "4":
        start_server()
    elif choice == "5":
        install_vllm(flash_attn=True)
        download_hf_model()
        start_server()
    elif choice == "0":
        print("  Exiting.")
    else:
        print("  Invalid choice.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 — vllm GPU inference server setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--install",       action="store_true",
                        help="Install vllm + flash-attn + dependencies in venv")
    parser.add_argument("--no-flash-attn", action="store_true", dest="no_flash_attn",
                        help="Skip flash-attn installation")
    parser.add_argument("--download",      action="store_true",
                        help="Download a Gemma 4 HuggingFace model checkpoint")
    parser.add_argument("--server",        action="store_true",
                        help="Start the vllm OpenAI-compatible API server")
    parser.add_argument("--all",           action="store_true",
                        help="Install + download + server")
    args = parser.parse_args()

    flash_attn = not args.no_flash_attn

    if args.all:
        install_vllm(flash_attn=flash_attn)
        download_hf_model()
        start_server()
    elif args.install:
        install_vllm(flash_attn=flash_attn)
    elif args.download:
        download_hf_model()
    elif args.server:
        start_server()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
