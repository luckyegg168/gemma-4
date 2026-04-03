#!/usr/bin/env python3
"""
install.py — Linux installer for Gemma 4 dependencies
Linux-specific: creates a virtualenv, checks system packages,
RAM/disk, GPU, and HuggingFace token.

Usage:
    ./install.py                  # interactive menu  (chmod +x first)
    python3 install.py            # interactive menu
    python3 install.py --base     # transformers baseline
    python3 install.py --llamacpp # llama-cpp-python (GPU CUDA)
    python3 install.py --vllm     # vllm (GPU server)
    python3 install.py --all      # everything
    python3 install.py --syscheck # system check only
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Guard: Linux only
# ---------------------------------------------------------------------------
if platform.system() != "Linux":
    print(f"[ERROR] This script is for Linux only. Detected: {platform.system()}")
    print("        Use install.py in the repo root for cross-platform installation.")
    sys.exit(1)

SEP = "=" * 60
VENV_DIR = os.path.expanduser("~/gemma4-env")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print(f"\n{SEP}\n {title}\n{SEP}")


def step(n: int, total: int, msg: str) -> None:
    print(f"\n[Step {n}/{total}] {msg}...")


def ok(msg: str) -> None:
    print(f"[OK]      {msg}")


def info(msg: str) -> None:
    print(f"[INFO]    {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def error(msg: str) -> None:
    print(f"[ERROR]   {msg}", file=sys.stderr)


def ask(prompt: str, default: bool = False) -> bool:
    hint = "[y/N]" if not default else "[Y/n]"
    try:
        answer = input(f"  {prompt} {hint}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def run_capture(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def setup_venv() -> str:
    """Create virtualenv at ~/gemma4-env if it doesn't exist. Returns venv python path."""
    if not os.path.isfile(VENV_PYTHON):
        info(f"Creating virtual environment at {VENV_DIR} ...")
        result = subprocess.run([sys.executable, "-m", "venv", VENV_DIR])
        if result.returncode != 0:
            error("Failed to create virtual environment.")
            error("Try:  sudo apt install python3-venv python3-full")
            sys.exit(1)
        ok(f"Venv created: {VENV_DIR}")
    else:
        ok(f"Venv ready:   {VENV_DIR}")
    return VENV_PYTHON


def pip(*packages: str, python: str = None, extra_index: str = None,
        no_build_isolation: bool = False, no_binary: str = None,
        env_extra: dict = None) -> bool:
    """Install pip packages into the venv. Returns True on success."""
    py = python or VENV_PYTHON
    cmd = [py, "-m", "pip", "install", "--upgrade"] + list(packages)
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    if no_binary:
        cmd += ["--no-binary", no_binary]

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def print_activate_hint() -> None:
    print()
    print("  -------------------------------------------------------")
    print(f"  Activate your environment before running Gemma 4:")
    print(f"    source {VENV_DIR}/bin/activate")
    print()
    print(f"  Or run scripts directly with the venv Python:")
    print(f"    {VENV_PYTHON} your_script.py")
    print("  -------------------------------------------------------")
    print()


# ---------------------------------------------------------------------------
# System checks
# ---------------------------------------------------------------------------

def check_python() -> bool:
    ver = sys.version_info
    if ver < (3, 9):
        error(f"Python 3.9+ required. Found {ver.major}.{ver.minor}.")
        return False
    ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")
    return True


def detect_nvidia_gpu() -> bool:
    """Return True if an NVIDIA GPU is accessible."""
    if shutil.which("nvidia-smi"):
        r = run_capture(["nvidia-smi",
                         "--query-gpu=name,memory.total,driver_version",
                         "--format=csv,noheader"])
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                print(f"  GPU: {line.strip()}")
            return True
    if shutil.which("nvcc"):
        r = run_capture(["nvcc", "--version"])
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "release" in line.lower():
                    info(f"CUDA: {line.strip()}")
                    break
            return True
    return False


def check_ram() -> None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    gb = kb / 1024 / 1024
                    status = "OK" if gb >= 8 else "WARNING (8 GB+ recommended)"
                    print(f"  RAM:  {gb:.1f} GB  [{status}]")
                    return
    except Exception:
        warn("Could not read /proc/meminfo")


def check_disk(path: str = None) -> None:
    if path is None:
        path = os.path.expanduser("~")
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / 1024 ** 3
        total_gb = stat.total / 1024 ** 3
        status = "OK" if free_gb >= 20 else "WARNING (20 GB+ recommended for models)"
        print(f"  Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total  [{status}]")
    except Exception:
        warn("Could not check disk usage.")


def check_system_packages() -> None:
    """Check for build tools needed for source builds."""
    needed = {"gcc": "build-essential", "make": "build-essential", "cmake": "cmake"}
    missing_pkgs = []
    for binary, pkg in needed.items():
        if not shutil.which(binary):
            missing_pkgs.append(pkg)

    # Check python3-dev via header file
    py_inc = run_capture(
        [sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('include'))"]
    ).stdout.strip()
    if py_inc and not os.path.isfile(os.path.join(py_inc, "Python.h")):
        missing_pkgs.append("python3-dev")

    # Check python3-venv
    r = run_capture([sys.executable, "-m", "venv", "--help"])
    if r.returncode != 0:
        missing_pkgs.append("python3-venv python3-full")

    if missing_pkgs:
        deduped = list(dict.fromkeys(missing_pkgs))
        warn(f"Missing system packages: {' '.join(deduped)}")
        info(f"Install with:  sudo apt install -y {' '.join(deduped)}")
    else:
        ok("Build tools present (gcc, make, cmake, python3-dev, python3-venv).")


def check_hf_login() -> bool:
    """Return True if huggingface-cli is logged in."""
    # Try venv cli first, then system
    venv_cli = os.path.join(VENV_DIR, "bin", "huggingface-cli")
    cli = venv_cli if os.path.isfile(venv_cli) else shutil.which("huggingface-cli")
    if cli:
        r = run_capture([cli, "whoami"])
    else:
        r = run_capture([VENV_PYTHON, "-m",
                         "huggingface_hub.commands.huggingface_cli", "whoami"])
    logged_in = r.returncode == 0 and r.stdout.strip() and "Not logged in" not in r.stdout
    if logged_in:
        ok(f"HuggingFace: logged in as {r.stdout.strip().splitlines()[0]}")
    else:
        warn("HuggingFace: not logged in.")
        info("Run option [5] to set your token.")
    return logged_in


def syscheck() -> None:
    banner("Gemma 4 — Linux System Check")

    print("\n--- Python ---")
    check_python()

    print("\n--- Hardware ---")
    check_ram()
    check_disk()
    gpu = detect_nvidia_gpu()
    if not gpu:
        warn("No NVIDIA GPU detected. CPU-only inference will be slow for large models.")
        info("Recommended: llama.cpp GGUF with E2B or E4B for CPU use.")

    print("\n--- Build tools ---")
    check_system_packages()

    print("\n--- Virtual environment ---")
    if os.path.isfile(VENV_PYTHON):
        ok(f"Venv exists: {VENV_DIR}")
    else:
        warn(f"No venv found at {VENV_DIR}  (run an install option to create it)")

    print("\n--- HuggingFace ---")
    check_hf_login()

    print("\n--- Installed packages (in venv) ---")
    if os.path.isfile(VENV_PYTHON):
        for pkg in ("torch", "transformers", "llama_cpp", "vllm", "huggingface_hub"):
            r = run_capture([VENV_PYTHON, "-c",
                             f"import {pkg}; v=getattr({pkg},'__version__','?'); print(f'{pkg} {{v}}')"])
            if r.returncode == 0:
                ok(r.stdout.strip())
            else:
                info(f"{pkg}: not installed in venv")
    else:
        info("Run an install option first to create the venv.")

    print()


# ---------------------------------------------------------------------------
# Install routines
# ---------------------------------------------------------------------------

def install_base() -> None:
    banner("Gemma 4 — Transformers Baseline (Linux)")

    step(1, 7, "Checking Python")
    if not check_python():
        sys.exit(1)

    step(2, 7, "Checking system build tools")
    check_system_packages()

    step(3, 7, "Checking hardware")
    check_ram()
    check_disk()
    gpu = detect_nvidia_gpu()
    if not gpu:
        warn("No NVIDIA GPU — PyTorch will run on CPU.")

    step(4, 7, "Creating virtual environment")
    setup_venv()

    step(5, 7, "Upgrading pip in venv")
    if pip("pip"):
        ok("pip upgraded.")
    else:
        warn("pip upgrade failed — continuing.")

    step(6, 7, "Installing core packages (transformers, torch, accelerate)")
    if pip("transformers", "torch", "accelerate"):
        ok("Core packages installed.")
    else:
        error("Core package installation failed.")
        sys.exit(1)

    pip("huggingface_hub")
    pip("Pillow", "librosa", "soundfile")
    ok("huggingface_hub + multimodal deps installed.")

    step(7, 7, "Flash Attention (optional — CUDA + gcc required)")
    if ask("Install flash-attn? Takes several minutes to compile"):
        if pip("flash-attn", no_build_isolation=True):
            ok("flash-attn installed.")
        else:
            warn("flash-attn failed — optional, skipping.")
    else:
        info("flash-attn skipped.")

    print(f"\n{SEP}")
    print(" Transformers baseline installed.")
    print(SEP)
    print("\n  Next steps:")
    print("  1. Accept Gemma 4 license: https://huggingface.co/google/gemma-4-E2B-it")
    print("  2. Run ./download-models.sh to fetch model weights")
    print("  3. Run ./start.sh to launch interactive chat")
    print_activate_hint()


def install_llamacpp() -> None:
    banner("Gemma 4 — llama-cpp-python Installer (Linux, GPU CUDA)")

    step(1, 5, "Checking Python")
    if not check_python():
        sys.exit(1)

    step(2, 5, "Creating virtual environment")
    setup_venv()
    pip("pip")
    ok("pip ready.")

    step(3, 5, "Detecting NVIDIA GPU")
    gpu_found = detect_nvidia_gpu()
    if gpu_found:
        ok("NVIDIA GPU detected — attempting GPU build.")
    else:
        warn("No NVIDIA GPU — installing CPU-only llama-cpp-python.")

    step(4, 5, "Installing llama-cpp-python")
    pkg = "llama-cpp-python[server]>=0.3.0"
    installed = False

    if gpu_found:
        info("Trying pre-built CUDA 12.1 wheel...")
        if pip(pkg, extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu121"):
            ok("GPU (CUDA 12.1) wheel installed.")
            installed = True
        else:
            info("CUDA 12.1 failed. Trying CUDA 11.8...")
            if pip(pkg, extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu118"):
                ok("GPU (CUDA 11.8) wheel installed.")
                installed = True
            else:
                info("Pre-built wheels failed. Building from source with CUDA...")
                check_system_packages()
                if pip(pkg, no_binary="llama-cpp-python",
                       env_extra={"CMAKE_ARGS": "-DGGML_CUDA=on", "FORCE_CMAKE": "1"}):
                    ok("GPU source build successful.")
                    installed = True
                else:
                    warn("GPU source build failed — falling back to CPU-only.")
                    gpu_found = False

    if not installed:
        info("Installing CPU-only llama-cpp-python...")
        if pip(pkg):
            ok("CPU-only llama-cpp-python installed.")
        else:
            error("llama-cpp-python installation failed.")
            sys.exit(1)

    step(5, 5, "Verifying installation")
    r = run_capture([VENV_PYTHON, "-c",
                     "from llama_cpp import Llama; print('llama-cpp-python OK')"])
    if r.returncode == 0:
        ok(r.stdout.strip())
    else:
        error("Verification failed — check CUDA/build tools and retry.")
        if r.stderr:
            print(r.stderr[:500])
        sys.exit(1)

    print(f"\n{SEP}")
    print(f" llama-cpp-python installed!  GPU={gpu_found}")
    print(SEP)
    print("\n  Next step: run ./download-models.sh (options [5]-[15] for GGUF)")
    print("             then ./start-llamacpp.sh")
    print_activate_hint()


def install_vllm() -> None:
    banner("Gemma 4 — vllm Installer (Linux GPU Only)")
    print("  NOTE: vllm requires an NVIDIA GPU (CUDA 11.8+).")

    step(1, 5, "Checking Python")
    if not check_python():
        sys.exit(1)

    step(2, 5, "Checking NVIDIA GPU (required for vllm)")
    if not detect_nvidia_gpu():
        warn("No NVIDIA GPU detected. vllm cannot run without a GPU.")
        if not ask("Continue anyway?"):
            info("Installation cancelled. Use --llamacpp for CPU fallback.")
            sys.exit(0)
    else:
        ok("GPU detected.")

    step(3, 5, "Creating virtual environment + PyTorch CUDA")
    setup_venv()
    pip("pip")
    if pip("torch", "torchvision", "torchaudio",
           extra_index="https://download.pytorch.org/whl/cu121"):
        ok("PyTorch CUDA 12.1 installed.")
    else:
        info("CUDA 12.1 failed. Trying CUDA 11.8...")
        if pip("torch", "torchvision", "torchaudio",
               extra_index="https://download.pytorch.org/whl/cu118"):
            ok("PyTorch CUDA 11.8 installed.")
        else:
            error("PyTorch CUDA installation failed.")
            sys.exit(1)

    step(4, 5, "Installing vllm + dependencies")
    if not pip("vllm"):
        error("vllm installation failed. Ensure CUDA drivers are installed.")
        sys.exit(1)
    pip("openai")
    pip("huggingface_hub>=0.22.0", "transformers>=4.47.0", "accelerate")
    ok("vllm and dependencies installed.")

    step(5, 5, "Verifying installation")
    r = run_capture([VENV_PYTHON, "-c",
                     "import vllm; print(f'vllm {vllm.__version__} OK')"])
    if r.returncode == 0:
        ok(r.stdout.strip())
    else:
        error("vllm verification failed. Check CUDA drivers and GPU access.")
        if r.stderr:
            print(r.stderr[:500])
        sys.exit(1)

    print(f"\n{SEP}")
    print(" vllm installation complete!")
    print(" NOTE: vllm requires GPU for inference.")
    print(" Use --llamacpp for CPU/lower-VRAM fallback.")
    print(SEP)
    print("\n  Start the server: ./start-vllm.sh")
    print_activate_hint()


def setup_hf_token() -> None:
    banner("Gemma 4 — HuggingFace Token Setup")
    print()
    print("  Gemma 4 models require accepting the license on HuggingFace.")
    print("  1. Visit:  https://huggingface.co/google/gemma-4-E2B-it")
    print("  2. Click 'Agree and access repository'")
    print("  3. Generate a token at: https://huggingface.co/settings/tokens")
    print()
    try:
        token = input("  Paste your HuggingFace token: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not token:
        warn("No token entered.")
        return

    venv_cli = os.path.join(VENV_DIR, "bin", "huggingface-cli")
    cli = venv_cli if os.path.isfile(venv_cli) else shutil.which("huggingface-cli")
    if cli:
        r = subprocess.run([cli, "login", "--token", token])
    else:
        r = subprocess.run([VENV_PYTHON, "-m",
                            "huggingface_hub.commands.huggingface_cli",
                            "login", "--token", token])
    if r.returncode == 0:
        ok("HuggingFace login successful.")
    else:
        error("Login failed — check your token and try again.")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def interactive_menu() -> None:
    banner("Gemma 4 — Linux Installer")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  Venv   : {VENV_DIR} ({'exists' if os.path.isfile(VENV_PYTHON) else 'will be created'})")
    gpu = detect_nvidia_gpu() if shutil.which("nvidia-smi") or shutil.which("nvcc") else False
    print(f"  GPU    : {'detected' if gpu else 'not detected (CPU mode)'}")
    print()
    print("  [1] System check                  (RAM, disk, GPU, packages)")
    print("  [2] Install transformers baseline  (CPU + GPU)")
    print("  [3] Install llama-cpp-python       (GPU CUDA primary, CPU fallback)")
    print("  [4] Install vllm                   (GPU only)")
    print("  [5] Set HuggingFace token")
    print("  [6] Install everything             ([2] + [3] + [4])")
    print("  [0] Exit")
    print()
    try:
        choice = input("  Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if choice == "1":
        syscheck()
    elif choice == "2":
        install_base()
    elif choice == "3":
        install_llamacpp()
    elif choice == "4":
        install_vllm()
    elif choice == "5":
        setup_hf_token()
    elif choice == "6":
        install_base()
        install_llamacpp()
        install_vllm()
    elif choice == "0":
        info("Exiting.")
    else:
        error("Invalid choice.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 Linux dependency installer"
    )
    parser.add_argument("--base",     action="store_true", help="Install transformers baseline")
    parser.add_argument("--llamacpp", action="store_true", help="Install llama-cpp-python (GPU CUDA)")
    parser.add_argument("--vllm",     action="store_true", help="Install vllm (GPU server)")
    parser.add_argument("--all",      action="store_true", help="Install everything")
    parser.add_argument("--syscheck", action="store_true", help="Run system check only")
    args = parser.parse_args()

    if args.syscheck:
        syscheck()
    elif args.all:
        install_base()
        install_llamacpp()
        install_vllm()
    elif args.base:
        install_base()
    elif args.llamacpp:
        install_llamacpp()
    elif args.vllm:
        install_vllm()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
