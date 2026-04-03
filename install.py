#!/usr/bin/env python3
"""
install.py — Cross-platform installer for Gemma 4 dependencies
Works on Windows, Linux, and macOS.
On Linux/macOS a virtualenv is created at ~/gemma4-env to avoid
PEP 668 "externally-managed-environment" errors.

Usage:
    python3 install.py            # interactive menu
    python3 install.py --base     # transformers baseline only
    python3 install.py --llamacpp # llama-cpp-python (GPU CUDA)
    python3 install.py --vllm     # vllm (GPU server, Linux only)
    python3 install.py --all      # all of the above
"""

import subprocess
import sys
import os
import platform
import shutil
import argparse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLATFORM = platform.system()  # "Windows", "Linux", "Darwin"
SEP = "=" * 60
USE_VENV = PLATFORM in ("Linux", "Darwin")
VENV_DIR = os.path.expanduser("~/gemma4-env")
VENV_PYTHON = (
    os.path.join(VENV_DIR, "Scripts", "python.exe") if PLATFORM == "Windows"
    else os.path.join(VENV_DIR, "bin", "python")
)


def banner(title: str) -> None:
    print(f"\n{SEP}")
    print(f" {title}")
    print(SEP)


def step(n: int, total: int, msg: str) -> None:
    print(f"\n[Step {n}/{total}] {msg}...")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


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


def run(cmd: list, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    kwargs = dict(check=check)
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)


def setup_venv() -> str:
    """Create virtualenv at ~/gemma4-env if needed. Returns the python path to use."""
    if not USE_VENV:
        return sys.executable
    if not os.path.isfile(VENV_PYTHON):
        print(f"  [INFO] Creating virtual environment at {VENV_DIR} ...")
        result = subprocess.run([sys.executable, "-m", "venv", VENV_DIR])
        if result.returncode != 0:
            error("Failed to create virtual environment.")
            if PLATFORM == "Linux":
                error("Try:  sudo apt install python3-venv python3-full")
            sys.exit(1)
        ok(f"Venv created: {VENV_DIR}")
    else:
        ok(f"Venv ready:   {VENV_DIR}")
    return VENV_PYTHON


def pip(*packages: str, python: str = None, extra_index: str = None,
        no_build_isolation: bool = False, no_binary: str = None,
        env_extra: dict = None) -> bool:
    """Install pip packages. Uses venv python on Linux/macOS. Returns True on success."""
    py = python or (VENV_PYTHON if USE_VENV else sys.executable)
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


def detect_nvidia_gpu() -> bool:
    """Return True if an NVIDIA GPU is accessible."""
    if shutil.which("nvidia-smi"):
        try:
            r = run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                check=False, capture=True
            )
            if r.returncode == 0 and r.stdout.strip():
                print(f"  GPU detected: {r.stdout.strip().splitlines()[0]}")
                return True
        except Exception:
            pass
    if shutil.which("nvcc"):
        try:
            r = run(["nvcc", "--version"], check=False, capture=True)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "release" in line.lower():
                        print(f"  CUDA found: {line.strip()}")
                        break
                return True
        except Exception:
            pass
    return False


def print_activate_hint() -> None:
    if not USE_VENV:
        return
    print()
    print("  -------------------------------------------------------")
    print(f"  Activate your environment before running Gemma 4:")
    print(f"    source {VENV_DIR}/bin/activate")
    print()
    print(f"  Or call the venv Python directly:")
    print(f"    {VENV_PYTHON} your_script.py")
    print("  -------------------------------------------------------")
    print()


# ---------------------------------------------------------------------------
# Install routines
# ---------------------------------------------------------------------------

def install_base() -> None:
    """Install transformers baseline."""
    banner("Gemma 4 — Transformers Baseline Installer")

    step(1, 6, "Checking Python")
    ver = sys.version_info
    if ver < (3, 9):
        error(f"Python 3.9+ required. Found {ver.major}.{ver.minor}.")
        sys.exit(1)
    ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")

    step(2, 6, "Setting up environment")
    setup_venv()

    step(3, 6, "Upgrading pip")
    if pip("pip"):
        ok("pip upgraded.")
    else:
        warn("pip upgrade failed — continuing.")

    step(4, 6, "Installing core packages (transformers, torch, accelerate)")
    if pip("transformers", "torch", "accelerate"):
        ok("Core packages installed.")
    else:
        error("Core package installation failed.")
        sys.exit(1)

    step(5, 6, "Installing huggingface_hub + multimodal deps")
    pip("huggingface_hub")
    pip("Pillow", "librosa", "soundfile")
    ok("huggingface_hub + multimodal deps installed.")

    step(6, 6, "Flash Attention (optional, Linux + CUDA only)")
    if PLATFORM == "Linux":
        if ask("Install flash-attn? Requires CUDA + gcc, takes a while"):
            if pip("flash-attn", no_build_isolation=True):
                ok("flash-attn installed.")
            else:
                warn("flash-attn failed — this is optional, continuing.")
        else:
            print("  [SKIP] flash-attn skipped.")
    else:
        print(f"  [SKIP] flash-attn not supported on {PLATFORM}.")

    print(f"\n{SEP}")
    print(" Transformers baseline installed.")
    print(SEP)
    print("\n  Next steps:")
    print("  1. Accept Gemma 4 license at https://huggingface.co/google/gemma-4-E2B-it")
    print("  2. Run scripts/linux/download-models.sh  (or .bat on Windows) to fetch weights")
    print("  3. Run scripts/linux/start.sh  (or start.bat) to launch interactive chat")
    print_activate_hint()


def install_llamacpp() -> None:
    """Install llama-cpp-python with GPU auto-detection."""
    banner("Gemma 4 — llama-cpp-python Installer (GPU CUDA)")

    step(1, 5, "Checking Python")
    ver = sys.version_info
    if ver < (3, 9):
        error(f"Python 3.9+ required. Found {ver.major}.{ver.minor}.")
        sys.exit(1)
    ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")

    step(2, 5, "Setting up environment")
    setup_venv()
    pip("pip")
    ok("pip ready.")

    step(3, 5, "Detecting NVIDIA GPU")
    gpu_found = detect_nvidia_gpu()
    if gpu_found:
        ok("NVIDIA GPU detected — will attempt GPU build.")
    else:
        warn("No NVIDIA GPU detected — will install CPU-only llama-cpp-python.")

    step(4, 5, "Installing llama-cpp-python")
    pkg = "llama-cpp-python[server]>=0.3.0"
    installed = False

    if gpu_found:
        print("  [INFO] Trying pre-built CUDA 12.1 wheel...")
        if pip(pkg, extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu121"):
            ok("GPU (CUDA 12.1) wheel installed.")
            installed = True
        else:
            print("  [INFO] CUDA 12.1 failed. Trying CUDA 11.8...")
            if pip(pkg, extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu118"):
                ok("GPU (CUDA 11.8) wheel installed.")
                installed = True
            else:
                print("  [INFO] Pre-built wheels failed. Building from source with CUDA...")
                if pip(pkg, no_binary="llama-cpp-python",
                       env_extra={"CMAKE_ARGS": "-DGGML_CUDA=on", "FORCE_CMAKE": "1"}):
                    ok("GPU source build successful.")
                    installed = True
                else:
                    warn("GPU build failed — falling back to CPU-only.")
                    gpu_found = False

    if not installed:
        print("  [INFO] Installing CPU-only llama-cpp-python...")
        if pip(pkg):
            ok("CPU-only llama-cpp-python installed.")
        else:
            error("llama-cpp-python installation failed.")
            sys.exit(1)

    step(5, 5, "Verifying installation")
    py = VENV_PYTHON if USE_VENV else sys.executable
    r = subprocess.run(
        [py, "-c", "from llama_cpp import Llama; print('llama-cpp-python OK')"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        ok("llama-cpp-python is ready.")
    else:
        error("Verification failed — check CUDA/build tools and retry.")
        if r.stderr:
            print(r.stderr[:500])
        sys.exit(1)

    print(f"\n{SEP}")
    print(f" llama-cpp-python installation complete!  GPU={gpu_found}")
    print(SEP)
    print("\n  Next step: download GGUF files (options [5]-[15] in download-models script)")
    print("             then run start-llamacpp.sh / start-llamacpp.bat")
    print_activate_hint()


def install_vllm() -> None:
    """Install vllm (GPU only)."""
    banner("Gemma 4 — vllm Installer (GPU Serving)")
    print("  NOTE: vllm requires an NVIDIA GPU (CUDA 11.8+).")

    if PLATFORM == "Windows":
        error("vllm does not support Windows. Use WSL2 with an NVIDIA GPU instead.")
        sys.exit(1)

    step(1, 5, "Checking Python")
    ver = sys.version_info
    if ver < (3, 9):
        error(f"Python 3.9+ required. Found {ver.major}.{ver.minor}.")
        sys.exit(1)
    ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")

    step(2, 5, "Checking NVIDIA GPU")
    if not detect_nvidia_gpu():
        warn("No NVIDIA GPU detected. vllm requires a GPU for inference.")
        if not ask("Continue anyway?"):
            print("  [INFO] Installation cancelled.")
            sys.exit(0)
    else:
        ok("GPU detected.")

    step(3, 5, "Setting up environment + PyTorch CUDA")
    setup_venv()
    pip("pip")
    if pip("torch", "torchvision", "torchaudio",
           extra_index="https://download.pytorch.org/whl/cu121"):
        ok("PyTorch CUDA 12.1 installed.")
    else:
        print("  [INFO] CUDA 12.1 failed. Trying CUDA 11.8...")
        if pip("torch", "torchvision", "torchaudio",
               extra_index="https://download.pytorch.org/whl/cu118"):
            ok("PyTorch CUDA 11.8 installed.")
        else:
            error("PyTorch CUDA installation failed.")
            sys.exit(1)

    step(4, 5, "Installing vllm and dependencies")
    if not pip("vllm"):
        error("vllm installation failed. Ensure CUDA drivers are installed.")
        sys.exit(1)
    pip("openai")
    pip("huggingface_hub>=0.22.0", "transformers>=4.47.0", "accelerate")
    ok("vllm and dependencies installed.")

    step(5, 5, "Verifying installation")
    py = VENV_PYTHON if USE_VENV else sys.executable
    r = subprocess.run(
        [py, "-c", "import vllm; print(f'vllm {vllm.__version__} OK')"],
        capture_output=True, text=True
    )
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
    print(" Use llama.cpp (--llamacpp) for CPU/lower-VRAM fallback.")
    print(SEP)
    print("\n  Start the vllm server: scripts/linux/start-vllm.sh")
    print_activate_hint()


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def interactive_menu() -> None:
    banner("Gemma 4 — Dependency Installer")
    print(f"  Platform : {PLATFORM}")
    print(f"  Python   : {sys.version.split()[0]}")
    if USE_VENV:
        status = "exists" if os.path.isfile(VENV_PYTHON) else "will be created"
        print(f"  Venv     : {VENV_DIR} ({status})")
    print()
    print("  [1] Transformers baseline  (all platforms, CPU + GPU)")
    print("  [2] llama-cpp-python       (GPU CUDA primary, CPU fallback)")
    print("  [3] vllm                   (GPU only — Linux / WSL2)")
    print("  [4] All of the above")
    print("  [0] Exit")
    print()
    try:
        choice = input("  Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if choice == "1":
        install_base()
    elif choice == "2":
        install_llamacpp()
    elif choice == "3":
        install_vllm()
    elif choice == "4":
        install_base()
        install_llamacpp()
        install_vllm()
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
        description="Gemma 4 cross-platform dependency installer"
    )
    parser.add_argument("--base",     action="store_true", help="Install transformers baseline")
    parser.add_argument("--llamacpp", action="store_true", help="Install llama-cpp-python (GPU CUDA)")
    parser.add_argument("--vllm",     action="store_true", help="Install vllm (GPU server)")
    parser.add_argument("--all",      action="store_true", help="Install everything")
    args = parser.parse_args()

    if args.all:
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
