#!/usr/bin/env python3
"""
setup_llamacpp.py — llama.cpp build, GGUF download, and model quantization
Cross-platform: Windows, Linux, macOS.  GPU (CUDA) default, CPU optional.

Usage:
    python setup_llamacpp.py                        # interactive menu
    python setup_llamacpp.py --compile              # clone + cmake build (GPU CUDA)
    python setup_llamacpp.py --compile --cpu        # CPU-only build (AVX2)
    python setup_llamacpp.py --compile --flash-attn # build with Flash Attention
    python setup_llamacpp.py --install-py           # install llama-cpp-python in venv
    python setup_llamacpp.py --install-py --cpu     # CPU-only llama-cpp-python
    python setup_llamacpp.py --install-py --flash-attn
    python setup_llamacpp.py --download             # download GGUF models (interactive)
    python setup_llamacpp.py --quantize             # quantize a .gguf file
    python setup_llamacpp.py --all                  # compile + install-py + download
"""

import argparse
import json
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
ARCH     = platform.machine().lower() # x86_64, arm64, etc.

# llama.cpp source directory (cloned here)
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
LLAMA_REPO    = "https://github.com/ggml-org/llama.cpp.git"
BUILD_DIR     = LLAMA_CPP_DIR / "build"

# Default models directory
MODELS_DIR = Path.home() / "gemma4-models"

# Virtual environment (Linux/macOS only; matches install.py)
USE_VENV    = PLATFORM in ("Linux", "Darwin")
VENV_DIR    = Path.home() / "gemma4-env"
VENV_PYTHON = (
    VENV_DIR / "Scripts" / "python.exe" if PLATFORM == "Windows"
    else VENV_DIR / "bin" / "python"
)

# ---------------------------------------------------------------------------
# GGUF model catalogue
# ---------------------------------------------------------------------------

# Each entry: (label, repo_id, filename, size_gb, gpu_layers, recommended)
GGUF_MODELS = [
    # E2B
    ("E2B  Q4_K_M  3.11 GB  ⭐ Recommended",
     "unsloth/gemma-4-E2B-it-GGUF",  "gemma-4-E2B-it-Q4_K_M.gguf",  3.11, 35, True),
    ("E2B  Q8_0    5.05 GB  High quality",
     "unsloth/gemma-4-E2B-it-GGUF",  "gemma-4-E2B-it-Q8_0.gguf",    5.05, 35, False),
    ("E2B  IQ4_XS  2.98 GB  Smallest",
     "unsloth/gemma-4-E2B-it-GGUF",  "gemma-4-E2B-it-IQ4_XS.gguf",  2.98, 35, False),
    # E4B
    ("E4B  Q4_K_M  4.98 GB  ⭐ Recommended",
     "unsloth/gemma-4-E4B-it-GGUF",  "gemma-4-E4B-it-Q4_K_M.gguf",  4.98, 42, True),
    ("E4B  Q8_0    8.19 GB  High quality",
     "unsloth/gemma-4-E4B-it-GGUF",  "gemma-4-E4B-it-Q8_0.gguf",    8.19, 42, False),
    # 26B-A4B MoE
    ("26B-A4B  MXFP4_MOE  16.7 GB  ⭐ MoE-optimised (unique)",
     "unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-MXFP4_MOE.gguf", 16.7, 30, True),
    ("26B-A4B  UD-Q4_K_M  16.9 GB  ⭐ Standard recommended",
     "unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 16.9, 30, True),
    ("26B-A4B  IQ4_XS     13.4 GB  Smallest 26B",
     "unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-IQ4_XS.gguf",  13.4, 30, False),
    ("26B-A4B  Q8_0       26.9 GB  Near full quality",
     "unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-Q8_0.gguf",    26.9, 30, False),
    # 31B dense
    ("31B  Q4_K_M  18.3 GB  ⭐ Most popular (~84K dl/mo)",
     "unsloth/gemma-4-31B-it-GGUF",  "gemma-4-31B-it-Q4_K_M.gguf",  18.3, 60, True),
    ("31B  IQ4_XS  16.4 GB  Smallest 31B",
     "unsloth/gemma-4-31B-it-GGUF",  "gemma-4-31B-it-IQ4_XS.gguf",  16.4, 60, False),
    ("31B  Q8_0    32.6 GB  Near full quality",
     "unsloth/gemma-4-31B-it-GGUF",  "gemma-4-31B-it-Q8_0.gguf",    32.6, 60, False),
    ("31B  BF16    61.4 GB  Full precision source",
     "unsloth/gemma-4-31B-it-GGUF",  "gemma-4-31B-it-BF16.gguf",    61.4, 60, False),
]

# Quantisation types supported by llama-quantize
QUANT_TYPES = [
    ("Q4_K_M",   "4-bit medium, good quality/size trade-off  ⭐"),
    ("Q4_K_S",   "4-bit small, slightly faster"),
    ("Q5_K_M",   "5-bit medium, better quality"),
    ("Q5_K_S",   "5-bit small"),
    ("Q6_K",     "6-bit, very close to original"),
    ("Q8_0",     "8-bit, near-lossless  ⭐"),
    ("IQ4_XS",   "4-bit importance-sampled, smaller than Q4_K_M"),
    ("IQ4_NL",   "4-bit importance-sampled (non-linear)"),
    ("IQ3_M",    "3-bit importance-sampled"),
    ("Q3_K_M",   "3-bit medium, low quality"),
    ("Q2_K",     "2-bit, very lossy"),
    ("F16",      "16-bit float (source for quantisation)"),
    ("BF16",     "BF16 (source for quantisation)"),
]

# ---------------------------------------------------------------------------
# Helpers (same style as install.py)
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
    """Create ~/gemma4-env on Linux/macOS. Returns the venv Python path."""
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
        no_build_isolation: bool = False, no_binary: str = None,
        env_extra: dict = None) -> bool:
    py = str(python or (VENV_PYTHON if USE_VENV else Path(sys.executable)))
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
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "release" in line.lower():
                    info(f"CUDA: {line.strip()}")
                    break
            return True
    return False


def get_cpu_jobs() -> int:
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except Exception:
        return 4


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
# 1. Compile llama.cpp from source
# ---------------------------------------------------------------------------

def compile_llamacpp(cpu_only: bool = False, flash_attn: bool = False) -> None:
    banner("llama.cpp — Source Build")

    step(1, 5, "Checking build tools (git, cmake)")
    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            error(f"'{tool}' not found. Please install it first.")
            if PLATFORM == "Linux":
                info(f"  sudo apt install {tool}")
            elif PLATFORM == "Darwin":
                info(f"  brew install {tool}")
            else:
                info("  Download from https://cmake.org and https://git-scm.com")
            sys.exit(1)
        r = subprocess.run([tool, "--version"], capture_output=True, text=True, check=False)
        ok(f"{tool}: {r.stdout.splitlines()[0].strip() if r.returncode == 0 else '?'}")

    step(2, 5, f"Cloning / updating llama.cpp → {LLAMA_CPP_DIR}")
    if not LLAMA_CPP_DIR.exists():
        info(f"Cloning {LLAMA_REPO} ...")
        run_cmd(["git", "clone", "--depth", "1", LLAMA_REPO, str(LLAMA_CPP_DIR)])
        ok("Cloned successfully.")
    else:
        info(f"Pulling latest changes from upstream...")
        try:
            run_cmd(["git", "-C", str(LLAMA_CPP_DIR), "pull", "--ff-only"])
            ok("Updated to latest commit.")
        except subprocess.CalledProcessError:
            warn("git pull failed — using existing source tree.")

    step(3, 5, "Detecting GPU")
    gpu_found = not cpu_only and detect_nvidia_gpu()
    if gpu_found:
        ok("NVIDIA GPU detected — building with CUDA.")
    elif cpu_only:
        info("CPU-only build requested.")
    else:
        warn("No NVIDIA GPU detected — building CPU-only.")

    step(4, 5, "CMake configure")
    cmake_args = [
        "cmake",
        "-B", str(BUILD_DIR),
        "-S", str(LLAMA_CPP_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if gpu_found:
        cmake_args += ["-DGGML_CUDA=ON"]
        info("CMake flag: -DGGML_CUDA=ON")
    if flash_attn and gpu_found:
        cmake_args += ["-DLLAMA_FLASH_ATTN=ON"]
        info("CMake flag: -DLLAMA_FLASH_ATTN=ON (Flash Attention)")
    if ARCH in ("arm64", "aarch64"):
        cmake_args += ["-DGGML_METAL=ON"] if PLATFORM == "Darwin" else []

    run_cmd(cmake_args, cwd=str(LLAMA_CPP_DIR))
    ok("CMake configured.")

    step(5, 5, f"Building (this may take several minutes, -j{get_cpu_jobs()})")
    run_cmd([
        "cmake", "--build", str(BUILD_DIR),
        "--config", "Release",
        "-j", str(get_cpu_jobs()),
    ])

    # Verify key binaries
    bins = ["llama-cli", "llama-quantize", "llama-server"]
    found = []
    missing = []
    for b in bins:
        for path in [
            BUILD_DIR / "bin" / b,
            BUILD_DIR / "bin" / f"{b}.exe",
            BUILD_DIR / "bin" / "Release" / b,
            BUILD_DIR / "bin" / "Release" / f"{b}.exe",
        ]:
            if path.is_file():
                found.append(str(path))
                break
        else:
            missing.append(b)

    for f in found:
        ok(f"Binary: {f}")
    for m in missing:
        warn(f"Binary not found: {m}  (build may have used a different name)")

    print(f"\n{SEP}")
    print(f" llama.cpp build complete!  GPU={gpu_found}  FlashAttn={flash_attn and gpu_found}")
    print(f" Source:   {LLAMA_CPP_DIR}")
    print(f" Build:    {BUILD_DIR}")
    print(SEP)
    print("\n  Next steps:")
    print("  • python setup_llamacpp.py --download   (get a GGUF model)")
    print("  • python setup_llamacpp.py --install-py (Python bindings)")


def _find_binary(name: str) -> Path | None:
    """Search BUILD_DIR subdirs for a compiled binary."""
    for subdir in ("bin", "bin/Release"):
        for ext in ("", ".exe"):
            p = BUILD_DIR / subdir / f"{name}{ext}"
            if p.is_file():
                return p
    return None


# ---------------------------------------------------------------------------
# 2. Install llama-cpp-python wheel (Python bindings)
# ---------------------------------------------------------------------------

def install_llamacpp_python(cpu_only: bool = False, flash_attn: bool = False) -> None:
    banner("llama-cpp-python — Python Bindings Installer")

    step(1, 4, "Checking Python")
    ver = sys.version_info
    if ver < (3, 9):
        error(f"Python 3.9+ required. Found {ver.major}.{ver.minor}.")
        sys.exit(1)
    ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")

    step(2, 4, "Setting up virtual environment")
    setup_venv()
    pip("pip")

    step(3, 4, "Detecting GPU & installing llama-cpp-python")
    gpu_found = not cpu_only and detect_nvidia_gpu()
    pkg = "llama-cpp-python[server]>=0.3.0"
    installed = False

    cmake_extra = []
    if gpu_found:
        cmake_extra.append("-DGGML_CUDA=on")
    if flash_attn and gpu_found:
        cmake_extra.append("-DLLAMA_FLASH_ATTN=ON")

    env_extra = {}
    if cmake_extra:
        env_extra["CMAKE_ARGS"] = " ".join(cmake_extra)
        env_extra["FORCE_CMAKE"] = "1"

    if gpu_found:
        info("Trying pre-built CUDA 12.1 wheel ...")
        if pip(pkg, extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu121"):
            ok("GPU (CUDA 12.1) wheel installed.")
            installed = True
        else:
            info("CUDA 12.1 failed — trying CUDA 11.8 wheel ...")
            if pip(pkg, extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu118"):
                ok("GPU (CUDA 11.8) wheel installed.")
                installed = True
            else:
                info("Pre-built wheels unavailable — building from source ...")
                if pip(pkg, no_binary="llama-cpp-python", env_extra=env_extra):
                    ok("Built from source (CUDA).")
                    installed = True
                else:
                    warn("GPU build failed — falling back to CPU.")
                    gpu_found = False

    if not installed:
        info("Installing CPU-only llama-cpp-python ...")
        if not pip(pkg):
            error("llama-cpp-python installation failed.")
            sys.exit(1)
        ok("CPU-only llama-cpp-python installed.")

    step(4, 4, "Verifying")
    py = str(VENV_PYTHON if USE_VENV else Path(sys.executable))
    r = subprocess.run([py, "-c",
        "from llama_cpp import Llama; print('llama-cpp-python OK')"],
        capture_output=True, text=True)
    if r.returncode == 0:
        ok("llama-cpp-python is ready.")
    else:
        error("Verification failed.")
        if r.stderr:
            print(r.stderr[:600])
        sys.exit(1)

    print(f"\n{SEP}")
    print(f" llama-cpp-python installed.  GPU={gpu_found}  FlashAttn={flash_attn and gpu_found}")
    print(SEP)
    print_activate_hint()


# ---------------------------------------------------------------------------
# 3. Download GGUF models
# ---------------------------------------------------------------------------

def download_gguf() -> None:
    banner("Gemma 4 — Download GGUF Models")

    # Ensure huggingface_hub is available
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        info("huggingface_hub not found — installing ...")
        py = str(VENV_PYTHON if USE_VENV else Path(sys.executable))
        subprocess.run([py, "-m", "pip", "install", "--upgrade",
                        "huggingface_hub", "hf_transfer"], check=False)
        import huggingface_hub  # noqa: F401

    print()
    print("  VRAM guide: E2B≤6 GB  |  E4B≤10 GB  |  26B≤20 GB  |  31B≤22 GB (Q4)")
    print()

    # Print menu
    sections = [
        ("E2B   (2.3B eff — 3–6 GB VRAM needed)",   slice(0, 3)),
        ("E4B   (4.5B eff — 5–9 GB VRAM needed)",   slice(3, 5)),
        ("26B-A4B  MoE  (17–28 GB VRAM needed)",    slice(5, 9)),
        ("31B   Dense  (18–34 GB VRAM needed)",      slice(9, 13)),
    ]
    idx = 1
    indices = {}
    for section_name, sl in sections:
        print(f"  --- {section_name} ---")
        for entry in GGUF_MODELS[sl]:
            label, repo, fname, size, _, rec = entry
            star = " ⭐" if rec else "   "
            print(f"  [{idx:>2}]{star} {label}")
            indices[idx] = entry
            idx += 1
        print()

    print("  [0] Cancel")
    print()

    try:
        raw = input("  Choose a model (or comma-separated list): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if raw == "0" or not raw:
        return

    choices = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            n = int(part)
            if n in indices:
                choices.append(indices[n])
            else:
                warn(f"Invalid choice: {n}")

    if not choices:
        return

    # Target directory
    print()
    default_dir = str(MODELS_DIR)
    try:
        raw_dir = input(f"  Save to directory [{default_dir}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw_dir = ""
    target_dir = Path(raw_dir) if raw_dir else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    print()
    # Enable hf_transfer for faster downloads if available
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from huggingface_hub import hf_hub_download
    for label, repo_id, filename, size_gb, _, _ in choices:
        print(f"\n  Downloading: {filename}  ({size_gb} GB)")
        print(f"  Repo:        {repo_id}")
        repo_subdir = target_dir / repo_id.split("/")[-1]
        repo_subdir.mkdir(parents=True, exist_ok=True)
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(repo_subdir),
                local_dir_use_symlinks=False,
            )
            ok(f"Saved to: {path}")
        except Exception as exc:
            error(f"Download failed: {exc}")
            warn("If you see 401/403, run: huggingface-cli login")

    print(f"\n{SEP}")
    print(f" Downloads complete.  Files saved under: {target_dir}")
    print(SEP)
    print("\n  Next: python setup_llamacpp.py --install-py  (if not done)")
    print("        then launch a chat with your GGUF file.")


# ---------------------------------------------------------------------------
# 4. Quantise a .gguf model
# ---------------------------------------------------------------------------

def quantize_model() -> None:
    banner("llama.cpp — Quantize Model")

    # Locate llama-quantize binary
    quantize_bin = _find_binary("llama-quantize")
    if quantize_bin is None:
        error("llama-quantize binary not found in build directory.")
        error(f"Run:  python setup_llamacpp.py --compile  first.")
        error(f"Expected under: {BUILD_DIR}")
        sys.exit(1)
    ok(f"llama-quantize: {quantize_bin}")

    # Source GGUF
    print()
    print("  Quantisation converts a high-precision model (F16/BF16) to a")
    print("  smaller format.  Input should be a BF16 or F16 .gguf file.")
    print()
    default_models = str(MODELS_DIR)
    try:
        src_raw = input(f"  Source .gguf file path: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    src = Path(src_raw.strip('"').strip("'"))
    if not src.is_file():
        error(f"Source file not found: {src}")
        sys.exit(1)

    # Quant type menu
    print()
    print("  Available quantisation types:")
    for i, (qtype, desc) in enumerate(QUANT_TYPES, 1):
        print(f"  [{i:>2}]  {qtype:<12}  {desc}")
    print()
    try:
        q_raw = input("  Choose quantisation type: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if q_raw.isdigit():
        q_idx = int(q_raw)
        if 1 <= q_idx <= len(QUANT_TYPES):
            quant_type = QUANT_TYPES[q_idx - 1][0]
        else:
            error("Invalid choice.")
            sys.exit(1)
    elif q_raw.upper() in [q[0] for q in QUANT_TYPES]:
        quant_type = q_raw.upper()
    else:
        error(f"Unknown quant type: {q_raw}")
        sys.exit(1)

    ok(f"Quantisation type: {quant_type}")

    # Output path (default: same dir, new name)
    stem = src.stem
    default_out = src.parent / f"{stem}-{quant_type}.gguf"
    try:
        out_raw = input(f"  Output file [{default_out}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        out_raw = ""
    output = Path(out_raw) if out_raw else default_out

    # Number of threads
    jobs = get_cpu_jobs()
    try:
        t_raw = input(f"  CPU threads [{jobs}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        t_raw = ""
    threads = t_raw if t_raw.isdigit() else str(jobs)

    # Imatrix optional
    imatrix = None
    if ask("Use importance matrix (imatrix) for IQ/importance-sampled quants?", False):
        try:
            im_raw = input("  Path to imatrix.dat file: ").strip()
        except (EOFError, KeyboardInterrupt):
            im_raw = ""
        if im_raw and Path(im_raw).is_file():
            imatrix = im_raw
        else:
            warn("imatrix file not found — proceeding without it.")

    # Build command
    cmd = [str(quantize_bin), "--nthread", threads]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += [str(src), str(output), quant_type]

    print(f"\n  Running: {' '.join(cmd)}\n")
    try:
        run_cmd(cmd)
        print()
        ok(f"Quantised model saved: {output}")
        # Show file sizes
        src_mb  = src.stat().st_size / 1_048_576
        out_mb  = output.stat().st_size / 1_048_576
        ratio   = src_mb / out_mb if out_mb else 0
        info(f"Source:  {src_mb:>8.1f} MB")
        info(f"Output:  {out_mb:>8.1f} MB  (compression {ratio:.2f}×)")
    except subprocess.CalledProcessError as exc:
        error(f"Quantisation failed (exit code {exc.returncode}).")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def interactive_menu() -> None:
    banner("Gemma 4 — llama.cpp Setup")
    print(f"  Platform  : {PLATFORM} ({ARCH})")
    print(f"  Python    : {sys.version.split()[0]}")
    if USE_VENV:
        status = "exists" if VENV_PYTHON.is_file() else "will be created"
        print(f"  Venv      : {VENV_DIR} ({status})")
    print(f"  llama.cpp : {LLAMA_CPP_DIR}")
    print(f"  Models    : {MODELS_DIR}")
    print()
    print("  [1] Compile llama.cpp from source  (GPU CUDA, Flash Attention)")
    print("  [2] Compile llama.cpp from source  (CPU-only build)")
    print("  [3] Install llama-cpp-python wheel  (GPU CUDA + Flash Attention)")
    print("  [4] Install llama-cpp-python wheel  (CPU-only)")
    print("  [5] Download GGUF model")
    print("  [6] Quantize a .gguf model")
    print("  [7] Everything: compile (GPU) + install-py (GPU) + download")
    print("  [0] Exit")
    print()
    try:
        choice = input("  Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if choice == "1":
        compile_llamacpp(cpu_only=False, flash_attn=True)
    elif choice == "2":
        compile_llamacpp(cpu_only=True, flash_attn=False)
    elif choice == "3":
        install_llamacpp_python(cpu_only=False, flash_attn=True)
    elif choice == "4":
        install_llamacpp_python(cpu_only=True, flash_attn=False)
    elif choice == "5":
        download_gguf()
    elif choice == "6":
        quantize_model()
    elif choice == "7":
        compile_llamacpp(cpu_only=False, flash_attn=True)
        install_llamacpp_python(cpu_only=False, flash_attn=True)
        download_gguf()
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
        description="Gemma 4 — llama.cpp build, GGUF download, and quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--compile",      action="store_true",
                        help="Clone + cmake-build llama.cpp from source")
    parser.add_argument("--install-py",   action="store_true", dest="install_py",
                        help="Install llama-cpp-python wheel in venv")
    parser.add_argument("--download",     action="store_true",
                        help="Download GGUF model (interactive list)")
    parser.add_argument("--quantize",     action="store_true",
                        help="Quantize a .gguf model using llama-quantize")
    parser.add_argument("--all",          action="store_true",
                        help="Compile + install-py + download")
    parser.add_argument("--cpu",          action="store_true",
                        help="CPU-only build (no CUDA)")
    parser.add_argument("--flash-attn",   action="store_true", dest="flash_attn",
                        help="Enable Flash Attention (requires CUDA)")
    args = parser.parse_args()

    gpu    = not args.cpu
    fa     = args.flash_attn

    if args.all:
        compile_llamacpp(cpu_only=not gpu, flash_attn=fa)
        install_llamacpp_python(cpu_only=not gpu, flash_attn=fa)
        download_gguf()
    elif args.compile:
        compile_llamacpp(cpu_only=not gpu, flash_attn=fa)
    elif args.install_py:
        install_llamacpp_python(cpu_only=not gpu, flash_attn=fa)
    elif args.download:
        download_gguf()
    elif args.quantize:
        quantize_model()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
