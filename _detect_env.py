"""Вспомогательный скрипт для install.bat — определяет окружение."""
import sys
import subprocess

# Python minor version
print(f"PY_MINOR={sys.version_info[1]}")

# PyTorch + CUDA
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"TORCH_VER={torch.__version__}")
    print(f"TORCH_CUDA={'1' if cuda_ok else '0'}")
except Exception:
    print("TORCH_VER=none")
    print("TORCH_CUDA=0")

# NVIDIA driver (major version only)
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if r.returncode == 0 and r.stdout.strip():
        major = int(r.stdout.strip().splitlines()[0].split(".")[0])
        print(f"DRIVER_MAJ={major}")
        print("HAS_GPU=1")
    else:
        print("DRIVER_MAJ=0")
        print("HAS_GPU=0")
except Exception:
    print("DRIVER_MAJ=0")
    print("HAS_GPU=0")
