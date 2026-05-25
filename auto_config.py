#!/usr/bin/env python3
"""
Airport Baggage Tracker — автоподбор параметров config.yaml под железо.

Запуск: python auto_config.py
        python auto_config.py --apply      # записать в config.yaml
        python auto_config.py --apply --no-backup

Определяет GPU VRAM и выбирает оптимальный профиль:
  Нет GPU / <2 GB  →  nano, 320px, onnx_cpu,  reid_every_n=5, shared_yolo=true
  2–4 GB           →  nano, 416px, onnx_gpu,  reid_every_n=4, shared_yolo=true
  4–8 GB           →  nano, 640px, onnx_gpu,  reid_every_n=3, shared_yolo=false
  8+ GB            →  small,640px, onnx_gpu,  reid_every_n=2, shared_yolo=false
"""

import sys
import os
import shutil
import argparse
from pathlib import Path

HERE = Path(__file__).parent

APPLY  = "--apply" in sys.argv
BACKUP = "--no-backup" not in sys.argv

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--apply",     action="store_true")
parser.add_argument("--no-backup", action="store_true")
args, _ = parser.parse_known_args()
APPLY  = args.apply
BACKUP = not args.no_backup


# ── Определяем железо ────────────────────────────────────────────────────────

def detect_hardware():
    """Возвращает (vram_gb, gpu_name, cuda_ok, ort_cuda_ok)."""
    vram_gb  = 0.0
    gpu_name = "CPU only"
    cuda_ok  = False
    ort_cuda = False

    try:
        import torch
        if torch.cuda.is_available():
            cuda_ok  = True
            props    = torch.cuda.get_device_properties(0)
            vram_gb  = props.total_memory / 1024**3
            gpu_name = props.name
    except Exception:
        pass

    try:
        import onnxruntime as ort
        ort_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        pass

    return vram_gb, gpu_name, cuda_ok, ort_cuda


def choose_profile(vram_gb: float, cuda_ok: bool, ort_cuda: bool) -> dict:
    """Возвращает словарь с рекомендуемыми параметрами."""
    has_gpu = cuda_ok and vram_gb >= 1.5

    if not has_gpu:
        return dict(
            profile      = "CPU / слабый GPU",
            yolo_model   = "yolo11n.pt",
            infer_imgsz  = 320,
            reid_engine  = "onnx_cpu",
            reid_every_n = 5,
            shared_yolo  = True,
            infer_every_n= 3,
        )
    elif vram_gb < 4:
        return dict(
            profile      = f"Слабый GPU ({vram_gb:.1f} GB VRAM)",
            yolo_model   = "yolo11n.pt",
            infer_imgsz  = 416,
            reid_engine  = "onnx_gpu" if ort_cuda else "onnx_cpu",
            reid_every_n = 4,
            shared_yolo  = True,
            infer_every_n= 2,
        )
    elif vram_gb < 8:
        return dict(
            profile      = f"Средний GPU ({vram_gb:.1f} GB VRAM)",
            yolo_model   = "yolo11n.pt",
            infer_imgsz  = 640,
            reid_engine  = "onnx_gpu" if ort_cuda else "onnx_cpu",
            reid_every_n = 3,
            shared_yolo  = False,
            infer_every_n= 2,
        )
    else:
        return dict(
            profile      = f"Мощный GPU ({vram_gb:.1f} GB VRAM)",
            yolo_model   = "yolo11s.pt",
            infer_imgsz  = 640,
            reid_engine  = "onnx_gpu" if ort_cuda else "onnx_cpu",
            reid_every_n = 2,
            shared_yolo  = False,
            infer_every_n= 1,
        )


# ── Патчим config.yaml ───────────────────────────────────────────────────────

def patch_config(cfg_path: Path, profile: dict, backup: bool) -> None:
    """Обновляет только performance-параметры, оставляя остальные нетронутыми."""
    import re

    text = cfg_path.read_text(encoding="utf-8")

    if backup:
        bak = cfg_path.with_suffix(".yaml.bak")
        shutil.copy(cfg_path, bak)
        print(f"  Резервная копия: {bak.name}")

    replacements = {
        r"(^\s*path\s*:\s*)(\S+)":          profile["yolo_model"],
        r"(^\s*reid_engine\s*:\s*)(\S+)":   profile["reid_engine"],
        r"(^\s*infer_imgsz\s*:\s*)(\d+)":   str(profile["infer_imgsz"]),
        r"(^\s*infer_every_n\s*:\s*)(\d+)":  str(profile["infer_every_n"]),
        r"(^\s*reid_every_n\s*:\s*)(\d+)":   str(profile["reid_every_n"]),
        r"(^\s*shared_yolo\s*:\s*)(\S+)":   str(profile["shared_yolo"]).lower(),
    }

    for pattern, value in replacements.items():
        text = re.sub(pattern, lambda m, v=value: m.group(1) + v, text, flags=re.MULTILINE)

    cfg_path.write_text(text, encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────────────────

print()
print("=" * 56)
print("  Airport Baggage Tracker — Автоподбор конфига")
print("=" * 56)

print("\n  Определяем железо...")
vram_gb, gpu_name, cuda_ok, ort_cuda = detect_hardware()

print(f"  GPU:          {gpu_name}")
print(f"  VRAM:         {vram_gb:.1f} GB" if vram_gb > 0 else "  VRAM:         нет")
print(f"  CUDA (torch): {'да' if cuda_ok else 'нет'}")
print(f"  CUDA (ORT):   {'да' if ort_cuda else 'нет'}")

profile = choose_profile(vram_gb, cuda_ok, ort_cuda)

print()
print(f"  Профиль: {profile['profile']}")
print()
print(f"  {'Параметр':<22}  {'Значение'}")
print(f"  {'-'*22}  {'-'*20}")
for k, v in profile.items():
    if k == "profile":
        continue
    print(f"  {k:<22}  {v}")

cfg_path = HERE / "config.yaml"
if not cfg_path.exists():
    print(f"\n  ОШИБКА: {cfg_path} не найден.")
    sys.exit(1)

if not APPLY:
    print()
    print("  ──────────────────────────────────────────────────")
    print("  Это предпросмотр. Для применения запустите:")
    print("    python auto_config.py --apply")
    print("  Резервная копия config.yaml.bak будет создана автоматически.")
    print()
    sys.exit(0)

print()
patch_config(cfg_path, profile, BACKUP)
print("  config.yaml обновлён.")
print()
print("  Запустите start.bat для проверки.")
print()
