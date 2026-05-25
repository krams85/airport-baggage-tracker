#!/usr/bin/env python3
"""
Airport Baggage Tracker — Setup & model downloader.

  python setup_models.py            — проверить зависимости
  python setup_models.py --install  — pip + (torchreid если нужен) + веса

Логика установки torchreid:
  - Если рядом есть osnet_x1_0_256x128.onnx → torchreid НЕ нужен (ONNX-движок работает без него)
  - Если ONNX нет → нужен torchreid для PyTorch-движка / экспорта
  - torchreid не совместим с Python 3.12+ (сломан setup.py) — в этом случае
    предлагается альтернатива или продолжение без него
"""

import sys
import os
import subprocess
from pathlib import Path

HERE    = Path(__file__).parent
INSTALL = "--install" in sys.argv
ONNX_FILE = HERE / "osnet_x1_0_256x128.onnx"


def run(cmd):
    print(f"  > {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd).returncode == 0

def pip(*args):
    return run([sys.executable, "-m", "pip", "install", *args,
                "--quiet", "--disable-pip-version-check"])

def can_import(name):
    r = subprocess.run([sys.executable, "-c", f"import {name}"],
                       capture_output=True)
    return r.returncode == 0

def python_version_tuple():
    return sys.version_info[:2]


print()
print("=" * 56)
print("  Airport Baggage Tracker — Setup")
print("=" * 56)
print(f"  Python     : {sys.version.split()[0]}")
print(f"  Folder     : {HERE}")
print(f"  TORCH_HOME : {os.environ.get('TORCH_HOME', '(not set)')}")
print()


# ── 0. Проверка PyTorch ───────────────────────────────────────────────────────
print("[0/?] Проверка PyTorch ...")

_torch_check = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True
)

if _torch_check.returncode != 0:
    _err = (_torch_check.stderr or "").strip()
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║  ОШИБКА ЗАГРУЗКИ PyTorch                            ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    if "WinError 1114" in _err or "c10.dll" in _err or "DLL" in _err:
        print("  DLL-ошибка torch. Переустановка CPU-версии...")
        run([sys.executable, "-m", "pip", "uninstall", "-y",
             "torch", "torchvision", "torchaudio"])
        ok = run([sys.executable, "-m", "pip", "install",
                  "torch", "torchvision", "torchaudio",
                  "--index-url", "https://download.pytorch.org/whl/cpu",
                  "--quiet", "--disable-pip-version-check"])
        if not ok:
            print("  ОШИБКА: не удалось установить CPU-версию torch.")
            sys.exit(1)
        _recheck = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True
        )
        if _recheck.returncode != 0:
            print("  ОШИБКА после переустановки:", _recheck.stderr.strip())
            sys.exit(1)
        print(f"  PyTorch CPU: {_recheck.stdout.strip()} — OK")
    else:
        print(f"  Ошибка: {_err}")
        sys.exit(1)
else:
    print(f"  PyTorch {_torch_check.stdout.strip()} — OK")

print()


# ── 0b. Проверка onnxruntime ──────────────────────────────────────────────────
_ort_check = subprocess.run(
    [sys.executable, "-c", "import onnxruntime; print(onnxruntime.__version__)"],
    capture_output=True, text=True
)

if _ort_check.returncode != 0:
    _ort_err = (_ort_check.stderr or "").strip()
    if _ort_err and ("DLL" in _ort_err or "WinError" in _ort_err or "_pybind_state" in _ort_err):
        print("[0b/?] onnxruntime — DLL ошибка, переустанавливаем CPU-версию...")
        run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu", "onnxruntime"])
        ok = run([sys.executable, "-m", "pip", "install", "onnxruntime",
                  "--quiet", "--disable-pip-version-check"])
        _recheck = subprocess.run(
            [sys.executable, "-c", "import onnxruntime; print(onnxruntime.__version__)"],
            capture_output=True, text=True
        )
        if _recheck.returncode == 0:
            print(f"  onnxruntime-cpu {_recheck.stdout.strip()} — OK")
        else:
            print("  ПРЕДУПРЕЖДЕНИЕ: onnxruntime не загружается. Используйте движок PyTorch.")
    else:
        print(f"[0b/?] onnxruntime не установлен — пропускаем.")
        print(f"       Для ONNX GPU: pip install onnxruntime-gpu")
        print(f"       Для ONNX CPU: pip install onnxruntime")
    print()
else:
    _ort_ver = _ort_check.stdout.strip()
    _prov_check = subprocess.run(
        [sys.executable, "-c",
         "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"],
        capture_output=True, text=True
    )
    _provs = _prov_check.stdout.strip() if _prov_check.returncode == 0 else "?"
    print(f"[0b/?] onnxruntime {_ort_ver}  |  провайдеры: {_provs} — OK")
    print()


# ── 1. Зависимости из requirements.txt ────────────────────────────────────────
if INSTALL:
    print("[1/3] requirements.txt ...")
    req = HERE / "requirements.txt"
    if req.exists():
        if not pip("-r", str(req)):
            print("  ОШИБКА установки зависимостей!")
            sys.exit(1)
    print("  OK")
    print()


# ── 2. torchreid — нужен ТОЛЬКО если нет ONNX-файла ─────────────────────────
step_torchreid = "2/3" if INSTALL else "пропуск"

if ONNX_FILE.exists():
    mb = ONNX_FILE.stat().st_size / 1024 / 1024
    print(f"[{step_torchreid}] torchreid — пропускаем (ONNX-файл уже есть: {mb:.1f} МБ)")
    print(f"  Приложение будет использовать ONNX Runtime для ReID.")
    print(f"  torchreid нужен только для движка 'pytorch' или переэкспорта ONNX.")
    print()
else:
    # ONNX файла нет — нужен torchreid
    py_ver = python_version_tuple()

    if INSTALL:
        print(f"[{step_torchreid}] torchreid ...")

        if can_import("torchreid"):
            print("  уже установлен — OK")
        else:
            # Проверяем совместимость с Python 3.12+
            if py_ver >= (3, 12):
                print(f"  ПРЕДУПРЕЖДЕНИЕ: Python {py_ver[0]}.{py_ver[1]} обнаружен.")
                print(f"  torchreid не совместим с Python 3.12+ из-за сломанного setup.py.")
                print()
                print(f"  Варианты решения:")
                print(f"  1. Скопируйте osnet_x1_0_256x128.onnx из другой установки")
                print(f"     (из папки 'Reid На реальных камерах' если она есть рядом)")
                print(f"  2. Установите Python 3.10 или 3.11 и запустите заново")
                print(f"  3. Используйте движок 'onnx_cpu' (установите: pip install onnxruntime)")
                print()
                # Пробуем найти ONNX рядом в других папках
                _candidates = list(HERE.parent.rglob("osnet_x1_0_256x128.onnx"))
                if _candidates:
                    import shutil
                    src_onnx = _candidates[0]
                    shutil.copy(str(src_onnx), str(ONNX_FILE))
                    print(f"  Найден и скопирован ONNX-файл из: {src_onnx}")
                    print(f"  Перезапустите setup — torchreid больше не понадобится!")
                    print()
                else:
                    print("  ONNX-файл не найден рядом. Приложение запустится в режиме")
                    print("  PyTorch без весов — ReID не будет работать.")
                    print()
                    # Не прерываем — позволяем запустить приложение
            else:
                print("  устанавливаем...")
                ok = run([sys.executable, "-m", "pip", "install",
                          "git+https://github.com/KaiyangZhou/deep-person-reid.git",
                          "--no-build-isolation",
                          "--quiet", "--disable-pip-version-check"])
                if not ok:
                    print("  ОШИБКА! Попробуйте вручную:")
                    print("    pip install git+https://github.com/KaiyangZhou/deep-person-reid.git --no-build-isolation")
                    sys.exit(1)
                print("  OK")
        print()
    else:
        if not can_import("torchreid"):
            print(f"  [!] torchreid не установлен и ONNX-файл не найден.")
            print(f"      Для работы ReID нужно одно из:")
            print(f"      1. Скопируйте osnet_x1_0_256x128.onnx в папку приложения")
            print(f"      2. pip install git+https://github.com/KaiyangZhou/deep-person-reid.git --no-build-isolation")
            sys.exit(1)


# ── 3. Веса OSNet (только для PyTorch-движка) ─────────────────────────────────
step = "3/3" if INSTALL else "1/1"

if ONNX_FILE.exists():
    print(f"[{step}] Модель OSNet x1.0 — ONNX ({ONNX_FILE.name}, {ONNX_FILE.stat().st_size/1024/1024:.1f} МБ) — OK")
    print()
    print("=" * 56)
    print("  Всё готово. Запуск приложения...")
    print("=" * 56)
    print()
    sys.exit(0)

# ONNX нет — нужны .pth веса для PyTorch-движка
print(f"[{step}] Веса OSNet x1.0 (.pth для PyTorch-движка)...")

try:
    import torch
    hub_dir   = Path(torch.hub.get_dir())
    osnet_pth = hub_dir / "checkpoints" / "osnet_x1_0_imagenet.pth"
except Exception as e:
    print(f"  Не удалось получить torch hub dir: {e}")
    osnet_pth = HERE / "torch_cache" / "hub" / "checkpoints" / "osnet_x1_0_imagenet.pth"

print(f"  Ожидаемый путь : {osnet_pth}")

if osnet_pth.exists():
    mb = osnet_pth.stat().st_size / 1024 / 1024
    print(f"  Уже скачаны ({mb:.1f} МБ) — OK")
    print()
    print("=" * 56)
    print("  Всё готово. Запуск приложения...")
    print("=" * 56)
    print()
    sys.exit(0)

print("  Файл не найден, скачиваем (~6 МБ)...")
print()

for mod in list(sys.modules.keys()):
    if "gdown" in mod or "tensorboard" in mod:
        del sys.modules[mod]

if not can_import("gdown"):
    print("  gdown не найден, устанавливаем...")
    pip("gdown>=4.6.0")

try:
    try:
        import tensorboard  # noqa
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules["tensorboard"] = MagicMock()
        sys.modules["torch.utils.tensorboard"] = MagicMock()

    import torchreid
    import time
    print("  torchreid.models.build_model(pretrained=True) ...")
    t0 = time.time()
    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=1000, pretrained=True
    )
    del model
    elapsed = time.time() - t0

    if osnet_pth.exists():
        mb = osnet_pth.stat().st_size / 1024 / 1024
        print(f"  Загружено за {elapsed:.0f} сек ({mb:.1f} МБ) — OK")
    else:
        torch_home = Path(os.environ.get("TORCH_HOME",
                          str(Path.home() / ".cache" / "torch")))
        found = list(torch_home.rglob("osnet_x1_0_imagenet.pth"))
        if found:
            print(f"  Файл найден по другому пути: {found[0]}")
        else:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: файл osnet_x1_0_imagenet.pth не найден.")

except Exception as e:
    import traceback
    print(f"  ОШИБКА: {e}")
    traceback.print_exc()
    print()
    print("  Используйте ONNX-движок вместо PyTorch:")
    print("    1. Скопируйте osnet_x1_0_256x128.onnx в папку приложения")
    print("    2. В config.yaml: reid_engine: onnx_cpu")
    print("    3. pip install onnxruntime")
    sys.exit(1)

print()
print("=" * 56)
print("  Всё готово. Запуск приложения...")
print("=" * 56)
print()
