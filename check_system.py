#!/usr/bin/env python3
"""
Airport Baggage Tracker — диагностика системы.
Запуск: python check_system.py
        check.bat
"""
import sys
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
OK   = "  [OK]  "
WARN = "  [!!]  "
FAIL = "  [XX]  "

results = []   # (level, text)  level: ok/warn/fail

def ok(msg):   results.append(("ok",   msg)); print(f"{OK}{msg}")
def warn(msg): results.append(("warn", msg)); print(f"{WARN}{msg}")
def fail(msg): results.append(("fail", msg)); print(f"{FAIL}{msg}")
def hdr(msg):  print(f"\n── {msg} {'─'*(50-len(msg))}")


print()
print("=" * 56)
print("  Airport Baggage Tracker — Диагностика системы")
print("=" * 56)


# ── Python ────────────────────────────────────────────────────────────────────
hdr("Python")
v = sys.version_info
if v >= (3, 10):
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fail(f"Python {v.major}.{v.minor} — нужен 3.10+")

if v >= (3, 12):
    warn("Python 3.12+: torchreid несовместим (используется ONNX-движок)")


# ── Основные пакеты ───────────────────────────────────────────────────────────
hdr("Пакеты")

def check_import(name, display=None, attr="__version__"):
    try:
        m = __import__(name)
        ver = getattr(m, attr, "?")
        ok(f"{display or name} {ver}")
        return m
    except Exception as e:
        fail(f"{display or name} — {e}")
        return None

check_import("torch")
check_import("cv2", "opencv-python", "__version__")
check_import("PyQt5.QtWidgets", "PyQt5")
check_import("yaml", "PyYAML", "__version__")
check_import("numpy")
check_import("ultralytics")
check_import("aiohttp")
check_import("openpyxl")


# ── CUDA / GPU ────────────────────────────────────────────────────────────────
hdr("CUDA / GPU")
try:
    import torch

    # Проверяем с каким вариантом собран torch (cpu vs cu118/cu121/...)
    torch_build = getattr(torch.version, "cuda", None)
    if torch_build is None or "cpu" in str(torch.__version__):
        fail(f"PyTorch установлен без CUDA (версия: {torch.__version__})")
        index = "cu124" if sys.version_info >= (3, 13) else "cu121"
        fail(f"  Переустановите: pip install torch torchvision "
             f"--index-url https://download.pytorch.org/whl/{index}")
    else:
        ok(f"PyTorch {torch.__version__}  (собран с CUDA {torch_build})")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / 1024**3
            ok(f"GPU {i}: {props.name}  |  VRAM: {vram:.1f} GB")
        ok(f"CUDA runtime: {torch.version.cuda}")
    else:
        fail("torch.cuda.is_available() = False")
        if torch_build:
            fail("  CUDA-сборка есть, но GPU недоступен.")
            fail("  Проверьте драйвер NVIDIA: nvidia-smi")
            fail("  Или переустановите: pip install torch torchvision "
                 "--index-url https://download.pytorch.org/whl/cu121")
        else:
            fail("  Установлен CPU-only torch — GPU работать не будет.")
except Exception as e:
    fail(f"torch.cuda: {e}")


# ── onnxruntime ───────────────────────────────────────────────────────────────
hdr("ONNX Runtime")
try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    ok(f"onnxruntime {ort.__version__}")
    if "CUDAExecutionProvider" in provs:
        ok("CUDAExecutionProvider — доступен")
    else:
        warn("CUDAExecutionProvider недоступен — ReID на CPU")
        warn("Установите: pip install onnxruntime-gpu")
    if "TensorrtExecutionProvider" in provs:
        # Проверяем что DLL реально загружается
        try:
            sess = ort.InferenceSession.__new__(ort.InferenceSession)
            ok("TensorrtExecutionProvider — доступен")
        except Exception:
            warn("TensorrtExecutionProvider: DLL зарегистрирован, но TRT не установлен (нормально)")
except ImportError:
    fail("onnxruntime не установлен")
    fail("  GPU: pip install onnxruntime-gpu")
    fail("  CPU: pip install onnxruntime")


# ── Файлы моделей ─────────────────────────────────────────────────────────────
hdr("Файлы моделей")
models = [
    HERE / "osnet_x1_0_256x128.onnx",
    HERE / "yolo11n.pt",
    HERE / "yolo11s.pt",
    HERE / "yolov8n.pt",
]
for p in models:
    if p.exists():
        mb = p.stat().st_size / 1024**2
        ok(f"{p.name}  ({mb:.1f} МБ)")
    else:
        if p.name == "osnet_x1_0_256x128.onnx":
            fail(f"{p.name} — ОТСУТСТВУЕТ (ReID не будет работать)")
        else:
            warn(f"{p.name} — отсутствует (опционально)")


# ── Qt платформа ──────────────────────────────────────────────────────────────
hdr("Qt платформа")
qt_ok = False
search = [
    Path(sys.prefix) / "Lib/site-packages/PyQt5/Qt5/plugins/platforms/qwindows.dll",
    Path(sys.prefix) / "Lib/site-packages/PyQt5/Qt/plugins/platforms/qwindows.dll",
]
plugin_env = os.environ.get("QT_PLUGIN_PATH", "")
if plugin_env:
    search.insert(0, Path(plugin_env) / "platforms/qwindows.dll")

for p in search:
    if p.exists():
        ok(f"qwindows.dll: {p.parent.parent}")
        qt_ok = True
        break
if not qt_ok:
    warn("qwindows.dll не найден в стандартных путях — установите PyQt5 через pip")


# ── config.yaml ───────────────────────────────────────────────────────────────
hdr("config.yaml")
cfg_path = HERE / "config.yaml"
if not cfg_path.exists():
    fail("config.yaml не найден")
else:
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ok("config.yaml загружен")

        cameras = cfg.get("cameras", [])
        enabled = [c for c in cameras if c.get("enabled", True)]
        ok(f"Камер в конфиге: {len(cameras)}  |  включено: {len(enabled)}")

        reid_engine = cfg.get("model", {}).get("reid_engine", "?")
        ok(f"reid_engine: {reid_engine}")

        for cam in enabled:
            name = cam.get("name", "?")
            mode = cam.get("mode", "rtsp")
            if mode == "file":
                fp = cam.get("file_path", "")
                if fp and Path(fp).exists():
                    ok(f"Камера '{name}': файл найден")
                elif fp:
                    fail(f"Камера '{name}': файл не найден → {fp}")
                else:
                    warn(f"Камера '{name}': file_path не задан")
            else:
                url = cam.get("rtsp_url", "")
                if url and not url.startswith("rtsp://admin:pass@"):
                    ok(f"Камера '{name}': RTSP настроен")
                else:
                    warn(f"Камера '{name}': RTSP URL — шаблонный или пустой")
    except Exception as e:
        fail(f"Ошибка чтения config.yaml: {e}")


# ── Итог ─────────────────────────────────────────────────────────────────────
hdr("Итог")
n_ok   = sum(1 for r in results if r[0] == "ok")
n_warn = sum(1 for r in results if r[0] == "warn")
n_fail = sum(1 for r in results if r[0] == "fail")

print()
print(f"  Пройдено:    {n_ok}")
print(f"  Внимание:    {n_warn}")
print(f"  Ошибок:      {n_fail}")
print()

if n_fail == 0 and n_warn == 0:
    print("  ✔  Всё готово к работе.")
elif n_fail == 0:
    print("  ⚠  Есть предупреждения — приложение запустится, но не всё оптимально.")
else:
    print("  ✘  Есть ошибки — устраните их перед запуском.")

print()
sys.exit(0 if n_fail == 0 else 1)
