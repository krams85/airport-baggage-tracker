"""
ReID Feature Extractor + In-memory Database with TTL.

Feature extractor: OSNet x1.0 (torchreid) — специализированная ReID-сеть.
Три бэкенда (выбирается через параметр engine):
  "pytorch"  — PyTorch CPU/GPU  (всегда доступен, fallback)
  "onnx_gpu" — ONNX Runtime GPU (требует onnxruntime-gpu, быстрее на NVIDIA)
  "onnx_cpu" — ONNX Runtime CPU (требует onnxruntime, без GPU)

ONNX-модель генерируется автоматически при первом запуске и сохраняется рядом
со скриптом как osnet_x1_0_256x128.onnx (~6 МБ).

Ключевые улучшения v2:
  - ONNX SessionOptions: graph optimization ORT_ENABLE_ALL, memory arena shrinkage,
    параллельные op-потоки — снижает latency и CPU-нагрузку
  - TensorRT провайдер поддерживается автоматически если доступен
  - extract_batch() используется везде вместо N × extract() — одна cuda-операция
    на кадр вместо N отдельных вызовов
  - FP16 ONNX: опционально через параметр onnx_half

Flow (production mode):
  source camera  →  extract_batch(crops)  →  db.add_or_update()
  query  camera  →  extract_batch(crops)  →  db.match()  →  MatchResult
"""

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("BaggageTracker.ReID")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ReIDEntry:
    track_id:   int
    counter_id: int             # desk # (только для source-камер, для отчётов)
    cam_name:   str
    cam_id:     int             # уникальный ID камеры — ключ маршрутизации
    embedding:  np.ndarray      # лучший эмбеддинг (наибольший вес)
    crop:       np.ndarray      # BGR, лучший кроп этого трека
    timestamp:  float = field(default_factory=time.time)
    _top_embeddings: list = field(default_factory=list, repr=False)
    _top_weights:    list = field(default_factory=list, repr=False)  # area × conf (#5)
    _mean_embedding: Optional[np.ndarray] = field(default=None, repr=False)  # (#3)
    color_hist: Optional[np.ndarray] = field(default=None, repr=False)  # HSV гистограмма кропа


@dataclass
class MatchResult:
    query_track_id:   int
    query_cam_name:   str
    query_counter_id: int
    query_crop:       np.ndarray
    source_entry:     ReIDEntry
    similarity:       float       # cosine, range [0, 1]
    timestamp:        float = field(default_factory=time.time)
    verdict_high:     float = 0.82
    verdict_mid:      float = 0.68

    @property
    def transit_seconds(self) -> float:
        return max(0.0, self.timestamp - self.source_entry.timestamp)

    @property
    def verdict(self) -> str:
        if self.similarity >= self.verdict_high:
            return "✔  Тот же багаж"
        if self.similarity >= self.verdict_mid:
            return "?  Вероятно тот же"
        return "✘  Другой"

    @property
    def verdict_color(self) -> str:
        if self.similarity >= self.verdict_high:
            return "#a6e3a1"
        if self.similarity >= self.verdict_mid:
            return "#f9e2af"
        return "#f38ba8"


# ── Feature extractor ──────────────────────────────────────────────────────────

class ReIDFeatureExtractor:
    """
    OSNet x1.0 (torchreid) ReID feature extractor.

    Входной размер: 128×256 px (стандарт ReID).
    Выходной вектор: 512-dim, L2-нормализованный → косинусное сходство.
    Нормализация: ImageNet mean/std (как в оригинальном torchreid).
    Thread-safe: lock вокруг каждого forward pass.

    Бэкенды:
      "pytorch"  — torch.no_grad() + model.to(device)
      "onnx_gpu" — ONNX Runtime CUDAExecutionProvider (быстрее на GTX/RTX/Quadro)
      "onnx_cpu" — ONNX Runtime CPUExecutionProvider

    v2 улучшения:
      - Оптимизированные SessionOptions для ONNX (ORT_ENABLE_ALL + memory arena)
      - TensorRT провайдер если доступен (автоматически)
      - extract_batch() эффективно обрабатывает весь кадр одним GPU-вызовом
      - onnx_half=True: FP16 ONNX (экспериментально, ~2× быстрее на RTX)
    """

    INPUT_W = 128
    INPUT_H = 256

    # ImageNet mean/std — правильная нормализация для OSNet
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    _ONNX_NAME = "osnet_x1_0_256x128.onnx"

    def __init__(
        self,
        device:    str  = "cpu",
        engine:    str  = "onnx_gpu",  # "pytorch" | "onnx_gpu" | "onnx_cpu"
        use_half:  bool = False,        # FP16 (только pytorch + CUDA)
        onnx_half: bool = False,        # FP16 ONNX (только onnx_gpu, экспериментально)
        ort_intra_threads: int = 0,     # 0 = авто
        ort_inter_threads: int = 0,
        onnx_model_path: str = "",      # путь к пользовательской ONNX-модели ReID
    ) -> None:
        self.device    = torch.device(device)
        self._on_cuda  = "cuda" in device
        self._engine   = engine
        self._use_half = use_half and self._on_cuda and engine == "pytorch"
        self._onnx_half = onnx_half and engine == "onnx_gpu"
        self._lock     = threading.Lock()
        self._custom_onnx_path = onnx_model_path.strip() if onnx_model_path else ""
        self._model        = None
        self._ort_session  = None
        self._ort_in_name  = ""
        self._ort_out_name = ""
        self._fallback_reason: Optional[str] = None
        self._ort_intra = ort_intra_threads
        self._ort_inter = ort_inter_threads

        if engine == "pytorch":
            try:
                self._init_pytorch()
            except RuntimeError as pt_exc:
                # torchreid недоступен (Python 3.12+) — пробуем ONNX если файл есть
                if os.path.exists(self._onnx_path()):
                    onnx_engine = "onnx_gpu" if self._on_cuda else "onnx_cpu"
                    logger.warning(
                        "PyTorch/torchreid недоступен (%s). "
                        "Обнаружен ONNX-файл — автопереключение на %s.",
                        pt_exc, onnx_engine
                    )
                    try:
                        self._init_onnx(use_gpu=self._on_cuda)
                        self._fallback_reason = f"pytorch→{onnx_engine}: {pt_exc}"
                        self._engine = onnx_engine
                    except Exception as onnx_exc2:
                        raise RuntimeError(
                            "Не удалось загрузить ReID ни через PyTorch, ни через ONNX.\n\n"
                            f"PyTorch ошибка: {pt_exc}\n"
                            f"ONNX ошибка:    {onnx_exc2}\n\n"
                            "Установите onnxruntime:\n"
                            "  GPU: pip install onnxruntime-gpu\n"
                            "  CPU: pip install onnxruntime\n"
                            "Затем перезапустите приложение."
                        ) from onnx_exc2
                else:
                    raise RuntimeError(
                        f"{pt_exc}\n\n"
                        "ONNX-файл osnet_x1_0_256x128.onnx не найден.\n"
                        "Установите onnxruntime и смените движок на ONNX:\n"
                        "  GPU: pip install onnxruntime-gpu\n"
                        "  CPU: pip install onnxruntime"
                    ) from pt_exc
        else:
            try:
                self._init_onnx(use_gpu=(engine == "onnx_gpu"))
            except Exception as onnx_exc:
                self._fallback_reason = str(onnx_exc)
                logger.warning(
                    "ONNX engine '%s' недоступен (%s). "
                    "Автоматический переход на PyTorch. "
                    "Смените движок в табе «Устройства» чтобы убрать это предупреждение.",
                    engine, onnx_exc
                )
                self._engine   = "pytorch"
                self._use_half = use_half and self._on_cuda
                self._init_pytorch()

        logger.info(
            "ReID: OSNet x1.0  engine=%s  device=%s%s%s",
            self._engine, device,
            " [FP16]" if self._use_half else "",
            " [ONNX-FP16]" if self._onnx_half else "",
        )

    # ── Публичные свойства ─────────────────────────────────────────────────────

    @property
    def effective_engine(self) -> str:
        return self._engine

    @property
    def fallback_occurred(self) -> bool:
        return self._fallback_reason is not None

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    # ── Инициализация бэкендов ─────────────────────────────────────────────────

    @staticmethod
    def _build_osnet() -> "torch.nn.Module":
        import sys
        from unittest.mock import MagicMock

        _OPTIONAL = ['gdown', 'tensorboard', 'torch.utils.tensorboard']
        _mocked: list = []
        for pkg in _OPTIONAL:
            if pkg not in sys.modules:
                try:
                    __import__(pkg)
                except ImportError:
                    sys.modules[pkg] = MagicMock()
                    _mocked.append(pkg)

        try:
            import torchreid
            return torchreid.models.build_model(
                name="osnet_x1_0", num_classes=1000, pretrained=True
            )
        except ImportError:
            for pkg in _mocked:
                sys.modules.pop(pkg, None)
            raise RuntimeError(
                "torchreid не установлен.\n"
                "Установите: pip install torchreid\n"
                "Или: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            )

    def _init_pytorch(self) -> None:
        m = self._build_osnet()
        m.eval().to(self.device)
        if self._use_half:
            m.half()
        self._model = m

    def _onnx_path(self) -> str:
        if self._custom_onnx_path:
            return self._custom_onnx_path
        if getattr(sys, "frozen", False):
            return str(Path(sys.executable).parent / self._ONNX_NAME)
        return str(Path(__file__).parent / self._ONNX_NAME)

    def _export_onnx_if_needed(self) -> str:
        path = self._onnx_path()
        if self._custom_onnx_path:
            # Custom path: user owns this file — we never auto-export to it
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"ReID ONNX модель не найдена: {path}\n"
                    f"Укажите корректный путь к .onnx файлу в настройках (reid_model_path)."
                )
            logger.info("ONNX ReID: используем пользовательскую модель: %s", path)
            return path
        if os.path.exists(path):
            return path

        logger.info("ONNX-модель не найдена, экспортируем OSNet → ONNX...")
        export_dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = self._build_osnet()
        m.eval().to(export_dev)
        dummy = torch.randn(1, 3, self.INPUT_H, self.INPUT_W).to(export_dev)

        torch.onnx.export(
            m, dummy, path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info("ONNX сохранён: %s", path)
        return path

    def _make_session_options(self, use_gpu: bool):
        """Создаёт оптимизированные SessionOptions для ONNX Runtime."""
        ort = self._import_onnxruntime()

        opts = ort.SessionOptions()

        # Максимальная оптимизация графа
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Для GPU: снижаем CPU-потоки, т.к. GPU выполняет вычисления
        # Для CPU: ставим 0 (auto = число ядер)
        if use_gpu:
            opts.intra_op_num_threads = self._ort_intra if self._ort_intra > 0 else 2
            opts.inter_op_num_threads = self._ort_inter if self._ort_inter > 0 else 1
        else:
            if self._ort_intra > 0:
                opts.intra_op_num_threads = self._ort_intra
            if self._ort_inter > 0:
                opts.inter_op_num_threads = self._ort_inter

        # Возвращаем GPU-память обратно после каждого батча (важно при нескольких моделях)
        opts.add_session_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0" if use_gpu else "cpu:0")

        # Детерминированное вычисление (стабильнее на GPU)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        return opts

    def _init_onnx(self, use_gpu: bool) -> None:
        ort = self._import_onnxruntime()
        path = self._export_onnx_if_needed()
        available = ort.get_available_providers()
        opts = self._make_session_options(use_gpu)

        providers: list = []
        use_trt = False
        if use_gpu:
            # TensorRT: get_available_providers() возвращает его даже без nvinfer DLL.
            # Проверяем DLL заранее через ctypes — иначе onnxruntime печатает
            # длинный EP Error в stderr ещё до нашего try/except.
            if "TensorrtExecutionProvider" in available and self._trt_dll_available():
                providers.append((
                    "TensorrtExecutionProvider",
                    {"device_id": 0, "trt_fp16_enable": self._onnx_half},
                ))
                use_trt = True
                logger.info("ONNX ReID: TensorrtExecutionProvider доступен (FP16=%s)", self._onnx_half)
            elif "TensorrtExecutionProvider" in available:
                logger.info("ONNX ReID: TensorRT в списке провайдеров, но nvinfer DLL не найден — пропускаем.")

            if "CUDAExecutionProvider" in available:
                providers.append((
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ))
                if not use_trt:
                    logger.info("ONNX ReID: CUDAExecutionProvider")
            elif use_gpu:
                logger.warning(
                    "CUDAExecutionProvider недоступен (провайдеры: %s) — используем CPU",
                    available,
                )

        providers.append("CPUExecutionProvider")

        try:
            self._ort_session = ort.InferenceSession(path, providers=providers, sess_options=opts)
        except Exception as exc:
            if use_trt and ("nvinfer" in str(exc) or "tensorrt" in str(exc).lower() or "Error 126" in str(exc)):
                # TensorRT DLL отсутствует — убираем его и пробуем снова
                logger.warning("TensorRT недоступен (%s) — используем CUDA без TRT.", exc)
                providers = [p for p in providers
                             if not (isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider")
                             and p != "TensorrtExecutionProvider"]
                self._ort_session = ort.InferenceSession(path, providers=providers, sess_options=opts)
            else:
                raise

        self._ort_in_name  = self._ort_session.get_inputs()[0].name
        self._ort_out_name = self._ort_session.get_outputs()[0].name
        actual = self._ort_session.get_providers()
        logger.info("ONNX ReID: активные провайдеры: %s", actual)

    @staticmethod
    def color_hist(crop: np.ndarray, bins: int = 16) -> np.ndarray:
        """
        Быстрая HSV цветовая гистограмма кропа (16×16×16 = 4096 bins, нормализована).
        Время: ~0.1 мс на 128×256 кроп — в 80× быстрее OSNet.
        Используется как быстрый предфильтр перед полным ReID-сравнением.
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None,
                         [bins, bins, bins],
                         [0, 180, 0, 256, 0, 256])
        h = h.flatten().astype(np.float32)
        total = h.sum()
        return h / total if total > 1e-6 else h

    @staticmethod
    def color_sim(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Intersection similarity between two normalized histograms. Range [0,1]."""
        return float(np.minimum(hist1, hist2).sum())

    @staticmethod
    def _trt_dll_available() -> bool:
        """Проверяет, загружается ли nvinfer_10.dll (TensorRT runtime).

        onnxruntime помечает TensorrtExecutionProvider как доступный даже без DLL,
        поэтому проверяем сами — до создания InferenceSession — чтобы избежать
        шумного «EP Error» в stderr при каждом запуске.
        """
        import ctypes
        # Список возможных имён: TRT 10.x и 8.x
        for dll_name in ("nvinfer_10.dll", "nvinfer_8.dll", "nvinfer.dll"):
            try:
                ctypes.CDLL(dll_name)
                return True
            except OSError:
                pass
        return False

    @staticmethod
    def _import_onnxruntime():
        import importlib
        try:
            return importlib.import_module("onnxruntime")
        except Exception as exc:
            err = str(exc)
            exc_type = type(exc).__name__
            logger.warning("onnxruntime import failed [%s]: %s", exc_type, err)
            if "DLL" in err or "WinError" in err or "_pybind_state" in err or "OSError" in exc_type:
                raise RuntimeError(
                    f"onnxruntime DLL ошибка ({exc_type}): {err}\n\n"
                    "Причина: конфликт DLL или несовместимость CUDA.\n"
                    "Решение:\n"
                    "  pip uninstall onnxruntime-gpu onnxruntime -y\n"
                    "  pip install onnxruntime\n\n"
                    "Приложение автоматически перешло на PyTorch."
                ) from exc
            elif "No module named" in err or isinstance(exc, ImportError):
                raise RuntimeError(
                    "onnxruntime не установлен.\n"
                    "GPU:  pip install onnxruntime-gpu\n"
                    "CPU:  pip install onnxruntime\n\n"
                    "Приложение автоматически перешло на PyTorch."
                ) from exc
            else:
                raise RuntimeError(
                    f"onnxruntime ошибка загрузки ({exc_type}): {err}\n\n"
                    "Попробуйте:\n"
                    "  pip uninstall onnxruntime-gpu onnxruntime -y\n"
                    "  pip install onnxruntime\n\n"
                    "Приложение автоматически перешло на PyTorch."
                ) from exc

    # ── Препроцессинг ──────────────────────────────────────────────────────────

    def _preprocess_one(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR crop → float32 1×3×H×W, ImageNet-нормализован."""
        img = cv2.resize(crop_bgr, (self.INPUT_W, self.INPUT_H),
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        return img.transpose(2, 0, 1)[np.newaxis]   # 1×3×H×W

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    # ── Публичный API ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Один кроп → 512-dim L2-нормализованный вектор. None при ошибке."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 8:
            return None
        try:
            x = self._preprocess_one(crop_bgr)
            return self._l2(self._forward(x).flatten())
        except Exception as exc:
            logger.debug("ReID extract error: %s", exc)
            return None

    @torch.no_grad()
    def extract_batch(self, crops_bgr: list) -> List[Optional[np.ndarray]]:
        """
        Список кропов → список 512-dim векторов (None для плохих кропов).

        Один GPU-вызов на весь список — значительно быстрее N × extract().
        Рекомендуется вызывать с ВСЕМИ кропами текущего кадра сразу.
        """
        if not crops_bgr:
            return []

        valid_idx: List[int] = []
        arrays: List[np.ndarray] = []
        results: List[Optional[np.ndarray]] = [None] * len(crops_bgr)

        for i, c in enumerate(crops_bgr):
            if c is not None and c.size > 0 and c.shape[0] >= 16 and c.shape[1] >= 8:
                arrays.append(self._preprocess_one(c))
                valid_idx.append(i)

        if not arrays:
            return results

        try:
            batch = np.concatenate(arrays, axis=0)  # N×3×H×W
            feats = self._forward(batch)              # N×512
            for rank, i in enumerate(valid_idx):
                results[i] = self._l2(feats[rank])
        except Exception as exc:
            logger.debug("ReID extract_batch error: %s", exc)

        return results

    # ── Внутренний forward ─────────────────────────────────────────────────────

    def _forward(self, x_np: np.ndarray) -> np.ndarray:
        """float32 NCHW numpy → float32 N×512 numpy."""
        if self._engine == "pytorch":
            t = torch.from_numpy(x_np).to(self.device)
            if self._use_half:
                t = t.half()
            with self._lock:
                out = self._model(t).cpu().float().numpy()
            return out
        else:
            with self._lock:
                out = self._ort_session.run(
                    [self._ort_out_name], {self._ort_in_name: x_np}
                )
            return out[0]


# ── In-memory ReID database ────────────────────────────────────────────────────

class ReIDDatabase:
    """
    Thread-safe store of ReIDEntry objects with automatic TTL expiry.
    One entry per (cam_name, track_id) pair — updated when better crop arrives.
    """

    def __init__(self, ttl_seconds: float = 420.0, max_size: int = 0):
        self.ttl = ttl_seconds
        self._max_size = max_size   # 0 = без ограничений
        self._entries: List[ReIDEntry] = []
        self._lock = threading.Lock()

    @staticmethod
    def _mean_emb(embeddings: list) -> Optional[np.ndarray]:
        """L2-нормализованное среднее из списка эмбеддингов (#3 gallery aggregation)."""
        if not embeddings:
            return None
        m = np.mean(embeddings, axis=0)
        n = np.linalg.norm(m)
        return (m / n) if n > 1e-8 else m

    def compute_diversity(self) -> float:
        """
        Среднее попарное косинусное расстояние между записями в БД.
        0.0 = все объекты идентичны, 1.0 = максимальное разнообразие.
        Используется для адаптивного порога: при низком diversity → поднимаем порог.
        """
        with self._lock:
            embs = [e._mean_embedding if e._mean_embedding is not None else e.embedding
                    for e in self._entries]
            if len(embs) < 2:
                return 1.0
            n = min(len(embs), 24)  # ограничиваем N^2 вычисления
            sample = embs[:n]
            dists: list = []
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    sim = float(np.dot(sample[i], sample[j]))
                    dists.append(1.0 - sim)
            return float(np.mean(dists)) if dists else 1.0

    def get_adaptive_threshold(self, base_threshold: float,
                               max_boost: float = 0.08) -> float:
        """
        Возвращает скорректированный порог:
        при diversity < 0.3 → поднимаем порог (объекты похожи, риск ошибок высок).
        """
        diversity = self.compute_diversity()
        # Линейная интерполяция: diversity 0.0→+max_boost, diversity 0.5+→+0
        boost = max(0.0, (0.5 - diversity) / 0.5) * max_boost
        return min(0.95, base_threshold + boost)

    def add_or_update(self, entry: ReIDEntry, top_k: int = 3,
                      weight: float = 0.0) -> None:
        """weight = area × confidence (#5). 0 → вычисляется из кропа."""
        if weight <= 0:
            weight = float(
                entry.crop.shape[0] * entry.crop.shape[1]
                if entry.crop is not None and entry.crop.size > 0 else 1.0
            )
        with self._lock:
            self._cleanup()
            key = (entry.cam_id, entry.track_id)
            for e in self._entries:
                if (e.cam_id, e.track_id) == key:
                    e._top_embeddings.append(entry.embedding)
                    e._top_weights.append(weight)
                    if len(e._top_embeddings) > top_k:
                        pairs = sorted(zip(e._top_weights, e._top_embeddings),
                                       reverse=True)[:top_k]
                        e._top_weights    = [w   for w, _   in pairs]
                        e._top_embeddings = [emb for _, emb in pairs]
                    best_idx = e._top_weights.index(max(e._top_weights))
                    e.embedding      = e._top_embeddings[best_idx]
                    e._mean_embedding = self._mean_emb(e._top_embeddings)
                    e.crop      = entry.crop
                    if entry.crop is not None and entry.crop.size > 0:
                        e.color_hist = ReIDFeatureExtractor.color_hist(entry.crop)
                    e.timestamp = entry.timestamp
                    return
            # Ограничение размера: удаляем самую старую запись
            if self._max_size > 0 and len(self._entries) >= self._max_size:
                self._entries.sort(key=lambda x: x.timestamp)
                self._entries.pop(0)
            entry._top_embeddings = [entry.embedding]
            entry._top_weights    = [weight]
            entry._mean_embedding = entry.embedding.copy()
            self._entries.append(entry)
            if entry.crop is not None and entry.crop.size > 0:
                entry.color_hist = ReIDFeatureExtractor.color_hist(entry.crop)

    def match(
        self,
        embedding: np.ndarray,
        from_cam_ids: Optional[List[int]] = None,
        exclude_counter_id: Optional[int] = None,
        min_age_sec: float = 0.0,
        top_k: int = 1,
    ) -> Optional[Tuple[ReIDEntry, float]]:
        """Return (best_entry, cosine_similarity) or None if DB empty.

        from_cam_ids — if set, only match entries from those cameras.
        min_age_sec  — skip entries younger than this (avoids matching
                       the same detection that just got added).
        """
        with self._lock:
            self._cleanup()
            now = time.time()
            best_sim, best = -1.0, None
            for e in self._entries:
                if from_cam_ids is not None and e.cam_id not in from_cam_ids:
                    continue
                if exclude_counter_id is not None and e.counter_id == exclude_counter_id:
                    continue
                if min_age_sec > 0 and (now - e.timestamp) < min_age_sec:
                    continue
                # #3 gallery aggregation: mean embedding более стабилен чем max
                gallery_emb = e._mean_embedding if e._mean_embedding is not None else e.embedding
                sim = float(np.dot(embedding, gallery_emb))
                if sim > best_sim:
                    best_sim, best = sim, e
            return (best, best_sim) if best is not None else None

    def match_voted(
        self,
        query_embeddings: list,
        from_cam_ids: Optional[List[int]] = None,
        min_age_sec: float = 0.0,
        vote_threshold: float = 0.65,
        min_votes: int = 3,
        query_color_hist: Optional[np.ndarray] = None,
        color_min_sim: float = 0.0,
    ) -> Optional[Tuple["ReIDEntry", float, int]]:
        """
        N×M косинусное сравнение: каждый query-эмбеддинг × каждый gallery-эмбеддинг.

        Возвращает (entry, composite_score, vote_count) или None.
        composite_score = 0.5×max_sim + 0.3×mean_sim + 0.2×vote_ratio
        """
        if not query_embeddings:
            return None
        with self._lock:
            self._cleanup()
            now = time.time()
            best_score, best_entry, best_votes = -1.0, None, 0

            for e in self._entries:
                if from_cam_ids is not None and e.cam_id not in from_cam_ids:
                    continue
                if min_age_sec > 0 and (now - e.timestamp) < min_age_sec:
                    continue
                # Fast color pre-filter
                if query_color_hist is not None and e.color_hist is not None:
                    csim = float(np.minimum(query_color_hist, e.color_hist).sum())
                    if csim < color_min_sim:
                        continue

                g_embs = e._top_embeddings if e._top_embeddings else [e.embedding]
                sims = [float(np.dot(q, g))
                        for q in query_embeddings for g in g_embs]
                if not sims:
                    continue

                votes      = sum(1 for s in sims if s >= vote_threshold)
                mean_sim   = float(np.mean(sims))
                max_sim    = float(np.max(sims))
                vote_ratio = votes / len(sims)
                score      = 0.5 * max_sim + 0.3 * mean_sim + 0.2 * vote_ratio

                if score > best_score:
                    best_score, best_entry, best_votes = score, e, votes

            if best_entry is None or best_votes < min_votes:
                return None
            return (best_entry, best_score, best_votes)

    def count(self) -> int:
        with self._lock:
            self._cleanup()
            return len(self._entries)

    def count_by_key(self, cam_id: int, track_id: int) -> int:
        with self._lock:
            self._cleanup()
            return sum(1 for e in self._entries
                       if (e.cam_id, e.track_id) == (cam_id, track_id))

    def get_old_entries(self, min_age_seconds: float) -> List[ReIDEntry]:
        with self._lock:
            self._cleanup()
            now = time.time()
            return [e for e in self._entries if now - e.timestamp >= min_age_seconds]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def save_to_file(self, path: str) -> int:
        """
        Сохраняет все записи БД в JSON-файл для восстановления после перезапуска.
        Возвращает количество сохранённых записей.
        """
        import json
        records = []
        with self._lock:
            self._cleanup()
            for e in self._entries:
                records.append({
                    "track_id":   e.track_id,
                    "counter_id": e.counter_id,
                    "cam_name":   e.cam_name,
                    "cam_id":     e.cam_id,
                    "timestamp":  e.timestamp,
                    "embedding":  e.embedding.tolist(),
                    "_top_embeddings": [x.tolist() for x in e._top_embeddings],
                    "_top_weights":    e._top_weights,
                    "_mean_embedding": e._mean_embedding.tolist()
                                       if e._mean_embedding is not None else None,
                })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "entries": records}, f)
        return len(records)

    def load_from_file(self, path: str) -> int:
        """
        Загружает записи из JSON-файла. Существующие записи НЕ очищаются.
        Возвращает количество загруженных записей.
        """
        import json
        if not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("entries", []) if isinstance(data, dict) else data
        except Exception as exc:
            logger.warning("ReID DB load_from_file failed: %s", exc)
            return 0

        loaded = 0
        with self._lock:
            for d in records:
                try:
                    e = ReIDEntry(
                        track_id   = int(d["track_id"]),
                        counter_id = int(d["counter_id"]),
                        cam_name   = str(d["cam_name"]),
                        cam_id     = int(d["cam_id"]),
                        embedding  = np.array(d["embedding"], dtype=np.float32),
                        crop       = np.zeros((32, 32, 3), dtype=np.uint8),
                        timestamp  = float(d["timestamp"]),
                    )
                    e._top_embeddings = [
                        np.array(x, dtype=np.float32) for x in d.get("_top_embeddings", [])
                    ]
                    e._top_weights    = list(d.get("_top_weights", []))
                    me = d.get("_mean_embedding")
                    e._mean_embedding = np.array(me, dtype=np.float32) if me else None
                    self._entries.append(e)
                    loaded += 1
                except Exception as exc:
                    logger.debug("Skipping corrupt ReID record: %s", exc)
        logger.info("ReID DB: loaded %d entries from %s", loaded, path)
        return loaded

    def _cleanup(self) -> None:
        now = time.time()
        self._entries = [e for e in self._entries if now - e.timestamp < self.ttl]
