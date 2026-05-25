#!/usr/bin/env python3
"""
Core tracking logic: config dataclasses, YAML I/O, QThread camera processor.
Imported by tracker_app.py — can also be used headlessly.

v2 изменения:
  - Дефолт reid_engine = "onnx_gpu" (ранее "pytorch") — значительно меньше CPU
  - Дефолт infer_every_n = 2 — YOLO на каждый 2-й кадр (плавно и быстро)
  - Дефолт reid_every_n = 3 — ReID раз в 3 детекции
  - Дефолт shared_yolo = True — одна модель YOLO на все камеры (экономит VRAM)
  - Дефолт infer_imgsz = 640, но можно снизить до 416/320 для скорости
  - Батчевый ReID: все кропы текущего кадра обрабатываются одним GPU-вызовом
    через extract_batch() вместо N × extract() — ключевое улучшение FPS
  - Новые параметры: reid_batch_crops, reid_aggregation_per_frame,
    motion_detect, display_width и др.
"""

import logging
import os
import queue as _queue
import threading as _threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import QThread, pyqtSignal

from reid_engine import (
    ReIDDatabase, ReIDEntry, ReIDFeatureExtractor, MatchResult,
)

logger = logging.getLogger("BaggageTracker")


class TrainingIdentityRegistry:
    """
    Thread-safe registry: maps (cam_id, track_id) → global_identity_id.

    Позволяет собирать датасет ReID в правильном формате:
    один чемодан с нескольких камер → одна папка (одна identity).

    Логика:
      - Source-камеры: новый трек → новая identity (порядковый номер).
      - Query-камеры: новый трек → ищем последнюю завершённую source-identity
        в окне `link_timeout` секунд. Если ровно одна → линкуем.
        Если несколько → берём самую свежую. Если ни одной → новая identity.
    """
    def __init__(self, link_timeout: float = 60.0):
        self._lock   = _threading.Lock()
        self._next   = 1
        self._map:   Dict[tuple, int]       = {}   # (cam_id, tid) → identity_id
        self._ended: List[tuple]            = []   # [(identity_id, end_time)]
        self.link_timeout = link_timeout

    def get_or_create(self, cam_id: int, tid: int,
                      is_source: bool, now: float) -> int:
        key = (cam_id, tid)
        with self._lock:
            if key in self._map:
                return self._map[key]
            if is_source:
                identity = self._next
                self._next += 1
            else:
                self._prune(now)
                if self._ended:
                    # Берём самую свежую завершённую source-identity
                    identity, _ = max(self._ended, key=lambda x: x[1])
                    self._ended = [(i, t) for i, t in self._ended if i != identity]
                else:
                    identity = self._next
                    self._next += 1
            self._map[key] = identity
            return identity

    def track_ended(self, cam_id: int, tid: int, now: float) -> None:
        key = (cam_id, tid)
        with self._lock:
            if key in self._map:
                self._ended.append((self._map[key], now))

    def reset(self) -> None:
        with self._lock:
            self._next = 1
            self._map.clear()
            self._ended.clear()

    def _prune(self, now: float) -> None:
        self._ended = [(i, t) for i, t in self._ended
                       if now - t <= self.link_timeout]


# ── GPU-декодирование кадров (NVDEC через cv2.cudacodec) ──────────────────────

class _GpuCapture:
    """
    Обёртка над cv2.cudacodec.VideoReader с API, совместимым с cv2.VideoCapture.

    Декодирование H264/H265 выполняется на GPU (NVDEC) — CPU освобождается от
    декодирования, что важно при нескольких камерах высокого разрешения.
    Кадр скачивается в CPU numpy-массив через GpuMat.download() — этот шаг быстр
    (PCIe 16x ~12 ГБ/с), но если хочется убрать и его, нужна полная CUDA-цепочка
    (GpuMat → CUDA tensor → YOLO), что требует специальной сборки Ultralytics.

    Требования:
      pip install opencv-contrib-python  (стандартный opencv-python НЕ содержит cudacodec)
      NVIDIA GPU + NVDEC (любая GeForce/RTX 10xx и новее)
    """

    def __init__(self, reader) -> None:
        self._reader = reader
        self._opened = True

    def read(self):
        try:
            ok, gpumat = self._reader.nextFrame()
            if not ok or gpumat is None:
                self._opened = False
                return False, None
            return True, gpumat.download()
        except Exception:
            self._opened = False
            return False, None

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        pass  # cudacodec не требует явного освобождения

    def get(self, prop: int) -> float:
        return 0.0

    def set(self, prop: int, value) -> bool:
        return False


# ── Фоновый поток чтения кадров ───────────────────────────────────────────────

class _FrameReader(_threading.Thread):
    """
    Читает кадры из VideoCapture в отдельном потоке.
    Всегда держит только самый свежий кадр (при медленном инференсе
    старые кадры вытесняются новыми — агрессивный дроп).
    """
    STOPPED = object()

    def __init__(self, cap: cv2.VideoCapture, file_fps: float = 0.0) -> None:
        super().__init__(daemon=True, name="FrameReader")
        self._cap      = cap
        self._q: _queue.Queue = _queue.Queue(maxsize=4)
        self._running  = True
        # >0 только для файлов — ограничивает скорость чтения до нативного FPS
        self._frame_dt = (1.0 / file_fps) if file_fps > 1 else 0.0

    def run(self) -> None:
        import time as _time
        t_next = _time.monotonic()
        while self._running:
            # Для файлов выдерживаем интервал между кадрами
            if self._frame_dt > 0:
                now = _time.monotonic()
                if now < t_next:
                    _time.sleep(t_next - now)
                t_next = _time.monotonic() + self._frame_dt

            try:
                ok, frame = self._cap.read()
            except Exception as exc:
                # cv2.error или другое исключение при чтении RTSP — сигнализируем стоп
                logger.warning("FrameReader: cap.read() exception (%s) — reconnecting", exc)
                ok, frame = False, None

            if not ok or frame is None:
                try:
                    self._q.put_nowait(self.STOPPED)
                except _queue.Full:
                    pass
                return
            # Дренируем — оставляем ≤1 ожидающего кадра
            while self._q.qsize() >= 2:
                try:
                    self._q.get_nowait()
                except _queue.Empty:
                    break
            try:
                self._q.put_nowait(frame)
            except _queue.Full:
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(frame)
                except _queue.Empty:
                    pass

    def read(self, timeout: float = 0.3):
        try:
            return self._q.get(timeout=timeout)
        except _queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False


_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Все 80 COCO классов ───────────────────────────────────────────────────────
COCO_NAMES: Dict[int, str] = {
    0:"person",       1:"bicycle",      2:"car",           3:"motorcycle",
    4:"airplane",     5:"bus",          6:"train",         7:"truck",
    8:"boat",         9:"traffic light",10:"fire hydrant", 11:"stop sign",
    12:"parking meter",13:"bench",      14:"bird",         15:"cat",
    16:"dog",         17:"horse",       18:"sheep",        19:"cow",
    20:"elephant",    21:"bear",        22:"zebra",        23:"giraffe",
    24:"backpack",    25:"umbrella",    26:"handbag",      27:"tie",
    28:"suitcase",    29:"frisbee",     30:"skis",         31:"snowboard",
    32:"sports ball", 33:"kite",        34:"baseball bat", 35:"baseball glove",
    36:"skateboard",  37:"surfboard",   38:"tennis racket",39:"bottle",
    40:"wine glass",  41:"cup",         42:"fork",         43:"knife",
    44:"spoon",       45:"bowl",        46:"banana",       47:"apple",
    48:"sandwich",    49:"orange",      50:"broccoli",     51:"carrot",
    52:"hot dog",     53:"pizza",       54:"donut",        55:"cake",
    56:"chair",       57:"couch",       58:"potted plant", 59:"bed",
    60:"dining table",61:"toilet",      62:"tv",           63:"laptop",
    64:"mouse",       65:"remote",      66:"keyboard",     67:"cell phone",
    68:"microwave",   69:"oven",        70:"toaster",      71:"sink",
    72:"refrigerator",73:"book",        74:"clock",        75:"vase",
    76:"scissors",    77:"teddy bear",  78:"hair drier",   79:"toothbrush",
}

_CLASS_COLORS: Dict[int, tuple] = {
    0:  (50,  180, 255),
    24: (0,   220, 110),
    25: (200, 100, 255),
    26: (255, 140,   0),
    27: (100, 200, 255),
    28: (30,  144, 255),
    63: (100, 255, 200),
    67: (100, 100, 255),
}

def _class_meta(cls_id: int, custom_names: dict = None) -> tuple:
    if custom_names and cls_id in custom_names:
        name = custom_names[cls_id]
    else:
        name  = COCO_NAMES.get(cls_id, f"cls{cls_id}")
    if cls_id in _CLASS_COLORS:
        return name, _CLASS_COLORS[cls_id]
    import colorsys
    h = (cls_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.90)
    return name, (int(b * 255), int(g * 255), int(r * 255))

CLASS_META: Dict[int, tuple] = {cid: _class_meta(cid) for cid in range(80)}


# ── Config dataclasses ─────────────────────────────────────────────────────────

@dataclass
class CameraEntry:
    name: str        = "New Camera"
    cam_id: int      = 0            # уникальный ID для маршрутизации ReID
    counter_id: int  = 1            # номер стойки (только source-камеры, для отчётов)
    mode: str        = "rtsp"
    rtsp_url: str    = ""
    file_path: str   = ""
    loop_video: bool = True
    enabled: bool    = True
    role: str        = "source"          # "source" | "transit" | "query"
    receives_from: Optional[int] = None  # cam_id камеры-предшественника
    roi: Optional[list] = None           # нормализованные [0..1], None = без ROI

    # ── Per-camera overrides (None = use global AppConfig value) ──────────────
    cam_confidence:     Optional[float]     = None
    cam_iou:            Optional[float]     = None
    cam_classes:        Optional[List[int]] = None   # None = global; [] = empty filter
    cam_infer_every_n:  Optional[int]       = None
    cam_infer_imgsz:    Optional[int]       = None
    cam_reid_every_n:   Optional[int]       = None
    cam_reid_min_crop_px: Optional[int]     = None
    cam_crop_pad: Optional[float]           = None
    cam_motion_detect:  Optional[bool]      = None
    cam_motion_min_area: Optional[int]      = None
    cam_training_save_interval: Optional[float] = None
    cam_training_bag_cooldown:  Optional[float] = None

    @property
    def source(self) -> str:
        return self.file_path if self.mode == "file" else self.rtsp_url

    def short_source(self) -> str:
        s = self.source
        return ("…" + s[-38:]) if len(s) > 40 else s


@dataclass
class AppConfig:
    cameras: List[CameraEntry]  = field(default_factory=list)
    # ── Модель ────────────────────────────────────────────────────────────────
    model_path: str             = "yolo11n.pt"          # nano быстрее small при том же качестве для багажа
    tracking_config: str        = "botsort.yaml"
    confidence: float           = 0.40
    iou: float                  = 0.50
    classes: List[int]          = field(default_factory=lambda: [24, 26, 28])
    custom_class_names: Dict[int, str] = field(default_factory=dict)  # {id: "name"} для классов вне COCO80
    # ── Устройства (per-component) ────────────────────────────────────────────
    device: str                 = "auto"        # legacy
    yolo_device: str            = "auto"
    yolo_half: bool             = False
    reid_engine: str            = "onnx_gpu"    # КЛЮЧЕВОЕ: ONNX GPU по умолчанию
    reid_device: str            = "auto"
    reid_half: bool             = False
    half: bool                  = False         # legacy
    # ── Производительность ────────────────────────────────────────────────────
    infer_every_n: int          = 2             # YOLO каждые 2 кадра
    infer_imgsz: int            = 640           # размер входа YOLO (416 = ~40% быстрее)
    display_fps_limit: int      = 25
    display_width: int          = 0             # 0 = без ресайза GUI
    # ── ReID анализ ───────────────────────────────────────────────────────────
    reid_every_n: int           = 3             # ReID каждые 3 детекции трека
    reid_batch_crops: bool      = True          # батчевый GPU-вызов на весь кадр
    track_min_hits: int         = 2
    track_max_age: int          = 30
    motion_detect: bool         = False
    motion_min_area: int        = 1000
    reid_embedding_cache: bool  = True
    # ── Режим работы ──────────────────────────────────────────────────────────
    app_mode: str               = "production"  # "training" | "production"
    reid_ttl_minutes: float     = 7.0
    reid_threshold: float       = 0.68
    reid_verdict_high: float    = 0.82
    reid_verdict_mid: float     = 0.68
    # ── ReID расширенные ──────────────────────────────────────────────────────
    reid_min_crop_px: int       = 32
    crop_pad: float             = 0.15  # отступ от bbox (0.15 = 15% каждой стороны); 0 = без отступа
    reid_aggregation: str       = "max"         # "max" | "mean"
    reid_max_db_size: int       = 0             # 0 = без ограничений
    reid_top_k: int             = 1
    reid_min_age_sec: float     = 3.0           # пропускать записи моложе N сек при match()
    # ── Оверлей на видео ──────────────────────────────────────────────────────
    overlay_bbox: bool          = True
    overlay_track_id: bool      = True
    overlay_conf: bool          = False
    overlay_class: bool         = True
    # ── Веб-дашборд ───────────────────────────────────────────────────────────
    web_port: int               = 8765
    reid_model_path: str         = ""  # путь к .onnx файлу ReID; пусто = искать osnet_x1_0_256x128.onnx рядом со скриптом
    # ── RTSP / декодирование ──────────────────────────────────────────────────
    stream_buffer: int          = 1
    reconnect_delay: float      = 5.0
    decode_device: str          = "auto"   # "auto"|"gpu"|"cpu" — аппаратный NVDEC или CPU FFmpeg
    # ── Оптимизации ROI ───────────────────────────────────────────────────────
    roi_crop_infer: bool        = True     # подавать YOLO только кроп ROI bbox (↑ FPS при малом ROI)
    # ── Снимки совпадений ─────────────────────────────────────────────────────
    snapshot_on_match: bool     = False
    snapshots_dir: str          = "Snapshots"
    # ── Датасет ───────────────────────────────────────────────────────────────
    yolo_images_dir: str        = "Dataset/datasetyolo/images"
    yolo_labels_dir: str        = "Dataset/datasetyolo/labels"
    reid_dir: str               = "Dataset/datasetReID"
    yolo_save_every_n: int      = 5
    reid_save_every_k: int      = 3
    training_link_timeout: float = 60.0  # секунд: окно линковки source→query треков
    training_save_interval: float = 0.3   # мин. секунд между сохранениями одного багажа
    training_bag_cooldown:  float = 1.0   # секунд паузы после исчезновения багажа
    # ── ReID voting (N×M embedding comparison) ────────────────────────────────
    reid_vote_threshold: float  = 0.65   # порог пары для "голоса"
    reid_min_votes:      int    = 3      # мин. голосов для признания совпадения
    reid_vote_every_n:   int    = 5      # собрать N эмбеддингов прежде чем голосовать
    reid_adaptive_threshold: bool  = True   # автоподстройка порога на основе разнообразия БД
    reid_adaptive_max_boost: float = 0.08   # максимальная надбавка к порогу (0.08 = +8%)
    transit_time_hint: float = 0.0   # ожидаемое время (сек) source→query; 0 = не используется
    color_prefilter_threshold: float = 0.08  # мин. цветовое сходство гистограмм; 0 = выключен
    # ── Персистентная БД ReID ────────────────────────────────────────────────
    reid_db_path:           str = ""   # путь к JSON-файлу для сохранения/загрузки БД
    reid_autosave_interval: int = 0    # сек; 0 = отключено
    # ── Обучение YOLO ─────────────────────────────────────────────────────────
    train_epochs: int           = 100
    train_batch: int            = 8
    train_imgsz: int            = 640
    train_lr0: float            = 0.01
    train_patience: int         = 50
    train_device: str           = "0"
    train_workers: int          = 2
    train_project: str          = "runs/train"
    # ── Масштаб (8+ камер) ────────────────────────────────────────────────────
    shared_yolo: bool           = False         # одна YOLO на все камеры (True = экономит VRAM, настраивается)


def resolve_device(cfg: "AppConfig", component: str = "yolo") -> str:
    if component == "reid":
        raw = cfg.reid_device
    elif component == "train":
        raw = cfg.train_device
    else:
        raw = cfg.yolo_device if cfg.yolo_device != "auto" else cfg.device

    if raw != "auto":
        return raw
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("GPU найден: %s → cuda:0 (%s)", name, component)
            return "cuda:0"
    except ImportError:
        pass
    logger.info("CUDA недоступен → CPU (%s)", component)
    return "cpu"


def gpu_info() -> List[str]:
    lines = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1024 ** 3
                lines.append(f"cuda:{i}  {props.name}  ({total:.1f} GB)")
        else:
            lines.append("CUDA недоступен")
    except ImportError:
        lines.append("torch не установлен")
    return lines


def load_config(path: str) -> AppConfig:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return AppConfig()

    cameras = [
        CameraEntry(
            name          = str(c.get("name", "Camera")),
            cam_id        = int(c.get("cam_id", idx + 1)),
            counter_id    = int(c.get("counter_id", 1)),
            mode          = str(c.get("mode", "rtsp")),
            rtsp_url      = str(c.get("rtsp_url", "")),
            file_path     = str(c.get("file_path", "")),
            loop_video    = bool(c.get("loop_video", True)),
            enabled       = bool(c.get("enabled", True)),
            role          = str(c.get("role", "source")),
            receives_from = c.get("receives_from", None),
            roi           = c.get("roi", None),
            cam_confidence      = c.get("cam_confidence", None),
            cam_iou             = c.get("cam_iou", None),
            cam_classes         = c.get("cam_classes", None),
            cam_infer_every_n   = int(c["cam_infer_every_n"]) if "cam_infer_every_n" in c else None,
            cam_infer_imgsz     = int(c["cam_infer_imgsz"]) if "cam_infer_imgsz" in c else None,
            cam_reid_every_n    = int(c["cam_reid_every_n"]) if "cam_reid_every_n" in c else None,
            cam_reid_min_crop_px = int(c["cam_reid_min_crop_px"]) if "cam_reid_min_crop_px" in c else None,
            cam_crop_pad        = c.get("cam_crop_pad", None),
            cam_motion_detect   = bool(c["cam_motion_detect"]) if "cam_motion_detect" in c else None,
            cam_motion_min_area = int(c["cam_motion_min_area"]) if "cam_motion_min_area" in c else None,
            cam_training_save_interval = c.get("cam_training_save_interval", None),
            cam_training_bag_cooldown  = c.get("cam_training_bag_cooldown",  None),
        )
        for idx, c in enumerate(raw.get("cameras", []))
    ]
    m = raw.get("model", {})
    d = raw.get("dataset", {})

    return AppConfig(
        cameras         = cameras,
        model_path      = str(m.get("path", "yolo11n.pt")),
        tracking_config = str(m.get("tracking_config", "botsort.yaml")),
        confidence      = float(m.get("confidence", 0.40)),
        iou             = float(m.get("iou", 0.5)),
        classes         = list(m.get("classes", [24, 26, 28])),
        custom_class_names = {int(k): str(v) for k, v in m.get("custom_class_names", {}).items()},
        yolo_images_dir = str(d.get("yolo_images_dir", "Dataset/datasetyolo/images")),
        yolo_labels_dir = str(d.get("yolo_labels_dir", "Dataset/datasetyolo/labels")),
        reid_dir        = str(d.get("reid_dir", "Dataset/datasetReID")),
        yolo_save_every_n = int(d.get("yolo_save_every_n_frames", 5)),
        reid_save_every_k = int(d.get("reid_save_every_k_detections", 3)),
        training_link_timeout = float(raw.get("training_link_timeout", 60.0)),
        training_save_interval = float(raw.get("training_save_interval", 0.3)),
        training_bag_cooldown  = float(raw.get("training_bag_cooldown",  1.0)),
        reid_vote_threshold = float(raw.get("reid_vote_threshold", 0.65)),
        reid_min_votes      = int(raw.get("reid_min_votes", 3)),
        reid_vote_every_n   = int(raw.get("reid_vote_every_n", 5)),
        reid_adaptive_threshold = bool(raw.get("reid_adaptive_threshold", True)),
        reid_adaptive_max_boost = float(raw.get("reid_adaptive_max_boost", 0.08)),
        transit_time_hint       = float(raw.get("transit_time_hint", 0.0)),
        color_prefilter_threshold = float(raw.get("color_prefilter_threshold", 0.08)),
        reid_db_path           = str(raw.get("reid_db_path", "")),
        reid_autosave_interval = int(raw.get("reid_autosave_interval", 0)),
        reid_model_path   = str(raw.get("reid_model_path", "")),
        reconnect_delay   = float(raw.get("reconnect_delay", 5.0)),
        device            = str(m.get("device", "auto")),
        yolo_device       = str(m.get("yolo_device", m.get("device", "auto"))),
        yolo_half         = bool(m.get("yolo_half", m.get("half", False))),
        reid_engine       = str(m.get("reid_engine", "onnx_gpu")),   # дефолт ONNX GPU
        reid_device       = str(m.get("reid_device", "auto")),
        reid_half         = bool(m.get("reid_half", False)),
        half              = bool(m.get("half", False)),
        infer_every_n     = int(m.get("infer_every_n", 2)),
        infer_imgsz       = int(m.get("infer_imgsz", 640)),
        display_fps_limit = int(m.get("display_fps_limit", 25)),
        display_width     = int(m.get("display_width", 0)),
        reid_every_n      = int(raw.get("reid_every_n", 3)),
        reid_batch_crops  = bool(raw.get("reid_batch_crops", True)),
        track_min_hits    = int(raw.get("track_min_hits", 2)),
        track_max_age     = int(raw.get("track_max_age", 30)),
        motion_detect     = bool(raw.get("motion_detect", False)),
        motion_min_area   = int(raw.get("motion_min_area", 1000)),
        reid_embedding_cache = bool(raw.get("reid_embedding_cache", True)),
        app_mode          = str(raw.get("app_mode", "production")),
        reid_ttl_minutes  = float(raw.get("reid_ttl_minutes", 7.0)),
        reid_threshold    = float(raw.get("reid_threshold", 0.68)),
        reid_verdict_high = float(raw.get("reid_verdict_high", 0.82)),
        reid_verdict_mid  = float(raw.get("reid_verdict_mid",  0.68)),
        reid_min_crop_px  = int(raw.get("reid_min_crop_px", 32)),
        crop_pad          = float(raw.get("crop_pad", 0.15)),
        reid_aggregation  = str(raw.get("reid_aggregation", "max")),
        reid_max_db_size  = int(raw.get("reid_max_db_size", 0)),
        reid_top_k        = int(raw.get("reid_top_k", 1)),
        reid_min_age_sec  = float(raw.get("reid_min_age_sec", 3.0)),
        overlay_bbox      = bool(raw.get("overlay_bbox", True)),
        overlay_track_id  = bool(raw.get("overlay_track_id", True)),
        overlay_conf      = bool(raw.get("overlay_conf", False)),
        overlay_class     = bool(raw.get("overlay_class", True)),
        web_port          = int(raw.get("web_port", 8765)),
        stream_buffer     = int(raw.get("stream_buffer", 1)),
        decode_device     = str(raw.get("decode_device", "auto")),
        roi_crop_infer    = bool(raw.get("roi_crop_infer", True)),
        snapshot_on_match = bool(raw.get("snapshot_on_match", False)),
        snapshots_dir     = str(raw.get("snapshots_dir", "Snapshots")),
        train_epochs      = int(raw.get("train_epochs", 100)),
        train_batch       = int(raw.get("train_batch", 8)),
        train_imgsz       = int(raw.get("train_imgsz", 640)),
        train_lr0         = float(raw.get("train_lr0", 0.01)),
        train_patience    = int(raw.get("train_patience", 50)),
        train_device      = str(raw.get("train_device", "0")),
        train_workers     = int(raw.get("train_workers", 2)),
        train_project     = str(raw.get("train_project", "runs/train")),
        shared_yolo       = bool(raw.get("shared_yolo", True)),
    )


def save_config(cfg: AppConfig, path: str) -> None:
    data = {
        "cameras": [
            {
                "name": c.name, "cam_id": c.cam_id, "counter_id": c.counter_id,
                "mode": c.mode, "rtsp_url": c.rtsp_url,
                "file_path": c.file_path, "loop_video": c.loop_video,
                "enabled": c.enabled, "role": c.role,
                "receives_from": c.receives_from,
                "roi": c.roi,
                **{k: getattr(c, k) for k in [
                    "cam_confidence","cam_iou","cam_classes",
                    "cam_infer_every_n","cam_infer_imgsz","cam_reid_every_n",
                    "cam_reid_min_crop_px","cam_crop_pad",
                    "cam_motion_detect","cam_motion_min_area",
                    "cam_training_save_interval","cam_training_bag_cooldown",
                ] if getattr(c, k) is not None},
            }
            for c in cfg.cameras
        ],
        "model": {
            "path": cfg.model_path,
            "tracking_config": cfg.tracking_config,
            "confidence": cfg.confidence,
            "iou": cfg.iou,
            "classes": cfg.classes,
            "custom_class_names": cfg.custom_class_names,
            "device": cfg.device,
            "yolo_device": cfg.yolo_device,
            "yolo_half":   cfg.yolo_half,
            "reid_engine": cfg.reid_engine,
            "reid_device": cfg.reid_device,
            "reid_half":   cfg.reid_half,
            "half": cfg.half,
            "infer_every_n": cfg.infer_every_n,
            "infer_imgsz": cfg.infer_imgsz,
            "display_fps_limit": cfg.display_fps_limit,
            "display_width": cfg.display_width,
        },
        "dataset": {
            "yolo_images_dir": cfg.yolo_images_dir,
            "yolo_labels_dir": cfg.yolo_labels_dir,
            "reid_dir": cfg.reid_dir,
            "yolo_save_every_n_frames": cfg.yolo_save_every_n,
            "reid_save_every_k_detections": cfg.reid_save_every_k,
        },
        "reid_model_path":      cfg.reid_model_path,
        "reconnect_delay":      cfg.reconnect_delay,
        "app_mode":             cfg.app_mode,
        "reid_every_n":         cfg.reid_every_n,
        "reid_batch_crops":     cfg.reid_batch_crops,
        "track_min_hits":       cfg.track_min_hits,
        "track_max_age":        cfg.track_max_age,
        "motion_detect":        cfg.motion_detect,
        "motion_min_area":      cfg.motion_min_area,
        "reid_embedding_cache": cfg.reid_embedding_cache,
        "reid_ttl_minutes":  cfg.reid_ttl_minutes,
        "reid_threshold":    cfg.reid_threshold,
        "reid_verdict_high": cfg.reid_verdict_high,
        "reid_verdict_mid":  cfg.reid_verdict_mid,
        "reid_min_crop_px":  cfg.reid_min_crop_px,
        "crop_pad":          cfg.crop_pad,
        "reid_aggregation":  cfg.reid_aggregation,
        "reid_max_db_size":  cfg.reid_max_db_size,
        "reid_top_k":        cfg.reid_top_k,
        "reid_min_age_sec":  cfg.reid_min_age_sec,
        "overlay_bbox":      cfg.overlay_bbox,
        "overlay_track_id":  cfg.overlay_track_id,
        "overlay_conf":      cfg.overlay_conf,
        "overlay_class":     cfg.overlay_class,
        "web_port":          cfg.web_port,
        "stream_buffer":     cfg.stream_buffer,
        "decode_device":     cfg.decode_device,
        "roi_crop_infer":    cfg.roi_crop_infer,
        "snapshot_on_match": cfg.snapshot_on_match,
        "snapshots_dir":     cfg.snapshots_dir,
        "train_epochs":   cfg.train_epochs,
        "train_batch":    cfg.train_batch,
        "train_imgsz":    cfg.train_imgsz,
        "train_lr0":      cfg.train_lr0,
        "train_patience": cfg.train_patience,
        "train_device":   cfg.train_device,
        "train_workers":  cfg.train_workers,
        "train_project":  cfg.train_project,
        "shared_yolo":    cfg.shared_yolo,
        "training_link_timeout": cfg.training_link_timeout,
        "training_save_interval": cfg.training_save_interval,
        "training_bag_cooldown":  cfg.training_bag_cooldown,
        "reid_vote_threshold": cfg.reid_vote_threshold,
        "reid_min_votes":      cfg.reid_min_votes,
        "reid_vote_every_n":   cfg.reid_vote_every_n,
        "reid_adaptive_threshold": cfg.reid_adaptive_threshold,
        "reid_adaptive_max_boost": cfg.reid_adaptive_max_boost,
        "transit_time_hint":       cfg.transit_time_hint,
        "color_prefilter_threshold": cfg.color_prefilter_threshold,
        "reid_db_path":           cfg.reid_db_path,
        "reid_autosave_interval": cfg.reid_autosave_interval,
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


# ── Stats snapshot ─────────────────────────────────────────────────────────────

class ProcessorStats:
    __slots__ = ("active_count", "total_seen", "yolo_saved",
                 "reid_saved", "fps", "status", "bags_sorted")

    def __init__(self):
        self.active_count: int = 0
        self.total_seen:   int = 0
        self.yolo_saved:   int = 0
        self.reid_saved:   int = 0
        self.fps: float        = 0.0
        self.status: str       = "stopped"
        self.bags_sorted: int  = 0


# ── Camera Processor ───────────────────────────────────────────────────────────

class CameraProcessor(QThread):
    """One QThread per camera — emits annotated frames and stats via signals."""

    frame_ready   = pyqtSignal(object)
    crop_ready    = pyqtSignal(object)
    stats_updated = pyqtSignal(object)
    log_msg       = pyqtSignal(str)
    match_found   = pyqtSignal(object)
    track_updated = pyqtSignal(int, object, str, str)
    # (track_id, crop_ndarray_or_None, cls_name, status)
    # status: "active" | "matched" | "lost"

    def __init__(
        self,
        cam: CameraEntry,
        cfg: AppConfig,
        reid_db: Optional["ReIDDatabase"] = None,
        reid_extractor: Optional["ReIDFeatureExtractor"] = None,
        engine=None,    # BatchInferenceEngine
        slot: int = 0,
        training_registry: Optional["TrainingIdentityRegistry"] = None,
    ) -> None:
        super().__init__()
        self.cam  = cam
        self.cfg  = cfg
        self._reid_db        = reid_db
        self._reid_extractor = reid_extractor
        self._engine = engine
        self._slot   = slot
        self._training_registry = training_registry
        self._stop_flag   = False
        self._paused      = False
        self.stats        = ProcessorStats()
        self._seen_ids: set              = set()
        self._reid_counts: Dict[int,int] = defaultdict(int)
        self._matched_tracks: set        = set()
        self._db_added_tracks: set       = set()
        self._db_empty_logged: set       = set()
        # #9 re-entry detection
        self._prev_track_ids: set        = set()
        self._track_last_emb: Dict[int, np.ndarray] = {}
        self._lost_buffer: Dict[int, tuple]          = {}  # tid→(emb, crop, ts)
        self._track_id_remap: Dict[int, int]         = {}  # new_tid→canonical_tid
        self._frame_n  = 0
        self._save_n   = 0
        self._model    = None
        self._device   = "cpu"
        self._use_half = False
        self._last_stats_t  = 0.0
        self._last_frame_t  = 0.0
        self._infer_skip    = 0
        self._last_annotated: Optional[np.ndarray] = None
        self._roi_cache_key: Optional[str] = None
        self._roi_frame_wh:  tuple         = (0, 0)
        self._roi_poly_pts:  Optional[np.ndarray] = None
        self._roi_mask_arr:  Optional[np.ndarray] = None
        self._roi_bbox:      Optional[tuple] = None   # (x1,y1,x2,y2) bbox ROI-полигона
        # Feature 3: per-timing for training saves
        self._last_reid_save_t: Dict[int, float] = {}   # tid → last save timestamp
        self._last_bag_end_t:   float = 0.0              # when last bag track disappeared
        # Feature 4: N×M embedding voting buffer
        self._query_emb_buffer: Dict[int, list] = {}   # tid → [embeddings] for query voting
        self._source_departure_times: Dict[int, float] = {}  # cam_id → {track_id: end_time}
        # Actually: List of (end_time, cam_id, track_id, embedding) sorted by end_time
        self._source_track_timeline: list = []  # [(end_ts, cam_id, track_id)]
        # LUT для затемнения вне ROI: 0.35 × pixel, uint8, SIMD-оптимизирован
        self._dark_lut: np.ndarray = np.array(
            [int(i * 0.35) for i in range(256)], dtype=np.uint8
        )

    def stop(self)                -> None: self._stop_flag = True
    def set_paused(self, v: bool) -> None:
        self._paused = v
        self.stats.status = "paused" if v else "running"
        self.stats_updated.emit(self.stats)

    def _c(self, attr: str):
        """Return per-camera override (cam_<attr>) if set, else global cfg.<attr>."""
        cam_val = getattr(self.cam, f"cam_{attr}", None)
        return cam_val if cam_val is not None else getattr(self.cfg, attr)

    def _source_chain(self) -> Optional[List[int]]:
        """
        Returns list of cam_ids that are upstream of this camera (the source chain).
        Example: CAM3 receives_from=CAM2, CAM2 receives_from=CAM1
        → returns [CAM2_id, CAM1_id]
        Returns None if this camera has no receives_from (match against ALL).
        """
        if self.cam.receives_from is None:
            return None
        chain: List[int] = []
        cam_map = {c.cam_id: c for c in self.cfg.cameras}
        current_id: Optional[int] = self.cam.receives_from
        visited: set = set()
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            chain.append(current_id)
            cam = cam_map.get(current_id)
            if cam:
                current_id = cam.receives_from
            else:
                break
        return chain or None

    def run(self) -> None:
        self._log(f"Starting '{self.cam.name}'")
        self._stop_flag = False

        self._device = resolve_device(self.cfg, component="yolo")
        self._use_half = self.cfg.yolo_half and "cuda" in self._device
        self._log(
            f"'{self.cam.name}' — device: {self._device}"
            + (" [FP16]" if self._use_half else " [FP32]")
        )

        if self._engine is not None:
            self._log(f"'{self.cam.name}' — BatchEngine slot {self._slot}")
        else:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.cfg.model_path)
                self._model.to(self._device)
                if self._use_half:
                    self._model.model.half()
            except Exception as e:
                self._log(f"Model load failed: {e}")
                self._set_status("error")
                return

            try:
                import torch
                if "cuda" in self._device:
                    idx   = int(self._device.split(":")[-1]) if ":" in self._device else 0
                    used  = torch.cuda.memory_allocated(idx) / 1024 ** 2
                    total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 2
                    self._log(f"GPU memory: {used:.0f} / {total:.0f} MB")
            except Exception:
                pass

        Path(self.cfg.yolo_images_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.yolo_labels_dir).mkdir(parents=True, exist_ok=True)

        while not self._stop_flag:
            self._set_status("connecting")
            cap = self._open()
            if cap is None:
                time.sleep(self.cfg.reconnect_delay)
                continue

            self._set_status("running")
            self._log(f"'{self.cam.name}' connected.")
            try:
                self._loop(cap)
            except Exception:
                logger.exception("Processor crash in '%s'", self.cam.name)
            finally:
                cap.release()

            if not self._stop_flag:
                self._log(f"'{self.cam.name}' lost — retry in {self.cfg.reconnect_delay:.0f}s")
                self._set_status("connecting")
                time.sleep(self.cfg.reconnect_delay)

        self._set_status("stopped")
        self._log(f"'{self.cam.name}' stopped.")

    def _open(self) -> Optional[cv2.VideoCapture]:
        src = self.cam.source
        if not src:
            self._log(f"'{self.cam.name}': source not configured.")
            return None

        if self.cam.mode == "rtsp":
            # ── GPU decode (NVDEC) ─────────────────────────────────────────────
            # cv2.cudacodec требует opencv-contrib-python с CUDA.
            # Освобождает CPU от H264-декодирования (особенно при 4+ камерах).
            if self.cfg.decode_device in ("auto", "gpu"):
                _has_cudacodec = (
                    hasattr(cv2, "cudacodec")
                    and hasattr(cv2.cudacodec, "VideoReader")
                )
                if _has_cudacodec:
                    try:
                        gpu_reader = cv2.cudacodec.VideoReader(src)
                        self._log(f"'{self.cam.name}': GPU decode (NVDEC) активен.")
                        return _GpuCapture(gpu_reader)
                    except Exception as exc:
                        if self.cfg.decode_device == "gpu":
                            self._log(
                                f"'{self.cam.name}': GPU decode запрошен, но не удался: {exc}"
                            )
                        # auto: молча переходим к CPU FFmpeg
                elif self.cfg.decode_device == "gpu":
                    self._log(
                        f"'{self.cam.name}': cv2.cudacodec недоступен "
                        "(нужен opencv-contrib-python с CUDA). Используем CPU FFmpeg."
                    )

            # ── CPU decode (FFmpeg) ────────────────────────────────────────────
            # TCP устраняет потерю UDP-пакетов → H264 decode errors.
            # err_detect=careful — мягкая обработка повреждённых блоков.
            # stimeout=10 с — быстрое обнаружение обрыва соединения.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp"
                "|max_delay;500000"
                "|err_detect;careful"
                "|fflags;nobuffer+discardcorrupt"
                "|stimeout;10000000"
            )
            backend = cv2.CAP_FFMPEG
        else:
            backend = cv2.CAP_ANY

        cap = cv2.VideoCapture(src, backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cfg.stream_buffer)
        if not cap.isOpened():
            self._log(f"'{self.cam.name}': cannot open {src!r}")
            cap.release()
            return None
        return cap

    def _loop(self, cap: cv2.VideoCapture) -> None:
        # ── Для файлов: читаем FPS чтобы не воспроизводить быстрее оригинала ─
        file_fps = 0.0
        if self.cam.mode == "file":
            file_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        reader = _FrameReader(cap, file_fps=file_fps)
        reader.start()

        consec_fail = 0
        fps_t  = time.time()
        fps_n  = 0
        min_dt = 1.0 / max(1, self.cfg.display_fps_limit)

        while not self._stop_flag:
            if self._paused:
                time.sleep(0.05)
                continue

            item = reader.read(timeout=0.3)

            if item is None:
                # Если поток ридера умер без сигнала STOPPED — форсируем реконнект
                if not reader.is_alive():
                    self._log(
                        f"'{self.cam.name}': FrameReader thread died — reconnecting"
                    )
                    reader.stop()
                    return
                continue
            if item is _FrameReader.STOPPED:
                if self.cam.mode == "file" and self.cam.loop_video:
                    reader.stop()
                    reader.join(timeout=1.0)   # ждём полного завершения потока перед seek
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    reader = _FrameReader(cap, file_fps=file_fps)
                    reader.start()
                    consec_fail = 0
                    continue
                consec_fail += 1
                if consec_fail >= 10:
                    reader.stop()
                    return
                time.sleep(0.05)
                continue

            frame = item
            consec_fail = 0
            self._frame_n += 1
            fps_n += 1
            now = time.time()
            if now - fps_t >= 1.0:
                self.stats.fps = fps_n / (now - fps_t)
                fps_n = 0
                fps_t = now

            # Пропуск инференса — отправляем кэшированный кадр
            self._infer_skip += 1
            if self._infer_skip < self._c("infer_every_n"):
                if now - self._last_frame_t >= min_dt:
                    src = self._last_annotated if self._last_annotated is not None else frame
                    emit_frame = src
                    if self.cfg.display_width > 0:
                        dh, dw = src.shape[:2]
                        if dw > self.cfg.display_width:
                            scale = self.cfg.display_width / dw
                            emit_frame = cv2.resize(
                                src, (self.cfg.display_width, int(dh * scale))
                            )
                    self.frame_ready.emit(emit_frame.copy())
                    self._last_frame_t = now
                continue
            self._infer_skip = 0

            # ── ROI crop: подаём YOLO только bbox ROI, не полный кадр ───────────
            # Если ROI задан и roi_crop_infer=True — вырезаем ограничивающий
            # прямоугольник полигона. YOLO работает на меньшем изображении:
            #   ROI 50% ширины → ~4× меньше пикселей → ~2-3× быстрее инференс.
            # Координаты боксов смещаются на roi_offset перед аннотацией.
            roi_offset = (0, 0)
            infer_frame = frame
            if self.cfg.roi_crop_infer and self.cam.roi and len(self.cam.roi) >= 3:
                fh, fw = frame.shape[:2]
                roi_pts = self.cam.roi
                if isinstance(roi_pts[0], (int, float)):
                    x1n, y1n, x2n, y2n = roi_pts
                    roi_pts = [[x1n,y1n],[x2n,y1n],[x2n,y2n],[x1n,y2n]]
                px = [int(p[0] * fw) for p in roi_pts]
                py = [int(p[1] * fh) for p in roi_pts]
                rx1 = max(0, min(px));  ry1 = max(0, min(py))
                rx2 = min(fw, max(px)); ry2 = min(fh, max(py))
                if rx2 - rx1 > 32 and ry2 - ry1 > 32:  # защита от вырожденного ROI
                    infer_frame = frame[ry1:ry2, rx1:rx2]
                    roi_offset  = (rx1, ry1)

            # Contiguous memory layout → faster GPU transfer (zero-copy eligible)
            if not infer_frame.flags['C_CONTIGUOUS']:
                infer_frame = np.ascontiguousarray(infer_frame)

            try:
                if self._engine is not None:
                    _r = self._engine.infer(self._slot, infer_frame)
                    if _r is None:
                        continue
                    results = [_r]
                else:
                    results = self._model.track(
                        infer_frame,
                        persist=True,
                        tracker=self.cfg.tracking_config,
                        conf=self._c("confidence"),
                        iou=self._c("iou"),
                        classes=self._c("classes"),
                        imgsz=self._c("infer_imgsz"),
                        device=self._device,
                        half=self._use_half,
                        verbose=False,
                    )
            except Exception as e:
                self._log(f"Inference error: {e}")
                continue

            annotated, tracks, crop = self._annotate(frame, results, roi_offset=roi_offset)
            self._last_annotated = annotated

            if self._frame_n % self.cfg.yolo_save_every_n == 0 and tracks:
                self._yolo_save(frame, results)

            if now - self._last_frame_t >= min_dt:
                emit_frame = annotated
                if self.cfg.display_width > 0:
                    h, w = annotated.shape[:2]
                    if w > self.cfg.display_width:
                        scale = self.cfg.display_width / w
                        emit_frame = cv2.resize(
                            annotated, (self.cfg.display_width, int(h * scale))
                        )
                self.frame_ready.emit(emit_frame.copy())
                self._last_frame_t = now

            if crop is not None:
                self.crop_ready.emit(crop.copy())

            self._seen_ids.update(tracks.keys())
            self.stats.active_count = len(tracks)
            self.stats.total_seen   = len(self._seen_ids)

            if now - self._last_stats_t >= 0.1:
                self.stats_updated.emit(self.stats)
                self._last_stats_t = now

        reader.stop()

    # ── Annotation + батчевый ReID ─────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, results, roi_offset: tuple = (0, 0)):
        """
        Аннотирует кадр: рисует боксы, ROI, HUD.

        roi_offset — (ox, oy) смещение в пикселях: добавляется к координатам боксов,
        полученных после инференса на ROI-кропе (см. roi_crop_infer в _loop).
        """
        out   = frame.copy()
        h, w  = frame.shape[:2]
        ox, oy = roi_offset          # смещение от ROI-кропа к полному кадру
        tracks: Dict[int, str] = {}
        last_crop = None
        track_crops: Dict[int, np.ndarray] = {}  # tid → crop for TrackingTab signal

        # ── ROI полигон (кэш) ─────────────────────────────────────────────────
        roi_poly = None
        roi_key  = str(self.cam.roi)
        if self.cam.roi and len(self.cam.roi) >= 3:
            if roi_key != self._roi_cache_key or (w, h) != self._roi_frame_wh:
                roi = self.cam.roi
                if isinstance(roi[0], (int, float)):
                    x1n, y1n, x2n, y2n = roi
                    roi = [[x1n,y1n],[x2n,y1n],[x2n,y2n],[x1n,y2n]]
                pts  = np.array([[int(p[0]*w), int(p[1]*h)] for p in roi], dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                self._roi_poly_pts  = pts
                self._roi_mask_arr  = mask
                self._roi_cache_key = roi_key
                self._roi_frame_wh  = (w, h)
                # bbox ROI-полигона — используется в _loop для кропа перед YOLO
                self._roi_bbox = (
                    int(pts[:, 0].min()), int(pts[:, 1].min()),
                    int(pts[:, 0].max()), int(pts[:, 1].max()),
                )

            roi_poly = self._roi_poly_pts
            mask     = self._roi_mask_arr
            # Затемнение через LUT (SIMD uint8, в 3-4× быстрее чем float32-путь)
            out_dark = cv2.LUT(out, self._dark_lut)
            out_dark[mask > 0] = out[mask > 0]
            out = out_dark
            cv2.polylines(out, [roi_poly], True, (0, 255, 200), 2)
            for pt in roi_poly:
                cv2.circle(out, tuple(pt), 4, (0, 255, 200), -1)
            lbl_x = int(roi_poly[:, 0].mean())
            lbl_y = max(int(roi_poly[:, 1].min()) - 6, 15)
            cv2.putText(out, "ROI", (lbl_x - 12, lbl_y),
                        _FONT, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        # ── Список задач для батчевого ReID (собираем перед обработкой) ────────
        reid_tasks: List[tuple] = []  # (tid, cls_name, crop_copy, conf)

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                if box.id is None:
                    continue
                tid    = int(box.id.item())
                cls_id = int(box.cls.item())
                conf   = float(box.conf.item())
                cls_name, color = _class_meta(cls_id, self.cfg.custom_class_names)
                # ox, oy — смещение ROI-кропа: 0,0 когда ROI не используется
                x1 = max(0, int(box.xyxy[0][0]) + ox)
                y1 = max(0, int(box.xyxy[0][1]) + oy)
                x2 = min(w, int(box.xyxy[0][2]) + ox)
                y2 = min(h, int(box.xyxy[0][3]) + oy)

                if roi_poly is not None:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if cv2.pointPolygonTest(roi_poly, (float(cx), float(cy)), False) < 0:
                        continue
                tracks[tid] = cls_name

                if self.cfg.overlay_bbox:
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                lbl_parts = []
                if self.cfg.overlay_track_id:
                    lbl_parts.append(f"#{tid}")
                if self.cfg.overlay_class:
                    lbl_parts.append(cls_name)
                if self.cfg.overlay_conf:
                    lbl_parts.append(f"{conf:.2f}")
                if lbl_parts and self.cfg.overlay_bbox:
                    lbl = " ".join(lbl_parts)
                    (lw, lh), base = cv2.getTextSize(lbl, _FONT, 0.48, 1)
                    cv2.rectangle(out, (x1, y1-lh-base-6), (x1+lw+4, y1), color, -1)
                    cv2.putText(out, lbl, (x1+2, y1-base-3),
                                _FONT, 0.48, (0,0,0), 1, cv2.LINE_AA)

                # Apply configurable padding around the detected bbox
                _cp = self._c("crop_pad")
                if _cp > 0:
                    _ph = max(1, int((y2 - y1) * _cp))
                    _pw = max(1, int((x2 - x1) * _cp))
                    crop = frame[max(0, y1-_ph):min(h, y2+_ph), max(0, x1-_pw):min(w, x2+_pw)]
                else:
                    crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    track_crops[tid] = crop
                    last_crop = crop
                    self._reid_counts[tid] += 1

                    if self.cfg.app_mode == "training":
                        _now = time.time()
                        _cooldown = self._c("training_bag_cooldown")
                        _interval = self._c("training_save_interval")
                        # Wait for cooldown after previous bag disappeared
                        if _now - self._last_bag_end_t >= _cooldown:
                            _last_t = self._last_reid_save_t.get(tid, 0.0)
                            if _now - _last_t >= _interval:
                                self._reid_save(crop, tid)
                                self._last_reid_save_t[tid] = _now

                    elif (self.cfg.app_mode == "production"
                          and self._reid_extractor is not None
                          and self._reid_db is not None):
                        reid_n = max(1, self._c("reid_every_n"))
                        if self._reid_counts[tid] % reid_n == 0:
                            min_px = self._c("reid_min_crop_px")
                            if crop.shape[0] >= min_px and crop.shape[1] >= min_px:
                                if not (self.cfg.reid_embedding_cache
                                        and tid in self._matched_tracks):
                                    reid_tasks.append((tid, cls_name, crop.copy(), conf))

        # ── Батчевый ReID — один GPU-вызов на весь кадр ────────────────────────
        if reid_tasks and self._reid_extractor is not None:
            if self.cfg.reid_batch_crops and len(reid_tasks) > 1:
                crops = [c for _, _, c, _ in reid_tasks]
                embs  = self._reid_extractor.extract_batch(crops)
                for (tid, cls_name, crop, conf), emb in zip(reid_tasks, embs):
                    if emb is not None:
                        self._handle_reid(tid, cls_name, crop, emb, conf)
            else:
                for (tid, cls_name, crop, conf) in reid_tasks:
                    emb = self._reid_extractor.extract(crop)
                    if emb is not None:
                        self._handle_reid(tid, cls_name, crop, emb, conf)

        # ── HUD overlay ───────────────────────────────────────────────────────
        mode_tag = "RABOTA" if self.cfg.app_mode == "production" else "OBUCHENIE"
        if self.cfg.app_mode == "production":
            role_tag = {"source": "-> SOURCE", "query": "<- QUERY", "transit": "~ TRANSIT"}.get(self.cam.role, "")
        else:
            role_tag = ""
        if self.cam.role == "source":
            cam_line = f"Stol #{self.cam.counter_id}  ID={self.cam.cam_id}  {role_tag}"
        else:
            recv = f"<-{self.cam.receives_from}" if self.cam.receives_from else ""
            cam_line = f"ID={self.cam.cam_id}{recv}  {role_tag}"
        for i, line in enumerate([
            self.cam.name,
            cam_line,
            f"Active: {len(tracks)}  Total: {len(self._seen_ids)}",
            f"FPS: {self.stats.fps:.1f}  {mode_tag}",
        ]):
            y = 22 + i * 22
            cv2.putText(out, line, (9, y+1), _FONT, 0.52, (0,0,0),     2, cv2.LINE_AA)
            cv2.putText(out, line, (8, y),   _FONT, 0.52, (0,255,200), 1, cv2.LINE_AA)

        # #9 re-entry: треки которые пропали → в буфер потерянных
        now_t = time.time()
        _lost_tids = self._prev_track_ids - set(tracks.keys())
        if self.cam.role in ("source", "transit"):
            for lost_tid in _lost_tids:
                if lost_tid in self._track_last_emb:
                    self._lost_buffer[lost_tid] = (
                        self._track_last_emb[lost_tid], now_t
                    )
                # Training: notify registry that this source track ended
                if self._training_registry is not None:
                    self._training_registry.track_ended(self.cam.cam_id, lost_tid, now_t)
                self._last_bag_end_t = now_t  # cooldown timer
                if self.cam.role in ("source", "transit"):
                    self._source_track_timeline.append(
                        (now_t, self.cam.cam_id, lost_tid)
                    )
                    # Keep only last 50 ended tracks
                    if len(self._source_track_timeline) > 50:
                        self._source_track_timeline = self._source_track_timeline[-50:]
        self._prev_track_ids = set(tracks.keys())

        # Emit track thumbnails for TrackingTab
        for tid, cls_name in tracks.items():
            c = track_crops.get(tid)
            self.track_updated.emit(tid, c, cls_name, "active")
        # Emit "lost" for tracks that just disappeared
        for tid in _lost_tids:
            self.track_updated.emit(tid, None, "", "lost")

        return out, tracks, last_crop

    def _check_reentry(self, tid: int, emb: np.ndarray,
                       ttl: float = 30.0) -> int:
        """#9: проверяем буфер потерянных треков. Возвращает canonical_tid."""
        now = time.time()
        best_sim, best_tid = -1.0, tid
        expired = [k for k, (_, ts) in self._lost_buffer.items() if now - ts > ttl]
        for k in expired:
            del self._lost_buffer[k]
        # Порог чуть выше основного — чтобы избежать ложных переассоциаций
        threshold = min(self.cfg.reid_threshold * 1.1, 0.95)
        for old_tid, (old_emb, _) in self._lost_buffer.items():
            if old_tid not in self._db_added_tracks:
                continue
            sim = float(np.dot(emb, old_emb))
            if sim > best_sim:
                best_sim, best_tid = sim, old_tid
        if best_sim >= threshold:
            del self._lost_buffer[best_tid]
            return best_tid
        return tid

    def _handle_reid(self, tid: int, cls_name: str, crop: np.ndarray,
                     emb: np.ndarray, conf: float = 1.0) -> None:
        role = self.cam.role
        # #5 confidence-weighted embedding: кропы с высоким conf и большой площадью
        # вносят больший вклад в mean gallery
        area   = crop.shape[0] * crop.shape[1] if crop is not None else 1
        weight = float(area) * max(0.01, conf)

        # Кэшируем последний эмбеддинг для детекции потери трека (#9)
        self._track_last_emb[tid] = emb

        if role in ("source", "transit"):
            is_new = tid not in self._db_added_tracks

            # #9 re-entry: новый tid может быть тем же объектом что только что пропал
            canonical_tid = tid
            if is_new:
                canonical_tid = self._check_reentry(tid, emb)
                if canonical_tid != tid:
                    self._track_id_remap[tid] = canonical_tid
                    self._db_added_tracks.add(tid)
                    self._log(
                        f"[ReID] Re-entry {self.cam.name}: "
                        f"#{tid} → #{canonical_tid}  sim повторная"
                    )

            self._reid_db.add_or_update(
                ReIDEntry(
                    track_id   = canonical_tid,
                    counter_id = self.cam.counter_id,
                    cam_name   = self.cam.name,
                    cam_id     = self.cam.cam_id,
                    embedding  = emb,
                    crop       = crop.copy(),
                ),
                weight=weight,
            )
            if is_new and canonical_tid == tid:
                self._db_added_tracks.add(tid)
                role_label = "SOURCE" if role == "source" else "TRANSIT"
                self._log(
                    f"[ReID] DB+ [{role_label}] {self.cam.name} #{tid} "
                    f"(cam_id={self.cam.cam_id})  total={self._reid_db.count()}"
                )

        elif role == "query" and tid not in self._matched_tracks:
            source_chain = self._source_chain()
            if self._reid_db.count() == 0:
                if tid not in self._db_empty_logged:
                    self._db_empty_logged.add(tid)
                    self._log(
                        f"[ReID] {self.cam.name} #{tid}: БД пуста, "
                        f"ожидание source/transit камер"
                    )
                return

            # Query: accumulate embeddings before voting
            if tid not in self._query_emb_buffer:
                self._query_emb_buffer[tid] = []
            self._query_emb_buffer[tid].append(emb)

            vote_every = self.cfg.reid_vote_every_n
            if len(self._query_emb_buffer[tid]) < vote_every:
                return  # not enough embeddings yet

            query_embs = self._query_emb_buffer[tid]
            self._query_emb_buffer[tid] = []  # reset buffer
            # Color histogram for pre-filtering (computed from the BEST crop of this buffer)
            _q_color = None
            if self.cfg.color_prefilter_threshold > 0 and crop is not None and crop.size > 0:
                try:
                    _q_color = ReIDFeatureExtractor.color_hist(crop)
                except Exception:
                    pass
            result = self._reid_db.match_voted(
                query_embeddings=query_embs,
                from_cam_ids=source_chain,
                min_age_sec=self.cfg.reid_min_age_sec,
                vote_threshold=self.cfg.reid_vote_threshold,
                min_votes=self.cfg.reid_min_votes,
                query_color_hist=_q_color,
                color_min_sim=self.cfg.color_prefilter_threshold,
            )
            if result is None:
                return
            src_entry, sim, votes = result
            if self.cfg.transit_time_hint > 0 and result is not None:
                expected_end = time.time() - self.cfg.transit_time_hint
                time_delta = abs(src_entry.timestamp - expected_end)
                # Bonus: up to +0.02 if timing is within ±5 seconds
                if time_delta < 5.0:
                    time_bonus = 0.02 * max(0, 1.0 - time_delta / 5.0)
                    sim = min(1.0, sim + time_bonus)
                result = (src_entry, sim, votes)
            _thr = (
                self._reid_db.get_adaptive_threshold(
                    self.cfg.reid_threshold,
                    self.cfg.reid_adaptive_max_boost,
                )
                if self.cfg.reid_adaptive_threshold
                else self.cfg.reid_threshold
            )
            if sim >= _thr:
                self._matched_tracks.add(tid)
                self.stats.bags_sorted += 1
                mr = MatchResult(
                    query_track_id   = tid,
                    query_cam_name   = self.cam.name,
                    query_counter_id = self.cam.counter_id,
                    query_crop       = crop.copy(),
                    source_entry     = src_entry,
                    similarity       = sim,
                    verdict_high     = self.cfg.reid_verdict_high,
                    verdict_mid      = self.cfg.reid_verdict_mid,
                )
                self._log(
                    f"[MATCH] {self.cam.name} #{tid} <-> "
                    f"{src_entry.cam_name} #{src_entry.track_id}  "
                    f"sim={sim:.3f}  votes={votes}  desk={src_entry.counter_id}"
                )
                self.match_found.emit(mr)
                if self.cfg.snapshot_on_match:
                    self._save_match_snapshot(mr)

    # ── Dataset savers ─────────────────────────────────────────────────────────

    def _yolo_save(self, frame: np.ndarray, results) -> None:
        b = results[0].boxes
        if b is None or len(b) == 0:
            return

        lines = []
        h, w = frame.shape[:2]
        for box in b:
            cid        = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            xc = max(0.0, min(1.0, ((x1 + x2) / 2) / w))
            yc = max(0.0, min(1.0, ((y1 + y2) / 2) / h))
            bw = max(0.0, min(1.0, (x2 - x1) / w))
            bh = max(0.0, min(1.0, (y2 - y1) / h))
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            return

        self._save_n += 1
        stem = f"c{self.cam.counter_id:02d}_{self._save_n:07d}"
        try:
            img_ok = cv2.imwrite(
                str(Path(self.cfg.yolo_images_dir) / f"{stem}.jpg"),
                frame, [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            if not img_ok:
                self._log(f"YOLO: не удалось записать {stem}.jpg")
                return
            (Path(self.cfg.yolo_labels_dir) / f"{stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            self.stats.yolo_saved += 1
        except Exception as e:
            self._log(f"YOLO save error: {e}")

    def _reid_save(self, crop: np.ndarray, tid: int) -> None:
        now = time.time()
        is_source = self.cam.role == "source"

        if self._training_registry is not None:
            identity = self._training_registry.get_or_create(
                self.cam.cam_id, tid, is_source, now
            )
            d = Path(self.cfg.reid_dir) / f"{identity:05d}"
        else:
            # Fallback: old structure cam/counter_id/track_id
            d = Path(self.cfg.reid_dir) / str(self.cam.counter_id) / str(tid)

        d.mkdir(parents=True, exist_ok=True)
        fname = f"c{self.cam.cam_id}_t{tid}_{self._reid_counts[tid]:05d}.jpg"
        try:
            cv2.imwrite(str(d / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.stats.reid_saved += 1
        except Exception as e:
            self._log(f"ReID save error: {e}")

    def _save_match_snapshot(self, mr: "MatchResult") -> None:
        try:
            snap_dir = Path(self.cfg.snapshots_dir)
            snap_dir.mkdir(parents=True, exist_ok=True)

            q_crop = mr.query_crop
            s_crop = mr.source_entry.crop
            h_target = max(q_crop.shape[0], s_crop.shape[0], 128)

            def _resize_h(img, h):
                scale = h / img.shape[0]
                return cv2.resize(img, (max(1, int(img.shape[1] * scale)), h))

            q_r = _resize_h(q_crop, h_target)
            s_r = _resize_h(s_crop, h_target)
            sep = np.full((h_target, 4, 3), 200, dtype=np.uint8)
            collage = np.hstack([s_r, sep, q_r])

            bar_h = 28
            bar = np.zeros((bar_h, collage.shape[1], 3), dtype=np.uint8)
            ts  = time.strftime("%Y-%m-%d %H:%M:%S")
            txt = (f"SOURCE: {mr.source_entry.cam_name} #{mr.source_entry.track_id}"
                   f"  ->  QUERY: {mr.query_cam_name} #{mr.query_track_id}"
                   f"  sim={mr.similarity:.3f}  {ts}")
            cv2.putText(bar, txt, (4, 19), _FONT, 0.40, (0, 220, 180), 1, cv2.LINE_AA)

            final = np.vstack([collage, bar])
            fname = f"match_{ts.replace(':','-').replace(' ','_')}_sim{mr.similarity:.3f}.jpg"
            cv2.imwrite(str(snap_dir / fname), final, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self._log(f"[SNAP] {fname}")
        except Exception as exc:
            self._log(f"[SNAP] Ошибка сохранения: {exc}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _set_status(self, s: str) -> None:
        self.stats.status = s
        self.stats_updated.emit(self.stats)

    def _log(self, msg: str) -> None:
        logger.info(msg)
        self.log_msg.emit(msg)
