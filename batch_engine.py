"""
GPU-batched inference engine for 8+ cameras.

Architecture
───────────────────────────────────────────────────────────────────────────────
One YOLO model is loaded once. Every camera occupies a fixed *slot* (an index
into the batch tensor). The engine worker wakes up every `collect_ms`
milliseconds, gathers the latest frame from each slot, builds one batch, and
calls model.track(batch, persist=True) once.

Because slots are fixed-position, ultralytics' persist=True mechanism stores
trackers[slot_idx] for each camera, so tracking state is maintained correctly
across batches even though frames come from different cameras.

Offline or idle slots are padded with a black frame so the batch size never
changes (that would force ultralytics to rebuild all trackers).

Camera thread usage:
    slot = engine.register()          # once, at startup
    result = engine.infer(slot, frame) # blocks ~collect_ms + inference time
    engine.release(slot)              # when camera is removed
───────────────────────────────────────────────────────────────────────────────
"""

import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("BaggageTracker.BatchEngine")


class BatchInferenceEngine:
    """Shared single-model batch inference for all camera threads."""

    def __init__(
        self,
        model_path:      str,
        tracking_config: str,
        confidence:      float,
        iou:             float,
        classes:         list,
        device:          str,
        half:            bool,
        collect_ms:      int = 8,
    ):
        self._model_path      = model_path
        self._tracking_config = tracking_config
        self._confidence      = confidence
        self._iou             = iou
        self._classes         = list(classes)
        self._device          = device
        self._use_half        = half and "cuda" in device
        self._collect_ms      = collect_ms        # frame-gather window (ms)

        # Slot state — protected by _lock
        self._lock    = threading.Lock()
        self._n_slots = 0
        self._frames:  List[Optional[np.ndarray]] = []
        self._results: List[object]               = []
        self._events:  List[threading.Event]      = []
        self._pending: List[bool]                 = []   # slot has new frame?

        self._model:  object                       = None
        self._thread: Optional[threading.Thread]   = None
        self._stop    = False

        # Shared blank frame cache (reused, never written)
        self._blank:  Optional[np.ndarray]         = None

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def n_slots(self) -> int:
        return self._n_slots

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Load model and start worker thread. Returns False on failure."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            self._model.to(self._device)
            if self._use_half:
                self._model.model.half()
        except Exception as exc:
            logger.error("BatchEngine: model load failed: %s", exc)
            return False

        try:
            import torch
            if "cuda" in self._device:
                i = int(self._device.split(":")[-1]) if ":" in self._device else 0
                used  = torch.cuda.memory_allocated(i) / 1024 ** 2
                total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 2
                logger.info("BatchEngine: GPU %s  %.0f / %.0f MB",
                            self._device, used, total)
        except Exception:
            pass

        self._stop   = False
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="BatchInferEngine"
        )
        self._thread.start()
        logger.info("BatchEngine started  device=%s  slots=%d  collect_ms=%d",
                    self._device, self._n_slots, self._collect_ms)
        return True

    def stop(self):
        self._stop = True
        with self._lock:
            for ev in self._events:
                ev.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    # ── Slot management ──────────────────────────────────────────────────────

    def register(self) -> int:
        """Reserve a fixed inference slot. Returns the slot index."""
        with self._lock:
            idx = self._n_slots
            self._n_slots += 1
            self._frames.append(None)
            self._results.append(None)
            self._events.append(threading.Event())
            self._pending.append(False)
        return idx

    def release(self, slot: int):
        """Mark a slot as permanently inactive (camera removed)."""
        # We don't shrink the list — that would shift other slots and break
        # the fixed-position tracker invariant. Just clear the frame.
        with self._lock:
            if 0 <= slot < self._n_slots:
                self._frames[slot]  = None
                self._pending[slot] = False

    # ── Per-frame API (called by camera threads) ─────────────────────────────

    def infer(self, slot: int, frame: np.ndarray, timeout: float = 2.0):
        """
        Submit `frame` for slot `slot` and block until the result is ready.
        Returns a single ultralytics Results object, or None on error/timeout.
        """
        with self._lock:
            if slot < 0 or slot >= self._n_slots or self._stop:
                return None
            ev = self._events[slot]
            ev.clear()
            self._frames[slot]  = frame      # engine reads this, no copy needed
            self._results[slot] = None
            self._pending[slot] = True

        if not ev.wait(timeout=timeout):
            logger.warning("BatchEngine: slot %d timed out", slot)
            return None

        with self._lock:
            return self._results[slot]

    # ── Worker thread ────────────────────────────────────────────────────────

    def _get_blank(self, h: int, w: int) -> np.ndarray:
        if self._blank is None or self._blank.shape[:2] != (h, w):
            self._blank = np.zeros((h, w, 3), dtype=np.uint8)
        return self._blank

    def _worker(self):
        while not self._stop:
            time.sleep(self._collect_ms / 1000.0)

            with self._lock:
                n        = self._n_slots
                frames   = list(self._frames)          # shallow copy of refs
                pending  = list(self._pending)

            if n == 0 or not any(pending):
                continue

            # Determine reference resolution (use first non-None frame)
            ref = next((f for f in frames if f is not None), None)
            if ref is None:
                continue
            h, w = ref.shape[:2]
            blank = self._get_blank(h, w)

            # Build fixed-size batch — pad idle/offline slots with blank frame
            batch = [f if f is not None else blank for f in frames]

            try:
                results = self._model.track(
                    batch,
                    persist       = True,
                    tracker       = self._tracking_config,
                    conf          = self._confidence,
                    iou           = self._iou,
                    classes       = self._classes,
                    device        = self._device,
                    half          = self._use_half,
                    verbose       = False,
                )
            except Exception as exc:
                logger.error("BatchEngine: inference error: %s", exc)
                # Unblock all pending slots so cameras don't hang
                with self._lock:
                    for i in range(n):
                        if pending[i]:
                            self._pending[i]  = False
                            self._results[i]  = None
                            self._events[i].set()
                continue

            # Dispatch results to waiting camera threads
            with self._lock:
                for i in range(n):
                    if pending[i]:
                        self._results[i]  = results[i] if results else None
                        self._pending[i]  = False
                        self._frames[i]   = None   # consumed
                        self._events[i].set()
