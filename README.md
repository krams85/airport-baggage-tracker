# ✈ Airport Baggage Tracker

Real-time baggage re-identification system for airports using YOLO object detection and OSNet ReID.  
Система отслеживания багажа в реальном времени для аэропортов на основе YOLO и OSNet ReID.

![Python](https://img.shields.io/badge/Python-3.9--3.11-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![YOLO](https://img.shields.io/badge/Detection-YOLOv11-orange)
![OSNet](https://img.shields.io/badge/ReID-OSNet_x1.0-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What it does

Tracks luggage between check-in desks and sorting belts across multiple cameras.  
Each bag is detected by YOLO, assigned a track ID, and a 512-dimensional visual embedding is extracted by OSNet.  
Source cameras store embeddings; query cameras match incoming bags against the database and report matches with confidence scores.

```
Check-in desk (source cam) ──► Conveyor (transit cam) ──► Sorting belt (query cam)
        │                                                          │
   embed bag                                                 match → ✔ Same bag (0.91)
```

---

## Features

| Category | Details |
|---|---|
| **Detection** | YOLOv8 / YOLOv11 (nano / small / medium), custom class IDs |
| **Tracking** | BoT-SORT, ByteTrack, StrongSORT |
| **ReID** | OSNet x1.0 — ONNX GPU / CPU / PyTorch fallback |
| **Matching** | N×M voting, color histogram pre-filter, adaptive threshold, temporal bonus |
| **Multi-camera** | Source → Transit → Query chains, per-camera overrides |
| **UI** | PyQt5, dark theme, live video tiles, drag-and-drop layout |
| **Tabs** | Monitor · Tracks · Cameras · Settings · Devices · ReID · Training · Statistics · Analytics · Matches · Log |
| **Storage** | SQLite matches DB, persistent ReID DB (JSON), CSV/Excel export |
| **Training** | Dataset collector (YOLO + ReID crops), YOLO fine-tune UI, OSNet fine-tune button |
| **Deployment** | Web dashboard (localhost:8765), system tray, auto-reconnect, watchdog |

---

## Screenshots

> _Add screenshots here_

---

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 6+ cores |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU mode) | NVIDIA 4+ GB VRAM (GTX 1660+) |
| CUDA | — | 11.8 or 12.x |
| OS | Windows 10/11 x64 | Windows 11 x64 |
| Python | 3.9 | 3.10 |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/airport-baggage-tracker.git
cd airport-baggage-tracker
```

### 2. Install dependencies

```bash
# Windows — automatic install
install.bat

# Or manually
pip install -r requirements.txt
```

> **GPU (NVIDIA):** `pip install onnxruntime-gpu`  
> **CPU only:** `pip install onnxruntime`

> **Optional — OSNet fine-tuning:**
> ```bash
> pip install git+https://github.com/KaiyangZhou/deep-person-reid.git --no-build-isolation
> ```

### 3. Download models

```bash
python setup_models.py
```

Downloads `yolo11n.pt` and exports `osnet_x1_0_256x128.onnx`.

### 4. Configure cameras

Edit `config.yaml` — set `file_path` (for video files) or `rtsp_url` for IP cameras.  
Set camera `role`: `source` for check-in desks, `query` for sorting belts.

### 5. Run

```bash
python tracker_app.py
# or with watchdog restart:
start.bat
```

---

## Camera Setup

```yaml
cameras:
  - name: Check-in Desk 1
    cam_id: 1
    role: source          # adds embeddings to DB
    mode: rtsp
    rtsp_url: rtsp://192.168.1.10:554/stream

  - name: Sorting Belt A
    cam_id: 2
    role: query           # matches against DB
    receives_from: 1      # links to cam_id=1
    mode: rtsp
    rtsp_url: rtsp://192.168.1.11:554/stream
```

Multi-hop chain: `source (1) → transit (2) → query (3, receives_from: 2)`

---

## Architecture

```
tracker_app.py      — PyQt5 GUI, all tabs, MainWindow
tracker_core.py     — AppConfig, CameraProcessor, _FrameReader, TrainingIdentityRegistry
reid_engine.py      — ReIDFeatureExtractor (ONNX/PyTorch), ReIDDatabase, MatchResult
match_storage.py    — SQLite persistence, CSV/Excel export
web_server.py       — aiohttp web dashboard (port 8765)
batch_engine.py     — shared YOLO engine for 8+ cameras
auto_config.py      — auto-select performance profile by VRAM
setup_models.py     — download / export models on first run
check_system.py     — dependency diagnostics
```

---

## ReID Matching Pipeline

1. **Color pre-filter** — HSV histogram intersection, skips OSNet for clearly different colors  
2. **OSNet** — 512-dim L2-normalized embedding via ONNX Runtime  
3. **N×M voting** — all N query embeddings vs all K gallery embeddings; score = 0.5·max + 0.3·mean + 0.2·vote_ratio  
4. **Adaptive threshold** — auto-raised when DB has low diversity  
5. **Temporal bonus** — +0.02 if transit time matches `transit_time_hint`  

Verdicts:
- ✅ **Same bag** — similarity ≥ 0.82 (green)
- ❓ **Probably same** — similarity ≥ 0.68 (yellow)
- ❌ **Different** — similarity < 0.68 (red)

---

## Training Mode

Switch `app_mode: training` in config or via Settings tab.  
The system collects:
- YOLO images + labels → `Dataset/datasetyolo/`
- ReID crops per identity → `Dataset/datasetReID/00001/`, `00002/`, ...

Then in the **Training** tab:
- Fine-tune YOLO on your dataset
- Fine-tune OSNet on collected ReID crops

---

## Configuration Reference

Key parameters in `config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `reid_ttl_minutes` | 7.0 | DB entry lifetime |
| `reid_threshold` | 0.68 | Minimum similarity to report match |
| `reid_verdict_high` | 0.82 | "Same bag" threshold |
| `reid_min_votes` | 3 | Votes needed (N×M voting) |
| `color_prefilter_threshold` | 0.08 | Color histogram min similarity |
| `reid_adaptive_threshold` | true | Auto-raise threshold for homogeneous DB |
| `transit_time_hint` | 0.0 | Expected travel time source→query (sec) |
| `infer_every_n` | 2 | Run YOLO every N frames |
| `track_min_hits` | 2 | Frames before track is confirmed |
| `crop_pad` | 0.15 | BBox padding for ReID crop |
| `roi_crop_infer` | true | Inference only within ROI polygon |
| `shared_yolo` | false | One YOLO model for all cameras (8+ cams) |

Full list: see `config.yaml` and `tracker_core.py → AppConfig`.

---

## Web Dashboard

Open `http://localhost:8765` in any browser while the app is running.  
Shows live match stream, camera statuses, and statistics.

---

## Diagnostics

```bash
check.bat          # Windows: full dependency check
python check_system.py   # cross-platform
```

---

## License

MIT — see [LICENSE](LICENSE)

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OSNet / torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
