# ✈ Airport Baggage Tracker

![Python](https://img.shields.io/badge/Python-3.9--3.11-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![YOLO](https://img.shields.io/badge/Detection-YOLOv11-orange)
![OSNet](https://img.shields.io/badge/ReID-OSNet_x1.0-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> 🇷🇺 [Русская версия](#-описание) | 🇬🇧 [English version](#-description)

---

## 🇷🇺 Описание

Система автоматически отслеживает движение багажа между стойками регистрации и лентами сортировки в аэропорту в реальном времени.

Видео с нескольких IP-камер поступает в детектор YOLO — он находит чемоданы и сумки, трекер присваивает каждому объекту уникальный ID, а нейросеть OSNet извлекает визуальный «отпечаток» каждого предмета багажа.

**Source-камеры** (стойки регистрации) сохраняют отпечатки в базу.  
**Query-камеры** (ленты сортировки) сравнивают проходящий багаж с базой и сообщают о совпадении с указанием стойки, уровня уверенности и времени в пути.

```
Стойка регистрации (source) ──► Конвейер (transit) ──► Лента сортировки (query)
         │                                                        │
   сохранить отпечаток                              совпадение → ✔ Тот же багаж (0.91)
```

Интерфейс — тёмная тема PyQt5 с живыми видеопотоками, вкладками аналитики, управлением камерами и режимом сбора датасета для дообучения моделей.

---

## 🇬🇧 Description

Automatically tracks luggage between check-in desks and sorting belts across multiple cameras in real time.

Each camera feed is processed by YOLO for object detection, a tracker assigns persistent IDs, and OSNet extracts a 512-dimensional visual embedding ("fingerprint") for each piece of luggage.

**Source cameras** (check-in desks) store embeddings in the database.  
**Query cameras** (sorting belts) match incoming bags against the database and report matches with desk number, confidence score, and transit time.

```
Check-in desk (source) ──► Conveyor (transit) ──► Sorting belt (query)
        │                                                  │
   store embedding                           match → ✔ Same bag (0.91)
```

Dark-themed PyQt5 interface with live video tiles, analytics, camera management, and a built-in dataset collection mode for model fine-tuning.

---

## ✨ Возможности / Features

| 🇷🇺 | 🇬🇧 |
|---|---|
| Детекция: YOLOv8 / YOLOv11 (nano/small/medium), пользовательские классы | Detection: YOLOv8 / YOLOv11 (nano/small/medium), custom class IDs |
| Трекинг: BoT-SORT, ByteTrack, StrongSORT | Tracking: BoT-SORT, ByteTrack, StrongSORT |
| ReID: OSNet x1.0 — ONNX GPU / CPU / PyTorch | ReID: OSNet x1.0 — ONNX GPU / CPU / PyTorch fallback |
| Матчинг: голосование N×M, цветовой пре-фильтр, адаптивный порог, временной бонус | Matching: N×M voting, color histogram pre-filter, adaptive threshold, temporal bonus |
| Мультикамерные цепочки source → transit → query | Multi-camera source → transit → query chains |
| Тёмный интерфейс PyQt5, живые тайлы, drag-and-drop | Dark PyQt5 UI, live video tiles, drag-and-drop layout |
| 11 вкладок: Монитор · Треки · Камеры · Настройки · Устройства · ReID · Обучение · Статистика · Аналитика · Совпадения · Журнал | 11 tabs: Monitor · Tracks · Cameras · Settings · Devices · ReID · Training · Statistics · Analytics · Matches · Log |
| SQLite · персистентная ReID БД (JSON) · экспорт CSV/Excel | SQLite matches DB · persistent ReID DB (JSON) · CSV/Excel export |
| Режим сбора датасета YOLO + ReID, fine-tune прямо из UI | Dataset collector (YOLO + ReID crops), fine-tune from UI |
| Веб-дашборд (localhost:8765), системный трей, авто-переподключение | Web dashboard (localhost:8765), system tray, auto-reconnect, watchdog |
| Профили конфигурации, автосохранение БД | Config profiles, DB auto-save |

---

## 💻 Требования / Requirements

| | Минимум / Minimum | Рекомендуется / Recommended |
|---|---|---|
| CPU | 4 ядра / 4 cores | 6+ ядер / 6+ cores |
| RAM | 8 ГБ / 8 GB | 16 ГБ / 16 GB |
| GPU | — (CPU режим / mode) | NVIDIA 4+ ГБ VRAM (GTX 1660+) |
| CUDA | — | 11.8 или/or 12.x |
| ОС / OS | Windows 10/11 x64 | Windows 11 x64 |
| Python | 3.9 | 3.10 |

---

## 🚀 Быстрый старт / Quick Start

### 1. Клонировать / Clone

```bash
git clone https://github.com/krams85/airport-baggage-tracker.git
cd airport-baggage-tracker
```

### 2. Установить зависимости / Install dependencies

```bash
# Windows — автоматически / automatic
install.bat

# Или вручную / or manually
pip install -r requirements.txt
```

> 🖥 **GPU (NVIDIA):** `pip install onnxruntime-gpu`  
> 💻 **CPU:** `pip install onnxruntime`

> **Опционально / Optional — дообучение OSNet / OSNet fine-tuning:**
> ```bash
> pip install git+https://github.com/KaiyangZhou/deep-person-reid.git --no-build-isolation
> ```

### 3. Скачать модели / Download models

```bash
python setup_models.py
```

Скачивает `yolo11n.pt`, экспортирует `osnet_x1_0_256x128.onnx`.  
Downloads `yolo11n.pt` and exports `osnet_x1_0_256x128.onnx`.

### 4. Настроить камеры / Configure cameras

Отредактируй `config.yaml` — укажи `file_path` (видеофайл) или `rtsp_url` (IP-камера).  
Edit `config.yaml` — set `file_path` (video file) or `rtsp_url` (IP camera).

Роль `source` — стойка регистрации, `query` — лента сортировки.  
Role `source` — check-in desk, `query` — sorting belt.

### 5. Запустить / Run

```bash
python tracker_app.py
# С перезапуском при падении / with watchdog restart:
start.bat
```

---

## 📷 Настройка камер / Camera Setup

```yaml
cameras:
  - name: Стойка 1 / Check-in Desk 1
    cam_id: 1
    role: source          # сохраняет отпечатки / stores embeddings
    mode: rtsp
    rtsp_url: rtsp://192.168.1.10:554/stream

  - name: Лента А / Sorting Belt A
    cam_id: 2
    role: query           # сравнивает с базой / matches against DB
    receives_from: 1      # ссылка на source / links to cam_id=1
    mode: rtsp
    rtsp_url: rtsp://192.168.1.11:554/stream
```

Цепочка / Multi-hop chain: `source (1) → transit (2) → query (3, receives_from: 2)`

---

## 🏗 Архитектура / Architecture

```
tracker_app.py      — PyQt5 GUI, все вкладки / all tabs, MainWindow
tracker_core.py     — AppConfig, CameraProcessor, _FrameReader, TrainingIdentityRegistry
reid_engine.py      — ReIDFeatureExtractor (ONNX/PyTorch), ReIDDatabase, MatchResult
match_storage.py    — SQLite, экспорт / export CSV/Excel
web_server.py       — aiohttp веб-дашборд / web dashboard (port 8765)
batch_engine.py     — общая YOLO для 8+ камер / shared YOLO for 8+ cameras
auto_config.py      — автовыбор профиля по VRAM / auto-select profile by VRAM
setup_models.py     — загрузка моделей / download models on first run
check_system.py     — диагностика / dependency diagnostics
```

---

## 🔍 Пайплайн матчинга / ReID Matching Pipeline

1. **Цветовой пре-фильтр / Color pre-filter** — пересечение HSV-гистограмм, пропускает OSNet для явно разных цветов / HSV histogram intersection, skips OSNet for clearly different colors
2. **OSNet** — 512-мерный L2-нормализованный вектор / 512-dim L2-normalized embedding via ONNX Runtime
3. **Голосование N×M / N×M voting** — все N эмбеддингов query против K gallery; score = 0.5·max + 0.3·mean + 0.2·vote_ratio
4. **Адаптивный порог / Adaptive threshold** — автоповышение при низком разнообразии БД / auto-raised when DB has low diversity
5. **Временной бонус / Temporal bonus** — +0.02 если время перехода соответствует `transit_time_hint`

**Вердикты / Verdicts:**
- ✅ **Тот же багаж / Same bag** — similarity ≥ 0.82
- ❓ **Вероятно тот же / Probably same** — similarity ≥ 0.68
- ❌ **Другой / Different** — similarity < 0.68

---

## 🎓 Режим обучения / Training Mode

Переключи `app_mode: training` в конфиге или через вкладку «Настройки».  
Switch `app_mode: training` in config or via the Settings tab.

Система собирает / The system collects:
- YOLO изображения + разметка / images + labels → `Dataset/datasetyolo/`
- ReID кропы по объектам / crops per identity → `Dataset/datasetReID/00001/`, `00002/`, ...

Затем во вкладке «Обучение» / Then in the **Training** tab:
- Дообучить YOLO на своих данных / Fine-tune YOLO on your dataset
- Дообучить OSNet на собранных кропах / Fine-tune OSNet on collected ReID crops

---

## ⚙️ Конфигурация / Configuration Reference

| Параметр | По умолчанию / Default | Описание / Description |
|---|---|---|
| `reid_ttl_minutes` | 7.0 | Время жизни записи в БД / DB entry lifetime |
| `reid_threshold` | 0.68 | Минимальное сходство / Minimum similarity to report match |
| `reid_verdict_high` | 0.82 | Порог «тот же» / "Same bag" threshold |
| `reid_min_votes` | 3 | Минимум голосов / Votes needed (N×M voting) |
| `color_prefilter_threshold` | 0.08 | Мин. цветовое сходство / Color histogram min similarity |
| `reid_adaptive_threshold` | true | Адаптивный порог / Auto-raise threshold for homogeneous DB |
| `transit_time_hint` | 0.0 | Ожидаемое время перехода (сек) / Expected travel time source→query (sec) |
| `infer_every_n` | 2 | YOLO каждые N кадров / Run YOLO every N frames |
| `track_min_hits` | 2 | Кадров до подтверждения трека / Frames before track is confirmed |
| `crop_pad` | 0.15 | Отступ кропа / BBox padding for ReID crop |
| `roi_crop_infer` | true | Инференс только в ROI / Inference only within ROI polygon |
| `shared_yolo` | false | Одна YOLO для всех камер / One YOLO model for all cameras (8+ cams) |

Полный список / Full list: см. `config.yaml` и `tracker_core.py → AppConfig`.

---

## 🌐 Веб-дашборд / Web Dashboard

Открой в браузере / Open in any browser: `http://localhost:8765`  
Показывает совпадения в реальном времени, статусы камер, статистику.  
Shows live match stream, camera statuses, and statistics.

---

## 🔧 Диагностика / Diagnostics

```bash
check.bat                  # Windows: полная проверка / full dependency check
python check_system.py     # кроссплатформенно / cross-platform
```

---

## 📄 Лицензия / License

MIT — см. / see [LICENSE](LICENSE)

---

## 👥 Авторы / Authors

- **Попов Олег** ([@krams85](https://github.com/krams85))
- **Иванов Сергей**

---

## 🙏 Использованные проекты / Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OSNet / torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
