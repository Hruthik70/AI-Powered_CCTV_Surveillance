# AI-Powered CCTV Surveillance System

An end-to-end intelligent surveillance platform for crowd management, workplace safety, and suspicious activity detection built with **OpenCV, YOLOv8, Multi-Object Tracking, FastAPI, and MongoDB**.

---

## 🏛️ System Pipeline Architecture

```
                         ┌─────────────────────┐
                         │   VIDEO SOURCE      │
                         │ MP4 / AVI / RTSP    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       OpenCV        │
                         │ Read, Resize, FPS   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    YOLO DETECTOR    │
                         │ Person / BoundingBox│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   OBJECT TRACKER    │
                         │ Persistent Track IDs│
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
              ▼                     ▼                      ▼
      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
      │ CROWD ENGINE │      │ ACTIVITY     │      │ SAFETY       │
      │ Person Count │      │ Movement /   │      │ Polygonal    │
      │ Overcrowding │      │ Loitering    │      │ Zones        │
      └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
             │                     │                      │
             └─────────────────────┼──────────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │   EVENT ENGINE      │
                         │ Decision Layer      │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │  ALERT ENGINE   │       │    DATABASE     │
              │ Cooldown / Spam │       │ MongoDB/Memory  │
              └────────┬────────┘       └────────┬────────┘
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │ REST / WS / MJPEG   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     DASHBOARD       │
                         │ Live Video / Events │
                         └─────────────────────┘
```

---

## 📂 Project Structure

```
AI-powered-CCTV-Surveillance/
│
├── backend/
│   ├── app.py                # FastAPI endpoints, WebSocket broadcaster, MJPEG stream
│   ├── config.py             # Camera configurations, thresholds, zones
│   ├── schemas.py            # Pydantic data schemas
│   └── pipeline_manager.py   # Multi-camera async processing pipeline
│
├── vision/
│   └── video_processor.py    # OpenCV video capture, FPS counter, frame looping
│
├── detection/
│   ├── yolo_detector.py      # Ultralytics YOLO inference & person filtering
│   └── tracker.py            # Multi-object tracking with persistent IDs & trajectories
│
├── engines/
│   ├── crowd_engine.py       # Crowd density & threshold evaluation
│   ├── safety_engine.py      # Polygonal zone violation monitoring (point-in-poly)
│   ├── activity_engine.py    # Anomaly detection (rapid movement, loitering)
│   ├── event_engine.py       # Central event synthesis layer
│   └── alert_engine.py       # Anti-spam deduplication & cooldown management
│
├── database/
│   └── db_service.py         # MongoDB async service with in-memory safe fallback
│
├── dashboard/
│   └── static/               # Real-time web UI (HTML, CSS, JS)
│
├── videos/
│   └── generate_sample.py    # Synthetic CCTV test clip generator
│
├── requirements.txt
├── main.py                   # One-click startup runner
└── README.md
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Surveillance System
```bash
python main.py
```

### 3. Access Dashboard & APIs
- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Live Video MJPEG Feed**: `http://localhost:8000/api/video/feed/CAM-01`
- **Real-time WebSocket Telemetry**: `ws://localhost:8000/ws/telemetry`

---

## 🧪 Verified Feature Highlights

1. **Anti-Spam Alert Cooldown (Phase 10 & 11)**: Prevents notification floods by enforcing a 10s cooldown per event type and track ID.
2. **Polygonal Restricted Zones (Phase 6 & 7)**: Configurable multi-point polygons with real-time ray casting violation checks.
3. **Multi-Object Tracking (Phase 5)**: Continuous tracking IDs and historical movement trajectories.
4. **Resilient Persistence (Phase 12)**: Seamlessly connects to MongoDB when available and automatically operates with an in-memory safe store otherwise.
