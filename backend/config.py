import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ZoneConfig(BaseModel):
    name: str
    type: str  # RESTRICTED, DANGER, ENTRY, EXIT, NORMAL
    polygon: List[List[int]]  # [[x1, y1], [x2, y2], ...]
    color: List[int] = [0, 0, 255]  # BGR

class CameraConfig(BaseModel):
    camera_id: str
    name: str
    location: str
    source: str  # Video path, webcam index, or RTSP URL
    crowd_threshold: int = 15
    crowd_warning_threshold: int = 8
    zones: List[ZoneConfig] = []
    fps_limit: int = 30
    alert_cooldown_seconds: float = 10.0

class AppConfig:
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME: str = "cctv_surveillance"
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8s.pt")
    CONFIDENCE_THRESHOLD: float = 0.05
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    DEFAULT_CAMERAS: List[CameraConfig] = [
        CameraConfig(
            camera_id="CAM-01",
            name="Synthetic Lobby & Zones",
            location="Simulation Area",
            source="videos/crowd.mp4",
            crowd_threshold=12,
            crowd_warning_threshold=7,
            alert_cooldown_seconds=10.0,
            zones=[
                ZoneConfig(
                    name="Restricted Zone Alpha",
                    type="RESTRICTED",
                    polygon=[[480, 80], [750, 80], [750, 320], [480, 320]],
                    color=[0, 0, 255]
                )
            ]
        ),
        CameraConfig(
            camera_id="CAM-02",
            name="Public Protest & Rally",
            location="City Center Square",
            source="videos/Protest.mp4",
            crowd_threshold=25,
            crowd_warning_threshold=12,
            alert_cooldown_seconds=10.0,
            zones=[]
        ),
        CameraConfig(
            camera_id="CAM-03",
            name="Workplace Facility",
            location="Production Floor",
            source="videos/Workplace.mp4",
            crowd_threshold=6,
            crowd_warning_threshold=4,
            alert_cooldown_seconds=10.0,
            zones=[]
        )
    ]
