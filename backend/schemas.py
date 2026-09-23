from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ZoneModel(BaseModel):
    name: str
    type: str  # RESTRICTED, DANGER, ENTRY, EXIT, NORMAL
    polygon: List[List[int]]
    color: Optional[List[int]] = [0, 0, 255]

class CameraModel(BaseModel):
    camera_id: str
    name: str
    location: str
    source: str
    status: str = "ACTIVE"
    crowd_threshold: int = 15
    crowd_warning_threshold: int = 10
    zones: List[ZoneModel] = []
    fps: float = 0.0
    current_people_count: int = 0
    crowd_status: str = "NORMAL"

class EventModel(BaseModel):
    event_id: str
    camera_id: str
    event_type: str  # NORMAL, OVERCROWDING, SAFETY_VIOLATION, SUSPICIOUS_ACTIVITY, LOITERING
    severity: str    # INFO, LOW, MEDIUM, HIGH, CRITICAL
    timestamp: str
    confidence: float
    message: str
    track_ids: List[int] = []
    metadata: Dict[str, Any] = {}

class AlertModel(BaseModel):
    alert_id: str
    event_id: str
    camera_id: str
    event_type: str
    severity: str
    message: str
    timestamp: str
    status: str = "ACTIVE"  # ACTIVE, ACKNOWLEDGED, RESOLVED
    track_ids: List[int] = []
    metadata: Dict[str, Any] = {}

class TelemetryFrame(BaseModel):
    camera_id: str
    timestamp: str
    fps: float
    people_count: int
    crowd_status: str
    active_events: List[EventModel]
    recent_alerts: List[AlertModel]
    tracks_count: int
