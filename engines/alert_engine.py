import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("AlertEngine")

class AlertEngine:
    """
    Phase 10 & 11: Alert Engine with Anti-Spam Deduplication & Cooldown Logic
    Prevents alert spam when an event persists across hundreds of consecutive frames.
    """
    def __init__(self, camera_id: str, cooldown_seconds: float = 10.0):
        self.camera_id = camera_id
        self.cooldown_seconds = cooldown_seconds
        # Key: (event_type, target_key) -> last_alert_time (float timestamp)
        self.active_alert_cooldowns: Dict[str, float] = {}

    def _generate_event_key(self, event: Dict[str, Any]) -> str:
        """
        Creates a distinct key for deduplication.
        For crowd: ("OVERCROWDING", camera_id)
        For safety violation: ("SAFETY_VIOLATION", zone_name, track_id)
        For suspicious activity: ("SUSPICIOUS_ACTIVITY", subtype, track_id)
        """
        etype = event.get("event_type")
        meta = event.get("metadata", {})
        track_ids = event.get("track_ids", [])
        track_str = ",".join(map(str, track_ids)) if track_ids else "all"
        
        if etype == "OVERCROWDING":
            return f"OVERCROWDING_{self.camera_id}"
        elif etype == "SAFETY_VIOLATION":
            zone = meta.get("zone_name", "zone")
            return f"SAFETY_{zone}_{track_str}"
        elif etype == "SUSPICIOUS_ACTIVITY":
            subtype = meta.get("subtype", "anomaly")
            return f"ACTIVITY_{subtype}_{track_str}"
        return f"{etype}_{track_str}"

    def process_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluates events against cooldown timers and creates new alert objects only when appropriate.
        """
        alerts = []
        now = time.time()
        now_iso = datetime.now().isoformat()
        
        for event in events:
            key = self._generate_event_key(event)
            last_alerted = self.active_alert_cooldowns.get(key, 0.0)
            
            # Check if cooldown has elapsed
            if now - last_alerted >= self.cooldown_seconds:
                self.active_alert_cooldowns[key] = now
                
                alert = {
                    "alert_id": str(uuid.uuid4()),
                    "event_id": event["event_id"],
                    "camera_id": self.camera_id,
                    "event_type": event["event_type"],
                    "severity": event["severity"],
                    "message": event["message"],
                    "timestamp": now_iso,
                    "status": "ACTIVE",
                    "track_ids": event.get("track_ids", []),
                    "metadata": event.get("metadata", {})
                }
                alerts.append(alert)
                logger.info(f"🚨 NEW ALERT [{alert['severity']}] {alert['event_type']} on {self.camera_id}: {alert['message']}")
                
        return alerts
