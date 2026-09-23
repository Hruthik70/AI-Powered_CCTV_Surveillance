import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple
from detection.tracker import TrackedObject
from engines.crowd_engine import CrowdEngine
from engines.safety_engine import SafetyEngine
from engines.activity_engine import ActivityEngine

class EventEngine:
    """
    Phase 9: Central Decision Layer (Event Engine)
    Synthesizes signals from Crowd, Safety, and Activity engines into structured events.
    """
    def __init__(self, camera_id: str, crowd_engine: CrowdEngine, safety_engine: SafetyEngine, activity_engine: ActivityEngine):
        self.camera_id = camera_id
        self.crowd_engine = crowd_engine
        self.safety_engine = safety_engine
        self.activity_engine = activity_engine

    def process(self, tracks: List[TrackedObject]) -> Tuple[List[Dict[str, Any]], str, int, List[Dict[str, Any]]]:
        """
        Runs all sub-engines and generates structured event records.
        Returns:
            events: List of event dicts
            crowd_status: "NORMAL" | "MODERATE" | "OVERCROWDED"
            person_count: int
            crowd_clusters: List of spatial hotspot dicts
        """
        person_count = len(tracks)
        crowd_status, crowd_severity, crowd_conf = self.crowd_engine.evaluate(person_count)
        crowd_clusters = self.crowd_engine.find_crowd_clusters(tracks)
        
        events = []
        now_iso = datetime.now().isoformat()
        
        # 1. Overcrowding Event
        if crowd_status == "OVERCROWDED":
            events.append({
                "event_id": str(uuid.uuid4()),
                "camera_id": self.camera_id,
                "event_type": "OVERCROWDING",
                "severity": crowd_severity,
                "timestamp": now_iso,
                "confidence": crowd_conf,
                "message": f"Overcrowding alert: {person_count} persons detected (threshold: {self.crowd_engine.crowd_threshold})",
                "track_ids": [t.track_id for t in tracks],
                "metadata": {
                    "count": person_count,
                    "threshold": self.crowd_engine.crowd_threshold
                }
            })
            
        # 1b. Spatial Crowd Cluster / Convergence Event
        for cluster in crowd_clusters:
            if cluster["count"] >= 4:
                events.append({
                    "event_id": str(uuid.uuid4()),
                    "camera_id": self.camera_id,
                    "event_type": "CROWD_CLUSTER",
                    "severity": "MEDIUM",
                    "timestamp": now_iso,
                    "confidence": 0.88,
                    "message": f"Dense crowd cluster detected: {cluster['count']} people grouped together.",
                    "track_ids": cluster["track_ids"],
                    "metadata": {
                        "cluster_bbox": cluster["bbox"],
                        "count": cluster["count"]
                    }
                })

        # 2. Safety Zone Violations
        safety_violations = self.safety_engine.evaluate(tracks)
        for v in safety_violations:
            events.append({
                "event_id": str(uuid.uuid4()),
                "camera_id": self.camera_id,
                "event_type": v["type"],
                "severity": v["severity"],
                "timestamp": now_iso,
                "confidence": v["confidence"],
                "message": v["message"],
                "track_ids": [v["track_id"]],
                "metadata": {
                    "zone_name": v["zone_name"],
                    "zone_type": v["zone_type"],
                    "position": v["position"]
                }
            })
            
        # 3. Behavioral Activity Anomalies
        activity_anomalies = self.activity_engine.evaluate(tracks)
        for a in activity_anomalies:
            events.append({
                "event_id": str(uuid.uuid4()),
                "camera_id": self.camera_id,
                "event_type": a["type"],
                "severity": a["severity"],
                "timestamp": now_iso,
                "confidence": a["confidence"],
                "message": a["message"],
                "track_ids": [a["track_id"]],
                "metadata": {
                    "subtype": a["subtype"]
                }
            })
            
        return events, crowd_status, person_count, crowd_clusters
