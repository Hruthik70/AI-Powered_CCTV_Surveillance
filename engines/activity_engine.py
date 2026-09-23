from typing import List, Dict, Any
from detection.tracker import TrackedObject

class ActivityEngine:
    """
    Phase 8: Activity & Anomaly Detection Engine
    Detects behavioral anomalies: Loitering, Rapid Running / Panic Movement, and Sudden Stops.
    """
    def __init__(self, loitering_threshold_frames: int = 75, rapid_movement_threshold: float = 12.0):
        self.loitering_threshold_frames = loitering_threshold_frames
        self.rapid_movement_threshold = rapid_movement_threshold

    def evaluate(self, tracks: List[TrackedObject]) -> List[Dict[str, Any]]:
        anomalies = []
        for trk in tracks:
            # Check for Loitering / Suspicious Lingering
            if trk.stationary_frames >= self.loitering_threshold_frames:
                anomalies.append({
                    "type": "SUSPICIOUS_ACTIVITY",
                    "subtype": "LOITERING",
                    "severity": "MEDIUM",
                    "track_id": trk.track_id,
                    "confidence": 0.85,
                    "message": f"Suspicious loitering detected: Person (Track #{trk.track_id}) stationary for {trk.stationary_frames} frames."
                })
            # Check for Rapid Running / Panic Movement
            elif trk.velocity >= self.rapid_movement_threshold:
                anomalies.append({
                    "type": "SUSPICIOUS_ACTIVITY",
                    "subtype": "RAPID_MOVEMENT",
                    "severity": "HIGH",
                    "track_id": trk.track_id,
                    "confidence": 0.89,
                    "message": f"Rapid running / abrupt sprint detected: Person (Track #{trk.track_id}) velocity {trk.velocity} px/frame."
                })
        return anomalies
