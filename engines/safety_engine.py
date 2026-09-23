import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from detection.tracker import TrackedObject
from backend.config import ZoneConfig

class SafetyEngine:
    """
    Phase 6 & 7: Polygonal Zones & Workplace Safety Monitoring
    Detects unauthorized intrusions into RESTRICTED or DANGER zones.
    """
    def __init__(self, zones: List[ZoneConfig] = None):
        self.zones = zones or []

    def set_zones(self, zones: List[ZoneConfig]):
        self.zones = zones

    def is_point_in_polygon(self, point: Tuple[int, int], polygon: List[List[int]]) -> bool:
        if len(polygon) < 3:
            return False
        pts = np.array(polygon, dtype=np.int32)
        # cv2.pointPolygonTest returns >= 0 if inside or on edge
        return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0

    def evaluate(self, tracks: List[TrackedObject]) -> List[Dict[str, Any]]:
        """
        Evaluates all active tracks against configured safety zones.
        Returns a list of violations detected in this frame.
        """
        violations = []
        
        for trk in tracks:
            feet_pos = trk.current_position
            for zone in self.zones:
                if zone.type in ("RESTRICTED", "DANGER"):
                    if self.is_point_in_polygon(feet_pos, zone.polygon):
                        violations.append({
                            "type": "SAFETY_VIOLATION",
                            "severity": "HIGH" if zone.type == "RESTRICTED" else "CRITICAL",
                            "zone_name": zone.name,
                            "zone_type": zone.type,
                            "track_id": trk.track_id,
                            "position": feet_pos,
                            "confidence": 0.95,
                            "message": f"Person (Track #{trk.track_id}) entered {zone.name} ({zone.type})"
                        })
        return violations
