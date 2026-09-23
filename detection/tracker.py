import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from detection.yolo_detector import Detection

class TrackedObject:
    def __init__(self, track_id: int, detection: Detection):
        self.track_id = track_id
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_name = detection.class_name
        self.history: List[Tuple[int, int]] = [detection.foot_position]
        self.max_history = 45
        self.age = 1
        self.lost_frames = 0
        self.current_position = detection.foot_position
        self.velocity: float = 0.0  # pixels per frame
        self.dx: float = 0.0
        self.dy: float = 0.0
        self.stationary_frames: int = 0
        self.current_zone: Optional[str] = None

    def update(self, detection: Detection):
        prev_pos = self.current_position
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.current_position = detection.foot_position
        self.history.append(self.current_position)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        self.dx = self.current_position[0] - prev_pos[0]
        self.dy = self.current_position[1] - prev_pos[1]
        dist = math.hypot(self.dx, self.dy)
        self.velocity = round(0.7 * self.velocity + 0.3 * dist, 2)
        
        if dist < 2.0:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0
            
        self.age += 1
        self.lost_frames = 0

    def predict(self):
        """Smoothly extrapolate bounding box and position on skipped frames."""
        x1, y1, x2, y2 = self.bbox
        shift_x = int(self.dx * 0.4)
        shift_y = int(self.dy * 0.4)
        self.bbox = (x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y)
        self.current_position = ((x1 + x2) // 2 + shift_x, y2 + shift_y)

    def mark_missed(self):
        self.lost_frames += 1

class MultiObjectTracker:
    """
    Phase 5: High-Performance Multi-Object Tracker with Inter-Frame Prediction
    """
    def __init__(self, max_distance: float = 90.0, max_lost: int = 15):
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = TrackedObject(self.next_id, det)
                self.next_id += 1
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        active_tracks = [self.tracks[tid] for tid in track_ids]
        
        num_tracks = len(active_tracks)
        num_dets = len(detections)
        
        matched_tracks = set()
        matched_dets = set()
        
        if num_dets > 0 and num_tracks > 0:
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
            for i, trk in enumerate(active_tracks):
                for j, det in enumerate(detections):
                    t_x, t_y = trk.current_position
                    d_x, d_y = det.foot_position
                    cost_matrix[i, j] = math.hypot(t_x - d_x, t_y - d_y)

            for _ in range(min(num_tracks, num_dets)):
                min_val = np.min(cost_matrix)
                if min_val > self.max_distance:
                    break
                row, col = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                cost_matrix[row, :] = 999999
                cost_matrix[:, col] = 999999
                
                track_id = track_ids[row]
                self.tracks[track_id].update(detections[col])
                matched_tracks.add(track_id)
                matched_dets.add(col)

        # Unmatched detections become new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks[self.next_id] = TrackedObject(self.next_id, det)
                self.next_id += 1

        # Unmatched existing tracks
        for tid, trk in list(self.tracks.items()):
            if tid not in matched_tracks:
                trk.mark_missed()
                if trk.lost_frames > self.max_lost:
                    del self.tracks[tid]

        return [t for t in self.tracks.values() if t.lost_frames == 0]

    def predict_all(self) -> List[TrackedObject]:
        for trk in self.tracks.values():
            trk.predict()
        return [t for t in self.tracks.values() if t.lost_frames == 0]
