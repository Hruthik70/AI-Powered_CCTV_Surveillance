import cv2
import time
import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Any

from backend.config import CameraConfig, ZoneConfig, AppConfig
from vision.video_processor import VideoProcessor
from detection.yolo_detector import YOLODetector
from detection.tracker import MultiObjectTracker
from engines.crowd_engine import CrowdEngine
from engines.safety_engine import SafetyEngine
from engines.activity_engine import ActivityEngine
from engines.event_engine import EventEngine
from engines.alert_engine import AlertEngine
from database.db_service import db_service

logger = logging.getLogger("CameraPipeline")

class CameraPipeline:
    def __init__(self, config: CameraConfig, detector: YOLODetector):
        self.config = config
        self.camera_id = config.camera_id
        self.detector = detector
        
        self.video_processor = VideoProcessor(config.source, target_fps=config.fps_limit)
        self.tracker = MultiObjectTracker(max_distance=90.0, max_lost=15)
        self.crowd_engine = CrowdEngine(config.crowd_threshold, config.crowd_warning_threshold)
        self.safety_engine = SafetyEngine(config.zones)
        self.activity_engine = ActivityEngine()
        self.event_engine = EventEngine(
            self.camera_id, self.crowd_engine, self.safety_engine, self.activity_engine
        )
        self.alert_engine = AlertEngine(self.camera_id, cooldown_seconds=config.alert_cooldown_seconds)
        
        # Frame skipping & FPS optimizations
        self.frame_index = 0
        self.inference_interval = 2  # Run YOLO every 2nd frame, extrapolate on intermediate
        self.cached_tracks = []
        self.cached_events = []
        self.cached_alerts = []
        self.cached_clusters = []
        self.cached_crowd_status = "NORMAL"
        self.cached_person_count = 0
        
        # Live state
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated_jpeg: Optional[bytes] = None
        self.latest_telemetry: Dict[str, Any] = {
            "camera_id": self.camera_id,
            "name": config.name,
            "location": config.location,
            "status": "ACTIVE",
            "fps": 0.0,
            "people_count": 0,
            "crowd_status": "NORMAL",
            "crowd_clusters": [],
            "active_events": [],
            "recent_alerts": [],
            "zones": [z.model_dump() if hasattr(z, 'model_dump') else z.__dict__ for z in config.zones]
        }
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

    def update_zones(self, zones: List[ZoneConfig]):
        self.config.zones = zones
        self.safety_engine.set_zones(zones)

    def draw_annotations(self, frame: np.ndarray, tracks, events, alerts, crowd_status: str, person_count: int, crowd_clusters: List[Dict[str, Any]]) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Zones (only if explicitly configured)
        for zone in self.config.zones:
            pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            
            is_violated = any(e.get("metadata", {}).get("zone_name") == zone.name for e in events)
            
            fill_color = (0, 0, 220) if is_violated else tuple(zone.color)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
            
            border_color = (0, 0, 255) if is_violated else tuple(zone.color)
            cv2.polylines(annotated, [pts], True, border_color, 2 if not is_violated else 3)
            
            if len(zone.polygon) > 0:
                zx, zy = zone.polygon[0]
                badge_text = f"ZONE: {zone.name} [{zone.type}]"
                cv2.putText(annotated, badge_text, (zx, max(20, zy - 8)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)

        # 2. Draw Spatial Crowd Hotspot Clusters
        for cl in crowd_clusters:
            cx1, cy1, cx2, cy2 = cl["bbox"]
            cluster_overlay = annotated.copy()
            cv2.rectangle(cluster_overlay, (cx1, cy1), (cx2, cy2), (0, 140, 255), -1)
            cv2.addWeighted(cluster_overlay, 0.18, annotated, 0.82, 0, annotated)
            
            cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)
            c_tag = f"CROWD HOTSPOT: {cl['count']} Persons"
            cv2.putText(annotated, c_tag, (cx1 + 6, max(25, cy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        # 3. Draw Tracked Pedestrians
        for trk in tracks:
            x1, y1, x2, y2 = trk.bbox
            tid = trk.track_id
            
            is_violator = any(tid in e.get("track_ids", []) and e.get("event_type") == "SAFETY_VIOLATION" for e in events)
            is_anomaly = any(tid in e.get("track_ids", []) and e.get("event_type") == "SUSPICIOUS_ACTIVITY" for e in events)
            
            if is_violator:
                box_color = (0, 0, 255)  # Red
                label_prefix = f"VIOLATION #{tid}"
            elif is_anomaly:
                box_color = (0, 165, 255) # Orange
                label_prefix = f"ANOMALY #{tid}"
            else:
                box_color = (0, 220, 0)  # Green
                label_prefix = f"ID #{tid}"

            if len(trk.history) > 1:
                pts = np.array(trk.history, dtype=np.int32)
                cv2.polylines(annotated, [pts], False, box_color, 1)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            
            label = f"{label_prefix}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(annotated, (x1, max(0, y1 - 18)), (x1 + t_size[0] + 4, max(18, y1)), box_color, -1)
            cv2.putText(annotated, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 4. Top HUD Status Bar
        hud_h = 42
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 22), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Camera ID & FPS
        cv2.putText(annotated, f"{self.camera_id} - {self.config.name}", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(annotated, f"FPS: {self.video_processor.current_fps:.1f}", (w - 110, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

        # People Count & Crowd Badge
        crowd_color = (0, 255, 0) if crowd_status == "NORMAL" else ((0, 200, 255) if crowd_status == "MODERATE" else (0, 0, 255))
        crowd_hud = f"PEOPLE: {person_count} [{crowd_status}]"
        cv2.putText(annotated, crowd_hud, (w // 2 - 90, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, crowd_color, 2)

        # 5. Active Alert Banner if any
        if alerts:
            alert_msg = f"ALERT: {alerts[-1]['message']}"
            cv2.rectangle(annotated, (0, h - 35), (w, h), (0, 0, 200), -1)
            cv2.putText(annotated, alert_msg, (20, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    async def run_loop(self):
        self.is_running = True
        logger.info(f"Pipeline started for {self.camera_id}")
        
        frame_interval = 1.0 / max(1, self.config.fps_limit)
        
        while self.is_running:
            loop_start = time.time()
            self.frame_index += 1
            
            # Step 1: Read Frame (OpenCV)
            success, frame = self.video_processor.read_frame()
            if not success or frame is None:
                await asyncio.sleep(0.04)
                continue
                
            self.latest_frame = frame
            
            # Decide whether to run full YOLO inference or fast tracker propagation
            should_run_yolo = (self.frame_index % self.inference_interval == 0)
            
            if should_run_yolo:
                # Step 2: YOLO Person Detection
                detections = self.detector.detect(frame, person_only=True)
                
                # Step 3: Multi-Object Tracking
                tracks = self.tracker.update(detections)
                
                # Step 4: Event Engine Processing
                events, crowd_status, person_count, crowd_clusters = self.event_engine.process(tracks)
                
                # Step 5: Alert Engine
                alerts = self.alert_engine.process_events(events)
                
                # Step 6: Database Persistence
                for e in events:
                    await db_service.save_event(e)
                for a in alerts:
                    await db_service.save_alert(a)
                    
                # Cache results for intermediate fast frames
                self.cached_tracks = tracks
                self.cached_events = events
                self.cached_alerts = alerts
                self.cached_clusters = crowd_clusters
                self.cached_crowd_status = crowd_status
                self.cached_person_count = person_count
            else:
                # Fast frame: extrapolate tracker positions without heavy YOLO run
                tracks = self.tracker.predict_all()
                events = self.cached_events
                alerts = self.cached_alerts
                crowd_clusters = self.cached_clusters
                crowd_status = self.cached_crowd_status
                person_count = len(tracks)

            # Step 7: Render Visual Annotations
            annotated_frame = self.draw_annotations(
                frame, tracks, events, alerts, crowd_status, person_count, crowd_clusters
            )
            
            # Step 8: Encode JPEG for Stream (quality 75 for fast network encoding)
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                self.latest_annotated_jpeg = buffer.tobytes()
                
            # Step 9: Update Telemetry State
            self.latest_telemetry = {
                "camera_id": self.camera_id,
                "name": self.config.name,
                "location": self.config.location,
                "status": "ACTIVE",
                "fps": self.video_processor.current_fps,
                "people_count": person_count,
                "crowd_status": crowd_status,
                "crowd_clusters": crowd_clusters,
                "active_events": events,
                "recent_alerts": alerts,
                "tracks_count": len(tracks),
                "zones": [z.model_dump() if hasattr(z, 'model_dump') else z.__dict__ for z in self.config.zones]
            }
            
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def start(self):
        if not self.is_running:
            self._task = asyncio.create_task(self.run_loop())

    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
        self.video_processor.release()

class SurveillancePipelineManager:
    def __init__(self):
        self.detector = YOLODetector(AppConfig.YOLO_MODEL, AppConfig.CONFIDENCE_THRESHOLD)
        self.pipelines: Dict[str, CameraPipeline] = {}

    def initialize_cameras(self, camera_configs: List[CameraConfig]):
        for cfg in camera_configs:
            pipeline = CameraPipeline(cfg, self.detector)
            self.pipelines[cfg.camera_id] = pipeline
            pipeline.start()

    def get_pipeline(self, camera_id: str) -> Optional[CameraPipeline]:
        return self.pipelines.get(camera_id)

    def get_all_telemetry(self) -> List[Dict[str, Any]]:
        return [p.latest_telemetry for p in self.pipelines.values()]

    def stop_all(self):
        for p in self.pipelines.values():
            p.stop()

pipeline_manager = SurveillancePipelineManager()
