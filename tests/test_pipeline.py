import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
import cv2
import asyncio
from vision.video_processor import VideoProcessor
from detection.yolo_detector import YOLODetector, Detection
from detection.tracker import MultiObjectTracker
from engines.crowd_engine import CrowdEngine
from engines.safety_engine import SafetyEngine
from engines.activity_engine import ActivityEngine
from engines.event_engine import EventEngine
from engines.alert_engine import AlertEngine
from backend.config import ZoneConfig, CameraConfig
from database.db_service import db_service

class TestCCTVSurveillancePipeline(unittest.TestCase):

    def test_phase1_video_processor(self):
        processor = VideoProcessor("videos/crowd.mp4", loop=True)
        success, frame = processor.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.shape), 3)
        processor.release()

    def test_phase2_and_3_yolo_detection(self):
        detector = YOLODetector(conf_threshold=0.3)
        # Create a test synthetic image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(img, person_only=True)
        self.assertIsInstance(dets, list)

    def test_phase4_crowd_engine(self):
        crowd_engine = CrowdEngine(crowd_threshold=15, warning_threshold=10)
        status, sev, _ = crowd_engine.evaluate(5)
        self.assertEqual(status, "NORMAL")
        status, sev, _ = crowd_engine.evaluate(12)
        self.assertEqual(status, "MODERATE")
        status, sev, _ = crowd_engine.evaluate(20)
        self.assertEqual(status, "OVERCROWDED")
        self.assertEqual(sev, "HIGH")

    def test_phase5_tracking(self):
        tracker = MultiObjectTracker()
        det1 = Detection(100, 100, 150, 200, 0.9, 0, "person")
        tracks_frame1 = tracker.update([det1])
        self.assertEqual(len(tracks_frame1), 1)
        tid = tracks_frame1[0].track_id
        
        # Next frame with slightly moved person
        det2 = Detection(104, 102, 154, 202, 0.91, 0, "person")
        tracks_frame2 = tracker.update([det2])
        self.assertEqual(len(tracks_frame2), 1)
        self.assertEqual(tracks_frame2[0].track_id, tid) # Persistent ID preserved

    def test_phase6_and_7_safety_zones(self):
        zone = ZoneConfig(
            name="Restricted Vault",
            type="RESTRICTED",
            polygon=[[100, 100], [300, 100], [300, 300], [100, 300]]
        )
        safety_engine = SafetyEngine([zone])
        
        # Person inside zone
        tracker = MultiObjectTracker()
        det_inside = Detection(150, 150, 180, 250, 0.9, 0, "person") # feet at (165, 250)
        tracks = tracker.update([det_inside])
        violations = safety_engine.evaluate(tracks)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["type"], "SAFETY_VIOLATION")

    def test_phase10_alert_cooldown_deduplication(self):
        alert_engine = AlertEngine("CAM-01", cooldown_seconds=5.0)
        event = {
            "event_id": "evt-1",
            "camera_id": "CAM-01",
            "event_type": "OVERCROWDING",
            "severity": "HIGH",
            "message": "Overcrowding detected",
            "track_ids": [1, 2, 3],
            "metadata": {"count": 25}
        }
        
        # First occurrence triggers an alert
        alerts1 = alert_engine.process_events([event])
        self.assertEqual(len(alerts1), 1)
        
        # Second immediate occurrence within cooldown produces NO duplicate alert
        alerts2 = alert_engine.process_events([event])
        self.assertEqual(len(alerts2), 0)

if __name__ == "__main__":
    unittest.main()
