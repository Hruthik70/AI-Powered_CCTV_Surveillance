import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
import os
import torch

logger = logging.getLogger("YOLODetector")

# Optimize CPU multi-threading for PyTorch
try:
    if not torch.cuda.is_available():
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(max(1, min(4, num_cores)))
except Exception:
    pass

class Detection:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float, class_id: int, class_name: str):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.confidence = round(float(confidence), 3)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
        
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
        
    @property
    def foot_position(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": list(self.center),
            "foot_position": list(self.foot_position)
        }

class YOLODetector:
    """
    High-Performance YOLO Detector with CPU multi-threading and optimized inference sizing.
    """
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.22, imgsz: int = 640):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {self.model_name}...")
            self.model = YOLO(self.model_name)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load YOLO model {self.model_name} ({e}). Falling back to yolov8n.pt")
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")
            except Exception:
                self.model = None

    def detect(self, frame: np.ndarray, person_only: bool = True) -> List[Detection]:
        if frame is None:
            return []
            
        detections: List[Detection] = []
        
        if self.model is not None:
            try:
                # Fast inference at 640 for high CPU throughput
                results = self.model(
                    frame, 
                    conf=self.conf_threshold, 
                    classes=[0] if person_only else None, 
                    imgsz=self.imgsz,
                    verbose=False
                )
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = r.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0].item())
                        
                        if person_only and cls_id != 0:
                            continue
                            
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append(Detection(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name
                        ))
                return detections
            except Exception as e:
                logger.error(f"YOLO inference error: {e}")
                
        return self._fallback_contour_detect(frame)

    def _fallback_contour_detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for c in contours:
            if cv2.contourArea(c) > 300:
                x, y, bw, bh = cv2.boundingRect(c)
                if 0.5 < bh / (bw + 1e-5) < 3.5 and bh > 25:
                    results.append(Detection(
                        x1=x, y1=y, x2=x+bw, y2=y+bh,
                        confidence=0.88,
                        class_id=0,
                        class_name="person"
                    ))
        return results
