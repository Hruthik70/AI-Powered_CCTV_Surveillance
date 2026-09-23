import cv2
import time
import os
import logging
from typing import Generator, Tuple, Optional
import numpy as np

logger = logging.getLogger("VideoProcessor")

class VideoProcessor:
    """
    Phase 1: Video Input & Frame Processor
    Handles video capture from file, RTSP, or camera index.
    Provides FPS calculation, resolution normalization, timestamping, and frame looping.
    """
    def __init__(self, source: str = "videos/crowd.mp4", target_fps: int = 30, loop: bool = True):
        self.source = source
        self.target_fps = target_fps
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0
        self._fps_history = []
        self._last_frame_time = time.time()
        self.width = 0
        self.height = 0
        self.is_open = False
        self._init_capture()

    def _init_capture(self):
        if self.cap is not None:
            self.cap.release()
            
        # If the file doesn't exist, try to generate synthetic crowd footage
        if isinstance(self.source, str) and not self.source.isdigit() and not self.source.startswith("rtsp://") and not self.source.startswith("http://"):
            if not os.path.exists(self.source):
                logger.info(f"Video file {self.source} not found. Generating default synthetic CCTV clip...")
                from videos.generate_sample import generate_synthetic_cctv_video
                generate_synthetic_cctv_video(self.source)
                
        # Parse numeric camera index or stream string
        src = int(self.source) if str(self.source).isdigit() else self.source
        self.cap = cv2.VideoCapture(src)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            self.is_open = False
            return
            
        self.is_open = True
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
        logger.info(f"Video source '{self.source}' opened ({self.width}x{self.height})")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open or self.cap is None:
            self._init_capture()
            if not self.is_open:
                return False, None
                
        ret, frame = self.cap.read()
        
        if not ret:
            if self.loop:
                # Loop back to beginning for recorded CCTV demo
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return False, None
            else:
                return False, None
                
        self.frame_count += 1
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps_history.append(instant_fps)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            self.current_fps = round(sum(self._fps_history) / len(self._fps_history), 1)
            
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            logger.info("Video processor released.")

if __name__ == "__main__":
    # Phase 1 standalone verification
    logging.basicConfig(level=logging.INFO)
    processor = VideoProcessor("videos/crowd.mp4")
    print("Testing VideoProcessor standalone playback. Press 'q' in window to exit.")
    
    while True:
        success, frame = processor.read_frame()
        if not success:
            break
            
        # Draw frame info HUD
        cv2.putText(frame, f"FPS: {processor.current_fps}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {processor.width}x{processor.height}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
        cv2.imshow("CCTV Video Feed (Phase 1)", frame)
        if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
            break
            
    processor.release()
    cv2.destroyAllWindows()
