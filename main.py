import uvicorn
import os
import sys

def main():
    print("=" * 60)
    print("   AI-POWERED CCTV SURVEILLANCE SYSTEM (Sentinel-AI)")
    print("   Pipeline: OpenCV -> YOLOv8 -> Tracker -> Engines -> FastAPI")
    print("=" * 60)
    print("📍 Dashboard URL: http://localhost:8000")
    print("📍 API Docs:     http://localhost:8000/docs")
    print("📍 WebSocket:    ws://localhost:8000/ws/telemetry")
    print("=" * 60)
    
    # Ensure default synthetic video exists if needed
    video_path = os.path.join("videos", "crowd.mp4")
    if not os.path.exists(video_path):
        print("Generating initial simulated crowd CCTV video in videos/crowd.mp4...")
        from videos.generate_sample import generate_synthetic_cctv_video
        generate_synthetic_cctv_video(video_path)

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
