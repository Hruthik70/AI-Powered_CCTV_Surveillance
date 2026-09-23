import uvicorn
import os
import sys
import torch

def main():
    print("=" * 60)
    print("   AI-POWERED CCTV SURVEILLANCE SYSTEM (Sentinel-AI)")
    print("   Pipeline: OpenCV -> YOLOv8 -> Tracker -> Engines -> FastAPI")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"🚀 HARDWARE ACCELERATION: GPU ENABLED ({gpu_name} | {vram} GB VRAM)")
    else:
        print("⚠️ HARDWARE ACCELERATION: RUNNING ON CPU (CUDA not detected in current Python runtime)")

    print("📍 Dashboard URL: http://localhost:8000")
    print("📍 API Docs:     http://localhost:8000/docs")
    print("📍 WebSocket:    ws://localhost:8000/ws/telemetry")
    print("=" * 60)
    
    video_path = os.path.join("videos", "crowd.mp4")
    if not os.path.exists(video_path):
        from videos.generate_sample import generate_synthetic_cctv_video
        generate_synthetic_cctv_video(video_path)

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
