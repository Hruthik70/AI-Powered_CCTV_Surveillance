import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import AppConfig, CameraConfig, ZoneConfig
from backend.schemas import EventModel, AlertModel, CameraModel
from database.db_service import db_service
from backend.pipeline_manager import pipeline_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SurveillanceAPI")

# Connected WebSocket clients
active_websockets: List[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect DB and start surveillance pipelines
    logger.info("Initializing CCTV Surveillance System...")
    await db_service.connect()
    
    # Save default cameras to DB
    for cam in AppConfig.DEFAULT_CAMERAS:
        cam_dict = cam.model_dump() if hasattr(cam, 'model_dump') else cam.__dict__
        await db_service.save_camera(cam_dict)
        
    pipeline_manager.initialize_cameras(AppConfig.DEFAULT_CAMERAS)
    
    # Background broadcaster for WebSocket clients
    broadcast_task = asyncio.create_task(websocket_broadcast_loop())
    
    yield
    
    # Shutdown
    logger.info("Shutting down surveillance pipelines...")
    broadcast_task.cancel()
    pipeline_manager.stop_all()

app = FastAPI(
    title="AI-Powered CCTV Surveillance System",
    description="Real-time multi-camera surveillance pipeline with YOLO detection, object tracking, crowd analysis, polygonal safety zones, and alert engine.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def websocket_broadcast_loop():
    while True:
        try:
            if active_websockets:
                telemetry = pipeline_manager.get_all_telemetry()
                stats = await db_service.get_statistics()
                payload = {
                    "cameras": telemetry,
                    "statistics": stats
                }
                for ws in list(active_websockets):
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        if ws in active_websockets:
                            active_websockets.remove(ws)
            await asyncio.sleep(0.35)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in websocket broadcaster: {e}")
            await asyncio.sleep(1.0)

# WebSocket Endpoint
@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            # Keep-alive receive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

# REST Endpoints
@app.get("/api/cameras")
async def get_cameras():
    return pipeline_manager.get_all_telemetry()

@app.get("/api/cameras/{camera_id}")
async def get_camera_detail(camera_id: str):
    p = pipeline_manager.get_pipeline(camera_id)
    if not p:
        raise HTTPException(status_code=404, detail="Camera not found")
    return p.latest_telemetry

class UpdateZonesRequest(BaseModel):
    zones: List[ZoneConfig]

@app.post("/api/cameras/{camera_id}/zones")
async def update_camera_zones(camera_id: str, req: UpdateZonesRequest):
    p = pipeline_manager.get_pipeline(camera_id)
    if not p:
        raise HTTPException(status_code=404, detail="Camera not found")
    p.update_zones(req.zones)
    # Persist camera zone update
    cam_data = await db_service.get_camera(camera_id)
    if cam_data:
        cam_data["zones"] = [z.model_dump() for z in req.zones]
        await db_service.save_camera(cam_data)
    return {"status": "success", "message": f"Updated {len(req.zones)} zones for {camera_id}"}

@app.get("/api/events")
async def get_events(camera_id: Optional[str] = None, limit: int = 50):
    return await db_service.get_events(camera_id=camera_id, limit=limit)

@app.get("/api/alerts")
async def get_alerts(camera_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50):
    return await db_service.get_alerts(camera_id=camera_id, status=status, limit=limit)

class AlertStatusUpdate(BaseModel):
    status: str  # ACKNOWLEDGED, RESOLVED

@app.patch("/api/alerts/{alert_id}")
async def update_alert_status(alert_id: str, req: AlertStatusUpdate):
    success = await db_service.update_alert_status(alert_id, req.status)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "success", "alert_id": alert_id, "new_status": req.status}

@app.get("/api/statistics")
async def get_statistics():
    return await db_service.get_statistics()

# Video Streaming MJPEG Endpoint
async def generate_mjpeg_frames(camera_id: str):
    pipeline = pipeline_manager.get_pipeline(camera_id)
    if not pipeline:
        return
    while True:
        if pipeline.latest_annotated_jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + pipeline.latest_annotated_jpeg + b'\r\n')
        await asyncio.sleep(0.04)

@app.get("/api/video/feed/{camera_id}")
async def video_feed(camera_id: str):
    pipeline = pipeline_manager.get_pipeline(camera_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Camera pipeline not found")
    return StreamingResponse(
        generate_mjpeg_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Static Dashboard UI Mount
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI CCTV Surveillance API Running. Please place UI in dashboard/static."}
