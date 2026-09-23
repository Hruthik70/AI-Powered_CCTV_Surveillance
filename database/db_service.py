import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.config import AppConfig

logger = logging.getLogger("DatabaseService")

class DatabaseService:
    def __init__(self, uri: str = AppConfig.MONGO_URI, db_name: str = AppConfig.DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.is_connected = False
        
        # In-memory fallback stores (if MongoDB is offline)
        self.mem_cameras: Dict[str, Dict[str, Any]] = {}
        self.mem_events: List[Dict[str, Any]] = []
        self.mem_alerts: List[Dict[str, Any]] = []
        self._max_history = 1000

    async def connect(self):
        try:
            import motor.motor_asyncio
            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                self.uri, serverSelectionTimeoutMS=2000
            )
            # Test connection
            await self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.is_connected = True
            logger.info("Connected to MongoDB successfully.")
        except Exception as e:
            self.is_connected = False
            logger.warning(f"MongoDB unavailable ({e}). Using robust In-Memory Database store.")

    async def save_camera(self, camera_data: Dict[str, Any]):
        camera_id = camera_data.get("camera_id")
        self.mem_cameras[camera_id] = camera_data
        if self.is_connected and self.db is not None:
            try:
                await self.db.cameras.update_one(
                    {"camera_id": camera_id},
                    {"$set": camera_data},
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Error saving camera to MongoDB: {e}")

    async def get_cameras(self) -> List[Dict[str, Any]]:
        if self.is_connected and self.db is not None:
            try:
                cursor = self.db.cameras.find({}, {"_id": 0})
                return await cursor.to_list(length=100)
            except Exception as e:
                logger.error(f"Error reading cameras from MongoDB: {e}")
        return list(self.mem_cameras.values())

    async def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.db is not None:
            try:
                cam = await self.db.cameras.find_one({"camera_id": camera_id}, {"_id": 0})
                if cam:
                    return cam
            except Exception as e:
                logger.error(f"Error reading camera {camera_id} from MongoDB: {e}")
        return self.mem_cameras.get(camera_id)

    async def save_event(self, event_data: Dict[str, Any]):
        self.mem_events.append(event_data)
        if len(self.mem_events) > self._max_history:
            self.mem_events.pop(0)
            
        if self.is_connected and self.db is not None:
            try:
                doc = {k: v for k, v in event_data.items() if k != "_id"}
                await self.db.events.insert_one(doc)
            except Exception as e:
                logger.error(f"Error saving event to MongoDB: {e}")

    async def get_events(self, camera_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if self.is_connected and self.db is not None:
            try:
                query = {"camera_id": camera_id} if camera_id else {}
                cursor = self.db.events.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
                return await cursor.to_list(length=limit)
            except Exception as e:
                logger.error(f"Error reading events from MongoDB: {e}")
        
        filtered = [e for e in self.mem_events if not camera_id or e.get("camera_id") == camera_id]
        return list(reversed(filtered))[:limit]

    async def save_alert(self, alert_data: Dict[str, Any]):
        self.mem_alerts.append(alert_data)
        if len(self.mem_alerts) > self._max_history:
            self.mem_alerts.pop(0)
            
        if self.is_connected and self.db is not None:
            try:
                doc = {k: v for k, v in alert_data.items() if k != "_id"}
                await self.db.alerts.insert_one(doc)
            except Exception as e:
                logger.error(f"Error saving alert to MongoDB: {e}")

    async def get_alerts(self, camera_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if self.is_connected and self.db is not None:
            try:
                query: Dict[str, Any] = {}
                if camera_id:
                    query["camera_id"] = camera_id
                if status:
                    query["status"] = status
                cursor = self.db.alerts.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
                return await cursor.to_list(length=limit)
            except Exception as e:
                logger.error(f"Error reading alerts from MongoDB: {e}")
                
        filtered = [
            a for a in self.mem_alerts 
            if (not camera_id or a.get("camera_id") == camera_id) and 
               (not status or a.get("status") == status)
        ]
        return list(reversed(filtered))[:limit]

    async def update_alert_status(self, alert_id: str, status: str) -> bool:
        updated = False
        for a in self.mem_alerts:
            if a.get("alert_id") == alert_id:
                a["status"] = status
                updated = True
                
        if self.is_connected and self.db is not None:
            try:
                res = await self.db.alerts.update_one(
                    {"alert_id": alert_id},
                    {"$set": {"status": status}}
                )
                return res.modified_count > 0
            except Exception as e:
                logger.error(f"Error updating alert status in MongoDB: {e}")
        return updated

    async def get_statistics(self) -> Dict[str, Any]:
        total_events = len(self.mem_events)
        total_alerts = len(self.mem_alerts)
        active_alerts = len([a for a in self.mem_alerts if a.get("status") == "ACTIVE"])
        
        if self.is_connected and self.db is not None:
            try:
                total_events = await self.db.events.count_documents({})
                total_alerts = await self.db.alerts.count_documents({})
                active_alerts = await self.db.alerts.count_documents({"status": "ACTIVE"})
            except Exception as e:
                logger.error(f"Error reading stats from MongoDB: {e}")
                
        return {
            "total_events": total_events,
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "db_connected": self.is_connected
        }

db_service = DatabaseService()
