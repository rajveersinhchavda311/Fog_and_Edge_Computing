"""
fog_server.py — the fog layer (tier 2 of the 3-tier architecture).

Receives telemetry from all edge nodes, maintains per-zone state, runs the
Pressure Cascade Detection Algorithm after every ingest, serves alert decisions
back to edge nodes, and pushes live state to the dashboard over WebSocket.

The fog server is what makes preemptive alerting possible: an individual edge
node only sees its own camera, but the fog server sees every zone at once and
can therefore warn a zone about a wave that has not reached it yet.

Run with:
    python -m uvicorn fog_server:app --host 0.0.0.0 --port 8000
or:
    python fog_server.py
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

import config
from cascade_detector import detect_cascade
from zone_manager import ZoneManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fog_server")

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")


zone_manager = ZoneManager()

# Most recent cascade detection result. Read by GET /cascade, GET /alert/{zone},
# and the WebSocket broadcast loop; rewritten by every POST /ingest.
latest_cascade: dict = {
    "cascade_detected": False,
    "cascade_score": 0.0,
    "zone_alerts": {},
    "zone_details": {},
    "surge_zones": [],
    "wave_pairs": [],
    "preemptive_alerts": {},
    "timestamp": 0.0,
}


class ConnectionManager:
    """Tracks connected dashboard WebSocket clients.

    Broadcast tolerates individual client failures: a dashboard that closed its
    tab mid-send must not interrupt delivery to the others, and must not raise
    into the broadcast loop and kill it.
    """

    def __init__(self):
        self.active: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active.append(websocket)
        logger.info("Dashboard connected (%d active)", len(self.active))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active:
                self.active.remove(websocket)
        logger.info("Dashboard disconnected (%d active)", len(self.active))

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            targets = list(self.active)

        dead: List[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self.active:
                        self.active.remove(ws)
            logger.info("Pruned %d dead dashboard connection(s)", len(dead))


manager = ConnectionManager()


def build_state_payload() -> dict:
    """The full system snapshot sent to dashboards over WebSocket and /zones."""
    return {
        "zones": zone_manager.snapshot_all_zones(),
        "cascade": latest_cascade,
        "timestamp": time.time(),
    }


async def broadcast_loop() -> None:
    """Push system state to all connected dashboards on a fixed cadence."""
    while True:
        try:
            await asyncio.sleep(config.WS_BROADCAST_INTERVAL_SEC)
            if manager.active:
                await manager.broadcast(build_state_payload())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Never let one bad broadcast cycle kill the loop for good.
            logger.warning("Broadcast loop error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(broadcast_loop())
    logger.info(
        "Fog server started. Zones configured: %s",
        ", ".join(sorted(config.ZONE_ADJACENCY.keys())),
    )
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info("Fog server stopped.")


app = FastAPI(title="CrowdShield Fog Server", lifespan=lifespan)


# ===== ENDPOINTS =====

@app.post("/ingest")
async def ingest(packet: dict):
    """Receive telemetry from an edge node and re-run cascade detection.

    Detection runs on every ingest rather than on a timer so that alert state is
    always derived from the most recent data any zone has reported — an edge
    node polling /alert immediately after POSTing sees a decision that already
    accounts for its own reading.

    Accepts a plain dict rather than a strict model so that a single malformed
    field from one edge node degrades that packet only, instead of returning 422
    and leaving the node retrying a packet that will never be accepted.
    """
    global latest_cascade

    required = ("zone_id", "density_avg", "timestamp")
    missing = [k for k in required if k not in packet]
    if missing:
        logger.warning("Rejected telemetry packet missing %s: %s", missing, packet)
        return JSONResponse(
            {"status": "error", "message": f"missing fields: {missing}"},
            status_code=400,
        )

    normalized = {
        "zone_id": str(packet["zone_id"]),
        "count": int(packet.get("count", 0)),
        "density_avg": float(packet["density_avg"]),
        "density_peak": float(packet.get("density_peak", 0.0)),
        "flow_x": float(packet.get("flow_x", 0.0)),
        "flow_y": float(packet.get("flow_y", 0.0)),
        "flow_magnitude": float(packet.get("flow_magnitude", 0.0)),
        "timestamp": float(packet["timestamp"]),
    }

    zone_manager.update(normalized)
    latest_cascade = detect_cascade(zone_manager)

    zone_id = normalized["zone_id"]
    return {
        "status": "ok",
        "zone_id": zone_id,
        "alert_level": latest_cascade["zone_alerts"].get(zone_id, "GREEN"),
        "cascade_detected": latest_cascade["cascade_detected"],
    }


@app.get("/zones")
async def get_zones():
    """Current state of every zone, for the dashboard."""
    return {"zones": zone_manager.snapshot_all_zones(), "timestamp": time.time()}


@app.get("/alert/{zone_id}")
async def get_alert(zone_id: str):
    """Current alert level for one zone. Polled by that zone's edge node.

    Returns GREEN for an unknown zone rather than 404: an edge node that starts
    before its first telemetry is processed should get a valid quiet response,
    not an error it has to special-case.
    """
    level = latest_cascade["zone_alerts"].get(zone_id, "GREEN")
    preemptive = latest_cascade["preemptive_alerts"].get(zone_id)

    eta: Optional[float] = None
    message = ""

    if preemptive is not None:
        eta = preemptive["eta_seconds"]
        message = (
            f"Preemptive alert: pressure wave {preemptive['triggered_by_wave']} "
            f"predicted to reach {zone_id} in {eta:.1f}s"
        )
    elif level == "RED":
        message = f"Confirmed pressure wave involving {zone_id}"
    elif level == "YELLOW":
        detail = latest_cascade["zone_details"].get(zone_id, {})
        confidence = detail.get("surge_confidence")
        message = (
            f"Surge buildup detected in {zone_id}"
            + (f" (confidence {confidence})" if confidence is not None else "")
        )

    return {"zone_id": zone_id, "level": level, "eta": eta, "message": message}


@app.get("/cascade")
async def get_cascade():
    """Most recent full cascade detection result, for the dashboard."""
    return latest_cascade


@app.get("/history/{zone_id}")
async def get_history(zone_id: str):
    """Density time series for one zone, for the dashboard chart."""
    history = zone_manager.get_history(zone_id)
    return {
        "zone_id": zone_id,
        "history": history,
        "count": len(history),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard page."""
    try:
        with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except OSError:
        return HTMLResponse(
            "<h1>Dashboard not found</h1>"
            f"<p>Expected at <code>{os.path.abspath(DASHBOARD_PATH)}</code></p>",
            status_code=404,
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Push zone and cascade state to a dashboard every WS_BROADCAST_INTERVAL_SEC.

    The initial snapshot is sent immediately on connect so a freshly opened
    dashboard renders at once instead of staying blank until the next tick.
    """
    await manager.connect(websocket)
    try:
        await websocket.send_json(build_state_payload())
        while True:
            # Keeps the connection open and consumes any client messages.
            # The periodic push itself comes from broadcast_loop().
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
        await manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.FOG_HOST, port=config.FOG_PORT)
