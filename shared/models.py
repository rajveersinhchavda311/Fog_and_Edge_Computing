"""
Shared data models for the fog/edge system.

These Pydantic models define the contract between edge nodes and the fog server.
Keeping them in shared/ means the edge node, fog server, and demo harness all
agree on the telemetry packet shape without duplicating field definitions.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TelemetryPacket(BaseModel):
    """One telemetry reading from a single edge node, POSTed to /ingest.

    Matches exactly what edge/edge_node.py build_telemetry_packet() produces.
    """

    zone_id: str
    count: int
    density_avg: float
    density_peak: float
    flow_x: float
    flow_y: float
    flow_magnitude: float
    timestamp: float


class HistoryEntry(BaseModel):
    """One point in a zone's rolling history buffer."""

    timestamp: float
    density_avg: float
    density_peak: float
    count: int
    flow_x: float
    flow_y: float
    flow_magnitude: float


class ZoneState(BaseModel):
    """Current state of one zone, as returned by GET /zones."""

    zone_id: str
    latest: Optional[HistoryEntry] = None
    history_length: int = 0
    adjacent_zones: List[str] = Field(default_factory=list)
    surge_start_time: Optional[float] = None


class AlertResponse(BaseModel):
    """Response from GET /alert/{zone_id}, polled by edge nodes.

    `level` is GREEN / YELLOW / RED. `eta` is seconds until a predicted surge
    arrival, present only for preemptive alerts from Component 3.
    """

    zone_id: str
    level: str
    eta: Optional[float] = None
    message: str = ""


class WavePair(BaseModel):
    """A confirmed pressure wave travelling from one zone to an adjacent one."""

    from_zone: str = Field(alias="from")
    to_zone: str = Field(alias="to")
    correlation: float
    lag_seconds: float
    flow_aligned: bool

    class Config:
        populate_by_name = True


class ZoneDetail(BaseModel):
    """Per-zone Component 1 output."""

    surge_confidence: float
    density: float
    is_surging: bool


class PreemptiveAlert(BaseModel):
    """Component 3 output: a predicted future surge in a not-yet-surging zone."""

    eta_seconds: float
    triggered_by_wave: str
    propagation_delay: float


class CascadeResult(BaseModel):
    """Full output of one run of the Pressure Cascade Detection Algorithm."""

    cascade_detected: bool = False
    cascade_score: float = 0.0
    zone_alerts: Dict[str, str] = Field(default_factory=dict)
    zone_details: Dict[str, ZoneDetail] = Field(default_factory=dict)
    surge_zones: List[str] = Field(default_factory=list)
    wave_pairs: List[WavePair] = Field(default_factory=list)
    preemptive_alerts: Dict[str, PreemptiveAlert] = Field(default_factory=dict)
    timestamp: float = 0.0
