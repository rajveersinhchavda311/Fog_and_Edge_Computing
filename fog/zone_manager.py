"""
zone_manager.py — per-zone telemetry state for the fog server.

Holds, for every zone: the latest telemetry packet, a rolling history buffer of
the last HISTORY_BUFFER_SIZE data points (60 seconds at 2-second intervals), and
the start time of the current surge episode.

The cascade detector reads exclusively through this class — it never touches
raw request data. That separation keeps the detection algorithm testable against
synthetic histories without needing a live HTTP server.
"""

import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

import config


class ZoneManager:
    """Thread-safe store of per-zone telemetry history and surge episode state.

    Thread safety matters here because FastAPI serves requests concurrently:
    an /ingest write can overlap a /zones or /cascade read. A single reentrant
    lock guards all mutation and all snapshot reads. It is an RLock rather than
    a plain Lock so that composite methods (e.g. get_all_zone_ids() called from
    within another locked method) do not self-deadlock.
    """

    def __init__(self, adjacency: Optional[Dict[str, List[str]]] = None):
        self.adjacency = adjacency if adjacency is not None else config.ZONE_ADJACENCY

        # zone_id -> deque of history dicts (oldest first, newest last)
        self.histories: Dict[str, deque] = {}
        # zone_id -> latest history dict
        self.latest: Dict[str, dict] = {}
        # zone_id -> timestamp when the current surge episode began (None if not surging)
        self.surge_start: Dict[str, Optional[float]] = {}

        self._lock = threading.RLock()

    # ---- ingest ----

    def update(self, packet: dict) -> None:
        """Record one telemetry packet from an edge node.

        Appends to the zone's rolling history, updates the latest snapshot, and
        maintains the surge episode start time.
        """
        zone_id = packet["zone_id"]

        entry = {
            "timestamp": packet["timestamp"],
            "density_avg": packet["density_avg"],
            "density_peak": packet["density_peak"],
            "count": packet["count"],
            "flow_x": packet["flow_x"],
            "flow_y": packet["flow_y"],
            "flow_magnitude": packet["flow_magnitude"],
        }

        with self._lock:
            if zone_id not in self.histories:
                self.histories[zone_id] = deque(maxlen=config.HISTORY_BUFFER_SIZE)
                self.surge_start[zone_id] = None

            self.histories[zone_id].append(entry)
            self.latest[zone_id] = entry
            self._update_surge_episode(zone_id, entry)

    def _update_surge_episode(self, zone_id: str, entry: dict) -> None:
        """Track when this zone's current surge episode began.

        Rising edge (density crosses the threshold from below): record the start
        timestamp. Falling edge (density drops back below): clear it, so the next
        rise is treated as a new, separate episode rather than continuing the old
        one. Caller must hold the lock.
        """
        above = entry["density_avg"] > config.SURGE_START_DENSITY_THRESHOLD

        if above and self.surge_start.get(zone_id) is None:
            self.surge_start[zone_id] = entry["timestamp"]
        elif not above:
            self.surge_start[zone_id] = None

    # ---- reads used by the cascade detector ----

    def get_all_zone_ids(self) -> List[str]:
        """All zones known to the system: those configured in the adjacency map
        plus any that have actually reported telemetry.

        Union rather than just the adjacency map so an unconfigured zone that
        starts reporting still appears in the dashboard and is still evaluated,
        and so configured-but-silent zones are still listed as GREEN rather than
        vanishing from the output entirely.
        """
        with self._lock:
            return sorted(set(self.adjacency.keys()) | set(self.histories.keys()))

    def get_adjacent_zones(self, zone_id: str) -> List[str]:
        return list(self.adjacency.get(zone_id, []))

    def get_density_timeseries(self, zone_id: str) -> np.ndarray:
        """Density_avg values for a zone, oldest first, as a float numpy array.

        Returns a numpy array (not a list) because every consumer — DTW,
        normalization, cross-correlation — does vectorized arithmetic on it.
        Returns an empty float array for an unknown zone so callers can uniformly
        use len() and slicing without a None check.
        """
        with self._lock:
            history = self.histories.get(zone_id)
            if not history:
                return np.array([], dtype=float)
            return np.array([e["density_avg"] for e in history], dtype=float)

    def get_flow_vector(self, zone_id: str) -> Tuple[float, float]:
        """Latest (flow_x, flow_y) for a zone; (0.0, 0.0) if none reported yet.

        Returning a zero vector rather than None means the flow-alignment dot
        product in Component 2 evaluates to 0 for an unknown zone, which fails
        the `> 0` aligned test — so a zone with no flow data can never falsely
        confirm a wave.
        """
        with self._lock:
            entry = self.latest.get(zone_id)
            if entry is None:
                return (0.0, 0.0)
            return (entry["flow_x"], entry["flow_y"])

    def get_surge_start_time(self, zone_id: str) -> Optional[float]:
        """Timestamp when this zone first crossed the surge density threshold in
        the current episode, or None if it is not currently surging."""
        with self._lock:
            return self.surge_start.get(zone_id)

    def get_latest(self, zone_id: str) -> Optional[dict]:
        with self._lock:
            entry = self.latest.get(zone_id)
            return dict(entry) if entry is not None else None

    def get_history(self, zone_id: str) -> List[dict]:
        """Full rolling history for a zone, oldest first, as plain dicts."""
        with self._lock:
            history = self.histories.get(zone_id)
            if not history:
                return []
            return [dict(e) for e in history]

    # ---- reads used by the HTTP API ----

    def snapshot_all_zones(self) -> List[dict]:
        """State of every known zone, for GET /zones and the WebSocket push."""
        with self._lock:
            return [
                {
                    "zone_id": zone_id,
                    "latest": self.get_latest(zone_id),
                    "history_length": len(self.histories.get(zone_id, ())),
                    "adjacent_zones": self.get_adjacent_zones(zone_id),
                    "surge_start_time": self.surge_start.get(zone_id),
                }
                for zone_id in self.get_all_zone_ids()
            ]

    def reset(self) -> None:
        """Clear all state. Used by tests and the demo harness."""
        with self._lock:
            self.histories.clear()
            self.latest.clear()
            self.surge_start.clear()
