"""
Fog server configuration.

Every threshold used by the Pressure Cascade Detection Algorithm lives here so
the algorithm's sensitivity can be tuned without touching detection logic.
Each constant documents what it controls and what happens if it is changed.

Deployment-dependent values (host, port, log path) can be overridden by
environment variable so the same image runs under Docker — where the log must be
written into a mounted volume — and locally, without editing this file.
"""

import os

# ===== SERVER =====

FOG_HOST = os.getenv("FOG_HOST", "0.0.0.0")
FOG_PORT = int(os.getenv("FOG_PORT", "8000"))

# How often the WebSocket broadcast loop pushes state to connected dashboards.
WS_BROADCAST_INTERVAL_SEC = 2.0


# ===== TIME BASE =====

# Seconds between consecutive telemetry data points. Edge nodes POST every 2s
# (edge/config.py TELEMETRY_INTERVAL_SEC), so one history slot == 2 seconds.
# Every lag-to-seconds conversion in the cascade detector multiplies by this.
DATA_INTERVAL = 2  # seconds between data points

# How much history the cascade detector reasons over, in seconds.
CASCADE_HISTORY_WINDOW = 60  # seconds

# Rolling history buffer depth per zone = 60s / 2s = 30 data points.
# Cross-correlation (Component 2) uses the full 30; DTW pattern matching
# (Component 1) uses the most recent 15 (last 30 seconds).
HISTORY_BUFFER_SIZE = CASCADE_HISTORY_WINDOW // DATA_INTERVAL  # 30


# ===== COMPONENT 1 — TEMPORAL PATTERN MATCHING (DTW) =====

# Number of points in the reference pre-stampede S-curve, and therefore the
# number of recent data points compared against it (15 points x 2s = 30s window).
REFERENCE_CURVE_POINTS = 15

# Steepness of the reference sigmoid. Higher = sharper transition from slow rise
# to plateau. 10.0 produces the classic "slow rise -> acceleration -> plateau"
# shape documented in crowd dynamics literature (Helbing et al. 2007, Still 2000).
REFERENCE_CURVE_STEEPNESS = 10.0

# DTW distance is converted to confidence via: 1 - (dtw_distance / MAX_DTW_DISTANCE).
# Larger value = more forgiving (higher confidence for a given distance), so more
# zones register as surging. Smaller = stricter, fewer surge detections.
MAX_DTW_DISTANCE = 5.0

# A zone must exceed this DTW-derived confidence to be considered "surging".
# Raising it reduces false surge detections but risks missing real buildups.
SURGE_CONFIDENCE_THRESHOLD = 0.6

# A zone must ALSO exceed this raw density before it can be called surging.
# This gates out zones whose density curve happens to be S-shaped but is
# occurring at a harmlessly low absolute crowd level.
MIN_DENSITY_THRESHOLD = 0.3

# Minimum history points before Component 1 will evaluate a zone at all.
MIN_POINTS_FOR_SURGE_CHECK = 5


# ===== COMPONENT 2 — CROSS-CORRELATION WAVE DETECTION =====

# Maximum lag (in data points) searched when correlating two adjacent zones.
# 10 points x 2s = +/- 20 seconds of propagation delay considered.
MAX_CORRELATION_LAG = 10

# Two adjacent zones must correlate above this to be called a real pressure
# wave rather than two coincidental independent spikes. Raising it demands
# near-identical curve shapes; lowering it admits weaker/noisier waves.
WAVE_CORRELATION_THRESHOLD = 0.7

# Minimum history points in BOTH zones before a pair is correlated. Correlation
# over very short series is statistically meaningless.
MIN_POINTS_FOR_CORRELATION = 10


# ===== COMPONENT 3 — PROPAGATION DELAY / ETA PREDICTION =====

# If a downstream zone's predicted surge arrival is within this many seconds,
# it receives a preemptive RED alert before it has surged at all. This is the
# preemptive warning horizon: larger = earlier warnings but more false alarms.
PREEMPTIVE_ALERT_WINDOW = 30  # seconds


# ===== SURGE EPISODE TRACKING =====

# Density at or above this marks the start of a surge episode (used by
# ZoneManager.get_surge_start_time). Kept equal to MIN_DENSITY_THRESHOLD so
# "surge episode started" and "dense enough to be surging" agree by default.
SURGE_START_DENSITY_THRESHOLD = MIN_DENSITY_THRESHOLD


# ===== CLOUD LAYER (TIER 3) =====

# Append-only JSONL log of cascade detections. This is the optional "cloud"
# tier of the 3-tier architecture: the fog server logs events durably here
# rather than shipping them to a remote service.
# Overridden to /app/fog/logs/cascade_events.jsonl in Dockerfile.fog, where
# docker-compose bind-mounts a host directory so events survive container
# restarts. Defaults to the working directory for local runs.
CASCADE_LOG_PATH = os.getenv("CASCADE_LOG_PATH", "cascade_events.jsonl")

# Only log when a cascade is actually detected, to keep the log signal-dense
# (the detector runs on every ingest, i.e. several times per second).
LOG_ONLY_DETECTED_CASCADES = True


# ===== ZONE TOPOLOGY =====

# Hardcoded demo topology: three zones in a line, zone_1 -- zone_2 -- zone_3.
ZONE_ADJACENCY = {
    "zone_1": ["zone_2"],
    "zone_2": ["zone_1", "zone_3"],
    "zone_3": ["zone_2"],
}
