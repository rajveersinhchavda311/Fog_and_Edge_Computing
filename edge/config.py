"""
Edge node configuration defaults. All values here can be overridden via
edge_node.py command-line arguments; these are just the fallback defaults.
"""

# Fog server connection
DEFAULT_FOG_URL = "http://localhost:8000"

# How often (in captured frames) to run CSRNet + optical flow inference.
# Lower = more frequent inference = more accurate but more CPU load.
DEFAULT_INFERENCE_INTERVAL = 10

# Path to the CSRNet checkpoint, relative to edge_node.py's working directory.
DEFAULT_MODEL_PATH = "../weights/PartAmodel_best.pth"

# How often (in seconds) to POST telemetry to the fog server /ingest endpoint,
# and how often to poll the fog server /alert/<zone_id> endpoint.
TELEMETRY_INTERVAL_SEC = 2.0
ALERT_POLL_INTERVAL_SEC = 2.0

# Frame resize target before optical flow / inference (keeps CPU load bounded
# and matches the resolution the reference video_pipeline.py used).
RESIZE_W, RESIZE_H = 640, 480

# Farneback optical flow parameters — copied from the reference implementation
# in video_pipeline.py so flow_x/flow_y/flow_magnitude are computed the same way.
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# Network request timeouts (seconds) — kept short so a slow/unreachable fog
# server never blocks the capture loop for long.
HTTP_POST_TIMEOUT_SEC = 1.0
HTTP_GET_TIMEOUT_SEC = 1.0

# Number of most-recent unsent telemetry packets to buffer and retry when the
# fog server is unreachable. Older packets beyond this are dropped (telemetry
# is a live stream, not a durable log — dropping stale data is intentional).
PACKET_BUFFER_SIZE = 5
