# CrowdShield — Fog/Edge Stampede Detection System

Restructured 3-tier fog/edge computing architecture for BCSE313L, built from the
original CrowdShield stampede-detection codebase (kept at the project root as
reference — nothing outside `fog_edge_system/` was modified during this rebuild).

## Architecture

```
Edge nodes (cameras)  →  Fog server (local LAN)  →  Cloud (optional logging)
```

- **Edge**: per-camera process running lightweight crowd density estimation
  (CSRNet) and emitting per-frame feature vectors for its assigned zone.
- **Fog**: LAN-local server aggregating feature streams from all edge nodes,
  running the **Pressure Cascade Detection Algorithm** — the novel contribution
  of this project — to detect crowd surge waves propagating across adjacent
  zones (not a simple per-zone threshold check).
- **Cloud**: optional, out of scope for early phases; logging/aggregation only.

This structure and the cascade detection algorithm are being built out
phase-by-phase; see `PROGRESS.md` for the full, timestamped build log.

## Directory Structure

```
fog_edge_system/
├── edge/
│   ├── edge_node.py        # Edge process: capture + inference + emit (Phase 1)
│   ├── csrnet_lite.py       # Lightweight CSRNet inference wrapper (Phase 1)
│   ├── model.py             # Copied from project root; CSRNet = CrowdStampedeYantra alias added
│   ├── predict.py           # Copied from project root, unmodified — CrowdCounter reference
│   └── config.py            # Edge node configuration (Phase 1)
├── fog/
│   ├── fog_server.py        # Fog aggregation server (Phase 2)
│   ├── cascade_detector.py  # Pressure Cascade Detection Algorithm (Phase 2/3)
│   ├── zone_manager.py      # Per-zone state/timeseries buffers (Phase 2)
│   └── config.py            # Fog server configuration (Phase 2)
├── dashboard/
│   └── index.html           # Live monitoring dashboard (Phase 4)
├── shared/
│   └── models.py            # Shared data models/schemas used by edge + fog
├── demo/
│   └── run_demo.py          # End-to-end demo orchestration (Phase 5)
├── weights/                 # CSRNet checkpoint goes here (NOT included, see below)
├── docker-compose.yml
├── Dockerfile.edge
├── Dockerfile.fog
├── requirements_edge.txt
├── requirements_fog.txt
└── PROGRESS.md               # Full timestamped development log (mandatory, ongoing)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### CSRNet Weights

A pretrained CSRNet checkpoint (~130MB, ShanghaiTech Part A) is required at
`weights/PartAmodel_best.pth` and is **not included in this repo** (exceeds GitHub's
file size limit). Get a compatible checkpoint from
[leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch) (same
architecture: VGG16 frontend + 6-layer dilated backend), or train your own with
`evaluation/train.py`.

To run inference directly:
```bash
python predict.py --image <path> --model ../weights/PartAmodel_best.pth
```

### Data

Sample/demo videos and datasets referenced by the code (`data/`) are also not
included — only `data/make_videos.py` is tracked. Drop your own crowd videos at
`data/video1.mp4`, `data/video2.mp4`, `data/video3.mp4` to run with Docker, or use the
synthetic-data-free `demo/run_demo.py` (see Quick Start below), which needs none of
this.

## Edge Node (Phase 1)

`edge/edge_node.py` runs one camera zone. Multiple instances run simultaneously,
one per camera, each pointed at the same fog server:

```bash
python edge_node.py --zone zone_1 --source 0 --fog-url http://localhost:8000 \
    --model-path ../weights/PartAmodel_best.pth
```

or with a video file:

```bash
python edge_node.py --zone zone_1 --source crowd_video.mp4 --fog-url http://localhost:8000
```

It runs three concurrent loops:
- **Capture + inference** (frame-paced): every `--interval` frames (default 10), runs
  CSRNet crowd counting (`CrowdCounter.predict`) and Farneback optical flow between
  consecutive frames, producing `flow_x`, `flow_y`, `flow_magnitude`.
- **Telemetry sender** (every 2s, background thread): POSTs the latest telemetry packet
  to `<fog-url>/ingest`. If the fog server is unreachable, buffers the last 5 unsent
  packets and keeps retrying — never crashes.
- **Alert poller** (every 2s, background thread): GETs `<fog-url>/alert/<zone_id>`; if the
  response level is `YELLOW` or `RED`, prints a console warning with ETA if available.

Defaults for fog URL, inference interval, model path, and all internal thresholds
(resize dimensions, Farneback parameters, HTTP timeouts, buffer size) live in
`edge/config.py`, documented inline with reasoning for each value.

Verified (see `PROGRESS.md` Phase 1 section for full test details): end-to-end run
against a live fog server, graceful degradation when the fog server is down, an
isolated buffer unit test, and recovery of buffered packets once the fog server comes
back online mid-run — all passed with no crashes.

## Fog Server (Phase 2)

The fog layer receives telemetry from all edge nodes, runs the Pressure Cascade
Detection Algorithm after every ingest, and serves alert decisions back to the edge.

```bash
cd fog
python fog_server.py            # or: python -m uvicorn fog_server:app --port 8000
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Receive telemetry from an edge node; triggers cascade detection |
| `GET /zones` | Current state of all zones |
| `GET /alert/{zone_id}` | Alert level (GREEN/YELLOW/RED) + ETA — polled by edge nodes |
| `GET /cascade` | Latest full cascade detection result |
| `GET /history/{zone_id}` | Density time series for a zone |
| `GET /` | Dashboard page |
| `WS /ws` | Live zone + cascade push every 2s |

### The Pressure Cascade Detection Algorithm

The novel contribution. It detects whether a density surge in one zone is a
*time-delayed copy* of a surge in an adjacent zone — a real propagating pressure
wave — rather than two coincidental spikes, and predicts when the next zone
downstream will be hit.

Three components, each gating the next:

1. **Temporal pattern matching** (`cascade_detector.compute_surge_confidence`) —
   Dynamic Time Warping compares each zone's recent density curve against a reference
   pre-stampede S-curve (slow rise → acceleration → plateau, per Helbing et al. 2007;
   Still 2000). This asks *is this the shape of a buildup*, not merely *is density high*.
   Gated additionally on absolute density, since DTW is scale-invariant.

2. **Cross-correlation wave detection** (`compute_wave_correlation`) — for each adjacent
   pair, normalized cross-correlation over lags of ±20s yields both a similarity score
   and the propagation delay. A wave is confirmed only if correlation exceeds threshold
   **and** the lag is positive (B follows A) **and** both zones' optical flow vectors
   point the same way (the crowd is physically moving A → B).

3. **Propagation delay / ETA prediction** (`compute_eta`) — given the measured delay and
   when the upstream zone began surging, predicts arrival at the next zone downstream.
   Zones predicted to be hit within `PREEMPTIVE_ALERT_WINDOW` seconds go RED *before*
   their own camera shows anything.

This is what the fog tier buys: a single edge node sees only its own camera, but the fog
server sees every zone at once and can warn a zone about a wave still in transit.

All thresholds live in `fog/config.py`.

**Verified to discriminate, not just threshold.** Three negative scenarios each defeat a
different confirmation condition — non-adjacent surges (no adjacency), simultaneous
adjacent surges (zero lag), and correlated surges with opposing flow (bottleneck squeeze,
not a wave). In all three both zones are individually flagged as surging, and no cascade
is reported. See `PROGRESS.md` Phase 2 for full test results.

## Dashboard (Phase 3)

Open `http://<fog-host>:8000/` while the fog server is running. Single self-contained
HTML file — no React, no build step, no npm, no external requests. Four rows:

1. **Zone cards** — people count, surge confidence bar, flow-direction arrow, alert badge
   (RED pulses), and a preemptive `⚠ SURGE ETA` box when a wave is predicted to arrive.
2. **Cascade status** — `NO CASCADE` / `CASCADE DETECTED`, cascade score, confirmed wave
   pairs with correlation and lag, and the preemptive alert list.
3. **Density chart** — pure-canvas rolling 60-second plot, one line per zone, line colour
   following that zone's alert level.
4. **Algorithm inspection** (collapsible) — per-zone DTW distances and both gate results,
   every evaluated zone pair including **rejected** ones with the reason each was vetoed,
   and the reference S-curve drawn against each zone's actual normalized curve.

Transport is WebSocket (`/ws`) with automatic fallback to 2-second polling of `/zones`
and `/cascade`, and automatic reconnection. If the fog server goes down the page keeps
the last known values on screen and shows a disconnected badge rather than blanking.

The inspection panel is the answer to "how does the algorithm work?" — the rejected rows
show which of the four conditions (source surging / correlation / positive lag / flow
alignment) prevented each false positive.

## Quick start — run the demo (Phase 5)

**No Docker, no GPU, no model weights, no video files, no camera.** Only `numpy` is
strictly required.

```bash
cd fog_edge_system/demo
python run_demo.py
```

Open **http://localhost:8000** to watch the dashboard alongside the console.

A ~95 second scenario drives three simulated zones through a full pressure cascade:
zone_1 surges → zone_2 repeats the same curve 10s later → **zone_3 is alarmed before its
own density has moved at all**, then actually surges 10 seconds later, confirming the
prediction. The telemetry is synthetic; the detection algorithm, API and dashboard are
the real production code.

If FastAPI/uvicorn are missing or the port is busy, the demo says so and continues in
console-only mode — the full algorithm demonstration still runs. See
[`demo/README.md`](demo/README.md) for flags and presentation tips.

## Docker (Phase 4)

Runs the whole system — one fog server plus three resource-constrained edge nodes:

```bash
# put crowd videos at data/video1.mp4, video2.mp4, video3.mp4
docker compose up --build
# dashboard -> http://localhost:8000/
```

| service | image size | constraints |
|---|---|---|
| `fog` | 309 MB | none (port 8000, logs volume) |
| `edge_1..3` | 1.62 GB | `cpus: 0.5`, `mem_limit: 512m` |

The edge services are deliberately capped to simulate real edge hardware — this is how
edge viability is demonstrated without physical devices. **Measured under load:** all
three edge containers ran CSRNet inference plus Farneback optical flow concurrently at
~49% CPU (pinned against the 0.5-core cap) using 286–361 MB of their 512 MB limit, none
OOM-killed. The fog server used 1.27% CPU and 51 MB.

Notes on the setup:
- Edge images use the **CPU-only** torch build (`2.5.1+cpu`, from PyTorch's own wheel
  index). The default CUDA wheels are several GB larger and useless without a GPU.
- CSRNet weights are **bind-mounted read-only** (`./weights:/app/weights:ro`) rather than
  baked in — one 125 MB host copy shared by all three edge containers, and swapping
  models needs no rebuild.
- The fog server has a healthcheck, and edges wait on `condition: service_healthy` so
  they do not start against a not-yet-listening server.
- Cascade events are written to `./fog/logs/cascade_events.jsonl` on the host via a
  volume, so they survive container removal.
- Edge nodes survive a fog restart: they buffer, warn, and resume automatically without
  crashing or being restarted.

`FOG_HOST`, `FOG_PORT`, and `CASCADE_LOG_PATH` can be overridden by environment variable;
all algorithm thresholds remain in `fog/config.py`.

## Status

**All five phases complete** — scaffolding, edge node, fog server + cascade algorithm,
dashboard, Docker, and the standalone demo. See `PROGRESS.md` for the full development
log, including a significant bug found and fixed in the cross-correlation lag convention
(which would have caused every genuine wave to be rejected), the API extension the
dashboard's inspection panel required, and a silent-failure log-path bug caught during
containerization.

**Known limitation affecting video-based demos:** CSRNet returns a constant count on
synthetic shape videos (it detects human heads, not drawn circles), so a cascade cannot be
triggered through the video path with generated test footage. Use real crowd video, feed
`POST /ingest` directly, or use the Phase 5 demo — which is why it exists.
