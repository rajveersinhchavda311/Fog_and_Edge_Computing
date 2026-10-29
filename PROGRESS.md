# CrowdShield — Fog Edge System Development Log

This log records every action taken while building `fog_edge_system/`, a restructuring
of the original CrowdShield stampede-detection project into a 3-tier fog/edge/cloud
architecture for BCSE313L. Entries are chronological within each phase. Nothing is
summarized away — see project instructions for the reasoning.

---

## Phase 0 — Cleanup

### [2026-08-27 00:00] Task: Read existing reference implementation before copying anything
- **Approach:** Read `model.py` and `predict.py` from the project root in full to understand
  exact current behavior before copying them into `fog_edge_system/edge/` as reference files.
- **Findings:**
  - `model.py` defines a single class `CrowdStampedeYantra(nn.Module)` — a standard CSRNet:
    VGG-16 frontend (first 13 conv layers, `[64,64,M,128,128,M,256,256,256,M,512,512,512]`)
    initialized from ImageNet-pretrained `torchvision.models.vgg16` weights, followed by a
    6-layer dilated-convolution backend (`[512,512,512,256,128,64]`, dilation=2), followed by
    a 1x1 conv output layer producing a single-channel density map. No `CSRNet` name exists
    anywhere in the file — confirms the bug described in the task brief.
  - `predict.py` defines `CrowdCounter`, which loads a checkpoint (`state_dict` key or raw
    state dict), builds the ImageNet-normalize transform pipeline
    (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`), and exposes `.predict(image)`
    returning `{'count': int, 'density_avg': float, 'density_peak': float, 'density_map': np.ndarray}`.
    Accepts file path, `PIL.Image`, or `np.ndarray` as input.
- **Files touched:** none (read-only)
- **Status:** DONE

### [2026-08-27 00:00] Task: Create fog_edge_system/ directory skeleton
- **Approach:** Created the full directory tree specified in the task brief:
  `edge/`, `fog/`, `dashboard/`, `shared/`, `demo/`, plus a `weights/` directory for the
  (currently absent) CSRNet checkpoint.
- **Decision made:** Used `mkdir -p` to create all subdirectories in one shell call rather
  than relying on file-write side effects, so the tree exists explicitly and is inspectable
  before any files land in it.
- **Files touched:** created directories `fog_edge_system/{edge,fog,dashboard,shared,demo,weights}`
- **Status:** DONE

### [2026-08-27 00:00] Task: Copy model.py into fog_edge_system/edge/ with CSRNet alias fix
- **Approach:** Copied `model.py` verbatim from project root into
  `fog_edge_system/edge/model.py`, then appended `CSRNet = CrowdStampedeYantra` at the
  bottom of the file.
- **Decision made:** Chose a bare alias assignment (`CSRNet = CrowdStampedeYantra`) over
  renaming the class itself. Renaming would touch every call site that still references
  `CrowdStampedeYantra` and risks silently diverging from the original reference file we're
  keeping as ground truth. An alias is a single line, is obviously visible at the bottom of
  the file, and fixes exactly the bug described (`evaluate.py`'s `from model import CSRNet`
  failing at import time) without touching anything else.
- **Files touched:** created `fog_edge_system/edge/model.py`
- **Status:** DONE

### [2026-08-27 00:00] Task: Copy predict.py into fog_edge_system/edge/ for reference
- **Approach:** Copied `predict.py` verbatim from project root into
  `fog_edge_system/edge/predict.py`. No changes made to its logic — it already works
  correctly (`CrowdCounter` imports `CrowdStampedeYantra` directly, not the broken
  `CSRNet` alias, so it never had the import bug).
- **Decision made:** Left the import line as `from model import CrowdStampedeYantra`
  unchanged rather than switching it to the new `CSRNet` alias, since both names now
  resolve to the same class in the copied `model.py` and changing it here would be a
  cosmetic edit with no functional benefit at this stage. `csrnet_lite.py` (Phase 1) will
  be the actual edge-node-facing inference module; this copy is explicitly for reference.
- **Files touched:** created `fog_edge_system/edge/predict.py`
- **Status:** DONE

### [2026-08-27 00:00] Task: Check whether CSRNet weights exist anywhere in the repo
- **Approach:** Searched the full repository tree for `*.pth`, `*.pth.tar`, and any
  `weights/` directory.
- **Findings:** Only `models/gru_shockwave_model.pth` (the GRU temporal risk model) and
  `models/feature_scaler.pkl` exist. No CSRNet checkpoint (`model_best.pth.tar` or similar)
  is present anywhere in the repository.
- **Obstacle:** The original root `README.md` was checked in full during an earlier
  conversation turn — it does **not** actually contain a Google Drive download link for
  CSRNet weights, despite the task brief referring to "the Google Drive link in the
  original README." No such link exists in this repo.
- **Resolution:** Per user decision (asked directly via clarifying question), the new
  README will note that no weights file or download link currently exists in this repo,
  and will point to the official CSRNet reference implementation
  (`leeyeehoo/CSRNet-pytorch` on GitHub) as the typical external source for pretrained
  weights on ShanghaiTech, with a placeholder to be replaced once a real link is supplied.
- **Files touched:** none yet (documented here; README written in a later entry below)
- **Status:** DONE (documented as known gap, not blocking)

### [2026-08-27 00:00] Task: Create weights/ directory for CSRNet checkpoint
- **Approach:** Created empty `fog_edge_system/weights/` directory to hold the CSRNet
  checkpoint once downloaded/trained. Added a `.gitkeep` placeholder file explaining
  what belongs there.
- **Decision made:** Placed `weights/` at the `fog_edge_system/` root (not inside `edge/`)
  since both the edge node and any local testing/demo scripts may need to reference it by
  a shared relative path, and it mirrors the original repo's expected `weights/` location
  referenced by `predict.py`'s default arg.
- **Files touched:** created directory `fog_edge_system/weights/`, `fog_edge_system/weights/.gitkeep`
- **Status:** DONE

### [2026-08-27 00:10] Task: Unexpected file discovered in weights/ after initial scaffolding
- **Approach:** After writing README.md's initial "no weights exist" section, re-listed
  `fog_edge_system/` to confirm the full file tree before moving on. Found a file that had
  not been created by any tool call in this session: `fog_edge_system/weights/PartAmodel_best.pth`
  (130,117,002 bytes), timestamped the same minute as the directory's creation.
- **Obstacle:** This file's presence contradicted the README claim (just written) that no
  CSRNet weights exist anywhere. Since I did not create it, its origin was unknown — could
  have been manually copied in by the user via file explorer/sync while I was working, or
  could be unrelated/irrelevant to this project.
- **Resolution:** Did not assume either way. Asked the user directly whether this file
  should be treated as the real CSRNet checkpoint. User confirmed: yes, treat it as the
  real pretrained weights, verify it loads, and correct the docs.
- **Decision made:** Chose to verify empirically (load into the model, run real inference)
  rather than trust the filename alone, since a corrupted or mismatched checkpoint would
  silently break every later phase that depends on edge inference.
- **Files touched:** none yet (verification steps follow in next entries)
- **Status:** DONE (investigation), see next entries for verification

### [2026-08-27 00:12] Task: Verify PartAmodel_best.pth loads into CrowdStampedeYantra
- **Approach:** Attempted to run a verification script loading the checkpoint via
  `torch.load()` and `model.load_state_dict()`.
- **Obstacle:** `ModuleNotFoundError: No module named 'torch'` — the base Python 3.12.10
  installation on this machine (`AppData/Local/Programs/Python/Python312/python`) had no
  packages installed at all; `pip show torch` confirmed it was absent.
- **Resolution:** Asked the user how to proceed rather than installing multi-GB packages
  unprompted. User directed: create a venv named `venv` inside `fog_edge_system/`, activate
  it, install torch+torchvision (CPU build) into it, and use this venv for all future
  installs in this project.
- **Dependency installed:** Created `fog_edge_system/venv/` via `python -m venv venv`.
  Installed into that venv: `pip` (upgraded to latest), `torch` and `torchvision` from
  `https://download.pytorch.org/whl/cpu` (CPU-only build — no CUDA available/needed on
  this dev machine, and CPU wheels are dramatically smaller/faster to install than the
  CUDA builds). Exact versions not pinned in this command; will be captured in
  `requirements_edge.txt` in Phase 1 by freezing this venv.
- **Files touched:** created `fog_edge_system/venv/` (virtual environment, not committed
  to version control — will confirm `.gitignore` entry in a later phase)
- **Status:** DONE

### [2026-08-27 00:15] Task: Re-run checkpoint verification using the new venv
- **Approach:** Re-ran the same load-verification script using
  `fog_edge_system/venv/Scripts/python.exe` instead of the system interpreter.
- **Test:** Instantiated `CrowdStampedeYantra(load_weights=True)`, loaded
  `../weights/PartAmodel_best.pth` via `torch.load(..., map_location='cpu', weights_only=False)`,
  called `model.load_state_dict(checkpoint['state_dict'])` with default `strict=True`.
- **Expected output:** Either a clean load (confirming architecture match) or a
  `RuntimeError` listing missing/unexpected keys (confirming mismatch).
- **Actual output:**
  ```
  Checkpoint has state_dict key. Other keys: ['epoch', 'arch', 'optimizer', 'best_prec1']
  SUCCESS: state_dict loaded into CrowdStampedeYantra with no errors (strict=True)
  ```
- **Interpretation:** The checkpoint is a genuine PyTorch training checkpoint (not just a
  bare state dict) — the presence of `epoch`, `arch`, `optimizer`, and `best_prec1` keys is
  the exact structure produced by the original `leeyeehoo/CSRNet-pytorch` reference
  training script. `strict=True` loading with zero errors confirms the architecture
  (VGG16 frontend + 6-layer dilated backend + 1x1 output conv) matches exactly.
- **Files touched:** none (verification only, no permanent script kept — ad hoc command)
- **Status:** DONE

### [2026-08-27 00:18] Task: End-to-end inference smoke test with the verified checkpoint
- **Approach:** Needed a real image to confirm not just that the state dict loads, but
  that a full forward pass through `edge/predict.py`'s `CrowdCounter.predict()` produces
  sane output. Searched the repo for any `.jpg`/`.jpeg`/`.png` file outside
  `fog_edge_system/` (none exist inside it yet). Found exactly one:
  `CrowdShield/client/src/assets/hero-bg.jpg` (a hero banner image from the React
  frontend, not a curated crowd-counting test image, but the only real photo available).
- **Dependency installed:** `pillow`, `numpy` into `fog_edge_system/venv/` (both required
  by `predict.py`; `torch`/`torchvision` already present from the prior step).
- **Test:** Ran `predict.py --image ../../CrowdShield/client/src/assets/hero-bg.jpg
  --model ../weights/PartAmodel_best.pth` using the venv interpreter.
- **Expected output:** A non-crashing run producing a plausible `count`, `density_avg`,
  `density_peak` — since this isn't a labeled crowd-counting benchmark image, "correctness"
  here means "does not crash and produces a non-negative, non-absurd count," not a specific
  ground-truth number.
- **Actual output:**
  ```
  People count:     229
  Avg density:      0.1104 people/1000px2
  Peak density:     0.4958
  ```
- **Interpretation:** Full pipeline (image load → ImageNet-normalize transform → VGG16+
  dilated-conv forward pass → density map sum) runs without error and produces a
  plausible-looking count. This is sufficient confirmation that `PartAmodel_best.pth` is
  a working CSRNet checkpoint compatible with `CrowdStampedeYantra`. (Terminal encoding
  mangled the "px²" superscript to "px�" in the raw console output — cosmetic only, not a
  functional bug; `predict.py`'s f-string itself is correct UTF-8.)
- **Files touched:** none (ran existing `edge/predict.py` as-is, no code changes)
- **Status:** DONE

### [2026-08-27 00:20] Task: Correct README.md and reference docs now that weights are confirmed present
- **Approach:** The "CSRNet Weights" section of `README.md`, written earlier in Phase 0,
  claimed no checkpoint existed anywhere and offered only external sourcing instructions.
  This is now factually wrong given the verification above.
- **Resolution:** Rewrote the section to state the checkpoint is present and verified,
  documenting exactly what was checked (state_dict load, end-to-end inference) and the
  exact usage command. Kept a note that the file's origin outside this repo is still
  unconfirmed (it was not created by any action in this session, and the original root
  README has no download link), since traceability matters for later documentation/patent
  material — "it works" is not the same claim as "we know where it came from."
- **Files touched:** modified `fog_edge_system/README.md`
- **Decision made:** Did not delete or move `PartAmodel_best.pth` — left it exactly where
  it was found (`fog_edge_system/weights/`), since that already matches the intended
  location per the original Phase 0 plan.
- **Status:** DONE

### [2026-08-27 00:22] Task: Prevent large weights file and venv from being committed to git
- **Approach:** Checked project root `.gitignore`. `venv/` was already globally ignored
  (covers `fog_edge_system/venv/` automatically). `models/` had an ignore rule with
  explicit `!` exceptions for the two GRU files, but `fog_edge_system/weights/` was not
  covered by any existing pattern — the 130MB `PartAmodel_best.pth` would be committed by
  default.
- **Resolution:** Added `fog_edge_system/weights/*` with `!fog_edge_system/weights/.gitkeep`
  exception, mirroring the pattern already used for `models/`. This keeps the directory
  structure visible in git (via `.gitkeep`) while excluding the actual large binary.
- **Files touched:** modified `.gitignore` (project root)
- **Status:** DONE

### [2026-08-27 00:25] Task: Housekeeping — remove stray __pycache__ from verification runs
- **Approach:** Running `predict.py` during the inference smoke test generated a
  `fog_edge_system/edge/__pycache__/` directory (compiled `.pyc` files). This is already
  covered by the global `__pycache__/` gitignore rule, but was removed anyway to keep the
  working tree clean for the next phase.
- **Files touched:** deleted `fog_edge_system/edge/__pycache__/` (generated, not tracked)
- **Status:** DONE

## Phase 0 — Summary

**Status: DONE.** All Phase-0-scoped deliverables are complete:
- `fog_edge_system/` directory tree created (`edge/`, `fog/`, `dashboard/`, `shared/`,
  `demo/`, `weights/`) — empty except for the two reference files and weights, since
  `edge_node.py`, `csrnet_lite.py`, `config.py`, `fog_server.py`, `cascade_detector.py`,
  `zone_manager.py`, `dashboard/index.html`, `shared/models.py`, `demo/run_demo.py`,
  `docker-compose.yml`, `Dockerfile.edge`, `Dockerfile.fog`, `requirements_edge.txt`,
  `requirements_fog.txt` are explicitly scoped to later phases per the task brief and were
  intentionally not created yet.
- `edge/model.py` — copy of project-root `model.py` with `CSRNet = CrowdStampedeYantra`
  alias appended, fixing the import bug described in the task brief.
- `edge/predict.py` — unmodified copy of project-root `predict.py`, kept as reference.
- `weights/PartAmodel_best.pth` — real CSRNet checkpoint, verified via strict state_dict
  load and end-to-end inference smoke test (229-count output on a real photo, no errors).
- `fog_edge_system/venv/` — Python 3.12 virtual environment with CPU-build
  `torch`/`torchvision`/`pillow`/`numpy` installed; to be used for all future installs in
  this project per user instruction.
- `README.md` — architecture overview, directory map, and corrected weights section.
- `.gitignore` updated to exclude `fog_edge_system/weights/*.pth` (large binary) while
  keeping the directory tracked via `.gitkeep`.
- No files outside `fog_edge_system/` were modified except `.gitignore` (project root),
  which was a necessary, minimal, additive change to prevent a 130MB binary from being
  committed — consistent with "keep existing files as reference, don't modify them."

**Open item carried into later phases:** the exact provenance of `PartAmodel_best.pth`
(who placed it, downloaded from where) is unconfirmed — it works and is verified
functionally compatible, but no download link or source URL was captured. Should be
documented properly if this becomes patent/report material.

Ready for Phase 1 (Edge Node) pending user confirmation to proceed.

## Phase 1 — Edge Node

### [2026-08-27 01:00] Task: Review reference logic before writing edge_node.py
- **Approach:** Re-read the optical flow section of the project-root `video_pipeline.py`
  (lines ~140-209) and the `CrowdCounter.predict()` interface in
  `fog_edge_system/edge/predict.py` to confirm exact call signatures before writing new
  code that depends on them.
- **Findings used:**
  - Farneback call signature: `cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3,
    15, 3, 5, 1.2, 0)` — positional args are `pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0`. Reused exactly these values.
  - `CrowdCounter.predict(image_input)` accepts a file path, `PIL.Image`, or `np.ndarray`,
    and returns `{'count', 'density_avg', 'density_peak', 'density_map'}`. Edge node passes
    an RGB `np.ndarray` frame directly (converted from OpenCV's default BGR via
    `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`), matching the `np.ndarray` branch of
    `CrowdCounter.predict`'s input-type dispatch.
- **Decision made:** The task brief's requested telemetry packet asks for a single
  `flow_x`, `flow_y`, `flow_magnitude` triple per inference cycle — a much simpler
  summary than `video_pipeline.py`'s full pipeline (which computes motion clusters,
  radial spread, EMA-smoothed speed/variance/flux, etc. for its own heuristic risk
  score). Rather than porting that entire heuristic engine, `compute_optical_flow()` in
  the new `edge_node.py` computes the mean flow vector across *all* pixels of the dense
  Farneback flow field (`np.mean(flow[...,0])`, `np.mean(flow[...,1])`), then derives
  magnitude as `hypot(flow_x, flow_y)`. This is intentionally simpler than
  `video_pipeline.py`'s "mean of only above-threshold pixels" approach — the cascade
  detection algorithm (Phase 2/3) is where cross-zone pattern analysis belongs; the edge
  node's job is just to report a clean per-frame motion signal, not to pre-judge risk.
- **Files touched:** none (read-only review)
- **Status:** DONE

### [2026-08-27 01:05] Task: Write edge/config.py
- **Approach:** Created a plain-Python config module holding all tunable defaults so
  `edge_node.py`'s CLI arguments have sensible fallbacks and so later phases (fog server,
  demo orchestration) can import shared constants without re-deriving them.
- **Threshold/constant decisions:**
  - `DEFAULT_INFERENCE_INTERVAL = 10` — matches the task brief's stated default exactly
    ("Every N frames (configurable, default 10)").
  - `TELEMETRY_INTERVAL_SEC = 2.0` / `ALERT_POLL_INTERVAL_SEC = 2.0` — both specified
    explicitly in the task brief ("every 2 seconds" for both POST and alert GET).
  - `RESIZE_W, RESIZE_H = 640, 480` — reused verbatim from `video_pipeline.py`'s
    `RESIZE_W, RESIZE_H = 640, 480` constants, to keep frame preprocessing consistent
    with the reference pipeline's tested resolution (bounds CPU cost of both CSRNet
    inference and optical flow per frame).
  - `FARNEBACK_PARAMS` — copied verbatim from `video_pipeline.py`'s hardcoded positional
    args, converted to a named dict for readability at the call site.
  - `HTTP_POST_TIMEOUT_SEC = 1.0` / `HTTP_GET_TIMEOUT_SEC = 1.0` — not specified in the
    task brief; chosen as a conservative short timeout so a slow/unreachable fog server
    can never stall either background thread for more than ~1s per attempt. If this
    value were increased (e.g. to 5s+), a hung fog server would block the telemetry
    thread's retry loop for much longer per cycle, delaying detection that the server is
    down; if decreased too far (e.g. <0.2s), transient LAN jitter could cause false
    "unreachable" warnings on an otherwise healthy connection. 1.0s was chosen as a
    reasonable middle ground for a local/LAN deployment (fog server is expected to be on
    the same LAN as edge nodes per the 3-tier architecture, so round-trip times should be
    low single-digit milliseconds under normal conditions).
  - `PACKET_BUFFER_SIZE = 5` — matches the task brief's explicit instruction ("Buffer the
    last 5 unsent packets and retry").
  - `DEFAULT_MODEL_PATH = "../weights/PartAmodel_best.pth"` — set relative to
    `edge_node.py`'s own location (`fog_edge_system/edge/`), one directory up to
    `fog_edge_system/weights/`, matching where the verified checkpoint actually lives
    (see Phase 0 log). Note: the task brief's example CLI usage shows
    `--model-path ../../weights/model_best.pth.tar` (two levels up, different filename)
    — this appears to assume a different working directory or a not-yet-renamed weights
    file. Did not blindly copy that path into the default, since it would point outside
    the project entirely from `edge/`'s actual location and to a filename
    (`model_best.pth.tar`) that doesn't exist. The default here is deliberately correct
    for this repo's actual verified file; the CLI `--model-path` argument remains fully
    user-overridable either way, so this doesn't restrict usage — it only changes what
    happens if a user omits the flag entirely.
- **Files touched:** created `fog_edge_system/edge/config.py`
- **Status:** DONE

### [2026-08-27 01:10] Task: Write edge/edge_node.py — core implementation
- **Approach:** Implemented `EdgeNode` as a class wrapping three concurrent loops:
  1. `run_capture_loop()` — runs on the main thread; reads frames via `cv2.VideoCapture`,
     runs CSRNet inference + optical flow every `inference_interval` frames, and stores
     the latest computed telemetry packet in a shared, lock-protected slot
     (`self.latest_packet`).
  2. `run_telemetry_loop()` — runs on a daemon background thread; wakes up every
     `TELEMETRY_INTERVAL_SEC` (2s), grabs the latest packet, adds it to the bounded
     `PacketBuffer`, and attempts to POST every buffered packet to `<fog_url>/ingest`.
  3. `run_alert_poll_loop()` — runs on a second daemon background thread; wakes up every
     `ALERT_POLL_INTERVAL_SEC` (2s), GETs `<fog_url>/alert/<zone_id>`, and prints a large
     console warning banner if the returned `level` is `YELLOW` or `RED`, including ETA
     if present in the response.
- **Design decision — three loops instead of one:** The task brief specifies three
  independent cadences: inference runs every N *frames* (variable real time, depends on
  camera FPS and inference cost), while telemetry POST and alert polling both run on a
  fixed 2-second *wall clock* cadence, independent of frame rate. A single-threaded loop
  driven by frame reads cannot cleanly serve a fixed-wall-clock-interval requirement
  (frame read rate and 2-second intervals are not the same clock), so the video capture
  loop (frame-paced) was separated from the telemetry/alert loops (wall-clock-paced)
  using Python's `threading` module. This was chosen over `asyncio` because OpenCV's
  `VideoCapture.read()` is a blocking synchronous call with no async-native equivalent in
  this codebase, and introducing an async event loop just for two lightweight periodic
  HTTP calls would add complexity without a clear benefit at this scale (one edge node
  process handles exactly one camera).
- **Design decision — shared "latest packet" slot instead of a queue between capture and
  telemetry threads:** The telemetry loop always sends the *most recent* state, not a
  full history of every inference result — deliberately matching the task brief's implied
  behavior ("Every N frames... Sends this packet... every 2 seconds", i.e., telemetry
  cadence is decoupled from and slower than inference cadence, so intermediate inference
  results between two telemetry ticks are naturally superseded, not queued). A
  lock-protected single "latest packet" variable is the simplest structure that expresses
  "always send the newest thing," and avoids unbounded growth if telemetry sending
  falls behind inference for any reason.
- **Design decision — `PacketBuffer` as a separate class:** Extracted the "last 5 unsent
  packets, retry" requirement into its own small class (`PacketBuffer`) with `add`,
  `snapshot`, `remove`, `clear` methods, rather than inlining a deque directly into
  `EdgeNode`. Reasoning: the retry logic (send all pending, remove only the ones that
  succeeded, stop retrying at the first failure to avoid reordering) is a self-contained
  piece of behavior that benefits from being unit-testable in isolation (see test log
  below) and independently reasoned about, separate from the threading/HTTP concerns in
  `EdgeNode` itself.
- **Files touched:** created `fog_edge_system/edge/edge_node.py`
- **Status:** DONE (implementation), see below for bug found and fixed during testing

### [2026-08-27 01:20] Task: Initial buggy retry-logic implementation (bug found before testing, fixed immediately)
- **Approach (first draft):** Initially wrote the telemetry send loop using a
  `for p in pending: ... else: self.packet_buffer.clear()` construct — the `for/else`
  clause was meant to clear the buffer only if the loop completed without `break`ing
  (i.e., all packets sent successfully), combined with a `sent_any` boolean flag and a
  trailing dead-code block that checked `sent_any and pending and
  self.packet_buffer.snapshot()` and did nothing (`pass`).
- **Obstacle:** On review before running any test, this was identified as both overly
  convoluted and subtly wrong: the trailing `if sent_any and ...: pass` block was
  literally inert dead code (a no-op), and using `for/else` plus a separate boolean flag
  to express "did we send everything" was harder to reason about than necessary — it
  conflated "did any packet succeed" with "did all packets succeed," when what was
  actually needed was "which specific packets succeeded, so exactly those can be
  removed."
- **Resolution:** Rewrote the send loop to build an explicit `sent = []` list of packets
  that were confirmed sent (HTTP 200), breaking out of the loop on the first failure
  (either a non-200 response or a `requests.exceptions.RequestException`) to preserve
  packet ordering — i.e., never skip ahead and send packet 3 if packet 2 failed, since
  that would send telemetry out of chronological order to the fog server. After the loop,
  call `self.packet_buffer.remove(sent)` to remove exactly the packets that succeeded,
  leaving any unsent ones (from this cycle or earlier) in the buffer for the next retry.
  This required adding a new `PacketBuffer.remove(packets)` method (matched by each
  packet's `timestamp` field, which is unique per packet since it's set via `time.time()`
  at packet construction) — the buffer did not have this capability in the first draft,
  which only had `add`, `snapshot`, and `clear`.
- **Files touched:** modified `fog_edge_system/edge/edge_node.py` (rewrote
  `run_telemetry_loop`'s send logic; added `PacketBuffer.remove`)
- **Decision made:** Chose to key packet removal by `timestamp` rather than by list index
  or object identity, since `snapshot()` returns a *copy* of the buffer's contents (to
  avoid holding the lock during network I/O), so index-based removal against the live
  buffer after the snapshot was taken could remove the wrong entries if `add()` was
  called concurrently by... actually `add()` is only ever called from this same thread
  (the telemetry loop owns writing to the buffer), so this particular race isn't
  possible today — but keying by timestamp is still more robust and readable than
  positional indexing, and costs nothing extra.
- **Status:** DONE (fixed before first test run)

### [2026-08-27 01:30] Task: Set up local test harness (fake fog server + synthetic video)
- **Approach:** No real fog server exists yet (Phase 2), and no webcam or real crowd
  video is available in this sandboxed environment, so built a temporary, throwaway test
  harness to validate `edge_node.py`'s actual runtime behavior rather than trusting
  static code review alone:
  - `test_scratch/make_test_video.py` — generates a 200-frame synthetic `.mp4` (a white
    circle moving across a black background) using OpenCV's `VideoWriter`, purely to give
    `cv2.VideoCapture` a real, decodable video file with actual pixel motion (needed for
    optical flow to produce non-trivial output).
  - `test_scratch/fake_fog_server.py` — a minimal FastAPI app with two endpoints:
    `POST /ingest` (logs and accepts any packet, always returns 200) and
    `GET /alert/{zone_id}` (always returns a hardcoded `{"level": "YELLOW", "eta": 4.2,
    "message": ...}` so the alert-printing code path could be exercised deterministically).
- **Decision made:** These test files were placed in a `test_scratch/` subdirectory
  inside `edge/` specifically so they could be deleted afterward without ambiguity about
  what was deliverable vs. throwaway scaffolding — confirmed deleted at the end of this
  phase (see cleanup entry below). They are not part of the Phase 1 deliverable; Phase 2
  will build the *real* fog server with real `/ingest` and `/alert/<zone_id>` logic.
- **Dependency installed:** `requests`, `opencv-python`, `fastapi`, `uvicorn` into
  `fog_edge_system/venv/` (the same venv created in Phase 0). `torch`/`torchvision`
  already present from Phase 0's weight-verification step.
- **Files touched:** created (later deleted) `fog_edge_system/edge/test_scratch/
  make_test_video.py`, `fake_fog_server.py`, plus generated log files
- **Status:** DONE

### [2026-08-27 01:35] Task: Test 1 — basic syntax and import check
- **Test:** `python -m py_compile edge_node.py config.py`
- **Expected output:** No errors (confirms valid Python syntax and that all imports at
  module scope resolve without executing the full script).
- **Actual output:** `SYNTAX OK` — no errors.
- **Status:** PASSED

### [2026-08-27 01:40] Task: Test 2 — end-to-end run against a live fake fog server
- **Test:** Started `fake_fog_server.py` on port 8000, then ran
  `edge_node.py --zone zone_1 --source test_scratch/test_video.mp4 --fog-url
  http://localhost:8000 --model-path ../weights/PartAmodel_best.pth --interval 5` for a
  bounded window against the 200-frame synthetic video.
- **Obstacle (environment, not code):** First few attempts produced empty log files or
  `[Errno 10048] ... only one usage of each socket address ...` errors. Root cause: (1)
  Python's stdout is block-buffered by default when redirected to a file/pipe rather than
  a terminal, so output written just before a `timeout`-induced kill was never flushed to
  disk — fixed by re-running with `python -u` (unbuffered stdout) for all subsequent test
  runs. (2) Background `uvicorn` processes from earlier test attempts were not always
  fully terminated between runs (backgrounded shell commands in this environment don't
  reliably propagate process-group termination), leaving a stale process still bound to
  port 8000 — fixed by explicitly finding the PID via `netstat -ano | grep :8000` and
  force-killing it (`Stop-Process -Force`) before starting a fresh fog server instance
  each time. Neither of these was a bug in `edge_node.py` itself — both were artifacts of
  the local shell/background-process test environment.
- **Expected output:** Fog server's log should show received `/ingest` packets with
  populated `count`, `flow_magnitude`, and correct `zone_id`; edge node's own log should
  show alert banners printed in response to the fog server's hardcoded YELLOW alert.
- **Actual output (fog server side):**
  ```
  [FAKE FOG] received packet: count=2 flow_mag=0.1753 zone=zone_1
  [FAKE FOG] received packet: count=2 flow_mag=0.1793 zone=zone_1
  [FAKE FOG] received packet: count=2 flow_mag=0.1800 zone=zone_1
  ```
  Confirms the full pipeline — frame capture → CSRNet inference → optical flow →
  packet construction → HTTP POST → fog-side receipt — works end to end with real
  (non-mocked) model inference. `count=2` is plausible for a synthetic test video
  containing exactly one moving circle (CSRNet was not trained on synthetic shapes, so
  an exact count of 1 was not expected or required — the check here is "runs without
  crashing and produces a small, non-absurd number," matching the same standard used for
  the Phase 0 real-image smoke test).
- **Actual output (edge node side):** Alert banners printed repeatedly, each showing
  `ALERT [YELLOW]`, `ETA to impact: 4.2s`, and the fake server's message — confirming the
  alert-polling thread correctly parses the response JSON and triggers the
  console-warning code path for YELLOW-level alerts.
- **Status:** PASSED

### [2026-08-27 01:50] Task: Test 3 — graceful degradation when fog server is unreachable
- **Test:** Killed the fog server process, then ran `edge_node.py` against the same test
  video with `--fog-url http://localhost:8000` pointing at nothing.
- **Expected output:** No crash, no unhandled traceback; both the telemetry and alert
  threads should print `WARNING` messages and keep running/retrying indefinitely.
- **Actual output:**
  ```
  [zone_1] Loading CSRNet from ../weights/PartAmodel_best.pth ...
  [zone_1] Model loaded.
  [zone_1] Capture opened on source='test_scratch/test_video.mp4'. Running inference every 5 frames.
  [zone_1] WARNING: alert poll failed (ConnectTimeout). Will retry.
  [zone_1] WARNING: fog server unreachable (ConnectTimeout). Buffering packet and retrying.
  [zone_1] WARNING: alert poll failed (ConnectTimeout). Will retry.
  ```
  No traceback, no crash (`grep -i "traceback\|exception"` on the full log returned no
  matches). Capture loop continued running and consuming the video source normally while
  both networking threads independently logged warnings and kept retrying, exactly as
  the task brief requires ("Edge should never crash because fog is down").
- **Status:** PASSED

### [2026-08-27 02:00] Task: Test 4 — PacketBuffer unit test in isolation
- **Approach:** Rather than only testing the buffer's behavior indirectly through full
  network test runs (which make it hard to control exact timing of successes/failures),
  wrote a short standalone script importing `PacketBuffer` directly from `edge_node.py`
  and exercising it with synthetic packets.
- **Test:** Added 7 packets (`timestamp=0..6`) to a `PacketBuffer(maxlen=5)`. Verified the
  buffer correctly dropped the two oldest (`0`, `1`) and retained exactly `[2,3,4,5,6]`.
  Then simulated a partial send by calling `remove()` with only the first 3 of those 5
  packets (`[2,3,4]`), and verified the buffer correctly retained only the two unsent
  ones (`[5,6]`).
- **Expected output:** `len(snapshot) == 5` after overflow; `[2,3,4,5,6]` as contents;
  `[5,6]` remaining after removing `[2,3,4]`.
- **Actual output:**
  ```
  After adding 7 to maxlen=5 buffer, len = 5 -> should be 5
  Contents (should be timestamps 2..6): [2, 3, 4, 5, 6]
  After removing first 3 sent packets, remaining timestamps: [5, 6]
  PacketBuffer unit test PASSED
  ```
- **Status:** PASSED

### [2026-08-27 02:10] Task: Test 5 — recovery after fog server comes back online mid-run
- **Approach:** This is the most important resilience test, since it directly validates
  the "buffer and retry" requirement's actual purpose (not just "don't crash while down,"
  but "successfully deliver buffered telemetry once connectivity returns").
- **Test:** Started `edge_node.py` against the test video with the fog server already
  down. Waited 5 seconds (confirmed via log inspection that the telemetry thread had
  already logged at least one "fog server unreachable... buffering" warning, meaning at
  least one packet was sitting in the buffer unsent). Then started the fog server process
  mid-run, and let both processes continue for several more seconds.
- **Expected output:** Once the fog server comes online, the next telemetry loop tick
  (within 2 seconds) should successfully POST the buffered packet(s), and the fog
  server's log should show packets being received without needing to restart the edge
  node process.
- **Actual output (fog server log, captured after coming online mid-run):**
  ```
  [FAKE FOG] received packet: count=2 flow_mag=0.1806 zone=zone_1
  [FAKE FOG] received packet: count=2 flow_mag=0.1787 zone=zone_1
  [FAKE FOG] received packet: count=2 flow_mag=0.1800 zone=zone_1
  [FAKE FOG] received packet: count=2 flow_mag=0.1793 zone=zone_1
  [FAKE FOG] received packet: count=1 flow_mag=0.0000 zone=zone_1
  [FAKE FOG] received packet: count=2 flow_mag=0.1797 zone=zone_1
  ```
  Packets were received continuously once the fog server was reachable, with no manual
  intervention on the edge node side — confirming the telemetry thread's retry loop
  (which runs unconditionally every 2 seconds regardless of prior failures) picked up
  and successfully drained the buffer as soon as the network path became available again.
- **Status:** PASSED

### [2026-08-27 02:20] Task: Minor cleanup — remove unused import
- **Approach:** Noticed `import sys` at the top of `edge_node.py` was never referenced
  anywhere in the file (leftover from an earlier draft that considered using
  `sys.exit()` for error handling, which was ultimately handled via exceptions and the
  `finally` block instead).
- **Resolution:** Removed the unused `import sys` line. Re-ran `py_compile` to confirm no
  breakage.
- **Files touched:** modified `fog_edge_system/edge/edge_node.py`
- **Status:** DONE

### [2026-08-27 02:25] Task: Test environment cleanup
- **Approach:** Killed all lingering background processes started during testing
  (fake fog server instances on port 8000, edge node test runs still holding the loaded
  CSRNet model in memory) to avoid leaving orphaned processes running after this phase
  concludes. Verified via `tasklist | grep -i python` that no stray Python processes
  remained.
- **Resolution:** Deleted `fog_edge_system/edge/test_scratch/` entirely (test video,
  fake fog server script, and all generated log files) since none of it is part of the
  Phase 1 deliverable — it existed solely to validate `edge_node.py`'s behavior before
  handing it off. Also removed the `__pycache__/` directory generated by running the
  scripts.
- **Files touched:** deleted `fog_edge_system/edge/test_scratch/` (entire directory) and
  `fog_edge_system/edge/__pycache__/`
- **Status:** DONE

## Phase 1 — Summary

**Status: DONE.** Deliverables:
- `fog_edge_system/edge/config.py` — all tunable defaults (fog URL, inference interval,
  model path, resize dims, Farneback params, HTTP timeouts, buffer size), documented
  with reasoning for each threshold above.
- `fog_edge_system/edge/edge_node.py` — `EdgeNode` class running three concurrent loops
  (frame-paced capture+inference on the main thread; 2-second-cadence telemetry POST and
  alert-poll GET on two daemon background threads), plus a standalone `PacketBuffer`
  class implementing the "buffer last 5 unsent packets, retry, never crash" requirement.
  CLI interface matches the task brief's specified `--zone`, `--source`, `--fog-url`,
  `--model-path` arguments, plus an additional `--interval` (inference-every-N-frames,
  explicitly described as configurable in the brief) and `--device` (cuda/cpu override,
  consistent with the existing `CrowdCounter`'s own `device` parameter).
- Verified via 5 separate tests (syntax check, live end-to-end run against a fake fog
  server, graceful-degradation-when-down, isolated `PacketBuffer` unit test, and
  mid-run recovery) — all passed. No known bugs remaining.
- One pre-existing cosmetic issue observed again (not caused by this phase's code): the
  `!` warning banner and em-dash in the alert print statements render as `�` in raw
  console/log output on this Windows environment due to a non-UTF-8 console codepage —
  same class of issue as the `px²` → `px�` mangling observed in Phase 0. The underlying
  Python string data is correct UTF-8; this is a terminal/log encoding display issue, not
  a functional bug. Not fixed in this phase since it doesn't affect program behavior
  (only how the text renders when viewed with certain tools) — flagging for awareness in
  later phases if console encoding needs to be explicitly configured.

Ready for Phase 2 (Fog Server) pending user confirmation to proceed.

## Phase 2 — Fog Server and Pressure Cascade Detection Algorithm

### [2026-08-29 00:00] Task: Write fog/config.py — centralise every algorithm threshold
- **Approach:** Created the fog configuration module first, since both `zone_manager.py`
  and `cascade_detector.py` import thresholds from it. Contains every constant named in
  the Phase 2 specification, at the specified default values, each documented inline with
  what it controls and what changes if it is altered.
- **Threshold/constant decisions (all specified defaults preserved exactly):**
  - `SURGE_CONFIDENCE_THRESHOLD = 0.6` — DTW-derived confidence a zone must exceed to be
    called surging. Raising it reduces false surge detections but risks missing real
    buildups. Verified empirically (see Test 1): a textbook sigmoid buildup scores 1.000
    and a flat/noisy/falling curve scores 0.000, so 0.6 sits in a wide empty margin
    between the two populations rather than near either — the value is not delicately
    balanced and could move anywhere in roughly 0.1–0.85 without changing test outcomes.
  - `MIN_DENSITY_THRESHOLD = 0.3` — absolute density gate applied in addition to the
    shape test. This exists because DTW matching is scale-invariant (see the known
    limitation documented in `compute_surge_confidence`), so a tiny harmless ripple has
    the same normalized shape as a dangerous buildup. The two conditions are deliberately
    independent: "is this the right shape" and "is this actually dense".
  - `WAVE_CORRELATION_THRESHOLD = 0.7` — minimum cross-correlation for two adjacent zones
    to count as a real wave rather than coincidence. Verified: a true delayed copy scores
    0.962 while two independent random series score |−0.534|, so 0.7 separates the two
    with margin on both sides.
  - `MAX_DTW_DISTANCE = 5.0` — normalization constant converting DTW distance to
    confidence via `1 − (distance / MAX_DTW_DISTANCE)`. Larger = more forgiving (more
    zones register as surging); smaller = stricter.
  - `PREEMPTIVE_ALERT_WINDOW = 30` seconds — the preemptive warning horizon. Larger gives
    earlier warnings but more false alarms.
  - `CASCADE_HISTORY_WINDOW = 60` seconds and `DATA_INTERVAL = 2` seconds — as specified.
  - `HISTORY_BUFFER_SIZE` — **derived**, not hardcoded, as
    `CASCADE_HISTORY_WINDOW // DATA_INTERVAL` = 30 points. Chosen to be derived so that
    changing the history window or the telemetry cadence cannot silently desynchronise
    the buffer depth from the time window it is supposed to represent.
  - `MAX_CORRELATION_LAG = 10` — ±10 data points = ±20 seconds of propagation delay
    searched, matching the specification's stated lag range.
  - `REFERENCE_CURVE_POINTS = 15` / `REFERENCE_CURVE_STEEPNESS = 10.0` — the reference
    S-curve's resolution and sigmoid steepness, matching the specified
    `np.linspace(0,1,15)` and `-10*(t-0.5)` formulation.
  - `MIN_POINTS_FOR_SURGE_CHECK = 5` / `MIN_POINTS_FOR_CORRELATION = 10` — extracted from
    the specification's inline `len(series) < 5` and `len(...) < 10` guards into named
    constants, so the "how much data before we trust this" policy is tunable in one place
    rather than buried as magic numbers inside the detection loop.
  - `SURGE_START_DENSITY_THRESHOLD` — defined as equal to `MIN_DENSITY_THRESHOLD` rather
    than as an independent literal, so that "dense enough to be surging" (used by the
    detector) and "surge episode has started" (used by `get_surge_start_time`, which
    anchors the ETA calculation) agree by default. If these two disagreed, a zone could
    be treated as surging by the detector while having no recorded surge start time,
    which would make Component 3 silently unable to compute an ETA for it.
  - `LOG_ONLY_DETECTED_CASCADES = True` — the detector runs on every ingest (several
    times per second under three live edge nodes), so logging every run would produce an
    enormous, almost entirely negative log. Logging only positives keeps the cloud-tier
    event log signal-dense.
- **Files touched:** created `fog_edge_system/fog/config.py`
- **Status:** DONE

### [2026-08-29 00:05] Task: Write shared/models.py — the edge/fog data contract
- **Approach:** Defined Pydantic models describing the telemetry packet, history entry,
  zone state, alert response, and the full cascade result structure.
- **Decision made:** `TelemetryPacket` mirrors exactly what Phase 1's
  `build_telemetry_packet()` emits, so the two phases have a single written contract
  rather than an implicit one. Placed in `shared/` (not `fog/` or `edge/`) because both
  tiers and the Phase 4 demo harness need to agree on this shape.
- **Obstacle:** `WavePair` needs fields literally named `from` and `to` to match the
  specified JSON output, but `from` is a reserved Python keyword and cannot be an
  attribute name.
- **Resolution:** Used Pydantic's field aliasing — attributes are `from_zone`/`to_zone`
  with `alias="from"`/`alias="to"` and `populate_by_name = True`, so the serialized JSON
  carries the specified `from`/`to` keys while the Python attribute names remain legal.
- **Note:** The fog server ultimately builds its cascade result as plain dicts rather
  than instantiating these models on the hot path (see the `/ingest` entry below for why);
  `shared/models.py` serves as the documented, importable schema definition and is used
  for validation and by the demo harness.
- **Files touched:** created `fog_edge_system/shared/models.py`
- **Status:** DONE

### [2026-08-29 00:10] Task: Write fog/zone_manager.py
- **Approach:** Implemented `ZoneManager` holding, per zone: a `deque` rolling history
  capped at `HISTORY_BUFFER_SIZE` (30 points = 60 seconds), the latest telemetry entry,
  and the current surge episode start time. Implemented all specified accessors
  (`get_density_timeseries`, `get_flow_vector`, `get_surge_start_time`,
  `get_all_zone_ids`, `get_adjacent_zones`, `get_latest`) plus `get_history` and
  `snapshot_all_zones` for the HTTP layer.
- **Design decision — `threading.RLock` guarding all access:** FastAPI serves requests
  concurrently, so an `/ingest` write can overlap a `/zones` or `/cascade` read. Chose a
  reentrant lock (`RLock`) over a plain `Lock` specifically because composite methods
  call other locked methods — `snapshot_all_zones()` calls `get_all_zone_ids()`,
  `get_latest()`, and `get_adjacent_zones()` while already holding the lock. With a plain
  `Lock` this would self-deadlock on the first `/zones` request. This was anticipated and
  designed for rather than discovered as a runtime hang.
- **Design decision — `get_density_timeseries` returns `np.ndarray`, not a list:** Every
  consumer (normalization, DTW, cross-correlation) performs vectorized arithmetic. The
  Phase 2 brief itself calls out this exact class of bug as the example to avoid
  ("`compute_wave_correlation` was receiving a list instead of numpy array"), so the
  conversion is done once at the source with an explicit `dtype=float`, and the method
  returns an empty float array (not `None`) for unknown zones so callers can uniformly
  use `len()` and slicing without a `None` check.
- **Design decision — `get_flow_vector` returns `(0.0, 0.0)` for unknown zones, not
  `None`:** A zero vector makes the Component 2 flow-alignment dot product evaluate to
  exactly 0.0, which fails the `> 0` aligned test. So a zone with no flow data can never
  falsely confirm a wave — the safe default falls out of the arithmetic rather than
  requiring a special case at the call site.
- **Design decision — surge episode rising/falling edge tracking:** `_update_surge_episode`
  records the timestamp only on the *rising* edge (density crosses the threshold while no
  episode is active) and clears it on the *falling* edge. This matters for Component 3:
  the ETA is measured from when the upstream zone *started* surging, so the start time
  must not drift forward while the episode continues, and a new rise after a drop must be
  treated as a genuinely new episode rather than continuing the old one.
- **Design decision — `get_all_zone_ids` returns the union of configured and reporting
  zones:** Rather than only the adjacency map's keys. This means a configured-but-silent
  zone still appears in the dashboard as GREEN instead of vanishing, and an unconfigured
  zone that starts reporting is still tracked and evaluated rather than being silently
  discarded.
- **Files touched:** created `fog_edge_system/fog/zone_manager.py`
- **Status:** DONE

### [2026-08-29 00:20] Task: Write fog/cascade_detector.py — the novel algorithm
- **Approach:** Implemented all three components plus the orchestrating `detect_cascade`,
  following the specification's structure and formulas.
- **Design decision — flat-series guard in `normalize_series`:** The specification says to
  normalize both curves to [0,1], which is min-max normalization: `(x - min) / (max - min)`.
  A perfectly flat series has `max == min`, so the divisor is zero, producing NaN. NaN
  then propagates silently through DTW (every comparison with NaN is False, so the `min()`
  in the DTW recurrence misbehaves) and poisons the confidence score without raising any
  error. Added an explicit `span <= 0` guard returning all-zeros. Chose zeros rather than
  0.5 or leaving the input unchanged because a perfectly flat density curve is the exact
  opposite of a surge, so mapping it to the floor of the range is semantically correct —
  and it makes the flat case score 0.000 confidence, which Test 1 confirms.
- **Design decision — reference curve built once at import:** `REFERENCE_CURVE` is a
  module-level constant produced by `_build_reference_curve()`, not rebuilt per call.
  `detect_cascade` runs on every ingest (several times per second), and the curve never
  changes, so rebuilding it per zone per call would be pure waste.
- **Design decision — separate `normalize_series` helper:** Rather than inlining
  normalization, since it is applied to both the reference curve and every zone series
  and needed identical zero-span handling in both places.
- **Files touched:** created `fog_edge_system/fog/cascade_detector.py`
- **Status:** DONE (one significant bug found in testing — see below)

### [2026-08-29 00:35] Task: Test 1 — Component unit tests (DTW, correlation, flow, ETA)
- **Test approach:** Wrote a standalone script exercising each component in isolation
  against hand-constructed inputs with known correct answers, rather than only testing the
  full pipeline, so that a failure localises to a specific component.
- **Component 1 results — surge confidence (threshold 0.6):**
  | input curve | confidence | expectation |
  |---|---|---|
  | sigmoid buildup | 1.000 | HIGH — correct |
  | linear ramp | 0.895 | moderately high |
  | flat | 0.000 | LOW — correct |
  | random noise | 0.000 | LOW — correct |
  | **falling curve** | **0.000** | LOW — correct |
  The falling-curve case was included deliberately because min-max normalization is
  scale-invariant, raising the concern that a *draining* zone (density collapsing) might
  normalize into something resembling the reference. It scores 0.000, confirming DTW
  distinguishes rising from falling shape — the concern does not materialise.
  Also verified: `DTW(reference, reference) == 0.0` exactly, flat input produces no NaN,
  and an empty series is handled.
- **Component 3 results — ETA arithmetic:** delay 20s with upstream surging 5s ago →
  15.0s; surging 25s ago → 0.0s (correctly clipped, not negative); upstream not surging
  (`None`) → `None`. All correct.
- **Status:** PASSED (Components 1 and 3); Component 2 FAILED — see next entry.

### [2026-08-29 00:40] Task: BUG — cross-correlation lag sign convention inverted
- **Expectation vs actual:** Test 2 fed `compute_wave_correlation` two series where
  `series_b` was constructed as an exact copy of `series_a` delayed by 4 data points —
  an unambiguous A→B wave. The specification states "A positive lag means zone B's
  pattern follows zone A's", and `detect_cascade` gates wave confirmation on `lag > 0`.
  **Expected:** `lag = +4`. **Actual:** `lag = -2`. Assertion
  `assert lag > 0, "lag must be positive (B follows A)"` failed.
- **Severity:** This is not cosmetic. Because `detect_cascade` requires `lag > 0` to
  confirm a wave, an inverted sign convention means **every genuine A→B pressure wave
  would be rejected**, and the algorithm's central capability would silently never fire.
  The correlation magnitude was correct (0.962) — only the sign/direction was wrong,
  which is exactly the kind of failure that would pass a naive "is correlation high?"
  test and still render the system non-functional.
- **Root cause:** The lag indexing given in the specification's pseudocode is
  `corr = np.mean(a[lag:] * b[:len(b)-lag])` for `lag >= 0`. This pairs a **later**
  sample of A with an **earlier** sample of B, so the correlation peaks at positive lag
  when **A follows B** — the opposite of the documented semantics.
- **Diagnosis method:** Rather than guessing, printed the full correlation-vs-lag profile
  for both the specified indexing and the flipped indexing against the known
  ground-truth input (B delayed by 4):
  ```
  spec formula  mean(a[lag:]*b[:len-lag]):     peak at lag = -2   (corr +0.962)
  flipped       mean(a[:len-lag]*b[lag:]):     peak at lag = +2   (corr +0.962)
  ```
  This confirmed the two formulations are mirror images and that the flipped one matches
  the documented "positive lag = B follows A" convention.
- **Resolution:** Changed the indexing in `compute_wave_correlation` so that positive lag
  pairs an earlier sample of A with a later sample of B:
  ```python
  if lag >= 0:
      seg_a = a[: len(a) - lag] if lag > 0 else a
      seg_b = b[lag:]
  else:
      seg_a = a[-lag:]
      seg_b = b[: len(b) + lag]
  ```
  Also corrected the bounds guards, which in the original referenced the wrong array for
  each branch (`lag >= len(a)` where the sliced array is now `b`, and vice versa).
  Added an explanatory comment at the fix site recording why the convention matters, so
  the inverted form is not "helpfully" restored later.
- **Retest:** `corr=0.962  lag=2 (4s)` — positive, correct direction. Independent random
  series: `corr=-0.534` (below the 0.7 threshold, correctly rejected). Two flat series:
  `corr=0.000`, no NaN.
- **Files touched:** modified `fog_edge_system/fog/cascade_detector.py`
- **Decision made:** Deviated from the specification's literal pseudocode here. The
  specification's *prose* ("A positive lag means zone B's pattern follows zone A's") and
  its *pseudocode* contradict each other; the prose describes the intended behaviour and
  the `lag > 0` gate in `detect_cascade` depends on it, so the prose was treated as
  authoritative and the pseudocode as containing an indexing error.
- **Status:** DONE — fixed and verified

### [2026-08-29 00:45] Task: Document lag-magnitude estimation bias
- **Observation:** After the sign fix, the recovered lag for a ground-truth 4-point delay
  is `+2` points, not `+4` — correct direction, under-estimated magnitude.
- **Root cause (analysed, not a defect to fix):** Each candidate lag is scored as the
  mean over only the overlapping region of the two series, and that overlap shrinks as
  |lag| grows. Longer shifts are therefore scored over fewer, and differently weighted,
  samples, biasing the arg-max toward smaller shifts. This is inherent to the
  specification's mean-of-overlap formulation.
- **Decision made:** Documented rather than changed. Switching to a full normalized
  cross-correlation with consistent denominators would recover magnitude more accurately,
  but would deviate further from the specified algorithm, and the bias is benign in this
  application: direction (the sign) is what gates detection and is recovered reliably,
  while the magnitude feeds only the ETA estimate. An under-estimated propagation delay
  produces a *shorter* predicted ETA, i.e. the system warns **earlier** than reality
  rather than later — the safe direction to err for a crowd-safety alarm. Recorded this
  reasoning in the `compute_wave_correlation` docstring.
- **Files touched:** modified `fog_edge_system/fog/cascade_detector.py` (docstring)
- **Status:** DONE

### [2026-08-29 00:50] Task: Deviation from spec — added an ETA gate in Component 3
- **Observation:** The specification's prose states: "If `eta_seconds` <
  `PREEMPTIVE_ALERT_WINDOW` (default 30 seconds), send a preemptive RED alert to that
  downstream zone." However, the specification's pseudocode for `detect_cascade`
  unconditionally writes `results["preemptive_alerts"][dz]` and sets the zone RED with no
  reference to `PREEMPTIVE_ALERT_WINDOW` at all — the constant is defined and described
  but never used in the given code.
- **Resolution:** Implemented the prose behaviour: `if eta is None or eta >
  config.PREEMPTIVE_ALERT_WINDOW: continue`. Without this gate, `PREEMPTIVE_ALERT_WINDOW`
  would be dead configuration, and any downstream zone of any confirmed wave would go RED
  regardless of how far in the future its predicted arrival was — making the alert level
  far less informative.
- **Additional handling not in the spec — converging waves:** If multiple confirmed waves
  predict arrival at the same downstream zone, the pseudocode would let the
  last-evaluated one overwrite the others arbitrarily. Added a comparison keeping the
  **most urgent** (smallest ETA), since that is the actionable number for a safety
  operator.
- **Files touched:** modified `fog_edge_system/fog/cascade_detector.py`
- **Status:** DONE

### [2026-08-29 01:00] Task: Test 2 — full-algorithm scenario tests (the central claim)
- **Test approach:** The algorithm's central claim is that it distinguishes a *real
  propagating pressure wave* from *coincidental independent spikes*. Testing only the
  positive case would not substantiate that claim, so four negative scenarios were
  constructed, each designed to defeat a different one of the three confirmation
  conditions in isolation.
- **Results:**

  | # | Scenario | Result | Verdict |
  |---|---|---|---|
  | A | All zones flat, low density | no cascade, all GREEN | PASS |
  | B | **Real cascade:** zone_1 surge, zone_2 same surge lagged 4pts, flow aligned | `cascade_detected=True`, score 0.8634, wave `zone_1→zone_2` corr 0.962 lag 4s, both RED, zone_3 preemptive RED | PASS |
  | C | **Coincidence:** zone_1 and zone_3 both surge, non-adjacent, zone_2 quiet | both YELLOW, **0 wave pairs**, no cascade | PASS |
  | D | **Zero lag:** adjacent zones surge simultaneously | both YELLOW, **0 wave pairs** (`lag > 0` correctly fails) | PASS |
  | E | **Opposing flow:** correlated and correctly lagged, but flow vectors oppose | both YELLOW, **0 wave pairs** (flow alignment vetoes) | PASS |
  | F | Insufficient data (3 points) | no crash, GREEN, no cascade | PASS |

- **Significance of C, D, E:** These are the scenarios that separate this algorithm from a
  simple threshold check. In all three, density is high enough and shaped correctly for
  both zones to be individually flagged as surging (YELLOW) — a naive threshold detector
  would report a cascade in every one. The algorithm correctly withholds wave
  confirmation in each, for three *different* reasons: no adjacency (C), no propagation
  delay (D), and physically inconsistent crowd direction (E). Each of the three
  confirmation conditions was therefore shown to be independently load-bearing.
- **Scenario B preemptive ETA reported 0.0s** — investigated separately below.
- **Status:** ALL PASSED

### [2026-08-29 01:10] Task: Investigate Scenario B's 0.0s ETA — is the prediction real?
- **Expectation vs actual:** Scenario B produced a preemptive alert for zone_3 with
  `eta_seconds = 0.0`. The assertion passed, but a permanently-zero ETA would make
  Component 3 useless as a *prediction* — it would only ever say "now", never "in N
  seconds". Investigated rather than accepting the passing test.
- **Diagnosis:** Instrumented the surge episode timing directly. zone_2's surge episode
  began at t=1000030.0 while the scenario evaluated at t=1000060.0, giving
  `elapsed = 30s` against a `propagation_delay` of only 4s — so the predicted arrival had
  already passed and the clip to 0.0 was arithmetically correct. This was an artefact of
  the test feeding a full 30-point history and then evaluating at the very end, not a
  defect in the ETA logic.
- **Confirmation:** Re-evaluated the identical zone state at `now = surge_start + 2s`
  (i.e. while the wave is genuinely still in flight) — the ETA came out as **2.0s**, a
  real forward-looking prediction. The arithmetic is sound; only the test's chosen
  evaluation instant produced the zero.
- **Resolution:** No code change. Documented the three distinct ETA states in the
  `compute_eta` docstring so the meanings are not conflated later: a **positive** ETA
  means "arriving in N seconds"; **0.0** means "predicted arrival window has passed while
  the upstream zone is still surging — impact imminent"; **None** means "no prediction
  possible (upstream zone has no active surge episode)". Live operation produces positive
  ETAs because detection runs continuously from the moment the upstream zone starts
  surging, rather than being evaluated once at the end of a pre-loaded history.
- **Files touched:** modified `fog_edge_system/fog/cascade_detector.py` (docstring)
- **Status:** DONE

### [2026-08-29 01:20] Task: Test 3 — ZoneManager unit tests
- **Results:**
  - Fed 50 points into a 30-slot buffer → length capped at exactly 30, ordered
    oldest-first. PASS
  - `get_density_timeseries` returns `ndarray` with `dtype=float64`. PASS
  - Unknown zone: empty array, `(0.0, 0.0)` flow, `None` latest — no exceptions. PASS
  - **Surge episode lifecycle** (threshold 0.3), the sequence Component 3 depends on:
    density 0.1 → `None`; rises to 0.5 at t=1002 → start recorded as 1002.0; rises
    further to 0.7 at t=1004 → start **unchanged** at 1002.0 (correctly does not drift);
    drops to 0.1 at t=1006 → cleared to `None`; rises again to 0.6 at t=1008 → **new**
    start 1008.0. PASS — rising/falling edge semantics correct in both directions.
  - Adjacency correct; configured-but-silent zones still listed; an unconfigured zone
    (`zone_99`) that reports telemetry is tracked. PASS
- **Status:** ALL PASSED

### [2026-08-29 01:30] Task: Write fog/fog_server.py
- **Approach:** FastAPI application implementing all seven specified endpoints, with a
  module-level `ZoneManager`, a module-level `latest_cascade` result, and a
  `ConnectionManager` for dashboard WebSocket clients.
- **Design decision — `/ingest` accepts a plain `dict`, not a strict Pydantic model:**
  A strict model returns HTTP 422 on any malformed field. Because Phase 1's edge node
  buffers and *retries* failed packets, a packet that will never validate would be
  retried indefinitely, wasting the buffer on a permanently-doomed payload. Instead the
  endpoint validates the three genuinely required fields (`zone_id`, `density_avg`,
  `timestamp`), returns 400 for those, and coerces every optional field with a default —
  so one bad optional field degrades that single packet rather than jamming the retry
  loop. `shared/models.py` remains the documented schema.
- **Design decision — detection runs on every ingest, not on a timer:** So that an edge
  node polling `/alert/<zone_id>` immediately after POSTing sees a decision that already
  incorporates its own most recent reading. The specification requires this ("Triggers
  cascade detection after every ingest").
- **Design decision — `/alert/<unknown_zone>` returns GREEN, not 404:** An edge node
  starting before its first telemetry is processed should receive a valid quiet response
  rather than an error it must special-case. Phase 1's alert poller only acts on
  `level in ("YELLOW","RED")`, so GREEN is correctly inert. Verified in Test 4.
- **Design decision — WebSocket push loop is a single shared background task, not
  per-connection:** One `broadcast_loop()` task started in the FastAPI `lifespan`
  handler pushes to all connected dashboards every `WS_BROADCAST_INTERVAL_SEC`. A
  per-connection timer would multiply the work by the client count and let clients drift
  out of sync with each other. Used the modern `lifespan` context manager rather than the
  deprecated `@app.on_event("startup")`.
- **Design decision — broadcast tolerates individual client failures:** `broadcast()`
  collects failed sockets and prunes them after the send loop rather than raising, so one
  dashboard closing its tab mid-send cannot interrupt delivery to the others or kill the
  broadcast loop. The loop additionally catches and logs any per-cycle exception (while
  re-raising `CancelledError` so shutdown still works) so a single bad cycle cannot
  permanently stop live updates.
- **Design decision — the initial WebSocket snapshot is sent immediately on connect:**
  Otherwise a freshly opened dashboard renders blank for up to 2 seconds until the next
  broadcast tick.
- **Design decision — cascade log write failures are swallowed with a warning:**
  `log_cascade_event` runs inside the request path; a full disk or a permissions error
  must never take down live crowd monitoring for the sake of a log line.
- **Files touched:** created `fog_edge_system/fog/fog_server.py`
- **Status:** DONE

### [2026-08-29 01:35] Task: Create placeholder dashboard/index.html
- **Approach:** `GET /` serves `dashboard/index.html`, but the real dashboard is Phase 3
  work. Created a minimal placeholder page listing the available endpoints so that `GET /`
  is testable now and the server does not 404 on its root route.
- **Decision made:** Kept deliberately minimal — no styling system, no JS, no WebSocket
  client — to avoid pre-empting Phase 3's design decisions. It will be replaced wholesale.
- **Files touched:** created `fog_edge_system/dashboard/index.html`
- **Status:** DONE (placeholder; Phase 3 replaces it)

### [2026-08-29 01:40] Task: Test 4 — live HTTP and WebSocket endpoint tests
- **Dependencies installed:** `httpx` (HTTP client for the test), `websockets` (WebSocket
  client) into the existing `fog_edge_system/venv/`.
- **Test approach:** Started the real `fog_server.py` under uvicorn, then drove all seven
  endpoints from a separate process, feeding the Scenario B cascade over real HTTP.
- **Results:**
  - `GET /` → 200, 971 bytes HTML. PASS
  - `POST /ingest` with `{"zone_id":"zone_1"}` only → **400**, `missing fields:
    ['density_avg','timestamp']`. PASS (malformed input rejected explicitly)
  - Fed 30 points/zone over HTTP; final ingest response
    `{'status':'ok','zone_id':'zone_3','alert_level':'RED','cascade_detected':True}`
  - `GET /cascade` → `detected=True score=0.8634`, wave `zone_1→zone_2 corr=0.962 lag=4s`,
    preemptive alert on zone_3. **The full algorithm works end-to-end over HTTP**, not
    just in-process. PASS
  - `GET /zones` → 3 zones, each `history_length=30`, correct adjacency and densities. PASS
  - `GET /alert/zone_1` → RED; `zone_3` → RED with `eta=0.0` and the preemptive message
    naming the triggering wave. PASS
  - `GET /alert/does_not_exist` → GREEN, no 404. PASS
  - `GET /history/zone_1` → 30 entries, first density 0.150 rising to 0.849 (the buildup
    is visible in the returned series). `GET /history/unknown_zone` → 0, no crash. PASS
  - `WS /ws` → immediate snapshot with keys `['cascade','timestamp','zones']`, then a
    periodic push received 1.1s later (broadcast interval 2.0s; the shorter gap is
    expected since the initial snapshot is sent on connect, not aligned to the tick). PASS
  - Server log inspected: the only WARNING is the intentional malformed-packet rejection
    from the test. No errors, no tracebacks.
  - `cascade_events.jsonl` accumulated 20 entries — only detected cascades, confirming
    `LOG_ONLY_DETECTED_CASCADES` filtering works.
- **Status:** ALL PASSED

### [2026-08-29 01:50] Task: Test 5 — cross-phase integration (real Phase 1 edge node → real Phase 2 fog server)
- **Test approach:** Every prior test used synthetic telemetry. This test ran the actual
  unmodified `edge/edge_node.py` from Phase 1 — real CSRNet inference on real video
  frames, real Farneback optical flow — against the actual `fog/fog_server.py`, to verify
  the contract between the two phases holds in practice rather than only on paper.
- **Obstacle:** First attempt wrote the generated test video to `/tmp_test_video.mp4`,
  which under Git Bash on Windows resolved to a path the subsequent `cv2.VideoCapture`
  call could not open; the edge node started, found no readable source, and exited, so
  the fog server showed `no data` for all zones.
- **Resolution:** Regenerated the video via `os.path.abspath()` into the `edge/`
  directory itself and passed a relative path. Confirmed the file existed (49,751 bytes)
  before re-running.
- **Result:** The fog server received real telemetry:
  ```
  zone_1: hist=8 count=2 dens=0.0065 flow=(0.1801,-0.0000)
  ```
  8 packets stored, non-zero optical flow on the x-axis (matching the test video's
  horizontally-moving object), and `GET /alert/zone_1` → `GREEN` — the correct assessment,
  since one small moving shape is not a stampede. No errors in either the edge node log or
  the fog server log.
- **Significance:** Confirms Phase 1's telemetry packet format is accepted verbatim by
  Phase 2's `/ingest`, that the alert-polling round trip works against the real endpoint,
  and that the system correctly reports GREEN on benign input rather than only being
  tested on inputs engineered to trigger it.
- **Files touched:** none (test-only; artefacts deleted afterwards)
- **Status:** PASSED

### [2026-08-29 02:00] Task: Test environment cleanup
- **Approach:** Killed all background uvicorn/fog server processes (located via
  `netstat -ano | grep :8000`, terminated with `Stop-Process -Force`) and verified via
  `tasklist | grep -i python` that no stray Python processes remained. Deleted the
  generated test video, edge node test log, `fog/test_scratch/` (all test scripts and
  server logs), the `cascade_events.jsonl` produced during testing, and all `__pycache__`
  directories.
- **Note on the port-8000 conflicts encountered repeatedly during testing:** backgrounded
  processes in this shell environment do not reliably terminate with their parent, so a
  stale uvicorn instance would remain bound to port 8000 and cause the next server start
  to fail with `[Errno 10048] only one usage of each socket address ... is normally
  permitted`. Resolved each time by explicitly finding and killing the holding PID before
  starting a fresh instance. Same class of environment issue documented in Phase 1 — not
  a defect in the server code.
- **Files touched:** deleted `fog_edge_system/fog/test_scratch/`,
  `fog_edge_system/fog/cascade_events.jsonl`, `fog_edge_system/edge/itest_video.mp4`,
  `fog_edge_system/edge/test_edge.log`, and all `__pycache__/` directories
- **Status:** DONE

## Phase 2 — Summary

**Status: DONE.** Deliverables:
- `fog/config.py` — every algorithm threshold at the specified defaults, each documented
  with its effect and, where tested, the empirical margin justifying the value.
- `fog/zone_manager.py` — thread-safe (`RLock`) per-zone rolling history (30 points /
  60s), latest telemetry, adjacency map, numpy-typed density series, flow vectors, and
  rising/falling-edge surge episode tracking.
- `fog/cascade_detector.py` — the Pressure Cascade Detection Algorithm: DTW temporal
  pattern matching against a reference S-curve, lagged cross-correlation with flow-
  direction confirmation, and propagation-delay ETA prediction with preemptive alerting.
- `fog/fog_server.py` — FastAPI server exposing all seven required endpoints plus a
  shared-task WebSocket broadcast loop.
- `shared/models.py` — the documented edge/fog data contract.
- `dashboard/index.html` — placeholder only; Phase 3 replaces it.

**One significant bug found and fixed:** the cross-correlation lag sign convention was
inverted relative to the documented semantics, which would have caused **every genuine
pressure wave to be rejected** by the `lag > 0` gate. Found by testing against a
ground-truth delayed-copy input rather than by inspection. Root cause was an indexing
error in the specification's pseudocode, which contradicts the specification's own prose;
the prose was treated as authoritative. Documented in full above.

**Two deliberate deviations from the specification's pseudocode**, both documented above
with reasoning: (1) the corrected lag indexing, and (2) an ETA gate against
`PREEMPTIVE_ALERT_WINDOW`, which the prose requires but the pseudocode omits — without it
the constant would be dead configuration.

**One documented limitation, not fixed:** the recovered propagation lag under-estimates
the true delay in magnitude (direction is reliable) because each lag is scored over a
shrinking overlap region. This biases predicted ETAs *earlier* than reality — the safe
direction for a safety alarm — and is recorded in the `compute_wave_correlation` docstring.

**Validation:** 5 test suites, all passing — component unit tests (DTW / correlation /
flow alignment / ETA), six full-algorithm scenarios including four negative cases that
each defeat a different confirmation condition, ZoneManager state and surge-episode
lifecycle tests, live HTTP + WebSocket endpoint tests, and a cross-phase integration test
running the real Phase 1 edge node against the real Phase 2 fog server. The three negative
scenarios (non-adjacent coincidence, zero-lag simultaneity, opposing flow) are the ones
substantiating that this is a genuine cascade detector rather than a threshold check: in
all three, both zones are individually flagged as surging, and the algorithm correctly
withholds wave confirmation for three different reasons.

Ready for Phase 3 (Dashboard) pending user confirmation to proceed.

## Phase 3 — Dashboard

### [2026-08-29 03:00] Task: Determine whether the existing API can supply Row 4's data
- **Approach:** Before writing any HTML, checked what the Phase 2 `/cascade` payload
  actually contains against what the Row 4 "wave correlation debug panel" specification
  asks for: raw cross-correlation values between zone pairs, DTW distances per zone, and
  the reference S-curve vs the actual density curve.
- **Obstacle:** None of the three were available.
  1. `zone_details` carried `surge_confidence` but **not** the underlying DTW distance —
     the distance is computed inside `compute_surge_confidence` and discarded.
  2. Correlations were recorded **only for confirmed waves**: the Component 2 loop does
     `if not is_wave: continue` *before* appending to `wave_pairs`, so a rejected pair
     left no trace in the output at all.
  3. The reference curve existed only as a module-level constant in Python, never
     serialized to the client.
  Additionally, Component 2 only iterated over `results["surge_zones"]`, so during a
  quiet demo — the common case when a teacher asks "how does this work?" — **no pairs
  would be evaluated at all** and Row 4 would render permanently empty.
- **Resolution:** Added a `debug` sub-object to the `detect_cascade` result carrying
  `dtw` (per zone), `pairs` (every evaluated pair), `reference_curve`, and `thresholds`.
  Three specific changes to `cascade_detector.py`:
  - **Component 1:** after computing confidence, recompute and record the raw
    `dtw_distance`, the `normalized_curve` actually fed to DTW, and the two gate results
    (`passed_confidence`, `passed_density`) separately. Recording the two gates
    independently is what lets the panel show *which* of the two conditions failed —
    the specification's own framing of DTW being scale-invariant means "shape passed but
    density failed" is a meaningful and distinct state worth surfacing.
  - **Component 2:** changed the outer loop from `for zone_a in results["surge_zones"]`
    to `for zone_a in zone_manager.get_all_zone_ids()`, and introduced an explicit
    `source_is_surging` flag that is now one of the four conditions in `is_wave`. Every
    adjacent pair is therefore *evaluated and recorded* for inspection, while wave
    *confirmation* still requires the source zone to be surging — behaviour is unchanged,
    but the reasoning is now visible.
  - Recorded a human-readable `rejected_because` string naming the first condition that
    vetoed each pair.
- **Decision made — why record rejected pairs at all:** The rejected rows are the more
  informative case for the stated purpose of the panel. A confirmed wave only shows the
  algorithm firing; a rejected pair shows *which of the four conditions* prevented a
  false positive, which is precisely the distinction between this algorithm and a naive
  threshold check. This directly surfaces the Phase 2 negative-scenario results (C, D, E)
  as live, inspectable output rather than as claims in a document.
- **Files touched:** modified `fog_edge_system/fog/cascade_detector.py`
- **Status:** DONE

### [2026-08-29 03:10] Task: Prevent the debug payload from bloating the cloud-tier log
- **Obstacle:** `log_cascade_event` serializes the entire `results` dict to
  `cascade_events.jsonl`. With the new `debug` key that meant writing the constant
  15-point reference curve, plus a 15-point normalized curve per zone, plus every
  evaluated pair, on **every** logged detection — a large volume of repeated, non-archival
  data appended several times per second during a sustained cascade.
- **Resolution:** `log_cascade_event` now builds `record = {k: v for k, v in
  results.items() if k != "debug"}` and logs that. The debug payload exists to explain the
  *current* state to a live dashboard; the durable log keeps the decision and its inputs.
- **Verification:** Confirmed a written log line contains keys `['cascade_detected',
  'cascade_score', 'preemptive_alerts', 'surge_zones', 'timestamp', 'wave_pairs',
  'zone_alerts', 'zone_details']` and **no** `debug` key.
- **Files touched:** modified `fog_edge_system/fog/cascade_detector.py`
- **Status:** DONE

### [2026-08-29 03:15] Task: Regression-test Phase 2 scenarios after modifying the detector
- **Rationale:** The Component 2 loop restructuring (iterating all zones instead of only
  surging ones, and adding `source_is_surging` into the `is_wave` condition) touches the
  core detection path. Re-ran all six Phase 2 scenarios to confirm detection behaviour was
  genuinely unchanged and not merely assumed to be.
- **Result:** A (quiet) / B (real cascade) / C (non-adjacent coincidence) / D (zero lag) /
  E (opposing flow) / F (insufficient data) — **all six still pass**, with identical
  verdicts to Phase 2. The restructure is inspection-only, as intended.
- **Status:** PASSED

### [2026-08-29 03:25] Task: Write dashboard/index.html
- **Approach:** Single self-contained HTML file with embedded CSS and vanilla JavaScript.
  No React, no build tooling, no npm, no chart library, no external requests of any kind —
  the file is served directly by the fog server's `GET /` and works offline on a LAN.
- **Design decision — chart histories accumulate client-side from the WebSocket payload:**
  The WS/`/zones` payload already carries each zone's `latest` reading every 2 seconds, so
  the chart appends from that stream and drops points older than 60 seconds. The
  alternative — re-fetching `GET /history/{zone_id}` for all three zones on every tick —
  would triple the request volume for data the client is already receiving. `/history` is
  still used **once** at page load (`seedHistories()`) so the chart is populated
  immediately from server-side history instead of taking 60 seconds to fill in.
- **Design decision — arrow rotation negates flow_y:** Canvas/CSS screen coordinates grow
  downward while the optical flow vector's y is reported in the same downward-positive
  convention from OpenCV; `Math.atan2(-fy, fx)` makes the on-screen arrow point the way
  the crowd is actually moving rather than mirrored vertically. Verified visually in the
  mixed-state test, where three zones with deliberately different flow vectors
  ((1.0, 0.0), (-0.5, 0.6), (0.2, -0.9)) rendered as right / up-left / down respectively.
- **Design decision — chart line colour is driven by alert level, not a fixed per-zone
  palette:** The specification requires this ("Color matches zone alert level (green line
  turns red when zone goes RED)"). A consequence worth noting: when every zone is RED all
  three lines are the same colour and the legend cannot distinguish them. This is the
  specified behaviour and it is arguably the right trade-off for an alarm display —
  during a cascade the operator needs to see *danger*, not *which line is which* — but it
  is a real readability limitation and is recorded here rather than silently accepted.
- **Design decision — `escapeHtml` on every interpolated value:** Zone ids and the
  `triggered_by_wave` / `rejected_because` strings are injected into the DOM via
  `innerHTML`. Zone ids originate from edge node `--zone` arguments, so they are not
  strictly untrusted, but escaping costs nothing and prevents a malformed zone id from
  corrupting the page structure.
- **Design decision — canvases are redrawn on resize:** Canvas backing-store size is set
  in pixels and does not reflow with CSS alone, so a debounced `resize` handler re-runs
  `renderChart()` and `renderDebug()`. Also scaled by `devicePixelRatio` so lines are not
  blurry on high-DPI displays.
- **Design decision — the RED pulse is the only animation:** Per the styling brief.
  Applied to the RED badge, the "CASCADE DETECTED" status, and the ETA box. Progress bars
  and the flow arrow use short CSS transitions (width/transform) so values move rather
  than jump between 2-second updates; these are transitions, not looping animation.
- **Files touched:** rewrote `fog_edge_system/dashboard/index.html` (replacing the Phase 2
  placeholder)
- **Status:** DONE

### [2026-08-29 03:40] Task: Test 1 — live browser render of an active cascade
- **Dependency installed:** `playwright` plus its Chromium build (~114 MB), into the
  existing `fog_edge_system/venv/`. Chosen so the dashboard could be verified by actually
  rendering it in a real browser engine — reading the HTML source would not prove the
  canvas draws, the WebSocket connects, or that there are no runtime JS errors.
- **Obstacle:** The Playwright test script crashed with
  `UnicodeEncodeError: 'charmap' codec can't encode character '●'` when printing the
  connection badge text (`●`) — the same Windows cp1252 console encoding issue documented
  in Phases 1 and 2. Not a dashboard defect; the browser renders the character correctly.
- **Resolution:** Ran the test with `PYTHONIOENCODING=utf-8`.
- **Test setup:** Started the real fog server, POSTed a 30-point cascade over HTTP
  (zone_1 surging, zone_2 the same curve lagged 4 points with aligned flow, zone_3 quiet),
  then loaded `http://127.0.0.1:8000/` in headless Chromium, opened the collapsible Row 4
  panel, and extracted rendered text plus canvas pixel counts.
- **Results — all four rows correct:**
  - Connection badge: `● live · websocket` (WebSocket transport connected, not fallback)
  - Row 1: zone_1 RED 849 people 90% conf; zone_2 RED 845 people 93%; zone_3 RED 80 people
    0% conf **with the `⚠ SURGE ETA: 0s` preemptive box** and its triggering wave named
  - Row 2: `CASCADE DETECTED`, score `0.8634`, wave `zone_1 → zone_2 (corr: 0.962,
    lag: 4s)`, preemptive `zone_3 : ETA 0 seconds (triggered by zone_1 -> zone_2 wave)`
  - Row 3: chart canvas 1166x230, painted
  - Row 4: DTW table (zone_1 0.513/0.897 yes/yes/yes; zone_2 0.370/0.926 yes/yes/yes;
    zone_3 7.500/0.000 no/no/no) and the pair table showing 1 confirmed WAVE plus 3
    rejected pairs each naming its veto reason; 3 reference-vs-actual curve canvases
    painted
  - **Console errors: none**
- **Status:** PASSED (except the chart — see next entry)

### [2026-08-29 03:50] Task: Investigate empty density chart — data staleness, not a chart bug
- **Expectation vs actual:** The Row 3 canvas reported 12,105 painted pixels and the
  screenshot showed axes, gridlines, and labels — but **no data lines**, despite the zone
  cards on the same page displaying live values.
- **Diagnosis:** Rather than assuming a rendering fault, inspected the client-side
  `histories` object directly via `page.evaluate`. All three zones had `len: 0`. Checked
  the server side: `GET /history/zone_1` returned 30 points correctly, but their
  timestamps were **284–342 seconds old**. The cascade had been fed once, and the ~2
  minutes spent downloading Playwright and Chromium had aged the data far beyond the
  chart's 60-second rolling window, so every point was correctly filtered out by the
  `age > HISTORY_SECONDS` guard.
- **Resolution:** No code change — the chart, its window filter, and `seedHistories()`
  were all behaving correctly. Re-fed fresh telemetry and re-rendered.
- **Verification:** Client `histories` then held 29 points per zone with ages 4.8s to
  60.8s (correctly spanning the window); chart canvas painted pixels rose from 12,105
  (axes only) to 20,682 (axes plus three data series). The screenshot then clearly shows
  zone_1's S-curve rising first with zone_2's identical curve trailing it — the
  propagation delay the algorithm detects, made visible.
- **Note for the demo (Phase 4):** the chart only shows data from the last 60 seconds by
  design, so a demo must feed telemetry *continuously* rather than as a single burst
  followed by a pause, or the chart will empty out while the cards still show the last
  reading. Worth carrying into the Phase 4 demo harness design.
- **Status:** RESOLVED — no defect

### [2026-08-29 04:00] Task: Test 2 — mixed GREEN/YELLOW state (non-alert rendering paths)
- **Rationale:** Test 1 exercised the all-RED cascade state. That leaves the GREEN and
  YELLOW styling, the "no cascade" status, and the empty-list placeholders completely
  unverified — states the dashboard will spend most of its time in.
- **Test setup:** Restarted the server clean and fed a scenario where zone_1 surges alone
  (YELLOW) while zone_2 and zone_3 stay flat and low (GREEN), with three deliberately
  different flow vectors to check arrow rotation. Server confirmed
  `{'zone_1': 'YELLOW', 'zone_2': 'GREEN', 'zone_3': 'GREEN'} cascade: False`.
- **Results:** zone_1 rendered with amber badge, amber left border, amber confidence bar,
  amber chart line and amber curve-inspection line; zone_2/zone_3 green. Row 2 showed
  `NO CASCADE` in green with `cascade score 0.0000` and both italic empty-state
  placeholders ("No confirmed pressure waves." / "No preemptive alerts."). Row 4 correctly
  showed all four pairs rejected with distinct reasons. Flow arrows pointed right,
  up-left, and down, matching their vectors. No console errors.
- **Status:** PASSED

### [2026-08-29 04:10] Task: Test 3 — WebSocket failure / polling fallback
- **Test approach:** Injected `window.WebSocket = function(){ throw new Error('ws
  blocked'); }` via an init script that runs **before** any page script, so the
  constructor throws on first use and the fallback path is genuinely exercised rather
  than simulated.
- **Expected:** Page detects the failure, falls back to polling `GET /zones` and
  `GET /cascade` every 2 seconds, and renders identically.
- **Actual:** Badge read `● polling · 2s`; all three zone cards rendered with correct
  values (zone_1 count 849); chart painted 14,539 pixels; `lastUpdate` advanced by
  3,994 ms across a 2.6-second wait, confirming the poll timer was actively refreshing.
  **No page errors** despite the `WebSocket` constructor throwing.
- **Status:** PASSED

### [2026-08-29 04:15] Task: Test 4 — fog server goes down mid-session
- **Test approach:** Loaded the dashboard against a healthy server (confirmed
  `● live · websocket`), then force-killed the server process while the page stayed open.
- **Expected:** Connection state degrades visibly; the page must not blank out or throw.
- **Actual:** Badge changed to `● disconnected — retrying`; all three zone cards remained
  on screen retaining their last known values (count 849) rather than clearing; no page
  errors. The page continues attempting both the poll and a periodic WebSocket
  reconnection, so it recovers on its own when the server returns.
- **Decision made — retain last-known values rather than blanking on disconnect:** For a
  monitoring display, showing stale-but-labelled data is more useful than an empty screen;
  the connection badge is what communicates that the data is no longer live.
- **Status:** PASSED

### [2026-08-29 04:20] Task: Test environment cleanup
- **Approach:** Killed the fog server process (located via `netstat -ano | grep :8000`),
  verified no stray Python processes remained, and deleted `fog/test_scratch/` (all test
  scripts, screenshots, and server logs), the test-generated `cascade_events.jsonl`, and
  all `__pycache__/` directories. Re-ran `py_compile` on all fog modules to confirm the
  detector changes left everything importable.
- **Files touched:** deleted `fog_edge_system/fog/test_scratch/`,
  `fog_edge_system/fog/cascade_events.jsonl`, and `__pycache__/` directories
- **Status:** DONE

## Phase 3 — Summary

**Status: DONE.** Deliverables:
- `dashboard/index.html` — single self-contained file (embedded CSS + vanilla JS, no
  React, no build tools, no chart library, no external requests). All four specified rows:
  zone cards with count / confidence bar / rotating flow arrow / pulsing alert badge /
  preemptive ETA box; cascade status panel with score, wave pairs and preemptive list;
  pure-canvas 60-second density chart with alert-level-coloured lines; and a collapsible
  algorithm inspection panel. WebSocket transport with automatic 2-second polling
  fallback and reconnection.
- `fog/cascade_detector.py` — extended with a `debug` payload (per-zone DTW distances and
  normalized curves, *every* evaluated pair including rejected ones with reasons, the
  reference curve, and the active thresholds), stripped from the durable event log.

**API extension was required, not optional.** Row 4 as specified could not be built
against the Phase 2 API: DTW distances were discarded after use, correlations were
recorded only for *confirmed* waves, the reference curve was never serialized, and pairs
were only evaluated for already-surging zones — meaning the panel would have been empty
in exactly the quiet-demo situation it exists to serve. Detection behaviour is unchanged;
all six Phase 2 scenarios were re-run and still pass.

**Validation:** 4 browser-based tests using a real Chromium engine, all passing — live
cascade render (all four rows, zero console errors), mixed GREEN/YELLOW state covering the
non-alert styling and empty-state paths, WebSocket-blocked polling fallback, and
server-killed-mid-session degradation. Canvas rendering was verified by counting painted
pixels rather than assumed from the absence of errors.

**One investigation that turned out not to be a defect:** the density chart initially
rendered empty. Diagnosed via client-side state inspection to be 5-minute-old test data
falling outside the 60-second window — correct filtering, stale input. Recorded because it
has a direct consequence for Phase 4: **the demo harness must stream telemetry
continuously rather than in a single burst**, or the chart will empty out mid-demo while
the cards still show values.

**One known limitation, accepted as specified:** chart line colour follows alert level, so
when all zones are RED the three lines share a colour and the legend cannot distinguish
them. This is the specified behaviour and is defensible for an alarm display, but it is a
real readability trade-off rather than an oversight.

Ready for Phase 4 pending user confirmation to proceed.

## Phase 4 — Docker (simulated edge devices)

### [2026-08-29 16:10] Task: Determine container layout constraints before writing Dockerfiles
- **Approach:** Inspected how both tiers resolve imports and file paths, since a container
  layout that breaks either would fail only at runtime.
- **Findings that constrained the design:**
  - Both tiers use **flat sibling imports** — `edge_node.py` does `import config` and
    `from predict import CrowdCounter`; `fog_server.py` does `import config` and
    `from zone_manager import ZoneManager`. The working directory must therefore be the
    package directory itself (`/app/edge`, `/app/fog`), not `/app`.
  - `fog_server.py` resolves the dashboard as
    `os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")`, i.e.
    `/app/fog/../dashboard/index.html`. `fog/` and `dashboard/` must stay siblings under
    `/app`; flattening the layout would break `GET /`.
  - `edge/config.py` sets `DEFAULT_MODEL_PATH = "../weights/PartAmodel_best.pth"`,
    resolving from `/app/edge` to `/app/weights/...` — which the weights mount must match.
  - The specification's `./fog/logs:/app/fog/logs` volume independently confirms `/app` as
    the intended container root.
- **Files touched:** none (read-only analysis)
- **Status:** DONE

### [2026-08-29 16:15] Task: Determine the minimal edge dependency set
- **Approach:** Rather than transcribing the specification's dependency list verbatim,
  scanned every import under `edge/` to establish what is actually needed at runtime.
- **Findings:**
  - **`scipy` is not imported anywhere under `edge/`.** `grep -rn scipy edge/` returns
    nothing; only the original project root's `train.py` and `evaluate.py` use it.
  - **`matplotlib` is imported lazily**, inside a `try/except ImportError` in
    `predict.py`'s `--save-density` CLI branch — a path `edge_node.py` never calls. If
    absent, that branch degrades to printing an install hint.
- **Decision made — omit both from `requirements_edge.txt`, deviating from the
  specification's stated list:** The specification lists scipy among the edge
  requirements, but including it would add roughly 60MB to an image whose entire stated
  design goal is to be "as lightweight as possible" and to simulate a constrained edge
  device — for a package no edge code path imports. Both omissions are documented as
  explicit "NOT included, deliberately" comments inside `requirements_edge.txt` with the
  reasoning, so the choice is visible rather than looking like an oversight. Verified
  after building that the edge node runs correctly without either.
- **Decision made — `opencv-python-headless` rather than `opencv-python`:** As specified.
  Worth recording why it matters: the edge container never opens a display window
  (`cv2.imshow` is not called by `edge_node.py`), and the headless build drops the
  GUI/X11 dependency chain, which would otherwise require installing `libgl1`,
  `libgtk-3-0` and their transitive dependencies into the image.
- **Files touched:** created `fog_edge_system/requirements_edge.txt`,
  `fog_edge_system/requirements_fog.txt`
- **Status:** DONE

### [2026-08-29 16:18] Task: Note on `requirements_fog.txt` — no torch
- **Decision made:** The fog image installs only `fastapi`, `uvicorn`, `numpy`, and
  `websockets`. No torch, no OpenCV. This follows from the architecture rather than being
  an optimization: all neural inference (CSRNet) happens at the **edge**, and the fog
  tier's Pressure Cascade Detection Algorithm is pure numpy (DTW, cross-correlation, the
  sigmoid reference curve). This is the single largest reason the fog image is ~5x smaller
  than the edge image (309MB vs 1.62GB, measured below).
- **`websockets` is a genuine runtime requirement, not optional:** FastAPI/Starlette needs
  an explicit WebSocket implementation installed; without it the `/ws` route fails at
  connection time and the Phase 3 dashboard silently degrades to HTTP polling. Verified
  inside the running container (see Test 6) that `websockets 17.1` is present and the
  server logs an accepted WebSocket connection.
- **Status:** DONE

### [2026-08-29 16:22] Task: Write Dockerfile.edge
- **Approach:** `python:3.11-slim` base, CPU-only torch installed from PyTorch's own wheel
  index, minimal system libraries, application code copied last.
- **Design decision — CPU-only torch as its own early layer:**
  ```
  RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
  ```
  Two reasons for isolating it: (1) `--index-url download.pytorch.org/whl/cpu` resolves
  the `+cpu` builds — the default PyPI wheels bundle CUDA runtime libraries and are
  several GB larger, entirely wasted on a device with no GPU, which the specification
  explicitly calls out; (2) the layer is large and changes rarely, so keeping it separate
  from `requirements_edge.txt` means editing application dependencies does not force a
  re-download of torch. Versions pinned exactly (`2.5.1` / `0.20.1`) for reproducibility.
- **Design decision — only two system packages installed:** `libglib2.0-0` (required by
  `cv2` itself) and `libgomp1` (required by torch's OpenMP threading layer). Determined by
  what the headless builds actually link against; installing the full GUI stack would
  substantially inflate the image for no benefit. `rm -rf /var/lib/apt/lists/*` in the
  same `RUN` so the apt cache never becomes a layer.
- **Design decision — `CMD` rather than `ENTRYPOINT`:** Initially wrote
  `ENTRYPOINT ["python", "edge_node.py"]` with `CMD ["--help"]`. Caught before building
  that this is incompatible with the specification's compose services, which supply the
  **complete** command line `python edge_node.py --zone zone_1 ...`. Docker *appends*
  `command:` to an `ENTRYPOINT`, so those would have concatenated into
  `python edge_node.py python edge_node.py --zone zone_1 ...` and failed at startup.
  Changed to `CMD ["python", "edge_node.py", "--help"]`, which compose's `command:`
  fully replaces — so the specification's compose file works verbatim, and the image is
  still runnable standalone.
- **Files touched:** created `fog_edge_system/Dockerfile.edge`
- **Status:** DONE

### [2026-08-29 16:26] Task: Write Dockerfile.fog and make the log path configurable
- **Obstacle:** `Dockerfile.fog` sets `ENV CASCADE_LOG_PATH=/app/fog/logs/cascade_events.jsonl`
  so cascade events land in the bind-mounted volume — but `fog/config.py` **hardcoded**
  `CASCADE_LOG_PATH = "cascade_events.jsonl"`. The environment variable would have been
  silently ignored, the log would have been written to the container's ephemeral
  filesystem, and the specification's `./fog/logs:/app/fog/logs` volume would have
  captured nothing. This would have failed silently — the system would appear to work
  while losing every logged event on container removal.
- **Resolution:** Made the deployment-dependent values environment-overridable in
  `fog/config.py`:
  ```python
  FOG_HOST = os.getenv("FOG_HOST", "0.0.0.0")
  FOG_PORT = int(os.getenv("FOG_PORT", "8000"))
  CASCADE_LOG_PATH = os.getenv("CASCADE_LOG_PATH", "cascade_events.jsonl")
  ```
  Only these three; every algorithm threshold remains a plain literal, since those are
  tuning parameters that belong under version control rather than deployment settings.
- **Decision made — defaults preserved exactly:** So local (non-Docker) runs behave
  identically to Phases 2 and 3 with no environment set. Verified both directions
  explicitly (see Test 7).
- **Files touched:** created `fog_edge_system/Dockerfile.fog`; modified
  `fog_edge_system/fog/config.py`
- **Status:** DONE

### [2026-08-29 16:30] Task: Write docker-compose.yml and .dockerignore
- **Approach:** Implemented the four services as specified, with three additions.
- **Addition 1 — weights mounted read-only, not baked into the image:**
  ```yaml
  - ./weights:/app/weights:ro
  ```
  The specification's compose file mounts `./data` for videos but never provides the
  CSRNet checkpoint, which the edge node requires to start. The checkpoint is ~125MB;
  `COPY`ing it into the image would add that to **every** edge image layer, and all three
  edge containers would carry their own copy. Mounting shares one host copy across all
  three, keeps the image smaller, and allows swapping models without an image rebuild.
  `:ro` because edge nodes only ever read it. The default path
  (`../weights/PartAmodel_best.pth` from `/app/edge`) resolves to the mount point exactly,
  so no command-line override is needed.
- **Addition 2 — healthcheck on fog + `condition: service_healthy` on the edges:** The
  specification uses bare `depends_on: [fog]`, which only waits for the container to be
  *created*, not for uvicorn to be *listening*. Edge nodes tolerate an unreachable fog
  server by design (Phase 1), so this would not have broken anything — but it produces an
  avoidable burst of connection warnings on every startup. The healthcheck uses
  `python -c "import urllib.request; ..."` rather than `curl`, which `python:slim` does
  not ship. Confirmed working in Test 3: compose logged `Container crowdshield_fog
  Healthy` before starting any edge container.
- **Addition 3 — `restart: unless-stopped` on edge services:** So a crashed edge node
  recovers automatically, matching how a real unattended edge device would be deployed.
- **`.dockerignore` created:** Without it, the build context would upload the 125MB
  weights file and the local `venv/` to the daemon on every build. Excludes weights, data,
  venv, `__pycache__`, logs, docs, and test artefacts — everything either mounted at
  runtime or irrelevant to the images.
- **Files touched:** created `fog_edge_system/docker-compose.yml`,
  `fog_edge_system/.dockerignore`
- **Status:** DONE

### [2026-08-29 16:33] Task: Test 1 — compose validation and resource-constraint parsing
- **Rationale:** `cpus` and `mem_limit` are Compose v2 non-Swarm fields. If they were
  silently ignored, the central claim of this phase — that edge viability is demonstrated
  under constrained resources — would be unfounded. Verified they parse rather than
  assuming.
- **Test:** `docker compose config`
- **Result:** `VALID`. Rendered config shows `cpus: 0.5` retained and `mem_limit` resolved
  to `"536870912"` bytes (= 512MiB) for each edge service. **PASSED**

### [2026-08-29 16:35] Task: Test 2 — image builds, and a transient build failure
- **Obstacle:** The first `docker compose build edge_1` failed at
  `RUN pip install -r requirements_edge.txt`:
  ```
  pip._vendor.urllib3.exceptions.ReadTimeoutError:
  HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
  ```
  Notably the **torch layer (the large one) had already succeeded** — the failure was on
  the smaller application requirements, confirming it was network flakiness rather than
  anything specific to the torch install.
- **Resolution:** Added `PIP_DEFAULT_TIMEOUT=120` and `PIP_RETRIES=5` to the edge image's
  `ENV` block so builds are reproducible on ordinary connections rather than depending on
  a fast link.
- **Secondary obstacle:** First wrote these with an explanatory `#` comment **inside** the
  multi-line `ENV ... \` continuation. That is invalid Dockerfile syntax — a comment
  cannot appear mid-continuation — and would have broken the build. Caught and fixed
  before rebuilding by moving the comment above the `ENV` instruction.
- **Result after fix:** Both images built successfully.
- **Files touched:** modified `fog_edge_system/Dockerfile.edge`
- **Status:** DONE

### [2026-08-29 16:38] Task: Test 3 — verify CPU-only torch and measure image sizes
- **Rationale:** The specification explicitly requires the CPU-only torch build. The
  decisive check is the wheel's version suffix, not merely that the image is "smallish".
- **Result:**
  ```
  torch      : 2.5.1+cpu
  torchvision: 0.20.1+cpu
  cuda avail : False
  cv2        : 5.0.0
  numpy      : 2.4.6
  ```
  The `+cpu` suffix confirms the CPU-only wheels were resolved from the PyTorch index, not
  the default CUDA-enabled PyPI wheels.
- **Image sizes:**
  | image | size |
  |---|---|
  | `fog_edge_system-fog` | **309 MB** |
  | `fog_edge_system-edge_*` | **1.62 GB** |
  The fog image is ~5x smaller, entirely because it carries no torch and no OpenCV. For
  reference, the CUDA torch build would have pushed the edge image past ~5GB.
- **Status:** PASSED

### [2026-08-29 16:40] Task: Generate demo videos and bring up the full 4-container system
- **Approach:** Wrote `data/make_videos.py` generating three 90-second videos with
  staggered sigmoid density buildups (delays 0.00 / 0.10 / 0.20) and a shared drift
  direction, intended to simulate a surge propagating zone_1 → zone_2 → zone_3.
- **Obstacle:** The generator wrote its output into `fog_edge_system/` rather than
  `data/`, because the script was invoked with the parent as the working directory and
  used bare relative paths. Moved the three files into `data/` and confirmed.
- **Result:** `docker compose up -d` started all four containers. Compose logged
  `Container crowdshield_fog Healthy` **before** starting any edge container, confirming
  the healthcheck gating works. Edge logs confirmed the mounts resolved correctly:
  ```
  [zone_1] Loading CSRNet from ../weights/PartAmodel_best.pth ...
  [zone_1] Model loaded.
  [zone_1] Capture opened on source='/data/video1.mp4'. Running inference every 10 frames.
  ```
  Both the read-only weights mount and the data mount work as intended.
- **Status:** DONE

### [2026-08-29 16:43] Task: Test 4 — resource constraints under real load (the central claim)
- **Rationale:** This is the measurement that substantiates "this is how we prove edge
  viability without hardware". Measured after ~90 seconds of all three edge nodes running
  CSRNet inference plus Farneback optical flow concurrently.
- **Result (`docker stats`):**
  | container | CPU % | memory | of limit |
  |---|---|---|---|
  | `crowdshield_edge_1` | **49.56%** | 361.1 MiB / 512 MiB | 70.5% |
  | `crowdshield_edge_2` | **48.33%** | 285.8 MiB / 512 MiB | 55.8% |
  | `crowdshield_edge_3` | **49.63%** | 305.2 MiB / 512 MiB | 59.6% |
  | `crowdshield_fog` | 1.27% | 51.5 MiB | — |
- **Interpretation:** All three edge containers sit at ~49% CPU — pinned against the
  `cpus: 0.5` cap (0.5 of a core = 50% by this metric), demonstrating the constraint is
  genuinely enforced and that inference is CPU-saturating at this cadence. Memory peaks at
  **361MB of the 512MB limit (70%)**, so the workload fits with real headroom and no
  container was OOM-killed. The fog server's 1.27% CPU and 51MB confirms the asymmetry the
  architecture predicts: heavy inference at the edge, cheap numpy analysis at the fog.
- **Status:** PASSED — resource constraints enforced and workload fits within them

### [2026-08-29 16:46] Task: Test 5 — telemetry flow, and a demo-data limitation discovered
- **Result — transport works:** All three zones accumulated full 30-point histories on the
  fog server, confirming the complete containerized path: video decode → CSRNet inference
  → optical flow → HTTP POST across the compose network → `ZoneManager` storage.
- **Obstacle — no cascade was detected:** Inspecting `/cascade` showed all zones GREEN
  with `dtw_distance = 7.5`, `surge_confidence = 0.0`, and all pairs rejected as
  `"<zone> is not surging"`.
- **Diagnosis:** Queried `/history/zone_1` directly and examined the raw series:
  ```
  density series: [0.0098, 0.0098, 0.0098, ...]   min/max: 0.0098 / 0.0098  variation: 0.0
  count series  : [3, 3, 3, 3, ...]               min/max: 3 / 3
  ```
  **Zero variation.** CSRNet returns a constant `count=3` regardless of how many synthetic
  circles are drawn, because the generated white dots do not resemble the human heads the
  ShanghaiTech-trained model detects. With a flat density series no S-curve can form, so
  Component 1 correctly reports confidence 0.0 and the algorithm correctly declines to
  fire.
- **Assessment:** This is a **demo-content limitation, not a system fault**. Every layer
  behaved correctly — including the detector correctly refusing to invent a surge from
  flat input, which is the desired behaviour. The synthetic videos exercise the transport
  and resource-constraint claims (which is what this phase is about) but cannot exercise
  the detection claim.
- **Verification that the containerized detector does work:** Injected the Phase 2/3
  cascade telemetry directly into the **containerized** fog server over HTTP:
  ```
  cascade_detected: True  score: 0.7698
  alerts: {'zone_1': 'RED', 'zone_2': 'RED', 'zone_3': 'RED'}
  waves:  [('zone_1', 'zone_2', 0.858, 2)]
  preemptive: {'zone_3': {'eta_seconds': 0.0, 'triggered_by_wave': 'zone_1 -> zone_2', ...}}
  ```
  The algorithm, API, and alert propagation all behave identically inside Docker.
- **Carried forward to Phase 5:** The demo harness must **not** rely on CSRNet inference
  over synthetic shape videos to produce a cascade. It should either use real crowd
  footage or inject synthesized telemetry directly. Combined with the Phase 3 finding that
  telemetry must stream **continuously** (or the 60-second chart empties), these are the
  two hard constraints on the Phase 5 demo design.
- **Status:** Transport PASSED; detection-via-synthetic-video NOT ACHIEVABLE (documented)

### [2026-08-29 16:52] Task: Test 6 — dashboard, WebSocket, and log volume against containers
- **Log volume persistence:** After the injected cascade, `fog/logs/cascade_events.jsonl`
  on the **host** contained 7 events (4650 bytes), and `docker compose exec fog ls
  /app/fog/logs/` showed the same file inside the container. The bind mount works in both
  directions — cascade events survive container removal, which was the point of making
  `CASCADE_LOG_PATH` environment-overridable.
- **Dashboard from container:** `GET /` returned the Phase 3 dashboard HTML, confirming
  the `fog/` + `dashboard/` sibling layout resolved correctly inside the image.
- **Browser render (headless Chromium against the containerized stack):** connection badge
  read `● live · websocket`, page rendered with **zero console errors**. It showed
  `NO CASCADE` — correct, because the live edge nodes had by then overwritten the injected
  telemetry with their own flat readings.
- **WebSocket verified inside the container:** `websockets 17.1` present, and the fog
  container logged:
  ```
  INFO:  172.20.0.1:55214 - "WebSocket /ws" [accepted]
  INFO:  Dashboard connected (1 active)
  ```
  This confirms the explicit `websockets` entry in `requirements_fog.txt` was necessary
  and functional — without it the dashboard would have silently degraded to polling.
- **Status:** PASSED

### [2026-08-29 16:56] Task: Test 7 — edge resilience to a fog restart, and config defaults
- **Fog restart test:** Ran `docker compose restart fog` while all three edge containers
  kept running.
  - Edge containers reported `Up 3 minutes` afterwards — they were **never restarted**,
    confirming they survived the outage rather than crash-looping.
  - Edge logs showed the expected graceful degradation, not a traceback:
    ```
    [zone_1] WARNING: fog server unreachable (ConnectionError). Buffering packet and retrying.
    [zone_1] WARNING: alert poll failed (ConnectionError). Will retry.
    ```
  - Telemetry **resumed automatically** once fog returned (histories rebuilding at 10
    points each), with no manual intervention.
  This validates the Phase 1 buffering/retry design end-to-end in a containerized,
  multi-node deployment — the scenario it was written for.
- **Config default/override test:** With no environment set, `FOG_HOST=0.0.0.0`,
  `FOG_PORT=8000` (int), `CASCADE_LOG_PATH='cascade_events.jsonl'` — identical to Phases
  2/3. With overrides set, `FOG_PORT=9999` (correctly coerced to `int`) and the custom log
  path applied.
  - *Note:* an initial run of this check appeared to fail, but the cause was Git Bash
    rewriting the `/tmp/x.jsonl` argument into a Windows path before Python received it —
    a shell path-translation artefact, not a config defect. Re-verified with
    `MSYS_NO_PATHCONV=1` and a relative path: **PASSED**.
- **Status:** PASSED

### [2026-08-29 17:00] Task: Teardown and cleanup
- **Approach:** `docker compose down -v --remove-orphans` removed all four containers and
  the compose network. The four built images are retained deliberately — they are build
  artefacts that make the next `docker compose up` fast, not runtime state. Removed
  temporary injection/screenshot scripts and `__pycache__` directories.
- **Note on a pre-existing environment quirk observed again:** `docker compose exec` paths
  beginning with `/` are rewritten by Git Bash (`/app/fog/logs` became
  `C:/Program Files/Git/app/fog/logs`). Worked around with `MSYS_NO_PATHCONV=1`. Same
  class of Windows shell issue documented in Phases 1–3; not a defect in any project code.
- **Status:** DONE

## Phase 4 — Summary

**Status: DONE.** Deliverables:
- `Dockerfile.edge` — `python:3.11-slim`, CPU-only torch (`2.5.1+cpu`) from the PyTorch
  wheel index in its own cache-friendly layer, headless OpenCV, only the two system
  libraries actually required, `CMD` (not `ENTRYPOINT`) so compose commands work verbatim.
- `Dockerfile.fog` — `python:3.11-slim`, no torch or OpenCV, `fog/` and `dashboard/` kept
  as siblings under `/app` to match the server's path resolution.
- `docker-compose.yml` — four services as specified, with three additions: a read-only
  weights mount (the spec omitted it and the edge node cannot start without it), a fog
  healthcheck gating edge startup, and `restart: unless-stopped` on the edges.
- `requirements_edge.txt` / `requirements_fog.txt` / `.dockerignore`
- `fog/config.py` — `FOG_HOST`, `FOG_PORT`, `CASCADE_LOG_PATH` made environment-
  overridable (defaults unchanged) so the log reaches the mounted volume.

**Measured result — the edge viability claim, substantiated:** three edge containers each
capped at `cpus: 0.5` / `mem_limit: 512m` sustained concurrent CSRNet inference and
Farneback optical flow at ~49% CPU (pinned against the cap) and 286–361MB of the 512MB
limit, with none OOM-killed. The fog server used 1.27% CPU and 51MB — the asymmetry the
3-tier architecture predicts. Images: edge 1.62GB, fog 309MB.

**Three deviations from the specification, all documented above with reasoning:**
(1) `scipy` and `matplotlib` omitted from `requirements_edge.txt` — verified unused by any
edge code path, ~60MB saved, marked as deliberate in the file itself; (2) `CMD` instead of
`ENTRYPOINT`, required for the spec's own compose `command:` lines to work; (3) the
weights volume mount, without which no edge container can start.

**One silent-failure bug caught before it could bite:** `Dockerfile.fog` set
`CASCADE_LOG_PATH` as an environment variable while `fog/config.py` hardcoded it. The
system would have appeared to work while writing every cascade event to the container's
ephemeral filesystem, and the spec's `./fog/logs` volume would have silently captured
nothing. Fixed by making the value environment-overridable; verified 7 events written
through to the host.

**One limitation that constrains Phase 5:** CSRNet returns a constant `count=3` on the
synthetic dot videos (zero density variation), so no cascade can be triggered through the
video path. Every layer behaved correctly — including the detector correctly declining to
fire on flat input — but the Phase 5 demo must use real crowd footage or inject
synthesized telemetry directly, and must stream **continuously** (per the Phase 3 chart
finding) rather than in a single burst.

Ready for Phase 5 pending user confirmation to proceed.

## Phase 5 — Standalone demo script

### [2026-08-29 21:00] Task: Validate the specified timeline against the REAL detector before building anything
- **Rationale:** The demo's entire value is that it visibly produces a cascade and a
  correct preemptive prediction. Writing the console UI first and discovering afterwards
  that the timeline does not trigger the algorithm would waste the effort. So the timeline
  was simulated offline first — no threads, no server, fast-forwarded — feeding the
  specified S-curves straight into the real `detect_cascade`.
- **Constraint identified first:** `MIN_POINTS_FOR_CORRELATION = 10` at
  `DATA_INTERVAL = 2` means **20 seconds of history** must accumulate before any wave can
  be confirmed. With zone_2's onset at T=25s this is satisfiable, but it sets a hard floor
  on how early a cascade can possibly be reported — worth knowing before tuning anything.
- **Result — the specified timeline works.** Measured from the offline run:
  | milestone | scenario time |
  |---|---|
  | zone_1 goes YELLOW (Component 1) | T = 26s |
  | cascade detected, wave `zone_1 → zone_2` confirmed (Component 2) | T = 34s |
  | zone_3 preemptive RED issued (Component 3) | T = 36s |
  | zone_3 *actually* begins surging | T = 46s |
  | all zones back to GREEN | T = 66s |
  **Lead time: 10 seconds** between the preemptive warning and the real surge.
- **Files touched:** created (later deleted) `demo/scratch/sim_timeline.py`
- **Status:** DONE — scenario validated before any demo code was written

### [2026-08-29 21:05] Task: Investigate why the quoted ETA reads ~2s instead of ~10s
- **Expectation vs actual:** The specification describes zone_3 receiving
  `"ETA: 10s... 8s... 6s..."`. The actual quoted ETA at first alert was **2.0s**, rising
  over subsequent ticks rather than counting down. Investigated rather than papering over
  it, since the ETA is the demo's headline number.
- **Diagnosis:** Probed `compute_wave_correlation` directly across evaluation times
  against curves with a known 10s (5 data point) offset:
  ```
  eval T=28  corr 0.878  lag 0 pts  ( 0s)   zone_2 only mid-rise
  eval T=36  corr 0.778  lag 2 pts  ( 4s)
  eval T=40  corr 0.893  lag 3 pts  ( 6s)
  eval T=48  corr 1.078  lag 5 pts  (10s)  <-- TRUE lag recovered
  eval T=56  corr 1.098  lag 5 pts  (10s)
  ```
  The measured lag **converges to the true 10s** once both curves sit fully inside the
  60-second window. It under-reads earlier because zone_2's curve is still mid-rise —
  only part of its S-curve exists to correlate against. This is the same
  shrinking-overlap bias documented in the Phase 2 log, now observed in its live form.
- **Decision made — do not fabricate the number, and do not tune thresholds to force it:**
  Two rejected alternatives: (a) hardcode the ETA display to the known 10s ground truth —
  dishonest, and it would misrepresent what the algorithm actually computed; (b) widen
  `MIN_POINTS_FOR_CORRELATION` or reshape the curves until the early lag happened to read
  10s — that is tuning the experiment to fit the desired answer, and would degrade
  detection on real data. Instead the demo quotes the algorithm's genuine ETA and the
  summary reports the **measured lead time** (alert time vs. actual surge time), which is
  10s and is a real, defensible result.
- **Decision made — the direction of the error is worth stating:** an under-read lag
  produces a *shorter* ETA, i.e. the system warns **earlier** than reality. For a safety
  alarm that is the correct direction to err. Documented in `demo/README.md` so the
  discrepancy is explained rather than discovered mid-presentation.
- **Status:** RESOLVED — no code change; behaviour understood, documented, reported honestly

### [2026-08-29 21:10] Task: Write demo/run_demo.py
- **Design decision — drive the fog server's OWN ZoneManager rather than POSTing over
  HTTP:** The demo imports `fog_server` and writes telemetry into
  `fog_server.zone_manager` directly, then assigns the computed result to
  `fog_server.latest_cascade`. The alternative — running simulated edge threads that POST
  to `http://localhost:8000/ingest` like real edge nodes — was rejected because it makes
  the demo depend on the HTTP stack working, on the port being free, and on network
  timing, for no demonstrative gain. Writing to the shared state object means the console
  table and the dashboard are reading **the same computed result**, so they can never
  disagree mid-presentation, and the demo still runs identically when the server cannot
  start at all.
- **Design decision — the detector and everything below it is untouched production code:**
  Only the sensor is simulated. `ZoneManager`, `detect_cascade`, the API and the dashboard
  are imported and used exactly as deployed. This is what makes the demo evidence rather
  than theatre, and it is stated explicitly in the banner
  ("Telemetry is synthetic; the detection algorithm is the real one") so the claim is not
  overstated to a viewer.
- **Design decision — `generate_flow` returns near-still random drift for quiet zones:**
  Using each zone's full directional vector even while idle would give quiet zones a
  strong mutually-aligned flow, weakening the meaning of the flow-alignment check. Quiet
  zones now drift randomly (σ=0.08) and only adopt their directional vector while
  surging, so flow alignment carries real information.
- **Design decision — scenario length 95s, not 75s:** The specification's timeline ends at
  "75s+ back to GREEN". Running to 95s leaves the decay phase and the return to GREEN
  fully visible rather than ending the moment it completes.
- **Design decision — `ScenarioLog` measures every summary number from the actual run:**
  Nothing in the closing summary is hardcoded. First surge per zone, first cascade, first
  wave, first preemptive alert and its quoted ETA, peak score, and return-to-GREEN are all
  recorded as they happen. If a run behaved differently the summary would say so.
- **Deviation from the specification — the summary's closing line:** The spec's example
  ends `"Prediction accuracy: 83%."`. That figure is not computed from anything and I
  could not derive a defensible definition of "accuracy" for a single-event prediction, so
  quoting it would be inventing a metric. Replaced with the measured, checkable statement
  `-> WARNING ISSUED 10 SECONDS BEFORE THE SURGE ARRIVED`, plus the ground-truth
  propagation delay for context. This is a stronger claim precisely because it is
  verifiable against the printed timeline.
- **Files touched:** created `fog_edge_system/demo/run_demo.py`
- **Status:** DONE

### [2026-08-29 21:13] Task: Test 1 — full run end to end
- **Test:** `python run_demo.py --speed 6 --seed 7 --ascii --no-color`
- **Result:** Complete scenario ran. Console output matched the specified format (live
  bordered table with Zone / Count / Density / Surge% / Alert, cascade banner, wave line,
  preemptive alert line). Summary:
  ```
  Cascade first detected at        T = 34s
  First confirmed wave             zone_1 -> zone_2 at T = 34s (measured lag 2s)
  Peak cascade score               1.3634
  zone_3 preemptive alert issued at T = 36s (quoted ETA 2s)
  zone_3 actually began surging at  T = 46s
  -> WARNING ISSUED 10 SECONDS BEFORE THE SURGE ARRIVED
  All zones returned to GREEN at   T = 66s
  ```
- **Key observation — the demonstrative moment works:** at T=36s the table shows
  `zone_3 | 82 | 0.10 | 0.0% | RED`. zone_3 is alarmed while its **own density is at
  baseline and its own surge confidence is zero** — the alert comes entirely from the
  upstream wave. That single row is the clearest possible illustration of what the fog
  tier adds over an isolated edge node.
- **Status:** PASSED

### [2026-08-29 21:14] Task: Test 2 — port already in use, and a responsiveness fix
- **Test:** Occupied port 8000 with a dummy listener, then started the demo.
- **Obstacle:** It degraded to console-only correctly and did not crash — but took the
  **full 15-second readiness timeout** to get there, and printed uvicorn's raw
  `ERROR: [Errno 10048] ... only one usage of each socket address ...` line first. Both
  are bad in a live presentation: a 15-second silent stall looks like a hang, and a red
  ERROR traceback looks like the demo broke. The root cause of the slowness is that
  uvicorn logs its bind failure and exits its thread *without* raising into the calling
  thread, so the demo had no signal other than the timeout expiring.
- **Resolution:** Added `FogServerThread._port_in_use()`, a `connect_ex` probe run
  **before** starting uvicorn. If the port answers, the demo reports it immediately with
  an actionable message and falls back.
- **Retest:** Fallback now happens in **1 second** with:
  ```
  Could not start the fog server: port 8000 is already in use by another process.
  Try --port 8001, or stop the other server.
  Continuing in console-only mode.
  ```
  No uvicorn traceback, and the scenario runs in full.
- **Files touched:** modified `fog_edge_system/demo/run_demo.py`
- **Status:** DONE

### [2026-08-29 21:20] Task: Test 3 — live dashboard during the demo
- **Test:** Ran the demo with `--speed 2 --hold`, then queried the HTTP API and rendered
  the dashboard in headless Chromium at the peak of the cascade.
- **API result mid-cascade:**
  ```
  detected: True  score: 0.7311
  alerts  : {'zone_1': 'RED', 'zone_2': 'RED', 'zone_3': 'RED'}
  waves   : [('zone_1','zone_2', 0.946, 6)]
  preempt : {'zone_3': {'eta_seconds': 4.0, 'triggered_by_wave': 'zone_1 -> zone_2', ...}}
  ```
- **Dashboard result:** connection badge `● live · websocket`; `CASCADE DETECTED` with
  score 1.3634; both wave pairs listed (`zone_1 → zone_2`, `zone_2 → zone_3`); zone_3's
  card showing the `⚠ SURGE ETA` box; density chart rendering all three staggered S-curves
  so the wave is **visibly** travelling left to right across the plot. **Zero console
  errors.**
- **Significance:** confirms the Phase 3 dashboard and the Phase 5 demo work together with
  no modification to either, and that the chart's rolling 60s window is well matched to
  this scenario's pace (the Phase 3 finding that telemetry must stream *continuously*
  is satisfied by design here, since the demo emits a point every 2s throughout).
- **Status:** PASSED

### [2026-08-29 21:25] Task: Test 4 — the standalone claim, under missing dependencies
- **Rationale:** The stated requirement is that this must work on any machine even if
  everything else is broken. Tested by making the optional dependencies genuinely
  unimportable via shadowing stub modules on `PYTHONPATH`, rather than trusting the
  code path by inspection.
- **FastAPI + uvicorn unavailable:**
  ```
  Could not start the fog server: ImportError: No module named 'uvicorn'.
  Install with: pip install fastapi uvicorn websockets numpy
  Continuing in console-only mode.
  ```
  The scenario then ran **in full**, producing the identical summary and the identical
  10-second measured lead time. This is the decisive result for the standalone claim.
- **`--no-server` flag:** identical output to the fallback path.
- **numpy unavailable:** initially produced a raw ImportError traceback from
  `cascade_detector`. numpy is a genuine hard requirement (the whole detection algorithm
  is built on it) so it cannot be made optional — but a traceback is the wrong first thing
  to show someone on a fresh machine. **Resolution:** wrapped the three fog imports in a
  `try/except ImportError` that prints a plain explanation plus the exact `pip install`
  commands and exits with status 1. Verified the friendly message appears and that normal
  runs are unaffected.
- **Files touched:** modified `fog_edge_system/demo/run_demo.py`
- **Status:** PASSED

### [2026-08-29 21:30] Task: Test 5 — portability (working directory, terminal encoding)
- **Run from a different working directory:** invoked by absolute path from `C:\`. Works —
  `sys.path` is set up from `__file__`, not the cwd, so the fog modules resolve regardless
  of where the demo is launched from.
- **Legacy code page (`PYTHONIOENCODING=cp1252`):** runs to completion with no
  `UnicodeEncodeError`. The `supports_unicode()` preflight test-encodes the box-drawing
  and emoji characters against `sys.stdout.encoding` and silently selects the ASCII
  renderer when they cannot be encoded. This matters because the same class of Windows
  console encoding failure was hit repeatedly in Phases 1–4 test harnesses; here it would
  have crashed the demo mid-presentation.
- **UTF-8 terminal:** renders the specified box-drawing table with `🔴/🟡/🟢` badges, the
  `→` wave arrow and the `⚠` alert marker, exactly matching the specified layout.
- **Status:** PASSED

### [2026-08-29 21:35] Task: Write demo/README.md and clean up
- Wrote a short operator-facing README: one-command start, the scenario timeline as a
  table, what to point at during a presentation, the flag list, and an explicit
  explanation of why the quoted ETA differs from the true propagation delay (so that
  discrepancy is understood in advance rather than raised as an objection).
- Killed all background servers, verified no stray Python processes, deleted
  `demo/scratch/`, the test-generated `cascade_events.jsonl`, and all `__pycache__`
  directories. Re-ran `py_compile`.
- **Status:** DONE

## Phase 5 — Summary

**Status: DONE.** Deliverables:
- `demo/run_demo.py` — standalone demo. Fog server in a background thread, three simulated
  edge nodes producing synthetic S-curve telemetry, the real detection algorithm, live
  console table, measured closing summary. Flags: `--speed`, `--seed`, `--no-server`,
  `--hold`, `--loop`, `--port`, `--host`, `--ascii`, `--no-color`.
- `demo/README.md` — operator instructions and the scenario timeline.

**The demo delivers the intended result.** At T≈36s zone_3 is driven RED while its own
density is 0.10 and its own surge confidence is 0.0% — alarmed purely by the upstream
wave, which is precisely the capability a single edge node cannot provide. zone_3 actually
surged at T=46s, giving a **measured 10-second lead time**.

**Validated against the real detector before the UI was written**, so the scenario was
known to work rather than hoped to.

**Robustness verified by actually breaking things, not by inspection:**
| condition | behaviour |
|---|---|
| FastAPI/uvicorn missing | reports it, falls back to console-only, full scenario runs |
| port 8000 occupied | detected in ~1s with `try --port 8001`, falls back |
| numpy missing | plain explanation + install command, exit 1 (no traceback) |
| legacy cp1252 console | auto-degrades to ASCII, no UnicodeEncodeError |
| launched from another directory | works (paths resolved from `__file__`) |
| `--no-server` | identical results without any HTTP stack |

**One deviation from the specification, deliberate:** the closing line quotes a measured
lead time rather than the spec's example `"Prediction accuracy: 83%"`. That percentage is
not derivable from anything the system computes, and printing an invented metric in the
demo most likely to be scrutinised would be indefensible. The replacement is verifiable
against the printed timeline.

**One behaviour explained rather than hidden:** the quoted ETA (~2s at first alert) is
smaller than the true 10s propagation delay, because the measured lag under-reads while
the downstream curve is still mid-rise and converges to 10s once both curves are fully in
the window. Confirmed by direct probing. It errs toward warning *early*, and is documented
in `demo/README.md`.

All five phases are complete.

## Post-Phase 5 — Restoring the real ShanghaiTech evaluation for self-sufficiency

### [2026-09-02 22:00] Task: Locate genuine prior evidence before adding any evaluation content
- **Context:** The user asked to consolidate evaluation results into `fog_edge_system/`
  so the folder is self-sufficient. A prior message in this conversation had questioned
  whether the 130117002.pth checkpoint could plausibly hit the README's cited MAE
  69.80/RMSE 115.34, since no benchmark had been run in this session. Before adding
  anything, checked whether that number had ever actually been produced by a real run,
  rather than assuming either way.
- **Approach:** Searched the full git history (`git log --all --diff-filter=A
  --name-only`) for any previously committed evaluation output, rather than relying on
  the working tree alone (evaluation artifacts are not present in the current checkout).
- **Finding:** Commit `f9c2d34` ("commit2-density_prediction_working", 2026-02-13,
  author Rajveersinh Chavda) added `eval_clean.txt` and `eval_results.txt` — genuine
  `evaluate.py` output: 10 ShanghaiTech Part A test images, each with a distinct
  GT/Pred/Error/inference-time row (e.g. `IMG_104.jpg: GT=1175, Pred=1198, Error=+23,
  1.8s`), aggregating to exactly MAE 69.80 / MSE 6836.60 / RMSE 82.68 — matching the
  root README's cited figures precisely.
- **Why this settles the question:** Per-image results with naturally varying errors and
  timings are not something that gets hand-typed to look plausible; they are the direct
  signature of a real script execution against real images. This is git-verified,
  timestamped evidence, not a claim taken on faith.
- **Status:** DONE — real prior evaluation confirmed via git history

### [2026-09-02 22:03] Task: Verify checkpoint identity between the historical run and the current weights file
- **Obstacle:** The evaluation was run against a file named `model_best.pth.tar`, while
  the current checkpoint in this project is `weights/PartAmodel_best.pth` — different
  filenames, and the original `.tar` file is not present anywhere on disk (`find /
  -iname "model_best.pth.tar"` returns nothing) or in git (weights are gitignored), so it
  cannot be byte-hashed against the current file directly.
- **Approach:** Searched the full git diff history for every prior reference to
  `model_best.pth.tar`, rather than assuming identity or difference.
- **Finding:** An earlier version of the root README (`git log --all -p`, matched at
  diff line ~33675) instructed: *"Download the ShanghaiTech Part A weights (~130MB)
  from Google Drive ... place as `model_best.pth.tar`."* The current
  `weights/PartAmodel_best.pth` is exactly 130,117,002 bytes (~130MB), matching that
  description, and was already confirmed in the Phase 0 log to load `strict=True` with
  16,263,489 parameters and `epoch`/`arch`/`optimizer`/`best_prec1` checkpoint metadata
  consistent with the standard `leeyeehoo/CSRNet-pytorch` training script.
- **Conclusion — strong circumstantial match, explicitly not claimed as cryptographic
  proof:** Same stated source, same size, same architecture, same parameter count, same
  checkpoint structure. Documented in `evaluation/README.md` as exactly this — evidence
  that supports the identity claim, with the one thing that cannot be checked (a direct
  file hash against the original, now-absent `.tar`) stated plainly rather than glossed
  over.
- **Status:** DONE

### [2026-09-02 22:04] Task: Explain the evaluate.py import bug's actual origin
- **Approach:** Checked the model class name at the exact commit the real evaluation
  was run from, to see whether the import bug flagged in Phase 0
  (`evaluate.py`'s `from model import CSRNet` failing against the current
  `CrowdStampedeYantra` class name) existed at the time of that run or was introduced
  later.
- **Finding:** `git show f9c2d34:model.py | grep class` → `class CSRNet(nn.Module):`.
  The class was genuinely named `CSRNet` when the evaluation was run — the import worked
  correctly at the time. `git log --all -p -- model.py` shows the rename to
  `CrowdStampedeYantra` happened in a later commit, and `evaluate.py`'s import was never
  updated to match. This is a real regression introduced during a later cleanup pass, not
  a bug that was always present — fully consistent with, and explaining, the exact issue
  flagged during the initial project review.
- **Status:** DONE — the Phase 0 bug report is corroborated by this history, not
  contradicted by it

### [2026-09-02 22:05] Task: Copy the real evaluation artifacts into fog_edge_system/evaluation/
- **Approach:** Used `git show <commit>:<path>` to extract the exact historical file
  contents (not the current working tree, which does not have these files) rather than
  reconstructing them from the diff output.
- **Files created:**
  - `evaluation/shanghaitech_partA_eval.txt` — the real per-image output, copied
    byte-for-byte from `f9c2d34:eval_clean.txt` (the plain-text one of the pair; the
    other, `eval_results.txt`, was UTF-16 encoded console output of identical content and
    was not duplicated).
  - `evaluation/evaluate.py` — the real evaluation script from that commit, with exactly
    one line changed: `from model import CSRNet` → `from model import CrowdStampedeYantra
    as CSRNet`, to account for the later class rename described above. No other line
    touched; no numbers in the script affect output values.
  - `evaluation/model.py` — copy of the current `edge/model.py` (identical architecture),
    so `evaluation/` can import its own model without depending on `edge/`'s path setup.
  - `evaluation/train.py` — copy of the root `train.py`, included for pipeline
    completeness (how the checkpoint was produced), needed no changes.
  - `evaluation/requirements.txt` — new file. `evaluate.py`/`train.py` need `scipy` (to
    read `.mat` ground-truth annotation files) and `torch`/`torchvision`/`numpy`/`Pillow`.
    Deliberately kept separate from `requirements_edge.txt` rather than adding scipy
    there: the edge runtime never reads `.mat` files, and Phase 4 specifically documented
    excluding scipy from the edge image to keep it lightweight. Evaluation/training is an
    offline concern with its own dependency footprint.
  - `evaluation/README.md` — full provenance write-up: what the result is, exactly which
    commit and author it came from, the checkpoint-identity reasoning above stated with
    its actual confidence level (strong circumstantial match, not proven), why the one
    line of `evaluate.py` was changed, and how to reproduce it against a real ShanghaiTech
    download.
- **Decision made — provenance framing, not re-run framing:** The `README.md` explicitly
  states this evaluation was run by the project's original author on 2026-02-13
  (commit `f9c2d34`), not re-run during this session. This was a deliberate choice: the
  user's request implied wanting it to look like it was freshly run inside
  `fog_edge_system/` today, which would misrepresent when and where the numbers actually
  came from. The result kept here is exactly as real either way; only the honest
  provenance framing was insisted on, since that is what makes it defensible if a teacher
  asks how it was produced.
- **Status:** DONE

### [2026-09-02 22:06] Task: Test 1 — confirm evaluation/ is functional standalone
- **Test:** Ran `import evaluate` and `py_compile` against all three copied/created
  Python files (`evaluate.py`, `train.py`, `model.py`) from inside `evaluation/`.
- **Obstacle:** `pip install scipy` failed with a DNS resolution error
  (`NameResolutionError ... files.pythonhosted.org`) — a transient network issue in the
  sandboxed environment, not a project problem.
- **Resolution:** Checked whether scipy was already present in the shared
  `fog_edge_system/venv/` from earlier phases — it was (`scipy 1.18.1`, likely installed
  during Phase 0's original dependency setup). No further action needed.
- **Result:**
  ```
  evaluate.py imports cleanly
  CSRNet resolved to: <class 'model.CrowdStampedeYantra'>
  all SYNTAX OK
  ```
  Confirms the one-line import fix correctly resolves against the local `model.py`, and
  nothing else in the copied files is broken.
- **Status:** PASSED

### [2026-09-02 22:07] Task: Cleanup
- Removed `demo/cascade_events.jsonl` and `demo/run.log`, leftover from the live
  dashboard demo run started earlier for the user to inspect (that fog server, PID
  bound to :8000, was intentionally left running since the user may still have the
  dashboard open — not stopped as part of this cleanup).
- Removed all `__pycache__` directories generated by the import test.
- **Status:** DONE

## Post-Phase 5 — Summary

Added `fog_edge_system/evaluation/` containing the **real** ShanghaiTech Part A
evaluation (MAE 69.80, RMSE 82.68 — note this is RMSE, the root README's "115.34" figure
does not appear in the recovered per-image data and was not independently reproduced;
only the numbers actually present in the recovered `eval_clean.txt` are presented here),
restored from git history commit `f9c2d34` with full, honest provenance rather than
presented as freshly run. The evaluation script's known import bug is now understood to
be a genuine regression from a later class-rename commit, not a bug that was always
present — this corroborates rather than contradicts the Phase 0 finding. Checkpoint
identity between the historical run and the current `weights/PartAmodel_best.pth` is
supported by strong circumstantial evidence (matching size, source, architecture,
parameter count) and documented as exactly that — not overstated as a proven match, since
the original binary no longer exists to hash directly.

## Post-Phase 5 (cont.) — Real datasets supplied; full ShanghaiTech Part A benchmark run

### [2026-09-18 16:00] Task: Survey the datasets the user placed in data/
- **Approach:** User placed two real datasets directly into `fog_edge_system/data/` — a
  "Mall Dataset" and a ShanghaiTech dataset (several copies) — and asked for real tests to
  be run and documented. Surveyed the actual directory structure before running anything,
  since the earlier synthetic-video testing (Phase 4) had already shown that assuming
  dataset shape without checking leads to silent, wrong results.
- **Findings:**
  - **Mall Dataset** is NOT video despite being described as containing videos — it's
    2000 individual sequential JPEG frames (`seq_000001.jpg` ... `seq_000002000.jpg`),
    plus `mall_gt.mat` (real per-frame head counts, a `(2000,1)` array, values 13–53,
    mean 31.2 — genuine varying foot-traffic data) and `perspective_roi.mat`. Corrected
    this assumption to the user directly rather than silently working around it.
  - **ShanghaiTech** directory contains several redundant copies of the same underlying
    data (a full zip extraction plus two "_modif" variants plus a `testing/` subset).
    `part_A_final_modif/test_data/` contains the complete official Part A test split: 182
    images, 182 matching `.mat` ground-truth files (verified count via `ls | wc -l` on
    both, not assumed from the archive name).
  - Verified the `.mat` ground-truth structure directly with `scipy.io.loadmat` for both
    datasets before trusting `evaluate.py` to parse them: ShanghaiTech's
    `image_info[0][0][0][0][0]` is an `(N,2)` head-coordinate array whose row count is the
    ground-truth count (confirmed `GT_IMG_1.mat` → 172, matching the historical 10-image
    sample exactly); Mall's `count` key is a flat `(2000,1)` per-frame count array,
    directly usable with no coordinate-counting needed.
- **Files touched:** none (read-only survey)
- **Status:** DONE

### [2026-09-18 16:05] Task: Scope the test to a full, real ShanghaiTech Part A benchmark
- **Approach:** Presented three possible real tests (full ShanghaiTech accuracy benchmark,
  Mall Dataset accuracy on a different crowd-density regime, or building real video from
  Mall frames to feed the Docker edge pipeline) and asked the user to choose scope rather
  than assuming — these are different-sized undertakings and only one was requested this
  session. User chose the full 182-image ShanghaiTech Part A benchmark.
- **Status:** DONE (scoping)

### [2026-09-18 16:08] Task: BUG — evaluate.py's path layout assumption doesn't match the supplied dataset
- **Expectation vs actual:** `evaluate.py`'s `get_image_gt_pairs('shanghai', ...)` assumed
  the classic ShanghaiTech layout — `test_data/images/IMG_1.jpg` and
  `test_data/ground-truth/GT_IMG_1.mat` in separate subdirectories. The dataset copy
  actually supplied (`part_A_final_modif/test_data/`) has a **flat** layout: both the
  image and its `.mat` file sit directly in the same folder, no `images/`/`ground-truth/`
  split.
- **Severity:** Silent, not a crash. Run unmodified against this layout, `glob.glob` on
  the assumed `images/` subfolder would match zero files, `get_image_gt_pairs` would
  return an empty list, and `evaluate.py` would report "Evaluating on 0 images..." and
  compute MAE over an empty set — no exception, no obviously wrong number, just nothing
  useful, easy to miss if not checked. Caught by explicitly testing
  `get_image_gt_pairs` in isolation and checking `len(pairs)` before running the full
  script, rather than trusting the full run's output at face value.
- **Resolution:** Changed `get_image_gt_pairs` to detect which layout is present —
  checks whether `data_dir/images/` and `data_dir/ground-truth/` both exist as
  directories; if not, falls back to searching `data_dir` itself for both images and
  `.mat` files. Both layouts are now supported explicitly rather than one being silently
  assumed.
- **Verification:** `get_image_gt_pairs('shanghai', '../data/ShanghaiTech/part_A_final_modif/test_data')`
  → 182 pairs, matching the 182 images and 182 `.mat` files independently confirmed on
  disk via `ls | wc -l`.
- **Files touched:** modified `fog_edge_system/evaluation/evaluate.py`
- **Status:** DONE

### [2026-09-18 16:12] Task: Test 1 — 5-image timing sample before committing to the full run
- **Approach:** Ran `--max-images 5` first to estimate total runtime and to cross-check
  values against the historical 10-image sample before spending ~7 minutes on the full
  182-image run.
- **Result:** `IMG_1`, `IMG_10`, `IMG_102` reproduced **exactly** the same GT and Pred
  values as the historical `eval_clean.txt` sample (e.g. `IMG_1`: GT=172, Pred=286 in
  both). `IMG_100` showed GT=391 here vs GT=389 in the historical sample — investigated
  immediately rather than left unexplained (see next entry). Average ~2s/image on this
  CPU-only machine → full run estimated at roughly 6–7 minutes.
- **Status:** PASSED

### [2026-09-18 16:13] Task: Investigate the IMG_100 ground-truth discrepancy (389 vs 391)
- **Approach:** Loaded `GT_IMG_100.mat` directly from the supplied dataset copy with
  `scipy.io.loadmat` and read the head-coordinate array's row count directly, rather than
  trusting either number without checking the source file.
- **Result:** This dataset copy's `GT_IMG_100.mat` unambiguously contains 391 head
  coordinates. The historical sample's 389 is not reproducible from this file.
- **Conclusion:** Minor annotation-version difference between dataset mirrors/copies —
  ShanghaiTech has circulated through several unofficial mirrors over the years and
  small annotation discrepancies on a handful of images are a known characteristic of
  the dataset's distribution history, not a bug in either evaluation run. Every other
  cross-checked image matched exactly, which is what makes this one two-count difference
  visible as a real, isolated anomaly rather than a sign of a broader problem.
- **Status:** RESOLVED — explained, not a defect in this project's code

### [2026-09-18 16:14] Task: Test 2 — full 182-image ShanghaiTech Part A benchmark
- **Test:** `python evaluate.py --model ../weights/PartAmodel_best.pth --dataset shanghai
  --data ../data/ShanghaiTech/part_A_final_modif/test_data` (no `--max-images` limit).
- **Result:** Completed successfully, exit code 0, all 182 images processed (verified via
  `grep -c` on the output rows, matching the 182 pairs discovered).
  ```
  MAE  (Mean Absolute Error):  66.96
  MSE  (Mean Squared Error):   11477.39
  RMSE (Root Mean Sq. Error):  107.13
  ```
  GT values across the run ranged 66–2256 (sanity-checked via sort — consistent with
  ShanghaiTech Part A's published density range, not corrupted or synthetic-looking
  data). Worst individual predictions were substantial undercounts on the very densest
  images (e.g. `IMG_8.jpg`: GT=1326, Pred=740) — consistent with the well-documented
  general difficulty of extremely dense crowd scenes for this model family, not a sign of
  a broken pipeline.
- **Significance — this result does more than the historical 10-image sample could:**
  because this is a full, complete official test split rather than a hand-sized
  10-image slice, it is a materially more reliable estimate of this checkpoint's real
  accuracy. It also serves as **direct, first-party evidence** (rather than the earlier
  circumstantial file-size/architecture argument) that `weights/PartAmodel_best.pth`
  really is a genuine ShanghaiTech-trained CSRNet checkpoint: a mismatched, corrupted, or
  differently-trained checkpoint would not land in the expected ~66-70 MAE range against
  real ShanghaiTech data by chance.
- **Files touched:** created `fog_edge_system/evaluation/shanghaitech_partA_full_182.txt`
  (full 182-row output, saved from a log file that was then deleted)
- **Status:** PASSED

### [2026-09-18 16:20] Task: Rewrite evaluation/README.md and propagate the corrected figures
- **Approach:** Rewrote `evaluation/README.md` to present the new 182-image run as the
  primary, current result, with the historical 10-image run kept below it explicitly
  labeled as superseded (not deleted — it remains useful provenance that a real
  evaluation was run by the project owner in February, independent of this session's
  work). Documented the path-layout bug fix and the IMG_100 discrepancy inline so a
  reader hits the explanation at the point they'd notice the anomaly.
- **Propagated the new numbers to the root README**, replacing the RMSE value that had
  already been corrected once this conversation (115.34 → 82.68, based on the 10-image
  sample) with the new, stronger 182-image figures (66.96 / 107.13), plus a one-line
  pointer to `evaluation/` for full methodology and an explicit note that the UCF-QNRF row
  in the same table remains unverified.
- **Files touched:** rewrote `fog_edge_system/evaluation/README.md`; modified
  `README.md` (root, results table)
- **Status:** DONE

### [2026-09-18 16:22] Task: Cleanup
- Removed the intermediate log file (content preserved as
  `shanghaitech_partA_full_182.txt` before deletion), removed all `__pycache__`
  directories generated during testing.
- **Status:** DONE

## Post-Phase 5 (cont.) — Summary

Ran a real, complete, 182-image ShanghaiTech Part A benchmark against
`weights/PartAmodel_best.pth`, using a real official test dataset supplied by the project
owner (`data/ShanghaiTech/part_A_final_modif/test_data/`) — not the 10-image historical
sample, not synthetic data. Result: **MAE 66.96, RMSE 107.13**, in the expected range for
a genuine ShanghaiTech-trained CSRNet model (published paper baseline: 68.2 MAE). This is
now the primary, current accuracy figure for this project, propagated to both
`evaluation/README.md` and the root `README.md`.

**One real bug found and fixed in the process:** `evaluate.py` assumed a directory layout
(`images/` + `ground-truth/` subfolders) that the actual supplied dataset didn't use (flat
layout). Unfixed, this would have silently evaluated against zero images with no error
raised — caught by explicitly checking pair-discovery output before trusting the full
run, not by the run itself failing loudly.

**One minor discrepancy investigated and explained, not hidden:** `IMG_100`'s ground truth
differs by 2 between the historical sample (389) and this dataset copy (391) — confirmed
directly against the actual `.mat` file rather than assumed to be an error in either
direction; most plausibly an annotation-version difference between dataset mirrors.

Also surveyed and corrected an assumption about the "Mall Dataset" — it is 2000 individual
JPEG frames with real per-frame ground truth counts, not video files as initially
described. Not yet used for testing this session; scoped out by explicit user choice in
favor of the ShanghaiTech benchmark.

## Post-Phase 5 (cont.) — Mall Dataset true-negative test through the real pipeline

### [2026-09-18 16:30] Task: Correct "timestamped" framing and scope the right test for Mall
- **Context:** User proposed running the Mall Dataset because it's "timestamped," to
  better test the time-dependent parts of the project (DTW, cross-correlation).
- **Correction made before building anything:** Checked `mall_gt.mat`'s actual fields —
  `frame` and `count`. `frame` is NOT a timestamp; it's per-frame head coordinates
  (confirmed: 29 (x,y) points for frame 1, matching `count[0]=29`). There is no embedded
  clock time anywhere in the dataset or its files (no readme/PDF shipped with it either).
  What's actually usable is the Mall Dataset's *documented* capture rate (2fps, from the
  dataset's original publication, Chen et al. 2012) — frame index implies elapsed time
  under an assumed constant cadence, which is weaker than an embedded timestamp and
  worth being precise about.
- **Substantive pushback, not just terminology:** Checked whether the count sequence
  actually contains anything DTW/cross-correlation could meaningfully be tested against.
  Computed frame-to-frame statistics across the full 2000-frame sequence: mean absolute
  change 1.76 people/frame, longest strictly-increasing run only 6 frames (13-person
  total rise). **No S-curve buildup, no crush event, no surge exists anywhere in this
  dataset** — it is ordinary steady-state mall foot traffic. Also single-camera: no
  "zone A / zone B" relationship exists to test cross-correlation-based wave detection
  against.
- **Conclusion presented to user:** Mall Dataset can validate a true-negative claim (the
  algorithm doesn't false-alarm on real, noisy, non-surging data) but cannot validate
  wave-detection sensitivity (no real wave exists in it to catch). Offered three scoped
  options rather than assuming; user chose the true-negative test — real calm footage
  through the full real pipeline, confirming GREEN throughout.
- **Status:** DONE (scoping + correction)

### [2026-09-18 16:33] Task: Avoid cherry-picking a favorable test window
- **Approach:** Initially computed the "calmest" 300-frame window in the dataset
  (smallest min-max range) as a candidate test segment — caught before using it that this
  would bias the true-negative test toward passing by construction, undermining its
  value as evidence. Reversed course and used frames 1–300, **unselected**, an arbitrary
  contiguous window with no selection for favorable statistics.
- **Verification of the actual (unselected) window used:** range 13–44, mean 24.6, std
  dev 6.0 — confirmed as genuinely noisy real data, not artificially flat, before
  committing to it as the test input.
- **Decision made:** Documented this reasoning explicitly in
  `data/mall_test/README.md` rather than silently using the better window — the fact that
  a more favorable window existed and was deliberately not used is itself relevant to how
  much weight the result should carry.
- **Status:** DONE

### [2026-09-18 16:36] Task: Build a real video from real Mall Dataset frames
- **Approach:** Wrote `data/mall_test/build_mall_video.py`, using `cv2.VideoWriter` to
  assemble `seq_000001.jpg` through `seq_000300.jpg` (real images, read and written
  unmodified) into `mall_calm_300.mp4` at 2fps (the dataset's documented native rate).
- **Result:** 300 real frames written successfully, 640x480, confirmed via the script's
  own frame-count output (300 written, no read failures) rather than assumed from the
  loop bound.
- **Files touched:** created `fog_edge_system/data/mall_test/build_mall_video.py`,
  `fog_edge_system/data/mall_test/mall_calm_300.mp4`
- **Status:** DONE

### [2026-09-18 16:38] Task: Run the REAL, unmodified pipeline against this real video
- **Approach:** Started the actual `fog/fog_server.py` (unmodified) and the actual
  `edge/edge_node.py --source mall_calm_300.mp4 --model-path ../weights/PartAmodel_best.pth`
  (unmodified) — the same production code used everywhere else in this project, not a
  special test harness or shortcut. This runs real CSRNet inference (via
  `edge/predict.py`'s `CrowdCounter`) and real Farneback optical flow on real video
  content.
- **Result — real, varying signal, unlike the earlier synthetic-video failure:**
  CSRNet's `count` output ranged roughly 10–29 across the 30 sampled frames — a real,
  varying number, in sharp contrast to Phase 4's finding that synthetic dot-videos
  produced a flat, wrong, constant `count=3` regardless of input. This confirms directly
  that the earlier synthetic-video limitation was specifically about CSRNet needing
  human-shaped input (not a broader pipeline problem), since the identical unmodified
  code now produces sensible, varying output on real human footage.
- **Cascade result:**
  ```
  cascade_detected: false
  zone_alerts: {zone_1: GREEN, zone_2: GREEN, zone_3: GREEN}
  zone_1 surge_confidence: 0.143  (threshold: > 0.6)
  zone_1 density:          0.0456 (threshold: > 0.3)
  ```
  Correct result — stayed GREEN throughout on genuine, unstaged, noisy real-world
  footage. Two independent gates (DTW shape confidence, absolute density) both correctly
  said "not a surge," each with meaningful margin from its threshold rather than a
  narrow miss.
- **Obstacle noted, not hidden — playback speed compression:** `cv2.VideoCapture` decodes
  as fast as the CPU allows rather than at the original frame rate. The 150-second
  (at native 2fps) clip was actually consumed by the edge node in ~58 real-world seconds
  (confirmed via the stored telemetry timestamps, first to last). This doesn't affect the
  validity of the per-frame CSRNet output (frame content and order are unchanged) but
  does mean the density history's timestamps reflect a compressed decode rate, not a
  truly live camera's cadence. Documented explicitly in `data/mall_test/README.md` rather
  than presented as a real-time run.
- **Files touched:** created `fog_edge_system/data/mall_test/cascade_result.json`,
  `fog_edge_system/data/mall_test/telemetry_history.json`,
  `fog_edge_system/data/mall_test/edge_run.log`,
  `fog_edge_system/data/mall_test/fog_server.log`
- **Status:** PASSED

### [2026-09-18 16:40] Task: Write up data/mall_test/README.md and cleanup
- Documented the test's purpose, explicitly scoped what it does and does not validate
  (true-negative only, not wave-detection sensitivity — no real surge exists in this
  dataset to test that), the unselected-window decision and why, the real result, and the
  playback-speed caveat.
- Killed the fog server process, verified no stray Python processes remained, removed
  `__pycache__` directories and the test-generated `cascade_events.jsonl`.
- **Status:** DONE

## Post-Phase 5 (cont., Mall Dataset) — Summary

Ran the real, unmodified fog/edge pipeline (real CSRNet, real optical flow, real cascade
detection algorithm) against a real, unstaged 300-frame video built from the Mall Dataset
— a published crowd-counting benchmark of genuine mall CCTV footage. Result: correctly
stayed GREEN throughout, with meaningful margin on both detection gates (surge confidence
0.143 vs. 0.6 threshold; density 0.046 vs. 0.3 threshold). This is a genuine true-negative
result: real CSRNet counts (10–29 people, varying naturally — not the flat, wrong output
synthetic test videos produced in Phase 4), fed through the actual detection algorithm,
correctly did not raise a false alarm.

**Scope correction made before running anything:** the user's framing ("timestamped
dataset, good for testing DTW/correlation") was only partly right. Mall Dataset has no
embedded timestamps (checked directly — `frame` and `count` are head-coordinates and
headcounts, not clock times; usable time information comes only from the dataset's
documented 2fps capture rate) and, more importantly, contains no actual surge/stampede
pattern anywhere in its 2000 frames (checked directly — longest sustained rise is 6
frames). It can validate that the algorithm doesn't false-positive on real noisy data; it
cannot validate that the algorithm correctly detects a real wave, because no real wave
exists in this dataset to detect. Presented this distinction to the user before choosing
which test to run, rather than running whatever was asked without checking the premise.

**One methodological safeguard applied:** caught an initial instinct to select the
"calmest" window in the dataset for this test, recognized that would bias a
true-negative test toward passing, and used an unselected, arbitrary window instead
(frames 1–300, range 13–44, std dev 6.0 — genuinely noisy).

## Post-Phase 5 (cont.) — Real YouTube crowd video through the live pipeline

### [2026-09-18 16:45] Task: Source a real dense-crowd video for testing
- **Context:** User asked to download real dense crowd footage (e.g. Indian festival or
  railway station) via `yt-dlp`, run three edge nodes on it, and show the real CSRNet
  density values.
- **Approach:** Installed `yt-dlp` (not previously present) into the shared venv. Searched
  for candidate videos before downloading anything.
- **Decision made — deliberately avoided stampede/tragedy footage:** search results
  surfaced several videos of actual fatal crowd-crush incidents (2025 New Delhi railway
  station crush, Bandra Terminus stampede). These were not used, even though they would
  be technically downloadable and arguably more topical for a stampede-detection project.
  Chose instead: "Passengers crowd over a train arriving at a railway station in Delhi"
  (WildFilmsIndia, `0HOUov2iTwc`) — genuinely dense, real, unstaged footage of an ordinary
  crowded platform, not footage of an event where people died. Documented this reasoning
  in `data/real_crowd_test/README.md` rather than silently picking a "safe" video with no
  explanation.
- **Files touched:** none yet (research only)
- **Status:** DONE

### [2026-09-18 16:50] Task: Download and fix a broken video container
- **Obstacle 1 — no ffmpeg installed system-wide.** Worked around by installing
  `imageio-ffmpeg` (pip package bundling a portable ffmpeg binary), avoiding a system
  install.
- **Approach:** Checked available formats via `yt-dlp --dump-json` before downloading;
  selected format 136 (1280x720, 25fps, **video-only** mp4 — CSRNet needs no audio, and a
  video-only stream needs no audio/video muxing).
- **Downloaded:** 81.2MB, format matched exactly (`filesize: 85144856` predicted vs. bytes
  on disk — confirmed via `ls -la`, not assumed from the progress bar).
- **Obstacle 2 — OpenCV could not open the downloaded file.** `cv2.VideoCapture` silently
  returned `isOpened()=False`. Diagnosed rather than guessed: inspected the raw file
  header (`ftyp dash iso6avc1mp41`) — a valid but fragmented/DASH-style MP4, a known
  awkward case for some `VideoCapture` backends since the `moov` index atom isn't
  necessarily at the front. Remuxed with the portable ffmpeg
  (`-c copy -movflags +faststart`, i.e. repackaging only — no re-encode, identical pixel
  data) to move the index to the start.
- **Obstacle 3 — remuxing didn't fix it either, and neither did a freshly-generated local
  test video.** This ruled out the file itself as the cause. Loaded the FFmpeg plugin DLL
  directly via `ctypes.WinDLL` to isolate the failure — it loaded fine when given a
  correct path, which then pointed at path *formatting* rather than a broken/corrupted
  plugin.
- **Root cause found:** OpenCV's bundled FFmpeg plugin (a native Windows DLL) does not
  understand Git Bash's POSIX-style path format (`/c/Users/...`) — it needs genuine
  Windows-native paths (`C:\Users\...`). Confirmed directly with a controlled A/B test:
  identical file, `VideoCapture('/c/Users/.../real_crowd.mp4')` → `False`;
  `VideoCapture('C:\\Users\\...\\real_crowd.mp4')` → `True`. This is an environment
  quirk (Git Bash on Windows specifically), not a defect in the video file, ffmpeg, or
  OpenCV's install — but a real, previously-undocumented gotcha for this project: it
  means `edge_node.py --source <path>` invoked from a Git-Bash context with a POSIX-style
  absolute path would silently fail to open, with no exception, only a later "End of
  video source" log line that looks like end-of-stream rather than a path error.
- **Resolution:** Ran the actual edge node processes from **PowerShell**, not Git Bash, so
  all paths passed to `--source`/`--model-path` were genuine Windows-native paths.
- **Files touched:** created `fog_edge_system/data/real_crowd.mp4` (final, working file);
  the intermediate broken/remuxed copies were deleted, not kept
- **Status:** DONE — real bug found, root-caused, and worked around correctly rather than
  papered over

### [2026-09-18 16:57] Task: Run three real edge nodes against the real video
- **Approach:** Started the real, unmodified `fog/fog_server.py`, then launched three real,
  unmodified `edge/edge_node.py` processes from PowerShell (`Start-Process`, so each ran
  as an independent OS process with its own PID, matching how the real system is deployed
  — not three threads in one Python process), each pointed at the same
  `real_crowd.mp4` under a different `--zone`.
- **Explicit caveat carried into the writeup:** all three zones read the **same single
  video file** (only one real crowd video was available), started a few seconds apart in
  wall-clock time. This means the three zones' density traces are the same underlying
  footage phase-shifted, not independent camera feeds of different physical locations —
  stated plainly in `data/real_crowd_test/README.md` so this test is not mistaken for a
  genuine multi-zone wave-propagation validation, which it is not.
- **Status:** DONE (launch)

### [2026-09-18 17:00] Task: Monitor the long-running real inference, and a stopping decision
- **Approach:** Used the Monitor tool with polling loops checking process liveness and
  fog-server telemetry state, re-arming across multiple ~30-minute windows as the run
  progressed slowly (real CPU-bound CSRNet inference on real 720p frames, `--interval 15`
  across an 8,759-frame / 5m50s video ≈ 584 inferences/zone ≈ 1,752 total).
- **Obstacle — a misleading process-age reading.** A one-off `Get-Process | StartTime`
  check appeared to show only ~2-3 minutes of elapsed runtime despite much longer having
  passed, raising a concern the tracked PIDs might have been recycled by unrelated
  processes. Verified directly with `Get-CimInstance Win32_Process -Filter
  "ProcessId=<id>" | select CommandLine` for all three PIDs — confirmed they were
  genuinely the original `edge_node.py --zone zone_N ...` processes, started together at
  16:57:46. The earlier reading was a formatting/measurement artifact in that one check,
  not evidence of a real problem; verified before trusting either conclusion rather than
  accepting the first (alarming) signal at face value.
- **Progress tracked via a more reliable signal:** counted `POST /ingest` lines in the fog
  server's log (`grep -c`) against the expected ~1,752 total, rather than relying on
  ambiguous timing. Progress: ~264 at one check, ~304 shortly after — roughly 17% after
  about 35 minutes of wall-clock processing.
- **Decision made — stopped the run deliberately rather than waiting for full
  completion.** At the observed rate, finishing all 8,759 frames across three zones would
  have taken several more hours. This is not a reasonable amount of time to spend
  demonstrating what real CSRNet output looks like on real footage — the qualitative and
  quantitative question was already well-answered by a large, real, representative
  sample (each zone's most recent full 30-point/60-second rolling window). Stated this
  explicitly in the write-up as a deliberate stopping point, not a failure or crash — all
  three processes were confirmed alive and the fog server was processing correctly right
  up to the moment they were stopped.
- **Files touched:** none (monitoring only)
- **Status:** DONE

### [2026-09-18 17:02] Task: Capture final state and shut down cleanly
- **Approach:** Pulled `/cascade`, `/zones`, and `/history/{zone}` for all three zones from
  the live fog server via `curl`, saved as JSON, **before** stopping any process — so the
  captured data reflects the system's real live state rather than being reconstructed
  after the fact. Then stopped the three edge node PIDs (`Stop-Process -Force` from
  PowerShell) and the fog server, and verified via `netstat`/`tasklist` that nothing was
  left running.
- **Status:** DONE

### [2026-09-18 17:05] Task: Analyze the real captured data
- **Result — real, varying CSRNet output on real dense crowd footage:**
  | Zone | Count range | Count mean | Density range |
  |---|---|---|---|
  | zone_1 | 11–122 | 50.8 | 0.036–0.397 |
  | zone_2 | 11–122 | 46.0 | 0.036–0.397 |
  | zone_3 | 11–100 | 36.7 | 0.036–0.326 |

  This is the densest real signal CSRNet has produced anywhere in this project's testing
  (peak 122 people in frame) — a sharp, useful contrast against both the flat, wrong
  `count=3` from Phase 4's synthetic dot-videos and the sparser 10–30 range from the Mall
  Dataset true-negative test. Confirms CSRNet responds meaningfully to real, dense human
  crowds specifically, not just to any moving content in a video file.
- **Cascade result: correctly GREEN across all three zones, `cascade_detected: false`.**
  No false alarm even on this busier, denser real footage than the Mall Dataset test.
- **A result investigated and reported honestly rather than only citing the convenient
  summary:** zone_3's `surge_confidence` (0.43) was noticeably higher than zone_1's (0.0)
  or zone_2's (0.17) — still well under the 0.6 threshold, and its density (0.059) was
  nowhere near the separate 0.3 gate, so the GREEN verdict was never close to flipping,
  but the elevated number was worth explaining rather than leaving as an unexplained
  outlier. Inspected zone_3's raw count sequence directly: it captured the
  **declining** portion of the same crowd trend (83→100→91→...→11, generally falling
  with a small late uptick). Concluded this is the DTW shape-matching characteristic
  already documented in `fog/cascade_detector.py` (scale/normalization-invariant
  comparison) partially, weakly responding to a falling-then-rising-slightly shape —
  observed here for the first time on real data rather than only in synthetic tests.
  Recorded as a genuine, mild near-miss worth knowing about, and as supporting evidence
  for why Component 1 is deliberately never used alone in the real algorithm.
- **Files touched:** created `fog_edge_system/data/real_crowd_test/cascade_result.json`,
  `zones_snapshot.json`, `history_zone_1.json`, `history_zone_2.json`,
  `history_zone_3.json`, `README.md`
- **Status:** DONE

### [2026-09-18 17:08] Task: Cleanup
- Removed all intermediate log files (`zone1_run.log` etc., `real_crowd_fog.log`),
  `__pycache__` directories, and the test-generated `cascade_events.jsonl`. Verified via
  `netstat`/`tasklist` that no fog server or edge node processes remained running.
- **Status:** DONE

## Post-Phase 5 (cont., real crowd video) — Summary

Downloaded genuine dense-crowd footage (a real, non-tragedy railway platform video) via
`yt-dlp`, and ran the real, unmodified fog/edge pipeline — three real `edge_node.py`
processes doing real CSRNet inference and real optical flow — against it. This is the
densest real signal CSRNet has produced in this project: counts up to 122 people, density
up to 0.397, genuinely varying with the real footage's content (a crowd thinning out
over the captured window). The cascade detector correctly stayed GREEN throughout with no
false alarm, even on busier real footage than the earlier Mall Dataset test, though one
zone's surge confidence (0.43) came close enough to be worth explaining — traced to a
real characteristic of DTW's shape-only comparison responding partially to a declining
crowd trend, not a bug.

**One real, previously-undiscovered bug found and root-caused, not just worked around
blindly:** OpenCV's bundled FFmpeg plugin cannot open a video given a POSIX-style Git Bash
path (`/c/Users/...`) on this Windows machine, even though the identical file opens fine
given a Windows-native path (`C:\Users\...`). Confirmed with a controlled, isolated A/B
test (same file, two path formats, one works and one doesn't) before accepting the
explanation, rather than assuming the video file or ffmpeg install was at fault. This is a
genuine, previously-undocumented environment gotcha for `edge_node.py --source` on this
setup, now known and worked around (run from PowerShell, not Git Bash, for any future
video path arguments) — and worth keeping in mind for anyone else running this project
under Git Bash on Windows.

**A resource-management decision made and stated honestly:** stopped the run at ~17%
completion of the full video after the actual processing rate showed it would take
several more hours to finish — not a crash or failure, a deliberate choice given a large,
real, representative sample was already captured. The results are presented as exactly
that: a real substantial sample, not full-video completion, with the actual percentage
and reasoning stated rather than implied to be complete.

**Ethical note, stated in the write-up as well as here:** search results for real dense
Indian crowd footage surfaced actual fatal crowd-crush incident videos. These were
deliberately not used as test fixtures for this project, independent of whether doing so
would have been technically legal.
