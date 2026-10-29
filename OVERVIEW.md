# What This Project Is

**CrowdShield** is a system that watches multiple cameras at a crowded venue (a concert,
a stadium entrance, a religious gathering) and tries to warn security staff **before** a
stampede happens — not just when it's already happening.

The core idea: a single camera can only see its own small patch of ground. But a
stampede doesn't start everywhere at once — it starts as a **pressure wave**: a crush
builds up in one spot, and a few seconds later it hits the next spot over, then the
next. If you're only watching one camera, you find out when the wave reaches *you*. If
something is watching *all* the cameras at once and knows how fast crowds move, it can
tell the next zone "this is coming for you in 10 seconds" — before that zone's own
camera has seen anything unusual at all.

That's what this system does.

---

## The problem in one picture

```
  Zone 1 (Gate A)        Zone 2 (Corridor)        Zone 3 (Gate B)
  ┌──────────┐            ┌──────────┐            ┌──────────┐
  │  camera  │            │  camera  │            │  camera  │
  │  crowd   │  ───────►  │  crowd   │  ───────►  │  crowd   │
  │ building │            │not surging│            │  calm    │
  └──────────┘            └──────────┘            └──────────┘
   getting crowded          fine... for now          fine

If Zone 1 is filling up and the crowd is physically moving toward Zone 2,
and Zone 2 shows the SAME pattern 10 seconds later, that's not a coincidence —
that's a wave. Zone 3 should be warned NOW, before it happens to them.
```

---

## The three tiers (this is the "fog computing" part)

The system is split into three layers that each do a different job. This split is the
whole architectural point of the project.

### 1. Edge — the cameras (`edge/`)

One small program runs per camera. Each one:
- Watches a video feed (a webcam, or a video file)
- Every few frames, asks a crowd-counting AI model: *"how many people, how densely
  packed, roughly?"* (this model is called **CSRNet** — see below)
- Also measures which direction the crowd is drifting, using optical flow (comparing
  consecutive frames to see motion)
- Packages this into a small JSON message and sends it to the fog server every 2
  seconds — it does **not** send video, just a handful of numbers

This is deliberately lightweight. It's built and tested to run on constrained hardware
(a Raspberry-Pi-class device, simulated in this project via Docker containers capped at
half a CPU core and 512MB RAM) — one per camera, at the edge of the network, close to
where the video actually is.

### 2. Fog — the brain (`fog/`)

One server, on the local network, that all the edge nodes report to. This is where the
actual intelligence lives. It:
- Keeps a rolling 60-second history of density readings for every zone
- Runs the **Pressure Cascade Detection Algorithm** (below) on every new reading
- Decides an alert level per zone: 🟢 GREEN (fine), 🟡 YELLOW (one zone surging alone),
  🔴 RED (a confirmed wave, or a zone about to be hit)
- Answers questions from edge nodes ("what's my alert level?") and serves a live
  dashboard

### 3. Cloud (optional) — the record

Every time a cascade is detected, it's logged to a simple append-only file
(`fog/logs/cascade_events.jsonl`) — a permanent, timestamped record of what happened
and when, for after-the-fact review.

---

## The novel part: how does it actually decide "this is a real wave"?

This is the thing that separates this project from "just alert when density is high."
A simple threshold check would false-alarm constantly — big crowds are normal at a
concert. The algorithm instead asks three separate questions, and only escalates when
**all** of them agree.

### Question 1 — "Does this zone's density curve *look like* the start of a crush?"

Before a real crush, density doesn't jump instantly — it follows a recognizable shape:
slow rise → speeds up → levels off (an S-curve). The algorithm compares each zone's
recent 30-second density trace against this reference S-curve shape (using a technique
called **Dynamic Time Warping**, which tolerates the buildup happening slightly faster
or slower than the reference). If the shape matches well *and* the density is actually
high (not just a small ripple), that zone is marked **surging**.

### Question 2 — "Is the neighboring zone's surge actually the SAME wave, delayed?"

If Zone 1 is surging, the algorithm checks Zone 1's density curve against every
neighboring zone's curve, sliding one in time against the other (**cross-correlation**)
to find: do these two curves match if I shift one of them by N seconds? If they match
well *and* the shift shows Zone 2 lagging *behind* Zone 1 (not the other way around)
*and* both zones' crowds are physically drifting in the same direction — that's
confirmed as one real wave traveling from Zone 1 to Zone 2. Both go RED.

This step is what rules out false alarms: two zones that are *both* busy but
unconnected (different event, different time, crowd moving the opposite way) do **not**
get confirmed as a wave. The algorithm can tell "these are the same crowd" apart from
"these are just both crowded."

### Question 3 — "When will the NEXT zone downstream get hit?"

Once a wave is confirmed with a measured delay (say, 10 seconds from Zone 1 to Zone 2),
and we know when Zone 2's surge started, simple arithmetic predicts when Zone 3 —
adjacent to Zone 2, not yet surging — will be hit: `time remaining = measured delay −
time already elapsed`. If that's under 30 seconds, Zone 3 goes RED **before its own
camera shows anything unusual at all**. This is the headline capability: a preemptive
warning issued from information a single camera could never have on its own.

---

## The full data flow, start to finish

```
1. Camera captures a frame
        │
2. Edge node runs CSRNet on it → people count + density
   Edge node compares this frame to the last one → crowd movement direction
        │
3. Edge node bundles {count, density, movement} into a small JSON packet
   and POSTs it to the fog server, every 2 seconds
        │
4. Fog server stores it in that zone's 60-second rolling history
        │
5. Fog server re-runs the cascade algorithm (all 3 questions above) on EVERY
   new packet from ANY zone — because a wave might now be visible
        │
6. Fog server updates each zone's alert level (GREEN/YELLOW/RED) and,
   for zones not yet surging, computes a preemptive ETA if a wave is inbound
        │
7. Two things read this result:
     a) Every edge node polls "what's my alert level?" every 2 seconds —
        if RED, it prints a big warning to its own console
     b) The dashboard (a web page) shows it live, over a WebSocket connection —
        zone cards, a live density chart, and a panel showing exactly why
        each wave was confirmed or rejected
```

---

## What's real, and what's simulated for demo purposes

This matters for being honest about what's been verified:

| Piece | Status |
|---|---|
| The cascade detection algorithm | **Real**, unit- and scenario-tested (including deliberately trying to trick it with coincidental spikes, simultaneous surges, and opposing crowd flow — it correctly rejects all of them) |
| The fog server, dashboard, API | **Real**, tested live with real HTTP/WebSocket traffic |
| CSRNet (the crowd-counting model) | **Real** pretrained model, architecture-verified against the published reference; see `evaluation/` for a real accuracy run on ShanghaiTech Part A |
| Edge ↔ fog networking, Docker packaging, resource limits | **Real**, tested with actual containers under real CPU/memory caps |
| The "standalone demo" you've been watching (`demo/run_demo.py`) | Uses **fake, formula-generated** density numbers on a scripted timeline, specifically so the whole system can be demonstrated with **no camera, no GPU, no dataset, and no Docker** — everything *downstream* of that fake number (the algorithm, the server, the dashboard) is the same real code that would run with a real camera |

In short: the "sensor" (camera → count) can be either real (CSRNet on a video/webcam) or
fake (the scripted demo). Everything after that point — the decision-making — is
identical either way.

---

## How to actually see it work

The fastest way, no setup required beyond `pip install numpy fastapi uvicorn websockets`:

```bash
cd fog_edge_system/demo
python run_demo.py
```

Then open **http://localhost:8000** in a browser. Watch Zone 1 build up, Zone 2 repeat
it 10 seconds later (cascade confirmed), and Zone 3 go red **before its own density has
moved at all** — then actually start surging ~10 seconds later, confirming the
prediction was right.

---

## Folder map

```
fog_edge_system/
├── edge/           one program per camera — real CSRNet + motion tracking
├── fog/            the brain — the cascade detection algorithm + server
├── dashboard/      the live web page (single HTML file, no build tools)
├── shared/         data format shared between edge and fog
├── demo/           standalone zero-dependency demo (fake sensor, real everything else)
├── evaluation/     real accuracy check of the CSRNet model on ShanghaiTech
├── weights/        the pretrained CSRNet model file
├── docker-compose.yml, Dockerfile.*   containerized deployment, resource-capped
└── PROGRESS.md     full development log — every decision, bug, and test, in detail
```
