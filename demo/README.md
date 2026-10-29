# Demo — run this

Standalone demonstration of the whole system. **No Docker, no GPU, no CSRNet weights,
no video files, no camera.**

```bash
cd fog_edge_system/demo
python run_demo.py
```

Then open **http://localhost:8000** to watch the live dashboard alongside the console.

Only requirement is `numpy`. For the dashboard as well:

```bash
pip install numpy fastapi uvicorn websockets
```

If FastAPI/uvicorn are missing, or port 8000 is busy, the demo says so and continues in
console-only mode — the algorithm demonstration still runs in full.

## What it does

The fog server runs in a background thread; three "edge nodes" are threads producing
synthetic telemetry instead of doing computer vision. **Everything downstream of the
telemetry packet is the real production code** — ZoneManager, the full Pressure Cascade
Detection Algorithm, the HTTP/WebSocket API, the dashboard. Only the sensor is simulated.

~95 second scenario:

| time | what happens |
|---|---|
| 0–15s | all zones low and quiet — **GREEN** |
| 15s | zone_1 begins an S-curve buildup → **YELLOW** (Component 1: DTW shape match) |
| 25s | zone_2 begins the *same* curve, 10s later, crowd flowing the same way |
| ~34s | **CASCADE DETECTED** — wave `zone_1 → zone_2` confirmed (Component 2) |
| ~36s | **zone_3 goes RED while its own density is still 0.10** (Component 3: ETA) |
| ~46s | zone_3 actually surges — the prediction was right |
| 55–75s | densities decay; zones return YELLOW → GREEN |

The headline moment is around T=36s: zone_3 is alarmed **before its own camera sees
anything**, because the fog layer can see the wave coming from upstream. A single edge
node could not do this.

The closing summary reports measured results from that run, e.g.:

```
  zone_3 preemptive alert issued at T = 36s (quoted ETA 2s)
  zone_3 actually began surging at  T = 46s
  -> WARNING ISSUED 10 SECONDS BEFORE THE SURGE ARRIVED
```

## Options

```
--speed 2        run faster (2 = twice as fast); good for a short slot
--hold           keep the dashboard up after the scenario ends
--loop           repeat forever (unattended display)
--no-server      console only, no HTTP server
--seed 7         reproducible run (identical output every time)
--port 8001      use a different port
--ascii          plain ASCII (legacy terminals)
--no-color       disable colour
```

For a presentation, `python run_demo.py --hold` is usually right: the dashboard stays up
afterwards so you can open the **Algorithm inspection** panel and show *why* each wave
was confirmed or rejected.

## A note on the quoted ETA

The ETA quoted early in a wave (~2s) is smaller than the true 10s propagation delay.
This is expected: while zone_2 is still mid-rise, only part of its curve exists to
correlate against, so the measured lag under-reads and converges to the true 10s once
both curves are fully inside the 60-second window. The effect is conservative — it warns
*earlier* than reality, which is the safe direction for a safety alarm. The **measured
lead time** in the summary (10s) is the honest headline number, taken from when the alert
fired versus when zone_3 actually surged.
