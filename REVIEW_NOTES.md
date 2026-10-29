# Pressure Cascade Detection — Review Notes

## 1. The idea in one paragraph

Edge cameras count people per zone. The fog server sees **all zones at once**, so it can
tell whether a density surge in one zone is a **wave travelling into the next zone**,
not just a high number. If a wave is confirmed, the zone *ahead* of it is warned
**before its own camera shows anything**. A single edge node cannot do this.

```
Edge (CSRNet count + optical flow, per zone) --> Fog (cascade detector) --> alerts / dashboard / log
```

Code: `fog/cascade_detector.py`. All thresholds: `fog/config.py`.

## 2. How the detector works

Runs after every telemetry ingest. Each component gates the next.
Data arrives every 2 s; history is 30 points (60 s) per zone.

### Component 1 — "Is this zone building up dangerously?" (per zone)
- Take the last 15 samples (30 s), min-max normalise to 0–1.
- Compare to a reference **S-curve** (slow rise → acceleration → plateau) using
  **Dynamic Time Warping (DTW)**.
- `confidence = 1 - DTW_distance / 5.0`, clipped to [0, 1].
- Zone is **surging (YELLOW)** if confidence > 0.6 **and** density > 0.3.

**Why DTW:** a real buildup can be faster or slower than the reference. DTW matches
the *shape* while letting the time axis stretch; a point-by-point distance would
reject a slow buildup that has the right shape.

**Why the separate density gate:** normalisation makes the shape test scale-blind (a tiny
S-curve looks like a huge one). So "right shape" and "actually dense" are two
independent conditions.

### Component 2 — "Is zone B repeating zone A with a delay?" (per adjacent pair)
- Z-score both 30-point series, compute **cross-correlation** over lags of ±10 points
  (±20 s). Best correlation gives similarity; its lag gives propagation delay.
- A pair is a confirmed **wave (RED)** only if **all four** hold:
  1. source zone A is surging
  2. correlation > 0.7
  3. lag > 0 (B follows A, not the reverse)
  4. flow vectors of A and B point the same way (dot product > 0)

**Why four conditions:** each blocks a different false alarm — two unrelated spikes
(low correlation), simultaneous spikes with no travel (zero lag), crowds moving in
opposite directions (flow), and a quiet source zone.

### Component 3 — "Who is hit next, and when?" (per downstream zone)
- `ETA = propagation_delay - time_since_upstream_zone_started_surging`, floored at 0.
- If ETA ≤ 30 s, the downstream zone gets a **preemptive RED** alert even though its
  own density hasn't moved. If several waves target one zone, the most urgent ETA is kept.

### Output
Per-zone alert level (GREEN / YELLOW / RED), confirmed wave pairs, preemptive alerts with
ETA, and a cascade score = Σ(correlation × source surge confidence). Every evaluated pair
— including rejected ones and the reason — is exposed in the dashboard's
"Algorithm inspection" panel. Detections are appended to a JSONL log (the "cloud" tier).

## 3. Design choices and why

| Choice | Reason |
|---|---|
| 3-tier edge / fog / cloud | Edge does heavy vision per camera; fog needs all zones to see cross-zone waves; cloud is only logging. |
| Fog is numpy-only (no torch) | DTW and correlation are pure numpy; fog image is ~5× smaller than edge. |
| Gate components in sequence | Cheap per-zone check first; pair analysis only matters for surging sources. |
| DTW for shape, cross-correlation for delay | DTW tolerates speed differences but doesn't give a lag; cross-correlation gives the lag. They answer different questions. |
| Z-score / min-max normalisation | Cameras differ in scale; compare shape, not absolute counts. |
| Flow direction as a veto | Correlation alone can't tell a travelling wave from coincidence. |
| Every threshold in `config.py` | Tunable without touching detection logic. |
| Record rejected pairs | Shows *why* a false alarm was avoided, which is the difference from a plain threshold. |
| Reference curve is a hand-built sigmoid (steepness 10) | Based on S-curve density buildup described in crowd-dynamics literature (Helbing et al. 2007; Still 2000). No labelled stampede data was available to learn one. |

## 4. Results

### Crowd counting (CSRNet) — real data
- Full **ShanghaiTech Part A test set, 182 images**: **MAE 66.96, RMSE 107.13**.
- Published CSRNet baseline MAE is 68.2, so the weights behave like a genuine ShanghaiTech model.
- Weakness: undercounts very dense crowds (e.g. IMG_8: true 1326, predicted 740).
- Files: `evaluation/shanghaitech_partA_full_182.txt`.

### Cascade detector — synthetic scenario (`demo/run_demo.py`)
Three zones in a line; zone_1 surges, zone_2 repeats the curve 10 s later.
- zone_1 → YELLOW at buildup (Component 1).
- Cascade `zone_1 → zone_2` confirmed at ~34 s.
- **zone_3 goes RED at ~36 s while its own density is still 0.10.**
- zone_3 actually surges at ~46 s → **10 s measured lead time**.

### Cascade detector — tests
- 5 test suites passing, including 6 full-algorithm scenarios; 4 are negative cases
  (non-adjacent coincidence, zero-lag simultaneity, opposing flow, quiet source).
  Each negative case is defeated by a different condition.

### Real footage — false-alarm check
- Real CSRNet + optical flow through the real edge → fog pipeline on Mall Dataset frames and
  on real railway-platform video (counts up to 122, density up to 0.40).
  Detector stayed **GREEN, no false alarm**.
- One zone's surge confidence reached 0.43 (threshold 0.6) on a declining crowd, a mild
  DTW near-miss. This is why Component 1 is never used alone.
- Only ~17 % of the railway video was processed (stopped for time); stated as a sample.

## 5. Limitations — say these before you are asked

1. **The cascade has not been tested on a real stampede/crush.** No such labelled data was
   available (and fatal-incident footage was deliberately not used). The positive result
   (10 s lead time) is on **synthetic telemetry**; the real-data runs only show it does
   **not** raise false alarms.
2. **The demo does not run CSRNet.** Only the sensors are simulated; the detector, API and
   dashboard are real code.
3. **Reference curve is hand-designed**, not learned. Thresholds (0.6, 0.7, 0.3, 30 s) are
   chosen by reasoning and testing, not tuned on real events.
4. **Lag is under-estimated.** The overlap shrinks as lag grows, so the recovered delay is
   biased low (a true 10 s may read ~2–6 s). Direction is reliable. ETAs therefore
   err toward warning *earlier*, the safe side. This is why the demo quotes ETA ≈ 2 s but the
   true lead time is 10 s.
5. **Fixed 3-zone line topology**, hardcoded in `config.py`.
6. **Counting error:** MAE ~67 and undercounting in very dense crowds. The detector uses
   density *trends*, which are less sensitive to a constant undercount, but absolute
   thresholds (density > 0.3) are affected.
7. Do **not** claim a prediction "accuracy %". It is a single-event result, reported as
   lead time.

## 6. Suggested script for the review (~2 min)

> "Stampedes are not just crowds that are dense; they are pressure waves that move
> between areas. Our system has edge nodes counting people per zone with CSRNet, and a fog
> server that looks at all zones together.
>
> The fog algorithm has three stages. First, DTW checks whether a zone's recent density
> curve matches the S-shaped buildup that precedes a crush. We use DTW because buildups
> happen at different speeds. Second, cross-correlation checks whether a neighbouring zone
> repeats that curve with a delay, and optical flow confirms people move the same way.
> That gives us a confirmed wave and its travel time. Third, we use that delay to predict
> when the next zone will be hit, and alert it before its own camera sees anything.
>
> CSRNet counting gets an MAE of 67 on the full 182-image ShanghaiTech Part A test set,
> matching the published 68. In the demo, zone 3 is alarmed 10 seconds before it actually
> surges. On real crowd footage the detector gave no false alarms.
>
> Honest limits: the 10-second result is from simulated telemetry because we have no labelled
> real stampede data, and the reference curve is hand-designed from the literature."

## 7. Likely questions

- **Why not just a density threshold?** It can't give early warning, and it fires on
  any dense-but-stable crowd. The negative scenarios show the cascade logic avoids that.
- **Why DTW and not just correlation for Component 1?** Correlation to a fixed template
  fails if the buildup is faster or slower; DTW absorbs the timing difference.
- **Is the 10 s real?** It's measured, on synthetic data with a known 10 s propagation delay.
- **Why is the quoted ETA smaller than the delay?** Known lag under-estimate, see limitation 4.
- **What would you do next?** Learn the reference curve and thresholds from labelled
  real incident data, and test a real multi-camera deployment.
