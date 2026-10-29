"""
cascade_detector.py — the Pressure Cascade Detection Algorithm.

Detects whether a density surge in one zone is a time-delayed copy of a surge in
an adjacent zone (a real propagating pressure wave), as opposed to two
coincidental independent spikes, and predicts when downstream zones will be hit.

Three components, run in order, each gating the next:

  Component 1 — Temporal Pattern Matching (per zone)
      Does this zone's density curve match the S-shaped buildup that precedes a
      crush event, rather than merely being "high"? Dynamic Time Warping against
      a reference sigmoid, converted to a confidence score.

  Component 2 — Cross-Correlation Wave Detection (per adjacent pair)
      Is zone B's curve a time-shifted copy of zone A's? Normalized
      cross-correlation over a range of lags yields both a similarity score and
      the propagation delay. A positive lag means B follows A. Flow vectors must
      also point the same way, confirming the crowd is physically moving A -> B.

  Component 3 — Propagation Delay / ETA Prediction (per downstream zone)
      Given the measured delay and when the upstream zone started surging, when
      will the next zone downstream be hit? Zones predicted to be hit within
      PREEMPTIVE_ALERT_WINDOW seconds get a RED alert before they surge at all.

Each component in isolation is standard signal processing. The novelty is the
combination — pattern-shape gating, then lag-and-direction confirmation, then
forward prediction — applied to fog-layer crowd safety, where the fog server
sees all zones at once and can warn a zone before its own camera shows anything.

Reference curve shape follows crowd dynamics literature documenting S-curve
density buildup before crush events (Helbing et al. 2007; Still 2000).
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


# ===== REFERENCE PRE-STAMPEDE CURVE =====

def _build_reference_curve() -> np.ndarray:
    """Normalized S-curve: slow rise -> acceleration -> plateau.

    Built once at import. Already spans ~0..1 by construction, but is explicitly
    min-max normalized so it is directly comparable to normalized zone series.
    """
    t = np.linspace(0, 1, config.REFERENCE_CURVE_POINTS)
    curve = 1.0 / (1.0 + np.exp(-config.REFERENCE_CURVE_STEEPNESS * (t - 0.5)))
    return normalize_series(curve)


def normalize_series(series: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1].

    A flat series has zero range, which would divide by zero and produce NaN
    (NaN then propagates through DTW and silently poisons the confidence score).
    Flat input is mapped to all-zeros instead: a perfectly flat density curve is
    the opposite of a surge, so zero is the semantically correct floor.
    """
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return series
    lo, hi = float(np.min(series)), float(np.max(series))
    span = hi - lo
    if span <= 0.0:
        return np.zeros_like(series)
    return (series - lo) / span


REFERENCE_CURVE = _build_reference_curve()


# ===== COMPONENT 1 — TEMPORAL PATTERN MATCHING =====

def dtw_distance(series_a: np.ndarray, series_b: np.ndarray) -> float:
    """Dynamic Time Warping distance between two 1-D series.

    DTW rather than pointwise distance because a real buildup may run faster or
    slower than the reference curve; DTW matches shape while allowing the time
    axis to stretch, so a slow buildup and a fast one both match the same
    reference.
    """
    n, m = len(series_a), len(series_b)
    if n == 0 or m == 0:
        return float("inf")

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(series_a[i - 1] - series_b[j - 1])
            dtw_matrix[i][j] = cost + min(
                dtw_matrix[i - 1][j],
                dtw_matrix[i][j - 1],
                dtw_matrix[i - 1][j - 1],
            )
    return float(dtw_matrix[n][m])


def compute_surge_confidence(series: np.ndarray) -> float:
    """Confidence in [0, 1] that a zone's density curve matches a pre-stampede buildup.

    Compares the most recent REFERENCE_CURVE_POINTS samples (30 seconds at the
    2-second data interval) against the reference S-curve using DTW, then maps
    distance to confidence: 0 distance -> 1.0, MAX_DTW_DISTANCE or worse -> 0.0.

    Note on a known limitation: min-max normalization makes this scale-invariant,
    so the shape test alone cannot distinguish a large buildup from a tiny one.
    That is deliberate — absolute magnitude is gated separately by
    MIN_DENSITY_THRESHOLD in detect_cascade, keeping "is this the right shape"
    and "is this actually dense" as two independent conditions.
    """
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return 0.0

    recent = series[-config.REFERENCE_CURVE_POINTS:]
    normalized = normalize_series(recent)

    distance = dtw_distance(normalized, REFERENCE_CURVE)
    if not np.isfinite(distance):
        return 0.0

    confidence = 1.0 - (distance / config.MAX_DTW_DISTANCE)
    return float(max(0.0, min(1.0, confidence)))


# ===== COMPONENT 2 — CROSS-CORRELATION WAVE DETECTION =====

def compute_wave_correlation(
    series_a: np.ndarray,
    series_b: np.ndarray,
    max_lag: int = None,
) -> Tuple[float, int]:
    """Best normalized cross-correlation between two zone density series, and its lag.

    Both series are z-scored (zero mean, unit variance) so correlation measures
    shape similarity independent of each zone's absolute density level and
    camera-specific scaling.

    Returns (best_corr, best_lag). A positive lag means series_b's pattern
    follows series_a's by that many data points — i.e. the wave travels A -> B.
    Multiply by config.DATA_INTERVAL for seconds.

    Only the magnitude of correlation is compared when searching for the best
    lag, but the signed value is returned, so an inverted (anti-correlated)
    relationship surfaces as a negative score and fails the positive threshold
    test in detect_cascade rather than masquerading as a strong match.

    Known bias: because each lag is scored as the mean over the overlapping
    region only, and that region shrinks as |lag| grows, this measure is biased
    toward smaller shifts. The recovered lag therefore tends to slightly
    UNDER-estimate the true delay (a genuine 8s delay may report as ~4-6s).
    Direction (the sign) is recovered reliably, which is what gates detection;
    the magnitude feeds the ETA estimate, so predicted arrival times are
    correspondingly conservative — they err toward warning early rather than late.
    """
    if max_lag is None:
        max_lag = config.MAX_CORRELATION_LAG

    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0, 0

    a = a - np.mean(a)
    b = b - np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    # A flat series carries no shape to correlate; guarding the divisor with 1.0
    # leaves it as all-zeros, which correlates to 0.0 against anything.
    a = a / (std_a if std_a > 0 else 1.0)
    b = b / (std_b if std_b > 0 else 1.0)

    best_corr = 0.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        # Positive lag pairs an EARLIER sample of A with a LATER sample of B, so
        # correlation peaks at positive lag when B's pattern follows A's — i.e.
        # the wave travels A -> B. (Pairing them the other way round inverts the
        # sign convention and reports a genuine A -> B wave as a negative lag,
        # which would then fail the `lag > 0` direction test in detect_cascade.)
        if lag >= 0:
            if lag >= len(b):
                continue
            seg_a = a[: len(a) - lag] if lag > 0 else a
            seg_b = b[lag:]
        else:
            if -lag >= len(a):
                continue
            seg_a = a[-lag:]
            seg_b = b[: len(b) + lag]

        n = min(len(seg_a), len(seg_b))
        if n == 0:
            continue
        corr = float(np.mean(seg_a[:n] * seg_b[:n]))

        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return best_corr, best_lag


def compute_flow_alignment(
    flow_a: Tuple[float, float],
    flow_b: Tuple[float, float],
) -> Tuple[float, bool]:
    """Dot product of two zones' flow vectors, and whether they point the same way.

    Positive means both crowds are moving in the same general direction,
    physically consistent with a wave travelling from one zone into the next.
    Negative means they are converging or moving apart — still potentially
    dangerous (a bottleneck squeeze) but not the propagating-wave case this
    algorithm confirms, so it is reported separately rather than counted as a wave.
    """
    alignment = float(flow_a[0] * flow_b[0] + flow_a[1] * flow_b[1])
    return alignment, alignment > 0


# ===== COMPONENT 3 — PROPAGATION DELAY / ETA PREDICTION =====

def compute_eta(
    propagation_delay: float,
    surge_start_time: Optional[float],
    now: Optional[float] = None,
) -> Optional[float]:
    """Seconds until a downstream zone is predicted to be hit.

    The wave took `propagation_delay` seconds to travel the previous hop, and
    the upstream zone started surging `elapsed` seconds ago; the remaining time
    is the difference, clipped at 0 once the predicted arrival has passed.

    Returns None when the upstream zone has no active surge episode, since
    without a start time there is no anchor from which to measure elapsed time.

    An ETA of exactly 0.0 means the predicted arrival window has already passed
    while the upstream zone is still surging — the wave should be arriving now.
    That is a live "impact imminent" state, distinct from None ("no prediction
    possible"), and both are distinct from a positive ETA ("arriving in N
    seconds"). Live operation produces positive ETAs because detection runs
    continuously from the moment the upstream zone starts surging.
    """
    if surge_start_time is None:
        return None
    now = time.time() if now is None else now
    elapsed = now - surge_start_time
    return float(max(0.0, propagation_delay - elapsed))


# ===== CLOUD LAYER (TIER 3) LOGGING =====

def log_cascade_event(results: dict, path: Optional[str] = None) -> None:
    """Append a cascade detection to the JSONL event log (the optional cloud tier).

    Logging failures are swallowed with a warning: the detector runs inside the
    request path, and a full disk or a permissions problem must never take down
    live crowd monitoring.
    """
    if config.LOG_ONLY_DETECTED_CASCADES and not results.get("cascade_detected"):
        return

    # The debug payload exists to explain the CURRENT state to the dashboard; it
    # repeats the constant reference curve and every normalized series on every
    # run, which would bloat an append-only log for no archival value. Logged
    # events keep the decision and its inputs, not the explainability extras.
    record = {k: v for k, v in results.items() if k != "debug"}

    path = path or config.CASCADE_LOG_PATH
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.warning("Failed to write cascade event log at %s: %s", path, e)


# ===== MAIN DETECTION LOOP =====

def detect_cascade(zone_manager, now: Optional[float] = None) -> dict:
    """Run all three components and return the full cascade assessment.

    Alert levels escalate monotonically: GREEN by default, YELLOW for a zone
    surging on its own, RED for zones in a confirmed wave pair or predicted to
    be hit imminently. A zone's level is never downgraded once raised within a
    single run, so a zone that is both surging and downstream of another wave
    keeps the more severe level.
    """
    now = time.time() if now is None else now

    results = {
        "cascade_detected": False,
        "cascade_score": 0.0,
        "zone_alerts": {},
        "zone_details": {},
        "surge_zones": [],
        "wave_pairs": [],
        "preemptive_alerts": {},
        "timestamp": now,
        # Explainability payload for the dashboard's algorithm-inspection panel.
        # Records the intermediate values behind every decision — including pairs
        # that were REJECTED and why — so the algorithm's reasoning is inspectable
        # rather than only its verdict.
        "debug": {
            "dtw": {},
            "pairs": [],
            "reference_curve": [round(float(v), 4) for v in REFERENCE_CURVE],
            "thresholds": {
                "surge_confidence": config.SURGE_CONFIDENCE_THRESHOLD,
                "min_density": config.MIN_DENSITY_THRESHOLD,
                "wave_correlation": config.WAVE_CORRELATION_THRESHOLD,
                "max_dtw_distance": config.MAX_DTW_DISTANCE,
                "preemptive_window": config.PREEMPTIVE_ALERT_WINDOW,
            },
        },
    }

    # ---- Component 1: per-zone surge pattern matching ----
    for zone_id in zone_manager.get_all_zone_ids():
        series = zone_manager.get_density_timeseries(zone_id)
        results["zone_alerts"][zone_id] = "GREEN"

        if len(series) < config.MIN_POINTS_FOR_SURGE_CHECK:
            continue

        surge_confidence = compute_surge_confidence(series)
        latest = zone_manager.get_latest(zone_id)
        density = latest["density_avg"] if latest else 0.0

        is_surging = (
            surge_confidence > config.SURGE_CONFIDENCE_THRESHOLD
            and density > config.MIN_DENSITY_THRESHOLD
        )

        results["zone_details"][zone_id] = {
            "surge_confidence": round(surge_confidence, 3),
            "density": round(density, 4),
            "is_surging": is_surging,
        }

        # Explainability: the raw DTW distance behind the confidence score, and
        # the normalized curve actually compared against the reference. Exposed
        # so the dashboard can show WHY a zone did or did not register a surge,
        # rather than only the final verdict.
        recent = series[-config.REFERENCE_CURVE_POINTS:]
        normalized = normalize_series(recent)
        distance = dtw_distance(normalized, REFERENCE_CURVE)
        results["debug"]["dtw"][zone_id] = {
            "dtw_distance": round(float(distance), 4) if np.isfinite(distance) else None,
            "max_dtw_distance": config.MAX_DTW_DISTANCE,
            "surge_confidence": round(surge_confidence, 3),
            "normalized_curve": [round(float(v), 4) for v in normalized],
            "passed_confidence": bool(
                surge_confidence > config.SURGE_CONFIDENCE_THRESHOLD
            ),
            "passed_density": bool(density > config.MIN_DENSITY_THRESHOLD),
        }

        if is_surging:
            results["surge_zones"].append(zone_id)
            results["zone_alerts"][zone_id] = "YELLOW"

    # ---- Component 2: adjacent-pair wave confirmation ----
    cascade_score = 0.0

    # Every adjacent pair is evaluated so the inspection panel always has data to
    # show, but only pairs whose SOURCE zone is surging can be confirmed as a
    # wave — a wave must originate from a zone that is actually surging.
    for zone_a in zone_manager.get_all_zone_ids():
        series_a = zone_manager.get_density_timeseries(zone_a)
        if len(series_a) < config.MIN_POINTS_FOR_CORRELATION:
            continue

        source_is_surging = zone_a in results["surge_zones"]

        for zone_b in zone_manager.get_adjacent_zones(zone_a):
            series_b = zone_manager.get_density_timeseries(zone_b)
            if len(series_b) < config.MIN_POINTS_FOR_CORRELATION:
                continue

            wave_corr, lag = compute_wave_correlation(series_a, series_b)
            flow_alignment, flow_aligned = compute_flow_alignment(
                zone_manager.get_flow_vector(zone_a),
                zone_manager.get_flow_vector(zone_b),
            )

            passed_corr = wave_corr > config.WAVE_CORRELATION_THRESHOLD
            passed_lag = lag > 0
            is_wave = source_is_surging and passed_corr and passed_lag and flow_aligned

            # Record every evaluated pair, confirmed or not. A rejected pair is
            # the more informative case for inspection: it shows which of the
            # conditions vetoed the wave (source not surging / weak correlation /
            # no propagation delay / crowds moving in opposing directions).
            if not source_is_surging:
                reason = f"{zone_a} is not surging"
            elif not passed_corr:
                reason = f"correlation {wave_corr:.3f} <= {config.WAVE_CORRELATION_THRESHOLD}"
            elif not passed_lag:
                reason = f"lag {lag} not positive (B does not follow A)"
            elif not flow_aligned:
                reason = f"flow opposed (dot {flow_alignment:.3f} <= 0)"
            else:
                reason = ""

            results["debug"]["pairs"].append({
                "from": zone_a,
                "to": zone_b,
                "correlation": round(wave_corr, 3),
                "lag_points": lag,
                "lag_seconds": lag * config.DATA_INTERVAL,
                "flow_alignment": round(flow_alignment, 3),
                "source_surging": source_is_surging,
                "passed_correlation": passed_corr,
                "passed_lag": passed_lag,
                "passed_flow": flow_aligned,
                "confirmed": is_wave,
                "rejected_because": reason,
            })

            if not is_wave:
                continue

            confidence_a = results["zone_details"][zone_a]["surge_confidence"]
            cascade_score += wave_corr * confidence_a
            propagation_delay = lag * config.DATA_INTERVAL

            results["wave_pairs"].append({
                "from": zone_a,
                "to": zone_b,
                "correlation": round(wave_corr, 3),
                "lag_seconds": propagation_delay,
                "flow_aligned": True,
            })

            # Both ends of a confirmed wave are in active danger.
            results["zone_alerts"][zone_a] = "RED"
            results["zone_alerts"][zone_b] = "RED"

            # ---- Component 3: predict the next zone downstream ----
            downstream_zones = [
                z for z in zone_manager.get_adjacent_zones(zone_b)
                if z != zone_a and z not in results["surge_zones"]
            ]
            for dz in downstream_zones:
                eta = compute_eta(
                    propagation_delay,
                    zone_manager.get_surge_start_time(zone_b),
                    now=now,
                )
                if eta is None or eta > config.PREEMPTIVE_ALERT_WINDOW:
                    continue

                existing = results["preemptive_alerts"].get(dz)
                # If several waves converge on one zone, keep the most urgent.
                if existing is not None and existing["eta_seconds"] <= eta:
                    continue

                results["preemptive_alerts"][dz] = {
                    "eta_seconds": round(eta, 1),
                    "triggered_by_wave": f"{zone_a} -> {zone_b}",
                    "propagation_delay": propagation_delay,
                }
                results["zone_alerts"][dz] = "RED"

    results["cascade_score"] = round(cascade_score, 4)
    results["cascade_detected"] = cascade_score > 0

    log_cascade_event(results)

    return results
