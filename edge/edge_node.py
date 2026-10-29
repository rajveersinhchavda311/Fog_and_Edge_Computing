"""
edge_node.py — single edge camera node.

Captures video from a webcam or file, periodically runs CSRNet crowd density
inference and Farneback optical flow, and streams telemetry to a fog server.
Also polls the fog server for alert status on its own zone.

Multiple instances of this script run simultaneously (one per camera/zone),
each pointed at the same fog server with a different --zone id.
"""

import argparse
import collections
import threading
import time

import cv2
import numpy as np
import requests

import config
from predict import CrowdCounter


class PacketBuffer:
    """Thread-safe bounded buffer of the most recent unsent telemetry packets.

    Deliberately a small ring buffer (maxlen=config.PACKET_BUFFER_SIZE) rather
    than an unbounded queue: telemetry is a live stream, so if the fog server
    is down for a while we only care about resending the most recent state
    once it comes back, not replaying a long backlog of stale packets.
    """

    def __init__(self, maxlen):
        self._buf = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, packet):
        with self._lock:
            self._buf.append(packet)

    def snapshot(self):
        with self._lock:
            return list(self._buf)

    def remove(self, packets):
        """Remove specific packets (by identity of their timestamp) that were
        successfully sent, leaving any not-yet-sent packets in place."""
        sent_timestamps = {p["timestamp"] for p in packets}
        with self._lock:
            remaining = [p for p in self._buf if p["timestamp"] not in sent_timestamps]
            self._buf.clear()
            self._buf.extend(remaining)

    def clear(self):
        with self._lock:
            self._buf.clear()


def build_telemetry_packet(zone_id, count, density_avg, density_peak, flow_x, flow_y, flow_magnitude):
    return {
        "zone_id": zone_id,
        "count": count,
        "density_avg": density_avg,
        "density_peak": density_peak,
        "flow_x": flow_x,
        "flow_y": flow_y,
        "flow_magnitude": flow_magnitude,
        "timestamp": time.time(),
    }


def compute_optical_flow(prev_gray, gray):
    """Farneback dense optical flow between two grayscale frames.

    Parameters copied from the reference implementation in the original
    video_pipeline.py (cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
    0.5, 3, 15, 3, 5, 1.2, 0)) so behavior matches the existing pipeline.

    Returns (flow_x, flow_y, flow_magnitude): the mean flow vector across all
    pixels (not just moving ones) and its magnitude — a simple crowd-movement
    direction/speed summary for this frame pair.
    """
    p = config.FARNEBACK_PARAMS
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        p["pyr_scale"], p["levels"], p["winsize"],
        p["iterations"], p["poly_n"], p["poly_sigma"], p["flags"],
    )
    flow_x = float(np.mean(flow[..., 0]))
    flow_y = float(np.mean(flow[..., 1]))
    flow_magnitude = float(np.hypot(flow_x, flow_y))
    return flow_x, flow_y, flow_magnitude


class EdgeNode:
    def __init__(self, zone_id, source, fog_url, model_path, inference_interval, device=None):
        self.zone_id = zone_id
        self.source = source
        self.fog_url = fog_url.rstrip("/")
        self.inference_interval = inference_interval

        print(f"[{self.zone_id}] Loading CSRNet from {model_path} ...")
        self.crowd_counter = CrowdCounter(model_path, device=device)
        print(f"[{self.zone_id}] Model loaded.")

        self.packet_buffer = PacketBuffer(config.PACKET_BUFFER_SIZE)
        self.latest_packet_lock = threading.Lock()
        self.latest_packet = None

        self.stop_event = threading.Event()

    def open_capture(self):
        source = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source!r}")
        return cap

    # ---- capture + inference loop (main thread) ----

    def run_capture_loop(self):
        cap = self.open_capture()
        print(f"[{self.zone_id}] Capture opened on source={self.source!r}. "
              f"Running inference every {self.inference_interval} frames.")

        prev_gray = None
        frame_idx = 0

        last_count = 0
        last_density_avg = 0.0
        last_density_peak = 0.0
        last_flow_x = 0.0
        last_flow_y = 0.0
        last_flow_magnitude = 0.0

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"[{self.zone_id}] End of video source or read failure. Stopping capture loop.")
                    break

                frame = cv2.resize(frame, (config.RESIZE_W, config.RESIZE_H))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if frame_idx % self.inference_interval == 0:
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = self.crowd_counter.predict(rgb_frame)
                        last_count = result["count"]
                        last_density_avg = result["density_avg"]
                        last_density_peak = result["density_peak"]
                    except Exception as e:
                        print(f"[{self.zone_id}] WARNING: CSRNet inference failed: {e}")

                    if prev_gray is not None:
                        try:
                            last_flow_x, last_flow_y, last_flow_magnitude = compute_optical_flow(prev_gray, gray)
                        except Exception as e:
                            print(f"[{self.zone_id}] WARNING: optical flow failed: {e}")

                    packet = build_telemetry_packet(
                        self.zone_id, last_count, last_density_avg, last_density_peak,
                        last_flow_x, last_flow_y, last_flow_magnitude,
                    )
                    with self.latest_packet_lock:
                        self.latest_packet = packet

                prev_gray = gray
                frame_idx += 1

        finally:
            cap.release()
            self.stop_event.set()
            print(f"[{self.zone_id}] Capture loop stopped.")

    # ---- telemetry sender thread ----

    def run_telemetry_loop(self):
        url = f"{self.fog_url}/ingest"
        while not self.stop_event.is_set():
            with self.latest_packet_lock:
                packet = self.latest_packet

            if packet is not None:
                self.packet_buffer.add(packet)
                pending = self.packet_buffer.snapshot()

                sent = []
                for p in pending:
                    try:
                        resp = requests.post(url, json=p, timeout=config.HTTP_POST_TIMEOUT_SEC)
                        if resp.status_code == 200:
                            sent.append(p)
                        else:
                            print(f"[{self.zone_id}] WARNING: fog server returned "
                                  f"status {resp.status_code} for /ingest")
                            break
                    except requests.exceptions.RequestException as e:
                        print(f"[{self.zone_id}] WARNING: fog server unreachable "
                              f"({e.__class__.__name__}). Buffering packet and retrying.")
                        break

                if sent:
                    self.packet_buffer.remove(sent)

            self.stop_event.wait(config.TELEMETRY_INTERVAL_SEC)

    # ---- alert polling thread ----

    def run_alert_poll_loop(self):
        url = f"{self.fog_url}/alert/{self.zone_id}"
        while not self.stop_event.is_set():
            try:
                resp = requests.get(url, timeout=config.HTTP_GET_TIMEOUT_SEC)
                if resp.status_code == 200:
                    data = resp.json()
                    level = data.get("level", "GREEN")
                    if level in ("YELLOW", "RED"):
                        eta = data.get("eta")
                        eta_str = f"{eta:.1f}s" if isinstance(eta, (int, float)) else "unknown"
                        print("\n" + "!" * 60)
                        print(f"  ALERT [{level}] — zone={self.zone_id}")
                        print(f"  ETA to impact: {eta_str}")
                        print(f"  message: {data.get('message', '')}")
                        print("!" * 60 + "\n")
            except requests.exceptions.RequestException as e:
                print(f"[{self.zone_id}] WARNING: alert poll failed "
                      f"({e.__class__.__name__}). Will retry.")

            self.stop_event.wait(config.ALERT_POLL_INTERVAL_SEC)

    # ---- lifecycle ----

    def run(self):
        telemetry_thread = threading.Thread(target=self.run_telemetry_loop, daemon=True)
        alert_thread = threading.Thread(target=self.run_alert_poll_loop, daemon=True)
        telemetry_thread.start()
        alert_thread.start()

        try:
            self.run_capture_loop()
        except KeyboardInterrupt:
            print(f"\n[{self.zone_id}] Interrupted by user.")
        finally:
            self.stop_event.set()
            telemetry_thread.join(timeout=3)
            alert_thread.join(timeout=3)
            print(f"[{self.zone_id}] Shutdown complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Edge camera node for fog/edge stampede detection.")
    parser.add_argument("--zone", required=True, help="Zone identifier, e.g. zone_1")
    parser.add_argument("--source", required=True, help="Video source: webcam index (e.g. 0) or path to video file")
    parser.add_argument("--fog-url", default=config.DEFAULT_FOG_URL, help="Fog server base URL")
    parser.add_argument("--model-path", default=config.DEFAULT_MODEL_PATH, help="Path to CSRNet checkpoint")
    parser.add_argument("--interval", type=int, default=config.DEFAULT_INFERENCE_INTERVAL,
                         help="Run inference every N frames")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu (default: auto-detect)")
    return parser.parse_args()


def main():
    args = parse_args()
    node = EdgeNode(
        zone_id=args.zone,
        source=args.source,
        fog_url=args.fog_url,
        model_path=args.model_path,
        inference_interval=args.interval,
        device=args.device,
    )
    node.run()


if __name__ == "__main__":
    main()
