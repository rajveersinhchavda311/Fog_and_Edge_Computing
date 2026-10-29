"""
run_demo.py — standalone demonstration of the CrowdShield fog/edge system.

Runs the COMPLETE system on one machine with no external dependencies:

    no Docker      no GPU      no CSRNet weights      no video files      no camera

The fog server runs in a background thread and the three "edge nodes" are threads
that generate synthetic-but-realistic telemetry instead of doing computer vision.
Everything downstream of the telemetry packet — the ZoneManager, the full
Pressure Cascade Detection Algorithm, the HTTP/WebSocket API and the dashboard —
is the REAL production code, unmodified. Only the sensor is simulated.

    python run_demo.py                 # ~95s scenario, dashboard on :8000
    python run_demo.py --no-server     # console only, no HTTP server
    python run_demo.py --speed 2       # run twice as fast
    python run_demo.py --loop          # repeat forever (for an unattended display)

The scenario deliberately exercises all three algorithm components in sequence:
a lone S-curve buildup in zone_1 (Component 1), the same curve reappearing in
zone_2 ten seconds later while the crowd flows the same way (Component 2), and a
predicted arrival in zone_3 issued before zone_3's own density has moved at all
(Component 3).
"""

import argparse
import math
import os
import random
import sys
import threading
import time
from typing import Dict, List, Optional

# --- make the sibling packages importable ------------------------------------
# fog/ and shared/ are siblings of demo/. The fog modules use flat imports
# (`import config`), so fog/ must be on sys.path itself rather than imported as
# a package. Done before those imports so this file runs from any directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FOG = os.path.join(_HERE, "..", "fog")
if _FOG not in sys.path:
    sys.path.insert(0, _FOG)

try:
    import config                                # noqa: E402  (fog/config.py)
    from cascade_detector import detect_cascade  # noqa: E402
    from zone_manager import ZoneManager         # noqa: E402
except ImportError as _e:
    # numpy is the one hard requirement — the detection algorithm is built on it.
    # FastAPI/uvicorn are optional (the demo falls back to console-only), so this
    # only fires for a genuinely missing core dependency or a broken checkout.
    # Reported plainly rather than as a traceback, since this is the first thing
    # someone sees on a fresh machine.
    sys.stderr.write(
        "\n  Cannot start: a required module is missing.\n"
        f"    {_e.__class__.__name__}: {_e}\n\n"
        "  The demo needs numpy. Install it with:\n"
        "      pip install numpy\n\n"
        "  For the live dashboard as well:\n"
        "      pip install numpy fastapi uvicorn websockets\n\n"
        f"  (Also confirm this file is still inside the project, next to '{_FOG}'.)\n\n"
    )
    raise SystemExit(1)


# ===== SCENARIO DEFINITION ===================================================

# Surge onset per zone, in scenario seconds. The 10-second spacing is the
# ground-truth propagation delay the algorithm has to rediscover from the data.
SURGE_START = {
    "zone_1": 15.0,
    "zone_2": 25.0,
    "zone_3": 35.0,
}

# Flow vectors, all pointing the same way (zone_1 -> zone_2 -> zone_3). Component
# 2 requires a positive dot product between adjacent zones to confirm a wave, so
# these encode "the crowd is physically moving down the corridor".
FLOW_VECTOR = {
    "zone_1": (0.6, 0.1),
    "zone_2": (0.5, 0.1),
    "zone_3": (0.5, 0.1),
}

SURGE_DURATION = 20.0      # seconds from onset to plateau
SCENARIO_LENGTH = 95.0     # total run time; > 75s so the decay back to GREEN is visible
PEOPLE_PER_DENSITY = 840   # density -> displayed head count (cosmetic only)

TRUE_PROPAGATION_DELAY = SURGE_START["zone_2"] - SURGE_START["zone_1"]  # 10.0s


def generate_density(t: float, surge_start: float, surge_duration: float = SURGE_DURATION) -> float:
    """Synthetic density for one zone at scenario time t.

    Four phases: low noisy baseline, S-curve rise, plateau-then-decline, and a
    return to baseline. The S-curve shape is what Component 1's DTW matches
    against its reference pre-stampede curve; the gaussian noise keeps the data
    from being unrealistically clean.
    """
    if t < surge_start:
        return 0.10 + random.gauss(0, 0.02)

    if t < surge_start + surge_duration:
        progress = (t - surge_start) / surge_duration
        density = 0.10 + 0.50 * (1 / (1 + math.exp(-10 * (progress - 0.5))))
        return density + random.gauss(0, 0.02)

    if t < surge_start + surge_duration + 20:
        decline_progress = (t - surge_start - surge_duration) / 20
        density = 0.60 - 0.40 * decline_progress
        return max(0.10, density + random.gauss(0, 0.02))

    return 0.10 + random.gauss(0, 0.02)


def generate_flow(zone_id: str, t: float, surging: bool) -> tuple:
    """Flow vector for a zone. Near-still while quiet, directional once surging."""
    fx, fy = FLOW_VECTOR[zone_id]
    if not surging:
        # Low baseline drift: small and random, so quiet zones do not accidentally
        # present a strong aligned flow.
        return (random.gauss(0, 0.08), random.gauss(0, 0.08))
    return (fx + random.gauss(0, 0.04), fy + random.gauss(0, 0.04))


# ===== CONSOLE PRESENTATION ==================================================

class Palette:
    """ANSI colours, disabled automatically when the terminal cannot show them."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.enabled else s

    def green(self, s):  return self._w("32", s)
    def yellow(self, s): return self._w("33", s)
    def red(self, s):    return self._w("31", s)
    def bold(self, s):   return self._w("1", s)
    def dim(self, s):    return self._w("2", s)


def supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        # Enable ANSI processing on Windows 10+ consoles; fall back to plain text.
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


def supports_unicode() -> bool:
    """Whether stdout can encode the box-drawing and emoji characters.

    Windows consoles frequently run a legacy code page (cp1252/cp437) that
    cannot encode these, which would raise UnicodeEncodeError mid-demo. Checked
    up front so the demo silently degrades to ASCII instead of crashing.
    """
    enc = getattr(sys.stdout, "encoding", None) or ""
    try:
        "┌─┐│└┘├┤┬┴┼╔╗╚╝║═⚠🔴🟡🟢→".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError, TypeError):
        return False


class Renderer:
    """Draws the banner, the live zone table and the closing summary."""

    def __init__(self, unicode_ok: bool, palette: Palette):
        self.u = unicode_ok
        self.c = palette

    # -- box drawing --
    def _chars(self):
        if self.u:
            return dict(h="─", v="│", tl="┌", tr="┐", bl="└", br="┘",
                        lt="├", rt="┤", tt="┬", bt="┴", x="┼")
        return dict(h="-", v="|", tl="+", tr="+", bl="+", br="+",
                    lt="+", rt="+", tt="+", bt="+", x="+")

    def banner(self, server_url: Optional[str], duration: float):
        lines = [
            "CrowdShield - Fog-Assisted Stampede Detection Demo",
            f"Dashboard: {server_url}" if server_url else "Console-only mode (--no-server)",
            f"3 simulated edge zones, ~{duration:.0f} second scenario",
            "Watch for: zone_1 surge -> zone_2 cascade -> zone_3 ETA",
        ]
        width = max(len(s) for s in lines) + 2
        if self.u:
            print("\n" + "╔" + "═" * width + "╗")
            for s in lines:
                print("║ " + s.ljust(width - 1) + "║")
            print("╚" + "═" * width + "╝")
        else:
            print("\n" + "=" * (width + 2))
            for s in lines:
                print("| " + s.ljust(width - 1) + "|")
            print("=" * (width + 2))
        print(self.c.dim("  No GPU, no model weights, no video files, no camera required."))
        print(self.c.dim("  Telemetry is synthetic; the detection algorithm is the real one.\n"))

    def badge(self, level: str) -> str:
        """Fixed-width alert badge, so table columns stay aligned."""
        if self.u:
            dot = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(level, "  ")
            text = f"{dot} {level}"
            # Emoji render as double-width; pad to a consistent visual width.
            pad = 9 - len(level)
            text = text + " " * max(0, pad)
        else:
            text = level.ljust(11)
        colour = {"GREEN": self.c.green, "YELLOW": self.c.yellow, "RED": self.c.red}
        return colour.get(level, lambda s: s)(text)

    def table(self, t: float, zones: List[str], snapshot: Dict, cascade: Dict):
        ch = self._chars()
        widths = (10, 7, 8, 9, 12)

        def row(cells, left, mid, right):
            return left + mid.join(ch["h"] * w for w in widths) + right

        print(f"\n{self.c.bold(f'[T={t:>5.1f}s]')} " + row(None, ch["tl"], ch["tt"], ch["tr"]))
        header = (ch["v"] + " Zone".ljust(widths[0])
                  + ch["v"] + " Count".ljust(widths[1])
                  + ch["v"] + " Density".ljust(widths[2])
                  + ch["v"] + " Surge%".ljust(widths[3])
                  + ch["v"] + " Alert".ljust(widths[4]) + ch["v"])
        print("          " + header)
        print("          " + row(None, ch["lt"], ch["x"], ch["rt"]))

        details = cascade.get("zone_details", {})
        alerts = cascade.get("zone_alerts", {})
        for z in zones:
            latest = snapshot.get(z)
            count = latest["count"] if latest else 0
            dens = latest["density_avg"] if latest else 0.0
            conf = details.get(z, {}).get("surge_confidence", 0.0)
            level = alerts.get(z, "GREEN")
            line = (ch["v"] + f" {z}".ljust(widths[0])
                    + ch["v"] + f" {count:>5}".ljust(widths[1])
                    + ch["v"] + f"  {dens:>5.2f}".ljust(widths[2])
                    + ch["v"] + f"  {conf * 100:>5.1f}%".ljust(widths[3])
                    + ch["v"] + " " + self.badge(level) + ch["v"])
            print("          " + line)
        print("          " + row(None, ch["bl"], ch["bt"], ch["br"]))

        # -- cascade verdict --
        arrow = "→" if self.u else "->"
        warn = "⚠" if self.u else "!"
        if cascade.get("cascade_detected"):
            print("          " + self.c.red(self.c.bold(
                f"CASCADE DETECTED! Score: {cascade['cascade_score']:.4f}")))
        for w in cascade.get("wave_pairs", []):
            print("          " + self.c.red(
                f"Wave: {w['from']} {arrow} {w['to']} "
                f"(corr: {w['correlation']:.2f}, lag: {w['lag_seconds']}s)"))
        for z, p in cascade.get("preemptive_alerts", {}).items():
            print("          " + self.c.red(self.c.bold(
                f"{warn} {z} PREEMPTIVE ALERT - Estimated surge in "
                f"{p['eta_seconds']:.1f} seconds")))


# ===== SIMULATED EDGE NODES ==================================================

class SimulatedEdgeNode:
    """One synthetic edge camera.

    Produces the same telemetry packet shape that the real edge_node.py emits
    (see shared/models.py TelemetryPacket), so the fog server cannot tell the
    difference between this and a real camera.
    """

    def __init__(self, zone_id: str, surge_start: float):
        self.zone_id = zone_id
        self.surge_start = surge_start

    def sample(self, t: float, wall_clock: float) -> dict:
        density = generate_density(t, self.surge_start)
        # "Surging" here only shapes the FLOW vector; it is scenario ground
        # truth, never fed to the detector, which must infer surges itself.
        surging = self.surge_start <= t < self.surge_start + SURGE_DURATION + 20
        fx, fy = generate_flow(self.zone_id, t, surging)
        return {
            "zone_id": self.zone_id,
            "count": int(density * PEOPLE_PER_DENSITY),
            "density_avg": round(density, 4),
            "density_peak": round(density * 1.4, 4),
            "flow_x": round(fx, 4),
            "flow_y": round(fy, 4),
            "flow_magnitude": round(math.hypot(fx, fy), 4),
            "timestamp": wall_clock,
        }


# ===== FOG SERVER (BACKGROUND THREAD) ========================================

class FogServerThread:
    """Runs the real FastAPI fog server in a daemon thread.

    Importing fog_server binds its module-level ZoneManager, which the demo then
    drives directly. That is deliberate: the demo feeds telemetry through the
    same state object the HTTP API reads from, so the dashboard and the console
    always show identical data with no risk of them diverging.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host, self.port = host, port
        self.thread: Optional[threading.Thread] = None
        self.server = None
        self.module = None
        self.error: Optional[str] = None

    def start(self, timeout: float = 15.0) -> bool:
        try:
            import uvicorn
            import fog_server
        except ImportError as e:
            self.error = (f"{e.__class__.__name__}: {e}. "
                          "Install with: pip install fastapi uvicorn websockets numpy")
            return False

        # Check the port up front. uvicorn logs its own bind error and exits the
        # thread without raising into this one, so probing here is what lets the
        # demo fail in a fraction of a second with an actionable message rather
        # than sitting through the full readiness timeout.
        if self._port_in_use():
            self.error = (f"port {self.port} is already in use by another process. "
                          f"Try --port 8001, or stop the other server.")
            return False

        self.module = fog_server
        cfg = uvicorn.Config(fog_server.app, host=self.host, port=self.port,
                             log_level="error", access_log=False)
        self.server = uvicorn.Server(cfg)

        self.thread = threading.Thread(target=self._run, daemon=True, name="fog-server")
        self.thread.start()

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.error:
                return False
            if getattr(self.server, "started", False):
                return True
            time.sleep(0.05)

        self.error = f"server did not become ready within {timeout:.0f}s"
        return False

    def _port_in_use(self) -> bool:
        """True if something is already listening on the target port."""
        import socket
        probe_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                return s.connect_ex((probe_host, self.port)) == 0
            except OSError:
                return False

    def _run(self):
        try:
            self.server.run()
        except OSError as e:
            # Almost always "address already in use" — report it usefully rather
            # than letting the thread die silently and the demo hang on startup.
            self.error = (f"could not bind {self.host}:{self.port} ({e}). "
                          "Another process is using that port; try --port 8001 "
                          "or stop the other server.")
        except Exception as e:                                  # pragma: no cover
            self.error = f"{e.__class__.__name__}: {e}"

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)


# ===== EVENT TRACKING (for the closing summary) ==============================

class ScenarioLog:
    """Records when each milestone actually happened, for the final report.

    Every number in the summary is measured from the run rather than assumed, so
    the demo reports what the algorithm genuinely did on this particular run.
    """

    def __init__(self):
        self.first_surge: Dict[str, float] = {}       # zone -> t it was first flagged surging
        self.first_cascade: Optional[float] = None
        self.first_wave: Optional[tuple] = None       # (t, from, to, lag_seconds)
        self.preemptive: Dict[str, float] = {}        # zone -> t of first preemptive alert
        self.preemptive_eta: Dict[str, float] = {}    # zone -> ETA quoted at that moment
        self.returned_green: Optional[float] = None
        self.max_score: float = 0.0

    def observe(self, t: float, cascade: Dict):
        for z, d in cascade.get("zone_details", {}).items():
            if d.get("is_surging") and z not in self.first_surge:
                self.first_surge[z] = t

        if cascade.get("cascade_detected"):
            if self.first_cascade is None:
                self.first_cascade = t
            self.max_score = max(self.max_score, cascade.get("cascade_score", 0.0))
            self.returned_green = None
        elif self.first_cascade is not None and self.returned_green is None:
            if all(v == "GREEN" for v in cascade.get("zone_alerts", {}).values()):
                self.returned_green = t

        for w in cascade.get("wave_pairs", []):
            if self.first_wave is None:
                self.first_wave = (t, w["from"], w["to"], w["lag_seconds"])

        for z, p in cascade.get("preemptive_alerts", {}).items():
            if z not in self.preemptive:
                self.preemptive[z] = t
                self.preemptive_eta[z] = p["eta_seconds"]


def print_summary(log: ScenarioLog, renderer: Renderer, palette: Palette):
    c = palette
    bar = "═" * 66 if renderer.u else "=" * 66
    print("\n" + bar)
    print(c.bold("  DEMO SUMMARY"))
    print(bar)

    if log.first_cascade is None:
        print(c.yellow("  No cascade was detected during this run."))
        print(bar + "\n")
        return

    print(f"  Cascade first detected at        T = {log.first_cascade:.0f}s")
    if log.first_wave:
        wt, wf, wto, wlag = log.first_wave
        arrow = "→" if renderer.u else "->"
        print(f"  First confirmed wave             {wf} {arrow} {wto} at T = {wt:.0f}s "
              f"(measured lag {wlag}s)")
    print(f"  Peak cascade score               {log.max_score:.4f}")

    # The headline claim: warning a zone before its own sensor saw anything.
    z3_alert = log.preemptive.get("zone_3")
    z3_actual = log.first_surge.get("zone_3")
    if z3_alert is not None:
        eta = log.preemptive_eta.get("zone_3", 0.0)
        print(f"\n  {c.bold('zone_3 preemptive alert issued at')} T = {z3_alert:.0f}s "
              f"(quoted ETA {eta:.0f}s)")
        if z3_actual is not None:
            lead = z3_actual - z3_alert
            print(f"  zone_3 actually began surging at  T = {z3_actual:.0f}s")
            print(c.bold(c.green(
                f"  -> WARNING ISSUED {lead:.0f} SECONDS BEFORE THE SURGE ARRIVED")))
            print(c.dim(f"     Ground-truth propagation delay was "
                        f"{TRUE_PROPAGATION_DELAY:.0f}s "
                        f"(zone_1 onset {SURGE_START['zone_1']:.0f}s, "
                        f"zone_2 onset {SURGE_START['zone_2']:.0f}s)."))
        else:
            print(c.dim("     zone_3 did not reach the surge threshold in this run."))

    if log.returned_green is not None:
        print(f"\n  All zones returned to GREEN at   T = {log.returned_green:.0f}s")

    print(bar + "\n")


# ===== MAIN LOOP =============================================================

def run_scenario(args) -> int:
    palette = Palette(supports_colour() and not args.no_color)
    renderer = Renderer(supports_unicode() and not args.ascii, palette)

    if args.seed is not None:
        random.seed(args.seed)

    server: Optional[FogServerThread] = None
    zone_manager: ZoneManager
    url: Optional[str] = None

    if args.no_server:
        # Console-only: drive a local ZoneManager directly. Guarantees the demo
        # still runs if FastAPI/uvicorn are unavailable or the port is taken.
        zone_manager = ZoneManager()
    else:
        server = FogServerThread(host=args.host, port=args.port)
        if server.start():
            zone_manager = server.module.zone_manager
            display_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
            url = f"http://{display_host}:{args.port}"
        else:
            # Degrade rather than abort: the algorithm demonstration is the point,
            # and it does not require the HTTP server.
            print(palette.yellow(f"\n  Could not start the fog server: {server.error}"))
            print(palette.yellow("  Continuing in console-only mode.\n"))
            server = None
            zone_manager = ZoneManager()

    renderer.banner(url, SCENARIO_LENGTH / args.speed)
    if url:
        print(palette.dim(f"  Open {url} in a browser to watch the live dashboard.\n"))
        time.sleep(2.0)  # a moment to open the browser before the scenario starts

    nodes = [SimulatedEdgeNode(z, SURGE_START[z]) for z in ("zone_1", "zone_2", "zone_3")]
    zones = [n.zone_id for n in nodes]
    log = ScenarioLog()

    tick = config.DATA_INTERVAL / args.speed     # wall-clock seconds per data point
    steps = int(SCENARIO_LENGTH / config.DATA_INTERVAL) + 1
    started = time.time()

    for step in range(steps):
        t = step * config.DATA_INTERVAL          # scenario time
        wall = time.time()

        for node in nodes:
            zone_manager.update(node.sample(t, wall))

        cascade = detect_cascade(zone_manager, now=wall)
        if server is not None:
            # Keep the HTTP/WebSocket API's view in sync with what we just computed,
            # so the dashboard and the console never disagree.
            server.module.latest_cascade = cascade

        log.observe(t, cascade)

        snapshot = {z: zone_manager.get_latest(z) for z in zones}
        renderer.table(t, zones, snapshot, cascade)

        # Sleep the remainder of this tick, accounting for time already spent.
        elapsed = time.time() - started
        target = (step + 1) * tick
        if target > elapsed:
            time.sleep(target - elapsed)

    print_summary(log, renderer, palette)

    if server is not None:
        if args.hold:
            print(palette.dim(f"  Server still running at {url} — press Ctrl+C to exit.\n"))
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print()
        server.stop()

    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Standalone CrowdShield fog/edge demo — no GPU, weights, or video required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_demo.py                  full demo with live dashboard
  python run_demo.py --speed 2        run at 2x speed
  python run_demo.py --no-server      console only (no FastAPI needed)
  python run_demo.py --hold           keep the dashboard up after the scenario
  python run_demo.py --loop           repeat forever (unattended display)
  python run_demo.py --ascii          plain ASCII output for legacy terminals
""")
    p.add_argument("--host", default="127.0.0.1", help="fog server bind address")
    p.add_argument("--port", type=int, default=8000, help="fog server port")
    p.add_argument("--speed", type=float, default=1.0,
                   help="scenario speed multiplier (2 = twice as fast)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for a byte-identical repeat run")
    p.add_argument("--no-server", action="store_true",
                   help="console only; do not start the HTTP server")
    p.add_argument("--hold", action="store_true",
                   help="keep the server running after the scenario ends")
    p.add_argument("--loop", action="store_true", help="repeat the scenario forever")
    p.add_argument("--ascii", action="store_true", help="force plain ASCII output")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    args = p.parse_args(argv)

    if args.speed <= 0:
        p.error("--speed must be greater than 0")
    if not (0 < args.port < 65536):
        p.error("--port must be between 1 and 65535")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        while True:
            rc = run_scenario(args)
            if not args.loop:
                return rc
            print("\n  Restarting scenario (--loop)...\n")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\n  Demo stopped.\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
