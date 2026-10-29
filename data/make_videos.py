"""Generate three demo videos simulating a crowd surge propagating zone_1 -> zone_2 -> zone_3."""
import cv2, numpy as np, os

W, H, FPS = 320, 240, 10
DURATION = 90                      # seconds; long enough for a sustained demo
N = FPS * DURATION

def density_at(t_norm, delay):
    """Sigmoid buildup, shifted later for downstream zones."""
    x = (t_norm - 0.45 - delay) * 12.0
    return 0.06 + 0.80 / (1.0 + np.exp(-x))

def render(path, delay, drift):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    rng = np.random.default_rng(abs(hash(path)) % (2**32))
    for i in range(N):
        t = i / N
        d = density_at(t, delay)
        frame = np.full((H, W, 3), 18, np.uint8)
        n = int(d * 130)                       # people count scales with density
        for k in range(n):
            # Crowd drifts in the shared direction, so flow vectors align.
            px = int((rng.random() * W + i * drift[0] * 2.0) % W)
            py = int((rng.random() * H + i * drift[1] * 2.0) % H)
            cv2.circle(frame, (px, py), 3, (200, 200, 200), -1)
        out.write(frame)
    out.release()
    print(f"{os.path.basename(path)}: {N} frames, {DURATION}s, delay={delay}")

render("video1.mp4", 0.00, ( 1.0,  0.1))
render("video2.mp4", 0.10, ( 0.9,  0.1))   # surge arrives later
render("video3.mp4", 0.20, ( 0.9,  0.1))   # later still
