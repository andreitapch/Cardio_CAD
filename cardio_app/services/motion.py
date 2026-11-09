import numpy as np
from matplotlib.colors import hsv_to_rgb


def flow_scale(flow, frames):
    sector = frames.max(axis=0)[::2, ::2] > 0
    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    inside = magnitude[:, sector[:magnitude.shape[1], :magnitude.shape[2]]]
    return float(np.percentile(inside, 50)), max(float(np.percentile(inside, 99)), 1e-3), sector


def flow_rgba(flow_frame, sector, low, top):
    magnitude = np.hypot(flow_frame[..., 0], flow_frame[..., 1])
    hue = (np.arctan2(flow_frame[..., 1], flow_frame[..., 0]) + np.pi) / (2 * np.pi)
    rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1))
    alpha = np.clip((magnitude - low) / max(top - low, 1e-3), 0, 1) * 0.6
    alpha[~sector[:magnitude.shape[0], :magnitude.shape[1]]] = 0
    return (np.dstack([rgb, alpha]) * 255).astype(np.uint8)


def summary(table):
    pairs = table.dropna(subset=["epe_a", "epe_b"])
    lines = []
    for slot in ("a", "b"):
        values = table[f"epe_{slot}"].dropna().to_numpy()
        text = f"{values.mean():.2f} +- {values.std(ddof=1):.2f} mm (n={len(values)})" if len(values) > 1 else "not computed"
        lines.append(f"{slot.upper()}: {text}")
    if len(pairs):
        a, b = pairs["epe_a"].to_numpy(), pairs["epe_b"].to_numpy()
        lines.append(f"B better on {(b < a).sum()}/{len(pairs)} clips, mean B - A {np.mean(b - a):+.2f} mm")
    return lines, pairs
