import random
from collections import deque

import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _border(r0, c0, r1, c1):
    for c in range(c0, c1 + 1):
        yield r0, c
        yield r1, c
    for r in range(r0 + 1, r1):
        yield r, c0
        yield r, c1


def _frames(g):
    result = []
    for color in [int(x) for x in np.unique(g) if int(x) != BG]:
        for pts in _components(g == color):
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if set(pts) == set(_border(r0, c0, r1, c1)):
                result.append({"color": color, "bbox": (r0, c0, r1, c1), "area": (r1 - r0 + 1) * (c1 - c0 + 1)})
    result.sort(key=lambda x: -x["area"])
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor even nested frames and copy the inner motif into even-layer gaps.

    Input structure:
    - Black background with concentric one-pixel rectangular frames of distinct colors.
    - Frames are separated by black gaps.
    - The innermost frame contains one small colored motif.

    Rule:
    - Number frames from outside to inside.
    - Odd frames keep their color; even frames are recolored to the original color of the preceding odd frame.
    - Copy the inner motif into each gap just inside an even frame, using that even frame's original color.

    Key insight:
    - Nesting depth controls both frame recoloring and where the inner motif is echoed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _frames(g)
    frame_colors = {f["color"] for f in frames}
    for idx, frame in enumerate(frames):
        if idx % 2 == 1:
            for r, c in _border(*frame["bbox"]):
                out[r, c] = frames[idx - 1]["color"]
    if not frames:
        return out
    inner = frames[-1]["bbox"]
    motif_mask = np.zeros_like(g, dtype=bool)
    for r in range(inner[0] + 1, inner[2]):
        for c in range(inner[1] + 1, inner[3]):
            if int(g[r, c]) not in frame_colors and int(g[r, c]) != BG:
                motif_mask[r, c] = True
    pts = np.argwhere(motif_mask)
    if pts.size == 0:
        return out
    pr0, pc0 = pts.min(axis=0)
    pr1, pc1 = pts.max(axis=0)
    rel = [(int(r - pr0), int(c - pc0)) for r, c in pts]
    ph, pw = int(pr1 - pr0 + 1), int(pc1 - pc0 + 1)
    for idx in range(1, len(frames) - 1, 2):
        even = frames[idx]
        next_inner = frames[idx + 1]["bbox"]
        r0, c0, r1, c1 = even["bbox"]
        band_r0, band_r1 = r0 + 1, next_inner[0] - 1
        band_c0, band_c1 = c0 + 1, c1 - 1
        if band_r0 > band_r1:
            continue
        tr = (band_r0 + band_r1 - ph + 1) // 2
        tc = (band_c0 + band_c1 - pw + 1) // 2
        for rr, cc in rel:
            r, c = tr + rr, tc + cc
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and out[r, c] == BG:
                out[r, c] = even["color"]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    h = w = rng.randint(19, 29)
    g = np.zeros((h, w), dtype=int)
    layers = rng.randint(3, 5)
    if layers % 2 == 0:
        layers += 1
    margin = 1
    for color in rng.sample(colors, layers):
        r0 = c0 = margin
        r1 = h - margin - 1
        c1 = w - margin - 1
        for r, c in _border(r0, c0, r1, c1):
            g[r, c] = color
        margin += rng.randint(2, 3)
        if h - 2 * margin < 5:
            break
    inners = _frames(g)
    inner = inners[-1]["bbox"]
    motif_color = rng.choice([c for c in colors if c not in {f["color"] for f in inners}])
    cr = (inner[0] + inner[2]) // 2
    cc = (inner[1] + inner[3]) // 2
    for dr, dc in ((0, 0), (1, 0), (0, 1), (-1, 0)):
        if inner[0] < cr + dr < inner[2] and inner[1] < cc + dc < inner[3]:
            g[cr + dr, cc + dc] = motif_color
    return g
