import random
from collections import deque

import numpy as np


BG = 0


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal each same-color landscape rectangle through the matching window by absolute-coordinate cropping.

    Input structure:
    - A black background contains colored rectangular window frames with black interiors.
    - Outside the frames, each frame color also appears as one solid scenery rectangle.
    - The scenery behind the window has been erased inside the window interior.

    Rule:
    - For each frame, find the scenery rectangle of the same color.
    - Intersect that scenery rectangle with the frame interior using global coordinates.
    - Draw the intersected pixels inside the window, remove the outside scenery, and keep the frame border unchanged.

    Key insight:
    - Matching uses color, but visibility is determined by a crop in absolute coordinates rather than by translating the scenery.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    frames = {}
    for color in [v for v in np.unique(g) if v != BG]:
        for cells in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(cells)
            if len(cells) == 2 * (r1 - r0 + 1) + 2 * (c1 - c0 - 1):
                frames[int(color)] = (r0, c0, r1, c1)
            else:
                for r, c in cells:
                    out[r, c] = BG
                sr0, sc0, sr1, sc1 = _bbox(cells)
                if int(color) in frames:
                    continue
                frames.setdefault(-int(color), (sr0, sc0, sr1, sc1))
    for color, fb in list(frames.items()):
        if color < 0:
            continue
        r0, c0, r1, c1 = fb
        scene = frames.get(-color)
        if scene is None:
            continue
        sr0, sc0, sr1, sc1 = scene
        rr0, rr1 = max(r0 + 1, sr0), min(r1 - 1, sr1)
        cc0, cc1 = max(c0 + 1, sc0), min(c1 - 1, sc1)
        if rr0 <= rr1 and cc0 <= cc1:
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 28), random.randint(18, 28)
    g = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7], random.randint(2, 3)):
        fh, fw = random.randint(5, 8), random.randint(5, 8)
        sh, sw = random.randint(3, 6), random.randint(3, 6)
        for _ in range(200):
            fr, fc = random.randint(0, h - fh), random.randint(0, w - fw)
            sr, sc = random.randint(0, h - sh), random.randint(0, w - sw)
            if np.any(g[max(0, fr - 1) : min(h, fr + fh + 1), max(0, fc - 1) : min(w, fc + fw + 1)]):
                continue
            if np.any(g[max(0, sr - 1) : min(h, sr + sh + 1), max(0, sc - 1) : min(w, sc + sw + 1)]):
                continue
            g[fr : fr + fh, fc] = g[fr : fr + fh, fc + fw - 1] = color
            g[fr, fc : fc + fw] = g[fr + fh - 1, fc : fc + fw] = color
            g[sr : sr + sh, sc : sc + sw] = color
            ir0, ir1 = max(fr + 1, sr), min(fr + fh - 2, sr + sh - 1)
            ic0, ic1 = max(fc + 1, sc), min(fc + fw - 2, sc + sw - 1)
            if ir0 <= ir1 and ic0 <= ic1:
                g[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
            break
    return g
