import random
from collections import deque

import numpy as np


def _components(mask: np.ndarray):
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        pts = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append(pts)
    return parts


def main(grid: np.ndarray) -> np.ndarray:
    """Settle interior pixels toward the side opposite each frame gap."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    for color in sorted(set(map(int, g.ravel())) - {0}):
        comps = _components(g == color)
        for pts in comps:
            rs = [p[0] for p in pts]
            cs = [p[1] for p in pts]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if len(pts) < 2 * (r1 - r0 + c1 - c0):
                continue
            border = set(pts)
            top = sum((r0, c) in border for c in range(c0, c1 + 1))
            bottom = sum((r1, c) in border for c in range(c0, c1 + 1))
            left = sum((r, c0) in border for r in range(r0, r1 + 1))
            right = sum((r, c1) in border for r in range(r0, r1 + 1))
            gap = min([("top", top), ("bottom", bottom), ("left", left), ("right", right)], key=lambda x: x[1])[0]
            inner = out[r0 + 1 : r1, c0 + 1 : c1].copy()
            colors = [(r, c, int(inner[r, c])) for r, c in zip(*np.where(inner != 0)) if int(inner[r, c]) != color]
            inner[:] = 0
            if gap in ("top", "bottom"):
                for c in range(inner.shape[1]):
                    vals = [int(v) for v in inner[:, c] if int(v) not in (0, color)]
                cols = {}
                for rr, cc, v in colors:
                    cols.setdefault(cc, []).append(v)
                for cc, vals in cols.items():
                    vals = vals[:]
                    if gap == "top":
                        for i, v in enumerate(reversed(vals)):
                            inner[inner.shape[0] - 1 - i, cc] = v
                    else:
                        for i, v in enumerate(vals):
                            inner[i, cc] = v
            else:
                rows = {}
                for rr, cc, v in colors:
                    rows.setdefault(rr, []).append(v)
                for rr, vals in rows.items():
                    vals = vals[:]
                    if gap == "left":
                        for i, v in enumerate(reversed(vals)):
                            inner[rr, inner.shape[1] - 1 - i] = v
                    else:
                        for i, v in enumerate(vals):
                            inner[rr, i] = v
            out[r0 + 1 : r1, c0 + 1 : c1] = inner
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    for frame_color in random.sample([1, 2, 3, 4], random.randint(2, 3)):
        for _ in range(100):
            h, w = random.randint(5, 7), random.randint(5, 7)
            r0 = random.randint(1, g.shape[0] - h - 1)
            c0 = random.randint(1, g.shape[1] - w - 1)
            if np.any(g[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] != 0):
                continue
            g[r0 : r0 + h, c0] = frame_color
            g[r0 : r0 + h, c0 + w - 1] = frame_color
            g[r0, c0 : c0 + w] = frame_color
            g[r0 + h - 1, c0 : c0 + w] = frame_color
            side = random.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                start = random.randint(c0 + 1, c0 + w - 3)
                row = r0 if side == "top" else r0 + h - 1
                g[row, start : start + 2] = 0
            else:
                start = random.randint(r0 + 1, r0 + h - 3)
                col = c0 if side == "left" else c0 + w - 1
                g[start : start + 2, col] = 0
            for _ in range(random.randint(3, 6)):
                rr = random.randint(r0 + 1, r0 + h - 2)
                cc = random.randint(c0 + 1, c0 + w - 2)
                if g[rr, cc] == 0:
                    g[rr, cc] = random.choice([5, 6, 7, 8, 9])
            break
    return g
