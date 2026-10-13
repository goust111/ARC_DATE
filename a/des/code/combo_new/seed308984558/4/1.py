import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5


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
    Fill each grey frame interior with the same-color edge stamp scaled to the interior rectangle.

    Input structure:
    - Grey hollow frames each contain one colored marker pixel.
    - Along one outer edge sit compact same-color stamp shapes.
    - Every frame interior size is an integer scaling of its matching stamp bbox.

    Rule:
    - Match a frame to the stamp sharing its interior marker color.
    - Remove the marker and scale the stamp's occupied cells to the frame interior by block replication.
    - Keep the grey border and any edge stamps unchanged.

    Key insight:
    - Color chooses the source stamp, and the stamp is enlarged by fitting its bbox exactly to the frame interior rather than by translation.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    stamps = {}
    frames = []
    for color in [v for v in np.unique(g) if v not in (BG, FRAME)]:
        comps = _components(g == int(color))
        for cells in comps:
            r0, c0, r1, c1 = _bbox(cells)
            if len(cells) == 1:
                frames.append((int(color), (r0, c0)))
            else:
                stamps[int(color)] = cells
    for cells in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(cells)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marks = [(r0 + 1 + r, c0 + 1 + c, int(inner[r, c])) for r in range(inner.shape[0]) for c in range(inner.shape[1]) if inner[r, c] != BG]
        if not marks:
            continue
        mr, mc, color = marks[0]
        out[mr, mc] = BG
        scells = stamps[color]
        sr0, sc0, sr1, sc1 = _bbox(scells)
        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=int)
        for r, c in scells:
            mask[r - sr0, c - sc0] = 1
        ih, iw = inner.shape
        sh, sw = mask.shape
        sy, sx = ih // sh, iw // sw
        for r in range(sh):
            for c in range(sw):
                if mask[r, c]:
                    out[r0 + 1 + r * sy : r0 + 1 + (r + 1) * sy, c0 + 1 + c * sx : c0 + 1 + (c + 1) * sx] = color
    return out


SHAPES = [np.array([[1, 1], [1, 0]]), np.array([[1, 1, 1], [0, 1, 0]])]


def generate_input() -> np.ndarray:
    h, w = random.randint(20, 28), random.randint(20, 28)
    g = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7], random.randint(2, 3)):
        shape = random.choice(SHAPES)
        sh, sw = shape.shape
        scale = random.randint(2, 3)
        ih, iw = sh * scale, sw * scale
        fh, fw = ih + 2, iw + 2
        for _ in range(200):
            r0 = random.randint(0, h - fh)
            c0 = random.randint(0, w - fw)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + fh + 1), max(0, c0 - 1) : min(w, c0 + fw + 1)]):
                continue
            g[r0 : r0 + fh, c0] = g[r0 : r0 + fh, c0 + fw - 1] = FRAME
            g[r0, c0 : c0 + fw] = g[r0 + fh - 1, c0 : c0 + fw] = FRAME
            g[r0 + fh // 2, c0 + fw // 2] = color
            break
        for _ in range(200):
            sr = random.randint(0, h - sh)
            sc = random.randint(0, w - sw)
            if np.any(g[max(0, sr - 1) : min(h, sr + sh + 1), max(0, sc - 1) : min(w, sc + sw + 1)]):
                continue
            g[sr : sr + sh, sc : sc + sw][shape == 1] = color
            break
    return g
