import random
from collections import deque

import numpy as np


BG = 0
CANVAS = 5
MARK = 4


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
    Rotate the template stamp according to canvas-edge markers and print it into the canvas.

    Input structure:
    - A black background contains one solid rectangular canvas and one small two-color stamp.
    - The stamp has a unique anchor color in one corner and a main body color elsewhere.
    - Yellow marker pixels sit just outside the canvas edges and indicate print locations.

    Rule:
    - Treat the unrotated stamp as anchor-at-top-left.
    - Top, right, bottom, and left markers request 0, 90, 180, and 270 degree clockwise rotations respectively.
    - Print the rotated stamp just inside that canvas edge, aligned by the marker's row or column center line, while keeping the original stamp and markers.

    Key insight:
    - The edge where the marker sits determines the rotation, and the marker coordinate chooses the aligned print position.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    canvas_cells = max(_components(g == CANVAS), key=len)
    r0, c0, r1, c1 = _bbox(canvas_cells)
    stamp_cells = max(_components((g != BG) & (g != CANVAS) & (g != MARK)), key=len)
    sr0, sc0, sr1, sc1 = _bbox(stamp_cells)
    stamp = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    markers = list(zip(*np.where(g == MARK), strict=False))
    for mr, mc in markers:
        if mr == r0 - 1:
            rot = 0
            piece = np.rot90(stamp, rot)
            rr = r0
            cc = min(max(mc - piece.shape[1] // 2, c0), c1 - piece.shape[1] + 1)
        elif mc == c1 + 1:
            rot = 3
            piece = np.rot90(stamp, rot)
            rr = min(max(mr - piece.shape[0] // 2, r0), r1 - piece.shape[0] + 1)
            cc = c1 - piece.shape[1] + 1
        elif mr == r1 + 1:
            rot = 2
            piece = np.rot90(stamp, rot)
            rr = r1 - piece.shape[0] + 1
            cc = min(max(mc - piece.shape[1] // 2, c0), c1 - piece.shape[1] + 1)
        else:
            rot = 1
            piece = np.rot90(stamp, rot)
            rr = min(max(mr - piece.shape[0] // 2, r0), r1 - piece.shape[0] + 1)
            cc = c0
        mask = piece != BG
        out[rr : rr + piece.shape[0], cc : cc + piece.shape[1]][mask] = piece[mask]
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    ch, cw = random.randint(7, 10), random.randint(7, 10)
    r0 = random.randint(4, h - ch - 4)
    c0 = random.randint(4, w - cw - 4)
    g[r0 : r0 + ch, c0 : c0 + cw] = CANVAS
    body, anchor = random.sample([1, 2, 3, 6, 7, 8, 9], 2)
    stamp = np.array([[anchor, body, 0], [body, body, body], [0, body, body]])
    sr = 1 if r0 > 5 else h - 5
    sc = random.randint(1, w - 4)
    g[sr : sr + 3, sc : sc + 3][stamp != 0] = stamp[stamp != 0]
    for side in random.sample(["top", "right", "bottom", "left"], random.randint(2, 4)):
        if side in ("top", "bottom"):
            mc = random.randint(c0 + 1, c0 + cw - 2)
            mr = r0 - 1 if side == "top" else r0 + ch
        else:
            mr = random.randint(r0 + 1, r0 + ch - 2)
            mc = c0 - 1 if side == "left" else c0 + cw
        g[mr, mc] = MARK
    return g
