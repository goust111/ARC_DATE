import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
BLUE = 1
YELLOW = 4
GREY = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _variant(seed, corner, color):
    if corner == "tl":
        return np.fliplr(seed) if color == BLUE else seed
    if corner == "tr":
        return seed if color == BLUE else np.fliplr(seed)
    if corner == "bl":
        return np.rot90(seed, 2) if color == BLUE else np.flipud(seed)
    return np.flipud(seed) if color == BLUE else np.rot90(seed, 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each frame's upper-left seed motif into all quadrants via corner markers.

    Input structure:
    - Grey hollow frames contain an even-sized interior split naturally into quadrants.
    - Only the upper-left interior quadrant contains a yellow seed motif.
    - Outside-diagonal corner markers around each frame are red or blue.

    Rule:
    - Treat the upper-left quadrant as the seed patch.
    - Place transformed copies into all four quadrants according to each corner marker:
      red means the natural symmetry transform for that quadrant, blue means the
      natural transform followed by an extra horizontal flip.
    - Keep the frame and external markers unchanged.

    Key insight:
    - The task combines two codes: quadrant location determines the baseline mirror,
      while red/blue toggles whether that baseline is used directly or horizontally
      flipped once more.
    """
    out = grid.copy()
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[0] % 2 or inner.shape[1] % 2:
            continue
        hh, ww = inner.shape[0] // 2, inner.shape[1] // 2
        seed = (inner[:hh, :ww] == YELLOW).astype(int) * YELLOW
        markers = {
            "tl": int(grid[r0 - 1, c0 - 1]),
            "tr": int(grid[r0 - 1, c1 + 1]),
            "bl": int(grid[r1 + 1, c0 - 1]),
            "br": int(grid[r1 + 1, c1 + 1]),
        }
        inner[:] = BLACK
        inner[:hh, :ww] = _variant(seed, "tl", markers["tl"])
        inner[:hh, ww:] = _variant(seed, "tr", markers["tr"])
        inner[hh:, :ww] = _variant(seed, "bl", markers["bl"])
        inner[hh:, ww:] = _variant(seed, "br", markers["br"])
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    placed = []
    for _ in range(random.randint(2, 3)):
        ih = random.choice([4, 6])
        iw = random.choice([4, 6])
        fh, fw = ih + 2, iw + 2
        for _ in range(100):
            r0 = random.randint(1, h - fh - 2)
            c0 = random.randint(1, w - fw - 2)
            if all(r0 + fh + 2 < pr0 or pr1 + 2 < r0 or c0 + fw + 2 < pc0 or pc1 + 2 < c0 for pr0, pc0, pr1, pc1 in placed):
                placed.append((r0, c0, r0 + fh - 1, c0 + fw - 1))
                break
        grid[r0, c0 : c0 + fw] = GREY
        grid[r0 + fh - 1, c0 : c0 + fw] = GREY
        grid[r0 : r0 + fh, c0] = GREY
        grid[r0 : r0 + fh, c0 + fw - 1] = GREY
        hh, ww = ih // 2, iw // 2
        seed = np.zeros((hh, ww), dtype=int)
        for _ in range(random.randint(2, max(2, hh * ww // 2))):
            seed[random.randint(0, hh - 1), random.randint(0, ww - 1)] = YELLOW
        grid[r0 + 1 : r0 + 1 + hh, c0 + 1 : c0 + 1 + ww] = seed
        for rr, cc in [(r0 - 1, c0 - 1), (r0 - 1, c0 + fw), (r0 + fh, c0 - 1), (r0 + fh, c0 + fw)]:
            grid[rr, cc] = random.choice([RED, BLUE])
    return grid
