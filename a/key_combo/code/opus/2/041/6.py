from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

if str(Path('/home/lixinyu/snap/new/solvers')) not in sys.path:
    sys.path.append(str(Path('/home/lixinyu/snap/new/solvers')))

from arc_utils import Color

BLACK = Color.BLACK
RED = Color.RED


def main(grid: np.ndarray) -> np.ndarray:
    """Cross erase on a 2x2 black marker and independently infer four quadrant wallpaper periods.

    Input structure: a full non-black periodic wallpaper split by color across four quadrants, a single 2x2 black marker
    with red pixels on marker row/column.

    Rule: locate the marker, erase its row and column to black while preserving red pixels, split the grid into 4 quadrants,
    infer a 2D periodic tile in each quadrant from non-black cells (including red), and fill remaining black pixels in each
    quadrant by that tile while keeping cross pixels unchanged.

    Key insight: the marker acts as both domain partition and additional holes; red pixels are protected from erasure and remain
    valid clues for period recovery.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects 2D ndarray")

    out = g.copy()
    h, w = g.shape

    marker = _find_black_marker(out)
    if marker is None:
        return out
    r0, c0 = marker
    r1, c1 = r0 + 1, c0 + 1

    mark_rows = {r0, r1}
    mark_cols = {c0, c1}

    # Cross erase, red exempt
    for r in mark_rows:
        for c in range(w):
            if int(out[r, c]) != int(RED):
                out[r, c] = int(BLACK)
    for c in mark_cols:
        for r in range(h):
            if int(out[r, c]) != int(RED):
                out[r, c] = int(BLACK)

    quadrants = [
        (0, 0, r0 - 1, c0 - 1),
        (0, c1 + 1, r0 - 1, w - 1),
        (r1 + 1, 0, h - 1, c0 - 1),
        (r1 + 1, c1 + 1, h - 1, w - 1),
    ]

    for qr0, qc0, qr1, qc1 in quadrants:
        if qr0 > qr1 or qc0 > qc1:
            continue
        region = out[qr0 : qr1 + 1, qc0 : qc1 + 1]
        if region.size == 0:
            continue
        ph, pw, tile = _infer_periodic_tile(region)
        for rr in range(region.shape[0]):
            for cc in range(region.shape[1]):
                if int(region[rr, cc]) != int(BLACK):
                    continue
                out[qr0 + rr, qc0 + cc] = int(tile[rr % ph, cc % pw])

    return out


def _find_black_marker(out: np.ndarray) -> tuple[int, int] | None:
    h, w = out.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            blk = out[r : r + 2, c : c + 2]
            if np.all(blk == int(BLACK)):
                candidates.append((r, c))
    return candidates[0] if len(candidates) == 1 else (candidates[0] if candidates else None)


def _infer_periodic_tile(region: np.ndarray) -> tuple[int, int, np.ndarray]:
    h, w = region.shape
    coords = [(r, c, int(region[r, c])) for r in range(h) for c in range(w) if int(region[r, c]) != int(BLACK)]
    if not coords:
        return 1, 1, np.array([[int(BLACK)]], dtype=int)

    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            tile: dict[tuple[int, int], int] = {}
            ok = True
            for r, c, v in coords:
                k = (r % ph, c % pw)
                if k in tile and tile[k] != v:
                    ok = False
                    break
                tile[k] = v
            if not ok:
                continue
            # fill missing offsets with majority known value for deterministic behavior
            if tile:
                vals = list(tile.values())
                fill = max(set(vals), key=lambda vv: (vals.count(vv), -int(vv)))
                full = np.full((ph, pw), int(fill), dtype=int)
                for (rr, cc), v in tile.items():
                    full[rr, cc] = int(v)
                return ph, pw, full
    _, _, v = coords[0]
    return 1, 1, np.array([[int(v)]], dtype=int)


def generate_input() -> np.ndarray:
    h, w = 13, 13
    g = np.full((h, w), int(BLACK), dtype=int)

    r0 = random.randint(4, 7)
    c0 = random.randint(4, 7)

    colors = [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, Color.MAROON]

    quads = [
        (0, 0, r0 - 1, c0 - 1),
        (0, c0 + 2, r0 - 1, w - 1),
        (r0 + 2, 0, h - 1, c0 - 1),
        (r0 + 2, c0 + 2, h - 1, w - 1),
    ]

    for qr0, qc0, qr1, qc1 in quads:
        ph = random.randint(1, 3)
        pw = random.randint(1, 3)
        tile = np.array(
            [[int(random.choice(colors)) for _ in range(pw)] for __ in range(ph)],
            dtype=int,
        )
        for r in range(qr0, qr1 + 1):
            for c in range(qc0, qc1 + 1):
                if random.random() < 0.16:
                    continue
                g[r, c] = int(tile[(r - qr0) % ph, (c - qc0) % pw])

    # add protected red anchors on marker row/col only
    for c in range(0, w):
        if random.random() < 0.22:
            rr = random.choice([r0, r0 + 1])
            if c not in {c0, c0 + 1}:
                g[rr, c] = int(Color.RED)
    for r in range(0, h):
        if random.random() < 0.22:
            cc = random.choice([c0, c0 + 1])
            if r not in {r0, r0 + 1}:
                g[r, cc] = int(Color.RED)

    # marker
    g[r0, c0] = int(BLACK)
    g[r0, c0 + 1] = int(BLACK)
    g[r0 + 1, c0] = int(BLACK)
    g[r0 + 1, c0 + 1] = int(BLACK)

    return g
