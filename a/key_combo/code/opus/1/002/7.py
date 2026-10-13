import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
CYAN = 8


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _draw_clipped(out, tile, r0, c0):
    th, tw = tile.shape
    for r in range(th):
        for c in range(tw):
            rr, cc = r0 + r, c0 + c
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and tile[r, c] != BG:
                out[rr, cc] = tile[r, c]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing diagonal square blocks to form a 2K tile, then repeat
    the tile using the red and blue control-bar gaps as horizontal and vertical
    periods.
    """
    g = np.asarray(grid, dtype=int)
    green_comps = _components(g == GREEN)
    assert len(green_comps) == 2, "expected two green square blocks"
    boxes = [_bbox(comp) for comp in green_comps]
    sizes = [b[2] - b[0] + 1 for b in boxes]
    assert sizes[0] == sizes[1] and all(b[3] - b[1] + 1 == sizes[0] for b in boxes)
    k = sizes[0]
    r0 = min(b[0] for b in boxes)
    c0 = min(b[1] for b in boxes)
    r1 = max(b[2] for b in boxes)
    c1 = max(b[3] for b in boxes)
    assert r1 - r0 + 1 == 2 * k and c1 - c0 + 1 == 2 * k

    tile = np.zeros((2 * k, 2 * k), dtype=int)
    for br0, bc0, _, _ in boxes:
        qr = 0 if br0 == r0 else 1
        qc = 0 if bc0 == c0 else 1
        tile[qr * k : (qr + 1) * k, qc * k : (qc + 1) * k] = GREEN
    for qr in (0, 1):
        for qc in (0, 1):
            if tile[qr * k, qc * k] == BG:
                tile[qr * k : (qr + 1) * k, qc * k : (qc + 1) * k] = CYAN

    red = list(zip(*np.where(g == RED)))
    blue = list(zip(*np.where(g == BLUE)))
    assert red and blue, "missing control bars"
    red_c0 = min(int(c) for _, c in red)
    blue_r0 = min(int(r) for r, _ in blue)
    h_gap = red_c0 - c1 - 1
    v_gap = blue_r0 - r1 - 1
    assert h_gap >= 1 and v_gap >= 1
    h_period = 2 * k + h_gap
    v_period = 2 * k + v_gap

    out = np.zeros_like(g)
    rr = r0
    while rr < out.shape[0]:
        cc = c0
        while cc < out.shape[1]:
            _draw_clipped(out, tile, rr, cc)
            cc += h_period
        rr += v_period
    return out


def _gen_attempt():
    k = random.choice([2, 3])
    h_gap = random.randint(1, 4)
    v_gap = random.randint(1, 4)
    tile_size = 2 * k
    min_w = tile_size + 2 * (tile_size + h_gap)
    min_h = tile_size + (tile_size + v_gap)
    h = random.randint(min_h + 2, min(30, min_h + 7))
    w = random.randint(min_w + 2, min(30, min_w + 7))
    r0 = random.randint(1, h - min_h - 1)
    c0 = random.randint(1, w - min_w - 1)
    grid = np.zeros((h, w), dtype=int)
    diag_main = random.random() < 0.5
    blocks = [(0, 0), (1, 1)] if diag_main else [(0, 1), (1, 0)]
    for qr, qc in blocks:
        grid[r0 + qr * k : r0 + (qr + 1) * k, c0 + qc * k : c0 + (qc + 1) * k] = GREEN

    obj_r1 = r0 + tile_size - 1
    obj_c1 = c0 + tile_size - 1
    red_c = obj_c1 + h_gap + 1
    red_r = random.randint(r0, obj_r1 - k + 1)
    grid[red_r : red_r + k, red_c] = RED
    blue_r = obj_r1 + v_gap + 1
    blue_c = random.randint(c0, obj_c1 - k + 1)
    grid[blue_r, blue_c : blue_c + k] = BLUE
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
