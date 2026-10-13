import random
from collections import Counter

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _frame_bbox(g, color):
    coords = list(zip(*np.where(g == color)))
    assert coords, "missing frame color"
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assign colors by frequency: most frequent completes the frame, least
    frequent counts the band width, and the middle color fills alternating
    concentric bands inside the completed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    counts = Counter(int(x) for x in g.ravel() if int(x) != BG)
    assert len(counts) == 3, "expected exactly three non-background colors"
    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    param_color, p = ordered[0]
    fill_color, _ = ordered[1]
    frame_color, _ = ordered[2]
    assert len({v for _, v in ordered}) == 3 and p >= 1
    _ = param_color

    r0, c0, r1, c1 = _frame_bbox(g, frame_color)
    out = np.zeros_like(g)
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            depth = min(r - r0 - 1, c - c0 - 1, r1 - 1 - r, c1 - 1 - c)
            if (depth // p) % 2 == 0:
                out[r, c] = fill_color
    return out


def _draw_frame(grid, box, color, gaps):
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    candidates = []
    for c in range(c0 + 2, c1 - 1):
        candidates.append((r0, c))
        candidates.append((r1, c))
    for r in range(r0 + 2, r1 - 1):
        candidates.append((r, c0))
        candidates.append((r, c1))
    for r, c in random.sample(candidates, gaps):
        grid[r, c] = BG


def _isolated(grid, r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != BG:
                return False
    return True


def _gen_attempt():
    colors = random.sample(COLORS, 3)
    frame_color, fill_color, param_color = colors
    p = random.randint(1, 3)
    min_side = max(7, 4 * p + 2)
    h = random.randint(min_side + 2, 20)
    w = random.randint(min_side + 2, 20)
    grid = np.zeros((h, w), dtype=int)
    fh = random.randint(min_side, min(h - 2, min_side + 4))
    fw = random.randint(min_side, min(w - 2, min_side + 4))
    r0 = random.randint(1, h - fh - 1)
    c0 = random.randint(1, w - fw - 1)
    box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
    _draw_frame(grid, box, frame_color, random.randint(1, 2))

    inner = [(r, c) for r in range(r0 + 1, r0 + fh - 1) for c in range(c0 + 1, c0 + fw - 1)]
    random.shuffle(inner)
    placed = 0
    corner_targets = sorted(inner, key=lambda rc: abs(rc[0] - (r0 + 1)) + abs(rc[1] - (c0 + 1)))
    for r, c in corner_targets:
        if grid[r, c] == BG and _isolated(grid, r, c):
            grid[r, c] = param_color
            placed += 1
            if placed == p:
                break
    if placed != p:
        return None

    mid_count = random.randint(p + 1, p + 5)
    for r, c in inner:
        if mid_count == 0:
            break
        if grid[r, c] == BG:
            grid[r, c] = fill_color
            mid_count -= 1
    if mid_count:
        return None

    counts = Counter(int(x) for x in grid.ravel() if int(x) != BG)
    if not (counts[frame_color] > counts[fill_color] > counts[param_color]):
        return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
