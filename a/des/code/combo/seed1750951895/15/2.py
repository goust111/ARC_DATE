import random
from collections import deque

import numpy as np


BG = 5
COLORS = [0, 1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 500


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _ordered_path(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    points = [(int(r), int(c)) for r, c in zip(*np.where(mask))]
    assert len(points) >= 2, "expected a path with at least two cells"
    point_set = set(points)

    seen = {points[0]}
    q = deque([points[0]])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, h, w):
            if (nr, nc) in point_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    assert seen == point_set, "foreground path must be one connected component"

    degree: dict[tuple[int, int], int] = {}
    for r, c in points:
        degree[(r, c)] = sum((nr, nc) in point_set for nr, nc in _neighbors(r, c, h, w))
    endpoints = sorted(cell for cell, deg in degree.items() if deg == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degree.values()), "path must not branch"

    ordered = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while cur != endpoints[1]:
        r, c = cur
        nxt = [
            (nr, nc)
            for nr, nc in _neighbors(r, c, h, w)
            if (nr, nc) in point_set and (nr, nc) != prev
        ]
        assert nxt, "broken path"
        assert len(nxt) == 1 or cur == endpoints[0], "ambiguous path step"
        nxt_cell = nxt[0]
        prev, cur = cur, nxt_cell
        ordered.append(cur)

    assert len(ordered) == len(points)
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the color sequence along the single 4-connected foreground path.

    The grey background and path geometry are unchanged. Only the ordered colors
    encountered from one endpoint to the other are reversed and written back to
    the same ordered path cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    mask = g != BG
    ordered = _ordered_path(mask)
    colors = [int(g[r, c]) for r, c in ordered]

    out = g.copy()
    for (r, c), color in zip(ordered, reversed(colors)):
        out[r, c] = color
    return out


def _build_path(rng: random.Random, h: int, w: int, target_len: int) -> list[tuple[int, int]] | None:
    start = (rng.randrange(1, h - 1), rng.randrange(1, w - 1))
    path = [start]
    occupied = {start}
    for _ in range(target_len * 4):
        if len(path) >= target_len:
            return path
        r, c = path[-1]
        candidates = []
        for nr, nc in _neighbors(r, c, h, w):
            if (nr, nc) in occupied:
                continue
            adjacent = sum((ar, ac) in occupied for ar, ac in _neighbors(nr, nc, h, w))
            if adjacent == 1:
                candidates.append((nr, nc))
        if not candidates:
            return path if len(path) >= max(8, target_len // 2) else None
        # Bias away from immediate dead ends while keeping enough randomness.
        scored = []
        for cell in candidates:
            nr, nc = cell
            onward = 0
            for rr, cc in _neighbors(nr, nc, h, w):
                if (rr, cc) in occupied or (rr, cc) == path[-1]:
                    continue
                onward += 1
            scored.append((onward, rng.random(), cell))
        scored.sort(reverse=True)
        pool = [cell for _, _, cell in scored[: max(1, min(3, len(scored)))]]
        nxt = rng.choice(pool)
        occupied.add(nxt)
        path.append(nxt)
    return path if len(path) >= max(8, target_len // 2) else None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(9, 24)
    w = rng.randint(9, 24)
    target_len = rng.randint(10, min(h * w // 3, 70))
    path = _build_path(rng, h, w, target_len)
    if path is None or len(path) < 8:
        return None

    grid = np.full((h, w), BG, dtype=int)
    palette = rng.sample(COLORS, rng.randint(3, min(7, len(COLORS))))
    colors: list[int] = []
    for i in range(len(path)):
        if i > 0 and rng.random() < 0.35:
            choices = [c for c in palette if c != colors[-1]]
            colors.append(rng.choice(choices))
        else:
            colors.append(rng.choice(palette))

    if colors == list(reversed(colors)):
        colors[0] = next(c for c in COLORS if c != colors[0] and c != colors[-1])

    for (r, c), color in zip(path, colors):
        grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
