import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4


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


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill matching boundary-marker rows, then use each lower barrier color to
    choose how the green sprite above it is reflected into a yellow copy.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    barriers: dict[int, int] = {}
    for r in range(h):
        left, right = int(g[r, 0]), int(g[r, w - 1])
        if left in (BLUE, RED) and left == right:
            out[r, :] = left
            barriers[r] = left

    barrier_rows = sorted(barriers)
    for comp in _components(g == GREEN):
        r0, c0, r1, c1 = _bbox(comp)
        below = [r for r in barrier_rows if r > r1]
        assert below, "each sprite needs a lower matching barrier"
        br = below[0]
        color = barriers[br]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        for r, c in comp:
            if color == RED:
                nr = r
                nc = c1 + 1 + (width - 1 - (c - c0))
            else:
                nr = r1 + 1 + (height - 1 - (r - r0))
                nc = c
            if 0 <= nr < h and 0 <= nc < w and nr not in barriers:
                out[nr, nc] = YELLOW
    return out


SPRITES = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (2, 0)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 1), (1, 1), (2, 1), (1, 0), (1, 2)],
]


def _can_place(grid, cells):
    for r, c in cells:
        if not (0 <= r < grid.shape[0] and 1 <= c < grid.shape[1] - 1):
            return False
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == GREEN and (nr, nc) not in cells:
                return False
    return True


def _gen_attempt():
    h = random.randint(15, 19)
    w = random.randint(14, 18)
    grid = np.zeros((h, w), dtype=int)

    rows = [1]
    while rows[-1] + 4 < h - 2 and len(rows) < 5:
        rows.append(rows[-1] + random.randint(4, 5))
    if len(rows) < 3:
        return None
    colors = [random.choice([BLUE, RED]) for _ in rows]
    if BLUE not in colors or RED not in colors:
        colors[random.randrange(len(colors))] = BLUE
        colors[random.randrange(len(colors))] = RED
    for r, color in zip(rows, colors):
        grid[r, 0] = color
        grid[r, w - 1] = color

    available = [r for r in range(1, h - 1) if r not in rows]
    for r in random.sample(available, random.randint(1, min(3, len(available)))):
        grid[r, 0], grid[r, w - 1] = (BLUE, RED) if random.random() < 0.5 else (RED, BLUE)

    placed = 0
    for upper, lower, lower_color in zip(rows, rows[1:], colors[1:]):
        if random.random() < 0.15:
            continue
        shape = random.choice(SPRITES)
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        if lower - upper - 1 < sh:
            continue
        for _ in range(80):
            if lower_color == RED:
                max_c0 = w - 2 * sw - 2
                if max_c0 < 1:
                    continue
                r0 = random.randint(upper + 1, lower - sh)
                c0 = random.randint(1, max_c0)
            else:
                r0 = random.randint(upper + 1, max(upper + 1, lower - 2 * sh))
                c0 = random.randint(1, w - sw - 2)
            cells = [(r0 + r, c0 + c) for r, c in shape]
            if _can_place(grid, cells):
                for r, c in cells:
                    grid[r, c] = GREEN
                placed += 1
                break

    if placed < 2:
        return None
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
