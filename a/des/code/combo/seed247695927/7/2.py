import random
from collections import Counter

import numpy as np


PALETTE_ORDER = (5, 2, 8, 9, 6)  # grey, red, teal, maroon, pink
ALL_COLORS = tuple(range(10))
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)
COUNT_RANGE = (0, 6)
MAX_TOTAL_OBJECTS = 15
MAX_PLACEMENT_TRIES = 400
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    counts = [_count_components(g == color) for color in PALETTE_ORDER]
    height = max(counts)
    assert height > 0

    out = np.full((height, len(PALETTE_ORDER)), bg, dtype=int)
    for col, (color, count) in enumerate(zip(PALETTE_ORDER, counts)):
        if count > 0:
            out[height - count :, col] = color
    return out


def _shape_library() -> list[list[tuple[int, int]]]:
    return [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ]


SHAPES = _shape_library()


def _normalize(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _transform_shape(cells: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    pts = list(cells)
    for _ in range(rng.randrange(4)):
        pts = [(-c, r) for r, c in pts]
    if rng.random() < 0.5:
        pts = [(r, -c) for r, c in pts]
    return _normalize(pts)


def _sample_counts(rng: random.Random) -> list[int]:
    while True:
        counts = [rng.randint(*COUNT_RANGE) for _ in PALETTE_ORDER]
        if sum(ct > 0 for ct in counts) == 0:
            continue
        if sum(counts) > MAX_TOTAL_OBJECTS:
            continue
        if max(counts) == 0:
            continue
        return counts


def _can_place(grid: np.ndarray, blocked: np.ndarray, cells: list[tuple[int, int]], r0: int, c0: int) -> bool:
    h, w = grid.shape
    for dr, dc in cells:
        r = r0 + dr
        c = c0 + dc
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], r0: int, c0: int) -> None:
    h, w = blocked.shape
    for dr, dc in cells:
        r = r0 + dr
        c = c0 + dc
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_objects(grid: np.ndarray, color: int, count: int, rng: random.Random, blocked: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _ in range(MAX_PLACEMENT_TRIES):
            cells = _transform_shape(rng.choice(SHAPES), rng)
            shape_h = max(r for r, _ in cells) + 1
            shape_w = max(c for _, c in cells) + 1
            if shape_h > h or shape_w > w:
                continue
            r0 = rng.randint(0, h - shape_h)
            c0 = rng.randint(0, w - shape_w)
            if not _can_place(grid, blocked, cells, r0, c0):
                continue
            for dr, dc in cells:
                grid[r0 + dr, c0 + dc] = color
            _mark_blocked(blocked, cells, r0, c0)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice([c for c in ALL_COLORS if c not in PALETTE_ORDER])
    counts = _sample_counts(rng)

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    color_order = list(zip(PALETTE_ORDER, counts))
    rng.shuffle(color_order)
    for color, count in color_order:
        if not _place_objects(grid, color, count, rng, blocked):
            return None

    if any(_count_components(grid == color) != count for color, count in zip(PALETTE_ORDER, counts)):
        return None
    if np.array_equal(grid, np.full_like(grid, bg)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
