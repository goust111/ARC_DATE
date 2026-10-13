import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 8, 9)

GRID_H_RANGE = (12, 25)
GRID_W_RANGE = (12, 25)
OBJECT_COUNT_RANGE = (1, 4)
RECT_H_RANGE = (3, 9)
RECT_W_RANGE = (3, 9)
PLACEMENT_GAP = 1
MAX_GENERATE_ATTEMPTS = 300

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_objects(grid: np.ndarray) -> list[tuple[int, int, int, int, tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    objects = []
    for comp in _components(g != BG):
        r0, c0, r1, c1 = _bbox(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert sub.size == len(comp), "foreground object must be a solid rectangle"

        seed_cells = np.argwhere((sub != BG) & (sub != GREY))
        assert seed_cells.shape[0] == 1, "each rectangle must contain exactly one seed"
        sr, sc = map(int, seed_cells[0])
        seed_color = int(sub[sr, sc])

        grey_or_seed = (sub == GREY) | (sub == seed_color)
        assert np.all(grey_or_seed), "rectangle may only contain grey plus its seed color"
        objects.append((r0, c0, r1, c1, (r0 + sr, c0 + sc), seed_color))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1, (sr, sc), seed_color in _find_objects(g):
        dist = min(sr - r0, r1 - sr, sc - c0, c1 - sc)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                ring = max(abs(r - sr), abs(c - sc))
                if 1 <= ring <= dist:
                    out[r, c] = seed_color if ring % 2 == 1 else GREY
    return out


def _clear_with_gap(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h0, w0 = occupied.shape
    rr0 = max(0, r0 - PLACEMENT_GAP)
    cc0 = max(0, c0 - PLACEMENT_GAP)
    rr1 = min(h0, r0 + h + PLACEMENT_GAP)
    cc1 = min(w0, c0 + w + PLACEMENT_GAP)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _candidate_seed_positions(h: int, w: int) -> list[tuple[int, int]]:
    positions = []
    for r in range(h):
        for c in range(w):
            dist = min(r, h - 1 - r, c, w - 1 - c)
            positions.append((r, c, dist))
    random.shuffle(positions)
    positions.sort(key=lambda item: (item[2], random.random()))
    return [(r, c) for r, c, _ in positions]


def _place_rectangles() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    target_count = random.randint(*OBJECT_COUNT_RANGE)
    objects = []
    attempts = 0
    max_attempts = 200

    want_positive = random.random() < 0.8
    have_positive = False

    while len(objects) < target_count and attempts < max_attempts:
        attempts += 1
        rh = random.randint(*RECT_H_RANGE)
        rw = random.randint(*RECT_W_RANGE)
        if rh > h or rw > w:
            continue
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        if not _clear_with_gap(occupied, r0, c0, rh, rw):
            continue

        positions = _candidate_seed_positions(rh, rw)
        if want_positive and len(objects) == target_count - 1 and not have_positive:
            positions = [p for p in positions if min(p[0], rh - 1 - p[0], p[1], rw - 1 - p[1]) >= 1]
            if not positions:
                continue

        sr, sc = random.choice(positions)
        seed_dist = min(sr, rh - 1 - sr, sc, rw - 1 - sc)
        have_positive = have_positive or seed_dist >= 1
        seed_color = random.choice(SEED_COLORS)

        grid[r0 : r0 + rh, c0 : c0 + rw] = GREY
        grid[r0 + sr, c0 + sc] = seed_color
        occupied[r0 : r0 + rh, c0 : c0 + rw] = True
        objects.append((r0, c0, rh, rw, sr, sc))

    if len(objects) != target_count:
        return None
    if want_positive and not have_positive:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _place_rectangles()
        if grid is None:
            continue
        objects = _find_objects(grid)
        if not objects:
            continue
        out = main(grid)
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
