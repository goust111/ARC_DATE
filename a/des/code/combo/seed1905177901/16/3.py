import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

DIRECTIONS = ("up", "down", "left", "right")
DIR_TO_COLOR = {
    "up": RED,
    "down": BLUE,
    "left": GREEN,
    "right": YELLOW,
}

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
OBJECT_COUNT_RANGE = (1, 4)
OBJECT_H_RANGE = (3, 7)
OBJECT_W_RANGE = (3, 7)
PLACEMENT_RETRIES = 80
GEN_ATTEMPTS = 300

_RNG = random.Random()
_GEN_COUNTER = 0


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_u_direction(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int, int, int]]:
    pts = set(coords)
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 3 and w >= 3, "U-shape bbox must have a non-empty cavity"

    top = sum((r0, c) in pts for c in range(c0, c1 + 1))
    bottom = sum((r1, c) in pts for c in range(c0, c1 + 1))
    left = sum((r, c0) in pts for r in range(r0, r1 + 1))
    right = sum((r, c1) in pts for r in range(r0, r1 + 1))

    candidates: list[str] = []
    if top == 2 and bottom == w and left == h and right == h:
        candidates.append("up")
    if bottom == 2 and top == w and left == h and right == h:
        candidates.append("down")
    if left == 2 and right == h and top == w and bottom == w:
        candidates.append("left")
    if right == 2 and left == h and top == w and bottom == w:
        candidates.append("right")

    assert len(candidates) == 1, "gray component must be a single valid U-shape"
    return candidates[0], (r0, c0, r1, c1)


def _mouth_cells(direction: str, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if direction == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if direction == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _fill_u_interior(
    out: np.ndarray,
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    fill_color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    q = deque()
    seen: set[tuple[int, int]] = set()
    for cell in _mouth_cells(direction, bbox):
        r, c = cell
        if source[r, c] == BG:
            q.append(cell)
            seen.add(cell)

    while q:
        r, c = q.popleft()
        out[r, c] = fill_color
        for nr, nc in neighbors4(r, c):
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in seen or source[nr, nc] != BG:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray U-shape from its opening using the color named by the edge legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    h, w = g.shape

    # Edge legend is fixed by side: top red, bottom blue, left green, right yellow.
    for c in range(w):
        if out[0, c] in (RED, BLUE, GREEN, YELLOW):
            out[0, c] = BG
        if out[h - 1, c] in (RED, BLUE, GREEN, YELLOW):
            out[h - 1, c] = BG
    for r in range(h):
        if out[r, 0] in (RED, BLUE, GREEN, YELLOW):
            out[r, 0] = BG
        if out[r, w - 1] in (RED, BLUE, GREEN, YELLOW):
            out[r, w - 1] = BG

    comps = _components(g == GRAY)
    assert comps, "expected at least one gray U-shape"
    for comp in comps:
        direction, bbox = _detect_u_direction(comp)
        _fill_u_interior(out, g, bbox, direction, DIR_TO_COLOR[direction])
    return out


def _draw_u(grid: np.ndarray, r0: int, c0: int, h: int, w: int, direction: str) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if direction in ("up", "down"):
        grid[r0 : r1 + 1, c0] = GRAY
        grid[r0 : r1 + 1, c1] = GRAY
        grid[r1 if direction == "up" else r0, c0 : c1 + 1] = GRAY
    else:
        grid[r0, c0 : c1 + 1] = GRAY
        grid[r1, c0 : c1 + 1] = GRAY
        grid[r0 : r1 + 1, c1 if direction == "left" else c0] = GRAY


def _fill_slice(
    bbox: tuple[int, int, int, int], direction: str
) -> tuple[slice, slice]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return slice(r0, r1), slice(c0 + 1, c1)
    if direction == "down":
        return slice(r0 + 1, r1 + 1), slice(c0 + 1, c1)
    if direction == "left":
        return slice(r0 + 1, r1), slice(c0, c1)
    return slice(r0 + 1, r1), slice(c0 + 1, c1 + 1)


def _expected_output(
    grid: np.ndarray,
    specs: list[tuple[int, int, int, int, str]],
    markers: dict[str, tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r, c in markers.values():
        out[r, c] = BG
    for r0, c0, h, w, direction in specs:
        rr, cc = _fill_slice((r0, c0, r0 + h - 1, c0 + w - 1), direction)
        out[rr, cc] = DIR_TO_COLOR[direction]
    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h_grid, w_grid = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r1 >= h_grid - 1 or c1 >= w_grid - 1:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h_grid - 1, r1 + 1)
    cc1 = min(w_grid - 1, c1 + 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _place_markers(grid: np.ndarray, rng: random.Random) -> dict[str, tuple[int, int]]:
    h, w = grid.shape
    markers = {
        "up": (0, rng.randint(1, w - 2)),
        "down": (h - 1, rng.randint(1, w - 2)),
        "left": (rng.randint(1, h - 2), 0),
        "right": (rng.randint(1, h - 2), w - 1),
    }
    grid[markers["up"]] = RED
    grid[markers["down"]] = BLUE
    grid[markers["left"]] = GREEN
    grid[markers["right"]] = YELLOW
    return markers


def _gen_attempt(rng: random.Random, required_direction: str) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    max_objs = min(OBJECT_COUNT_RANGE[1], max(1, (h * w) // 45))
    n_objs = rng.randint(OBJECT_COUNT_RANGE[0], max_objs)

    directions = [required_direction]
    while len(directions) < n_objs:
        directions.append(rng.choice(DIRECTIONS))
    rng.shuffle(directions)

    specs: list[tuple[int, int, int, int, str]] = []
    for direction in directions:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            obj_h = rng.randint(*OBJECT_H_RANGE)
            obj_w = rng.randint(*OBJECT_W_RANGE)
            obj_h = min(obj_h, h - 2)
            obj_w = min(obj_w, w - 2)
            if obj_h < 3 or obj_w < 3:
                continue

            r0 = rng.randint(1, h - obj_h - 1)
            c0 = rng.randint(1, w - obj_w - 1)
            if not _can_place(occupied, r0, c0, obj_h, obj_w):
                continue

            _draw_u(grid, r0, c0, obj_h, obj_w, direction)
            occupied[r0 : r0 + obj_h, c0 : c0 + obj_w] = True
            specs.append((r0, c0, obj_h, obj_w, direction))
            placed = True
            break
        if not placed:
            return None

    markers = _place_markers(grid, rng)
    expected = _expected_output(grid, specs, markers)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    required_direction = DIRECTIONS[_GEN_COUNTER % len(DIRECTIONS)]
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG, required_direction)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise AssertionError("failed to generate a valid input")
