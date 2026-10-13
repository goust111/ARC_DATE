from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

OUTPUT_SIZE = 4
GRID_SIZE_RANGE = (12, 15)
COUNT_PAIRS = [
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (3, 1),
]
COUNT_PAIR_WEIGHTS = [1, 2, 2, 2, 2, 2]
BLUE_DISTRACTOR_RANGE = (3, 7)
GREEN_DISTRACTOR_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_objects(grid: np.ndarray) -> tuple[int, int]:
    full = 0
    damaged = 0

    for comp in _components_mask(grid == GREEN):
        area = int(comp.shape[0])
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        if area == 4 and h == 2 and w == 2:
            if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN):
                full += 1
            continue

        if area == 3 and h == 2 and w == 2:
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if int(np.count_nonzero(sub == GREEN)) == 3 and int(np.count_nonzero(sub == RED)) == 1:
                damaged += 1

    return full, damaged


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count full and damaged green 2x2 blocks and encode the two counts on opposite diagonals.

    Input structure:
    - A square black grid contains isolated green 2x2 full blocks, isolated damaged 2x2 green blocks with one red replacement cell, and blue single-pixel distractors.
    - Full blocks are solid 2x2 green squares.
    - Damaged blocks are 2x2 footprints with exactly three green cells and one red cell.

    Rule:
    - Find 4-connected green components.
    - Count a component as full if it is a solid 2x2 green square.
    - Count a component as damaged if it has three green cells in a 2x2 box and the missing box cell is red.
    - Return a 4x4 black grid whose main diagonal is filled with the first C_full green cells and whose anti-diagonal is filled with the first C_dmg red cells; if both writes target the same cell, keep green.

    Key insight:
    - The red pixel is not a distractor: it marks the missing corner of a damaged green 2x2 block, so the task splits into two separate counts that are merged into one diagonal code.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full, damaged = _count_objects(g)

    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), BLACK, dtype=int)
    for i in range(damaged):
        out[i, OUTPUT_SIZE - 1 - i] = RED
    for i in range(full):
        out[i, i] = GREEN
    return out


def _can_place(occupied: np.ndarray, blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    n = occupied.shape[0]
    if r0 < 0 or c0 < 0 or r0 + h > n or c0 + w > n:
        return False
    if np.any(occupied[r0 : r0 + h, c0 : c0 + w]):
        return False
    if np.any(blocked[r0 : r0 + h, c0 : c0 + w]):
        return False
    return True


def _reserve(occupied: np.ndarray, blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    n = occupied.shape[0]
    occupied[r0 : r0 + h, c0 : c0 + w] = True
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(n, r0 + h + 1)
    cc1 = min(n, c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _candidate_positions(occupied: np.ndarray, blocked: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    n = occupied.shape[0]
    return [
        (r, c)
        for r in range(n - h + 1)
        for c in range(n - w + 1)
        if _can_place(occupied, blocked, r, c, h, w)
    ]


def _place_full_block(grid: np.ndarray, occupied: np.ndarray, blocked: np.ndarray) -> bool:
    options = _candidate_positions(occupied, blocked, 2, 2)
    if not options:
        return False
    r0, c0 = random.choice(options)
    grid[r0 : r0 + 2, c0 : c0 + 2] = GREEN
    _reserve(occupied, blocked, r0, c0, 2, 2)
    return True


def _place_damaged_block(grid: np.ndarray, occupied: np.ndarray, blocked: np.ndarray) -> bool:
    options = _candidate_positions(occupied, blocked, 2, 2)
    if not options:
        return False
    r0, c0 = random.choice(options)
    patch = np.full((2, 2), GREEN, dtype=int)
    miss_r, miss_c = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    patch[miss_r, miss_c] = RED
    grid[r0 : r0 + 2, c0 : c0 + 2] = patch
    _reserve(occupied, blocked, r0, c0, 2, 2)
    return True


def _place_blue_pixel(grid: np.ndarray, occupied: np.ndarray, blocked: np.ndarray) -> bool:
    options = _candidate_positions(occupied, blocked, 1, 1)
    if not options:
        return False
    r0, c0 = random.choice(options)
    grid[r0, c0] = BLUE
    _reserve(occupied, blocked, r0, c0, 1, 1)
    return True


def _place_green_distractor(grid: np.ndarray, occupied: np.ndarray, blocked: np.ndarray) -> bool:
    shapes = [
        np.array([[GREEN]], dtype=int),
        np.array([[GREEN, GREEN]], dtype=int),
        np.array([[GREEN], [GREEN]], dtype=int),
    ]
    random.shuffle(shapes)
    for shape in shapes:
        h, w = shape.shape
        options = _candidate_positions(occupied, blocked, h, w)
        if not options:
            continue
        r0, c0 = random.choice(options)
        grid[r0 : r0 + h, c0 : c0 + w] = shape
        _reserve(occupied, blocked, r0, c0, h, w)
        return True
    return False


def _expected_output(full: int, damaged: int) -> np.ndarray:
    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), BLACK, dtype=int)
    for i in range(damaged):
        out[i, OUTPUT_SIZE - 1 - i] = RED
    for i in range(full):
        out[i, i] = GREEN
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*GRID_SIZE_RANGE)
    full_count, damaged_count = random.choices(COUNT_PAIRS, weights=COUNT_PAIR_WEIGHTS, k=1)[0]

    grid = np.full((n, n), BLACK, dtype=int)
    occupied = np.zeros((n, n), dtype=bool)
    blocked = np.zeros((n, n), dtype=bool)

    placements = ["full"] * full_count + ["damaged"] * damaged_count
    random.shuffle(placements)
    for kind in placements:
        ok = _place_full_block(grid, occupied, blocked) if kind == "full" else _place_damaged_block(grid, occupied, blocked)
        if not ok:
            return None

    blue_count = random.randint(*BLUE_DISTRACTOR_RANGE)
    for _ in range(blue_count):
        if not _place_blue_pixel(grid, occupied, blocked):
            return None

    green_distractors = random.randint(*GREEN_DISTRACTOR_RANGE)
    for _ in range(green_distractors):
        if not _place_green_distractor(grid, occupied, blocked):
            return None

    expected = _expected_output(full_count, damaged_count)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
