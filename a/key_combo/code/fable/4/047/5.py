from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
TEAL = 8
SOURCE_COLORS = (1, 3, 4, 5, 6, 7, 9)
GENERATION_ATTEMPTS = 300


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _simulate_race(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, TEAL)]
    assert len(colors) == 2, "expected exactly two source colors"

    out = g.copy()
    frontiers = {
        color: {tuple(map(int, cell)) for cell in np.argwhere(out == color)}
        for color in colors
    }

    while True:
        proposals: dict[tuple[int, int], set[int]] = {}
        for color, frontier in frontiers.items():
            for r, c in frontier:
                for nr, nc in _neighbors4(out.shape, r, c):
                    if int(out[nr, nc]) != BLACK:
                        continue
                    proposals.setdefault((nr, nc), set()).add(int(color))

        if not proposals:
            break

        next_frontiers = {color: set() for color in colors}
        for cell, claimers in proposals.items():
            r, c = cell
            if len(claimers) == 1:
                color = next(iter(claimers))
                out[r, c] = color
                next_frontiers[color].add(cell)
            else:
                out[r, c] = RED
        frontiers = next_frontiers

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let two colored sources race through black corridors while simultaneous arrivals turn into red blocking walls.

    Input structure:
    - Cyan pixels are fixed maze walls and black pixels are open corridors.
    - Exactly two differently colored source points sit on corridor cells.
    - The corridors may contain chambers, side branches, and narrow choke points.

    Rule:
    - Starting from both source colors at once, expand one 4-neighbor step per round into currently black corridor cells.
    - If only one source reaches a black cell in that round, recolor it with that source color.
    - If both sources reach the same black cell in the same round, recolor it red and treat it as a wall from the next round onward.
    - Keep expanding until no black corridor cell can be claimed; cyan walls stay cyan and any corridor cut off by the new red walls stays black.

    Key insight:
    - Territory is determined by dynamic geodesic propagation, not straight-line closeness: tie cells harden into red walls immediately, and those walls can strand deeper corridor cells as untouched black dead zones.
    """
    return _simulate_race(grid)


def _carve_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = BLACK


def _carve_line(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = BLACK
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = BLACK
        return
    raise ValueError("only axis-aligned corridor segments are supported")


def _carve_branch(grid: np.ndarray, attach_r: int, attach_c: int, direction: int, stem: int, cap_width: int) -> None:
    end_r = attach_r + direction * stem
    _carve_line(grid, attach_r, attach_c, end_r, attach_c)
    half = cap_width // 2
    _carve_line(grid, end_r, attach_c - half, end_r, attach_c + half)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_with_margin(grid: np.ndarray, margin: int = 1) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(grid != TEAL)
    r0 = max(0, r0 - margin)
    c0 = max(0, c0 - margin)
    r1 = min(grid.shape[0] - 1, r1 + margin)
    c1 = min(grid.shape[1] - 1, c1 + margin)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _choose_attach_columns(left_source_c: int, room_left: int, right_source_c: int, room_right: int) -> tuple[list[int], list[int]]:
    left_choices = list(range(left_source_c + 1, room_left - 1))
    right_choices = list(range(room_right + 2, right_source_c))
    random.shuffle(left_choices)
    random.shuffle(right_choices)
    return left_choices, right_choices


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((25, 25), TEAL, dtype=int)

    room_h = random.randint(3, 6)
    room_w = random.choice((5, 7, 9))
    approach = random.randint(3, 7)
    bottom_tail = random.randint(2, 5)
    top_tail = random.choice((0, 0, 2, 3, 4))

    room_top = 12 - room_h // 2
    room_left = 12 - room_w // 2
    room_bottom = room_top + room_h - 1
    room_right = room_left + room_w - 1
    door_r = room_top + room_h // 2
    center_c = room_left + room_w // 2
    left_source_c = room_left - approach
    right_source_c = room_right + approach

    if left_source_c < 2 or right_source_c > 22:
        return None

    _carve_rect(grid, room_top, room_left, room_h, room_w)
    _carve_line(grid, door_r, left_source_c, door_r, room_left)
    _carve_line(grid, door_r, room_right, door_r, right_source_c)
    _carve_line(grid, room_bottom + 1, center_c, room_bottom + bottom_tail, center_c)
    if top_tail:
        _carve_line(grid, room_top - top_tail, center_c, room_top - 1, center_c)

    left_choices, right_choices = _choose_attach_columns(left_source_c, room_left, right_source_c, room_right)
    branch_specs = []
    if left_choices and random.random() < 0.9:
        branch_specs.append((door_r, left_choices[0], random.choice((-1, 1)), random.randint(2, 4), random.choice((1, 3))))
    if len(left_choices) > 1 and random.random() < 0.35:
        branch_specs.append((door_r, left_choices[1], random.choice((-1, 1)), random.randint(2, 3), 1))
    if right_choices and random.random() < 0.9:
        branch_specs.append((door_r, right_choices[0], random.choice((-1, 1)), random.randint(2, 4), random.choice((1, 3))))
    if len(right_choices) > 1 and random.random() < 0.35:
        branch_specs.append((door_r, right_choices[1], random.choice((-1, 1)), random.randint(2, 3), 1))

    for attach_r, attach_c, direction, stem, cap_width in branch_specs:
        end_r = attach_r + direction * stem
        half = cap_width // 2
        if not (1 <= end_r < grid.shape[0] - 1):
            return None
        if attach_c - half < 1 or attach_c + half >= grid.shape[1] - 1:
            return None
        _carve_branch(grid, attach_r, attach_c, direction, stem, cap_width)

    source_a, source_b = random.sample(SOURCE_COLORS, 2)
    grid[door_r, left_source_c] = source_a
    grid[door_r, right_source_c] = source_b

    grid = _crop_with_margin(grid, margin=1)
    if random.random() < 0.5:
        grid = np.fliplr(grid)
    if random.random() < 0.5:
        grid = np.flipud(grid)
    grid = np.rot90(grid, k=random.randint(0, 3)).copy()

    out = main(grid)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if np.array_equal(grid, out):
        return None
    if [int(color) for color in np.unique(grid) if int(color) not in (BLACK, TEAL)] != sorted([source_a, source_b]):
        return None
    if int(np.count_nonzero(out == RED)) < 2:
        return None
    if int(np.count_nonzero(out == BLACK)) < max(bottom_tail, 2):
        return None
    if int(np.count_nonzero(out == source_a)) <= 1 or int(np.count_nonzero(out == source_b)) <= 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-race input")
