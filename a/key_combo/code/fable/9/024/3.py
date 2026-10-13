from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

HEIGHT_RANGE = (10, 18)
WIDTH_RANGE = (10, 18)
BLUE_COUNT_RANGE = (8, 16)
ANCHOR_MARGIN = 2
MIN_ANCHOR_DISTANCE = 6
MAX_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _parity(cell: tuple[int, int]) -> int:
    return (cell[0] + cell[1]) % 2


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _chebyshev_clear(cell: tuple[int, int], occupied: set[tuple[int, int]], gap: int = 1) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > gap for rr, cc in occupied)


def _nonblack_isolated(grid: np.ndarray) -> bool:
    for r, c in np.argwhere(grid != BLACK):
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BLACK:
                return False
    return True


def _parity_output(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    red_anchor = np.argwhere(out == RED)
    green_anchor = np.argwhere(out == GREEN)
    assert red_anchor.shape == (1, 2)
    assert green_anchor.shape == (1, 2)

    red_anchor_cell = (int(red_anchor[0, 0]), int(red_anchor[0, 1]))
    green_anchor_cell = (int(green_anchor[0, 0]), int(green_anchor[0, 1]))
    red_phase = _parity(red_anchor_cell)
    assert _parity(green_anchor_cell) != red_phase

    for r, c in np.argwhere(out == BLUE):
        out[int(r), int(c)] = RED if _parity((int(r), int(c))) == red_phase else GREEN
    return out


def _nearest_anchor_output(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    red_anchor = tuple(map(int, np.argwhere(out == RED)[0]))
    green_anchor = tuple(map(int, np.argwhere(out == GREEN)[0]))
    for r, c in np.argwhere(out == BLUE):
        cell = (int(r), int(c))
        d_red = _manhattan(cell, red_anchor)
        d_green = _manhattan(cell, green_anchor)
        assert d_red != d_green
        out[cell] = RED if d_red < d_green else GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each blue singleton by the invisible global checkerboard phase named by the two anchors.

    Input structure:
    - The background is black.
    - There are many isolated blue single pixels plus exactly one red anchor pixel and one green anchor pixel.
    - The two anchors lie on opposite row-plus-column parity phases of the same unseen global checkerboard.

    Rule:
    - Keep the two anchors and the black background unchanged.
    - Use global row-plus-column parity to split the whole grid into two checkerboard phases.
    - Recolor every blue pixel to red if it has the same parity as the red anchor; otherwise recolor it to green.

    Key insight:
    - The anchors are not distance sources; they only reveal which global parity phase should become red and which should become green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(np.unique(g)).issubset({BLACK, BLUE, RED, GREEN}), "unexpected colors present"

    red_anchor = np.argwhere(g == RED)
    green_anchor = np.argwhere(g == GREEN)
    blue_coords = np.argwhere(g == BLUE)
    assert red_anchor.shape == (1, 2), "expected exactly one red anchor"
    assert green_anchor.shape == (1, 2), "expected exactly one green anchor"
    assert blue_coords.shape[0] > 0, "expected at least one blue pixel"
    assert _nonblack_isolated(g), "all colored pixels must be isolated 4-connected singletons"

    red_anchor_cell = (int(red_anchor[0, 0]), int(red_anchor[0, 1]))
    green_anchor_cell = (int(green_anchor[0, 0]), int(green_anchor[0, 1]))
    red_phase = _parity(red_anchor_cell)
    assert _parity(green_anchor_cell) != red_phase, "anchor phases must be opposite"

    out = g.copy()
    for r, c in blue_coords:
        out[int(r), int(c)] = RED if _parity((int(r), int(c))) == red_phase else GREEN
    return out


def _anchor_candidates(height: int, width: int, parity: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(ANCHOR_MARGIN, height - ANCHOR_MARGIN):
        for c in range(ANCHOR_MARGIN, width - ANCHOR_MARGIN):
            if (r + c) % 2 == parity:
                cells.append((r, c))
    return cells


def _candidate_groups(
    height: int, width: int, red_anchor: tuple[int, int], green_anchor: tuple[int, int]
) -> dict[str, list[tuple[int, int]]]:
    red_phase = _parity(red_anchor)
    occupied = {red_anchor, green_anchor}
    groups = {"red_agree": [], "red_mislead": [], "green_agree": [], "green_mislead": []}

    for r in range(height):
        for c in range(width):
            cell = (r, c)
            if cell in occupied or not _chebyshev_clear(cell, occupied):
                continue

            d_red = _manhattan(cell, red_anchor)
            d_green = _manhattan(cell, green_anchor)
            if d_red == d_green:
                continue

            if _parity(cell) == red_phase:
                key = "red_agree" if d_red < d_green else "red_mislead"
            else:
                key = "green_agree" if d_green < d_red else "green_mislead"
            groups[key].append(cell)
    return groups


def _choose_seed_cells(groups: dict[str, list[tuple[int, int]]], occupied: set[tuple[int, int]]):
    keys = ["red_agree", "green_agree", "red_mislead", "green_mislead"]
    candidate_lists = {}
    for key in keys:
        cells = groups[key][:]
        random.shuffle(cells)
        candidate_lists[key] = cells

    chosen: list[tuple[int, int]] = []

    def search(i: int) -> bool:
        if i == len(keys):
            return True
        for cell in candidate_lists[keys[i]]:
            if _chebyshev_clear(cell, occupied | set(chosen)):
                chosen.append(cell)
                if search(i + 1):
                    return True
                chosen.pop()
        return False

    return chosen if search(0) else None


def _greedy_extra_cells(
    height: int, width: int, occupied: set[tuple[int, int]], target_count: int
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if cell in occupied:
            continue
        if _chebyshev_clear(cell, occupied | set(chosen)):
            chosen.append(cell)
            if len(chosen) == target_count:
                return chosen
    return None


def _build_input(
    height: int,
    width: int,
    red_anchor: tuple[int, int],
    green_anchor: tuple[int, int],
    blue_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[red_anchor] = RED
    grid[green_anchor] = GREEN
    for cell in blue_cells:
        grid[cell] = BLUE
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(np.unique(grid)).issubset({BLACK, BLUE, RED, GREEN}):
        return False
    if not _nonblack_isolated(grid):
        return False

    red_anchor = np.argwhere(grid == RED)
    green_anchor = np.argwhere(grid == GREEN)
    blue_coords = np.argwhere(grid == BLUE)
    if red_anchor.shape != (1, 2) or green_anchor.shape != (1, 2) or blue_coords.shape[0] < 4:
        return False

    red_anchor_cell = (int(red_anchor[0, 0]), int(red_anchor[0, 1]))
    green_anchor_cell = (int(green_anchor[0, 0]), int(green_anchor[0, 1]))
    if _parity(red_anchor_cell) == _parity(green_anchor_cell):
        return False

    red_phase = _parity(red_anchor_cell)
    has_red_agree = has_red_mislead = False
    has_green_agree = has_green_mislead = False
    red_phase_count = 0
    green_phase_count = 0
    for r, c in blue_coords:
        cell = (int(r), int(c))
        d_red = _manhattan(cell, red_anchor_cell)
        d_green = _manhattan(cell, green_anchor_cell)
        if d_red == d_green:
            return False
        if _parity(cell) == red_phase:
            red_phase_count += 1
            has_red_agree |= d_red < d_green
            has_red_mislead |= d_green < d_red
        else:
            green_phase_count += 1
            has_green_agree |= d_green < d_red
            has_green_mislead |= d_red < d_green

    if min(red_phase_count, green_phase_count) < 2:
        return False
    if not (has_red_agree and has_red_mislead and has_green_agree and has_green_mislead):
        return False

    expected = _parity_output(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(expected, _nearest_anchor_output(grid)):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    red_phase = random.randint(0, 1)

    red_candidates = _anchor_candidates(height, width, red_phase)
    green_candidates = _anchor_candidates(height, width, 1 - red_phase)
    random.shuffle(red_candidates)
    random.shuffle(green_candidates)

    for red_anchor in red_candidates:
        for green_anchor in green_candidates:
            if _manhattan(red_anchor, green_anchor) < MIN_ANCHOR_DISTANCE:
                continue
            if not _chebyshev_clear(green_anchor, {red_anchor}):
                continue

            groups = _candidate_groups(height, width, red_anchor, green_anchor)
            if any(not groups[key] for key in groups):
                continue

            occupied = {red_anchor, green_anchor}
            seed_cells = _choose_seed_cells(groups, occupied)
            if seed_cells is None:
                continue
            occupied.update(seed_cells)

            target_blue_count = random.randint(*BLUE_COUNT_RANGE)
            extra_count = target_blue_count - len(seed_cells)
            extras = _greedy_extra_cells(height, width, occupied, extra_count)
            if extras is None:
                continue

            grid = _build_input(height, width, red_anchor, green_anchor, seed_cells + extras)
            if _validate_instance(grid):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-checkerboard sample")
