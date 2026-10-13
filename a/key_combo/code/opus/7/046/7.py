from __future__ import annotations

import random
import sys

import numpy as np


ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import Color, in_bounds


BLACK = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
ORANGE = int(Color.ORANGE)
TEAL = int(Color.TEAL)

GRID_MIN = 10
GRID_MAX = 14
MARKER_COUNT_MIN = 3
MARKER_COUNT_MAX = 5
GENERATION_RETRIES = 400

ROWCOL_LAYOUTS = (
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
)

DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONALS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _expand_markers(grid: np.ndarray) -> np.ndarray:
    expanded = np.asarray(grid, dtype=int).copy()

    for r, c in np.argwhere(grid == RED):
        for dr, dc in DIAGONALS:
            rr = int(r + dr)
            cc = int(c + dc)
            if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BLACK:
                expanded[rr, cc] = YELLOW

    for r, c in np.argwhere(grid == BLUE):
        for dr, dc in ORTHOGONALS:
            rr = int(r + dr)
            cc = int(c + dc)
            if (
                in_bounds(grid.shape, rr, cc)
                and int(grid[rr, cc]) == BLACK
                and int(expanded[rr, cc]) == BLACK
            ):
                expanded[rr, cc] = ORANGE

    return expanded


def _open_row_corridors(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    rows: list[int] = []
    for r in range(h):
        if int(grid[r, 0]) == TEAL and int(grid[r, w - 1]) == TEAL and np.all(grid[r, 1 : w - 1] == BLACK):
            rows.append(int(r))
    return rows


def _open_col_corridors(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    cols: list[int] = []
    for c in range(w):
        if int(grid[0, c]) == TEAL and int(grid[h - 1, c]) == TEAL and np.all(grid[1 : h - 1, c] == BLACK):
            cols.append(int(c))
    return cols


def _corridor_hits(
    targets: list[tuple[int, int]], row_corridors: set[int], col_corridors: set[int]
) -> tuple[set[int], set[int]]:
    hit_rows = {int(r) for r, _c in targets if int(r) in row_corridors}
    hit_cols = {int(c) for _r, c in targets if int(c) in col_corridors}
    return hit_rows, hit_cols


def _targets_for_marker(shape: tuple[int, int], marker: tuple[int, int], color: int) -> list[tuple[int, int]]:
    deltas = DIAGONALS if color == RED else ORTHOGONALS
    r, c = marker
    targets: list[tuple[int, int]] = []
    for dr, dc in deltas:
        rr = int(r + dr)
        cc = int(c + dc)
        if in_bounds(shape, rr, cc):
            targets.append((rr, cc))
    return targets


def _source_markers_for_target(target: tuple[int, int], color: int) -> list[tuple[int, int]]:
    deltas = DIAGONALS if color == RED else ORTHOGONALS
    tr, tc = target
    return [(int(tr - dr), int(tc - dc)) for dr, dc in deltas]


def _blocking_candidates(
    grid: np.ndarray,
    color: int,
    kind: str,
    value: int,
    row_corridors: set[int],
    col_corridors: set[int],
    reserved_targets: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], set[tuple[int, int]]]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []

    if kind == "row":
        target_cells = [(value, c) for c in range(1, w - 1)]
        random.shuffle(target_cells)
    else:
        target_cells = [(r, value) for r in range(1, h - 1)]
        random.shuffle(target_cells)

    for target in target_cells:
        tr, tc = target
        if kind == "row" and tc in col_corridors:
            continue
        if kind == "col" and tr in row_corridors:
            continue

        marker_sources = _source_markers_for_target(target, color)
        random.shuffle(marker_sources)
        for mr, mc in marker_sources:
            if not (1 <= mr < h - 1 and 1 <= mc < w - 1):
                continue
            if mr in row_corridors or mc in col_corridors:
                continue
            if int(grid[mr, mc]) != BLACK:
                continue

            targets = _targets_for_marker(grid.shape, (mr, mc), color)
            visible_targets = {(rr, cc) for rr, cc in targets if int(grid[rr, cc]) == BLACK}
            if not visible_targets or visible_targets & reserved_targets:
                continue

            hit_rows, hit_cols = _corridor_hits(targets, row_corridors, col_corridors)
            if kind == "row":
                if hit_rows != {value} or hit_cols:
                    continue
            else:
                if hit_cols != {value} or hit_rows:
                    continue

            candidates.append(((mr, mc), visible_targets))
    return candidates


def _nonblocking_candidates(
    grid: np.ndarray,
    color: int,
    row_corridors: set[int],
    col_corridors: set[int],
    reserved_targets: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], set[tuple[int, int]]]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
    marker_cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(marker_cells)

    for mr, mc in marker_cells:
        if mr in row_corridors or mc in col_corridors:
            continue
        if int(grid[mr, mc]) != BLACK:
            continue

        targets = _targets_for_marker(grid.shape, (mr, mc), color)
        visible_targets = {(rr, cc) for rr, cc in targets if int(grid[rr, cc]) == BLACK}
        if not visible_targets or visible_targets & reserved_targets:
            continue

        hit_rows, hit_cols = _corridor_hits(targets, row_corridors, col_corridors)
        if hit_rows or hit_cols:
            continue

        candidates.append(((mr, mc), visible_targets))
    return candidates


def _place_marker(
    grid: np.ndarray,
    marker: tuple[int, int],
    color: int,
    visible_targets: set[tuple[int, int]],
    reserved_targets: set[tuple[int, int]],
) -> None:
    r, c = marker
    grid[r, c] = color
    reserved_targets.update(visible_targets)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand red and blue markers, then recolor only the still-unobstructed teal-to-teal corridors green.

    Input structure:
    - The grid has a black background with teal pixels marking selected opposite-edge row and column endpoints.
    - The interior contains isolated red and blue marker pixels.
    - No green is present in the input; yellow and orange are reserved for expansion results.

    Rule:
    - First, each red marker paints its in-bounds black diagonal neighbors yellow, and each blue marker paints its in-bounds black orthogonal neighbors orange.
    - After that expansion step, any row with teal cells at both outer ends and an all-black interior is a valid horizontal corridor, so its interior is recolored green.
    - Likewise, any column with teal cells at both outer ends and an all-black interior is a valid vertical corridor, and its interior is recolored green.

    Key insight:
    - Corridor validity is determined only after marker expansion, because the newly created yellow and orange pixels can splash into an otherwise clear path and block it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    expanded = _expand_markers(g)
    out = expanded.copy()
    h, w = g.shape

    for r in _open_row_corridors(expanded):
        out[r, 1 : w - 1] = GREEN

    for c in _open_col_corridors(expanded):
        out[1 : h - 1, c] = GREEN

    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    row_count, col_count = random.choice(ROWCOL_LAYOUTS)

    if row_count >= h - 2 or col_count >= w - 2:
        return None

    row_corridors = sorted(random.sample(range(1, h - 1), row_count))
    col_corridors = sorted(random.sample(range(1, w - 1), col_count))
    open_row = random.choice(row_corridors)
    open_col = random.choice(col_corridors)

    grid = np.zeros((h, w), dtype=int)
    for r in row_corridors:
        grid[r, 0] = TEAL
        grid[r, w - 1] = TEAL
    for c in col_corridors:
        grid[0, c] = TEAL
        grid[h - 1, c] = TEAL

    row_set = set(row_corridors)
    col_set = set(col_corridors)
    reserved_targets: set[tuple[int, int]] = set()
    marker_specs: list[tuple[int, tuple[int, int]]] = []

    blockable_rows = [r for r in row_corridors if r != open_row]
    blockable_cols = [c for c in col_corridors if c != open_col]
    block_choices = [("row", r) for r in blockable_rows] + [("col", c) for c in blockable_cols]
    if not block_choices:
        return None

    blocking_plan: list[tuple[int, str, int]] = []
    if blockable_rows and blockable_cols and random.random() < 0.7:
        colors = [RED, BLUE]
        random.shuffle(colors)
        blocking_plan.append((colors[0], "row", random.choice(blockable_rows)))
        blocking_plan.append((colors[1], "col", random.choice(blockable_cols)))
    else:
        block_kind, block_value = random.choice(block_choices)
        blocking_plan.append((random.choice((RED, BLUE)), block_kind, block_value))

    for color, kind, value in blocking_plan:
        candidates = _blocking_candidates(grid, color, kind, value, row_set, col_set, reserved_targets)
        if not candidates:
            return None
        marker, visible_targets = random.choice(candidates)
        _place_marker(grid, marker, color, visible_targets, reserved_targets)
        marker_specs.append((color, marker))

    target_marker_count = random.randint(MARKER_COUNT_MIN, MARKER_COUNT_MAX)
    while len(marker_specs) < target_marker_count:
        colors_needed = {RED, BLUE} - {color for color, _marker in marker_specs}
        if colors_needed:
            color_order = list(colors_needed) + [RED, BLUE]
        else:
            color_order = [random.choice((RED, BLUE)), RED, BLUE]

        placed = False
        for color in color_order:
            candidates = _nonblocking_candidates(grid, color, row_set, col_set, reserved_targets)
            if not candidates:
                continue
            marker, visible_targets = random.choice(candidates)
            _place_marker(grid, marker, color, visible_targets, reserved_targets)
            marker_specs.append((color, marker))
            placed = True
            break
        if not placed:
            return None

    if len(marker_specs) < MARKER_COUNT_MIN or len(marker_specs) > MARKER_COUNT_MAX:
        return None
    if {color for color, _marker in marker_specs} != {RED, BLUE}:
        return None

    pre_rows = _open_row_corridors(grid)
    pre_cols = _open_col_corridors(grid)
    if len(pre_rows) + len(pre_cols) < 3:
        return None

    expanded = _expand_markers(grid)
    post_rows = _open_row_corridors(expanded)
    post_cols = _open_col_corridors(expanded)
    blocked_count = (len(pre_rows) - len(post_rows)) + (len(pre_cols) - len(post_cols))

    if open_row not in post_rows or open_col not in post_cols:
        return None
    if blocked_count < 1:
        return None
    if not np.any(expanded == YELLOW) or not np.any(expanded == ORANGE):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
