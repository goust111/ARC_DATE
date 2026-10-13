from __future__ import annotations

import random

import numpy as np


BG = 0
ANCHOR_COLOR = 8
STAIR_COLOR = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 9)
STOP_KINDS = ("horizontal", "vertical")

GRID_HEIGHT_RANGE = (16, 28)
GRID_WIDTH_RANGE = (16, 28)
STEP_RANGE = (1, 6)
MIN_STAIR_CELLS = 12
GENERATION_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _trace_arm(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    w: int,
    h: int,
    horizontal_dir: int,
    vertical_dir: int,
) -> tuple[list[tuple[int, int]], str]:
    r, c = anchor
    cells: list[tuple[int, int]] = []
    while True:
        for _ in range(w):
            nr, nc = r, c + horizontal_dir
            if not _in_bounds(shape, nr, nc):
                return cells, "horizontal"
            r, c = nr, nc
            cells.append((r, c))
        for _ in range(h):
            nr, nc = r + vertical_dir, c
            if not _in_bounds(shape, nr, nc):
                return cells, "vertical"
            r, c = nr, nc
            cells.append((r, c))


def _parse_instance(grid: np.ndarray) -> tuple[tuple[int, int], int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset(set(range(10))), "expected ARC color ids"
    assert int(np.min(g)) >= 0

    anchor_coords = np.argwhere(g == ANCHOR_COLOR)
    assert anchor_coords.shape[0] == 1, "expected exactly one cyan anchor pixel"
    anchor = tuple(map(int, anchor_coords[0]))

    foreground = sorted(set(map(int, np.unique(g))) - {BG, ANCHOR_COLOR})
    assert len(foreground) == 1, "expected exactly one non-anchor foreground color"
    bar_color = foreground[0]

    coords = np.argwhere(g == bar_color)
    assert coords.size > 0, "expected a visible vertical bar pair"
    cols = sorted(set(map(int, coords[:, 1])))
    assert len(cols) == 2, "expected exactly two bar columns"

    row_runs: list[list[int]] = []
    for col in cols:
        rows = sorted(map(int, coords[coords[:, 1] == col, 0]))
        assert rows == list(range(rows[0], rows[-1] + 1)), "bar pixels must be vertically contiguous"
        row_runs.append(rows)

    assert row_runs[0] == row_runs[1], "paired bars must share the same row interval"
    h = len(row_runs[0])
    w = cols[1] - cols[0]
    assert h >= 1 and w >= 1
    assert coords.shape[0] == 2 * h, "only the two bars should use the bar color"
    return anchor, bar_color, h, w


def _stair_paths(
    shape: tuple[int, int], anchor: tuple[int, int], w: int, h: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], str, str]:
    arm_ne, stop_ne = _trace_arm(shape, anchor, w, h, horizontal_dir=1, vertical_dir=-1)
    arm_sw, stop_sw = _trace_arm(shape, anchor, w, h, horizontal_dir=-1, vertical_dir=1)
    return arm_ne, arm_sw, stop_ne, stop_sw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw two point-symmetric gray staircase arms whose horizontal and vertical step sizes are encoded by a distant bar pair.

    Input structure:
    - The grid has a black background, exactly one cyan anchor pixel, and one pair
      of same-color 1-pixel-wide vertical bars.
    - The two bars occupy different columns but share one contiguous row interval.
    - The bar height encodes the vertical step size, and the column distance
      between the two bars encodes the horizontal step size.

    Rule:
    - Preserve the input content.
    - Starting from the anchor, repeatedly draw `w` cells to the right and then
      `h` cells upward until the path would leave the grid.
    - Also draw the point-symmetric arm by repeatedly drawing `w` cells to the
      left and then `h` cells downward until the path would leave the grid.
    - Finally restore the cyan anchor pixel.

    Key insight:
    - The bars are not transformed objects; they are a geometric codebook whose
      height and spacing uniquely determine the staircase's vertical and
      horizontal step lengths.
    """
    g = np.asarray(grid, dtype=int)
    anchor, _bar_color, h, w = _parse_instance(g)
    out = g.copy()

    arm_ne, arm_sw, _stop_ne, _stop_sw = _stair_paths(g.shape, anchor, w, h)
    for r, c in arm_ne + arm_sw:
        if out[r, c] == BG:
            out[r, c] = STAIR_COLOR

    out[anchor] = ANCHOR_COLOR
    return out


def _anchor_candidates(height: int, width: int, h: int, w: int) -> list[tuple[int, int, str, str]]:
    candidates: list[tuple[int, int, str, str]] = []
    for r in range(h, height - h):
        for c in range(w, width - w):
            if r < h or height - 1 - r < h or c < w or width - 1 - c < w:
                continue
            arm_ne, arm_sw, stop_ne, stop_sw = _stair_paths((height, width), (r, c), w, h)
            if len(arm_ne) < h + w or len(arm_sw) < h + w:
                continue
            if len(set(arm_ne) | set(arm_sw)) < MIN_STAIR_CELLS:
                continue
            candidates.append((r, c, stop_ne, stop_sw))
    return candidates


def _bar_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c0) for r in range(r0, r0 + h)} | {(r, c0 + w) for r in range(r0, r0 + h)}


def _bar_candidates(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = shape
    ar, ac = anchor
    candidates: list[tuple[int, int]] = []

    quadrants = [
        (range(1, max(1, ar - h)), range(1, max(1, ac - w - 1))),
        (range(ar + 2, height - h), range(ac + 2, width - w - 1)),
    ]
    for row_range, col_range in quadrants:
        for r0 in row_range:
            for c0 in col_range:
                cells = _bar_cells(r0, c0, h, w)
                if any(not _in_bounds(shape, r, c) for r, c in cells):
                    continue
                if any((r, c) in forbidden for r, c in cells):
                    continue
                if min(abs(r - ar) + abs(c - ac) for r, c in cells) < h + w + 1:
                    continue
                candidates.append((r0, c0))
    return candidates


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    anchor, _bar_color, h, w = _parse_instance(g)
    expected = g.copy()
    arm_ne, arm_sw, _stop_ne, _stop_sw = _stair_paths(g.shape, anchor, w, h)
    for r, c in arm_ne + arm_sw:
        if expected[r, c] == BG:
            expected[r, c] = STAIR_COLOR
    expected[anchor] = ANCHOR_COLOR
    return expected


def _semantic_checks(grid: np.ndarray) -> bool:
    anchor, bar_color, h, w = _parse_instance(grid)
    assert bar_color != STAIR_COLOR, "bar color should stay visually distinct from the staircase"
    assert h != w, "generator should keep the two encoded parameters distinguishable"

    arm_ne, arm_sw, stop_ne, stop_sw = _stair_paths(grid.shape, anchor, w, h)
    assert stop_ne in STOP_KINDS and stop_sw in STOP_KINDS
    assert len(arm_ne) >= h + w and len(arm_sw) >= h + w

    out = main(grid)
    expected = _expected_output(grid)
    assert np.array_equal(out, expected)
    assert np.array_equal(out[grid != BG], grid[grid != BG]), "input content must be preserved"

    gray_coords = {tuple(map(int, rc)) for rc in np.argwhere(out == STAIR_COLOR)}
    expected_gray = set(arm_ne) | set(arm_sw)
    assert gray_coords == expected_gray
    assert len(gray_coords) >= MIN_STAIR_CELLS
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    h = random.randint(*STEP_RANGE)
    w = random.randint(*STEP_RANGE)
    if h == w:
        return None

    anchor_pool = _anchor_candidates(height, width, h, w)
    if not anchor_pool:
        return None

    random.shuffle(anchor_pool)
    desired_stops = [(a, b) for a in STOP_KINDS for b in STOP_KINDS]
    random.shuffle(desired_stops)
    chosen_anchor: tuple[int, int] | None = None
    for stop_ne, stop_sw in desired_stops:
        for r, c, cand_stop_ne, cand_stop_sw in anchor_pool:
            if (cand_stop_ne, cand_stop_sw) == (stop_ne, stop_sw):
                chosen_anchor = (r, c)
                break
        if chosen_anchor is not None:
            break
    if chosen_anchor is None:
        r, c, _stop_ne, _stop_sw = random.choice(anchor_pool)
        chosen_anchor = (r, c)

    arm_ne, arm_sw, _stop_ne, _stop_sw = _stair_paths((height, width), chosen_anchor, w, h)
    forbidden = set(arm_ne) | set(arm_sw) | {chosen_anchor}
    bar_positions = _bar_candidates((height, width), chosen_anchor, h, w, forbidden)
    if not bar_positions:
        return None

    bar_color = random.choice(BAR_COLORS)
    r0, c0 = random.choice(bar_positions)
    grid = np.zeros((height, width), dtype=int)
    grid[chosen_anchor] = ANCHOR_COLOR
    for r, c in _bar_cells(r0, c0, h, w):
        grid[r, c] = bar_color

    try:
        if _semantic_checks(grid):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid staircase-coded input")
