from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, in_bounds


BG = Color.BLACK
RECT = Color.TEAL
GAP = Color.RED
BAR = Color.YELLOW
RAY = Color.ORANGE

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
RECT_H_RANGE = (6, 8)
RECT_W_RANGE = (7, 10)
MISSING_RANGE = (4, 8)
GENERATION_RETRIES = 400


def _infer_instance(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rect_mask = g == RECT
    rect_bbox = bbox(rect_mask)
    assert rect_bbox is not None, "expected teal rectangle pixels"
    r0, c0, r1, c1 = rect_bbox
    assert r1 + 1 < g.shape[0], "expected a trigger bar directly below the rectangle"

    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((rect == BG) | (rect == RECT)), "rectangle area may only contain teal or background"
    assert np.all(rect[0, [0, -1]] == RECT) and np.all(rect[-1, [0, -1]] == RECT), "corner cells must be present"
    assert np.all(np.any(rect == RECT, axis=0)), "every rectangle column must keep at least one teal pixel"
    assert np.all(np.any(rect == RECT, axis=1)), "every rectangle row must keep at least one teal pixel"

    bar_row = r1 + 1
    assert np.all(g[bar_row, c0 : c1 + 1] == BAR), "expected a full yellow trigger bar matching the rectangle width"
    return r0, c0, r1, c1, bar_row


def _trace_ray(shape: tuple[int, int], teal_mask: np.ndarray, start_r: int, start_c: int, dc: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    rr = int(start_r)
    cc = int(start_c)
    step_c = int(dc)
    while in_bounds(shape, rr, cc) and not bool(teal_mask[rr, cc]):
        cells.append((rr, cc))
        rr -= 1
        cc += step_c
    return cells


def _spec_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1, bar_row = _infer_instance(g)
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = np.where(
        out[r0 : r1 + 1, c0 : c1 + 1] == BG,
        GAP,
        out[r0 : r1 + 1, c0 : c1 + 1],
    )

    teal_mask = g == RECT
    trigger_cols = [
        int(c)
        for c in range(c0, c1 + 1)
        if int(g[r1, c]) == BG and int(g[bar_row, c]) == BAR
    ]
    assert trigger_cols, "expected at least one bottom-edge gap touching the trigger bar"

    for c in trigger_cols:
        for dc in (-1, 1):
            for rr, cc in _trace_ray(g.shape, teal_mask, r1 - 1, c + dc, dc):
                out[rr, cc] = RAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the incomplete teal rectangle, mark its gaps red, and splash diagonal orange rays from bottom-edge gaps touching the yellow bar.

    Input structure:
    - The grid has a black background, one incomplete teal filled rectangle, and one yellow horizontal trigger bar.
    - The teal pixels define a rectangular bounding box with several black missing cells inside it.
    - The yellow bar sits immediately below the rectangle and has the same width as the rectangle bbox.

    Rule:
    - Infer the teal bbox and fill every black cell inside it with red.
    - Every red gap on the bbox bottom row is a trigger because the yellow bar is directly below it.
    - From each trigger gap, shoot orange rays one step up-left and up-right, continuing until the ray leaves the grid or the next cell is teal.
    - Orange may overwrite red gap cells and black background cells, but teal rectangle pixels and the yellow bar stay unchanged.

    Key insight:
    - The same missing cells serve two roles: bottom-row gaps identify which positions emit rays, while other gaps only turn orange if those rays can reach them without being blocked by teal.
    """
    return _spec_output(grid)


def _local_ray_path(height: int, width: int, bottom_col: int, dc: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    rr = height - 2
    cc = int(bottom_col) + int(dc)
    while 0 <= rr < height and 0 <= cc < width:
        cells.append((rr, cc))
        rr -= 1
        cc += int(dc)
    return cells


def _all_gap_cells(height: int, width: int) -> list[tuple[int, int]]:
    corners = {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)}
    return [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in corners
    ]


def _rows_cols_still_identifiable(height: int, width: int, holes: set[tuple[int, int]]) -> bool:
    for r in range(height):
        if sum((r, c) in holes for c in range(width)) >= width:
            return False
    for c in range(width):
        if sum((r, c) in holes for r in range(height)) >= height:
            return False
    return True


def _globalize(r0: int, c0: int, cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r0 + r, c0 + c) for r, c in cells}


def _semantic_profile(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1, bar_row = _infer_instance(g)
    out = _spec_output(g)

    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    holes = {
        (r0 + int(r), c0 + int(c))
        for r, c in np.argwhere(rect == BG)
    }
    bottom_holes = {(r, c) for r, c in holes if r == r1}
    inner_holes = holes - bottom_holes
    orange = {
        (int(r), int(c))
        for r, c in np.argwhere(out == RAY)
    }

    blocked_dirs = 0
    for _r, c in bottom_holes:
        for dc in (-1, 1):
            rr = r1 - 1
            cc = c + dc
            if in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == RECT:
                blocked_dirs += 1

    return {
        "bbox": (r0, c0, r1, c1),
        "bar_row": bar_row,
        "holes": holes,
        "bottom_holes": bottom_holes,
        "inner_holes": inner_holes,
        "orange": orange,
        "out": out,
        "blocked_dirs": blocked_dirs,
    }


def _validate_candidate(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(map(int, np.unique(g))) <= {BG, RECT, BAR}:
        return False

    try:
        prof = _semantic_profile(g)
    except AssertionError:
        return False

    r0, c0, r1, c1 = prof["bbox"]
    holes = prof["holes"]
    bottom_holes = prof["bottom_holes"]
    inner_holes = prof["inner_holes"]
    orange = prof["orange"]
    out = prof["out"]

    if not (MISSING_RANGE[0] <= len(holes) <= MISSING_RANGE[1]):
        return False
    if len(bottom_holes) < 1 or len(inner_holes) < 1:
        return False
    if prof["blocked_dirs"] < 1:
        return False

    covered_inner = [cell for cell in inner_holes if int(out[cell]) == RAY]
    surviving_inner = [cell for cell in inner_holes if int(out[cell]) == GAP]
    if not covered_inner or not surviving_inner:
        return False

    if not any(c < c0 or c > c1 or r < r0 for r, c in orange):
        return False

    if any(int(out[cell]) != GAP for cell in bottom_holes):
        return False
    if np.any(out[g == RECT] != RECT):
        return False
    if np.any(out[prof["bar_row"], c0 : c1 + 1] != BAR):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rect_h = random.randint(*RECT_H_RANGE)
    rect_w = random.randint(*RECT_W_RANGE)
    grid_h = random.randint(max(GRID_H_RANGE[0], rect_h + 4), GRID_H_RANGE[1])
    grid_w = random.randint(max(GRID_W_RANGE[0], rect_w + 4), GRID_W_RANGE[1])

    max_r0 = min(grid_h - rect_h - 3, max(1, grid_h // 2 - rect_h // 2))
    if max_r0 < 1:
        return None
    r0 = random.randint(1, max_r0)
    c0 = random.randint(1, grid_w - rect_w - 2)

    target_holes = random.randint(*MISSING_RANGE)
    holes: set[tuple[int, int]] = set()

    main_dc = random.choice((-1, 1))
    if main_dc == -1:
        main_cols = [c for c in range(1, rect_w - 1) if 2 <= c <= min(4, rect_w - 2)]
    else:
        main_cols = [c for c in range(1, rect_w - 1) if 2 <= (rect_w - 1 - c) <= 4]
    if not main_cols:
        return None

    main_c = random.choice(main_cols)
    open_path = _local_ray_path(rect_h, rect_w, main_c, main_dc)
    if not (2 <= len(open_path) <= 4):
        return None

    holes.add((rect_h - 1, main_c))
    holes.update(open_path)

    keep_teal = set()
    blocked_neighbor = (rect_h - 2, main_c - main_dc)
    if 0 <= blocked_neighbor[0] < rect_h and 0 <= blocked_neighbor[1] < rect_w:
        keep_teal.add(blocked_neighbor)

    extra_bottom_target = random.choices([0, 1, 2], weights=[2, 5, 3])[0]
    bottom_candidates = [c for c in range(1, rect_w - 1) if c != main_c and abs(c - main_c) >= 2]
    random.shuffle(bottom_candidates)
    for extra_c in bottom_candidates:
        if extra_bottom_target == 0 or len(holes) >= target_holes:
            break
        holes.add((rect_h - 1, extra_c))
        extra_bottom_target -= 1
        side_options = [dc for dc in (-1, 1) if 0 <= extra_c + dc < rect_w]
        random.shuffle(side_options)
        for dc in side_options[:1]:
            short_cell = (rect_h - 2, extra_c + dc)
            if len(holes) < target_holes and short_cell not in keep_teal and short_cell[0] > 0 and 0 < short_cell[1] < rect_w - 1 and random.random() < 0.45:
                holes.add(short_cell)

    interior_candidates = [
        (r, c)
        for r in range(1, rect_h - 1)
        for c in range(1, rect_w - 1)
        if (r, c) not in holes and (r, c) not in open_path
    ]
    if not interior_candidates:
        return None
    off_ray_gap = random.choice(interior_candidates)
    holes.add(off_ray_gap)

    all_candidates = _all_gap_cells(rect_h, rect_w)
    random.shuffle(all_candidates)
    for cell in all_candidates:
        if len(holes) >= target_holes:
            break
        if cell in holes or cell in keep_teal:
            continue
        test_holes = set(holes)
        test_holes.add(cell)
        if _rows_cols_still_identifiable(rect_h, rect_w, test_holes):
            holes = test_holes

    if len(holes) != target_holes:
        return None
    if not _rows_cols_still_identifiable(rect_h, rect_w, holes):
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = RECT
    for rr, cc in _globalize(r0, c0, holes):
        grid[rr, cc] = BG
    grid[r0 + rect_h, c0 : c0 + rect_w] = BAR

    return grid if _validate_candidate(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
