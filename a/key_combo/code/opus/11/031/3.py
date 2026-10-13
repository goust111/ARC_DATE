from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, components_mask, get_bg
except Exception:  # pragma: no cover - fallback for direct loading outside the ARC env
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        GREY = 5

    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(grid, return_counts=True)
        return int(values[np.argmax(counts)])

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        assert connectivity == 4
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        comps: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(coords, dtype=int))
        return comps


BG = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
GREY = int(Color.GREY)

GRID_ROWS = (12, 16)
GRID_COLS = (12, 16)
PIVOT_MARGIN = 3
SHAPE_AREA = (3, 6)
DECOR_COUNT = (3, 6)
MAX_ATTEMPTS = 500
SIDES = ("left", "right")


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _shape_component(grid: np.ndarray) -> np.ndarray:
    comps = components_mask(grid == BLUE, connectivity=4)
    assert len(comps) == 1, "expected exactly one blue 4-connected component"
    coords = comps[0]
    assert SHAPE_AREA[0] <= len(coords) <= SHAPE_AREA[1], "blue shape size out of range"
    return coords


def _reflection_info(shape: np.ndarray, pivot_r: int, pivot_c: int, grid_shape: tuple[int, int]) -> tuple[np.ndarray, int, str]:
    h, w = grid_shape
    reflected = np.array(
        [[2 * pivot_r - int(r), 2 * pivot_c - int(c)] for r, c in shape],
        dtype=int,
    )
    assert np.all((0 <= reflected[:, 0]) & (reflected[:, 0] < h))
    assert np.all((0 <= reflected[:, 1]) & (reflected[:, 1] < w))
    orig_set = {tuple(map(int, rc)) for rc in shape}
    refl_set = {tuple(map(int, rc)) for rc in reflected}
    assert not (orig_set & refl_set), "original and reflected shapes must not overlap"

    orig_centroid_r = _round_half_up(float(shape[:, 0].mean()))
    refl_centroid_r = _round_half_up(float(reflected[:, 0].mean()))
    d = abs(refl_centroid_r - orig_centroid_r)
    assert 1 <= d <= h // 2, "roll amount must be a positive bounded integer"

    reflected_side = "left" if float(reflected[:, 1].mean()) < pivot_c else "right"
    return reflected, d, reflected_side


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG, "expected black background"

    pivots = np.argwhere(g == GREY)
    assert len(pivots) == 1, "expected exactly one grey pivot"
    pivot_r, pivot_c = map(int, pivots[0])
    h, w = g.shape
    assert 0 < pivot_r < h - 1 and 0 < pivot_c < w - 1

    shape = _shape_component(g)
    rows = shape[:, 0]
    cols = shape[:, 1]
    assert np.all(rows < pivot_r), "blue shape must lie above the pivot to encode a downward roll"
    assert np.all(cols != pivot_c), "blue shape cannot use the pivot column"
    if np.all(cols < pivot_c):
        original_side = "left"
    elif np.all(cols > pivot_c):
        original_side = "right"
    else:
        raise AssertionError("blue shape must lie entirely on one side of the pivot")

    reflected, d, reflected_side = _reflection_info(shape, pivot_r, pivot_c, g.shape)
    assert reflected_side != original_side

    decor_mask = (g == RED) | (g == GREEN)
    decor_coords = np.argwhere(decor_mask)
    color_counts = {
        RED: int(np.count_nonzero(g == RED)),
        GREEN: int(np.count_nonzero(g == GREEN)),
    }
    assert DECOR_COUNT[0] <= color_counts[RED] <= DECOR_COUNT[1]
    assert DECOR_COUNT[0] <= color_counts[GREEN] <= DECOR_COUNT[1]
    assert np.all(decor_coords[:, 1] != pivot_c), "decorations cannot use the pivot column"

    left_decor = int(np.count_nonzero(decor_coords[:, 1] < pivot_c))
    right_decor = int(np.count_nonzero(decor_coords[:, 1] > pivot_c))
    assert left_decor >= 2 and right_decor >= 2, "both halves need visible data content"

    stage = g.copy()
    for r, c in reflected:
        assert stage[r, c] == BG, "reflected shape must land on empty cells"
        stage[r, c] = BLUE

    return {
        "grid": g,
        "pivot": (pivot_r, pivot_c),
        "shape": shape,
        "reflected": reflected,
        "d": d,
        "original_side": original_side,
        "reflected_side": reflected_side,
        "stage": stage,
    }


def _roll_stage(stage: np.ndarray, pivot_c: int, side: str, shift: int) -> np.ndarray:
    out = stage.copy()
    if side == "left":
        out[:, :pivot_c] = np.roll(stage[:, :pivot_c], shift=shift, axis=0)
    else:
        out[:, pivot_c + 1 :] = np.roll(stage[:, pivot_c + 1 :], shift=shift, axis=0)
    out[:, pivot_c] = stage[:, pivot_c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the blue shape through the grey pivot and roll only the reflection side by the reflected vertical displacement.

    Input structure:
    - The grid has a black background, exactly one grey pivot pixel, one blue 4-connected shape, and scattered red/green decoration pixels.
    - The blue shape lies entirely on one side of the pivot column and above the pivot row.
    - The pivot column itself is just a divider and contains only the grey pivot.

    Rule:
    - Reflect every blue shape pixel 180 degrees through the grey pivot, keeping both the original and reflected blue copies.
    - Round the original and reflected shape row centroids to the nearest integer and let d be their absolute difference.
    - Split the grid by the pivot column; the reflected shape lands entirely in exactly one half.
    - Cyclically roll that half downward by d rows, leaving the other half and the pivot column unchanged.

    Key insight:
    - The pivot has a double role: it is both the point-symmetry center that creates the reflected blue copy and the column that selects which half of the data grid should be rolled.
    """
    info = _analyze_input(grid)
    _pivot_r, pivot_c = info["pivot"]
    return _roll_stage(info["stage"], pivot_c, info["reflected_side"], info["d"])


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _grow_shape(max_up: int, max_horiz: int, side: str, area: int) -> set[tuple[int, int]] | None:
    sign = -1 if side == "left" else 1
    allowed = {(dr, sign * dc) for dr in range(-max_up, 0) for dc in range(1, max_horiz + 1)}
    seeds = [(dr, sign * dc) for dr in range(-min(max_up, 2), 0) for dc in range(1, min(max_horiz, 2) + 1)]
    seeds = [cell for cell in seeds if cell in allowed]
    if not seeds:
        return None

    chosen = {random.choice(seeds)}
    while len(chosen) < area:
        frontier = sorted(
            cell
            for cell in allowed - chosen
            if any(nb in chosen for nb in _neighbors4(cell))
        )
        if not frontier:
            return None
        chosen.add(random.choice(frontier))

    rows = {dr for dr, _dc in chosen}
    cols = {dc for _dr, dc in chosen}
    if len(rows) < 2 or len(cols) < 2:
        return None
    return chosen


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _sample_separated(candidates: list[tuple[int, int]], count: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    for cell in candidates:
        if cell in blocked:
            continue
        if any(_manhattan(cell, other) < 2 for other in blocked):
            continue
        if any(_manhattan(cell, other) < 2 for other in picked):
            continue
        picked.append(cell)
        if len(picked) == count:
            return picked
    return None


def _target_positions(h: int, w: int, pivot_c: int, cells: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if c == pivot_c or (r, c) in cells:
                continue
            if c < pivot_c:
                left.append((r, c))
            else:
                right.append((r, c))
    return left, right


def _build_case() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    pivot_r = random.randint(PIVOT_MARGIN, h - PIVOT_MARGIN - 1)
    pivot_c = random.randint(PIVOT_MARGIN, w - PIVOT_MARGIN - 1)
    original_side = random.choice(SIDES)

    max_up = pivot_r
    max_horiz = pivot_c if original_side == "left" else (w - 1 - pivot_c)
    area = random.randint(*SHAPE_AREA)
    shape_offsets = _grow_shape(max_up, max_horiz, original_side, area)
    if shape_offsets is None:
        return None

    shape = {(pivot_r + dr, pivot_c + dc) for dr, dc in shape_offsets}
    reflected = {(2 * pivot_r - r, 2 * pivot_c - c) for r, c in shape}
    if any(not (0 <= r < h and 0 <= c < w) for r, c in reflected):
        return None
    if shape & reflected:
        return None

    temp = np.zeros((h, w), dtype=int)
    temp[pivot_r, pivot_c] = GREY
    for r, c in shape:
        temp[r, c] = BLUE

    shape_arr = np.array(sorted(shape), dtype=int)
    try:
        _reflected_arr, d, reflected_side = _reflection_info(shape_arr, pivot_r, pivot_c, temp.shape)
    except AssertionError:
        return None
    if reflected_side == original_side:
        return None

    occupied = set(shape) | set(reflected) | {(pivot_r, pivot_c)}
    left_pool, right_pool = _target_positions(h, w, pivot_c, occupied)

    red_total = random.randint(*DECOR_COUNT)
    green_total = random.randint(*DECOR_COUNT)
    left_red = random.randint(1, red_total - 1)
    left_green = random.randint(1, green_total - 1)
    right_red = red_total - left_red
    right_green = green_total - left_green

    if reflected_side == "left" and left_red + left_green < 3:
        return None
    if reflected_side == "right" and right_red + right_green < 3:
        return None

    left_need = left_red + left_green
    right_need = right_red + right_green
    if len(left_pool) < left_need or len(right_pool) < right_need:
        return None

    left_cells = _sample_separated(left_pool, left_need, occupied)
    if left_cells is None:
        return None
    occupied |= set(left_cells)
    right_cells = _sample_separated(right_pool, right_need, occupied)
    if right_cells is None:
        return None

    random.shuffle(left_cells)
    random.shuffle(right_cells)
    left_red_cells = left_cells[:left_red]
    left_green_cells = left_cells[left_red:]
    right_red_cells = right_cells[:right_red]
    right_green_cells = right_cells[right_red:]

    grid = np.zeros((h, w), dtype=int)
    grid[pivot_r, pivot_c] = GREY
    for r, c in shape:
        grid[r, c] = BLUE
    for r, c in left_red_cells + right_red_cells:
        grid[r, c] = RED
    for r, c in left_green_cells + right_green_cells:
        grid[r, c] = GREEN

    if not _oracle_checks(grid):
        return None
    return grid


def _oracle_checks(grid: np.ndarray) -> bool:
    try:
        info = _analyze_input(grid)
    except Exception:
        return False

    stage = info["stage"]
    pivot_c = info["pivot"][1]
    d = info["d"]
    side = info["reflected_side"]
    out = _roll_stage(stage, pivot_c, side, d)

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, out):
        return False

    target = stage[:, :pivot_c] if side == "left" else stage[:, pivot_c + 1 :]
    other = stage[:, pivot_c + 1 :] if side == "left" else stage[:, :pivot_c]
    rolled_target = actual[:, :pivot_c] if side == "left" else actual[:, pivot_c + 1 :]
    rolled_other = actual[:, pivot_c + 1 :] if side == "left" else actual[:, :pivot_c]

    if np.array_equal(target, rolled_target):
        return False
    if not np.array_equal(other, rolled_other):
        return False
    if not np.array_equal(actual[:, pivot_c], stage[:, pivot_c]):
        return False

    wrong_half = "right" if side == "left" else "left"
    if np.array_equal(_roll_stage(stage, pivot_c, wrong_half, d), actual):
        return False
    if np.array_equal(_roll_stage(stage, pivot_c, side, 0), actual):
        return False
    if np.array_equal(_roll_stage(grid.copy(), pivot_c, side, d), actual):
        return False

    blue_area = len(info["shape"])
    if int(np.count_nonzero(actual == BLUE)) != 2 * blue_area:
        return False
    if int(np.count_nonzero(grid == RED)) < 3 or int(np.count_nonzero(grid == GREEN)) < 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pivot-reflection roll puzzle")
