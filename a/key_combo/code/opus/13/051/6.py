from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MASK = 6
TEAL = 8

PALETTE = (BLUE, RED, GREEN)
GRID_SIZES = (13, 15, 17)
SHAPES = (
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
)
MAX_GEN_ATTEMPTS = 500


def _mirror_col(width: int, c: int) -> int:
    return int(width - 1 - c)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _vertical_mask_candidate(grid: np.ndarray, candidate: int) -> bool:
    g = np.asarray(grid, dtype=int)
    if not np.any(g == int(candidate)):
        return False

    height, width = g.shape
    used = False
    for r in range(height):
        for c in range((width + 1) // 2):
            mc = _mirror_col(width, c)
            a = int(g[r, c])
            b = int(g[r, mc])
            a_mask = a == int(candidate)
            b_mask = b == int(candidate)
            if c == mc:
                if a_mask:
                    return False
                continue
            if a_mask and b_mask:
                return False
            if a_mask or b_mask:
                used = True
                continue
            if a != b:
                return False
    return used


def _infer_mask_color(grid: np.ndarray) -> int:
    colors = [int(color) for color in np.unique(grid)]
    candidates = [color for color in colors if _vertical_mask_candidate(grid, color)]
    assert len(candidates) == 1, f"expected one mask color, got {candidates}"
    return int(candidates[0])


def _repair_vertical_symmetry(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    mask_color = _infer_mask_color(g)
    out = g.copy()
    for r in range(height):
        for c in range(width):
            if int(out[r, c]) != mask_color:
                continue
            mc = _mirror_col(width, c)
            assert mc != c, "masked center-axis cells are not recoverable"
            fill = int(out[r, mc])
            assert fill != mask_color, "mirror mate of every masked cell must stay visible"
            out[r, c] = fill
    return out


def _column_runs_of_color(grid: np.ndarray, color: int) -> list[tuple[int, int, int]]:
    height, width = grid.shape
    runs: list[tuple[int, int, int]] = []
    for c in range(width):
        r = 0
        while r < height:
            if int(grid[r, c]) != int(color):
                r += 1
                continue
            start = r
            while r + 1 < height and int(grid[r + 1, c]) == int(color):
                r += 1
            runs.append((c, start, r))
            r += 1
    return runs


def _find_rails(grid: np.ndarray) -> list[tuple[int, int, int]]:
    runs = [
        (c, r0, r1)
        for c, r0, r1 in _column_runs_of_color(grid, GREY)
        if r1 - r0 + 1 >= 4
    ]
    assert len(runs) == 2, f"expected exactly two gray rails, got {runs}"
    runs.sort(key=lambda item: item[0])
    left, right = runs
    assert left[1:] == right[1:], "the two rails must share the same row extent"
    return runs


def _crop_from_intact_grid(grid: np.ndarray) -> np.ndarray:
    repaired = np.asarray(grid, dtype=int)
    rails = _find_rails(repaired)
    left_col = int(rails[0][0])
    right_col = int(rails[1][0])
    top = int(rails[0][1])
    bottom = int(rails[0][2])

    expected_markers = {
        (top - 1, left_col),
        (bottom + 1, left_col),
        (top - 1, right_col),
        (bottom + 1, right_col),
    }
    assert top - 1 >= 0 and bottom + 1 < repaired.shape[0]
    teal_positions = {
        (int(r), int(c))
        for r, c in np.argwhere(repaired == TEAL)
        if int(c) in (left_col, right_col)
    }
    assert teal_positions == expected_markers, f"unexpected teal markers: {teal_positions}"
    return _crop_bbox(repaired, (top - 1, left_col, bottom + 1, right_col))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a vertically symmetric damaged scene, then crop the rectangle marked by the restored rails and corner pixels.

    Input structure:
    - The grid has a black background and is symmetric about the vertical middle axis before damage.
    - Two gray vertical rails sit at mirror columns, and four teal corner markers lie one row beyond the rail endpoints.
    - Other blue/red/green foreground pixels also respect the same left-right symmetry.
    - A magenta mask color overwrites several original pixels, including at least one crop-boundary marker or rail cell.

    Rule:
    - Identify the unique mask color whose removal makes every vertical mirror pair consistent.
    - Replace each masked pixel with the color at its vertical mirror mate.
    - In the repaired grid, find the two gray vertical rails of length at least four and sort them left to right.
    - Use their columns plus the teal pixels one row beyond their endpoints to define the crop rectangle, then return that subgrid.

    Key insight:
    - The crop markers themselves can be damaged, so the rails and teal corners are only trustworthy after restoring every masked cell from its visible mirror partner.
    """
    repaired = _repair_vertical_symmetry(np.asarray(grid, dtype=int))
    return _crop_from_intact_grid(repaired)


def _paint_pair(grid: np.ndarray, coords: Iterable[tuple[int, int]], color: int) -> None:
    width = grid.shape[1]
    for r, c in coords:
        grid[int(r), int(c)] = int(color)
        grid[int(r), _mirror_col(width, int(c))] = int(color)


def _can_place_pair(
    blocked: np.ndarray,
    width: int,
    coords: Iterable[tuple[int, int]],
) -> bool:
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        mc = _mirror_col(width, cc)
        if blocked[rr, cc] or blocked[rr, mc]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, coords: Iterable[tuple[int, int]]) -> None:
    width = blocked.shape[1]
    occupied = np.zeros_like(blocked, dtype=bool)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        occupied[rr, cc] = True
        occupied[rr, _mirror_col(width, cc)] = True
    blocked |= _dilate8(occupied)


def _shape_coords(anchor_r: int, anchor_c: int, shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return [(int(anchor_r + dr), int(anchor_c + dc)) for dr, dc in shape]


def _zone_candidates(
    shape: tuple[tuple[int, int], ...],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> list[tuple[int, int]]:
    h = max(dr for dr, _ in shape) + 1
    w = max(dc for _, dc in shape) + 1
    r_lo, r_hi = map(int, row_range)
    c_lo, c_hi = map(int, col_range)
    rows = list(range(r_lo, r_hi - h + 2))
    cols = list(range(c_lo, c_hi - w + 2))
    return [(r, c) for r in rows for c in cols]


def _place_one_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> bool:
    height, width = grid.shape
    positions = _zone_candidates(shape, row_range, col_range)
    random.shuffle(positions)
    for anchor_r, anchor_c in positions:
        coords = _shape_coords(anchor_r, anchor_c, shape)
        if any(not (0 <= r < height and 0 <= c < width // 2) for r, c in coords):
            continue
        if not _can_place_pair(blocked, width, coords):
            continue
        _paint_pair(grid, coords, color)
        _mark_blocked(blocked, coords)
        return True
    return False


def _place_from_choices(
    grid: np.ndarray,
    blocked: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    *,
    shapes: Iterable[tuple[tuple[int, int], ...]] = SHAPES,
    colors: Iterable[int] = PALETTE,
) -> bool:
    shape_list = list(shapes)
    color_list = list(colors)
    random.shuffle(shape_list)
    random.shuffle(color_list)
    for shape in shape_list:
        for color in color_list:
            if _place_one_pair(grid, blocked, shape, color, row_range, col_range):
                return True
    return False


def _break_rows(top: int, line_len: int) -> list[int]:
    rows: list[int] = []
    visible_run = 0
    for idx in range(line_len):
        if visible_run == 3:
            rows.append(top + idx)
            visible_run = 0
        else:
            visible_run += 1
    return rows


def _try_crop_without_repair(grid: np.ndarray) -> np.ndarray | None:
    try:
        return _crop_from_intact_grid(np.asarray(grid, dtype=int))
    except AssertionError:
        return None


def _build_sample() -> tuple[np.ndarray, np.ndarray]:
    size = random.choice(GRID_SIZES)
    grid = np.zeros((size, size), dtype=int)
    mid = size // 2

    line_len = random.randint(4, min(8, size - 4))
    top = random.randint(2, size - line_len - 3)
    bottom = top + line_len - 1
    left_col = random.randint(2, max(2, mid - 3))
    right_col = _mirror_col(size, left_col)
    crop_r0 = top - 1
    crop_r1 = bottom + 1

    grid[top : bottom + 1, left_col] = GREY
    grid[top : bottom + 1, right_col] = GREY
    grid[crop_r0, left_col] = TEAL
    grid[crop_r1, left_col] = TEAL
    grid[crop_r0, right_col] = TEAL
    grid[crop_r1, right_col] = TEAL

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[:, mid] = True
    boundary_mask = np.zeros_like(grid, dtype=bool)
    boundary_mask[top : bottom + 1, left_col] = True
    boundary_mask[top : bottom + 1, right_col] = True
    boundary_mask[crop_r0, left_col] = True
    boundary_mask[crop_r1, left_col] = True
    boundary_mask[crop_r0, right_col] = True
    boundary_mask[crop_r1, right_col] = True
    blocked |= _dilate8(boundary_mask)

    inside_ok = _place_from_choices(
        grid,
        blocked,
        (crop_r0 + 1, crop_r1 - 1),
        (left_col + 1, mid - 1),
    )
    assert inside_ok

    outside_zones: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if crop_r0 >= 3:
        outside_zones.append(((0, crop_r0 - 2), (1, mid - 1)))
    if crop_r1 <= size - 4:
        outside_zones.append(((crop_r1 + 2, size - 1), (1, mid - 1)))
    outside_zones.append(((crop_r0, crop_r1), (1, max(1, left_col - 2))))
    random.shuffle(outside_zones)
    placed_outside = False
    for row_range, col_range in outside_zones:
        if row_range[0] > row_range[1] or col_range[0] > col_range[1]:
            continue
        if _place_from_choices(
            grid,
            blocked,
            row_range,
            col_range,
        ):
            placed_outside = True
            break
    assert placed_outside

    for _ in range(random.randint(2, 5)):
        shape = random.choice(SHAPES)
        color = random.choice(PALETTE)
        zone_kind = random.choice(("inside", "above", "below", "outer_left"))
        if zone_kind == "inside":
            row_range = (crop_r0 + 1, crop_r1 - 1)
            col_range = (left_col + 1, mid - 1)
        elif zone_kind == "above":
            if crop_r0 < 3:
                continue
            row_range = (0, crop_r0 - 2)
            col_range = (1, mid - 1)
        elif zone_kind == "below":
            if crop_r1 > size - 4:
                continue
            row_range = (crop_r1 + 2, size - 1)
            col_range = (1, mid - 1)
        else:
            if left_col < 3:
                continue
            row_range = (crop_r0, crop_r1)
            col_range = (1, left_col - 2)
        _place_one_pair(grid, blocked, shape, color, row_range, col_range)

    full = grid.copy()
    side_col = left_col if random.choice((True, False)) else right_col
    rail_rows = _break_rows(top, line_len)
    marker_rows = [random.choice((crop_r0, crop_r1))]
    mandatory = {(int(r), int(side_col)) for r in rail_rows}
    mandatory.update((int(r), int(side_col)) for r in marker_rows)

    if side_col == left_col:
        side_foreground = [
            (int(r), int(c))
            for r, c in np.argwhere(full != BG)
            if int(c) < mid
        ]
    else:
        side_foreground = [
            (int(r), int(c))
            for r, c in np.argwhere(full != BG)
            if int(c) > mid
        ]
    color_extras = [
        (r, c)
        for r, c in side_foreground
        if full[r, c] in PALETTE and (r, c) not in mandatory
    ]
    boundary_extras = [
        (r, c)
        for r, c in side_foreground
        if full[r, c] in (GREY, TEAL) and (r, c) not in mandatory
    ]
    random.shuffle(color_extras)
    random.shuffle(boundary_extras)
    target_mask_count = random.randint(3, 6)
    masked = set(mandatory)
    if color_extras:
        masked.add(color_extras.pop())
    extras_pool = color_extras + boundary_extras
    random.shuffle(extras_pool)
    while len(masked) < target_mask_count and extras_pool:
        masked.add(extras_pool.pop())
    if len(masked) < 3:
        raise AssertionError("not enough maskable foreground cells")

    damaged = full.copy()
    for r, c in masked:
        assert full[r, c] != BG
        assert full[r, _mirror_col(size, c)] != MASK
        damaged[r, c] = MASK

    expected = _crop_from_intact_grid(full)
    return damaged, expected


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        if inp.ndim != 2 or expected.ndim != 2:
            return False
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            return False
        if _infer_mask_color(inp) != MASK:
            return False
        solved = main(inp)
        if not np.array_equal(solved, expected):
            return False

        repaired = _repair_vertical_symmetry(inp)
        if not np.array_equal(repaired, np.fliplr(repaired)):
            return False

        raw_crop = _try_crop_without_repair(inp)
        if raw_crop is not None and np.array_equal(raw_crop, expected):
            return False

        rails = _find_rails(repaired)
        left_col = rails[0][0]
        right_col = rails[1][0]
        if expected.shape[1] != right_col - left_col + 1:
            return False
        inner = expected[1:-1, 1:-1]
        if not np.any(np.isin(inner, PALETTE)):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp, expected = _build_sample()
        except AssertionError:
            continue
        if _validate_generated(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid sample")
