from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, get_bg


BG = 0
PALETTE = tuple(range(1, 10))
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
MAX_GENERATION_ATTEMPTS = 400
GEOMETRY_TRIES = 160
SELECTION_TRIES = 1600
MIN_STRAIGHT_LEN = 4
MAX_STRAIGHT_LEN = 14
MAX_TOTAL_CROSSINGS = 4


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _straight_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        return [(r0, c) for c in _inclusive_range(c0, c1)]
    if c0 == c1:
        return [(r, c0) for r in _inclusive_range(r0, r1)]
    raise AssertionError("straight path endpoints must share a row or a column")


def _l_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] <= b[0]:
        (rt, ct), (rb, cb) = a, b
    else:
        (rt, ct), (rb, cb) = b, a
    assert rt < rb and ct != cb
    cells = [(rt, c) for c in _inclusive_range(ct, cb)]
    cells.extend((r, cb) for r in range(rt + 1, rb + 1))
    return cells


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> tuple[str, list[tuple[int, int]]]:
    if a[0] == b[0] or a[1] == b[1]:
        return "straight", _straight_cells(a, b)
    return "l", _l_cells(a, b)


def _extract_paths(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "expected a black background"

    paths: list[dict] = []
    l_count = 0
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        kind, cells = _path_cells(a, b)
        if kind == "l":
            l_count += 1
        paths.append(
            {
                "color": int(color),
                "kind": kind,
                "endpoints": (a, b),
                "cells": cells,
                "length": len(cells),
            }
        )

    assert l_count == 1, "expected exactly one non-axis-aligned pair"
    lengths = [path["length"] for path in paths]
    assert len(lengths) == len(set(lengths)), "all path lengths must be distinct"
    return sorted(paths, key=lambda path: (int(path["length"]), int(path["color"])))


def _render_overlay(grid: np.ndarray, paths: list[dict], *, reverse: bool = False) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = int(get_bg(out))
    ordered = sorted(paths, key=lambda path: int(path["length"]), reverse=reverse)
    for path in ordered:
        color = int(path["color"])
        for r, c in path["cells"]:
            if out[r, c] == bg:
                out[r, c] = color
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    paths = _extract_paths(grid)
    return _render_overlay(np.asarray(grid, dtype=int), paths)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair, drawing shorter paths first so earlier crossings stay visible.

    Input structure:
    - The background is black and every foreground color appears in exactly two marker pixels.
    - All but one same-color pair share one row or one column.
    - Exactly one same-color pair is split across different rows and columns.

    Rule:
    - Draw a straight inclusive segment for each aligned pair.
    - For the one unaligned pair, start from the higher marker, go horizontally to the other marker's column, then vertically to the lower marker.
    - Sort all paths by their cell counts from shortest to longest and paint them transparently on the input grid, so later paths never overwrite an already colored crossing cell.

    Key insight:
    - Crossing priority is determined only by total path length; the L path joins the same ordering using its full Manhattan bend length.
    """
    return _solve(grid)


def _sample_spaced_positions(
    size: int,
    count: int,
    *,
    margin: int = 2,
    min_gap: int = 2,
) -> list[int] | None:
    pool = list(range(margin, size - margin))
    if len(pool) < count:
        return None
    for _ in range(200):
        picks = sorted(random.sample(pool, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _sample_layout(cross_mode: str) -> tuple[int, int]:
    layouts = [(2, 1), (1, 2), (2, 2)]
    if cross_mode == "l_only":
        weights = (3, 3, 1)
    elif cross_mode == "mixed":
        weights = (2, 2, 3)
    else:
        weights = (2, 2, 2)
    return random.choices(layouts, weights=weights, k=1)[0]


def _gap_range_for_rank(rank: str) -> tuple[int, int]:
    return (2, 4) if rank == "short" else (5, 8)


def _axis_pairs(
    size: int,
    blocked: set[int],
    between_values: list[int],
    *,
    rank: str,
    requirement: str,
) -> list[tuple[int, int]]:
    gap_min, gap_max = _gap_range_for_rank(rank)
    available = [x for x in range(2, size - 2) if x not in blocked]
    pairs: list[tuple[int, int]] = []
    for lo, hi in combinations(available, 2):
        gap = hi - lo
        if not (gap_min <= gap <= gap_max):
            continue
        between = sum(lo < value < hi for value in between_values)
        if requirement == "none" and between != 0:
            continue
        if requirement == "some" and between == 0:
            continue
        pairs.append((lo, hi))
    random.shuffle(pairs)
    return pairs


def _straight_length_ok(length: int, l_length: int, rank: str) -> bool:
    if not (MIN_STRAIGHT_LEN <= length <= MAX_STRAIGHT_LEN):
        return False
    if rank == "short":
        return length > l_length
    return length < l_length


def _horizontal_candidates(
    width: int,
    row: int,
    v_cols: list[int],
    l_end_col: int,
    *,
    can_cross_l: bool,
    l_length: int,
    rank: str,
) -> list[tuple[int, int, int, int, int]]:
    reserved = set(v_cols) | {l_end_col}
    endpoints = [c for c in range(1, width - 1) if c not in reserved]
    candidates: list[tuple[int, int, int, int, int]] = []
    for c0, c1 in combinations(endpoints, 2):
        length = c1 - c0 + 1
        if not _straight_length_ok(length, l_length, rank):
            continue
        v_mask = 0
        for idx, col in enumerate(v_cols):
            if c0 < col < c1:
                v_mask |= 1 << idx
        cross_l = int(can_cross_l and c0 < l_end_col < c1)
        candidates.append((c0, c1, length, v_mask, cross_l))
    random.shuffle(candidates)
    return candidates


def _vertical_candidates(
    height: int,
    col: int,
    h_rows: list[int],
    l_top_row: int,
    l_bottom_row: int,
    *,
    can_cross_l: bool,
    l_length: int,
    rank: str,
) -> list[tuple[int, int, int, int, int]]:
    reserved = set(h_rows) | {l_top_row, l_bottom_row}
    endpoints = [r for r in range(1, height - 1) if r not in reserved]
    candidates: list[tuple[int, int, int, int, int]] = []
    for r0, r1 in combinations(endpoints, 2):
        length = r1 - r0 + 1
        if not _straight_length_ok(length, l_length, rank):
            continue
        h_mask = 0
        for idx, row in enumerate(h_rows):
            if r0 < row < r1:
                h_mask |= 1 << idx
        cross_l = int(can_cross_l and r0 < l_top_row < r1)
        candidates.append((r0, r1, length, h_mask, cross_l))
    random.shuffle(candidates)
    return candidates


def _cross_mode_ok(hv_cross_count: int, l_cross_count: int, cross_mode: str) -> bool:
    if cross_mode == "hv_only":
        return hv_cross_count >= 1 and l_cross_count == 0
    if cross_mode == "l_only":
        return hv_cross_count == 0 and l_cross_count >= 1
    return hv_cross_count >= 1 and l_cross_count >= 1


def _choose_candidates(
    h_options: list[list[tuple[int, int, int, int, int]]],
    v_options: list[list[tuple[int, int, int, int, int]]],
    *,
    l_length: int,
    cross_mode: str,
) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int, int, int, int]]] | None:
    if any(not options for options in h_options + v_options):
        return None

    for _ in range(SELECTION_TRIES):
        h_selected = [random.choice(options) for options in h_options]
        v_selected = [random.choice(options) for options in v_options]

        lengths = [cand[2] for cand in h_selected] + [cand[2] for cand in v_selected] + [l_length]
        if len(lengths) != len(set(lengths)):
            continue

        hv_cross_count = 0
        for h_idx, h_cand in enumerate(h_selected):
            for v_idx, v_cand in enumerate(v_selected):
                if ((h_cand[3] >> v_idx) & 1) and ((v_cand[3] >> h_idx) & 1):
                    hv_cross_count += 1

        l_cross_count = sum(cand[4] for cand in h_selected) + sum(cand[4] for cand in v_selected)
        total_crossings = hv_cross_count + l_cross_count
        if total_crossings == 0 or total_crossings > MAX_TOTAL_CROSSINGS:
            continue
        if not _cross_mode_ok(hv_cross_count, l_cross_count, cross_mode):
            continue
        return h_selected, v_selected

    return None


def _build_paths(
    colors: list[int],
    h_rows: list[int],
    h_selected: list[tuple[int, int, int, int, int]],
    v_cols: list[int],
    v_selected: list[tuple[int, int, int, int, int]],
    l_top_row: int,
    l_bottom_row: int,
    l_start_col: int,
    l_end_col: int,
) -> list[dict]:
    color_iter = iter(colors)
    paths: list[dict] = []

    for row, cand in zip(h_rows, h_selected):
        c0, c1 = cand[:2]
        endpoints = ((row, c0), (row, c1))
        cells = _straight_cells(*endpoints)
        paths.append(
            {
                "color": int(next(color_iter)),
                "kind": "straight",
                "endpoints": endpoints,
                "cells": cells,
                "length": len(cells),
            }
        )

    for col, cand in zip(v_cols, v_selected):
        r0, r1 = cand[:2]
        endpoints = ((r0, col), (r1, col))
        cells = _straight_cells(*endpoints)
        paths.append(
            {
                "color": int(next(color_iter)),
                "kind": "straight",
                "endpoints": endpoints,
                "cells": cells,
                "length": len(cells),
            }
        )

    l_endpoints = ((l_top_row, l_start_col), (l_bottom_row, l_end_col))
    l_cells = _l_cells(*l_endpoints)
    paths.append(
        {
            "color": int(next(color_iter)),
            "kind": "l",
            "endpoints": l_endpoints,
            "cells": l_cells,
            "length": len(l_cells),
        }
    )
    return paths


def _render_input(shape: tuple[int, int], paths: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for path in paths:
        color = int(path["color"])
        for r, c in path["endpoints"]:
            grid[r, c] = color
    return grid


def _count_and_validate_crossings(shape: tuple[int, int], paths: list[dict]) -> int:
    h, w = shape
    endpoint_list: list[tuple[int, int]] = []
    cell_sets: list[set[tuple[int, int]]] = []

    for path in paths:
        cells = list(path["cells"])
        cell_set = set(cells)
        if len(cells) != len(cell_set):
            return 0
        if len(cell_set) != int(path["length"]):
            return 0
        for r, c in cell_set:
            if not (0 <= r < h and 0 <= c < w):
                return 0
        for endpoint in path["endpoints"]:
            if endpoint not in cell_set:
                return 0
            endpoint_list.append(endpoint)
        cell_sets.append(cell_set)

    if len(endpoint_list) != len(set(endpoint_list)):
        return 0

    for a, b in combinations(endpoint_list, 2):
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
            return 0

    total_crossings = 0
    for i, path in enumerate(paths):
        own_endpoints = set(path["endpoints"])
        for j, other_cells in enumerate(cell_sets):
            if i == j:
                continue
            if own_endpoints & other_cells:
                return 0
        for j in range(i + 1, len(paths)):
            inter = cell_sets[i] & cell_sets[j]
            if not inter:
                continue
            if len(inter) != 1:
                return 0
            total_crossings += 1
    return total_crossings


def _semantic_validation(inp: np.ndarray, paths: list[dict]) -> bool:
    out = _render_overlay(inp, paths)
    if not np.array_equal(_solve(inp), out):
        return False

    saw_crossing = False
    for i, path_a in enumerate(paths):
        cells_a = set(path_a["cells"])
        for path_b in paths[i + 1 :]:
            inter = cells_a & set(path_b["cells"])
            if not inter:
                continue
            saw_crossing = True
            crossing = next(iter(inter))
            shorter = path_a if int(path_a["length"]) < int(path_b["length"]) else path_b
            if int(out[crossing]) != int(shorter["color"]):
                return False

    if not saw_crossing:
        return False

    reversed_out = _render_overlay(inp, paths, reverse=True)
    if np.array_equal(out, reversed_out):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    cross_mode = random.choice(("hv_only", "l_only", "mixed"))
    l_rank = random.choice(("short", "long"))
    h_count, v_count = _sample_layout(cross_mode)
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    h_rows = _sample_spaced_positions(height, h_count)
    v_cols = _sample_spaced_positions(width, v_count)
    if h_rows is None or v_cols is None:
        return None

    if cross_mode == "hv_only":
        axis_mode = "none"
    else:
        axis_mode = random.choice(("row", "col", "both"))

    row_requirement = "some" if axis_mode in {"row", "both"} else "none"
    col_requirement = "some" if axis_mode in {"col", "both"} else "none"

    row_pairs = _axis_pairs(height, set(h_rows), h_rows, rank=l_rank, requirement=row_requirement)
    col_pairs = _axis_pairs(width, set(v_cols), v_cols, rank=l_rank, requirement=col_requirement)
    if not row_pairs or not col_pairs:
        return None

    for _ in range(GEOMETRY_TRIES):
        l_top_row, l_bottom_row = random.choice(row_pairs)
        lo_col, hi_col = random.choice(col_pairs)
        if random.randrange(2):
            l_start_col, l_end_col = lo_col, hi_col
        else:
            l_start_col, l_end_col = hi_col, lo_col

        l_length = (l_bottom_row - l_top_row) + abs(l_end_col - l_start_col) + 1
        h_options = [
            _horizontal_candidates(
                width,
                row,
                v_cols,
                l_end_col,
                can_cross_l=l_top_row < row < l_bottom_row,
                l_length=l_length,
                rank=l_rank,
            )
            for row in h_rows
        ]
        v_options = [
            _vertical_candidates(
                height,
                col,
                h_rows,
                l_top_row,
                l_bottom_row,
                can_cross_l=min(l_start_col, l_end_col) < col < max(l_start_col, l_end_col),
                l_length=l_length,
                rank=l_rank,
            )
            for col in v_cols
        ]
        chosen = _choose_candidates(h_options, v_options, l_length=l_length, cross_mode=cross_mode)
        if chosen is None:
            continue

        colors = random.sample(PALETTE, h_count + v_count + 1)
        paths = _build_paths(
            colors,
            h_rows,
            chosen[0],
            v_cols,
            chosen[1],
            l_top_row,
            l_bottom_row,
            l_start_col,
            l_end_col,
        )
        crossing_count = _count_and_validate_crossings((height, width), paths)
        if not crossing_count:
            continue

        inp = _render_input((height, width), paths)
        if _semantic_validation(inp, paths):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid transparent-length-order line input")
