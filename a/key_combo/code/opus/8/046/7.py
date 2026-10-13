from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, get_bg


BG = Color.BLACK
BAR = Color.RED
SPRITE = Color.GREEN
LINK = Color.TEAL

GRID_MIN = 17
GRID_MAX = 22
BAR_COUNT_OPTIONS = (3, 4)
GAP_RANGE = range(2, 9)
BAR_LENGTH_RANGE = range(3, 7)
GEN_ATTEMPTS = 800

SHAPES = [
    {(-1, 0), (0, 0), (1, 0)},
    {(0, -1), (0, 0), (0, 1)},
    {(-1, 0), (0, 0), (1, 0), (0, 1)},
    {(-1, 0), (0, 0), (1, 0), (0, -1)},
    {(0, -1), (0, 0), (0, 1), (1, 0)},
    {(0, -1), (0, 0), (0, 1), (-1, 0)},
    {(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1)},
    {(-1, 0), (0, 0), (1, 0), (1, -1), (0, 1)},
    {(-1, 0), (0, 0), (0, -1), (0, 1), (1, 1)},
]


def _rotate(shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(-c, r) for r, c in shape}


def _mirror(shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, -c) for r, c in shape}


def _random_shape() -> set[tuple[int, int]]:
    shape = set(random.choice(SHAPES))
    for _ in range(random.randrange(4)):
        shape = _rotate(shape)
    if random.random() < 0.5:
        shape = _mirror(shape)
    return shape


def _shape_bounds(shape: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return min(rows), min(cols), max(rows), max(cols)


def _cells_from_center(center: tuple[int, int], shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in shape}


def _center_from_coords(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox_coords(coords)
    return (r0 + r1) // 2, (c0 + c1) // 2


def _edge_span_on_row(coords: np.ndarray, row: int) -> tuple[int, int] | None:
    cols = [int(c) for r, c in coords if int(r) == int(row)]
    if not cols:
        return None
    return min(cols), max(cols)


def _edge_span_on_col(coords: np.ndarray, col: int) -> tuple[int, int] | None:
    rows = [int(r) for r, c in coords if int(c) == int(col)]
    if not rows:
        return None
    return min(rows), max(rows)


def _translate_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.asarray(coords, dtype=int).copy()
    out[:, 0] += int(dr)
    out[:, 1] += int(dc)
    return out


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _bar_info(coords: np.ndarray) -> dict:
    coords = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    if c0 == c1:
        rows = sorted(int(r) for r in coords[:, 0])
        assert rows == list(range(r0, r1 + 1))
        return {
            "orientation": "v",
            "fixed": int(c0),
            "start": int(r0),
            "end": int(r1),
            "coords": coords,
        }
    assert r0 == r1
    cols = sorted(int(c) for c in coords[:, 1])
    assert cols == list(range(c0, c1 + 1))
    return {
        "orientation": "h",
        "fixed": int(r0),
        "start": int(c0),
        "end": int(c1),
        "coords": coords,
    }


def _candidate_slide(sprite_coords: np.ndarray, bar: dict) -> tuple[int, tuple[int, int]] | None:
    rows = set(int(r) for r in sprite_coords[:, 0])
    cols = set(int(c) for c in sprite_coords[:, 1])
    min_r = int(sprite_coords[:, 0].min())
    max_r = int(sprite_coords[:, 0].max())
    min_c = int(sprite_coords[:, 1].min())
    max_c = int(sprite_coords[:, 1].max())

    if bar["orientation"] == "v":
        bar_rows = set(range(int(bar["start"]), int(bar["end"]) + 1))
        if not (rows & bar_rows):
            return None
        fixed = int(bar["fixed"])
        if max_c < fixed:
            dist = fixed - max_c - 1
            return (dist, (0, 1)) if dist > 0 else None
        if min_c > fixed:
            dist = min_c - fixed - 1
            return (dist, (0, -1)) if dist > 0 else None
        return None

    bar_cols = set(range(int(bar["start"]), int(bar["end"]) + 1))
    if not (cols & bar_cols):
        return None
    fixed = int(bar["fixed"])
    if max_r < fixed:
        dist = fixed - max_r - 1
        return (dist, (1, 0)) if dist > 0 else None
    if min_r > fixed:
        dist = min_r - fixed - 1
        return (dist, (-1, 0)) if dist > 0 else None
    return None


def _solve_assignments(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    bar_components = find_components(
        g,
        bg=bg,
        colors=[BAR],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    sprite_components = find_components(
        g,
        bg=bg,
        colors=[SPRITE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(bar_components) == len(sprite_components)
    assert len(bar_components) in BAR_COUNT_OPTIONS

    bars = [_bar_info(component["coords"]) for component in bar_components]
    assignments = []
    for component in sprite_components:
        coords = np.asarray(component["coords"], dtype=int)
        candidates = []
        for idx, bar in enumerate(bars):
            candidate = _candidate_slide(coords, bar)
            if candidate is None:
                continue
            dist, direction = candidate
            candidates.append((int(dist), int(idx), tuple(direction)))

        assert candidates, "each sprite must have a reachable bar"
        candidates.sort(key=lambda item: (item[0], item[1]))
        assert len(candidates) == 1 or candidates[0][0] < candidates[1][0], "nearest bar must be unique"
        dist, bar_index, direction = candidates[0]
        assignments.append(
            {
                "start_coords": coords,
                "bar_index": int(bar_index),
                "distance": int(dist),
                "direction": tuple(direction),
                "end_coords": _translate_coords(coords, direction[0] * dist, direction[1] * dist),
            }
        )

    assert len({item["bar_index"] for item in assignments}) == len(assignments), "bar pairing must be one-to-one"
    return bars, assignments


def _settled_base_from_assignments(grid: np.ndarray, bars: list[dict], assignments: list[dict]) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    out[out == SPRITE] = bg

    settled = []
    for item in assignments:
        end_coords = np.asarray(item["end_coords"], dtype=int)
        _paint_coords(out, end_coords, SPRITE)
        settled.append(
            {
                "coords": end_coords,
                "center": _center_from_coords(end_coords),
                "bar_index": int(item["bar_index"]),
                "direction": tuple(item["direction"]),
            }
        )

    return out, settled


def _apply_connectors(out: np.ndarray, base: np.ndarray, settled: list[dict], bg: int) -> set[tuple[int, int]]:
    connectors: set[tuple[int, int]] = set()
    by_row: dict[int, list[dict]] = defaultdict(list)
    by_col: dict[int, list[dict]] = defaultdict(list)
    for item in settled:
        row, col = item["center"]
        by_row[int(row)].append(item)
        by_col[int(col)].append(item)

    for row, items in by_row.items():
        ordered = sorted(items, key=lambda item: item["center"][1])
        for left, right in zip(ordered, ordered[1:]):
            left_span = _edge_span_on_row(left["coords"], row)
            right_span = _edge_span_on_row(right["coords"], row)
            if left_span is None or right_span is None:
                continue
            start = int(left_span[1]) + 1
            end = int(right_span[0]) - 1
            if start > end:
                continue
            if np.all(base[row, start : end + 1] == bg):
                for col in range(start, end + 1):
                    connectors.add((int(row), int(col)))

    for col, items in by_col.items():
        ordered = sorted(items, key=lambda item: item["center"][0])
        for top, bottom in zip(ordered, ordered[1:]):
            top_span = _edge_span_on_col(top["coords"], col)
            bottom_span = _edge_span_on_col(bottom["coords"], col)
            if top_span is None or bottom_span is None:
                continue
            start = int(top_span[1]) + 1
            end = int(bottom_span[0]) - 1
            if start > end:
                continue
            if np.all(base[start : end + 1, col] == bg):
                for row in range(start, end + 1):
                    connectors.add((int(row), int(col)))

    for r, c in connectors:
        out[r, c] = LINK
    return connectors


def _apply_bar_copies(out: np.ndarray, bars: list[dict], settled: list[dict]) -> set[tuple[int, int]]:
    copies: set[tuple[int, int]] = set()
    by_bar = {int(item["bar_index"]): item for item in settled}
    for idx, bar in enumerate(bars):
        item = by_bar[int(idx)]
        dr, dc = item["direction"]
        shift = dr + dc
        assert abs(shift) == 1
        if bar["orientation"] == "v":
            copy_col = int(bar["fixed"]) + int(dc)
            for row in range(int(bar["start"]), int(bar["end"]) + 1):
                copies.add((int(row), int(copy_col)))
        else:
            copy_row = int(bar["fixed"]) + int(dr)
            for col in range(int(bar["start"]), int(bar["end"]) + 1):
                copies.add((int(copy_row), int(col)))
    for r, c in copies:
        out[r, c] = LINK
    return copies


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each green sprite onto its red bar, connect newly aligned stopped sprites with teal segments, and mirror each bar with a teal copy on the far side.

    Input structure:
    - The grid is black background with 3-4 red straight bars and the same number of green 4-connected sprites.
    - Each green sprite is offset from exactly one red bar along a single cardinal axis with a unique nearest-bar displacement.
    - Bars are horizontal or vertical line segments; sprites are small irregular foreground objects.

    Rule:
    - For every green sprite, find the unique shortest orthogonal translation that makes it stop with edge contact against a red bar, and move the whole sprite there.
    - After all sprites stop, group sprite bbox-centers by shared row and by shared column; among adjacent centers in each sorted group, fill the all-background gap between the two sprites with teal if that row/column corridor is unobstructed.
    - For each red bar, place a teal copy of the same bar shape on the side opposite the touching sprite, directly adjacent to the red bar.

    Key insight:
    - The alignment anchors do not exist in the input: only after resolving every slide can you see which stopped sprites newly share a row or column and therefore deserve a teal connector.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    bars, assignments = _solve_assignments(g)
    base, settled = _settled_base_from_assignments(g, bars, assignments)
    out = base.copy()
    _apply_connectors(out, base, settled, bg)
    _apply_bar_copies(out, bars, settled)
    return out


def _feasible_moves(
    center: tuple[int, int],
    shape: set[tuple[int, int]],
    grid_shape: tuple[int, int],
    *,
    axis: str,
) -> list[tuple[tuple[int, int], int]]:
    height, width = grid_shape
    row, col = center
    min_r, min_c, max_r, max_c = _shape_bounds(shape)
    options = []

    if axis == "v":
        if row + max_r + 2 < height:
            top_room = row + min_r
            for gap in GAP_RANGE:
                if gap <= top_room:
                    options.append(((1, 0), int(gap)))
        if row + min_r - 2 >= 0:
            bottom_room = (height - 1) - (row + max_r)
            for gap in GAP_RANGE:
                if gap <= bottom_room:
                    options.append(((-1, 0), int(gap)))
        return options

    if col + max_c + 2 < width:
        left_room = col + min_c
        for gap in GAP_RANGE:
            if gap <= left_room:
                options.append(((0, 1), int(gap)))
    if col + min_c - 2 >= 0:
        right_room = (width - 1) - (col + max_c)
        for gap in GAP_RANGE:
            if gap <= right_room:
                options.append(((0, -1), int(gap)))
    return options


def _sample_bar_cells(
    settled_cells: set[tuple[int, int]],
    center: tuple[int, int],
    direction: tuple[int, int],
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    height, width = grid_shape
    rows = [r for r, _ in settled_cells]
    cols = [c for _, c in settled_cells]
    dr, dc = direction

    if dr != 0:
        bar_row = max(rows) + 1 if dr > 0 else min(rows) - 1
        copy_row = bar_row + dr
        if not (0 <= bar_row < height and 0 <= copy_row < height):
            return None
        support_cols = sorted({c for _, c in settled_cells})
        anchor = random.choice(support_cols + [center[1]])
        options = []
        for length in BAR_LENGTH_RANGE:
            for start in range(anchor - length + 1, anchor + 1):
                end = start + length - 1
                if 0 <= start and end < width and any(start <= col <= end for col in support_cols):
                    options.append((start, end))
        if not options:
            return None
        start, end = random.choice(options)
        return {(int(bar_row), c) for c in range(int(start), int(end) + 1)}

    bar_col = max(cols) + 1 if dc > 0 else min(cols) - 1
    copy_col = bar_col + dc
    if not (0 <= bar_col < width and 0 <= copy_col < width):
        return None
    support_rows = sorted({r for r, _ in settled_cells})
    anchor = random.choice(support_rows + [center[0]])
    options = []
    for length in BAR_LENGTH_RANGE:
        for start in range(anchor - length + 1, anchor + 1):
            end = start + length - 1
            if 0 <= start and end < height and any(start <= row <= end for row in support_rows):
                options.append((start, end))
    if not options:
        return None
    start, end = random.choice(options)
    return {(r, int(bar_col)) for r in range(int(start), int(end) + 1)}


def _render_specs(
    grid_shape: tuple[int, int],
    specs: list[dict],
    *,
    use_input_positions: bool,
) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for spec in specs:
        for r, c in spec["bar_cells"]:
            grid[r, c] = BAR
    for spec in specs:
        cell_key = "input_cells" if use_input_positions else "settled_cells"
        for r, c in spec[cell_key]:
            assert grid[r, c] == BG
            grid[r, c] = SPRITE
    return grid


def _center_tuple_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)


def _has_input_alignment(specs: list[dict]) -> bool:
    centers = [_center_tuple_from_cells(spec["input_cells"]) for spec in specs]
    rows = [r for r, _ in centers]
    cols = [c for _, c in centers]
    return len(rows) != len(set(rows)) or len(cols) != len(set(cols))


def _line_connector_pairs(grid: np.ndarray, settled: list[dict]) -> tuple[int, int]:
    bg = int(get_bg(grid))
    probe = grid.copy()
    connectors = _apply_connectors(probe, grid, settled, bg)
    rows = defaultdict(list)
    cols = defaultdict(list)
    for item in settled:
        r, c = item["center"]
        rows[int(r)].append(item)
        cols[int(c)].append(item)
    aligned_groups = sum(1 for items in rows.values() if len(items) >= 2) + sum(1 for items in cols.values() if len(items) >= 2)
    return len(connectors), aligned_groups


def _pairwise_unaligned_count(centers: list[tuple[int, int]]) -> int:
    count = 0
    for i, (r0, c0) in enumerate(centers):
        for r1, c1 in centers[i + 1 :]:
            if r0 != r1 and c0 != c1:
                count += 1
    return count


def _components_count(grid: np.ndarray, color: int) -> int:
    comps = find_components(
        grid,
        bg=BG,
        colors=[color],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    return len(comps)


def _intended_output(grid_shape: tuple[int, int], specs: list[dict]) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    base = _render_specs(grid_shape, specs, use_input_positions=False)
    settled = []
    for idx, spec in enumerate(specs):
        coords = np.array(sorted(spec["settled_cells"]), dtype=int)
        settled.append(
            {
                "coords": coords,
                "center": _center_from_coords(coords),
                "bar_index": idx,
                "direction": spec["direction"],
            }
        )
    out = base.copy()
    connectors = _apply_connectors(out, base, settled, BG)
    bars = [_bar_info(np.array(sorted(spec["bar_cells"]), dtype=int)) for spec in specs]
    copies = _apply_bar_copies(out, bars, settled)
    return out, connectors, copies


def _paths_clear(specs: list[dict], grid_shape: tuple[int, int]) -> bool:
    occupied_bars = set().union(*(spec["bar_cells"] for spec in specs))
    settled_occupied = [spec["settled_cells"] for spec in specs]
    input_occupied = [spec["input_cells"] for spec in specs]
    for idx, spec in enumerate(specs):
        dr, dc = spec["direction"]
        gap = int(spec["gap"])
        for step in range(gap + 1):
            translated = {(r + dr * step, c + dc * step) for r, c in spec["input_cells"]}
            if any(not (0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]) for r, c in translated):
                return False
            for jdx, other in enumerate(specs):
                if idx == jdx:
                    continue
                if translated & other["bar_cells"]:
                    return False
                if step < gap and translated & input_occupied[jdx]:
                    return False
                if step > 0 and translated & settled_occupied[jdx]:
                    return False
            if translated & (occupied_bars - spec["bar_cells"]):
                return False
    return True


def _specs_valid(specs: list[dict], grid_shape: tuple[int, int]) -> bool:
    input_cells = [spec["input_cells"] for spec in specs]
    settled_cells = [spec["settled_cells"] for spec in specs]
    bar_cells = [spec["bar_cells"] for spec in specs]

    def disjoint(groups: list[set[tuple[int, int]]]) -> bool:
        seen: set[tuple[int, int]] = set()
        for group in groups:
            if seen & group:
                return False
            seen |= group
        return True

    if not disjoint(input_cells + bar_cells):
        return False
    if not disjoint(settled_cells + bar_cells):
        return False
    if _has_input_alignment(specs):
        return False
    if not _paths_clear(specs, grid_shape):
        return False

    intended, connectors, copies = _intended_output(grid_shape, specs)
    if connectors & copies:
        return False

    input_grid = _render_specs(grid_shape, specs, use_input_positions=True)
    if _components_count(input_grid, BAR) != len(specs):
        return False
    if _components_count(input_grid, SPRITE) != len(specs):
        return False
    if _components_count(_render_specs(grid_shape, specs, use_input_positions=False), SPRITE) != len(specs):
        return False
    if np.array_equal(input_grid, intended):
        return False

    try:
        bars, assignments = _solve_assignments(input_grid)
    except AssertionError:
        return False
    base, settled = _settled_base_from_assignments(input_grid, bars, assignments)
    settled_centers = sorted(item["center"] for item in settled)
    target_centers = sorted(_center_tuple_from_cells(spec["settled_cells"]) for spec in specs)
    if settled_centers != target_centers:
        return False
    if _pairwise_unaligned_count(target_centers) == 0:
        return False

    connector_count, aligned_groups = _line_connector_pairs(base, settled)
    if connector_count == 0 or aligned_groups == 0:
        return False

    for item in assignments:
        idx = int(item["bar_index"])
        if tuple(item["direction"]) != tuple(specs[idx]["direction"]):
            return False
        if int(item["distance"]) != int(specs[idx]["gap"]):
            return False

    actual = main(input_grid)
    return np.array_equal(actual, intended)


def _pick_unique(values: list[int], count: int, *, min_gap: int = 0) -> list[int] | None:
    choices = values[:]
    random.shuffle(choices)
    selected = []
    for value in choices:
        if all(abs(value - prev) >= min_gap for prev in selected):
            selected.append(value)
            if len(selected) == count:
                return sorted(selected)
    return None


def _sample_row_mode(grid_shape: tuple[int, int], count: int) -> list[dict] | None:
    height, width = grid_shape
    row_values = list(range(4, height - 4))
    col_values = list(range(4, width - 4))
    pair_cols = _pick_unique(col_values, 2, min_gap=5)
    if pair_cols is None:
        return None
    align_row = random.choice(row_values)

    centers = [(align_row, pair_cols[0]), (align_row, pair_cols[1])]
    axes = ["v", "v"]
    while len(centers) < count:
        row = random.choice(row_values)
        col = random.choice(col_values)
        if row == align_row or col in pair_cols:
            continue
        if any((row, col) == item for item in centers):
            continue
        centers.append((row, col))
        axes.append(random.choice(["h", "v"]))

    return _build_specs(grid_shape, centers, axes)


def _sample_col_mode(grid_shape: tuple[int, int], count: int) -> list[dict] | None:
    height, width = grid_shape
    row_values = list(range(4, height - 4))
    col_values = list(range(4, width - 4))
    pair_rows = _pick_unique(row_values, 2, min_gap=5)
    if pair_rows is None:
        return None
    align_col = random.choice(col_values)

    centers = [(pair_rows[0], align_col), (pair_rows[1], align_col)]
    axes = ["h", "h"]
    while len(centers) < count:
        row = random.choice(row_values)
        col = random.choice(col_values)
        if col == align_col or row in pair_rows:
            continue
        if any((row, col) == item for item in centers):
            continue
        centers.append((row, col))
        axes.append(random.choice(["h", "v"]))

    return _build_specs(grid_shape, centers, axes)


def _sample_both_mode(grid_shape: tuple[int, int]) -> list[dict] | None:
    height, width = grid_shape
    row_values = list(range(4, height - 4))
    col_values = list(range(4, width - 4))
    pair_cols = _pick_unique(col_values, 2, min_gap=5)
    pair_rows = _pick_unique(row_values, 2, min_gap=5)
    if pair_cols is None or pair_rows is None:
        return None

    align_row = random.choice([r for r in row_values if r not in pair_rows])
    align_col = random.choice([c for c in col_values if c not in pair_cols])
    centers = [
        (align_row, pair_cols[0]),
        (align_row, pair_cols[1]),
        (pair_rows[0], align_col),
        (pair_rows[1], align_col),
    ]
    if len(set(centers)) != 4:
        return None
    axes = ["v", "v", "h", "h"]
    return _build_specs(grid_shape, centers, axes)


def _build_specs(grid_shape: tuple[int, int], centers: list[tuple[int, int]], axes: list[str]) -> list[dict] | None:
    specs = []
    for center, axis in zip(centers, axes):
        shape = _random_shape()
        options = _feasible_moves(center, shape, grid_shape, axis=axis)
        if not options:
            return None
        direction, gap = random.choice(options)
        settled_cells = _cells_from_center(center, shape)
        input_cells = {(r - direction[0] * gap, c - direction[1] * gap) for r, c in settled_cells}
        bar_cells = _sample_bar_cells(settled_cells, center, direction, grid_shape)
        if bar_cells is None:
            return None
        specs.append(
            {
                "shape": shape,
                "center": center,
                "axis": axis,
                "direction": tuple(direction),
                "gap": int(gap),
                "settled_cells": settled_cells,
                "input_cells": input_cells,
                "bar_cells": bar_cells,
            }
        )

    return specs if _specs_valid(specs, grid_shape) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid_shape = (
            random.randint(GRID_MIN, GRID_MAX),
            random.randint(GRID_MIN, GRID_MAX),
        )
        mode = random.choice(["row", "col", "both"])
        if mode == "both":
            specs = _sample_both_mode(grid_shape)
        elif mode == "row":
            specs = _sample_row_mode(grid_shape, random.choice(BAR_COUNT_OPTIONS))
        else:
            specs = _sample_col_mode(grid_shape, random.choice(BAR_COUNT_OPTIONS))

        if specs is None:
            continue
        return _render_specs(grid_shape, specs, use_input_positions=True)

    raise RuntimeError("failed to generate a valid puzzle instance")
