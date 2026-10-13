from __future__ import annotations

import random
from collections import deque

import numpy as np


BG_COLOR = 0
BAR_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 7, 8)
BAR_LENGTHS = (3, 5, 7, 9)
ORIENTATIONS = ("H", "V")

GEN_MIN_SIDE = 19
GEN_MAX_SIDE = 25
GEN_BAR_COUNT_MIN = 5
GEN_BAR_COUNT_MAX = 8
GEN_ATTEMPTS = 400
GROUP_ATTEMPTS = 140
EXTRA_ATTEMPTS = 180


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bar_cells(center: tuple[int, int], length: int, orient: str) -> list[tuple[int, int]]:
    cr, cc = center
    half = length // 2
    if orient == "H":
        return [(cr, cc + d) for d in range(-half, half + 1)]
    return [(cr + d, cc) for d in range(-half, half + 1)]


def _rotated_cells(center: tuple[int, int], length: int, orient: str) -> list[tuple[int, int]]:
    cr, cc = center
    half = length // 2
    if orient == "H":
        return [(cr + d, cc) for d in range(-half, half + 1)]
    return [(cr, cc + d) for d in range(-half, half + 1)]


def _marker_cell(spec: dict) -> tuple[int, int]:
    cr, cc = spec["center"]
    half = spec["length"] // 2
    side = spec["marker_side"]
    if spec["orient"] == "H":
        return (cr, cc - half - 1) if side < 0 else (cr, cc + half + 1)
    return (cr - half - 1, cc) if side < 0 else (cr + half + 1, cc)


def _extract_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape

    visited = np.zeros((h, w), dtype=bool)
    bars: list[dict] = []
    used_markers: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) != BAR_COLOR:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(g.shape, nr, nc)
                        and not visited[nr, nc]
                        and int(g[nr, nc]) == BAR_COLOR
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = sorted({rr for rr, _ in coords})
            cols = sorted({cc for _, cc in coords})
            if len(rows) == 1:
                orient = "H"
                row = rows[0]
                assert cols == list(range(cols[0], cols[-1] + 1))
                length = len(cols)
                center = (row, cols[length // 2])
                endpoints = ((row, cols[0]), (row, cols[-1]))
                marker_checks = ((row, cols[0] - 1), (row, cols[-1] + 1))
            else:
                assert len(cols) == 1
                orient = "V"
                col = cols[0]
                assert rows == list(range(rows[0], rows[-1] + 1))
                length = len(rows)
                center = (rows[length // 2], col)
                endpoints = ((rows[0], col), (rows[-1], col))
                marker_checks = ((rows[0] - 1, col), (rows[-1] + 1, col))

            assert length in BAR_LENGTHS and length % 2 == 1
            assert set(coords) == set(_bar_cells(center, length, orient))

            marker_candidates: list[tuple[tuple[int, int], int]] = []
            for mr, mc in marker_checks:
                if not _in_bounds(g.shape, mr, mc):
                    continue
                color = int(g[mr, mc])
                if color not in (BG_COLOR, BAR_COLOR):
                    marker_candidates.append(((mr, mc), color))
            assert len(marker_candidates) == 1

            (marker_pos, target_color) = marker_candidates[0]
            assert target_color in MARKER_COLORS
            assert marker_pos not in used_markers
            used_markers.add(marker_pos)

            bars.append(
                {
                    "center": center,
                    "length": length,
                    "orient": orient,
                    "endpoints": endpoints,
                    "marker": marker_pos,
                    "target_color": target_color,
                }
            )

    stray_markers = {
        tuple(map(int, rc))
        for rc in np.argwhere((g != BG_COLOR) & (g != BAR_COLOR))
    }
    assert stray_markers == used_markers
    assert bars
    return bars


def _render_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(specs, key=lambda spec: (-spec["length"], spec["center"][0], spec["center"][1]))
    for spec in ordered:
        for r, c in _rotated_cells(spec["center"], spec["length"], spec["orient"]):
            if out[r, c] == BG_COLOR:
                out[r, c] = spec["target_color"] if "target_color" in spec else spec["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each gray bar about its center, recolor it by its endpoint marker, and resolve overlaps by length.

    Input structure:
    - A black background contains several disjoint gray one-cell-thick bars of odd length.
    - Each gray bar is a straight horizontal or vertical 4-connected segment.
    - Exactly one non-black non-gray marker pixel sits one step beyond one endpoint of each bar and gives that bar's target color.

    Rule:
    - Identify every gray bar, its length, midpoint, and the marker color attached beyond one endpoint.
    - Clear the whole canvas to black, removing both the original bars and all marker pixels.
    - Rotate each bar by 90 degrees around its own center, so horizontal bars become vertical and vertical bars become horizontal while keeping the same length.
    - Paint the rotated bar entirely in its marker color.
    - If multiple rotated bars cover the same pixel, the longest bar wins; if several tied longest bars overlap there, use the tied bar whose original center is earliest in reading order.

    Key insight:
    - The final color of an overlap pixel depends on three linked facts from the input: which marker belongs to the bar, where the bar lands after center-fixed rotation, and how that bar's original length compares to the others.
    """
    bars = _extract_bars(np.asarray(grid, dtype=int))
    return _render_specs(np.asarray(grid).shape, bars)


def _can_place_spec(
    spec: dict,
    shape: tuple[int, int],
    gray_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
) -> bool:
    bar = _bar_cells(spec["center"], spec["length"], spec["orient"])
    marker = _marker_cell(spec)
    if not _in_bounds(shape, marker[0], marker[1]):
        return False
    if any(not _in_bounds(shape, r, c) for r, c in bar):
        return False

    occupied = gray_cells | marker_cells
    bar_set = set(bar)
    if marker in occupied or any(cell in occupied for cell in bar):
        return False

    for r, c in bar:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in gray_cells or (nr, nc) in marker_cells:
                return False

    if any((nr, nc) in gray_cells or (nr, nc) in marker_cells for nr, nc in _neighbors4(*marker)):
        return False

    own_neighbors = [nbr for nbr in _neighbors4(*marker) if nbr in bar_set]
    return len(own_neighbors) == 1


def _apply_spec(
    spec: dict,
    gray_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
) -> None:
    gray_cells.update(_bar_cells(spec["center"], spec["length"], spec["orient"]))
    marker_cells.add(_marker_cell(spec))


def _group_positions(axis_len: int, lengths: list[int], want_tie: bool) -> list[int] | None:
    halfs = [length // 2 for length in lengths]
    for _ in range(GROUP_ATTEMPTS):
        if want_tie:
            gap_hi = min(2 * halfs[0], 5)
            if gap_hi < 2:
                continue
            offsets = [0, random.randint(2, gap_hi)]
        else:
            gap1_hi = min(halfs[0] + halfs[1], 6)
            gap2_hi = min(halfs[1] + halfs[2], 6)
            if gap1_hi < 2 or gap2_hi < 2:
                continue
            gap1 = random.randint(2, gap1_hi)
            gap2 = random.randint(2, gap2_hi)
            offsets = [0, gap1, gap1 + gap2]

        base_min = max(halfs[i] - offsets[i] for i in range(len(lengths)))
        base_max = min(axis_len - 1 - halfs[i] - offsets[i] for i in range(len(lengths)))
        if base_min > base_max:
            continue

        base = random.randint(base_min, base_max)
        positions = [base + offset for offset in offsets]
        if all(positions[i + 1] - positions[i] >= 2 for i in range(len(positions) - 1)):
            return positions
    return None


def _fixed_axis_range(side_len: int, max_half: int) -> tuple[int, int]:
    return max_half + 1, side_len - max_half - 2


def _place_overlap_group(
    shape: tuple[int, int],
    orient: str,
    lengths: list[int],
    colors: list[int],
    gray_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
    want_tie: bool,
) -> list[dict] | None:
    axis_len = shape[0] if orient == "H" else shape[1]
    side_len = shape[1] if orient == "H" else shape[0]
    positions = _group_positions(axis_len, lengths, want_tie)
    if positions is None:
        return None

    fixed_lo, fixed_hi = _fixed_axis_range(side_len, max(lengths) // 2)
    if fixed_lo > fixed_hi:
        return None

    fixed_choices = list(range(fixed_lo, fixed_hi + 1))
    random.shuffle(fixed_choices)
    for fixed in fixed_choices[:GROUP_ATTEMPTS]:
        local_gray = set(gray_cells)
        local_markers = set(marker_cells)
        specs: list[dict] = []
        ok = True
        for pos, length, color in zip(positions, lengths, colors):
            center = (pos, fixed) if orient == "H" else (fixed, pos)
            spec = {
                "center": center,
                "length": length,
                "orient": orient,
                "marker_side": random.choice((-1, 1)),
                "color": color,
                "target_color": color,
            }
            if not _can_place_spec(spec, shape, local_gray, local_markers):
                ok = False
                break
            _apply_spec(spec, local_gray, local_markers)
            specs.append(spec)
        if ok:
            gray_cells.update(local_gray - gray_cells)
            marker_cells.update(local_markers - marker_cells)
            return specs
    return None


def _random_extra_spec(
    shape: tuple[int, int],
    gray_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
) -> dict | None:
    for _ in range(EXTRA_ATTEMPTS):
        orient = random.choice(ORIENTATIONS)
        length = random.choice(BAR_LENGTHS)
        side = random.choice((-1, 1))
        half = length // 2

        if orient == "H":
            row_lo, row_hi = half, shape[0] - 1 - half
            if side < 0:
                col_lo, col_hi = half + 1, shape[1] - 1 - half
            else:
                col_lo, col_hi = half, shape[1] - 2 - half
        else:
            col_lo, col_hi = half, shape[1] - 1 - half
            if side < 0:
                row_lo, row_hi = half + 1, shape[0] - 1 - half
            else:
                row_lo, row_hi = half, shape[0] - 2 - half

        if row_lo > row_hi or col_lo > col_hi:
            continue

        spec = {
            "center": (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)),
            "length": length,
            "orient": orient,
            "marker_side": side,
            "color": random.choice(MARKER_COLORS),
            "target_color": None,
        }
        spec["target_color"] = spec["color"]
        if _can_place_spec(spec, shape, gray_cells, marker_cells):
            _apply_spec(spec, gray_cells, marker_cells)
            return spec
    return None


def _build_grid(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in _bar_cells(spec["center"], spec["length"], spec["orient"]):
            grid[r, c] = BAR_COLOR
        mr, mc = _marker_cell(spec)
        grid[mr, mc] = spec["color"]
    return grid


def _overlap_details(specs: list[dict]) -> list[dict]:
    rotated_sets = [set(_rotated_cells(spec["center"], spec["length"], spec["orient"])) for spec in specs]
    pairs: list[dict] = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            inter = rotated_sets[i] & rotated_sets[j]
            if inter:
                pairs.append({"i": i, "j": j, "cells": inter})
    return pairs


def _winner_for_cell(specs: list[dict], cell: tuple[int, int]) -> dict:
    covering = [spec for spec in specs if cell in set(_rotated_cells(spec["center"], spec["length"], spec["orient"]))]
    max_length = max(spec["length"] for spec in covering)
    contenders = [spec for spec in covering if spec["length"] == max_length]
    return min(contenders, key=lambda spec: spec["center"])


def _validate_generated_case(grid: np.ndarray, specs: list[dict]) -> None:
    parsed = _extract_bars(grid)
    assert len(parsed) == len(specs)

    by_center = {spec["center"]: spec for spec in specs}
    for bar in parsed:
        spec = by_center[bar["center"]]
        assert bar["length"] == spec["length"]
        assert bar["orient"] == spec["orient"]
        assert bar["target_color"] == spec["color"]
        assert bar["marker"] == _marker_cell(spec)

    pairs = _overlap_details(specs)
    assert len(pairs) >= 2
    assert any(
        specs[pair["i"]]["length"] != specs[pair["j"]]["length"]
        and specs[pair["i"]]["color"] != specs[pair["j"]]["color"]
        for pair in pairs
    )
    assert any(
        specs[pair["i"]]["length"] == specs[pair["j"]]["length"]
        and specs[pair["i"]]["color"] != specs[pair["j"]]["color"]
        for pair in pairs
    )

    expected = _render_specs(grid.shape, specs)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    saw_length_conflict = False
    saw_tie_conflict = False
    for pair in pairs:
        cells = pair["cells"]
        left = specs[pair["i"]]
        right = specs[pair["j"]]
        if left["length"] != right["length"] and left["color"] != right["color"]:
            for cell in cells:
                winner = _winner_for_cell(specs, cell)
                if winner["length"] == max(left["length"], right["length"]):
                    saw_length_conflict = True
                    break
        if left["length"] == right["length"] and left["color"] != right["color"]:
            for cell in cells:
                winner = _winner_for_cell(specs, cell)
                earlier = min((left, right), key=lambda spec: spec["center"])
                if winner["center"] == earlier["center"] and expected[cell] == earlier["color"]:
                    saw_tie_conflict = True
                    break
    assert saw_length_conflict
    assert saw_tie_conflict


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE), random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE))
    gray_cells: set[tuple[int, int]] = set()
    marker_cells: set[tuple[int, int]] = set()
    specs: list[dict] = []

    diff_orient = random.choice(ORIENTATIONS)
    tie_orient = random.choice(ORIENTATIONS)

    diff_lengths = random.sample(BAR_LENGTHS, 3)
    diff_colors = random.sample(MARKER_COLORS, 2) + [random.choice(MARKER_COLORS)]
    diff_specs = _place_overlap_group(
        shape,
        diff_orient,
        diff_lengths,
        diff_colors,
        gray_cells,
        marker_cells,
        want_tie=False,
    )
    if diff_specs is None:
        return None
    specs.extend(diff_specs)

    tie_length = random.choice((3, 5, 7))
    tie_colors = random.sample(MARKER_COLORS, 2)
    tie_specs = _place_overlap_group(
        shape,
        tie_orient,
        [tie_length, tie_length],
        tie_colors,
        gray_cells,
        marker_cells,
        want_tie=True,
    )
    if tie_specs is None:
        return None
    specs.extend(tie_specs)

    total_bars = random.randint(GEN_BAR_COUNT_MIN, GEN_BAR_COUNT_MAX)
    while len(specs) < total_bars:
        extra = _random_extra_spec(shape, gray_cells, marker_cells)
        if extra is None:
            return None
        specs.append(extra)

    random.shuffle(specs)
    grid = _build_grid(shape, specs)
    try:
        _validate_generated_case(grid, specs)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bar-rotation recoloring task")
