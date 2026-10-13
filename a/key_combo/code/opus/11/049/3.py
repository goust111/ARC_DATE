from __future__ import annotations

import itertools
import random

import numpy as np


_BG = 0
_RED = 2
_MARKER_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
_GRID_H_RANGE = (10, 13)
_GRID_W_RANGE = (10, 13)
_MARKER_COUNT = 3
_MIN_GAP = 2
_MAX_ATTEMPTS = 400


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_markers(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background(g)
    coords = np.argwhere(g != bg)
    markers: list[dict[str, int]] = []
    for idx, (r_raw, c_raw) in enumerate(coords.tolist()):
        r = int(r_raw)
        c = int(c_raw)
        color = int(g[r, c])
        markers.append({"id": idx, "r": r, "c": c, "color": color})

    assert len(markers) >= 2, "expected at least two markers"
    rows = [marker["r"] for marker in markers]
    cols = [marker["c"] for marker in markers]
    colors = [marker["color"] for marker in markers]
    assert len(set(rows)) == len(rows), "marker rows must be distinct"
    assert len(set(cols)) == len(cols), "marker columns must be distinct"
    assert len(set(colors)) == len(colors), "marker colors must be distinct"
    assert _RED not in colors, "red is reserved for ray crossings"
    assert all(0 < r < g.shape[0] - 1 for r in rows), "markers must stay off the top/bottom edge"
    assert all(0 < c < g.shape[1] - 1 for c in cols), "markers must stay off the left/right edge"
    return bg, markers


def _render_rays(shape: tuple[int, int], bg: int, markers: list[dict[str, int]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    marker_positions = {(marker["r"], marker["c"]) for marker in markers}

    for marker in markers:
        out[marker["r"], :] = marker["color"]
        out[:, marker["c"]] = marker["color"]

    for row_marker in markers:
        for col_marker in markers:
            if row_marker["id"] == col_marker["id"]:
                continue
            pos = (row_marker["r"], col_marker["c"])
            if pos not in marker_positions:
                out[pos] = _RED

    for marker in markers:
        out[marker["r"], marker["c"]] = marker["color"]
    return out


def _region_specs(
    shape: tuple[int, int], markers: list[dict[str, int]]
) -> list[dict[str, object]]:
    h, w = map(int, shape)
    by_row = {marker["r"]: marker for marker in markers}
    by_col = {marker["c"]: marker for marker in markers}
    row_bounds = [-1] + sorted(by_row) + [h]
    col_bounds = [-1] + sorted(by_col) + [w]
    regions: list[dict[str, object]] = []

    for top, bottom in zip(row_bounds, row_bounds[1:]):
        r0 = top + 1
        r1 = bottom - 1
        if r0 > r1:
            continue
        for left, right in zip(col_bounds, col_bounds[1:]):
            c0 = left + 1
            c1 = right - 1
            if c0 > c1:
                continue

            row_ids: list[int] = []
            col_ids: list[int] = []
            marker_ids: list[int] = []
            for boundary_row in (top, bottom):
                if boundary_row in by_row:
                    marker = by_row[boundary_row]
                    row_ids.append(marker["id"])
                    marker_ids.append(marker["id"])
            for boundary_col in (left, right):
                if boundary_col in by_col:
                    marker = by_col[boundary_col]
                    col_ids.append(marker["id"])
                    marker_ids.append(marker["id"])

            uniq_ids = tuple(sorted(set(marker_ids)))
            assert uniq_ids, "every rectangle must be bounded by at least one marker"
            regions.append(
                {
                    "rows": (r0, r1),
                    "cols": (c0, c1),
                    "row_ids": tuple(sorted(set(row_ids))),
                    "col_ids": tuple(sorted(set(col_ids))),
                    "marker_ids": uniq_ids,
                }
            )

    return regions


def _filled_output(
    shape: tuple[int, int],
    bg: int,
    markers: list[dict[str, int]],
    chooser,
) -> np.ndarray:
    out = _render_rays(shape, bg, markers)
    by_id = {marker["id"]: marker for marker in markers}

    for region in _region_specs(shape, markers):
        winner_id = int(chooser(region["marker_ids"], by_id))
        winner_color = int(by_id[winner_id]["color"])
        r0, r1 = region["rows"]
        c0, c1 = region["cols"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == bg] = winner_color
    return out


def _default_chooser(marker_ids: tuple[int, ...], by_id: dict[int, dict[str, int]]) -> int:
    return max(marker_ids, key=lambda marker_id: by_id[marker_id]["r"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marker into cross-shaped rays, mark ray crossings red, then fill each rectangular compartment by the lowest source-row priority.

    Input structure:
    - The background is black.
    - Exactly three isolated non-background marker pixels appear on distinct interior rows and distinct interior columns.
    - The marker colors are distinct and red is not used by the markers.

    Rule:
    - Each marker emits four straight rays of its own color along its full row and full column until the grid boundary.
    - Crossings between rays from different markers become red, while the original marker pixels keep their own colors.
    - The completed rays partition the grid into axis-aligned rectangular regions whose sides are made of rays and/or the outer grid boundary.
    - Fill every such region without covering the rays; the fill color is the color of the boundary marker whose source row index is greatest among all markers contributing a side of that region.

    Key insight:
    - Region color is not chosen by geometry size, by color id, or by the nearest side; it is chosen by comparing the source rows of all ray owners that bound the region.
    """
    bg, markers = _extract_markers(grid)
    return _filled_output(np.asarray(grid).shape, bg, markers, _default_chooser)


def _spaced_positions(length: int, count: int, min_gap: int) -> tuple[int, ...]:
    choices = [
        combo
        for combo in itertools.combinations(range(1, length - 1), count)
        if all(b - a >= min_gap for a, b in zip(combo, combo[1:]))
    ]
    assert choices, f"no valid spaced positions for length={length}"
    return random.choice(choices)


def _winner_ids(shape: tuple[int, int], markers: list[dict[str, int]]) -> set[int]:
    by_id = {marker["id"]: marker for marker in markers}
    return {
        max(region["marker_ids"], key=lambda marker_id: by_id[marker_id]["r"])
        for region in _region_specs(shape, markers)
    }


def _check_non_degenerate(inp: np.ndarray, out: np.ndarray, markers: list[dict[str, int]]) -> bool:
    h, w = inp.shape
    regions = _region_specs((h, w), markers)
    by_id = {marker["id"]: marker for marker in markers}
    all_ids = {marker["id"] for marker in markers}

    if len(regions) < 12:
        return False
    if _winner_ids((h, w), markers) != all_ids:
        return False
    if not any(len(region["marker_ids"]) == 1 for region in regions):
        return False
    if not any(len(region["marker_ids"]) >= 2 for region in regions):
        return False
    if not any(
        len(region["col_ids"]) > 0
        and max(region["marker_ids"], key=lambda marker_id: by_id[marker_id]["r"]) in region["col_ids"]
        and max(region["marker_ids"], key=lambda marker_id: by_id[marker_id]["r"]) not in region["row_ids"]
        for region in regions
    ):
        return False

    line_mask = np.zeros_like(inp, dtype=bool)
    for marker in markers:
        line_mask[marker["r"], :] = True
        line_mask[:, marker["c"]] = True
    fill_mask = (~line_mask) & (out != _BG)
    if np.count_nonzero(fill_mask) < max(10, (h * w) // 5):
        return False

    red_crossings = int(np.count_nonzero(out == _RED))
    if red_crossings != len(markers) * (len(markers) - 1):
        return False

    alt_min_row = _filled_output(
        inp.shape, _BG, markers, lambda ids, id_map: min(ids, key=lambda marker_id: id_map[marker_id]["r"])
    )
    alt_max_col = _filled_output(
        inp.shape, _BG, markers, lambda ids, id_map: max(ids, key=lambda marker_id: id_map[marker_id]["c"])
    )
    alt_global_bottom = _filled_output(
        inp.shape,
        _BG,
        markers,
        lambda ids, id_map: max(id_map, key=lambda marker_id: id_map[marker_id]["r"]),
    )
    if np.array_equal(out, alt_min_row):
        return False
    if np.array_equal(out, alt_max_col):
        return False
    if np.array_equal(out, alt_global_bottom):
        return False
    return True


def _attempt_input() -> np.ndarray | None:
    h = random.randint(*_GRID_H_RANGE)
    w = random.randint(*_GRID_W_RANGE)
    rows = list(_spaced_positions(h, _MARKER_COUNT, _MIN_GAP))
    cols = list(_spaced_positions(w, _MARKER_COUNT, _MIN_GAP))
    colors = random.sample(_MARKER_COLORS, _MARKER_COUNT)
    perm = random.sample(range(_MARKER_COUNT), _MARKER_COUNT)

    grid = np.zeros((h, w), dtype=int)
    markers: list[dict[str, int]] = []
    for idx, row in enumerate(rows):
        col = cols[perm[idx]]
        color = colors[idx]
        grid[row, col] = color
        markers.append({"id": idx, "r": row, "c": col, "color": color})

    out = main(grid)
    expected = _filled_output(grid.shape, _BG, markers, _default_chooser)
    if not np.array_equal(out, expected):
        return None
    if not _check_non_degenerate(grid, out, markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-fill arbitration input")
