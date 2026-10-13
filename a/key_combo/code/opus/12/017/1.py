from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]

_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
_GEN_RETRY_LIMIT = 400
_MIN_OVERLAP = 2


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _line_cells(marker: Coord, trail_dir: Coord, length: int) -> list[Coord]:
    r0, c0 = marker
    dr, dc = trail_dir
    return [(r0 + step * dr, c0 + step * dc) for step in range(length + 1)]


def _extract_sources(grid: np.ndarray, bg: int) -> list[dict]:
    sources: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert 2 <= len(coords) <= 5, "each source must be marker plus 1-4 trail cells"

        rows = sorted({r for r, _c in coords})
        cols = sorted({c for _r, c in coords})
        if len(rows) == 1:
            ordered = [(rows[0], c) for c in sorted(c for _r, c in coords)]
            assert ordered == [(rows[0], c) for c in range(ordered[0][1], ordered[-1][1] + 1)]
        else:
            assert len(cols) == 1, "each colored source must be a straight orthogonal segment"
            ordered = [(r, cols[0]) for r in sorted(r for r, _c in coords)]
            assert ordered == [(r, cols[0]) for r in range(ordered[0][0], ordered[-1][0] + 1)]

        sources.append(
            {
                "color": color,
                "ordered": ordered,
                "length": len(ordered) - 1,
                "cells": set(ordered),
            }
        )

    assert 3 <= len(sources) <= 4, "expected 3-4 distinct colored sources"
    lengths = [src["length"] for src in sources]
    assert len(set(lengths)) == len(lengths), "trail lengths must be unique"
    return sources


def _candidate_from_choice(source: dict, choose_last: bool) -> dict:
    ordered: list[Coord] = source["ordered"]
    if choose_last:
        marker = ordered[-1]
        prev = ordered[-2]
    else:
        marker = ordered[0]
        prev = ordered[1]

    trail_dir = (prev[0] - marker[0], prev[1] - marker[1])
    ray_dir = (-trail_dir[0], -trail_dir[1])
    return {
        "color": source["color"],
        "length": source["length"],
        "marker": marker,
        "trail_dir": trail_dir,
        "ray_dir": ray_dir,
        "cells": set(source["cells"]),
    }


def _ray_cells(grid: np.ndarray, bg: int, marker: Coord, ray_dir: Coord) -> list[Coord]:
    cells: list[Coord] = []
    r = marker[0] + ray_dir[0]
    c = marker[1] + ray_dir[1]
    while _in_bounds(grid.shape, r, c):
        if int(grid[r, c]) != bg:
            break
        cells.append((r, c))
        r += ray_dir[0]
        c += ray_dir[1]
    return cells


def _valid_assignments(grid: np.ndarray) -> list[list[dict]]:
    bg = _get_bg(grid)
    sources = _extract_sources(grid, bg)
    valid: list[list[dict]] = []

    for bits in itertools.product((False, True), repeat=len(sources)):
        chosen: list[dict] = []
        ok = True
        coverage: Counter[Coord] = Counter()

        for src, choose_last in zip(sources, bits):
            cand = _candidate_from_choice(src, choose_last)
            path = _ray_cells(grid, bg, cand["marker"], cand["ray_dir"])
            if not path:
                ok = False
                break
            cand["path"] = path
            chosen.append(cand)
            coverage.update(path)

        if not ok:
            continue

        overlap_cells = [cell for cell, count in coverage.items() if count >= 2]
        if not overlap_cells:
            continue
        valid.append(chosen)

    return valid


def _render_from_assignment(shape: tuple[int, int], bg: int, assignment: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    winners: dict[Coord, tuple[int, int]] = {}
    for ray in assignment:
        for cell in ray["path"]:
            current = winners.get(cell)
            if current is None or ray["length"] > current[0]:
                winners[cell] = (ray["length"], ray["color"])

    for (r, c), (_length, color) in winners.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored marker's ray away from its same-color trail, letting longer trails win overlaps, then erase the sources.

    Input structure:
    - The grid has black background and 3-4 distinct non-black colors.
    - Each color forms exactly one straight horizontal or vertical segment made of one marker pixel plus a same-color trail of length 1-4.
    - The chosen marker endpoint has background space in the direction opposite its trail, and at least one pair of rays overlap on background cells.

    Rule:
    - For each colored segment, identify the endpoint that acts as the marker, with the rest of the segment giving the trail direction and trail length.
    - Cast a ray from that marker in the direction opposite the trail, advancing only through original background cells and stopping before any non-background cell or the grid edge.
    - If multiple rays claim the same background cell, keep the color whose trail is longer.
    - After all rays are resolved, erase every original marker and trail cell to background.

    Key insight:
    - The same-color trail encodes two things at once: which way the ray fires and how strong that ray is at conflicts, because trail length is the priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)

    assignments = _valid_assignments(g)
    assert len(assignments) == 1, "expected a unique marker/trail interpretation"
    return _render_from_assignment(g.shape, bg, assignments[0])


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in _line_cells(spec["marker"], spec["trail_dir"], spec["length"]):
            grid[r, c] = spec["color"]
    return grid


def _render_expected(inp: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, dict[Coord, list[tuple[int, int]]]]:
    out = np.zeros_like(inp)
    claims: dict[Coord, list[tuple[int, int]]] = {}
    for spec in specs:
        marker = spec["marker"]
        ray_dir = (-spec["trail_dir"][0], -spec["trail_dir"][1])
        r = marker[0] + ray_dir[0]
        c = marker[1] + ray_dir[1]
        while _in_bounds(inp.shape, r, c):
            if int(inp[r, c]) != 0:
                break
            claims.setdefault((r, c), []).append((spec["length"], spec["color"]))
            r += ray_dir[0]
            c += ray_dir[1]

    for (r, c), options in claims.items():
        best = max(options, key=lambda item: item[0])
        out[r, c] = best[1]
    return out, claims


def _cells_touch(specs: list[dict]) -> bool:
    occupied: dict[Coord, int] = {}
    for spec in specs:
        color = spec["color"]
        for cell in _line_cells(spec["marker"], spec["trail_dir"], spec["length"]):
            if cell in occupied:
                return True
            occupied[cell] = color

    coords = list(occupied)
    for i, (r1, c1) in enumerate(coords):
        for r2, c2 in coords[i + 1 :]:
            if occupied[(r1, c1)] == occupied[(r2, c2)]:
                continue
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _spec_from_horizontal_left(row: int, length: int, color: int) -> dict:
    return {"marker": (row, length), "trail_dir": (0, -1), "length": length, "color": color}


def _spec_from_horizontal_right(row: int, length: int, color: int, width: int) -> dict:
    return {"marker": (row, width - 1 - length), "trail_dir": (0, 1), "length": length, "color": color}


def _spec_from_vertical_top(col: int, length: int, color: int) -> dict:
    return {"marker": (length, col), "trail_dir": (-1, 0), "length": length, "color": color}


def _spec_from_vertical_bottom(col: int, length: int, color: int, height: int) -> dict:
    return {"marker": (height - 1 - length, col), "trail_dir": (1, 0), "length": length, "color": color}


def _gen_two_horizontal_pairs(height: int, width: int, lengths: list[int], colors: list[int]) -> list[dict] | None:
    rows = random.sample(list(range(1, height - 1)), 2)
    rows.sort()
    if rows[1] - rows[0] < 3:
        return None

    specs = [
        _spec_from_horizontal_left(rows[0], lengths[0], colors[0]),
        _spec_from_horizontal_right(rows[0], lengths[1], colors[1], width),
        _spec_from_horizontal_left(rows[1], lengths[2], colors[2]),
        _spec_from_horizontal_right(rows[1], lengths[3], colors[3], width),
    ]

    for row, left_len, right_len in ((rows[0], lengths[0], lengths[1]), (rows[1], lengths[2], lengths[3])):
        left_marker = left_len
        right_marker = width - 1 - right_len
        if right_marker - left_marker - 1 < _MIN_OVERLAP:
            return None
        if row in (0, height - 1):
            return None
    return specs


def _gen_cross_pairs(height: int, width: int, lengths: list[int], colors: list[int]) -> list[dict] | None:
    h_left, h_right, v_top, v_bottom = lengths
    overlap_cols = list(range(h_left + 1, width - 1 - h_right))
    overlap_rows = list(range(v_top + 1, height - 1 - v_bottom))
    if len(overlap_cols) < _MIN_OVERLAP or len(overlap_rows) < _MIN_OVERLAP:
        return None

    valid_rows = [r for r in overlap_rows if r >= v_top + 2 and r <= height - v_bottom - 3]
    if not valid_rows:
        return None
    row = random.choice(valid_rows)
    col = random.choice(overlap_cols)

    return [
        _spec_from_horizontal_left(row, h_left, colors[0]),
        _spec_from_horizontal_right(row, h_right, colors[1], width),
        _spec_from_vertical_top(col, v_top, colors[2]),
        _spec_from_vertical_bottom(col, v_bottom, colors[3], height),
    ]


def _gen_pair_with_crossing_solo(height: int, width: int, lengths: list[int], colors: list[int]) -> list[dict] | None:
    pair_vertical = bool(random.getrandbits(1))
    if pair_vertical:
        top_len, bottom_len, solo_len = lengths
        overlap_rows = list(range(top_len + 1, height - 1 - bottom_len))
        if len(overlap_rows) < _MIN_OVERLAP:
            return None
        valid_cols = [c for c in range(solo_len + 2, width - 2)]
        if not valid_cols:
            return None
        col = random.choice([c for c in range(1, width - 1)])
        row = random.choice(overlap_rows)
        solo_from_left = bool(random.getrandbits(1))
        solo_col = solo_len if solo_from_left else width - 1 - solo_len
        if solo_col == col:
            return None
        specs = [
            _spec_from_vertical_top(col, top_len, colors[0]),
            _spec_from_vertical_bottom(col, bottom_len, colors[1], height),
            (
                _spec_from_horizontal_left(row, solo_len, colors[2])
                if solo_from_left
                else _spec_from_horizontal_right(row, solo_len, colors[2], width)
            ),
        ]
        return specs

    left_len, right_len, solo_len = lengths
    overlap_cols = list(range(left_len + 1, width - 1 - right_len))
    if len(overlap_cols) < _MIN_OVERLAP:
        return None

    solo_from_top = bool(random.getrandbits(1))
    row = random.randint(solo_len + 2, height - 3) if solo_from_top else random.randint(2, height - solo_len - 3)
    col = random.choice(overlap_cols)

    specs = [
        _spec_from_horizontal_left(row, left_len, colors[0]),
        _spec_from_horizontal_right(row, right_len, colors[1], width),
        (
            _spec_from_vertical_top(col, solo_len, colors[2])
            if solo_from_top
            else _spec_from_vertical_bottom(col, solo_len, colors[2], height)
        ),
    ]
    return specs


def _semantic_checks(inp: np.ndarray, specs: list[dict], expected: np.ndarray, claims: dict[Coord, list[tuple[int, int]]]) -> bool:
    if _cells_touch(specs):
        return False

    lengths = [spec["length"] for spec in specs]
    if len(set(lengths)) != len(lengths):
        return False

    contested = [cell for cell, options in claims.items() if len(options) >= 2]
    if not contested:
        return False

    assignments = _valid_assignments(inp)
    if len(assignments) != 1:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False

    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(actual, expected):
        return False
    if not np.any(expected != 0):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(("two_horizontal_pairs", "cross_pairs", "pair_with_crossing_solo"))
    if layout == "two_horizontal_pairs":
        height = random.randint(10, 18)
        width = random.randint(12, 18)
        lengths = random.sample([1, 2, 3, 4], 4)
        colors = random.sample(_COLORS, 4)
        specs = _gen_two_horizontal_pairs(height, width, lengths, colors)
    elif layout == "cross_pairs":
        height = random.randint(12, 18)
        width = random.randint(12, 18)
        lengths = random.sample([1, 2, 3, 4], 4)
        colors = random.sample(_COLORS, 4)
        specs = _gen_cross_pairs(height, width, lengths, colors)
    else:
        height = random.randint(10, 18)
        width = random.randint(10, 18)
        lengths = random.sample([1, 2, 3, 4], 3)
        colors = random.sample(_COLORS, 3)
        specs = _gen_pair_with_crossing_solo(height, width, lengths, colors)

    if specs is None:
        return None

    inp = _render_input((height, width), specs)
    expected, claims = _render_expected(inp, specs)
    if not _semantic_checks(inp, specs, expected, claims):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/12/017/1")
