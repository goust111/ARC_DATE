from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Marker = tuple[int, int, int]  # row, col, color

_COLORS = tuple(range(10))
_GEN_RETRIES = 500
_MIN_SIZE = 20
_MAX_SIZE = 30
_QUADRANTS = ("TL", "TR", "BL", "BR")


def _background(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _vertical_span(row: int, height: int) -> tuple[int, int]:
    assert 2 * int(row) != int(height) - 1, "marker must not be tied between top and bottom edges"
    if 2 * int(row) < int(height) - 1:
        return 0, int(row)
    return int(row), int(height) - 1


def _horizontal_span(col: int, width: int) -> tuple[int, int]:
    assert 2 * int(col) != int(width) - 1, "marker must not be tied between left and right edges"
    if 2 * int(col) < int(width) - 1:
        return 0, int(col)
    return int(col), int(width) - 1


def _extract_markers(g: np.ndarray, bg: int) -> list[Marker]:
    height, width = map(int, g.shape)
    markers: list[Marker] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        coords = np.argwhere(g == int(color))
        assert coords.shape == (1, 2), "each non-background color must be one marker pixel"
        row, col = map(int, coords[0])
        _vertical_span(row, height)
        _horizontal_span(col, width)
        markers.append((row, col, int(color)))

    assert len(markers) >= 3, "expected several marker pixels"
    occupied = {(r, c) for r, c, _color in markers}
    for row, col, _color in markers:
        for rr in range(max(0, row - 1), min(height, row + 2)):
            for cc in range(max(0, col - 1), min(width, col + 2)):
                if (rr, cc) != (row, col):
                    assert (rr, cc) not in occupied, "marker pixels must be isolated"
    return markers


def _crossing_colors(markers: list[Marker], height: int, width: int) -> dict[Coord, int]:
    marker_cells = {(r, c) for r, c, _color in markers}
    crossings: dict[Coord, int] = {}
    for v_idx, (v_row, v_col, v_color) in enumerate(markers):
        v0, v1 = _vertical_span(v_row, height)
        for h_idx, (h_row, h_col, _h_color) in enumerate(markers):
            if v_idx == h_idx:
                continue
            h0, h1 = _horizontal_span(h_col, width)
            if not (v0 <= h_row <= v1 and h0 <= v_col <= h1):
                continue
            cell = (h_row, v_col)
            if cell in marker_cells:
                continue
            prev = crossings.get(cell)
            assert prev is None or prev == v_color, "ambiguous vertical owners for one crossing"
            crossings[cell] = int(v_color)
    return crossings


def _crossing_colors_by_paths(markers: list[Marker], height: int, width: int) -> dict[Coord, int]:
    vertical: dict[Coord, set[int]] = {}
    horizontal: dict[Coord, set[int]] = {}
    for idx, (row, col, _color) in enumerate(markers):
        v0, v1 = _vertical_span(row, height)
        for rr in range(v0, v1 + 1):
            vertical.setdefault((rr, col), set()).add(idx)
        h0, h1 = _horizontal_span(col, width)
        for cc in range(h0, h1 + 1):
            horizontal.setdefault((row, cc), set()).add(idx)

    marker_cells = {(r, c) for r, c, _color in markers}
    crossings: dict[Coord, int] = {}
    for cell in set(vertical) & set(horizontal):
        owners = {
            v_idx
            for v_idx in vertical[cell]
            for h_idx in horizontal[cell]
            if v_idx != h_idx
        }
        if not owners or cell in marker_cells:
            continue
        owner_colors = {markers[idx][2] for idx in owners}
        assert len(owner_colors) == 1, "ambiguous vertical owners for one crossing"
        crossings[cell] = int(next(iter(owner_colors)))
    return crossings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep marker pixels and paint only crossings of their invisible nearest-edge L rays.

    Input structure:
    - A dominant background contains several isolated one-cell colored markers.
    - Marker colors are all distinct, and each marker has a unique nearer vertical
      edge and a unique nearer horizontal edge.
    - No visible guide lines are drawn; the rays are inferred from marker positions
      relative to the grid boundaries.

    Rule:
    - For each marker, construct an invisible L path: one vertical segment from the
      marker to the nearer top or bottom edge, and one horizontal segment from the
      marker to the nearer left or right edge.
    - Whenever one marker's vertical segment and another marker's horizontal segment
      pass through the same cell, paint that cell with the vertical segment owner's
      color.
    - Original marker cells take priority over crossing paint, and every other cell
      remains background.

    Key insight:
    - The visible output is not the rays themselves but the sparse intersection set
      of two invisible ray families; crossing color belongs to the marker whose path
      is vertical at that cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _background(g)
    markers = _extract_markers(g, bg)

    out = np.full((height, width), int(bg), dtype=int)
    for (row, col), color in _crossing_colors(markers, height, width).items():
        out[row, col] = int(color)
    for row, col, color in markers:
        out[row, col] = int(color)
    return out


# --------------- generator ---------------


def _side_values(length: int, upper_or_left: bool) -> list[int]:
    out: list[int] = []
    for value in range(1, int(length) - 1):
        if 2 * value == int(length) - 1:
            continue
        if upper_or_left and 2 * value < int(length) - 1:
            out.append(value)
        if not upper_or_left and 2 * value > int(length) - 1:
            out.append(value)
    return out


def _pick_spaced(values: list[int], count: int) -> list[int] | None:
    if len(values) < count:
        return None
    for _ in range(300):
        sample = sorted(random.sample(values, count))
        if all(b - a >= 2 for a, b in zip(sample, sample[1:])):
            return sample
    return None


def _base_coords(height: int, width: int) -> dict[str, list[Coord]] | None:
    top_rows = _pick_spaced(_side_values(height, True), 4)
    bottom_rows = _pick_spaced(_side_values(height, False), 4)
    left_cols = _pick_spaced(_side_values(width, True), 4)
    right_cols = _pick_spaced(_side_values(width, False), 4)
    if top_rows is None or bottom_rows is None or left_cols is None or right_cols is None:
        return None

    return {
        # First marker in each pair is arranged to own a vertical crossing;
        # second marker contributes the horizontal segment that reaches it.
        "TL": [(top_rows[1], left_cols[0]), (top_rows[0], left_cols[1])],
        "TR": [(top_rows[3], right_cols[1]), (top_rows[2], right_cols[0])],
        "BL": [(bottom_rows[0], left_cols[2]), (bottom_rows[1], left_cols[3])],
        "BR": [(bottom_rows[2], right_cols[3]), (bottom_rows[3], right_cols[2])],
    }


def _render(markers: list[Marker], height: int, width: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), int(bg), dtype=int)
    out = np.full((height, width), int(bg), dtype=int)
    for (row, col), color in _crossing_colors(markers, height, width).items():
        out[row, col] = int(color)
    for row, col, color in markers:
        inp[row, col] = int(color)
        out[row, col] = int(color)
    return inp, out


def _horizontal_owner_output(markers: list[Marker], height: int, width: int, bg: int) -> np.ndarray:
    out = np.full((height, width), int(bg), dtype=int)
    marker_cells = {(r, c) for r, c, _color in markers}
    for v_idx, (v_row, v_col, _v_color) in enumerate(markers):
        v0, v1 = _vertical_span(v_row, height)
        for h_idx, (h_row, h_col, h_color) in enumerate(markers):
            if v_idx == h_idx:
                continue
            h0, h1 = _horizontal_span(h_col, width)
            cell = (h_row, v_col)
            if cell in marker_cells:
                continue
            if v0 <= h_row <= v1 and h0 <= v_col <= h1:
                out[cell] = int(h_color)
    for row, col, color in markers:
        out[row, col] = int(color)
    return out


def _full_ray_output(markers: list[Marker], height: int, width: int, bg: int) -> np.ndarray:
    out = np.full((height, width), int(bg), dtype=int)
    marker_cells = {(r, c) for r, c, _color in markers}
    for v_idx, (_v_row, v_col, v_color) in enumerate(markers):
        for h_idx, (h_row, _h_col, _h_color) in enumerate(markers):
            if v_idx == h_idx:
                continue
            cell = (h_row, v_col)
            if cell not in marker_cells:
                out[cell] = int(v_color)
    for row, col, color in markers:
        out[row, col] = int(color)
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, markers: list[Marker], bg: int) -> bool:
    height, width = map(int, inp.shape)
    try:
        solved = main(inp)
        parsed = _extract_markers(inp, bg)
        by_paths = _crossing_colors_by_paths(parsed, height, width)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    pair_crossings = _crossing_colors(markers, height, width)
    if pair_crossings != by_paths:
        return False

    added = [(r, c) for r, c in pair_crossings]
    if not (3 <= len(added) <= 16):
        return False
    if len({pair_crossings[cell] for cell in added}) < 3:
        return False

    vertical_dirs = {
        "up" if _vertical_span(row, height)[0] == 0 else "down"
        for row, _col, _color in markers
    }
    horizontal_dirs = {
        "left" if _horizontal_span(col, width)[0] == 0 else "right"
        for _row, col, _color in markers
    }
    if vertical_dirs != {"up", "down"} or horizontal_dirs != {"left", "right"}:
        return False

    misses_vertical = 0
    misses_horizontal = 0
    for v_idx, (v_row, v_col, _v_color) in enumerate(markers):
        v0, v1 = _vertical_span(v_row, height)
        for h_idx, (h_row, h_col, _h_color) in enumerate(markers):
            if v_idx == h_idx:
                continue
            h0, h1 = _horizontal_span(h_col, width)
            v_ok = v0 <= h_row <= v1
            h_ok = h0 <= v_col <= h1
            if v_ok and not h_ok:
                misses_horizontal += 1
            if h_ok and not v_ok:
                misses_vertical += 1
    if misses_vertical < len(markers) // 2 or misses_horizontal < len(markers) // 2:
        return False

    if np.array_equal(_horizontal_owner_output(markers, height, width, bg), expected):
        return False
    if np.array_equal(_full_ray_output(markers, height, width, bg), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(_MIN_SIZE, _MAX_SIZE)
    width = random.randint(_MIN_SIZE, _MAX_SIZE)
    bg = random.choice(_COLORS)

    coords_by_quad = _base_coords(height, width)
    if coords_by_quad is None:
        return None

    if random.random() < 0.35:
        active_quads = random.sample(list(_QUADRANTS), 3)
    else:
        active_quads = list(_QUADRANTS)
    coords = [coord for quad in active_quads for coord in coords_by_quad[quad]]
    random.shuffle(coords)

    marker_colors = random.sample([color for color in _COLORS if color != bg], len(coords))
    markers = [(row, col, color) for (row, col), color in zip(coords, marker_colors)]

    inp, expected = _render(markers, height, width, bg)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if _semantic_checks(inp, expected, markers, bg):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible L-ray crossing input")
