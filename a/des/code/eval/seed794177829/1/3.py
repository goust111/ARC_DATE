from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

INPUT_COLORS = (RED, BLUE, YELLOW)
COLOR_BITS = {
    RED: 1,
    BLUE: 2,
    YELLOW: 4,
}
MIX_FROM_BITS = {
    1: RED,
    2: BLUE,
    4: YELLOW,
    3: MAGENTA,
    5: ORANGE,
    6: GREEN,
    7: MAROON,
}

GRID_MIN = 8
GRID_MAX = 15
MIN_GAP = 2
GEN_RETRIES = 400
SCENARIOS = (
    "orthogonal_only",
    "pair_same",
    "triple_maroon",
    "double_pair",
)


def _marker_direction(r: int, c: int, height: int, width: int) -> tuple[int, int]:
    directions = []
    if r == 0:
        directions.append((1, 0))
    if r == height - 1:
        directions.append((-1, 0))
    if c == 0:
        directions.append((0, 1))
    if c == width - 1:
        directions.append((0, -1))
    assert len(directions) == 1, "marker must lie on exactly one side of the frame"
    return directions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Project border markers inward and keep only pixels covered by at least two rays."""
    g = np.asarray(grid, dtype=int)
    height, width = g.shape

    cover_count = np.zeros((height, width), dtype=int)
    color_mask = np.zeros((height, width), dtype=int)

    marker_coords = np.argwhere((g != BG) & (g != GREY))
    for r_raw, c_raw in marker_coords:
        r = int(r_raw)
        c = int(c_raw)
        color = int(g[r, c])
        assert color in COLOR_BITS, "marker color must be red, blue, or yellow"
        dr, dc = _marker_direction(r, c, height, width)

        rr = r
        cc = c
        while 0 <= rr < height and 0 <= cc < width:
            cover_count[rr, cc] += 1
            color_mask[rr, cc] |= COLOR_BITS[color]
            rr += dr
            cc += dc

    out = np.full_like(g, BG)
    keep = cover_count >= 2
    for bits, mixed_color in MIX_FROM_BITS.items():
        out[keep & (color_mask == bits)] = mixed_color
    return out


def _sample_spaced_positions(start: int, end: int, count: int, min_gap: int = MIN_GAP) -> list[int] | None:
    if count == 0:
        return []
    if end < start:
        return None
    max_count = 1 + (end - start) // min_gap
    if count > max_count:
        return None

    candidates = list(range(start, end + 1))
    for _ in range(200):
        picked = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _other_colors(*excluded: int) -> list[int]:
    blocked = set(map(int, excluded))
    return [color for color in INPUT_COLORS if color not in blocked]


def _marker_coord(edge: str, pos: int, height: int, width: int) -> tuple[int, int]:
    if edge == "top":
        return 0, pos
    if edge == "bottom":
        return height - 1, pos
    if edge == "left":
        return pos, 0
    assert edge == "right"
    return pos, width - 1


def _blank_grid(height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY
    return grid


def _place_markers(height: int, width: int, markers: list[tuple[str, int, int]]) -> np.ndarray:
    grid = _blank_grid(height, width)
    used_cells: set[tuple[int, int]] = set()
    for edge, pos, color in markers:
        r, c = _marker_coord(edge, int(pos), height, width)
        assert (r, c) not in used_cells, "markers must not overlap"
        used_cells.add((r, c))
        grid[r, c] = int(color)
    return grid


def _maybe_extra_single_markers(
    primary_positions: list[int],
    axis: str,
) -> list[tuple[str, int, int]]:
    extras: list[tuple[str, int, int]] = []
    for pos in primary_positions:
        edge = random.choice(("top", "bottom")) if axis == "col" else random.choice(("left", "right"))
        extras.append((edge, int(pos), random.choice(INPUT_COLORS)))
    return extras


def _build_orthogonal_only(height: int, width: int) -> list[tuple[str, int, int]] | None:
    max_cols = min(3, 1 + (width - 3) // MIN_GAP)
    max_rows = min(3, 1 + (height - 3) // MIN_GAP)
    n_cols = random.randint(1, max_cols)
    n_rows = random.randint(1, max_rows)

    cols = _sample_spaced_positions(1, width - 2, n_cols)
    rows = _sample_spaced_positions(1, height - 2, n_rows)
    if cols is None or rows is None:
        return None

    target_same = random.choice([False, True])
    first_col_color = random.choice(INPUT_COLORS)
    first_row_color = first_col_color if target_same else random.choice(_other_colors(first_col_color))

    markers: list[tuple[str, int, int]] = [
        (random.choice(("top", "bottom")), cols[0], first_col_color),
        (random.choice(("left", "right")), rows[0], first_row_color),
    ]

    for col in cols[1:]:
        markers.append((random.choice(("top", "bottom")), col, random.choice(INPUT_COLORS)))
    for row in rows[1:]:
        markers.append((random.choice(("left", "right")), row, random.choice(INPUT_COLORS)))
    return markers


def _build_pair_same(height: int, width: int) -> list[tuple[str, int, int]] | None:
    axis = random.choice(("vertical", "horizontal"))
    pair_color = random.choice(INPUT_COLORS)
    keep_same_at_cross = random.choice([False, True])

    if axis == "vertical":
        total_cols = 1 + random.randint(0, 1)
        cols = _sample_spaced_positions(1, width - 2, total_cols)
        rows = _sample_spaced_positions(1, height - 2, random.randint(1, 2))
        if cols is None or rows is None:
            return None
        pair_col = cols[0]
        cross_color = pair_color if keep_same_at_cross else random.choice(_other_colors(pair_color))
        markers = [("top", pair_col, pair_color), ("bottom", pair_col, pair_color)]
        markers.append((random.choice(("left", "right")), rows[0], cross_color))
        for row in rows[1:]:
            markers.append((random.choice(("left", "right")), row, random.choice(INPUT_COLORS)))
        markers.extend(_maybe_extra_single_markers(cols[1:], "col"))
        return markers

    total_rows = 1 + random.randint(0, 1)
    rows = _sample_spaced_positions(1, height - 2, total_rows)
    cols = _sample_spaced_positions(1, width - 2, random.randint(1, 2))
    if rows is None or cols is None:
        return None
    pair_row = rows[0]
    cross_color = pair_color if keep_same_at_cross else random.choice(_other_colors(pair_color))
    markers = [("left", pair_row, pair_color), ("right", pair_row, pair_color)]
    markers.append((random.choice(("top", "bottom")), cols[0], cross_color))
    for col in cols[1:]:
        markers.append((random.choice(("top", "bottom")), col, random.choice(INPUT_COLORS)))
    markers.extend(_maybe_extra_single_markers(rows[1:], "row"))
    return markers


def _build_triple_maroon(height: int, width: int) -> list[tuple[str, int, int]] | None:
    axis = random.choice(("vertical", "horizontal"))
    pair_a, pair_b = random.sample(INPUT_COLORS, 2)
    third_color = _other_colors(pair_a, pair_b)[0]

    if axis == "vertical":
        total_cols = 1 + random.randint(0, 1)
        cols = _sample_spaced_positions(1, width - 2, total_cols)
        rows = _sample_spaced_positions(1, height - 2, random.randint(1, 2))
        if cols is None or rows is None:
            return None
        markers = [("top", cols[0], pair_a), ("bottom", cols[0], pair_b)]
        markers.append((random.choice(("left", "right")), rows[0], third_color))
        for row in rows[1:]:
            markers.append((random.choice(("left", "right")), row, random.choice(INPUT_COLORS)))
        markers.extend(_maybe_extra_single_markers(cols[1:], "col"))
        return markers

    total_rows = 1 + random.randint(0, 1)
    rows = _sample_spaced_positions(1, height - 2, total_rows)
    cols = _sample_spaced_positions(1, width - 2, random.randint(1, 2))
    if rows is None or cols is None:
        return None
    markers = [("left", rows[0], pair_a), ("right", rows[0], pair_b)]
    markers.append((random.choice(("top", "bottom")), cols[0], third_color))
    for col in cols[1:]:
        markers.append((random.choice(("top", "bottom")), col, random.choice(INPUT_COLORS)))
    markers.extend(_maybe_extra_single_markers(rows[1:], "row"))
    return markers


def _build_double_pair(height: int, width: int) -> list[tuple[str, int, int]] | None:
    total_cols = 1 + random.randint(0, 1)
    total_rows = 1 + random.randint(0, 1)
    cols = _sample_spaced_positions(1, width - 2, total_cols)
    rows = _sample_spaced_positions(1, height - 2, total_rows)
    if cols is None or rows is None:
        return None

    crossing_colors = list(INPUT_COLORS) + [random.choice(INPUT_COLORS)]
    random.shuffle(crossing_colors)
    top_color, bottom_color, left_color, right_color = crossing_colors

    markers = [
        ("top", cols[0], top_color),
        ("bottom", cols[0], bottom_color),
        ("left", rows[0], left_color),
        ("right", rows[0], right_color),
    ]
    markers.extend(_maybe_extra_single_markers(cols[1:], "col"))
    markers.extend(_maybe_extra_single_markers(rows[1:], "row"))
    return markers


SCENARIO_BUILDERS = {
    "orthogonal_only": _build_orthogonal_only,
    "pair_same": _build_pair_same,
    "triple_maroon": _build_triple_maroon,
    "double_pair": _build_double_pair,
}


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    scenario = random.choice(SCENARIOS)
    markers = SCENARIO_BUILDERS[scenario](height, width)
    if markers is None:
        return None

    grid = _place_markers(height, width, markers)
    out = main(grid)
    if not np.any(out != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-mixing puzzle input")
