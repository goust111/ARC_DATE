from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BACKGROUND_COLOR = 0
MARKER_COLORS = (1, 2, 3)
SEPARATOR_COLOR = 5
BASE_COLOR = 7

TILE_GRID_SIZE = 3
INDICATOR_MAX_LEN = 3
GEN_TILE_SIZE_RANGE = (3, 4)
GEN_TOP_MARGIN_RANGE = (0, 3)
GEN_BOTTOM_MARGIN_RANGE = (0, 3)
GEN_LEFT_MARGIN_RANGE = (0, 2)
GEN_GAP_RANGE = (1, 3)
GEN_RIGHT_MARGIN_RANGE = (0, 2)
GEN_INDICATOR_WIDTH = INDICATOR_MAX_LEN
GEN_RETRY_LIMIT = 400
GEN_ACTIVE_COLOR_COUNTS = (2, 3)
GEN_EXTRA_BAR_PROB = 0.35
GEN_MISSING_BAR_PROB = 0.4
GEN_STRAIGHT_SHAPE_PROB = 0.55


def main(grid: np.ndarray) -> np.ndarray:
    """Validate indicator bars against lattice marker counts, then connect matching colors and erase mismatches.

    Input structure:
    A black background contains a 3x3 orange tile lattice on the left, split by two grey separator rows and two grey
    separator columns.  Marker pixels of a few colors appear inside the tiles, and each color keeps one fixed local
    offset across every tile where it appears.  On the right, horizontal color bars sit on black background.

    Rule:
    For each marker color, count how many tiles contain that color in the lattice and compare the count with the
    length of the same-color bar on the right.  If the count matches the bar length and is positive, keep the markers
    and connect aligned same-color tile positions across intermediate tiles at the shared local offset.  Otherwise,
    erase every lattice marker of that color back to orange and erase its bar back to black.

    Key insight:
    The right-side bars are not labels; they are an external verification signal that decides whether a color survives
    and expands into lattice connections or is removed entirely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    layout = _detect_lattice(g)
    markers_by_color, offsets_by_color = _collect_markers(g, layout)

    indicator_start = layout["right"] + 1
    indicator = g[:, indicator_start:] if indicator_start < g.shape[1] else np.zeros((g.shape[0], 0), dtype=int)
    bar_colors = set(map(int, np.unique(indicator))) - {BACKGROUND_COLOR}
    colors = set(markers_by_color) | bar_colors

    lattice_mask = np.zeros_like(g, dtype=bool)
    lattice_mask[layout["top"] : layout["bottom"] + 1, layout["left"] : layout["right"] + 1] = True
    indicator_mask = np.zeros_like(g, dtype=bool)
    if indicator_start < g.shape[1]:
        indicator_mask[:, indicator_start:] = True

    out = g.copy()
    matched_colors = []
    for color in sorted(colors):
        tile_count = len(markers_by_color.get(color, set()))
        bar_len = int(np.count_nonzero(indicator == color))
        if tile_count == bar_len and tile_count > 0:
            matched_colors.append(color)
            continue
        out[lattice_mask & (g == color)] = BASE_COLOR
        out[indicator_mask & (g == color)] = BACKGROUND_COLOR

    for color in matched_colors:
        offset_r, offset_c = offsets_by_color[color]
        tiles = markers_by_color[color]

        for tile_r in range(TILE_GRID_SIZE):
            cols = sorted(tile_c for row, tile_c in tiles if row == tile_r)
            if len(cols) >= 2:
                for tile_c in range(cols[0], cols[-1] + 1):
                    _paint_tile_cell(out, layout, tile_r, tile_c, offset_r, offset_c, color)

        for tile_c in range(TILE_GRID_SIZE):
            rows = sorted(tile_r for tile_r, col in tiles if col == tile_c)
            if len(rows) >= 2:
                for tile_r in range(rows[0], rows[-1] + 1):
                    _paint_tile_cell(out, layout, tile_r, tile_c, offset_r, offset_c, color)

    return out


def _detect_lattice(grid: np.ndarray) -> dict[str, int]:
    sep_mask = grid == SEPARATOR_COLOR
    row_counts = sep_mask.sum(axis=1)
    col_counts = sep_mask.sum(axis=0)
    row_max = int(row_counts.max())
    col_max = int(col_counts.max())
    assert row_max >= 5 and col_max >= 5, "separator lines not found"

    sep_rows = [int(r) for r in np.flatnonzero(row_counts == row_max)]
    sep_cols = [int(c) for c in np.flatnonzero(col_counts == col_max)]
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected a 3x3 lattice"

    tile_h = sep_rows[1] - sep_rows[0] - 1
    tile_w = sep_cols[1] - sep_cols[0] - 1
    assert tile_h >= 2 and tile_w >= 2, "tile size too small"

    top = sep_rows[0] - tile_h
    left = sep_cols[0] - tile_w
    bottom = sep_rows[1] + tile_h
    right = sep_cols[1] + tile_w
    assert 0 <= top <= bottom < grid.shape[0]
    assert 0 <= left <= right < grid.shape[1]
    assert row_max == right - left + 1
    assert col_max == bottom - top + 1

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "tile_h": tile_h,
        "tile_w": tile_w,
    }


def _collect_markers(
    grid: np.ndarray, layout: dict[str, int]
) -> tuple[dict[int, set[tuple[int, int]]], dict[int, tuple[int, int]]]:
    row_tiles = _tile_index_ranges(layout["top"], layout["tile_h"])
    col_tiles = _tile_index_ranges(layout["left"], layout["tile_w"])
    row_lookup = {
        row: (tile_r, local_r)
        for tile_r, rows in enumerate(row_tiles)
        for local_r, row in enumerate(rows)
    }
    col_lookup = {
        col: (tile_c, local_c)
        for tile_c, cols in enumerate(col_tiles)
        for local_c, col in enumerate(cols)
    }

    markers_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
    offsets_by_color: dict[int, tuple[int, int]] = {}
    for row in range(layout["top"], layout["bottom"] + 1):
        for col in range(layout["left"], layout["right"] + 1):
            color = int(grid[row, col])
            if color in (BACKGROUND_COLOR, SEPARATOR_COLOR, BASE_COLOR):
                continue
            tile_r, local_r = row_lookup[row]
            tile_c, local_c = col_lookup[col]
            offset = (local_r, local_c)
            if color in offsets_by_color:
                assert offsets_by_color[color] == offset, "marker color must preserve its local offset"
            else:
                offsets_by_color[color] = offset
            markers_by_color[color].add((tile_r, tile_c))

    return dict(markers_by_color), offsets_by_color


def _tile_index_ranges(start: int, tile_size: int) -> list[list[int]]:
    return [
        list(range(start + tile * (tile_size + 1), start + tile * (tile_size + 1) + tile_size))
        for tile in range(TILE_GRID_SIZE)
    ]


def _tile_cell(
    layout: dict[str, int], tile_r: int, tile_c: int, local_r: int, local_c: int
) -> tuple[int, int]:
    row = layout["top"] + tile_r * (layout["tile_h"] + 1) + local_r
    col = layout["left"] + tile_c * (layout["tile_w"] + 1) + local_c
    return row, col


def _paint_tile_cell(
    out: np.ndarray,
    layout: dict[str, int],
    tile_r: int,
    tile_c: int,
    local_r: int,
    local_c: int,
    color: int,
) -> None:
    row, col = _tile_cell(layout, tile_r, tile_c, local_r, local_c)
    if int(out[row, col]) in (BASE_COLOR, color):
        out[row, col] = color


def _sample_shape() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    if random.random() < GEN_STRAIGHT_SHAPE_PROB:
        if random.random() < 0.5:
            tile_r = random.randrange(TILE_GRID_SIZE)
            return (
                {(tile_r, 0), (tile_r, TILE_GRID_SIZE - 1)},
                {(tile_r, tile_c) for tile_c in range(TILE_GRID_SIZE)},
            )
        tile_c = random.randrange(TILE_GRID_SIZE)
        return (
            {(0, tile_c), (TILE_GRID_SIZE - 1, tile_c)},
            {(tile_r, tile_c) for tile_r in range(TILE_GRID_SIZE)},
        )

    corner_r = random.choice((0, TILE_GRID_SIZE - 1))
    corner_c = random.choice((0, TILE_GRID_SIZE - 1))
    input_tiles = {
        (corner_r, corner_c),
        (corner_r, TILE_GRID_SIZE - 1 - corner_c),
        (TILE_GRID_SIZE - 1 - corner_r, corner_c),
    }
    output_tiles = {(corner_r, tile_c) for tile_c in range(TILE_GRID_SIZE)}
    output_tiles.update((tile_r, corner_c) for tile_r in range(TILE_GRID_SIZE))
    return input_tiles, output_tiles


def _sample_bar_len(tile_count: int, matched: bool) -> int:
    if matched:
        return tile_count
    if random.random() < GEN_MISSING_BAR_PROB:
        return 0
    choices = [length for length in range(1, INDICATOR_MAX_LEN + 1) if length != tile_count]
    return random.choice(choices)


def _valid_specs(specs: list[dict[str, object]]) -> bool:
    offsets = [spec["offset"] for spec in specs]
    if len(set(offsets)) != len(offsets):
        return False
    tile_load = Counter(tile for spec in specs for tile in spec["in_tiles"])
    return all(load <= 2 for load in tile_load.values())


def _render_input(layout: dict[str, int], active_specs: list[dict[str, object]], bar_specs: list[dict[str, object]]) -> np.ndarray | None:
    height = layout["height"]
    width = layout["width"]
    grid = np.zeros((height, width), dtype=int)

    grid[layout["top"] : layout["bottom"] + 1, layout["left"] : layout["right"] + 1] = BASE_COLOR
    for sep_row in (layout["top"] + layout["tile_h"], layout["top"] + 2 * layout["tile_h"] + 1):
        grid[sep_row, layout["left"] : layout["right"] + 1] = SEPARATOR_COLOR
    for sep_col in (layout["left"] + layout["tile_w"], layout["left"] + 2 * layout["tile_w"] + 1):
        grid[layout["top"] : layout["bottom"] + 1, sep_col] = SEPARATOR_COLOR

    for spec in active_specs:
        color = int(spec["color"])
        local_r, local_c = spec["offset"]
        for tile_r, tile_c in spec["in_tiles"]:
            row, col = _tile_cell(layout, tile_r, tile_c, local_r, local_c)
            if int(grid[row, col]) != BASE_COLOR:
                return None
            grid[row, col] = color

    for spec in bar_specs:
        if int(spec["bar_len"]) == 0:
            continue
        row = int(spec["bar_row"])
        color = int(spec["color"])
        length = int(spec["bar_len"])
        left = layout["indicator_left"]
        grid[row, left : left + length] = color

    return grid


def _expected_from_specs(
    inp: np.ndarray,
    layout: dict[str, int],
    active_specs: list[dict[str, object]],
    bar_specs: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()

    for spec in active_specs:
        if spec["matched"]:
            continue
        local_r, local_c = spec["offset"]
        for tile_r, tile_c in spec["in_tiles"]:
            row, col = _tile_cell(layout, tile_r, tile_c, local_r, local_c)
            out[row, col] = BASE_COLOR

    for spec in bar_specs:
        if spec["matched"] or int(spec["bar_len"]) == 0:
            continue
        row = int(spec["bar_row"])
        left = layout["indicator_left"]
        out[row, left : left + int(spec["bar_len"])] = BACKGROUND_COLOR

    for spec in active_specs:
        if not spec["matched"]:
            continue
        color = int(spec["color"])
        local_r, local_c = spec["offset"]
        for tile_r, tile_c in spec["out_tiles"]:
            row, col = _tile_cell(layout, tile_r, tile_c, local_r, local_c)
            if int(out[row, col]) in (BASE_COLOR, color):
                out[row, col] = color

    return out


def _make_layout(tile_size: int) -> dict[str, int]:
    top = random.randint(*GEN_TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*GEN_BOTTOM_MARGIN_RANGE)
    left = random.randint(*GEN_LEFT_MARGIN_RANGE)
    gap = random.randint(*GEN_GAP_RANGE)
    right_margin = random.randint(*GEN_RIGHT_MARGIN_RANGE)

    lattice_span = TILE_GRID_SIZE * tile_size + (TILE_GRID_SIZE - 1)
    height = top + lattice_span + bottom_margin
    width = left + lattice_span + gap + GEN_INDICATOR_WIDTH + right_margin
    right = left + lattice_span - 1
    bottom = top + lattice_span - 1
    indicator_left = right + 1 + gap

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "tile_h": tile_size,
        "tile_w": tile_size,
        "height": height,
        "width": width,
        "indicator_left": indicator_left,
    }


def _sample_active_specs(tile_size: int) -> list[dict[str, object]] | None:
    active_count = random.choice(GEN_ACTIVE_COLOR_COUNTS)
    colors = random.sample(MARKER_COLORS, active_count)
    matched_count = random.randint(1, active_count - 1)
    matched_colors = set(random.sample(colors, matched_count))
    offsets = random.sample(
        [(row, col) for row in range(tile_size) for col in range(tile_size)],
        active_count,
    )

    for _ in range(80):
        specs = []
        for color, offset in zip(colors, offsets):
            input_tiles, output_tiles = _sample_shape()
            matched = color in matched_colors
            specs.append(
                {
                    "color": color,
                    "offset": offset,
                    "in_tiles": input_tiles,
                    "out_tiles": output_tiles,
                    "matched": matched,
                    "bar_len": _sample_bar_len(len(input_tiles), matched),
                }
            )
        if _valid_specs(specs):
            return specs
    return None


def _attach_bar_rows(
    layout: dict[str, int], active_specs: list[dict[str, object]]
) -> list[dict[str, object]] | None:
    bar_specs = [
        {
            "color": int(spec["color"]),
            "matched": bool(spec["matched"]),
            "bar_len": int(spec["bar_len"]),
        }
        for spec in active_specs
        if int(spec["bar_len"]) > 0
    ]

    unused_colors = [color for color in MARKER_COLORS if color not in {int(spec["color"]) for spec in active_specs}]
    if unused_colors and random.random() < GEN_EXTRA_BAR_PROB:
        bar_specs.append(
            {
                "color": random.choice(unused_colors),
                "matched": False,
                "bar_len": random.randint(1, INDICATOR_MAX_LEN),
            }
        )

    candidate_rows = list(range(layout["height"]))
    if len(candidate_rows) < len(bar_specs):
        return None
    chosen_rows = random.sample(candidate_rows, len(bar_specs))
    for spec, row in zip(bar_specs, chosen_rows):
        spec["bar_row"] = row
    return bar_specs


def _gen_attempt() -> np.ndarray | None:
    tile_size = random.randint(*GEN_TILE_SIZE_RANGE)
    layout = _make_layout(tile_size)
    if max(layout["height"], layout["width"]) > 30:
        return None

    active_specs = _sample_active_specs(tile_size)
    if active_specs is None:
        return None

    bar_specs = _attach_bar_rows(layout, active_specs)
    if bar_specs is None:
        return None

    inp = _render_input(layout, active_specs, bar_specs)
    if inp is None:
        return None

    expected = _expected_from_specs(inp, layout, active_specs, bar_specs)
    if np.array_equal(expected, inp):
        return None

    try:
        observed = main(inp)
    except Exception:
        return None
    if not np.array_equal(observed, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
