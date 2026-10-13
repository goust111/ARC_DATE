from __future__ import annotations

import math
import random
from itertools import combinations

import numpy as np


BG = 0
SEPARATOR = 8
USABLE_COLORS = [color for color in range(10) if color not in {BG, SEPARATOR}]
GENERATION_RETRIES = 400


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave a nonempty cell before each divider"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a nonempty final cell"
    spans.append((start, length - 1))
    return spans


def _grid_cells(grid: np.ndarray):
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    row_spans = _spans(int(grid.shape[0]), sep_rows)
    col_spans = _spans(int(grid.shape[1]), sep_cols)
    cells = [
        [grid[r0 : r1 + 1, c0 : c1 + 1] for c0, c1 in col_spans]
        for r0, r1 in row_spans
    ]
    return row_spans, col_spans, cells


def _corner_positions(height: int, width: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]


def _pair_and_markers(cell: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    h, w = map(int, cell.shape)
    corners = {pos: int(cell[pos]) for pos in _corner_positions(h, w)}
    markers = [pos for pos, value in corners.items() if value != BG]
    assert len(markers) == 2, "each cell must expose exactly two corner seeds"
    marker_set = set(markers)
    diagonal_a = {(0, 0), (h - 1, w - 1)}
    diagonal_b = {(0, w - 1), (h - 1, 0)}
    assert marker_set == diagonal_a or marker_set == diagonal_b, "corner seeds must occupy opposite corners"
    colors = tuple(sorted(int(corners[pos]) for pos in markers))
    assert colors[0] != colors[1], "seed pairs must use two distinct colors"
    return colors, markers


def _is_legend_cell(cell: np.ndarray) -> bool:
    return int(np.count_nonzero(cell != BG)) > 2


def _detect_legend_side(cells: list[list[np.ndarray]]) -> str:
    n_rows = len(cells)
    n_cols = len(cells[0])
    right = (
        n_cols >= 2
        and all(_is_legend_cell(cells[r][-1]) for r in range(n_rows))
        and all(not _is_legend_cell(cells[r][c]) for r in range(n_rows) for c in range(n_cols - 1))
    )
    bottom = (
        n_rows >= 2
        and all(_is_legend_cell(cells[-1][c]) for c in range(n_cols))
        and all(not _is_legend_cell(cells[r][c]) for r in range(n_rows - 1) for c in range(n_cols))
    )
    assert right ^ bottom, "expected exactly one edge strip to be the legend"
    return "right" if right else "bottom"


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = shape
    pat_h, pat_w = map(int, pattern.shape)
    row_idx = np.arange(out_h) % pat_h
    col_idx = np.arange(out_w) % pat_w
    return pattern[row_idx[:, None], col_idx[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each separator-bounded region by looking up its corner-color pair in an edge legend of sample patterns.

    Input structure:
    - The background is black, and azure full rows and columns divide the canvas into rectangular cells.
    - Every non-legend cell contains exactly two single-pixel seeds in opposite corners on an otherwise black interior.
    - The legend is the entire rightmost cell-column or the entire bottom cell-row, and each legend cell contains a visible sample pattern plus the same kind of diagonal corner-color pair.

    Rule:
    - Read the two seed colors in each main cell as an unordered pair.
    - Find the legend cell whose diagonal corner markers use the same unordered pair.
    - Remove the two legend markers from that sample, tile the remaining legend cell pattern across the whole target cell, and overwrite the target cell with that tiling.
    - Leave the azure separator lines and the legend strip unchanged.

    Key insight:
    - The two corner seeds are not decoration: together they form a lookup key that selects one legend sample, and that sample must then be repeated to fit the target region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert np.all((g == BG) | (g == SEPARATOR) | ((0 <= g) & (g <= 9))), "grid must use ARC-style color ids"

    row_spans, col_spans, cells = _grid_cells(g)
    legend_side = _detect_legend_side(cells)
    n_rows = len(cells)
    n_cols = len(cells[0])

    legend_map: dict[tuple[int, int], np.ndarray] = {}
    if legend_side == "right":
        legend_positions = [(r, n_cols - 1) for r in range(n_rows)]
        main_positions = [(r, c) for r in range(n_rows) for c in range(n_cols - 1)]
    else:
        legend_positions = [(n_rows - 1, c) for c in range(n_cols)]
        main_positions = [(r, c) for r in range(n_rows - 1) for c in range(n_cols)]

    for r, c in legend_positions:
        pair, markers = _pair_and_markers(cells[r][c])
        assert pair not in legend_map, "legend pairs must be unique"
        pattern = cells[r][c].copy()
        for mr, mc in markers:
            pattern[mr, mc] = BG
        legend_map[pair] = pattern

    out = g.copy()
    for r, c in main_positions:
        pair, _ = _pair_and_markers(cells[r][c])
        assert pair in legend_map, "every main-cell pair must appear in the legend"
        tiled = _tile_pattern(legend_map[pair], cells[r][c].shape)
        r0, r1 = row_spans[r]
        c0, c1 = col_spans[c]
        out[r0 : r1 + 1, c0 : c1 + 1] = tiled

    return out


def _sizes_to_spans(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        spans.append((cursor, cursor + size - 1))
        cursor += size
        if idx != len(sizes) - 1:
            separators.append(cursor)
            cursor += 1
    return spans, separators, cursor


def _place_markers(base: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    cell = base.copy()
    h, w = map(int, cell.shape)
    cell[0, 0] = cell[0, w - 1] = cell[h - 1, 0] = cell[h - 1, w - 1] = BG
    positions = random.choice(
        [[(0, 0), (h - 1, w - 1)], [(0, w - 1), (h - 1, 0)]]
    )
    colors = list(pair)
    random.shuffle(colors)
    for (r, c), color in zip(positions, colors):
        cell[r, c] = color
    return cell


def _seed_cell(height: int, width: int, pair: tuple[int, int]) -> np.ndarray:
    return _place_markers(np.full((height, width), BG, dtype=int), pair)


def _pattern_signature(pattern: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, pattern.shape)), tuple(map(int, pattern.reshape(-1)))


def _make_pattern(height: int, width: int, colors: list[int]) -> np.ndarray:
    corners = set(_corner_positions(height, width))
    kinds = ["solid", "checker", "hstripe", "vstripe", "cross", "frame", "diag"]
    for _ in range(80):
        primary = random.choice(colors)
        secondary_choices = [BG] + [color for color in colors if color != primary]
        secondary = random.choice(secondary_choices)
        kind = random.choice(kinds)
        pattern = np.full((height, width), BG, dtype=int)

        for r in range(height):
            for c in range(width):
                if (r, c) in corners:
                    continue
                if kind == "solid":
                    value = primary
                elif kind == "checker":
                    value = primary if (r + c) % 2 == 0 else secondary
                elif kind == "hstripe":
                    value = primary if r % 2 == 0 else secondary
                elif kind == "vstripe":
                    value = primary if c % 2 == 0 else secondary
                elif kind == "cross":
                    value = primary if (r == height // 2 or c == width // 2) else secondary
                elif kind == "frame":
                    value = primary if (r in (0, height - 1) or c in (0, width - 1)) else secondary
                else:
                    value = primary if (r - c) % 2 == 0 else secondary
                pattern[r, c] = value

        pattern[0, 0] = pattern[0, width - 1] = pattern[height - 1, 0] = pattern[height - 1, width - 1] = BG
        if int(np.count_nonzero(pattern != BG)) < 3:
            continue
        if len(set(map(int, pattern[pattern != BG].reshape(-1)))) == 0:
            continue
        return pattern
    raise RuntimeError("failed to build a visible legend pattern")


def _min_seed_color_count(n_pairs: int) -> int:
    for count in range(2, len(USABLE_COLORS) + 1):
        if math.comb(count, 2) >= n_pairs:
            return count
    raise RuntimeError("not enough colors to encode unique seed pairs")


def _legend_specs(dimensions: list[tuple[int, int]]):
    n_keys = len(dimensions)
    seed_count = _min_seed_color_count(n_keys)
    palette = random.sample(USABLE_COLORS, len(USABLE_COLORS))
    seed_colors = palette[:seed_count]
    pattern_colors = palette[seed_count:]
    pair_pool = list(combinations(seed_colors, 2))
    random.shuffle(pair_pool)
    pairs = [tuple(sorted(map(int, pair_pool[i]))) for i in range(n_keys)]

    used_patterns: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    specs = []
    for pair, (height, width) in zip(pairs, dimensions):
        for _ in range(80):
            pattern = _make_pattern(height, width, pattern_colors)
            signature = _pattern_signature(pattern)
            if signature in used_patterns:
                continue
            used_patterns.add(signature)
            specs.append({"pair": pair, "pattern": pattern})
            break
        else:
            return None
    return specs


def _assign_keys(main_positions: list[tuple[int, int]], key_count: int, orientation: str):
    shuffled = main_positions[:]
    random.shuffle(shuffled)
    assignments: dict[tuple[int, int], int] = {}
    for pos, key_idx in zip(shuffled, range(key_count)):
        assignments[pos] = key_idx
    for pos in shuffled[key_count:]:
        assignments[pos] = random.randrange(key_count)

    extras = shuffled[key_count:]
    if extras and key_count > 1:
        pos = random.choice(extras)
        forbidden = pos[0] if orientation == "right" else pos[1]
        choices = [key_idx for key_idx in range(key_count) if key_idx != forbidden]
        assignments[pos] = random.choice(choices)
    return assignments


def _sample_right_layout():
    n_rows = random.randint(2, 4)
    n_main_cols = random.randint(2, 3)
    legend_width = random.randint(3, 4)
    row_heights = [random.randint(3, 5) for _ in range(n_rows)]
    width_multipliers = [random.choice([1, 2]) for _ in range(n_main_cols)]
    if max(width_multipliers) == 1:
        width_multipliers[random.randrange(n_main_cols)] = 2
    main_widths = [legend_width * mult for mult in width_multipliers]

    total_height = sum(row_heights) + (n_rows - 1)
    total_width = sum(main_widths) + legend_width + n_main_cols
    if total_height > 30 or total_width > 30:
        return None
    return row_heights, main_widths, legend_width


def _sample_bottom_layout():
    n_main_rows = random.randint(2, 3)
    n_cols = random.randint(2, 4)
    legend_height = random.randint(3, 4)
    col_widths = [random.randint(3, 5) for _ in range(n_cols)]
    height_multipliers = [random.choice([1, 2]) for _ in range(n_main_rows)]
    if max(height_multipliers) == 1:
        height_multipliers[random.randrange(n_main_rows)] = 2
    main_heights = [legend_height * mult for mult in height_multipliers]

    total_height = sum(main_heights) + legend_height + n_main_rows
    total_width = sum(col_widths) + (n_cols - 1)
    if total_height > 30 or total_width > 30:
        return None
    return main_heights, col_widths, legend_height


def _render_right():
    layout = _sample_right_layout()
    if layout is None:
        return None
    row_heights, main_widths, legend_width = layout
    row_spans, sep_rows, total_h = _sizes_to_spans(row_heights)
    col_spans, sep_cols, total_w = _sizes_to_spans(main_widths + [legend_width])
    grid = np.full((total_h, total_w), BG, dtype=int)
    if sep_rows:
        grid[sep_rows, :] = SEPARATOR
    if sep_cols:
        grid[:, sep_cols] = SEPARATOR

    specs = _legend_specs([(height, legend_width) for height in row_heights])
    if specs is None:
        return None

    legend_col = len(col_spans) - 1
    for row_idx, spec in enumerate(specs):
        r0, r1 = row_spans[row_idx]
        c0, c1 = col_spans[legend_col]
        grid[r0 : r1 + 1, c0 : c1 + 1] = _place_markers(spec["pattern"], spec["pair"])

    inp = grid.copy()
    main_positions = [(r, c) for r in range(len(row_spans)) for c in range(len(main_widths))]
    assignments = _assign_keys(main_positions, len(specs), "right")
    for row_idx, col_idx in main_positions:
        spec = specs[assignments[(row_idx, col_idx)]]
        r0, r1 = row_spans[row_idx]
        c0, c1 = col_spans[col_idx]
        inp[r0 : r1 + 1, c0 : c1 + 1] = _seed_cell(r1 - r0 + 1, c1 - c0 + 1, spec["pair"])

    expected = inp.copy()
    for row_idx, col_idx in main_positions:
        spec = specs[assignments[(row_idx, col_idx)]]
        r0, r1 = row_spans[row_idx]
        c0, c1 = col_spans[col_idx]
        expected[r0 : r1 + 1, c0 : c1 + 1] = _tile_pattern(spec["pattern"], (r1 - r0 + 1, c1 - c0 + 1))
    return inp, expected


def _render_bottom():
    layout = _sample_bottom_layout()
    if layout is None:
        return None
    main_heights, col_widths, legend_height = layout
    row_spans, sep_rows, total_h = _sizes_to_spans(main_heights + [legend_height])
    col_spans, sep_cols, total_w = _sizes_to_spans(col_widths)
    grid = np.full((total_h, total_w), BG, dtype=int)
    if sep_rows:
        grid[sep_rows, :] = SEPARATOR
    if sep_cols:
        grid[:, sep_cols] = SEPARATOR

    specs = _legend_specs([(legend_height, width) for width in col_widths])
    if specs is None:
        return None

    legend_row = len(row_spans) - 1
    for col_idx, spec in enumerate(specs):
        r0, r1 = row_spans[legend_row]
        c0, c1 = col_spans[col_idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = _place_markers(spec["pattern"], spec["pair"])

    inp = grid.copy()
    main_positions = [(r, c) for r in range(len(main_heights)) for c in range(len(col_spans))]
    assignments = _assign_keys(main_positions, len(specs), "bottom")
    for row_idx, col_idx in main_positions:
        spec = specs[assignments[(row_idx, col_idx)]]
        r0, r1 = row_spans[row_idx]
        c0, c1 = col_spans[col_idx]
        inp[r0 : r1 + 1, c0 : c1 + 1] = _seed_cell(r1 - r0 + 1, c1 - c0 + 1, spec["pair"])

    expected = inp.copy()
    for row_idx, col_idx in main_positions:
        spec = specs[assignments[(row_idx, col_idx)]]
        r0, r1 = row_spans[row_idx]
        c0, c1 = col_spans[col_idx]
        expected[r0 : r1 + 1, c0 : c1 + 1] = _tile_pattern(spec["pattern"], (r1 - r0 + 1, c1 - c0 + 1))
    return inp, expected


def _gen_attempt():
    rendered = _render_right() if random.random() < 0.5 else _render_bottom()
    if rendered is None:
        return None
    inp, expected = rendered
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the legend-pattern tiling task")
