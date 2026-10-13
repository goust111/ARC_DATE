from __future__ import annotations

import itertools
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return 0 <= int(r) < height and 0 <= int(c) < width


def _find_solid_blocks(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    blocks: list[tuple[int, int, int, tuple[int, int, int, int]]] = []

    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(g[row, col]) == int(bg):
                continue
            color = int(g[row, col])
            stack = [(int(row), int(col))]
            seen[row, col] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if (
                        _in_bounds(g.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_coords(arr)
            if (
                len(coords) == 9
                and r1 - r0 == 2
                and c1 - c0 == 2
                and c0 >= 2
                and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)
            ):
                blocks.append((r0 + 1, c0 + 1, color, (r0, c0, r1, c1)))

    blocks.sort(key=lambda item: (item[0], item[1], item[2]))
    return blocks


def _intersection_positions(blocks: list[tuple[int, int, int, tuple[int, int, int, int]]]) -> set[tuple[int, int]]:
    return {
        (int(row_i), int(col_j))
        for i, (row_i, _col_i, _color_i, _bbox_i) in enumerate(blocks)
        for j, (_row_j, col_j, _color_j, _bbox_j) in enumerate(blocks)
        if i != j
    }


def _apply_rule(
    grid: np.ndarray,
    blocks: list[tuple[int, int, int, tuple[int, int, int, int]]],
    *,
    intersection_mode: str = "legend",
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    height, width = g.shape
    out = g.copy()

    for row, col, color, _bbox in blocks:
        for cc in range(2, width):
            if int(out[row, cc]) == int(bg):
                out[row, cc] = color
        for rr in range(height):
            if int(out[rr, col]) == int(bg):
                out[rr, col] = color

    intersections = _intersection_positions(blocks)
    row_to_color = {int(row): int(color) for row, _col, color, _bbox in blocks}
    col_to_color = {int(col): int(color) for _row, col, color, _bbox in blocks}

    for row, col in intersections:
        out[row, col] = bg

    if intersection_mode == "background":
        return out

    for row, col in intersections:
        if intersection_mode == "legend":
            out[row, col] = int(g[row, 0])
        elif intersection_mode == "row":
            out[row, col] = row_to_color[int(row)]
        elif intersection_mode == "column":
            out[row, col] = col_to_color[int(col)]
        else:
            raise ValueError(f"unknown intersection_mode: {intersection_mode}")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each interior 3x3 solid block into a cross, clear cross-over holes, then refill those holes from the left legend column.

    Input structure:
    - The background is black and column 0 is a full-height non-black legend string.
    - Column 1 is a black separator, and columns 2..end contain only 2 to 4 solid monochrome 3x3 blocks.
    - The blocks have distinct center rows and distinct center columns, stay at least one cell away from the outer border, and each block row has a different legend color in column 0.

    Rule:
    - Copy the input unchanged.
    - For each 3x3 block, paint its color along its center row across columns 2..end and along its center column across all rows, but only onto background cells; the original 3x3 block stays unchanged.
    - Every crossing between one block's center row and another block's center column is reset to background, then recolored with the legend value from column 0 on that crossing row.

    Key insight:
    - The generated cross intersections are not final pixels: they must first be discovered by the cross simulation and then decoded by row lookup into the left-edge legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 7 and width >= 8, "grid is too small for the described layout"

    bg = _get_bg(g)
    assert int(bg) == 0, "expected black background"
    assert np.all(g[:, 1] == bg), "expected a black separator in column 1"
    assert np.all(g[:, 0] != bg), "expected a non-black legend column in column 0"
    legend_colors = {int(v) for v in g[:, 0]}
    assert 2 <= len(legend_colors) <= 4, "expected a 2 to 4 color legend"

    blocks = _find_solid_blocks(g, bg)
    assert 2 <= len(blocks) <= 4, "expected 2 to 4 solid 3x3 blocks"

    rows = [int(row) for row, _col, _color, _bbox in blocks]
    cols = [int(col) for _row, col, _color, _bbox in blocks]
    assert len(set(rows)) == len(rows), "expected distinct block center rows"
    assert len(set(cols)) == len(cols), "expected distinct block center columns"

    for i, (row_i, col_i, color_i, bbox_i) in enumerate(blocks):
        r0, c0, r1, c1 = bbox_i
        assert r0 >= 1 and r1 <= height - 2, "expected a one-cell vertical border gap"
        assert c0 >= 3 and c1 <= width - 2, "expected the blocks to stay inside the main area"
        assert int(g[row_i, 0]) != int(color_i), "legend lookup color must differ from the block color on its row"
        for row_j, col_j, _color_j, _bbox_j in blocks[i + 1 :]:
            assert abs(int(row_i) - int(row_j)) >= 3, "expected separated row bands"
            assert abs(int(col_i) - int(col_j)) >= 3, "expected separated column bands"

    return _apply_rule(g, blocks, intersection_mode="legend")


HEIGHT_RANGE = (14, 19)
WIDTH_RANGE = (16, 21)
BLOCK_COUNT_RANGE = (2, 4)
LEGEND_COLOR_COUNT_RANGE = (2, 4)
MIN_CENTER_ROW = 2
MIN_CENTER_COL = 4
MIN_CENTER_SEP = 3
MAX_ATTEMPTS = 400
ALL_COLORS = tuple(range(1, 10))


def _spaced_positions(start: int, stop: int, count: int, min_sep: int) -> list[int] | None:
    positions = list(range(int(start), int(stop)))
    combos = [
        combo
        for combo in itertools.combinations(positions, int(count))
        if all(b - a >= int(min_sep) for a, b in zip(combo, combo[1:]))
    ]
    if not combos:
        return None
    chosen = list(random.choice(combos))
    random.shuffle(chosen)
    return [int(v) for v in chosen]


def _assign_block_colors(count: int, pool: list[int]) -> list[int]:
    distinct_count = random.randint(1, min(3, count, len(pool)))
    palette = random.sample(pool, distinct_count)
    colors = [int(palette[i % distinct_count]) for i in range(count)]
    random.shuffle(colors)
    return colors


def _build_legend(height: int, palette: list[int], block_rows: list[int]) -> np.ndarray:
    legend = np.array([random.choice(palette) for _ in range(height)], dtype=int)

    rows = list(range(height))
    random.shuffle(rows)
    for color, row in zip(palette, rows):
        legend[row] = int(color)

    anchor_colors = [random.choice(palette)]
    anchor_colors.append(random.choice([c for c in palette if c != anchor_colors[0]]))
    while len(anchor_colors) < len(block_rows):
        anchor_colors.append(random.choice(palette))
    random.shuffle(anchor_colors)

    for row, color in zip(block_rows, anchor_colors):
        legend[int(row)] = int(color)

    assert len({int(legend[row]) for row in block_rows}) >= 2
    return legend


def _render_input(
    height: int,
    width: int,
    legend: np.ndarray,
    blocks: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[:, 0] = legend
    grid[:, 1] = 0
    for row, col, color in blocks:
        grid[row - 1 : row + 2, col - 1 : col + 2] = int(color)
    return grid


def _validate_instance(grid: np.ndarray, blocks: list[tuple[int, int, int]]) -> None:
    detected = [(row, col, color) for row, col, color, _bbox in _find_solid_blocks(grid, 0)]
    expected_blocks = sorted((int(row), int(col), int(color)) for row, col, color in blocks)
    assert detected == expected_blocks, "block detection must match the generated layout"

    actual = main(grid)
    detected_full = _find_solid_blocks(grid, 0)
    expected = _apply_rule(grid, detected_full, intersection_mode="legend")
    cleared = _apply_rule(grid, detected_full, intersection_mode="background")
    row_filled = _apply_rule(grid, detected_full, intersection_mode="row")
    col_filled = _apply_rule(grid, detected_full, intersection_mode="column")

    assert np.array_equal(actual, expected), "main must match the rule-level simulation"
    assert not np.array_equal(expected, cleared), "legend refill step must affect the result"
    assert not np.array_equal(expected, row_filled), "row-block shortcut must fail"
    assert not np.array_equal(expected, col_filled), "column-block shortcut must fail"

    intersections = sorted(_intersection_positions(detected_full))
    assert len(intersections) == len(detected_full) * (len(detected_full) - 1)
    fill_colors = {int(expected[row, col]) for row, col in intersections}
    assert len(fill_colors) >= 2, "intersection rows must decode to at least two legend colors"

    row_to_block = {int(row): int(color) for row, _col, color in blocks}
    col_to_block = {int(col): int(color) for _row, col, color in blocks}
    for row, col in intersections:
        legend_color = int(grid[row, 0])
        assert int(expected[row, col]) == legend_color
        assert legend_color != row_to_block[int(row)]
        assert legend_color != col_to_block[int(col)]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    count = random.randint(*BLOCK_COUNT_RANGE)

    rows = _spaced_positions(MIN_CENTER_ROW, height - 2, count, MIN_CENTER_SEP)
    cols = _spaced_positions(MIN_CENTER_COL, width - 2, count, MIN_CENTER_SEP)
    if rows is None or cols is None:
        return None

    legend_color_count = random.randint(*LEGEND_COLOR_COUNT_RANGE)
    legend_palette = random.sample(list(ALL_COLORS), legend_color_count)
    remaining = [color for color in ALL_COLORS if color not in legend_palette]
    if not remaining:
        return None

    block_colors = _assign_block_colors(count, remaining)
    legend = _build_legend(height, legend_palette, rows)
    blocks = [(int(row), int(col), int(color)) for row, col, color in zip(rows, cols, block_colors)]
    grid = _render_input(height, width, legend, blocks)

    if _get_bg(grid) != 0:
        return None

    try:
        _validate_instance(grid, blocks)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
