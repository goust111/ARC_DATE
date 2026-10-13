from __future__ import annotations

import numpy as np

from arc_utils import colors_present, components_mask, get_bg
from itertools import combinations
import random


def _ranges_between(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    points = [-1] + list(dividers) + [int(size)]
    return [
        (points[i] + 1, points[i + 1] - 1)
        for i in range(len(points) - 1)
        if points[i] + 1 <= points[i + 1] - 1
    ]


def _divider_grid(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    height, width = g.shape
    candidates: list[tuple[int, list[int], list[int]]] = []

    for color in colors_present(g, exclude={bg}):
        full_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if full_rows and full_cols:
            candidates.append((int(color), full_rows, full_cols))

    assert len(candidates) == 1, "expected one color forming both horizontal and vertical dividers"
    return candidates[0]


def _cell_colors(
    g: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    bg: int,
    divider: int,
) -> np.ndarray:
    cells = np.full((len(row_ranges), len(col_ranges)), -1, dtype=int)
    for cell_r, (r0, r1) in enumerate(row_ranges):
        for cell_c, (c0, c1) in enumerate(col_ranges):
            vals = set(map(int, g[r0 : r1 + 1, c0 : c1 + 1].ravel())) - {bg}
            assert divider not in vals, "divider color must not appear inside a cell"
            if vals:
                assert len(vals) == 1, "each occupied cell must contain one sprite color"
                cells[cell_r, cell_c] = vals.pop()
    return cells


def _reflected_to_missing(
    g: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    have: set[tuple[int, int]],
    missing: tuple[int, int],
) -> np.ndarray:
    missing_r, missing_c = missing
    predictions = []
    for cell_r, cell_c in have:
        r0, r1 = row_ranges[cell_r]
        c0, c1 = col_ranges[cell_c]
        pred = g[r0 : r1 + 1, c0 : c1 + 1]
        if cell_r != missing_r:
            pred = pred[::-1, :]
        if cell_c != missing_c:
            pred = pred[:, ::-1]
        predictions.append(pred)

    first = predictions[0]
    assert all(np.array_equal(first, pred) for pred in predictions[1:]), (
        "the three reflected source sprites must agree on the missing sprite"
    )
    return first.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each 2-by-2 lattice-cell quartet by adding its missing reflected sprite.

    Input structure:
    - The background is the most common color.
    - One non-background color forms complete horizontal and vertical divider
      lines, partitioning the grid into rectangular cells.
    - Every occupied cell away from the dividers contains pixels of exactly one
      non-background, non-divider color.
    - For each sprite color, occupied cells occur as connected three-cell groups
      inside a 2-by-2 block of equal-sized cells.

    Rule:
    - Split the grid at the divider rows and columns.
    - For each three-cell monochrome group in a 2-by-2 cell block, find the one
      empty cell in that block.
    - Reflect each of the three present cell-local sprites vertically and/or
      horizontally according to its relative position to the missing cell; all
      three reflections must produce the same mask.
    - Paint that agreed reflected sprite into the missing cell, leaving dividers
      and all existing cells unchanged.

    Key insight:
    - The sprites are mirror variants around the center of a 2-by-2 block of
      lattice cells, so any three visible variants uniquely determine the fourth.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    divider, divider_rows, divider_cols = _divider_grid(g, bg)
    row_ranges = _ranges_between(g.shape[0], divider_rows)
    col_ranges = _ranges_between(g.shape[1], divider_cols)
    assert len(row_ranges) >= 2 and len(col_ranges) >= 2, "expected at least one 2-by-2 cell block"

    cells = _cell_colors(g, row_ranges, col_ranges, bg, divider)
    out = g.copy()

    for color in colors_present(g, exclude={bg, divider}):
        for comp in components_mask(cells == color, connectivity=4):
            assert comp.shape[0] == 3, "each sprite-color group must contain three occupied cells"
            group_r0, group_c0 = map(int, comp.min(axis=0))
            group_r1, group_c1 = map(int, comp.max(axis=0))
            assert group_r1 - group_r0 == 1 and group_c1 - group_c0 == 1, (
                "each sprite-color group must fit one 2-by-2 cell block"
            )

            block = {
                (r, c)
                for r in (group_r0, group_r1)
                for c in (group_c0, group_c1)
            }
            have = {(int(r), int(c)) for r, c in comp}
            missing_cells = block - have
            assert len(missing_cells) == 1, "expected one missing cell in the 2-by-2 block"
            missing = next(iter(missing_cells))
            missing_r, missing_c = missing
            assert cells[missing_r, missing_c] == -1, "missing quartet cell must be empty"

            heights = {row_ranges[r][1] - row_ranges[r][0] + 1 for r, _ in block}
            widths = {col_ranges[c][1] - col_ranges[c][0] + 1 for _, c in block}
            assert len(heights) == 1 and len(widths) == 1, "quartet cells must be equal-sized"

            pred = _reflected_to_missing(g, row_ranges, col_ranges, have, missing)
            assert set(map(int, pred.ravel())) <= {bg, int(color)}, (
                "reflected sprite must contain only background and its color"
            )

            r0, r1 = row_ranges[missing_r]
            c0, c1 = col_ranges[missing_c]
            assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == bg), "target cell must be empty"
            out[r0 : r1 + 1, c0 : c1 + 1] = pred

    return out

# --------------- generator (o5x) ---------------

# Canvas and lattice geometry.
BACKGROUND_COLOR = 0
CANVAS_HEIGHT_MIN, CANVAS_HEIGHT_MAX = 17, 26
CANVAS_WIDTH_MIN, CANVAS_WIDTH_MAX = 18, 23
CELL_SIZE_CHOICES = (4, 5)
DIVIDER_ROW_CHOICES = (3, 4)
DIVIDER_COL_COUNT = 3
TRUNCATED_EDGE_SIZE_CHOICES = (2, 3)

# Sprite groups and colors.
GROUP_COUNT_CHOICES = (1, 2, 3)
COLOR_POOL = range(1, 10)
SPRITE_PIXEL_MIN, SPRITE_PIXEL_MAX = 4, 6

# Local geometry and retry limits.
BLOCK_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_RETRIES = 300


def _gen_cell_ranges(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    ranges = []
    dividers = []
    pos = 0
    for i, size in enumerate(sizes):
        ranges.append((pos, pos + size - 1))
        pos += size
        if i != len(sizes) - 1:
            dividers.append(pos)
            pos += 1
    return ranges, dividers, pos


def _gen_block_cells(block: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = block
    return {(r + dr, c + dc) for dr, dc in BLOCK_OFFSETS}


def _gen_block_combos(row_sizes, col_sizes, cell_size, num_groups):
    starts = [
        (r, c)
        for r in range(len(row_sizes) - 1)
        for c in range(len(col_sizes) - 1)
        if row_sizes[r] == cell_size
        and row_sizes[r + 1] == cell_size
        and col_sizes[c] == cell_size
        and col_sizes[c + 1] == cell_size
    ]
    valid = []
    for combo in combinations(starts, num_groups):
        occupied: set[tuple[int, int]] = set()
        ok = True
        for block in combo:
            cells = _gen_block_cells(block)
            if occupied & cells:
                ok = False
                break
            occupied |= cells
        if ok:
            valid.append(combo)
    return valid


def _gen_layout_options(cell_size, num_divider_rows, num_groups):
    row_count = num_divider_rows + 1
    col_count = DIVIDER_COL_COUNT + 1
    last_sizes = [cell_size] + [s for s in TRUNCATED_EDGE_SIZE_CHOICES if s < cell_size]
    options = []

    for row_last in last_sizes:
        row_sizes = [cell_size] * (row_count - 1) + [row_last]
        height = sum(row_sizes) + num_divider_rows
        if not CANVAS_HEIGHT_MIN <= height <= CANVAS_HEIGHT_MAX:
            continue

        for col_last in last_sizes:
            col_sizes = [cell_size] * (col_count - 1) + [col_last]
            width = sum(col_sizes) + DIVIDER_COL_COUNT
            if not CANVAS_WIDTH_MIN <= width <= CANVAS_WIDTH_MAX:
                continue

            combos = _gen_block_combos(row_sizes, col_sizes, cell_size, num_groups)
            if combos:
                options.append((row_sizes, col_sizes, combos))

    return options


def _gen_sprite(cell_size: int, pixel_count: int) -> np.ndarray:
    rows = [r for r in range(cell_size) if r != cell_size - 1 - r]
    cols = [c for c in range(cell_size) if c != cell_size - 1 - c]
    anchor = (random.choice(rows), random.choice(cols))
    banned = {
        (anchor[0], cell_size - 1 - anchor[1]),
        (cell_size - 1 - anchor[0], anchor[1]),
    }
    cells = {anchor}

    while len(cells) < pixel_count:
        frontier = []
        for r, c in cells:
            for dr, dc in ORTHOGONAL_STEPS:
                nr, nc = r + dr, c + dc
                pos = (nr, nc)
                if (
                    0 <= nr < cell_size
                    and 0 <= nc < cell_size
                    and pos not in cells
                    and pos not in banned
                ):
                    frontier.append(pos)
        cells.add(random.choice(frontier))

    sprite = np.zeros((cell_size, cell_size), dtype=bool)
    for r, c in cells:
        sprite[r, c] = True

    assert not np.array_equal(sprite, sprite[:, ::-1])
    assert not np.array_equal(sprite, sprite[::-1, :])
    return sprite


def _gen_variant(sprite: np.ndarray, dr: int, dc: int) -> np.ndarray:
    variant = sprite
    if dr:
        variant = variant[::-1, :]
    if dc:
        variant = variant[:, ::-1]
    return variant.copy()


def _gen_cell_slice(row_ranges, col_ranges, cell_r, cell_c) -> tuple[slice, slice]:
    r0, r1 = row_ranges[cell_r]
    c0, c1 = col_ranges[cell_c]
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _gen_paint_cell(grid, row_ranges, col_ranges, cell_r, cell_c, mask, color) -> None:
    cell = grid[_gen_cell_slice(row_ranges, col_ranges, cell_r, cell_c)]
    cell[mask] = color


def _gen_clear_cell(grid, row_ranges, col_ranges, cell_r, cell_c) -> None:
    grid[_gen_cell_slice(row_ranges, col_ranges, cell_r, cell_c)] = BACKGROUND_COLOR


def _gen_choose_layout():
    cell_size = random.choice(CELL_SIZE_CHOICES)
    num_divider_rows = random.choice(DIVIDER_ROW_CHOICES)
    num_groups = random.choice(GROUP_COUNT_CHOICES)

    layout_options = _gen_layout_options(cell_size, num_divider_rows, num_groups)
    if not layout_options:
        return None

    row_sizes, col_sizes, block_combos = random.choice(layout_options)
    blocks = random.choice(block_combos)
    return cell_size, num_groups, row_sizes, col_sizes, blocks


def _gen_render_sprite_block(output, row_ranges, col_ranges, block, color, cell_size):
    pixel_count = random.randint(SPRITE_PIXEL_MIN, SPRITE_PIXEL_MAX)
    sprite = _gen_sprite(cell_size, pixel_count)
    block_r, block_c = block
    cells = []
    for dr, dc in BLOCK_OFFSETS:
        cell = (block_r + dr, block_c + dc)
        _gen_paint_cell(
            output,
            row_ranges,
            col_ranges,
            cell[0],
            cell[1],
            _gen_variant(sprite, dr, dc),
            color,
        )
        cells.append(cell)
    return cells


def _gen_render_groups(output, row_ranges, col_ranges, blocks, sprite_colors, cell_size):
    group_cells = []
    for block, color in zip(blocks, sprite_colors):
        group_cells.append(
            _gen_render_sprite_block(output, row_ranges, col_ranges, block, color, cell_size)
        )
    return group_cells


def _gen_attempt() -> np.ndarray | None:
    layout = _gen_choose_layout()
    if layout is None:
        return None

    cell_size, num_groups, row_sizes, col_sizes, blocks = layout
    row_ranges, divider_rows, height = _gen_cell_ranges(row_sizes)
    col_ranges, divider_cols, width = _gen_cell_ranges(col_sizes)

    colors = random.sample(COLOR_POOL, num_groups + 1)
    divider = colors[0]
    sprite_colors = colors[1:]

    output = np.zeros((height, width), dtype=int)
    for r in divider_rows:
        output[r, :] = divider
    for c in divider_cols:
        output[:, c] = divider

    group_cells = _gen_render_groups(output, row_ranges, col_ranges, blocks, sprite_colors, cell_size)

    inp = output.copy()
    for cells in group_cells:
        missing_r, missing_c = random.choice(cells)
        _gen_clear_cell(inp, row_ranges, col_ranges, missing_r, missing_c)

    try:
        predicted = main(inp)
    except Exception:
        return None
    if not np.array_equal(predicted, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 2546ccf6")
