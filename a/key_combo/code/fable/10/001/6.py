from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
GLYPH_MASK = np.array(
    [
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
    ],
    dtype=bool,
)
GLYPH_SIZE = 3
GLYPH_RESERVED = 4
OPS = tuple(range(8))
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_op(arr: np.ndarray, op: int) -> np.ndarray:
    out = np.asarray(arr)
    if op >= 4:
        out = np.fliplr(out)
    return np.rot90(out, k=op % 4).copy()


def _divider_color_and_lines(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    height, width = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1, "expected one divider color"
    return candidates[0]


def _cell_ranges(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    assert dividers, "expected divider lines"
    ranges = []
    points = list(dividers) + [int(size)]
    for i, d0 in enumerate(dividers):
        d1 = points[i + 1]
        r0 = d0 + 1
        r1 = d1 - 1
        assert r0 <= r1, "divider spacing must leave a cell interval"
        ranges.append((r0, r1))
    return ranges


def _reference_glyph(grid: np.ndarray, top_div: int, left_div: int) -> np.ndarray:
    bay = grid[:top_div, :left_div]
    assert bay.size > 0, "expected a top-left reference bay"
    assert set(map(int, np.unique(bay))) <= {BG, GREY}, "reference bay must contain only background and grey"
    return _crop_mask(bay == GREY)


def _match_op(reference_mask: np.ndarray, target_mask: np.ndarray) -> int:
    matches = [op for op in OPS if np.array_equal(_apply_op(reference_mask, op), target_mask)]
    assert len(matches) == 1, "glyph pose difference must define a unique operation"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each cell's grey glyph pose difference from the reference glyph to transform that cell's sprite.

    Input structure:
    - The background is black.
    - One non-background color forms full horizontal and vertical divider lines;
      the first divider row and first divider column leave a top-left bay that
      contains one isolated grey reference glyph.
    - The region below and to the right of those first dividers is partitioned
      into equal square cells.
    - Every cell contains one colored sprite plus one grey copy of the same
      chiral glyph in some rotated or mirrored pose.

    Rule:
    - Read the grey glyph in the top-left bay as the reference pose.
    - In each cell, compare the local grey glyph to the reference and find the
      unique rotation-or-reflection that maps the reference glyph to that pose.
    - Apply that same transformation to the colored sprite using the whole cell
      as the transformation canvas.
    - Remove all grey glyphs, keep the divider lattice unchanged, and write the
      transformed sprite back into its original cell.

    Key insight:
    - The task never states an explicit symbol-to-operation codebook: the needed
      operation is exactly the pose difference between the reference glyph and
      the glyph shown inside the current cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    assert bg == BG, "expected black background"
    divider, divider_rows, divider_cols = _divider_color_and_lines(g, bg)
    row_ranges = _cell_ranges(g.shape[0], divider_rows)
    col_ranges = _cell_ranges(g.shape[1], divider_cols)
    assert len(row_ranges) >= 2 and len(col_ranges) >= 2, "expected a multi-cell lattice"

    cell_heights = {r1 - r0 + 1 for r0, r1 in row_ranges}
    cell_widths = {c1 - c0 + 1 for c0, c1 in col_ranges}
    assert len(cell_heights) == 1 and len(cell_widths) == 1, "cells must be equal-sized"
    assert next(iter(cell_heights)) == next(iter(cell_widths)), "cells must be square for rotation"

    reference_mask = _reference_glyph(g, divider_rows[0], divider_cols[0])
    assert reference_mask.shape == (GLYPH_SIZE, GLYPH_SIZE), "unexpected reference glyph size"

    out = g.copy()
    out[g == GREY] = bg

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(cell)))
            assert divider not in colors, "divider color must not appear inside a cell"
            assert GREY in colors, "every cell must contain a grey glyph"

            local_glyph = _crop_mask(cell == GREY)
            assert local_glyph.shape == reference_mask.shape, "local glyph size must match reference"
            op = _match_op(reference_mask, local_glyph)

            sprite_colors = colors - {bg, GREY}
            assert len(sprite_colors) == 1, "each cell must contain exactly one sprite color"
            sprite_canvas = cell.copy()
            sprite_canvas[sprite_canvas == GREY] = bg

            transformed = _apply_op(sprite_canvas, op)
            out[r0 : r1 + 1, c0 : c1 + 1] = transformed

    return out


assert len({tuple(_apply_op(GLYPH_MASK.astype(int), op).ravel()) for op in OPS}) == 8


def _build_layout(
    cell_size: int, cell_rows: int, cell_cols: int, top_margin: int, left_margin: int
) -> tuple[tuple[int, int], list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    divider_rows = [top_margin + i * (cell_size + 1) for i in range(cell_rows)]
    divider_cols = [left_margin + i * (cell_size + 1) for i in range(cell_cols)]
    row_ranges = _cell_ranges(top_margin + cell_rows * cell_size + cell_rows, divider_rows)
    col_ranges = _cell_ranges(left_margin + cell_cols * cell_size + cell_cols, divider_cols)
    return (
        (row_ranges[-1][1] + 1, col_ranges[-1][1] + 1),
        divider_rows,
        divider_cols,
        row_ranges,
        col_ranges,
    )


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    target[patch.astype(bool)] = color


def _random_connected_sprite(cell_size: int, color: int) -> np.ndarray | None:
    allowed = [
        (r, c)
        for r in range(cell_size)
        for c in range(cell_size)
        if not (r < GLYPH_RESERVED and c < GLYPH_RESERVED)
    ]
    start = random.choice(allowed)
    pixels = {start}
    target_count = random.randint(5, min(8, len(allowed)))

    while len(pixels) < target_count:
        frontier = []
        for r, c in pixels:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in pixels:
                    continue
                if 0 <= nr < cell_size and 0 <= nc < cell_size and (nr, nc) in allowed:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        pixels.add(random.choice(frontier))

    canvas = np.zeros((cell_size, cell_size), dtype=int)
    for r, c in pixels:
        canvas[r, c] = color

    box = _bbox(canvas == color)
    assert box is not None
    r0, c0, r1, c1 = box
    if r1 == r0 or c1 == c0:
        return None
    variants = {tuple(_apply_op(canvas, op).ravel()) for op in OPS}
    if len(variants) != 8:
        return None
    return canvas


def _expected_output_from_spec(
    inp: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    cell_ops: list[int],
) -> np.ndarray:
    out = inp.copy()
    out[out == GREY] = BG
    op_iter = iter(cell_ops)
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            op = next(op_iter)
            cell = inp[r0 : r1 + 1, c0 : c1 + 1].copy()
            cell[cell == GREY] = BG
            out[r0 : r1 + 1, c0 : c1 + 1] = _apply_op(cell, op)
    return out


def _sample_cell_ops(n_cells: int) -> list[int]:
    assert n_cells >= 4
    proper_rotation = random.choice([1, 2, 3])
    mirror = random.choice([4, 5, 6, 7])
    chosen = [0, proper_rotation, mirror]
    remaining = [op for op in OPS if op not in chosen]
    extra_needed = min(n_cells, 6) - len(chosen)
    chosen.extend(random.sample(remaining, extra_needed))
    while len(chosen) < n_cells:
        chosen.append(random.choice(OPS))
    random.shuffle(chosen)
    assert any(op in {1, 2, 3} for op in chosen)
    assert any(op >= 4 for op in chosen)
    return chosen


def _attempt_generate() -> np.ndarray | None:
    cell_rows, cell_cols = random.choice([(2, 2), (2, 3), (3, 2)])
    cell_size = random.choice([6, 7])
    top_margin = random.choice([4, 5])
    left_margin = random.choice([4, 5])
    shape, divider_rows, divider_cols, row_ranges, col_ranges = _build_layout(
        cell_size, cell_rows, cell_cols, top_margin, left_margin
    )
    if max(shape) > 30:
        return None

    n_cells = cell_rows * cell_cols
    chosen_colors = random.sample(PALETTE, n_cells + 1)
    divider = chosen_colors[0]
    sprite_colors = chosen_colors[1:]
    ref_op = random.choice([1, 2, 3, 4, 5, 6, 7])
    ref_mask = _apply_op(GLYPH_MASK.astype(int), ref_op).astype(bool)
    cell_ops = _sample_cell_ops(n_cells)

    grid = np.zeros(shape, dtype=int)
    grid[divider_rows, :] = divider
    grid[:, divider_cols] = divider

    ref_r = random.randint(0, divider_rows[0] - GLYPH_SIZE)
    ref_c = random.randint(0, divider_cols[0] - GLYPH_SIZE)
    _place_patch(grid, ref_mask, ref_r, ref_c, GREY)

    op_iter = iter(cell_ops)
    absolute_ops = []
    for (r0, r1), (c0, c1), color in zip(
        [rr for rr in row_ranges for _ in col_ranges],
        col_ranges * len(row_ranges),
        sprite_colors,
    ):
        op = next(op_iter)
        sprite = _random_connected_sprite(cell_size, color)
        if sprite is None:
            return None
        cell = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(cell != BG):
            return None
        cell[:] = sprite
        glyph_mask = _apply_op(ref_mask.astype(int), op).astype(bool)
        _place_patch(cell, glyph_mask, 0, 0, GREY)
        if np.any((sprite != BG) & (cell == GREY)):
            return None
        local_mask = _crop_mask(cell == GREY)
        if _match_op(ref_mask, local_mask) != op:
            return None
        absolute_matches = [
            abs_op
            for abs_op in OPS
            if np.array_equal(_apply_op(GLYPH_MASK.astype(int), abs_op).astype(bool), local_mask)
        ]
        if len(absolute_matches) != 1:
            return None
        absolute_ops.append(absolute_matches[0])

    if all(abs_op == rel_op for abs_op, rel_op in zip(absolute_ops, cell_ops)):
        return None

    expected = _expected_output_from_spec(grid, row_ranges, col_ranges, cell_ops)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pose-difference puzzle input")
