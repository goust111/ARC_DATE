from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _separator_row(g: np.ndarray) -> int:
    candidates: list[int] = []
    for r in range(g.shape[0]):
        row = g[r]
        if int(row[0]) == BLACK:
            continue
        if np.all(row == row[0]):
            candidates.append(r)
    assert len(candidates) == 1, "expected one solid non-black horizontal separator"
    return candidates[0]


def _extract_blocks(row: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    cols = np.flatnonzero(row != BLACK)
    assert cols.size > 0, "template row must contain colored blocks"
    left, right = int(cols[0]), int(cols[-1])
    assert np.all(row[left : right + 1] != BLACK), "template blocks must be contiguous"

    blocks: list[tuple[int, int]] = []
    seen: set[int] = set()
    c = left
    while c <= right:
        color = int(row[c])
        start = c
        while c <= right and int(row[c]) == color:
            c += 1
        assert color not in seen, "each template block color must appear once per row"
        seen.add(color)
        blocks.append((color, c - start))

    assert len(blocks) >= 2, "template row must contain multiple color blocks"
    return left, blocks


def _instruction_colors(row: np.ndarray) -> list[int]:
    cols = np.flatnonzero(row != BLACK)
    assert cols.size > 0, "instruction row must contain colored pixels"
    colors = [int(row[c]) for c in cols]
    assert len(colors) == len(set(colors)), "instruction colors must be unique per row"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each upper template row's color blocks using the left-to-right color order encoded below the separator.

    Input structure:
    - The background is black and one solid non-black horizontal row separates the grid into upper template rows and lower instruction rows.
    - Each upper row contains one contiguous run of differently colored block columns, with each block one or more adjacent same-color cells.
    - The matching lower row contains one single colored pixel for each upper block, using the same color set.

    Rule:
    - Pair upper and lower rows by their distance from the separator.
    - Read the colors of the lower row's non-black pixels from left to right.
    - Rebuild the upper row's contiguous block run in that color order, preserving each color block's original width and keeping the run's left edge fixed.
    - Return only the rebuilt upper part, omitting the separator and lower rows.

    Key insight:
    The lower sparse pixels are an ordering code for the upper color blocks; their positions matter only through their left-to-right color sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    upper = g[:sep]
    lower = g[sep + 1 :]
    assert upper.shape[0] == lower.shape[0] > 0, "upper and lower row counts must match"

    out = np.full_like(upper, BLACK)
    for r in range(upper.shape[0]):
        left, blocks = _extract_blocks(upper[r])
        widths_by_color = {color: width for color, width in blocks}
        order = _instruction_colors(lower[r])
        assert set(order) == set(widths_by_color), "paired rows must use the same color set"

        c = left
        for color in order:
            width = widths_by_color[color]
            out[r, c : c + width] = color
            c += width

    return out


def _spaced_positions(width: int, count: int) -> list[int]:
    assert width >= 2 * count - 1
    base = sorted(random.sample(range(width - count + 1), count))
    return [pos + i for i, pos in enumerate(base)]


def _render_expected(
    width: int,
    specs: list[tuple[int, list[tuple[int, int]], list[int]]],
) -> np.ndarray:
    out = np.full((len(specs), width), BLACK, dtype=int)
    for r, (left, blocks, order) in enumerate(specs):
        widths_by_color = {color: block_width for color, block_width in blocks}
        c = left
        for color in order:
            block_width = widths_by_color[color]
            out[r, c : c + block_width] = color
            c += block_width
    return out


def _gen_specs(row_count: int) -> tuple[int, list[tuple[int, list[tuple[int, int]], list[int]]]]:
    raw_specs: list[tuple[list[tuple[int, int]], list[int]]] = []
    used_color_sets: set[tuple[int, ...]] = set()
    any_reordered = False

    for _ in range(row_count):
        for _attempt in range(100):
            block_count = random.randint(3, 6)
            colors = random.sample(PALETTE, block_count)
            color_key = tuple(sorted(colors))
            if color_key not in used_color_sets:
                used_color_sets.add(color_key)
                break
        else:
            raise RuntimeError("failed to sample distinct row color sets")

        widths = [random.randint(1, 3) for _ in colors]
        blocks = list(zip(colors, widths))
        order = colors[:]
        random.shuffle(order)
        any_reordered = any_reordered or order != colors
        raw_specs.append((blocks, order))

    if not any_reordered:
        blocks, order = raw_specs[-1]
        order = order[1:] + order[:1]
        raw_specs[-1] = (blocks, order)

    min_width = 8
    for blocks, _order in raw_specs:
        min_width = max(min_width, sum(width for _color, width in blocks))
        min_width = max(min_width, 2 * len(blocks) - 1)
    width = random.randint(min_width, 30)

    specs: list[tuple[int, list[tuple[int, int]], list[int]]] = []
    for blocks, order in raw_specs:
        block_width = sum(width for _color, width in blocks)
        left = random.randint(0, width - block_width)
        specs.append((left, blocks, order))
    return width, specs


def _build_input(
    width: int,
    specs: list[tuple[int, list[tuple[int, int]], list[int]]],
) -> np.ndarray:
    row_count = len(specs)
    grid = np.full((2 * row_count + 1, width), BLACK, dtype=int)
    grid[row_count, :] = GRAY

    for r, (left, blocks, order) in enumerate(specs):
        c = left
        for color, block_width in blocks:
            grid[r, c : c + block_width] = color
            c += block_width

        positions = _spaced_positions(width, len(order))
        for c, color in zip(positions, order):
            grid[row_count + 1 + r, c] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(2, 6)
    width, specs = _gen_specs(row_count)
    grid = _build_input(width, specs)
    expected = _render_expected(width, specs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
