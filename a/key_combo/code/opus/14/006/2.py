from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
CONTENT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    factor = int(factor)
    if factor <= 0:
        raise ValueError("scale factor must be positive")
    return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))


def _non_divider_spans(dividers: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_divider in enumerate(dividers.astype(bool).tolist()):
        if not is_divider and start is None:
            start = idx
        elif is_divider and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(dividers)))
    return spans


def _extract_pattern(grid: np.ndarray) -> np.ndarray:
    divider_rows = np.all(grid == RED, axis=1)
    divider_cols = np.all(grid == RED, axis=0)

    row_spans = _non_divider_spans(divider_rows)
    col_spans = _non_divider_spans(divider_cols)
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected a partition grid"

    row_heights = {r1 - r0 for r0, r1 in row_spans}
    col_widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_heights) == 1 and len(col_widths) == 1, "expected equal-sized regions"

    pattern = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            region = grid[r0:r1, c0:c1]
            colors = [int(v) for v in np.unique(region) if int(v) not in (BLACK, RED)]
            assert len(colors) <= 1, "each region may contain at most one content color"
            if colors:
                assert int(np.count_nonzero(region == colors[0])) == 1, "content marker must be a single pixel"
                pattern[rr, cc] = colors[0]
    return pattern


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a red-partitioned grid into a low-resolution pattern and enlarge it by the number of filled regions.

    Input structure:
    - A black background is cut by full red divider rows and columns into an R x C array of equal rectangular regions.
    - Each region is either empty or contains exactly one non-black, non-red marker pixel.
    - Divider lines are one cell thick and span the full width or height.

    Rule:
    - Detect the full red rows and columns to recover the region layout.
    - Build an R x C pattern where each region contributes its marker color, or black if the region is empty.
    - Count the non-black cells in that pattern and use that count as the integer pixel-upscale factor.
    - Output only the enlarged pattern, without the red dividers.

    Key insight:
    - The same partition structure encodes both the image content and the scale factor, so every region must be inspected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pattern = _extract_pattern(g)
    scale = int(np.count_nonzero(pattern))
    assert scale >= 1, "expected at least one filled region"
    return _scale_sprite(pattern, scale)


def _choose_region_size(parts: int) -> int:
    max_size = (30 - (parts - 1)) // parts
    choices = [size for size in range(3, max_size + 1, 2)]
    if not choices:
        raise ValueError("no valid region sizes")
    return random.choice(choices)


def _sample_positions(rows: int, cols: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(100):
        chosen = random.sample(cells, count)
        used_rows = {r for r, _ in chosen}
        used_cols = {c for _, c in chosen}
        if rows > 1 and count > 1 and len(used_rows) < 2:
            continue
        if cols > 1 and count > 1 and len(used_cols) < 2:
            continue
        return chosen
    return random.sample(cells, count)


def _sample_colors(count: int) -> list[int]:
    palette_size = random.randint(1, min(4, count, len(CONTENT_COLORS)))
    palette = random.sample(CONTENT_COLORS, palette_size)
    colors = palette.copy()
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _build_input(rows: int, cols: int, cell_h: int, cell_w: int, filled: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)
    grid = np.zeros((height, width), dtype=int)

    row_spans: list[tuple[int, int]] = []
    cursor = 0
    for r in range(rows):
        row_spans.append((cursor, cursor + cell_h))
        cursor += cell_h
        if r < rows - 1:
            grid[cursor, :] = RED
            cursor += 1

    col_spans: list[tuple[int, int]] = []
    cursor = 0
    for c in range(cols):
        col_spans.append((cursor, cursor + cell_w))
        cursor += cell_w
        if c < cols - 1:
            grid[:, cursor] = RED
            cursor += 1

    for (rr, cc), color in zip(filled, colors):
        r0, r1 = row_spans[rr]
        c0, c1 = col_spans[cc]
        grid[r0 + cell_h // 2, c0 + cell_w // 2] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        cell_h = _choose_region_size(rows)
        cell_w = _choose_region_size(cols)

        max_filled = min(4, rows * cols)
        filled_count = random.randint(2, max_filled)
        filled = _sample_positions(rows, cols, filled_count)
        colors = _sample_colors(filled_count)

        grid = _build_input(rows, cols, cell_h, cell_w, filled, colors)
        pattern = _extract_pattern(grid)
        if int(np.count_nonzero(pattern)) != filled_count:
            continue

        out = main(grid)
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if out.shape != (rows * filled_count, cols * filled_count):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
