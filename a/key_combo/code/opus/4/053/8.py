from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _extract_legend(grid: np.ndarray) -> list[int]:
    row = [int(x) for x in np.asarray(grid[0], dtype=int)]
    length = 0
    while length < len(row) and row[length] not in (BLACK, GREY):
        length += 1
    legend = row[:length]
    if not (3 <= length <= 4):
        raise ValueError("legend length must be 3 or 4")
    if len(set(legend)) != length:
        raise ValueError("legend colors must be distinct")
    return legend


def _column_heights(grid: np.ndarray) -> list[int]:
    body = np.asarray(grid[1:-1], dtype=int)
    return [int(np.count_nonzero(body[:, c] != BLACK)) for c in range(grid.shape[1])]


def _render_output(shape: tuple[int, int], legend: list[int], heights: list[int]) -> np.ndarray:
    height, width = shape
    out = np.zeros((height, width), dtype=int)
    out[0, : len(legend)] = legend
    out[-1, :] = GREY
    for c, stack_height in enumerate(heights):
        for depth in range(1, stack_height + 1):
            out[height - 1 - depth, c] = legend[(depth - 1) % len(legend)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each column's loose pixels onto the baseline and recolor them by cyclic stack depth from the top legend.

    Input structure:
    - The top row begins with a 3- or 4-cell legend strip of distinct non-black, non-grey colors, followed by black cells.
    - The bottom row is a full grey baseline.
    - Between them are scattered non-black foreground pixels on a black background; their original colors are arbitrary and are not part of the output rule.

    Rule:
    - Keep the top legend strip and the bottom grey baseline fixed.
    - In every column, let all other non-black pixels fall straight down and stack immediately above the baseline.
    - Assign stack depth from bottom to top: the baseline-adjacent pixel has depth 1, the next has depth 2, and so on.
    - Recolor depth d with the legend color at cyclic position ((d - 1) mod L), where L is the legend length.

    Key insight:
    - The output ignores every falling pixel's original color and depends only on two visible cues: how many pixels end up stacked in that column, and the legend's left-to-right color order.
    """
    g = np.asarray(grid, dtype=int)
    legend = _extract_legend(g)
    heights = _column_heights(g)
    return _render_output(g.shape, legend, heights)


def _sample_body_rows(height: int, count: int) -> list[int]:
    final_rows = {height - 1 - depth for depth in range(1, count + 1)}
    for _ in range(20):
        rows = sorted(random.sample(range(1, height - 1), count))
        if set(rows) != final_rows:
            return rows
    return sorted(final_rows)


def _sample_active_columns(width: int, legend_len: int, n_active: int) -> list[int]:
    for _ in range(100):
        cols = sorted(random.sample(range(width), n_active))
        if any(c < legend_len for c in cols) and any(c >= legend_len for c in cols):
            return cols
    raise RuntimeError("failed to place active columns on both sides of the legend span")


def _semantic_checks(grid: np.ndarray, required_counts: list[int]) -> None:
    legend = _extract_legend(grid)
    height, width = grid.shape
    assert np.all(grid[0, len(legend) :] == BLACK)
    assert np.all(grid[-1, :] == GREY)
    assert not np.any(grid[1:-1] == GREY)

    heights = _column_heights(grid)
    active_cols = [c for c, count in enumerate(heights) if count > 0]
    assert active_cols
    assert any(c < len(legend) for c in active_cols)
    assert any(c >= len(legend) for c in active_cols)
    assert set(required_counts).issubset(set(heights[c] for c in active_cols))
    assert max(heights[c] for c in active_cols) >= len(legend) + 1

    body_colors = {int(v) for v in grid[1:-1][grid[1:-1] != BLACK]}
    assert body_colors
    assert body_colors.isdisjoint(set(legend))

    nonfinal_cols = 0
    multicolor_cols = 0
    for c in active_cols:
        rows = list(np.flatnonzero(grid[1:-1, c] != BLACK) + 1)
        final_rows = {height - 1 - depth for depth in range(1, len(rows) + 1)}
        if set(rows) != final_rows:
            nonfinal_cols += 1
        colors = {int(grid[r, c]) for r in rows}
        if len(colors) >= 2:
            multicolor_cols += 1
    assert nonfinal_cols >= max(2, len(active_cols) // 2)
    assert multicolor_cols >= 1

    expected = _render_output(grid.shape, legend, heights)
    assert np.array_equal(main(grid), expected)

    recolored = grid.copy()
    source_pool = [color for color in PALETTE if color not in legend]
    for r, c in np.argwhere(recolored[1:-1] != BLACK):
        recolored[int(r) + 1, int(c)] = random.choice(source_pool)
    assert np.array_equal(main(recolored), expected)

    moved = grid.copy()
    move_col = random.choice(active_cols)
    moved[1:-1, move_col] = BLACK
    moved_rows = _sample_body_rows(height, heights[move_col])
    for r in moved_rows:
        moved[r, move_col] = random.choice(source_pool)
    assert np.array_equal(main(moved), expected)

    reduced = grid.copy()
    tall_col = max(active_cols, key=lambda c: heights[c])
    tall_rows = list(np.flatnonzero(reduced[1:-1, tall_col] != BLACK) + 1)
    reduced[tall_rows[0], tall_col] = BLACK
    assert not np.array_equal(main(reduced), expected)


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.choice([3, 4])
    required_counts = list(range(1, legend_len + 2))

    min_width = max(9, len(required_counts) + 2, legend_len + 4)
    width = random.randint(min_width, 14)
    height = random.randint(max(9, legend_len + 5), 14)
    max_stack = min(6, height - 2)
    if max_stack < legend_len + 1:
        return None

    legend = random.sample(PALETTE, legend_len)
    source_pool = [color for color in PALETTE if color not in legend]
    source_colors = random.sample(source_pool, random.randint(2, min(4, len(source_pool))))

    n_active = random.randint(max(5, len(required_counts) + 1), min(width - 1, 8))
    active_cols = _sample_active_columns(width, legend_len, n_active)

    counts = required_counts + [random.randint(1, max_stack) for _ in range(n_active - len(required_counts))]
    if max_stack == 6 and 6 not in counts and random.random() < 0.4:
        counts[random.randrange(len(counts))] = 6
    random.shuffle(counts)

    grid = np.zeros((height, width), dtype=int)
    grid[0, :legend_len] = legend
    grid[-1, :] = GREY

    multicolor_done = False
    for c, count in zip(active_cols, counts):
        rows = _sample_body_rows(height, count)
        colors = [random.choice(source_colors) for _ in range(count)]
        if count >= 2 and (not multicolor_done or random.random() < 0.6):
            if len(set(colors)) == 1:
                alternatives = [color for color in source_colors if color != colors[0]]
                colors[0] = random.choice(alternatives)
            multicolor_done = True
        for r, color in zip(rows, colors):
            grid[r, c] = color

    if not multicolor_done:
        candidates = [c for c, count in zip(active_cols, counts) if count >= 2]
        if not candidates:
            return None
        c = random.choice(candidates)
        rows = list(np.flatnonzero(grid[1:-1, c] != BLACK) + 1)
        current = int(grid[rows[0], c])
        alternatives = [color for color in source_colors if color != current]
        grid[rows[0], c] = random.choice(alternatives)

    try:
        _semantic_checks(grid, required_counts)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
