from __future__ import annotations

from collections import Counter
import random

import numpy as np


COUNT_MAX = 9
GEN_RETRIES = 300
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _mode(values: np.ndarray) -> int:
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _detect_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best: tuple[tuple[int, int, int], int, list[tuple[int, int]], list[tuple[int, int]]] | None = None

    for color0 in np.unique(g):
        color = int(color0)
        row_lines = _spans(np.all(g == color, axis=1))
        col_lines = _spans(np.all(g == color, axis=0))
        if len(row_lines) < 3 or len(col_lines) < COUNT_MAX + 1:
            continue
        if row_lines[0][0] != 0 or row_lines[-1][1] != h:
            continue
        if col_lines[0][0] != 0 or col_lines[-1][1] != w:
            continue

        row_thicknesses = {r1 - r0 for r0, r1 in row_lines}
        col_thicknesses = {c1 - c0 for c0, c1 in col_lines}
        if len(row_thicknesses) != 1 or len(col_thicknesses) != 1:
            continue
        row_thickness = next(iter(row_thicknesses))
        col_thickness = next(iter(col_thicknesses))
        if row_thickness != col_thickness:
            continue

        row_cells = [(row_lines[i][1], row_lines[i + 1][0]) for i in range(len(row_lines) - 1)]
        col_cells = [(col_lines[i][1], col_lines[i + 1][0]) for i in range(len(col_lines) - 1)]
        row_sizes = {r1 - r0 for r0, r1 in row_cells}
        col_sizes = {c1 - c0 for c0, c1 in col_cells}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        row_size = next(iter(row_sizes))
        col_size = next(iter(col_sizes))
        if row_size != col_size or row_size < 3:
            continue

        score = (len(row_cells) * len(col_cells), len(row_cells) + len(col_cells), row_thickness)
        if best is None or score > best[0]:
            best = (score, color, row_cells, col_cells)

    if best is None:
        raise ValueError("failed to detect grid layout")
    return best[1], best[2], best[3]


def _detect_background(grid: np.ndarray, line_color: int, board_rows: list[tuple[int, int]], col_cells: list[tuple[int, int]]) -> int:
    local_mode_votes: Counter[int] = Counter()
    presence_votes: Counter[int] = Counter()
    total_pixels: Counter[int] = Counter()

    for r0, r1 in board_rows:
        for c0, c1 in col_cells:
            patch = np.asarray(grid[r0:r1, c0:c1], dtype=int)
            colors, counts = np.unique(patch, return_counts=True)
            filtered = [(int(color), int(count)) for color, count in zip(colors.tolist(), counts.tolist()) if int(color) != line_color]
            if not filtered:
                continue
            best_local = max(filtered, key=lambda item: item[1])[0]
            local_mode_votes[best_local] += 1
            for color, count in filtered:
                presence_votes[color] += 1
                total_pixels[color] += count

    if not local_mode_votes:
        raise ValueError("failed to detect background color")
    return max(local_mode_votes, key=lambda color: (local_mode_votes[color], presence_votes[color], total_pixels[color]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each small motif with the legend color keyed by its non-background pixel count.
    """
    g = np.asarray(grid, dtype=int)
    line_color, row_cells, col_cells = _detect_layout(g)
    if len(row_cells) < 2 or len(col_cells) < COUNT_MAX:
        raise ValueError("grid is missing board rows or legend columns")

    board_rows = row_cells[:-1]
    legend_row = row_cells[-1]
    background = _detect_background(g, line_color, board_rows, col_cells)

    legend_colors = []
    for c0, c1 in col_cells[:COUNT_MAX]:
        legend_colors.append(_mode(g[legend_row[0]:legend_row[1], c0:c1]))

    out = g.copy()
    for r0, r1 in board_rows:
        for c0, c1 in col_cells:
            patch = g[r0:r1, c0:c1]
            count = int(np.count_nonzero((patch != background) & (patch != line_color)))
            fill_color = background if not (1 <= count <= len(legend_colors)) else int(legend_colors[count - 1])
            out[r0:r1, c0:c1] = fill_color

    for c0, c1 in col_cells:
        out[legend_row[0]:legend_row[1], c0:c1] = background

    return out


def _sample_connected_shape(rng: random.Random, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    chosen = {rng.choice(cells)}
    while len(chosen) < count:
        frontier = []
        for r, c in chosen:
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < 3 and 0 <= cc < 3 and (rr, cc) not in chosen:
                    frontier.append((rr, cc))
        if not frontier:
            break
        chosen.add(rng.choice(frontier))
    return list(chosen)


def _sample_pattern_pixels(rng: random.Random, cell_size: int, count: int) -> list[tuple[int, int]]:
    offset_r = rng.randint(0, cell_size - 3)
    offset_c = rng.randint(0, cell_size - 3)
    if count == 9:
        rel = [(r, c) for r in range(3) for c in range(3)]
    elif rng.random() < 0.5:
        rel = _sample_connected_shape(rng, count)
    else:
        rel = rng.sample([(r, c) for r in range(3) for c in range(3)], count)
    return [(offset_r + r, offset_c + c) for r, c in rel]


def _cell_spans(count: int, cell_size: int, thickness: int) -> list[tuple[int, int]]:
    spans = []
    cursor = thickness
    for _ in range(count):
        spans.append((cursor, cursor + cell_size))
        cursor += cell_size + thickness
    return spans


def _render_expected(
    shape: tuple[int, int],
    board_counts: list[list[int]],
    legend_colors: list[int],
    background: int,
    line_color: int,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, line_color, dtype=int)
    for row_idx, (r0, r1) in enumerate(row_cells[:-1]):
        for col_idx, (c0, c1) in enumerate(col_cells):
            out[r0:r1, c0:c1] = int(legend_colors[board_counts[row_idx][col_idx] - 1])
    legend_r0, legend_r1 = row_cells[-1]
    for c0, c1 in col_cells:
        out[legend_r0:legend_r1, c0:c1] = background
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    board_rows = rng.randint(2, 4)
    board_cols = COUNT_MAX
    cell_size = rng.randint(5, 6)
    thickness = rng.randint(1, 2)

    colors = list(range(10))
    background = rng.choice(colors)
    line_color = rng.choice([c for c in colors if c != background])
    usable = [c for c in colors if c not in (background, line_color)]
    if len(usable) < 3:
        return None

    legend_palette = rng.sample(usable, rng.randint(3, min(6, len(usable))))
    legend_colors = [rng.choice(legend_palette) for _ in range(COUNT_MAX)]
    if len(set(legend_colors)) < 3:
        return None

    counts_flat = [rng.randint(1, COUNT_MAX) for _ in range(board_rows * board_cols)]
    if len(set(counts_flat)) < 4 or min(counts_flat) > 2 or max(counts_flat) < 8:
        return None
    board_counts = [counts_flat[i * board_cols : (i + 1) * board_cols] for i in range(board_rows)]

    pattern_colors = [rng.choice(usable) for _ in range(board_rows * board_cols)]
    if len(set(pattern_colors)) < 2:
        return None

    total_rows = board_rows + 1
    height = total_rows * cell_size + (total_rows + 1) * thickness
    width = board_cols * cell_size + (board_cols + 1) * thickness
    grid = np.full((height, width), line_color, dtype=int)

    row_cells = _cell_spans(total_rows, cell_size, thickness)
    col_cells = _cell_spans(board_cols, cell_size, thickness)

    for r0, r1 in row_cells[:-1]:
        for c0, c1 in col_cells:
            grid[r0:r1, c0:c1] = background

    legend_r0, legend_r1 = row_cells[-1]
    for idx, (c0, c1) in enumerate(col_cells):
        grid[legend_r0:legend_r1, c0:c1] = int(legend_colors[idx])

    color_idx = 0
    for row_idx, (r0, r1) in enumerate(row_cells[:-1]):
        for col_idx, (c0, c1) in enumerate(col_cells):
            count = board_counts[row_idx][col_idx]
            pattern_color = int(pattern_colors[color_idx])
            color_idx += 1
            for dr, dc in _sample_pattern_pixels(rng, cell_size, count):
                grid[r0 + dr, c0 + dc] = pattern_color

    expected = _render_expected(
        grid.shape,
        board_counts,
        legend_colors,
        background,
        line_color,
        row_cells,
        col_cells,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
