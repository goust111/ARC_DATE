from __future__ import annotations

from collections import Counter
import random

import numpy as np


LINE_COLOR = 8
GRID_MIN_CELLS = 2
GRID_MAX_CELLS = 6
CELL_SIZE_MIN = 3
CELL_SIZE_MAX = 6
PALETTE_SIZE_MIN = 2
PALETTE_SIZE_MAX = 4
GEN_RETRIES = 400


def _border_color(grid: np.ndarray) -> int:
    border = np.concatenate(
        [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
    )
    return int(Counter(map(int, border.tolist())).most_common(1)[0][0])


def _runs(indexes: list[int]) -> list[tuple[int, int]]:
    assert indexes, "expected at least one separator line"
    out: list[tuple[int, int]] = []
    start = prev = int(indexes[0])
    for idx in indexes[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            out.append((start, prev))
            start = prev = idx
    out.append((start, prev))
    return out


def _full_line_runs(grid: np.ndarray, color: int, axis: int) -> list[tuple[int, int]]:
    size = int(grid.shape[axis])
    indexes: list[int] = []
    for idx in range(size):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if bool(np.all(line == int(color))):
            indexes.append(int(idx))
    return _runs(indexes)


def _spans_between_runs(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for (a0, a1), (b0, _b1) in zip(runs, runs[1:], strict=False):
        start = int(a1 + 1)
        end = int(b0 - 1)
        if start <= end:
            spans.append((start, end))
    return spans


def _dominant_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _recolor_cells(
    grid: np.ndarray,
    line_color: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.array(grid, dtype=int).copy()
    majors = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals = cell[cell != int(line_color)]
            assert vals.size > 0, "each cell must contain non-separator pixels"
            major = _dominant_color(vals)
            majors[i, j] = int(major)
            clean[r0 : r1 + 1, c0 : c1 + 1] = int(major)
    return clean, majors


def _merge_boundaries(
    clean: np.ndarray,
    line_color: int,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    majors: np.ndarray,
) -> np.ndarray:
    out = np.array(clean, dtype=int).copy()

    for i, (r0, r1) in enumerate(row_spans):
        for boundary_idx, (c0, c1) in enumerate(col_runs[1:-1], start=1):
            left = int(majors[i, boundary_idx - 1])
            right = int(majors[i, boundary_idx])
            if left == right:
                out[r0 : r1 + 1, c0 : c1 + 1] = left

    for boundary_idx, (r0, r1) in enumerate(row_runs[1:-1], start=1):
        for j, (c0, c1) in enumerate(col_spans):
            top = int(majors[boundary_idx - 1, j])
            bottom = int(majors[boundary_idx, j])
            if top == bottom:
                out[r0 : r1 + 1, c0 : c1 + 1] = top

    for row_boundary_idx, (r0, r1) in enumerate(row_runs[1:-1], start=1):
        for col_boundary_idx, (c0, c1) in enumerate(col_runs[1:-1], start=1):
            neighbors = {
                int(majors[row_boundary_idx - 1, col_boundary_idx - 1]),
                int(majors[row_boundary_idx - 1, col_boundary_idx]),
                int(majors[row_boundary_idx, col_boundary_idx - 1]),
                int(majors[row_boundary_idx, col_boundary_idx]),
            }
            if len(neighbors) == 1:
                out[r0 : r1 + 1, c0 : c1 + 1] = next(iter(neighbors))
            else:
                out[r0 : r1 + 1, c0 : c1 + 1] = int(line_color)

    return out


def _parse_grid(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    line_color = _border_color(grid)
    assert np.all(grid[0, :] == int(line_color))
    assert np.all(grid[-1, :] == int(line_color))
    assert np.all(grid[:, 0] == int(line_color))
    assert np.all(grid[:, -1] == int(line_color))

    row_runs = _full_line_runs(grid, int(line_color), axis=0)
    col_runs = _full_line_runs(grid, int(line_color), axis=1)
    row_spans = _spans_between_runs(row_runs)
    col_spans = _spans_between_runs(col_runs)

    assert row_spans and col_spans, "expected at least one cell"
    return int(line_color), row_runs, col_runs, row_spans, col_spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise each frame-delimited cell by majority vote, then erase internal separators only where neighboring cell colors match.

    Input structure:
    - A solid azure border and full azure rows/columns partition the grid into equal-sized rectangular cells.
    - Each cell interior contains exactly two non-azure colors: one majority color and one minority noise color.
    - Different cells may share the same majority color or use different majority colors.

    Rule:
    - Identify the separator color from the outer border and locate every full separator row and column.
    - For each cell, choose the most frequent non-separator color and repaint the entire cell interior with that dominant color.
    - For each internal separator segment, compare the dominant colors of the two adjacent cells: erase the segment if they match, otherwise keep it; always keep the outer border.

    Key insight:
    - The puzzle is not solved by denoising alone: after recovering each cell's dominant color, you must use equality between neighboring cells to decide which separator pieces survive.
    """
    g = np.asarray(grid, dtype=int)
    line_color, row_runs, col_runs, row_spans, col_spans = _parse_grid(g)
    clean, majors = _recolor_cells(g, int(line_color), row_spans, col_spans)
    return _merge_boundaries(
        clean, int(line_color), row_runs, col_runs, row_spans, col_spans, majors
    )


def _layout_choices() -> list[tuple[int, int, int, int]]:
    choices: list[tuple[int, int, int, int]] = []
    for n_rows in range(GRID_MIN_CELLS, GRID_MAX_CELLS + 1):
        for n_cols in range(GRID_MIN_CELLS, GRID_MAX_CELLS + 1):
            for cell_h in range(CELL_SIZE_MIN, CELL_SIZE_MAX + 1):
                for cell_w in range(CELL_SIZE_MIN, CELL_SIZE_MAX + 1):
                    height = n_rows * cell_h + n_rows + 1
                    width = n_cols * cell_w + n_cols + 1
                    if height <= 30 and width <= 30:
                        choices.append((n_rows, n_cols, cell_h, cell_w))
    return choices


LAYOUT_CHOICES = _layout_choices()
COLOR_POOL = [c for c in range(10) if c != LINE_COLOR]


def _cell_spans(n_cells: int, cell_size: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 1
    for _ in range(n_cells):
        end = int(start + cell_size - 1)
        spans.append((int(start), end))
        start = int(end + 2)
    return spans


def _adjacency_stats(majors: np.ndarray) -> tuple[int, int]:
    same = 0
    total = 0
    rows, cols = majors.shape
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                total += 1
                same += int(majors[r, c] == majors[r + 1, c])
            if c + 1 < cols:
                total += 1
                same += int(majors[r, c] == majors[r, c + 1])
    return int(same), int(total)


def _sample_majors(rng: random.Random, n_rows: int, n_cols: int) -> np.ndarray | None:
    palette_size = rng.randint(PALETTE_SIZE_MIN, PALETTE_SIZE_MAX)
    palette = rng.sample(COLOR_POOL, k=palette_size)
    majors = np.array(
        [[rng.choice(palette) for _ in range(n_cols)] for _ in range(n_rows)], dtype=int
    )
    same, total = _adjacency_stats(majors)
    diff = int(total - same)
    if same == 0 or diff == 0:
        return None
    ratio = same / total
    if total >= 4 and not (0.25 <= ratio <= 0.75):
        return None
    if len(set(map(int, majors.ravel().tolist()))) < 2:
        return None
    return majors


def _render_output(
    majors: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    line_color: int,
) -> np.ndarray:
    height = int(row_spans[-1][1] + 2)
    width = int(col_spans[-1][1] + 2)
    out = np.full((height, width), int(line_color), dtype=int)

    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(majors[i, j])

    for i, (r0, r1) in enumerate(row_spans):
        for j in range(len(col_spans) - 1):
            if int(majors[i, j]) == int(majors[i, j + 1]):
                split = int(col_spans[j][1] + 1)
                out[r0 : r1 + 1, split : split + 1] = int(majors[i, j])

    for i in range(len(row_spans) - 1):
        split = int(row_spans[i][1] + 1)
        for j, (c0, c1) in enumerate(col_spans):
            if int(majors[i, j]) == int(majors[i + 1, j]):
                out[split : split + 1, c0 : c1 + 1] = int(majors[i, j])

    for i in range(len(row_spans) - 1):
        split_r = int(row_spans[i][1] + 1)
        for j in range(len(col_spans) - 1):
            split_c = int(col_spans[j][1] + 1)
            neighbors = {
                int(majors[i, j]),
                int(majors[i, j + 1]),
                int(majors[i + 1, j]),
                int(majors[i + 1, j + 1]),
            }
            if len(neighbors) == 1:
                out[split_r, split_c] = next(iter(neighbors))

    return out


def _render_input(
    rng: random.Random,
    majors: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    line_color: int,
) -> np.ndarray:
    height = int(row_spans[-1][1] + 2)
    width = int(col_spans[-1][1] + 2)
    inp = np.full((height, width), int(line_color), dtype=int)

    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            major = int(majors[i, j])
            area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            minor_choices = [c for c in COLOR_POOL if c != major]
            minor = int(rng.choice(minor_choices))
            max_noise = max(1, min(area // 3, area - 1))
            noise = int(rng.randint(1, max_noise))

            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), major, dtype=int)
            coords = rng.sample(range(area), k=noise)
            for idx in coords:
                rr, cc = divmod(int(idx), int(patch.shape[1]))
                patch[rr, cc] = int(minor)
            inp[r0 : r1 + 1, c0 : c1 + 1] = patch

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        n_rows, n_cols, cell_h, cell_w = rng.choice(LAYOUT_CHOICES)
        row_spans = _cell_spans(n_rows, cell_h)
        col_spans = _cell_spans(n_cols, cell_w)
        majors = _sample_majors(rng, n_rows, n_cols)
        if majors is None:
            continue

        inp = _render_input(rng, majors, row_spans, col_spans, LINE_COLOR)
        expected = _render_output(majors, row_spans, col_spans, LINE_COLOR)

        try:
            got = main(inp)
        except Exception:
            continue
        if np.array_equal(got, expected):
            return inp

    raise RuntimeError("failed to generate a valid input")
