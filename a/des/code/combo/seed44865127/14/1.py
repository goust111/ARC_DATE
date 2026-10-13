from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_lattice(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int], int]:
    h, w = g.shape
    candidates: list[tuple[int, list[int], list[int], int]] = []

    for color0 in np.unique(g):
        color = int(color0)
        if color == bg:
            continue

        row_hits = np.count_nonzero(g == color, axis=1)
        col_hits = np.count_nonzero(g == color, axis=0)
        rows = [int(r) for r in np.flatnonzero(row_hits >= w - 2)]
        cols = [int(c) for c in np.flatnonzero(col_hits >= h - 1)]

        if len(rows) < 3 or len(cols) < 3:
            continue
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue

        row_steps = np.diff(rows)
        col_steps = np.diff(cols)
        if row_steps.size == 0 or col_steps.size == 0:
            continue
        if not np.all(row_steps == row_steps[0]) or not np.all(col_steps == col_steps[0]):
            continue
        if int(row_steps[0]) <= 1 or int(row_steps[0]) != int(col_steps[0]):
            continue

        cell_size = int(row_steps[0] - 1)
        if cell_size <= 0:
            continue

        line_mask = np.zeros_like(g, dtype=bool)
        line_mask[rows, :] = True
        line_mask[:, cols] = True
        off_color = {
            (int(r), int(c))
            for r, c in np.argwhere(line_mask & (g != color))
        }
        if not off_color.issubset({(0, 0), (0, w - 1)}):
            continue

        candidates.append((color, rows, cols, cell_size))

    assert len(candidates) == 1, f"expected one separator lattice, got {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each checker-grid cell by the parity of its foreground-pixel count.

    Visible conventions used in this task family:
    - A 1-pixel separator lattice, including an outer frame, divides the grid
      into equal square cells.
    - The top-left corner pixel is the odd-color legend.
    - The top-right corner pixel is the even-color legend.
    - All other non-background pixels inside cells belong to the counted
      foreground pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _most_common_color(g)
    sep, sep_rows, sep_cols, _cell_size = _find_lattice(g, bg)

    odd_color = int(g[0, 0])
    even_color = int(g[0, w - 1])
    assert odd_color not in (bg, sep)
    assert even_color not in (bg, sep)
    assert odd_color != even_color
    assert int(np.count_nonzero(g == odd_color)) == 1
    assert int(np.count_nonzero(g == even_color)) == 1

    out = g.copy()
    for r0, r1 in zip(sep_rows, sep_rows[1:]):
        for c0, c1 in zip(sep_cols, sep_cols[1:]):
            cell = g[r0 + 1 : r1, c0 + 1 : c1]
            assert cell.size > 0
            count = int(np.count_nonzero(cell != bg))
            assert count >= 1
            fill = odd_color if count % 2 == 1 else even_color
            out[r0 + 1 : r1, c0 + 1 : c1] = fill

    out[sep_rows, :] = sep
    out[:, sep_cols] = sep
    return out


# --------------- generator ---------------

GRID_SIDE_RANGE = (2, 5)
CELL_SIZE_CHOICES = (6, 7)
ODD_COUNTS = (1, 3, 5, 7)
EVEN_COUNTS = (2, 4, 6, 8)
MAX_GENERATION_ATTEMPTS = 300


def _paint_cell_foreground(
    grid: np.ndarray,
    top: int,
    left: int,
    cell_size: int,
    count: int,
    color: int,
) -> None:
    chosen = random.sample(range(cell_size * cell_size), count)
    for idx in chosen:
        dr, dc = divmod(idx, cell_size)
        grid[top + dr, left + dc] = color


def _sample_cell_counts(n_cells: int) -> list[int]:
    parities = [1, 0] + [random.randint(0, 1) for _ in range(n_cells - 2)]
    random.shuffle(parities)
    return [
        random.choice(ODD_COUNTS if parity else EVEN_COUNTS)
        for parity in parities
    ]


def _build_expected(
    h: int,
    w: int,
    sep_rows: list[int],
    sep_cols: list[int],
    bg: int,
    sep: int,
    odd_color: int,
    even_color: int,
    cell_counts: list[int],
) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    out[sep_rows, :] = sep
    out[:, sep_cols] = sep

    idx = 0
    for r0, r1 in zip(sep_rows, sep_rows[1:]):
        for c0, c1 in zip(sep_cols, sep_cols[1:]):
            fill = odd_color if cell_counts[idx] % 2 == 1 else even_color
            out[r0 + 1 : r1, c0 + 1 : c1] = fill
            idx += 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*GRID_SIDE_RANGE)
    n_cols = random.randint(*GRID_SIDE_RANGE)
    cell_size = random.choice(CELL_SIZE_CHOICES)

    bg, sep, fg, odd_color, even_color = random.sample(range(10), 5)
    h = n_rows * cell_size + (n_rows + 1)
    w = n_cols * cell_size + (n_cols + 1)

    sep_rows = [i * (cell_size + 1) for i in range(n_rows + 1)]
    sep_cols = [i * (cell_size + 1) for i in range(n_cols + 1)]

    inp = np.full((h, w), bg, dtype=int)
    inp[sep_rows, :] = sep
    inp[:, sep_cols] = sep
    inp[0, 0] = odd_color
    inp[0, w - 1] = even_color

    cell_counts = _sample_cell_counts(n_rows * n_cols)
    idx = 0
    for r0, r1 in zip(sep_rows, sep_rows[1:]):
        for c0, c1 in zip(sep_cols, sep_cols[1:]):
            _paint_cell_foreground(inp, r0 + 1, c0 + 1, cell_size, cell_counts[idx], fg)
            idx += 1

    expected = _build_expected(
        h=h,
        w=w,
        sep_rows=sep_rows,
        sep_cols=sep_cols,
        bg=bg,
        sep=sep,
        odd_color=odd_color,
        even_color=even_color,
        cell_counts=cell_counts,
    )

    if _most_common_color(inp) != bg:
        return None
    if not any(count % 2 == 1 for count in cell_counts):
        return None
    if not any(count % 2 == 0 for count in cell_counts):
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-fill checkerboard input")
