from __future__ import annotations

import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _stripe_bounds(row: np.ndarray, bg: int) -> tuple[int, int] | None:
    cols = np.flatnonzero(row != int(bg))
    if cols.size == 0:
        return None
    start = int(cols[0])
    end = int(cols[-1])
    assert end - start + 1 == int(cols.size), "each row must contain one contiguous foreground stripe"
    return start, end


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the left control column, then use marked rows as selectors.

    The control column is the first column. A non-background pixel in that column marks the row.
    Using zero-based row indices, marked even rows reverse the color sequence inside that row's
    unique contiguous foreground stripe. Marked odd rows and unmarked rows stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[1] >= 2

    bg = _dominant_color(g)
    out = g[:, 1:].copy()

    for r in range(g.shape[0]):
        bounds = _stripe_bounds(out[r], bg)
        if bounds is None:
            continue
        if int(g[r, 0]) != bg and r % 2 == 0:
            c0, c1 = bounds
            out[r, c0 : c1 + 1] = out[r, c0 : c1 + 1][::-1]

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (6, 12)
PAYLOAD_WIDTH_RANGE = (8, 16)
PALETTE = tuple(range(10))
PALETTE_SIZE_RANGE = (2, 3)
MAX_RUNS = 4
GEN_ATTEMPTS = 300


def _split_positive(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, int(total)), parts - 1))
    prev = 0
    out: list[int] = []
    for cut in cuts + [int(total)]:
        out.append(cut - prev)
        prev = cut
    return out


def _sample_sequence(length: int, colors: list[int]) -> np.ndarray:
    max_runs = min(MAX_RUNS, length)
    for _ in range(80):
        run_count = random.randint(2, max_runs)
        run_lengths = _split_positive(length, run_count)

        run_colors = [random.choice(colors)]
        for _ in range(run_count - 1):
            choices = [color for color in colors if color != run_colors[-1]]
            run_colors.append(random.choice(choices))

        seq = [color for color, run_len in zip(run_colors, run_lengths) for _ in range(run_len)]
        if len(set(seq)) < 2:
            continue
        if seq == list(reversed(seq)):
            continue
        return np.array(seq, dtype=int)
    raise RuntimeError("failed to build a non-palindromic stripe sequence")


def _choose_marked_rows(height: int) -> set[int]:
    even_rows = [r for r in range(height) if r % 2 == 0]
    odd_rows = [r for r in range(height) if r % 2 == 1]
    base = max(1, height // 3)

    for _ in range(80):
        even_count = random.randint(max(1, base - 1), min(len(even_rows), base + 1))
        odd_count = random.randint(max(1, base - 1), min(len(odd_rows), base + 1))
        if even_count + odd_count >= height:
            continue
        marked = set(random.sample(even_rows, even_count))
        marked.update(random.sample(odd_rows, odd_count))
        return marked

    raise RuntimeError("failed to choose balanced marked rows")


def _paint_row(grid: np.ndarray, row: int, bg: int, stripe_colors: list[int]) -> None:
    payload_width = grid.shape[1] - 1
    max_len = max(3, min(payload_width - 2, payload_width // 2 + 1))
    length = random.randint(3, max_len)

    free = payload_width - length
    if free >= 2:
        start = random.randint(1, free - 1)
    else:
        start = random.randint(0, free)

    seq = _sample_sequence(length, stripe_colors)
    grid[row, 1:] = bg
    grid[row, 1 + start : 1 + start + length] = seq


def _expected_output(grid: np.ndarray, bg: int) -> np.ndarray:
    out = grid[:, 1:].copy()
    for r in range(grid.shape[0]):
        if int(grid[r, 0]) == bg or r % 2 == 1:
            continue
        bounds = _stripe_bounds(out[r], bg)
        assert bounds is not None
        c0, c1 = bounds
        out[r, c0 : c1 + 1] = out[r, c0 : c1 + 1][::-1]
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    payload_width = random.randint(*PAYLOAD_WIDTH_RANGE)

    bg = random.choice(PALETTE)
    remaining = [color for color in PALETTE if color != bg]
    marker = random.choice(remaining)
    stripe_palette_size = random.randint(*PALETTE_SIZE_RANGE)
    stripe_colors = random.sample([color for color in remaining if color != marker], stripe_palette_size)

    grid = np.full((height, payload_width + 1), bg, dtype=int)
    marked_rows = _choose_marked_rows(height)
    if not any(r % 2 == 0 for r in marked_rows):
        return None
    if not any(r % 2 == 1 for r in marked_rows):
        return None
    if len(marked_rows) == height:
        return None

    for r in range(height):
        _paint_row(grid, r, bg, stripe_colors)
        if r in marked_rows:
            grid[r, 0] = marker

    if _dominant_color(grid) != bg:
        return None

    expected = _expected_output(grid, bg)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
