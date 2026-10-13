from __future__ import annotations

from collections import Counter

import numpy as np


RNG = np.random.default_rng()
BG = 0
PALETTE = np.arange(1, 10, dtype=int)
HEIGHT_RANGE = (4, 10)
WIDTH_RANGE = (10, 20)
MIN_COLORS_PER_ROW = 2
MAX_COLORS_PER_ROW = 4
MAX_GENERATION_ATTEMPTS = 300
ROW_ATTEMPTS = 200


def _row_counts_and_order(row: np.ndarray) -> tuple[Counter[int], list[int]]:
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    for col, value in enumerate(np.asarray(row, dtype=int)):
        color = int(value)
        if color == BG:
            continue
        counts[color] += 1
        first_seen.setdefault(color, int(col))

    order = sorted(counts, key=lambda color: (-counts[color], first_seen[color]))
    return counts, order


def _transform_row(row: np.ndarray) -> np.ndarray:
    arr = np.asarray(row, dtype=int)
    counts, order = _row_counts_and_order(arr)
    out = np.full(arr.shape, BG, dtype=int)

    cursor = 0
    for color in order:
        count = int(counts[color])
        out[cursor : cursor + count] = color
        cursor += count
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Pack each row's foreground colors into left-aligned runs sorted by frequency."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full_like(g, BG)
    for r in range(g.shape[0]):
        out[r] = _transform_row(g[r])
    return out


def _nonbg_segment_count(row: np.ndarray) -> int:
    segments = 0
    prev_color: int | None = None
    for value in np.asarray(row, dtype=int):
        color = int(value)
        if color == BG:
            prev_color = None
            continue
        if prev_color != color:
            segments += 1
        prev_color = color
    return segments


def _row_is_scattered(row: np.ndarray) -> bool:
    arr = np.asarray(row, dtype=int)
    fg_pos = np.flatnonzero(arr != BG)
    if fg_pos.size == 0:
        return False

    colors = np.unique(arr[fg_pos])
    if colors.size < MIN_COLORS_PER_ROW:
        return False

    span = arr[int(fg_pos[0]) : int(fg_pos[-1]) + 1]
    has_internal_bg = bool(np.any(span == BG))
    repeated_segments = _nonbg_segment_count(arr) > int(colors.size)
    return has_internal_bg or repeated_segments


def _max_distinct_colors(width: int) -> int:
    max_fg = width - 2
    result = MIN_COLORS_PER_ROW
    for n_colors in range(MIN_COLORS_PER_ROW, MAX_COLORS_PER_ROW + 1):
        if n_colors * (n_colors + 1) // 2 <= max_fg:
            result = n_colors
    return result


def _sample_distinct_counts(width: int, n_colors: int) -> list[int] | None:
    max_fg = width - 2
    counts = np.arange(1, n_colors + 1, dtype=int)
    budget = int(RNG.integers(0, max_fg - int(counts.sum()) + 1))
    if budget < 0:
        return None

    while budget > 0:
        valid_starts = [start for start in range(n_colors) if n_colors - start <= budget]
        if not valid_starts:
            break
        start = int(RNG.choice(np.asarray(valid_starts, dtype=int)))
        counts[start:] += 1
        budget -= n_colors - start

    RNG.shuffle(counts)
    return [int(value) for value in counts]


def _sample_tie_counts(width: int, n_colors: int) -> list[int] | None:
    max_fg = width - 2
    max_base = min(4, max_fg // n_colors)
    if max_base < 1:
        return None

    for _ in range(ROW_ATTEMPTS):
        base = int(RNG.integers(1, max_base + 1))
        counts = np.full(n_colors, base, dtype=int)

        if n_colors >= 3 and RNG.random() < 0.5:
            chosen = RNG.choice(n_colors, size=int(RNG.integers(1, n_colors)), replace=False)
            for idx in np.asarray(chosen, dtype=int):
                room = int(max_fg - counts.sum())
                if room <= 0:
                    break
                counts[int(idx)] += int(RNG.integers(1, min(3, room) + 1))

        if counts.sum() <= max_fg and len(set(map(int, counts))) < n_colors:
            return [int(value) for value in counts]
    return None


def _sample_row(width: int, mode: str) -> np.ndarray | None:
    if mode == "distinct":
        max_colors = _max_distinct_colors(width)
    else:
        max_colors = min(MAX_COLORS_PER_ROW, width - 2)
    if max_colors < MIN_COLORS_PER_ROW:
        return None

    for _ in range(ROW_ATTEMPTS):
        n_colors = int(RNG.integers(MIN_COLORS_PER_ROW, max_colors + 1))
        if mode == "distinct":
            counts = _sample_distinct_counts(width, n_colors)
        else:
            counts = _sample_tie_counts(width, n_colors)
        if counts is None:
            continue

        colors = RNG.choice(PALETTE, size=n_colors, replace=False)
        parts = [np.full(count, int(color), dtype=int) for color, count in zip(colors, counts)]
        bg_count = int(width - sum(counts))
        row = np.concatenate(parts + [np.full(bg_count, BG, dtype=int)])

        for _ in range(ROW_ATTEMPTS):
            trial = row.copy()
            RNG.shuffle(trial)
            if not _row_is_scattered(trial):
                continue
            if np.array_equal(trial, _transform_row(trial)):
                continue
            return trial
    return None


def _sample_row_modes(height: int) -> list[str]:
    modes = ["tie", "distinct"]
    while len(modes) < height:
        modes.append("tie" if RNG.random() < 0.5 else "distinct")
    RNG.shuffle(modes)
    return modes


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    width = int(RNG.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))

    rows: list[np.ndarray] = []
    tie_rows = 0
    distinct_rows = 0
    for mode in _sample_row_modes(height):
        row = _sample_row(width, mode)
        if row is None:
            return None
        counts, _ = _row_counts_and_order(row)
        if len(set(counts.values())) < len(counts):
            tie_rows += 1
        else:
            distinct_rows += 1
        rows.append(row)

    grid = np.stack(rows).astype(int)
    if tie_rows == 0 or distinct_rows == 0:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape == grid.shape and np.any(grid != out):
            return grid
    raise RuntimeError("failed to generate a valid input")
