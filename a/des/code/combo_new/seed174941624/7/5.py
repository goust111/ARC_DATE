from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ORIENTATION_MODES = ("horizontal", "vertical", "mixed")
MAX_GENERATION_ATTEMPTS = 400
PARTITION_ATTEMPTS = 200


def _spans_from_cuts(size: int, cuts: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for cut in sorted(cuts) + [size]:
        start = prev + 1
        stop = cut
        if start < stop:
            spans.append((start, stop))
        prev = cut
    return spans


def _layout_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    cuts: list[int] = []
    pos = 0
    for i, size in enumerate(sizes):
        spans.append((pos, pos + int(size)))
        pos += int(size)
        if i + 1 < len(sizes):
            cuts.append(pos)
            pos += 1
    return spans, cuts, pos


def _orientation_counts(heights: list[int], widths: list[int]) -> tuple[int, int]:
    horizontal = 0
    vertical = 0
    for height in heights:
        for width in widths:
            if width > height:
                horizontal += 1
            elif height > width:
                vertical += 1
            else:
                raise AssertionError("square regions are not allowed")
    return horizontal, vertical


def _find_separator(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.reshape(-1)))
    candidates: list[tuple[int, int, int, list[int], list[int]]] = []

    for color, total in counts.items():
        rows = [r for r in range(h) if bool((g[r, :] == int(color)).all())]
        cols = [c for c in range(w) if bool((g[:, c] == int(color)).all())]
        if not rows and not cols:
            continue

        line_total = h * len(cols) + w * len(rows) - len(rows) * len(cols)
        if line_total != int(total):
            continue

        mask = np.zeros((h, w), dtype=bool)
        if rows:
            mask[rows, :] = True
        if cols:
            mask[:, cols] = True
        if not bool((g[mask] == int(color)).all()) or (~mask).sum() == 0:
            continue

        bg = int(Counter(map(int, g[~mask].reshape(-1))).most_common(1)[0][0])
        valid = True
        for r0, r1 in _spans_from_cuts(h, rows):
            for c0, c1 in _spans_from_cuts(w, cols):
                region = g[r0:r1, c0:c1]
                extras = {int(v) for v in region.reshape(-1) if int(v) not in (bg, int(color))}
                if len(extras) > 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            candidates.append((len(rows) + len(cols), line_total, int(color), rows, cols))

    assert candidates, "expected a separator color carried only by full-width/full-height lines"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _score, _cells, color, rows, cols = candidates[0]
    return int(color), [int(r) for r in rows], [int(c) for c in cols]


def _find_background(grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]) -> int:
    g = np.asarray(grid, dtype=int)
    mask = np.zeros(g.shape, dtype=bool)
    if sep_rows:
        mask[sep_rows, :] = True
    if sep_cols:
        mask[:, sep_cols] = True
    assert (~mask).sum() > 0, "separator lines cannot cover the full grid"
    return int(Counter(map(int, g[~mask].reshape(-1))).most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed pixel into a stripe whose direction comes from its rectangular chamber and whose length comes from the chamber's seed count.

    Input structure:
    - The background is black, and one-pixel-wide grey separator lines run across the full grid width or height.
    - These full lines partition the canvas into rectangular regions; every region is strictly wider than tall or strictly taller than wide.
    - Each region contains one seed color only, appearing as several isolated single pixels, and different regions use different seed colors.

    Rule:
    - Detect the full-line separator color and use those rows and columns to recover every rectangular region.
    - In each region, count the number N of seed pixels and determine the growth direction from the region shape: rightward if width > height, downward if height > width.
    - Replace each seed by a same-colored stripe of length N in that direction, including the seed itself and truncating at the region boundary when necessary.
    - Keep all separator lines unchanged and leave every other cell as it was.

    Key insight:
    - The stripe direction is not encoded by seed arrangement; it is a property of the enclosing rectangle, while the stripe length is the shared population count of that whole region.
    """
    g = np.asarray(grid, dtype=int)
    sep_color, sep_rows, sep_cols = _find_separator(g)
    bg = _find_background(g, sep_rows, sep_cols)
    row_spans = _spans_from_cuts(g.shape[0], sep_rows)
    col_spans = _spans_from_cuts(g.shape[1], sep_cols)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0:r1, c0:c1]
            height, width = map(int, region.shape)
            assert height != width, "regions must be non-square"

            colors = {int(v) for v in region.reshape(-1) if int(v) not in (bg, sep_color)}
            assert len(colors) == 1, "each region must contain exactly one seed color"
            color = colors.pop()

            seeds = np.argwhere(region == int(color))
            count = int(len(seeds))
            assert count >= 1, "each region must contain at least one seed"

            if width > height:
                for rr, cc in seeds:
                    end = min(width, int(cc) + count)
                    out[int(r0 + rr), int(c0 + cc) : int(c0 + end)] = int(color)
            else:
                for rr, cc in seeds:
                    end = min(height, int(rr) + count)
                    out[int(r0 + rr) : int(r0 + end), int(c0 + cc)] = int(color)

    return out


def _sample_partition(mode: str) -> tuple[list[int], list[int]] | None:
    for _ in range(PARTITION_ATTEMPTS):
        n_rows = random.randint(1, 3)
        max_cols = min(3, len(SEED_COLORS) // n_rows)
        n_cols = random.randint(1, max_cols)
        if n_rows * n_cols < 2:
            continue

        heights = [random.randint(2, 10) for _ in range(n_rows)]
        widths = [random.randint(2, 10) for _ in range(n_cols)]
        if any(height == width for height in heights for width in widths):
            continue

        total_h = sum(heights) + (n_rows - 1)
        total_w = sum(widths) + (n_cols - 1)
        if total_h > 30 or total_w > 30:
            continue

        horizontal, vertical = _orientation_counts(heights, widths)
        if mode == "horizontal" and vertical != 0:
            continue
        if mode == "vertical" and horizontal != 0:
            continue
        if mode == "mixed" and not (horizontal > 0 and vertical > 0):
            continue

        return heights, widths
    return None


def _sample_seed_count(height: int, width: int) -> int:
    capacity = min(6, (height * width + 1) // 2)
    assert capacity >= 1
    if capacity == 1:
        return 1
    if random.random() < 0.25:
        return 1
    return random.randint(2, capacity)


def _sample_seed_cells(height: int, width: int, count: int) -> list[tuple[int, int]]:
    parity_groups = [
        [(r, c) for r in range(height) for c in range(width) if (r + c) % 2 == 0],
        [(r, c) for r in range(height) for c in range(width) if (r + c) % 2 == 1],
    ]
    choices = [group for group in parity_groups if len(group) >= count]
    assert choices, "requested more isolated seeds than the region can hold"
    return random.sample(random.choice(choices), count)


def _render_partition(heights: list[int], widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans, sep_rows, total_h = _layout_from_sizes(heights)
    col_spans, sep_cols, total_w = _layout_from_sizes(widths)

    grid = np.full((total_h, total_w), BG, dtype=int)
    if sep_rows:
        grid[sep_rows, :] = SEP
    if sep_cols:
        grid[:, sep_cols] = SEP
    return grid, row_spans, col_spans


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(ORIENTATION_MODES)
    partition = _sample_partition(mode)
    if partition is None:
        return None
    heights, widths = partition
    input_grid, row_spans, col_spans = _render_partition(heights, widths)

    n_regions = len(row_spans) * len(col_spans)
    colors = random.sample(SEED_COLORS, n_regions)
    color_index = 0
    horizontal = 0
    vertical = 0

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            color = colors[color_index]
            color_index += 1

            height = int(r1 - r0)
            width = int(c1 - c0)
            seed_count = _sample_seed_count(height, width)
            for rr, cc in _sample_seed_cells(height, width, seed_count):
                input_grid[int(r0 + rr), int(c0 + cc)] = int(color)

            if width > height:
                horizontal += 1
            else:
                vertical += 1

    if mode == "horizontal" and vertical != 0:
        return None
    if mode == "vertical" and horizontal != 0:
        return None
    if mode == "mixed" and not (horizontal > 0 and vertical > 0):
        return None

    try:
        output = main(input_grid)
    except Exception:
        return None

    if np.array_equal(input_grid, output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input grid")
