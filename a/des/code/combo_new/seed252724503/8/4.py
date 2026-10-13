from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 300


def _find_separator_and_palette(g: np.ndarray) -> tuple[int, np.ndarray]:
    h, w = g.shape
    candidates: list[tuple[int, np.ndarray]] = []

    for r in range(h - 1):
        row = g[r]
        if int(row[0]) == BG or not bool(np.all(row == row[0])):
            continue
        if not bool(np.any(g[:r] != BG)):
            continue

        palette = np.empty(w, dtype=int)
        ok = True
        for c in range(w):
            vals = {int(v) for v in g[r + 1 :, c] if int(v) != BG}
            if len(vals) != 1:
                ok = False
                break
            palette[c] = vals.pop()

        if not ok:
            continue
        if len(set(map(int, palette))) < 2:
            continue
        if any(int(a) == int(b) for a, b in zip(palette, palette[1:])):
            continue
        candidates.append((int(r), palette))

    assert len(candidates) == 1, "expected one separator row above a striped ground palette"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the upper sparse figure straight down onto the shadow row, using the ground stripe colors instead of the figure colors.

    Input structure:
    - Black background surrounds two functional layers.
    - Above a single full-width monochrome separator row sits a sparse multi-color floating pattern.
    - Below the separator is a ground palette encoded as one-cell-wide vertical color stripes, with one stable stripe color per column.

    Rule:
    - Identify every column that contains at least one non-background floating-pattern pixel above the separator.
    - For each such column, paint exactly one shadow pixel in the first row below the separator.
    - The shadow pixel keeps the same column but takes its color from that column's ground stripe, not from the floating pixel.
    - Leave the floating pattern, separator, and ground palette unchanged everywhere else.

    Key insight:
    - The top figure contributes only column positions, while the lower striped layer supplies the output colors for those projected positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D grid"

    sep_r, palette = _find_separator_and_palette(g)
    shadow_r = sep_r + 1

    shadow_cols = np.flatnonzero(np.any(g[:sep_r] != BG, axis=0))
    assert shadow_cols.size > 0, "expected at least one floating pixel above the separator"

    out = g.copy()
    out[shadow_r, shadow_cols] = palette[shadow_cols]
    return out


def _sample_palette_colors(width: int) -> list[int]:
    distinct = random.randint(3, min(6, width, len(COLOR_POOL)))
    base = random.sample(COLOR_POOL, distinct)
    colors = [random.choice(base)]
    while len(colors) < width:
        choices = [color for color in base if color != colors[-1]]
        colors.append(random.choice(choices))
    return colors


def _floating_coords(sep_r: int, width: int, repeat_mode: bool) -> set[tuple[int, int]]:
    max_row = sep_r - 2
    length = random.randint(4, min(width - 2, 14))
    start_c = random.randint(1, width - length - 1)
    row = random.randint(1, max(1, max_row - 1))

    coords: list[tuple[int, int]] = []
    path_rows: list[int] = []
    for offset in range(length):
        c = start_c + offset
        coords.append((row, c))
        path_rows.append(row)
        if offset < length - 1:
            row = min(max(1, row + random.choice([-1, 0, 1])), max_row)

    used = set(coords)
    if repeat_mode:
        extra_count = random.randint(1, max(1, length // 2))
        for idx in random.sample(range(length), extra_count):
            c = start_c + idx
            base_r = path_rows[idx]
            options = [rr for rr in (base_r - 1, base_r + 1) if 0 <= rr <= max_row and (rr, c) not in used]
            if not options:
                continue
            extra_r = random.choice(options)
            used.add((extra_r, c))
            step = 1 if extra_r > base_r else -1
            third_r = extra_r + step
            if random.random() < 0.3 and 0 <= third_r <= max_row and (third_r, c) not in used:
                used.add((third_r, c))

    return used


def _paint_floating_pattern(grid: np.ndarray, coords: set[tuple[int, int]]) -> None:
    palette_size = random.randint(2, min(4, len(COLOR_POOL)))
    colors = random.sample(COLOR_POOL, palette_size)
    ordered = list(coords)
    random.shuffle(ordered)
    assigned = [random.choice(colors) for _ in ordered]
    if len(set(assigned)) == 1:
        alternatives = [color for color in colors if color != assigned[0]]
        if alternatives:
            assigned[-1] = random.choice(alternatives)
    for (r, c), color in zip(ordered, assigned):
        grid[r, c] = int(color)


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(12, 30)
    width = random.randint(8, 30)
    sep_r = random.randint(4, height - 4)
    shadow_r = sep_r + 1
    palette_r0 = sep_r + 2

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_r, :] = SEPARATOR_COLOR

    palette_colors = _sample_palette_colors(width)
    for c, color in enumerate(palette_colors):
        grid[palette_r0:, c] = int(color)

    repeat_mode = random.random() < 0.5
    coords = _floating_coords(sep_r, width, repeat_mode)
    if len(coords) < 4:
        return None
    _paint_floating_pattern(grid, coords)

    expected = grid.copy()
    shadow_cols = sorted({c for _r, c in coords})
    expected[shadow_r, shadow_cols] = np.array([palette_colors[c] for c in shadow_cols], dtype=int)

    if not bool(np.any(grid[:sep_r] != BG)):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is None:
            continue
        grid, expected = candidate
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection puzzle")
