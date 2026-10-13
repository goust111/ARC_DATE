from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _detect_palette(grid: np.ndarray, bg: int) -> tuple[list[int], list[tuple[int, int]]]:
    h, w = grid.shape
    non_bg_colors = {int(color) for color in np.unique(grid) if int(color) != bg}
    assert 2 <= len(non_bg_colors) <= 4

    candidates: list[list[tuple[int, int]]] = []
    for col in (0, w - 1):
        if int(np.count_nonzero(grid[:, col] != bg)) != len(non_bg_colors):
            continue
        for start_row, step in ((0, 1), (h - 1, -1)):
            rows: list[int] = []
            row = start_row
            while 0 <= row < h and int(grid[row, col]) != bg:
                rows.append(int(row))
                row += step
            if step < 0:
                rows.reverse()
            if len(rows) != len(non_bg_colors):
                continue
            colors = [int(grid[r, col]) for r in rows]
            if len(set(colors)) != len(colors):
                continue
            if set(colors) == non_bg_colors:
                candidates.append([(int(r), int(col)) for r in rows])

    assert len(candidates) == 1, "expected exactly one corner priority palette"
    palette_coords = candidates[0]
    palette_order = [int(grid[r, c]) for r, c in palette_coords]
    return palette_order, palette_coords


def _extract_pairs(
    grid: np.ndarray, bg: int, palette_order: list[int], palette_coords: list[tuple[int, int]]
) -> dict[int, tuple[str, int, int, int]]:
    palette_set = set(palette_coords)
    non_bg_colors = {int(color) for color in np.unique(grid) if int(color) != bg}
    assert set(palette_order) == non_bg_colors

    pairs: dict[int, tuple[str, int, int, int]] = {}
    for color in palette_order:
        locs = [
            (int(r), int(c))
            for r, c in np.argwhere(grid == color)
            if (int(r), int(c)) not in palette_set
        ]
        assert len(locs) == 2, "each color must have exactly two non-palette anchors"
        (r1, c1), (r2, c2) = sorted(locs)
        if r1 == r2:
            pairs[int(color)] = ("h", int(r1), min(int(c1), int(c2)), max(int(c1), int(c2)))
        else:
            assert c1 == c2, "anchors must share a row or a column"
            pairs[int(color)] = ("v", int(c1), min(int(r1), int(r2)), max(int(r1), int(r2)))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    palette_order, palette_coords = _detect_palette(g, bg)
    pairs = _extract_pairs(g, bg, palette_order, palette_coords)

    out = g.copy()
    for color in reversed(palette_order):
        orientation, fixed, start, end = pairs[int(color)]
        if orientation == "h":
            out[fixed, start : end + 1] = int(color)
        else:
            out[start : end + 1, fixed] = int(color)

    for (r, c), color in zip(palette_coords, palette_order):
        out[r, c] = int(color)
    return out


def _sample_spaced(low: int, high: int, count: int, min_gap: int = 2) -> list[int] | None:
    if count == 0:
        return []
    values = np.arange(low, high + 1, dtype=int)
    if len(values) < count:
        return None
    for _ in range(120):
        picks = sorted(int(v) for v in RNG.choice(values, size=count, replace=False))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _count_crossings(pairs: dict[int, tuple[str, int, int, int]]) -> tuple[int, dict[int, int]]:
    horizontals = [(color, spec) for color, spec in pairs.items() if spec[0] == "h"]
    verticals = [(color, spec) for color, spec in pairs.items() if spec[0] == "v"]
    counts = {int(color): 0 for color in pairs}
    total = 0
    for h_color, (_h, row, c0, c1) in horizontals:
        for v_color, (_v, col, r0, r1) in verticals:
            if c0 < col < c1 and r0 < row < r1:
                total += 1
                counts[int(h_color)] += 1
                counts[int(v_color)] += 1
    return total, counts


def _validate_input(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    try:
        palette_order, palette_coords = _detect_palette(grid, bg)
        pairs = _extract_pairs(grid, bg, palette_order, palette_coords)
    except AssertionError:
        return False

    orientations = {spec[0] for spec in pairs.values()}
    if orientations != {"h", "v"}:
        return False

    total_crossings, per_color_crossings = _count_crossings(pairs)
    if total_crossings == 0 or any(count == 0 for count in per_color_crossings.values()):
        return False

    out = main(grid)
    for (r, c), color in zip(palette_coords, palette_order):
        if int(out[r, c]) != int(color):
            return False
    return int(np.count_nonzero(out != bg)) > int(np.count_nonzero(grid != bg))


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.integers(2, 5))
    size_min = max(10, 2 * pair_count + 6)
    size = int(RNG.integers(size_min, 19))
    grid = np.full((size, size), BG, dtype=int)

    colors = [int(c) for c in RNG.choice(COLOR_POOL, size=pair_count, replace=False)]
    palette_order = [int(c) for c in RNG.permutation(colors)]
    corner = int(RNG.integers(0, 4))
    palette_col = 0 if corner in (0, 2) else size - 1
    if corner in (0, 1):
        palette_rows = list(range(pair_count))
    else:
        palette_rows = list(range(size - pair_count, size))
    for row, color in zip(palette_rows, palette_order):
        grid[int(row), int(palette_col)] = int(color)

    horizontal_count = int(RNG.integers(1, pair_count))
    vertical_count = pair_count - horizontal_count

    rows = _sample_spaced(2, size - 3, horizontal_count, min_gap=2)
    cols = _sample_spaced(2, size - 3, vertical_count, min_gap=2)
    if rows is None or cols is None:
        return None
    rows = sorted(rows)
    cols = sorted(cols)

    shuffled_colors = [int(c) for c in RNG.permutation(colors)]
    horizontal_colors = shuffled_colors[:horizontal_count]
    vertical_colors = shuffled_colors[horizontal_count:]

    pairs: dict[int, tuple[str, int, int, int]] = {}

    for color, row in zip(horizontal_colors, rows):
        lo = int(RNG.integers(0, vertical_count))
        hi = int(RNG.integers(lo, vertical_count))
        left_choices = [c for c in range(1, cols[lo]) if c not in cols]
        right_choices = [c for c in range(cols[hi] + 1, size - 1) if c not in cols]
        if not left_choices or not right_choices:
            return None
        c0 = int(RNG.choice(left_choices))
        c1 = int(RNG.choice(right_choices))
        grid[int(row), int(c0)] = int(color)
        grid[int(row), int(c1)] = int(color)
        pairs[int(color)] = ("h", int(row), min(int(c0), int(c1)), max(int(c0), int(c1)))

    for color, col in zip(vertical_colors, cols):
        lo = int(RNG.integers(0, horizontal_count))
        hi = int(RNG.integers(lo, horizontal_count))
        top_choices = [r for r in range(1, rows[lo]) if r not in rows]
        bottom_choices = [r for r in range(rows[hi] + 1, size - 1) if r not in rows]
        if not top_choices or not bottom_choices:
            return None
        r0 = int(RNG.choice(top_choices))
        r1 = int(RNG.choice(bottom_choices))
        grid[int(r0), int(col)] = int(color)
        grid[int(r1), int(col)] = int(color)
        pairs[int(color)] = ("v", int(col), min(int(r0), int(r1)), max(int(r0), int(r1)))

    total_crossings, per_color_crossings = _count_crossings(pairs)
    if total_crossings == 0 or any(count == 0 for count in per_color_crossings.values()):
        return None

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
