from __future__ import annotations

from collections import Counter
import random

import numpy as np


MIN_GRID_SIDE = 14
MAX_GRID_SIDE = 28
MIN_PATTERN_HEIGHT = 7
MAX_PATTERN_HEIGHT = 15
MIN_PATTERN_WIDTH = 8
MAX_PATTERN_WIDTH = 18
MIN_MARGIN = 1
MAX_MARGIN = 5
MIN_STRIPES = 2
MAX_STRIPES = 5
MIN_STRIPE_HEIGHT = 2
PATTERN_COLOR_MIN = 2
PATTERN_COLOR_MAX = 4
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _find_pattern_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    non_bg = grid != int(bg)
    col_counts = non_bg.sum(axis=0)
    max_count = int(col_counts.max())
    assert max_count > 6, "expected a tall rectangular pattern"

    candidate_runs = [
        run
        for run in _runs(col_counts == max_count)
        if run[1] - run[0] > 6
    ]
    assert candidate_runs, "failed to locate the pattern columns"
    c0, c1_excl = max(candidate_runs, key=lambda rc: rc[1] - rc[0])

    full_rows = np.all(non_bg[:, c0:c1_excl], axis=1)
    row_runs = [run for run in _runs(full_rows) if run[1] - run[0] > 6]
    assert row_runs, "failed to locate the pattern rows"
    r0, r1_excl = max(row_runs, key=lambda rc: rc[1] - rc[0])
    return int(r0), int(c0), int(r1_excl), int(c1_excl)


def _anchor_rows(grid: np.ndarray, bg: int, r0: int, c0: int, r1_excl: int, c1_excl: int) -> list[int]:
    assert c0 > 0 and c1_excl < grid.shape[1], "anchor columns must exist"
    left = grid[r0:r1_excl, c0 - 1]
    right = grid[r0:r1_excl, c1_excl]

    rows: list[int] = []
    colors: list[int] = []
    for rr in range(left.shape[0]):
        lv = int(left[rr])
        rv = int(right[rr])
        if lv == int(bg) and rv == int(bg):
            continue
        assert lv == rv != int(bg), "anchors must appear as same-color pairs on both sides"
        rows.append(int(rr))
        colors.append(lv)

    assert rows, "expected at least one anchor pair"
    assert len(rows) == len(set(rows))
    assert len(colors) == len(set(colors)), "each anchor pair must use a distinct color"
    assert all(0 < row < (r1_excl - r0) for row in rows), "anchors must be internal split lines"
    return sorted(rows)


def _apply_rule(pattern: np.ndarray, split_rows: list[int]) -> np.ndarray:
    cuts = [0] + [int(row) for row in split_rows] + [int(pattern.shape[0])]
    parts = []
    for idx, (start, end) in enumerate(zip(cuts, cuts[1:]), start=1):
        band = pattern[start:end].copy()
        assert band.size > 0, "every strip must be nonempty"
        if idx % 2 == 0:
            band = band[:, ::-1]
        parts.append(band)
    return np.vstack(parts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color anchors on both sides of a large rectangle and mirror every even strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    r0, c0, r1_excl, c1_excl = _find_pattern_bbox(g, bg)
    pattern = g[r0:r1_excl, c0:c1_excl]
    split_rows = _anchor_rows(g, bg, r0, c0, r1_excl, c1_excl)
    return _apply_rule(pattern, split_rows)


def _sample_composition(total: int, count: int, min_size: int) -> list[int] | None:
    remainder = int(total) - int(count) * int(min_size)
    if remainder < 0:
        return None
    cuts = sorted(random.sample(range(remainder + count - 1), count - 1)) if count > 1 else []
    parts: list[int] = []
    prev = -1
    for cut in cuts + [remainder + count - 1]:
        parts.append(int(cut - prev))
        prev = cut
    return [part + min_size - 1 for part in parts]


def _ensure_palette_coverage(pattern: np.ndarray, palette: list[int]) -> None:
    flat_indices = list(range(pattern.size))
    random.shuffle(flat_indices)
    for idx, color in zip(flat_indices, palette):
        r, c = divmod(idx, pattern.shape[1])
        pattern[r, c] = int(color)


def _break_horizontal_symmetry(arr: np.ndarray, palette: list[int]) -> None:
    if not np.array_equal(arr, arr[:, ::-1]):
        return
    assert arr.shape[1] >= 2
    r = random.randrange(arr.shape[0])
    c = random.randrange(arr.shape[1] // 2)
    mirror_c = arr.shape[1] - 1 - c
    choices = [color for color in palette if color != int(arr[r, mirror_c])]
    if not choices:
        choices = list(palette)
    arr[r, c] = int(random.choice(choices))


def _random_runs(length: int, min_seg: int = 1, max_seg: int = 4) -> list[int]:
    segments: list[int] = []
    used = 0
    while used < length:
        remaining = length - used
        seg = random.randint(min_seg, min(max_seg, remaining))
        segments.append(seg)
        used += seg
    if sum(segments) > length:
        segments[-1] -= sum(segments) - length
    return segments


def _make_stripe(height: int, width: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    mode = random.choice(["noise", "row_runs", "col_bands", "diagonal"])
    stripe = np.empty((height, width), dtype=int)

    if mode == "noise":
        stripe[:] = np.random.choice(palette, size=(height, width))
    elif mode == "row_runs":
        for r in range(height):
            runs = _random_runs(width)
            pos = 0
            last_color = None
            for run in runs:
                choices = [color for color in palette if color != last_color]
                color = int(random.choice(choices))
                stripe[r, pos : pos + run] = color
                pos += run
                last_color = color
    elif mode == "col_bands":
        bands = _random_runs(width, min_seg=1, max_seg=3)
        pos = 0
        last_color = None
        for band in bands:
            choices = [color for color in palette if color != last_color]
            color = int(random.choice(choices))
            stripe[:, pos : pos + band] = color
            pos += band
            last_color = color
        for _ in range(random.randint(1, max(1, height))):
            rr = random.randrange(height)
            cc = random.randrange(width)
            stripe[rr, cc] = int(random.choice(palette))
    else:
        rr, cc = np.indices((height, width))
        formula = random.choice(
            [
                (rr + cc + random.randrange(k)) % k,
                (2 * rr + cc + random.randrange(k)) % k,
                (rr + 2 * cc + random.randrange(k)) % k,
            ]
        )
        stripe[:] = np.asarray(palette, dtype=int)[formula]
        if random.choice([False, True]):
            stripe[:] = np.flipud(stripe)

    _break_horizontal_symmetry(stripe, palette)
    return stripe


def _build_pattern(height: int, width: int, stripe_heights: list[int], palette: list[int]) -> np.ndarray:
    stripes = [_make_stripe(part_h, width, palette) for part_h in stripe_heights]
    pattern = np.vstack(stripes)
    _ensure_palette_coverage(pattern, palette)
    if np.array_equal(pattern, pattern[:, ::-1]):
        _break_horizontal_symmetry(pattern, palette)
    return pattern


def _render_case(
    grid_h: int,
    grid_w: int,
    bg: int,
    pattern: np.ndarray,
    anchor_rows: list[int],
    anchor_colors: list[int],
    top: int,
    left: int,
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    ph, pw = pattern.shape
    grid[top : top + ph, left : left + pw] = pattern
    for row, color in zip(anchor_rows, anchor_colors):
        rr = top + int(row)
        grid[rr, left - 1] = int(color)
        grid[rr, left + pw] = int(color)
    return grid


def _sample_layout() -> tuple[int, int, int, int, int, int] | None:
    pattern_h = random.randint(MIN_PATTERN_HEIGHT, MAX_PATTERN_HEIGHT)
    pattern_w = random.randint(MIN_PATTERN_WIDTH, MAX_PATTERN_WIDTH)

    max_stripes = min(MAX_STRIPES, pattern_h // MIN_STRIPE_HEIGHT)
    if max_stripes < MIN_STRIPES:
        return None
    stripe_count = random.randint(MIN_STRIPES, max_stripes)

    grid_h_min = max(MIN_GRID_SIDE, pattern_h + 2 * MIN_MARGIN)
    grid_h_max = min(MAX_GRID_SIDE, pattern_h + 2 * MAX_MARGIN)
    grid_w_min = max(MIN_GRID_SIDE, pattern_w + 2 + 2 * MIN_MARGIN)
    grid_w_max = min(MAX_GRID_SIDE, pattern_w + 2 + 2 * MAX_MARGIN)
    if grid_h_min > grid_h_max or grid_w_min > grid_w_max:
        return None

    grid_h = random.randint(grid_h_min, grid_h_max)
    grid_w = random.randint(grid_w_min, grid_w_max)
    return grid_h, grid_w, pattern_h, pattern_w, stripe_count, stripe_count - 1


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    sampled = _sample_layout()
    if sampled is None:
        return None
    grid_h, grid_w, pattern_h, pattern_w, stripe_count, anchor_count = sampled

    stripe_heights = _sample_composition(pattern_h, stripe_count, MIN_STRIPE_HEIGHT)
    if stripe_heights is None:
        return None
    anchor_rows = list(np.cumsum(stripe_heights[:-1], dtype=int))

    bg = random.randrange(10)
    total_non_bg_needed = anchor_count + PATTERN_COLOR_MAX
    if total_non_bg_needed > 9:
        return None
    available = [color for color in range(10) if color != bg]
    random.shuffle(available)

    pattern_color_count = random.randint(PATTERN_COLOR_MIN, min(PATTERN_COLOR_MAX, len(available) - anchor_count))
    pattern_palette = [int(color) for color in available[:pattern_color_count]]
    anchor_palette = [int(color) for color in available[pattern_color_count : pattern_color_count + anchor_count]]
    if len(anchor_palette) != anchor_count:
        return None

    pattern = _build_pattern(pattern_h, pattern_w, stripe_heights, pattern_palette)
    expected = _apply_rule(pattern, anchor_rows)
    if np.array_equal(pattern, expected):
        return None

    top_choices = list(range(MIN_MARGIN, grid_h - pattern_h - MIN_MARGIN + 1))
    left_choices = list(range(MIN_MARGIN + 1, grid_w - pattern_w - MIN_MARGIN))
    if not top_choices or not left_choices:
        return None

    target_top = (grid_h - pattern_h) / 2.0
    target_left = (grid_w - pattern_w) / 2.0
    top = min(top_choices, key=lambda value: abs(value - target_top) + random.random() * 0.2)
    left = min(left_choices, key=lambda value: abs(value - target_left) + random.random() * 0.2)

    grid = _render_case(grid_h, grid_w, bg, pattern, anchor_rows, anchor_palette, top, left)
    counts = Counter(map(int, grid.ravel()))
    if counts[int(bg)] != max(counts.values()):
        return None
    if sum(1 for count in counts.values() if count == counts[int(bg)]) != 1:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is not None:
            grid, _ = built
            return grid
    raise RuntimeError("failed to generate a valid anchor-strip ARC input")
