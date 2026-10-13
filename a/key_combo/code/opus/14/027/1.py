from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
GUIDE_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_MIN = 13
GRID_MAX = 19
RECT_COUNT_CHOICES = (3, 4, 5)
OFFSET_VALUES = np.arange(0, 6, dtype=int)
MAX_GENERATION_ATTEMPTS = 500


def _distinct_scan(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != int(value):
            out.append(int(value))
    return out


def _extract_guide(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    color_mask = (grid != BLACK) & (grid != GREY)
    coords = np.argwhere(color_mask)
    assert coords.size > 0, "guide bar is missing"

    rows = np.unique(coords[:, 0])
    assert len(rows) == 1, "guide bar must occupy exactly one border row"
    row = int(rows[0])
    h, w = grid.shape
    assert row in (0, h - 1), "guide bar must lie on the top or bottom border"

    cols = np.sort(coords[:, 1])
    expected = np.arange(int(cols[0]), int(cols[-1]) + 1)
    assert np.array_equal(cols, expected), "guide bar must be horizontally contiguous"
    assert int(cols[0]) == 0 or int(cols[-1]) == w - 1, "guide bar must be flush to a corner"

    colors = [int(grid[row, c]) for c in cols]
    assert len(colors) == len(set(colors)), "guide colors must be distinct"
    return colors, coords


def _recover_rectangles(mask: np.ndarray, guide_len: int) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    row_ids = np.where(mask.any(axis=1))[0]
    col_ids = np.where(mask.any(axis=0))[0]
    assert len(row_ids) > 0 and len(col_ids) > 0, "expected at least one gray rectangle"

    left_profile: list[int] = []
    right_profile: list[int] = []
    for r in row_ids:
        cols = np.where(mask[int(r)])[0]
        assert np.array_equal(cols, np.arange(int(cols[0]), int(cols[-1]) + 1)), "gray rows must be contiguous"
        left_profile.append(int(cols[0]))
        right_profile.append(int(cols[-1]))

    top_profile: list[int] = []
    bottom_profile: list[int] = []
    for c in col_ids:
        rows = np.where(mask[:, int(c)])[0]
        assert np.array_equal(rows, np.arange(int(rows[0]), int(rows[-1]) + 1)), "gray columns must be contiguous"
        top_profile.append(int(rows[0]))
        bottom_profile.append(int(rows[-1]))

    for profile in (left_profile, right_profile, top_profile, bottom_profile):
        assert all(profile[i] <= profile[i + 1] for i in range(len(profile) - 1)), "profiles must be monotone"

    tops = _distinct_scan(top_profile)
    lefts = _distinct_scan(left_profile)
    bottoms = _distinct_scan(bottom_profile)
    rights = _distinct_scan(right_profile)
    assert len(tops) == len(lefts) == len(bottoms) == len(rights) == guide_len, "guide length must match depth count"

    rectangles: list[tuple[int, int, int, int]] = []
    counts = np.zeros(mask.shape, dtype=int)
    rebuilt = np.zeros_like(mask, dtype=bool)
    for i in range(guide_len):
        r0, c0, r1, c1 = tops[i], lefts[i], bottoms[i], rights[i]
        assert r1 - r0 + 1 >= 2 and c1 - c0 + 1 >= 2, "rectangle sides must be at least 2"
        rectangles.append((r0, c0, r1, c1))
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
        rebuilt[r0 : r1 + 1, c0 : c1 + 1] = True

    assert np.array_equal(rebuilt, mask), "gray mask must equal the union of recovered rectangles"
    depths = set(map(int, np.unique(counts[mask])))
    assert depths == set(range(1, guide_len + 1)), "every depth from 1..K must appear"
    assert int(counts.max()) == guide_len, "guide length must equal the maximum overlap depth"
    return rectangles, counts


def _render_depths(rectangles: list[tuple[int, int, int, int]], guide_colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rectangles:
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1

    out = np.zeros(shape, dtype=int)
    for depth, color in enumerate(guide_colors, start=1):
        out[counts == depth] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Quantize overlapping gray rectangles into a depth heatmap using the corner color guide.

    Input structure:
    - Black (0) is the background.
    - The only non-black, non-gray cells form one horizontal color-guide bar flush to a corner on the top or bottom border.
    - All remaining foreground cells are gray (5) and form the union of several filled rectangles.

    Rule:
    - Treat the guide bar as an ordered legend: its first color means overlap depth 1, its second color means depth 2, and so on.
    - Recover the gray rectangles, count for each cell how many rectangles cover it, and recolor every depth-d cell with the d-th guide color.
    - Leave uncovered cells black and erase the guide bar itself back to black.

    Key insight:
    - The gray shape is not the final object; it is a stack of overlapping rectangles whose per-cell depth must be measured before the ordered corner legend can translate those depths into colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    guide_colors, guide_coords = _extract_guide(g)
    work = g.copy()
    work[guide_coords[:, 0], guide_coords[:, 1]] = BLACK
    gray_mask = work == GREY

    rectangles, counts = _recover_rectangles(gray_mask, len(guide_colors))
    out = np.zeros_like(g)
    for depth, color in enumerate(guide_colors, start=1):
        out[counts == depth] = int(color)
    return out


def _sample_offset_pair(rng: np.random.Generator, n_rects: int) -> tuple[list[int], list[int]] | None:
    for _ in range(200):
        upper = sorted(rng.choice(OFFSET_VALUES, size=n_rects, replace=False).astype(int).tolist(), reverse=True)
        lower = sorted(rng.choice(OFFSET_VALUES, size=n_rects, replace=False).astype(int).tolist())
        spans = [upper[i] + lower[i] + 1 for i in range(n_rects)]
        if all(2 <= span <= 6 for span in spans):
            return upper, lower
    return None


def _place_guide(grid: np.ndarray, colors: list[int], corner: int) -> None:
    h, w = grid.shape
    k = len(colors)
    if corner == 0:
        grid[0, :k] = colors
    elif corner == 1:
        grid[0, w - k : w] = colors
    elif corner == 2:
        grid[h - 1, :k] = colors
    else:
        grid[h - 1, w - k : w] = colors


def _check_instance(grid: np.ndarray, rectangles: list[tuple[int, int, int, int]], guide_colors: list[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    color_mask = (grid != BLACK) & (grid != GREY)
    if int(color_mask.sum()) != len(guide_colors):
        return False

    gray_mask = grid == GREY
    if not np.any(gray_mask):
        return False

    counts = np.zeros(grid.shape, dtype=int)
    for r0, c0, r1, c1 in rectangles:
        if not (0 <= r0 < r1 < grid.shape[0] and 0 <= c0 < c1 < grid.shape[1]):
            return False
        if not (2 <= r1 - r0 + 1 <= 6 and 2 <= c1 - c0 + 1 <= 6):
            return False
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1

    if int(counts.max()) != len(guide_colors):
        return False
    if set(map(int, np.unique(counts[counts > 0]))) != set(range(1, len(guide_colors) + 1)):
        return False
    if not np.array_equal(gray_mask, counts > 0):
        return False

    expected = _render_depths(rectangles, guide_colors, grid.shape)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False

    recovered_rectangles, recovered_counts = _recover_rectangles(gray_mask, len(guide_colors))
    if recovered_rectangles != rectangles:
        return False
    return np.array_equal(recovered_counts, counts)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = int(rng.integers(GRID_MIN, GRID_MAX))
        w = int(rng.integers(GRID_MIN, GRID_MAX))
        n_rects = int(rng.choice(np.array(RECT_COUNT_CHOICES, dtype=int)))

        vertical = _sample_offset_pair(rng, n_rects)
        horizontal = _sample_offset_pair(rng, n_rects)
        if vertical is None or horizontal is None:
            continue
        up, down = vertical
        left, right = horizontal

        min_core_r = 1 + max(up)
        max_core_r = h - 2 - max(down)
        min_core_c = 1 + max(left)
        max_core_c = w - 2 - max(right)
        if min_core_r > max_core_r or min_core_c > max_core_c:
            continue

        core_r = int(rng.integers(min_core_r, max_core_r + 1))
        core_c = int(rng.integers(min_core_c, max_core_c + 1))

        rectangles = [
            (core_r - up[i], core_c - left[i], core_r + down[i], core_c + right[i])
            for i in range(n_rects)
        ]

        grid = np.zeros((h, w), dtype=int)
        for r0, c0, r1, c1 in rectangles:
            grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

        guide_colors = rng.choice(np.array(GUIDE_PALETTE, dtype=int), size=n_rects, replace=False).astype(int).tolist()
        _place_guide(grid, guide_colors, int(rng.integers(4)))

        if _check_instance(grid, rectangles, guide_colors):
            return grid

    raise RuntimeError("failed to generate a valid overlapping-depth heatmap input")
