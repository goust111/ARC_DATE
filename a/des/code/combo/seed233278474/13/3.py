import random
from collections import Counter

import numpy as np


BG = 0
SEPARATOR = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)

REGION_SIZE_RANGE = (4, 9)
GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_cross(grid: np.ndarray, bg: int | None = None) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if bg is None:
        bg = _background_color(g)

    row_candidates: list[tuple[int, int]] = []
    for r in range(h):
        color = int(g[r, 0])
        if color != bg and np.all(g[r, :] == color):
            row_candidates.append((r, color))

    col_candidates: list[tuple[int, int]] = []
    for c in range(w):
        color = int(g[0, c])
        if color != bg and np.all(g[:, c] == color):
            col_candidates.append((c, color))

    matches = [(r, c, color) for r, color in row_candidates for c, color2 in col_candidates if color == color2]
    assert len(matches) == 1, "expected exactly one full separator row/column cross"

    sep_row, sep_col, sep_color = matches[0]
    assert 0 < sep_row < h - 1 and 0 < sep_col < w - 1, "separator cross must split the grid into four regions"
    return int(sep_row), int(sep_col), int(sep_color)


def _paint_rect(out: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int) -> None:
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = int(color)


def _region_specs(shape: tuple[int, int], sep_row: int, sep_col: int) -> list[tuple[int, int, int, int]]:
    h, w = map(int, shape)
    return [
        (0, sep_row, 0, sep_col),
        (0, sep_row, sep_col + 1, w),
        (sep_row + 1, h, 0, sep_col),
        (sep_row + 1, h, sep_col + 1, w),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each separator-defined subregion, count the colored edge markers and
    convert those counts into four corner-anchored solid rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    sep_row, sep_col, sep_color = _find_separator_cross(g, bg)
    out = g.copy()

    for r0, r1, c0, c1 in _region_specs(g.shape, sep_row, sep_col):
        sub = g[r0:r1, c0:c1]
        region_h, region_w = map(int, sub.shape)
        assert region_h >= 3 and region_w >= 3, "each subregion needs an interior surrounded by edge rows/cols"

        interior = sub[1:-1, 1:-1]
        assert np.all(interior == bg), "subregion interior must start as background"

        region_colors = set(map(int, np.unique(sub))) - {bg}
        assert len(region_colors) == 1, "each subregion must use exactly one marker color"
        color = region_colors.pop()

        assert np.all((sub == bg) | (sub == color)), "subregion may only contain background and its marker color"

        top = int(np.count_nonzero(sub[0, :] == color))
        bottom = int(np.count_nonzero(sub[-1, :] == color))
        left = int(np.count_nonzero(sub[:, 0] == color))
        right = int(np.count_nonzero(sub[:, -1] == color))

        inner_r0, inner_r1 = r0 + 1, r1 - 1
        inner_c0, inner_c1 = c0 + 1, c1 - 1
        inner_h = int(inner_r1 - inner_r0)
        inner_w = int(inner_c1 - inner_c0)

        assert 0 <= top <= inner_h and 0 <= bottom <= inner_h
        assert 0 <= left <= inner_w and 0 <= right <= inner_w

        _paint_rect(out, inner_r0, inner_r0 + top, inner_c0, inner_c0 + left, color)
        _paint_rect(out, inner_r0, inner_r0 + top, inner_c1 - right, inner_c1, color)
        _paint_rect(out, inner_r1 - bottom, inner_r1, inner_c0, inner_c0 + left, color)
        _paint_rect(out, inner_r1 - bottom, inner_r1, inner_c1 - right, inner_c1, color)

    out[sep_row, :] = sep_color
    out[:, sep_col] = sep_color
    return out


def _sample_counts(rng: random.Random, max_dim: int) -> tuple[int, int, int, int]:
    mode = rng.choices(("sparse", "mixed", "dense"), weights=(3, 4, 3), k=1)[0]

    if mode == "sparse":
        counts = [0 if rng.random() < 0.45 else rng.randint(1, max_dim) for _ in range(4)]
    elif mode == "dense":
        low = max(1, (max_dim + 1) // 2)
        counts = [rng.randint(low, max_dim) for _ in range(4)]
        zero_index = rng.randrange(4)
        if rng.random() < 0.25:
            counts[zero_index] = 0
    else:
        counts = [rng.randint(0, max_dim) for _ in range(4)]

    if counts[0] == 0 and counts[1] == 0:
        counts[rng.randrange(2)] = rng.randint(1, max_dim)
    if counts[2] == 0 and counts[3] == 0:
        counts[2 + rng.randrange(2)] = rng.randint(1, max_dim)

    return tuple(map(int, counts))


def _edge_positions(r0: int, r1: int, c0: int, c1: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1 - 1)],
        "bottom": [(r1 - 1, c) for c in range(c0 + 1, c1 - 1)],
        "left": [(r, c0) for r in range(r0 + 1, r1 - 1)],
        "right": [(r, c1 - 1) for r in range(r0 + 1, r1 - 1)],
    }


def _place_markers(
    grid: np.ndarray,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    color: int,
    counts: tuple[int, int, int, int],
) -> None:
    r0, r1, c0, c1 = bounds
    positions = _edge_positions(r0, r1, c0, c1)
    for edge_name, count in zip(("top", "bottom", "left", "right"), counts):
        if count == 0:
            continue
        chosen = rng.sample(positions[edge_name], count)
        for r, c in chosen:
            grid[r, c] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    top_h = rng.randint(*REGION_SIZE_RANGE)
    bottom_h = rng.randint(*REGION_SIZE_RANGE)
    left_w = rng.randint(*REGION_SIZE_RANGE)
    right_w = rng.randint(*REGION_SIZE_RANGE)

    height = top_h + bottom_h + 1
    width = left_w + right_w + 1
    sep_row = top_h
    sep_col = left_w

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_row, :] = SEPARATOR
    grid[:, sep_col] = SEPARATOR

    colors = rng.sample(PALETTE, 4)
    changed = False

    for bounds, color in zip(_region_specs(grid.shape, sep_row, sep_col), colors):
        r0, r1, c0, c1 = bounds
        inner_h = (r1 - r0) - 2
        inner_w = (c1 - c0) - 2
        max_dim = min(inner_h, inner_w)
        counts = _sample_counts(rng, max_dim)

        _place_markers(grid, rng, bounds, int(color), counts)
        top, bottom, left, right = counts
        if (top or bottom) and (left or right):
            changed = True

    if not changed:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
