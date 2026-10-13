from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
MAROON = 9

BASE_COLORS = (BLUE, RED, GREEN, YELLOW)
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    sep = int(rows[0])
    assert 0 < sep < grid.shape[0] - 1
    assert not np.any(grid[:sep] == GREY)
    assert not np.any(grid[sep + 1 :] == GREY)
    return sep


def _extract_terrain_heights(grid: np.ndarray, sep: int) -> list[int]:
    top = grid[:sep]
    assert top.size > 0
    assert not np.any((top != BG) & (top != ORANGE))

    heights: list[int] = []
    for c in range(grid.shape[1]):
        h = 0
        r = sep - 1
        while r >= 0 and int(grid[r, c]) == ORANGE:
            h += 1
            r -= 1
        assert h >= 1, "every terrain column must have positive height"
        assert np.all(grid[: sep - h, c] == BG)
        heights.append(int(h))
    return heights


def _extract_band_info(grid: np.ndarray, sep: int) -> tuple[list[int], list[int]]:
    lower = grid[sep + 1 :]
    assert lower.size > 0
    assert not np.any(lower == ORANGE)
    assert not np.any(lower == GREY)
    assert not np.any(lower == PINK)
    assert not np.any(lower == MAROON)

    base_colors: list[int] = []
    seed_lengths: list[int] = []
    for c in range(grid.shape[1]):
        base = int(lower[0, c])
        assert base in BASE_COLORS, "band seed must start with a base color"
        seed = 1
        while seed < lower.shape[0] and int(lower[seed, c]) == base:
            seed += 1
        assert 1 <= seed <= 2, "band seed length must be 1 or 2"
        assert np.all(lower[seed:, c] == BG), "only the top seed pixels may be pre-colored"
        base_colors.append(base)
        seed_lengths.append(int(seed))

    for c in range(grid.shape[1] - 1):
        assert base_colors[c] != base_colors[c + 1], "adjacent band colors must differ"
    return base_colors, seed_lengths


def _fill_colors(heights: list[int], base_colors: list[int]) -> tuple[list[int], int, int]:
    max_h = max(heights)
    min_h = min(heights)
    max_idxs = [i for i, h in enumerate(heights) if h == max_h]
    min_idxs = [i for i, h in enumerate(heights) if h == min_h]
    assert len(max_idxs) == 1 and len(min_idxs) == 1

    max_idx = int(max_idxs[0])
    min_idx = int(min_idxs[0])
    fills: list[int] = []
    width = len(base_colors)
    for c in range(width):
        if c == max_idx:
            fills.append(MAROON)
        elif c == min_idx:
            fills.append(PINK)
        else:
            fills.append(int(base_colors[(c + 1) % width]))
    return fills, max_idx, min_idx


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep = _find_separator_row(g)
    heights = _extract_terrain_heights(g, sep)
    base_colors, seed_lengths = _extract_band_info(g, sep)
    fill_colors, _max_idx, _min_idx = _fill_colors(heights, base_colors)

    lower_h = g.shape[0] - sep - 1
    out = g.copy()
    for c, (height, seed_len, fill_color) in enumerate(zip(heights, seed_lengths, fill_colors, strict=True)):
        assert seed_len + height <= lower_h
        start = sep + 1 + seed_len
        stop = start + height
        out[start:stop, c] = int(fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each lower color band by a terrain-encoded height, with special colors on the extreme-height columns.

    Input structure:
    - The grid has black background, one full-width grey separator row, and the same number of columns above and below it.
    - Above the separator, each column contains a bottom-aligned orange terrain stack whose height encodes one number.
    - Below the separator, each column is a one-cell-wide color band whose top 1 or 2 cells are its own base color from blue, red, green, or yellow.

    Rule:
    - Read the orange height in every column above the separator.
    - In the matching lower column, start immediately below the existing base-color seed and fill downward by exactly that many cells.
    - Normally the fill color for column c is the base color of column c+1, wrapping from the last column back to the first.
    - The unique tallest terrain column instead fills with maroon, and the unique shortest terrain column instead fills with pink.
    - Keep the terrain, separator, and original base-color seed cells unchanged.

    Key insight:
    - The upper terrain is not transformed directly: it is an instruction strip whose per-column heights control the fill amount of the lower bands, while cyclic neighbor colors and the two terrain extremes jointly determine fill color.
    """
    return _transform(grid)


def _sample_heights(width: int, top_h: int) -> list[int] | None:
    max_height = min(6, top_h)
    if max_height < 3:
        return None

    for _ in range(200):
        heights = [random.randint(1, max_height) for _ in range(width)]
        if len(set(heights)) < min(4, max_height):
            continue

        max_v = max(heights)
        min_v = min(heights)
        max_idxs = [i for i, h in enumerate(heights) if h == max_v]
        min_idxs = [i for i, h in enumerate(heights) if h == min_v]
        if len(max_idxs) != 1 or len(min_idxs) != 1:
            continue

        max_idx = max_idxs[0]
        min_idx = min_idxs[0]
        if width - 1 in (max_idx, min_idx):
            continue

        diffs = [heights[i + 1] - heights[i] for i in range(width - 1)]
        if not any(d > 0 for d in diffs):
            continue
        if not any(d < 0 for d in diffs):
            continue

        local_turns = sum(
            1
            for i in range(1, len(diffs))
            if (diffs[i - 1] > 0 > diffs[i]) or (diffs[i - 1] < 0 < diffs[i])
        )
        if local_turns < 1:
            continue

        return heights
    return None


def _sample_base_colors(width: int) -> list[int] | None:
    for _ in range(200):
        colors = [random.choice(BASE_COLORS)]
        for _ in range(1, width):
            prev = colors[-1]
            choices = [c for c in BASE_COLORS if c != prev]
            colors.append(random.choice(choices))

        if len(set(colors)) < 3:
            continue
        return colors
    return None


def _sample_seed_lengths(width: int) -> list[int]:
    seeds = [random.randint(1, 2) for _ in range(width)]
    if len(set(seeds)) == 1:
        seeds[random.randrange(width)] = 3 - seeds[0]
    return seeds


def _build_grid(
    width: int,
    top_h: int,
    bottom_h: int,
    heights: list[int],
    base_colors: list[int],
    seed_lengths: list[int],
) -> np.ndarray:
    grid = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
    grid[top_h, :] = GREY

    for c, height in enumerate(heights):
        grid[top_h - height : top_h, c] = ORANGE

    for c, (base, seed_len) in enumerate(zip(base_colors, seed_lengths, strict=True)):
        grid[top_h + 1 : top_h + 1 + seed_len, c] = int(base)
    return grid


def _validate_case(inp: np.ndarray) -> bool:
    try:
        sep = _find_separator_row(inp)
        heights = _extract_terrain_heights(inp, sep)
        base_colors, seed_lengths = _extract_band_info(inp, sep)
        fill_colors, max_idx, min_idx = _fill_colors(heights, base_colors)
        out = main(inp)
    except AssertionError:
        return False

    if out.shape != inp.shape:
        return False
    if not np.array_equal(out[: sep + 1], inp[: sep + 1]):
        return False

    lower_in = inp[sep + 1 :]
    lower_out = out[sep + 1 :]
    lower_h = lower_in.shape[0]
    new_mask = lower_out != lower_in

    if np.any(out == GREY) and not np.array_equal(out[sep], np.full(inp.shape[1], GREY, dtype=int)):
        return False

    for c, (height, seed_len, base, fill_color) in enumerate(
        zip(heights, seed_lengths, base_colors, fill_colors, strict=True)
    ):
        if seed_len + height > lower_h - 1:
            return False
        if not np.all(lower_out[:seed_len, c] == base):
            return False
        if not np.all(lower_out[seed_len : seed_len + height, c] == fill_color):
            return False
        if not np.all(lower_out[seed_len + height :, c] == BG):
            return False
        if np.any(new_mask[:seed_len, c]):
            return False
        if not np.all(new_mask[seed_len : seed_len + height, c]):
            return False
        if np.any(new_mask[seed_len + height :, c]):
            return False

    if int(np.count_nonzero(new_mask)) != int(sum(heights)):
        return False
    if max_idx == min_idx:
        return False
    if inp.shape[1] - 1 in (max_idx, min_idx):
        return False
    if lower_out[seed_lengths[-1], -1] != base_colors[0]:
        return False
    if fill_colors[max_idx] != MAROON or fill_colors[min_idx] != PINK:
        return False

    normal_indices = [c for c in range(inp.shape[1]) if c not in (max_idx, min_idx)]
    if not normal_indices:
        return False
    borrowed_colors = {fill_colors[c] for c in normal_indices}
    if len(borrowed_colors) < 2:
        return False
    if not {PINK, MAROON}.issubset(set(fill_colors)):
        return False
    if len(set(seed_lengths)) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        width = random.randint(8, 16)
        top_h = random.randint(6, 9)
        heights = _sample_heights(width, top_h)
        if heights is None:
            continue

        seed_lengths = _sample_seed_lengths(width)
        needed_bottom = max(seed + height for seed, height in zip(seed_lengths, heights, strict=True)) + 1
        max_bottom = min(12, 29 - top_h)
        if needed_bottom > max_bottom:
            continue
        bottom_h = random.randint(needed_bottom, max_bottom)

        base_colors = _sample_base_colors(width)
        if base_colors is None:
            continue

        grid = _build_grid(width, top_h, bottom_h, heights, base_colors, seed_lengths)
        if _validate_case(grid):
            return grid

    raise RuntimeError("failed to generate a valid terrain-band puzzle input")
