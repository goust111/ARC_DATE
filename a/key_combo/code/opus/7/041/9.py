from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
NEUTRAL = 5
SWATCH_SIZE = 2
STENCIL_SIZE = 3
MAX_GENERATION_RETRIES = 400
PATTERN_RETRIES = 500
MASK_FILL_RANGE = range(4, 8)
AVAILABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(map(bool, values)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4_count(pattern: np.ndarray, r: int, c: int) -> int:
    color = int(pattern[r, c])
    count = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < pattern.shape[0] and 0 <= cc < pattern.shape[1]:
            count += int(pattern[rr, cc] == color)
    return count


def _expand_pattern(pattern: np.ndarray, stencils: dict[int, np.ndarray]) -> np.ndarray:
    side_h, side_w = pattern.shape
    out = np.full((side_h * STENCIL_SIZE, side_w * STENCIL_SIZE), BACKGROUND, dtype=int)
    for r in range(side_h):
        for c in range(side_w):
            color = int(pattern[r, c])
            rotation = _neighbors4_count(pattern, r, c)
            mask = np.rot90(stencils[color], -rotation)
            block = out[
                r * STENCIL_SIZE : (r + 1) * STENCIL_SIZE,
                c * STENCIL_SIZE : (c + 1) * STENCIL_SIZE,
            ]
            block[mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each lower-grid pixel into its paired template, rotated by same-color neighborhood count.

    Input structure:
    - The grid has a top legend band and a lower dense index grid on a black background.
    - In the legend, 2x2 solid color swatches appear on the left and 3x3 neutral stencil masks appear on the right.
    - The legend and the lower index grid are separated by blank rows.

    Rule:
    - Pair swatches and stencil masks by their left-to-right order in the legend.
    - For each pixel in the lower index grid, choose the paired stencil for that pixel's color.
    - Count how many 4-neighbors of that pixel have the same color, rotate the stencil clockwise by 90 degrees times that count, then recolor stencil cells with the pixel color while leaving holes black.

    Key insight:
    - Color selects which template to use, but local same-color context selects the template rotation, so equal colors can expand differently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    row_runs = _runs(np.any(g != bg, axis=1))
    assert len(row_runs) == 2, "expected legend band and pattern band"
    (legend_r0, legend_r1), (pattern_r0, pattern_r1) = row_runs

    legend = g[legend_r0 : legend_r1 + 1, :]
    lower = g[pattern_r0 : pattern_r1 + 1, :]
    assert legend.shape[0] == STENCIL_SIZE, "expected 3-row legend"

    pattern_box = _bbox(lower != bg)
    assert pattern_box is not None, "expected a lower index grid"
    pr0, pc0, pr1, pc1 = pattern_box
    pattern = lower[pr0 : pr1 + 1, pc0 : pc1 + 1]
    assert np.all(pattern != bg), "lower index grid must be dense"

    pattern_colors = set(map(int, np.unique(pattern)))
    legend_colors = set(map(int, np.unique(legend))) - {bg}
    neutral_candidates = legend_colors - pattern_colors
    assert len(neutral_candidates) == 1, "expected one neutral stencil color"
    neutral = next(iter(neutral_candidates))

    swatches: list[tuple[int, int]] = []
    for color in pattern_colors:
        mask = legend == color
        box = _bbox(mask)
        assert box is not None, "every pattern color must have a legend swatch"
        r0, c0, r1, c1 = box
        assert int(mask.sum()) == SWATCH_SIZE * SWATCH_SIZE, "swatches must be 2x2"
        assert (r1 - r0 + 1, c1 - c0 + 1) == (SWATCH_SIZE, SWATCH_SIZE), "swatches must be compact 2x2 blocks"
        swatches.append((c0, color))
    swatch_order = [color for c0, color in sorted(swatches)]

    stencil_runs = _runs(np.any(legend == neutral, axis=0))
    assert len(stencil_runs) == len(swatch_order), "legend must contain one stencil per swatch"

    stencils: dict[int, np.ndarray] = {}
    for color, (c0, c1) in zip(swatch_order, stencil_runs):
        assert c1 - c0 + 1 == STENCIL_SIZE, "stencils must be 3x3"
        mask = legend[:, c0 : c1 + 1] == neutral
        assert mask.shape == (STENCIL_SIZE, STENCIL_SIZE)
        assert np.any(mask) and not np.all(mask), "stencils must contain both solid cells and holes"
        stencils[color] = mask

    return _expand_pattern(pattern, stencils)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _rotation_keys(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, np.rot90(mask, -k).ravel())) for k in range(4))


def _build_stencil_catalog() -> list[np.ndarray]:
    catalog: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...] | tuple[int, ...]] = set()
    for bits in range(1 << (STENCIL_SIZE * STENCIL_SIZE)):
        flat = [(bits >> idx) & 1 for idx in range(STENCIL_SIZE * STENCIL_SIZE)]
        mask = np.array(flat, dtype=bool).reshape(STENCIL_SIZE, STENCIL_SIZE)
        if int(mask.sum()) not in MASK_FILL_RANGE:
            continue
        if not np.all(np.any(mask, axis=0)) or not np.all(np.any(mask, axis=1)):
            continue
        if not _is_connected(mask):
            continue
        rotations = _rotation_keys(mask)
        if len(set(rotations)) != 4:
            continue
        canonical = min(rotations)
        if canonical in seen:
            continue
        seen.add(canonical)
        catalog.append(mask)
    assert len(catalog) >= 3, "need enough asymmetric stencil masks"
    return catalog


STENCIL_CATALOG = _build_stencil_catalog()


def _same_color_mods(pattern: np.ndarray, color: int) -> set[int]:
    mods = set()
    for r, c in np.argwhere(pattern == color):
        mods.add(_neighbors4_count(pattern, int(r), int(c)) % 4)
    return mods


def _pattern_is_valid(pattern: np.ndarray, colors: tuple[int, ...]) -> bool:
    counts = Counter(map(int, pattern.ravel()))
    if any(counts[color] < 3 for color in colors):
        return False
    if len({_neighbors4_count(pattern, r, c) for r, c in np.ndindex(pattern.shape)}) < 3:
        return False
    return all(len(_same_color_mods(pattern, color)) >= 2 for color in colors)


def _make_pattern(colors: tuple[int, ...], side: int) -> np.ndarray:
    total = side * side
    for _ in range(PATTERN_RETRIES):
        counts = [3] * len(colors)
        for _ in range(total - 3 * len(colors)):
            counts[random.randrange(len(colors))] += 1
        flat: list[int] = []
        for color, count in zip(colors, counts):
            flat.extend([color] * count)
        random.shuffle(flat)
        pattern = np.array(flat, dtype=int).reshape(side, side)
        if _pattern_is_valid(pattern, colors):
            return pattern
    raise RuntimeError("failed to construct a non-degenerate pattern grid")


def _render_input(colors: tuple[int, ...], stencils: tuple[np.ndarray, ...], pattern: np.ndarray) -> np.ndarray:
    gap_rows = random.randint(1, 2)
    legend_gap = random.randint(2, 3)
    bottom_pad = random.randint(0, 2)
    swatch_band_width = len(colors) * SWATCH_SIZE + (len(colors) - 1)
    stencil_band_width = len(colors) * STENCIL_SIZE + (len(colors) - 1)
    legend_width = swatch_band_width + legend_gap + stencil_band_width
    width = min(30, max(legend_width, pattern.shape[1]) + random.randint(0, 4))
    legend_c0 = random.randint(0, width - legend_width)
    pattern_c0 = random.randint(0, width - pattern.shape[1])
    height = STENCIL_SIZE + gap_rows + pattern.shape[0] + bottom_pad

    grid = np.full((height, width), BACKGROUND, dtype=int)

    c = legend_c0
    for idx, color in enumerate(colors):
        grid[0:SWATCH_SIZE, c : c + SWATCH_SIZE] = color
        c += SWATCH_SIZE
        if idx != len(colors) - 1:
            c += 1

    c += legend_gap
    for idx, mask in enumerate(stencils):
        grid[0:STENCIL_SIZE, c : c + STENCIL_SIZE] = np.where(mask, NEUTRAL, BACKGROUND)
        c += STENCIL_SIZE
        if idx != len(colors) - 1:
            c += 1

    pattern_r0 = STENCIL_SIZE + gap_rows
    grid[
        pattern_r0 : pattern_r0 + pattern.shape[0],
        pattern_c0 : pattern_c0 + pattern.shape[1],
    ] = pattern
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        color_count = random.choice((2, 3))
        colors = tuple(random.sample(AVAILABLE_COLORS, color_count))
        stencils = tuple(random.sample(STENCIL_CATALOG, color_count))
        side = random.choice((4, 5))
        pattern = _make_pattern(colors, side)
        grid = _render_input(colors, stencils, pattern)
        expected = _expand_pattern(pattern, dict(zip(colors, stencils)))
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
