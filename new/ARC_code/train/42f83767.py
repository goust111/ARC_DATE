import numpy as np

from arc_utils import bbox, find_components, get_bg, paste
import math
import random


def _runs(values):
    runs = []
    start = None
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the lower color grid by replacing each pixel with its matching top stencil.

    Input structure
    The input has a top legend and a lower dense color grid, separated by blank rows.
    The legend contains left-to-right 2x2 color swatches and, to their right,
    the same number of square neutral-color stencil masks.

    Rule
    Pair the swatches and stencil masks by left-to-right order. For every pixel in
    the lower color grid, paste the paired stencil mask as one output block,
    recoloring the mask pixels with that pixel's color and leaving mask holes as bg.

    Key insight
    The lower grid is not copied directly; it is an index grid whose colors select
    the stencil shape from the legend, so each input pixel becomes one square block.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    row_runs = _runs(np.any(g != bg, axis=1))
    assert len(row_runs) == 2
    (top_r0, top_r1), (pattern_r0, pattern_r1) = row_runs
    assert top_r0 == 0

    stencil_size = top_r1 - top_r0 + 1
    top = g[top_r0 : top_r1 + 1, :]
    lower = g[pattern_r0 : pattern_r1 + 1, :]

    pattern_box = bbox(lower != bg)
    assert pattern_box is not None
    pr0, pc0, pr1, pc1 = pattern_box
    assert pr0 == 0 and pr1 == lower.shape[0] - 1
    pattern = lower[:, pc0 : pc1 + 1]
    assert np.all(pattern != bg)
    pattern_colors = set(map(int, np.unique(pattern)))

    top_colors = set(map(int, np.unique(top))) - {bg}
    neutral_colors = top_colors - pattern_colors
    assert len(neutral_colors) == 1
    neutral = next(iter(neutral_colors))

    swatches = []
    for comp in find_components(top, bg=bg, colors=pattern_colors):
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["area"] == 4
        assert r1 - r0 + 1 == 2
        assert c1 - c0 + 1 == 2
        swatches.append((c0, int(comp["color"])))
    assert len(swatches) == len(pattern_colors)
    assert len({c0 for c0, _color in swatches}) == len(swatches)

    swatch_colors = [color for _c0, color in sorted(swatches)]
    assert set(swatch_colors) == pattern_colors

    stencil_col_runs = _runs(np.any(top == neutral, axis=0))
    assert len(stencil_col_runs) == len(swatch_colors)

    stencils = {}
    for color, (c0, c1) in zip(swatch_colors, stencil_col_runs):
        assert c1 - c0 + 1 == stencil_size
        mask = top[:, c0 : c1 + 1] == neutral
        assert mask.shape == (stencil_size, stencil_size)
        assert np.any(mask)
        stencils[color] = mask

    h, w = pattern.shape
    out = np.full((h * stencil_size, w * stencil_size), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(pattern[r, c])
            assert color in stencils
            block = out[
                r * stencil_size : (r + 1) * stencil_size,
                c * stencil_size : (c + 1) * stencil_size,
            ]
            block[stencils[color]] = color
    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and legend geometry.
BACKGROUND, NEUTRAL_COLOR, INPUT_ROWS = 0, 5, 15
SWATCH_SIZE, SWATCH_STRIDE, STENCIL_GAP = 2, 3, 1

# Structural size and palette bounds.
COLOR_COUNT_RANGE = (2, 4)
STENCIL_SIZE_RANGE, PATTERN_SIDE_RANGE = range(3, 7), range(4, 8)
MAX_OUTPUT_SIDE, AVAILABLE_COLORS = 30, [1, 2, 3, 4, 6, 7, 8, 9]
VALID_SIZE_PAIRS = [(s, p) for s in STENCIL_SIZE_RANGE for p in PATTERN_SIDE_RANGE if s * p <= MAX_OUTPUT_SIDE]

# Stencil, pattern, and retry sampling.
STENCIL_FILL_RANGE = (0.60, 0.89)
UNIQUE_STENCIL_RETRIES, GENERATION_RETRIES = 80, 300
GAP_ROWS_RANGE = (1, 3)
PATTERN_MODES = ["iid", "rows", "cols", "checker", "tile"]
TILE_SIDE_RANGE = (2, 3)


def _grow_stencil(size):
    min_fill = max(2 * size - 1, math.ceil(STENCIL_FILL_RANGE[0] * size * size))
    max_fill = math.floor(STENCIL_FILL_RANGE[1] * size * size)
    target = random.randint(min_fill, max_fill)

    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[:, random.randrange(size)] = True

    while int(mask.sum()) < target:
        frontier = []
        for r in range(size):
            for c in range(size):
                if mask[r, c]:
                    continue
                if (
                    (r > 0 and mask[r - 1, c])
                    or (r + 1 < size and mask[r + 1, c])
                    or (c > 0 and mask[r, c - 1])
                    or (c + 1 < size and mask[r, c + 1])
                ):
                    frontier.append((r, c))
        r, c = random.choice(frontier)
        mask[r, c] = True
    return mask


def _make_stencils(count, size):
    stencils = []
    seen = set()
    for _ in range(count):
        for _retry in range(UNIQUE_STENCIL_RETRIES):
            mask = _grow_stencil(size)
            key = tuple(mask.ravel())
            if key not in seen:
                seen.add(key)
                stencils.append(mask)
                break
        else:
            return None
    return stencils


def _make_pattern(colors, side):
    mode = random.choice(PATTERN_MODES)
    colors = list(colors)
    if mode == "rows":
        row_colors = [random.choice(colors) for _ in range(side)]
        pattern = np.array([[row_colors[r]] * side for r in range(side)], dtype=int)
    elif mode == "cols":
        col_colors = [random.choice(colors) for _ in range(side)]
        pattern = np.array([col_colors for _ in range(side)], dtype=int)
    elif mode == "checker":
        order = random.sample(colors, len(colors))
        offset = random.randrange(len(order))
        pattern = np.array(
            [[order[(r + c + offset) % len(order)] for c in range(side)] for r in range(side)],
            dtype=int,
        )
    elif mode == "tile":
        tile_h = random.randint(TILE_SIDE_RANGE[0], min(TILE_SIDE_RANGE[1], side))
        tile_w = random.randint(TILE_SIDE_RANGE[0], min(TILE_SIDE_RANGE[1], side))
        tile = np.array(
            [[random.choice(colors) for _c in range(tile_w)] for _r in range(tile_h)],
            dtype=int,
        )
        pattern = np.tile(tile, (math.ceil(side / tile_h), math.ceil(side / tile_w)))[:side, :side]
    else:
        flat = [random.choice(colors) for _ in range(side * side)]
        pattern = np.array(flat, dtype=int).reshape(side, side)

    missing = [color for color in colors if color not in set(map(int, pattern.ravel()))]
    if missing:
        cells = random.sample(range(side * side), len(missing))
        for idx, color in zip(cells, missing):
            pattern[idx // side, idx % side] = color
    return pattern


def _make_legend_grid(colors, stencils, size):
    count = len(colors)
    width = count * (size + SWATCH_STRIDE + STENCIL_GAP) - STENCIL_GAP
    grid = np.zeros((INPUT_ROWS, width), dtype=int)
    for i, color in enumerate(colors):
        swatch = np.full((SWATCH_SIZE, SWATCH_SIZE), color, dtype=int)
        paste(grid, swatch, 0, i * SWATCH_STRIDE)

    stencil_start = count * SWATCH_STRIDE
    for i, mask in enumerate(stencils):
        patch = np.where(mask, NEUTRAL_COLOR, BACKGROUND)
        paste(grid, patch, 0, stencil_start + i * (size + STENCIL_GAP))
    return grid


def _expand(pattern, colors, stencils):
    side = pattern.shape[0]
    size = stencils[0].shape[0]
    out = np.zeros((side * size, side * size), dtype=int)
    stencil_by_color = dict(zip(colors, stencils))
    for r in range(side):
        for c in range(side):
            color = int(pattern[r, c])
            block = out[r * size : (r + 1) * size, c * size : (c + 1) * size]
            block[stencil_by_color[color]] = color
    return out


def _gen_attempt():
    size, side = random.choice(VALID_SIZE_PAIRS)
    count = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(AVAILABLE_COLORS, count)
    stencils = _make_stencils(count, size)
    if stencils is None:
        return None

    grid = _make_legend_grid(colors, stencils, size)
    gap = random.randint(*GAP_ROWS_RANGE)
    pattern = _make_pattern(colors, side)
    pattern_col = random.randint(0, grid.shape[1] - side)
    paste(grid, pattern, size + gap, pattern_col)

    expected = _expand(pattern, colors, stencils)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 42f83767 input")
