from __future__ import annotations

import random

import numpy as np


AZURE = 8
GREY = 5
SWATCH_SIZE = 2
TEMPLATE_SIZE = 2


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(flags):
        if bool(flag) and start is None:
            start = i
        elif not bool(flag) and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _crop_non_azure(block: np.ndarray) -> np.ndarray:
    coords = np.argwhere(block != AZURE)
    assert coords.size, "expected a non-empty block"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return block[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _parse_legend(top: np.ndarray) -> dict[int, np.ndarray]:
    assert top.shape[0] == SWATCH_SIZE, "legend pairs must be two cells tall"

    mapping: dict[int, np.ndarray] = {}
    c = 0
    while c <= top.shape[1] - 2 * SWATCH_SIZE:
        if np.all(top[:, c] == AZURE):
            c += 1
            continue

        swatch = top[:, c : c + SWATCH_SIZE]
        template = top[:, c + SWATCH_SIZE : c + 2 * SWATCH_SIZE]
        color = int(swatch[0, 0])
        assert color not in (AZURE, GREY), "swatch color must differ from Azure and Grey"
        assert np.all(swatch == color), "swatches must be solid 2x2 color blocks"
        assert set(map(int, np.unique(template))).issubset({AZURE, GREY}), "template must use only Grey and Azure"
        assert np.any(template == GREY), "template must contain Grey foreground"
        assert color not in mapping, "legend swatch colors must be unique"

        mapping[color] = template == GREY
        c += 2 * SWATCH_SIZE

    assert mapping, "expected at least one color-template pair"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each lower-grid color pixel into the matching legend template.

    Input structure:
    - Azure is the background. A two-row top legend and a lower dense color grid are separated by one or more all-Azure rows.
    - The legend contains left-to-right pairs; each pair is a solid 2x2 color swatch followed immediately by a 2x2 Grey/Azure template mask.
    - The lower grid contains only swatch colors from the legend and may be horizontally offset within the Azure background.

    Rule:
    - Read the legend pairs from left to right and map each swatch color to the adjacent template.
    - For every cell of the lower color grid, copy that color's template into the corresponding output block.
    - Grey template cells become the lower cell's color; Azure template cells remain Azure background.

    Key insight:
    The lower grid is an index grid, not a picture to copy directly: each color selects a stencil shape that is recolored and tiled into the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_runs = _runs(np.any(g != AZURE, axis=1))
    assert len(row_runs) == 2, "expected a top legend and one lower grid separated by Azure rows"
    top_r0, top_r1 = row_runs[0]
    lower_r0, lower_r1 = row_runs[1]

    top = _crop_non_azure(g[top_r0 : top_r1 + 1])
    lower = _crop_non_azure(g[lower_r0 : lower_r1 + 1])
    assert np.all(lower != AZURE), "lower index grid must be dense and non-Azure"

    mapping = _parse_legend(top)
    lower_colors = set(map(int, np.unique(lower)))
    assert lower_colors.issubset(set(mapping)), "all lower-grid colors must appear in the legend"

    h, w = lower.shape
    out = np.full((h * TEMPLATE_SIZE, w * TEMPLATE_SIZE), AZURE, dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(lower[r, c])
            mask = mapping[color]
            block = out[
                r * TEMPLATE_SIZE : (r + 1) * TEMPLATE_SIZE,
                c * TEMPLATE_SIZE : (c + 1) * TEMPLATE_SIZE,
            ]
            block[mask] = color

    return out


# --------------- generator ---------------

AVAILABLE_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]
PAIR_COUNT_RANGE = (2, 5)
LOWER_ROW_RANGE = (2, 12)
LOWER_COL_RANGE = (3, 14)
PAIR_GAP_RANGE = (1, 2)
SEPARATOR_ROW_RANGE = (1, 3)
BOTTOM_MARGIN_RANGE = (0, 2)
EXTRA_WIDTH_RANGE = (0, 5)
MAX_ATTEMPTS = 300

PATTERN_MODES = ("iid", "rows", "cols", "checker", "tile")
MASKS_2X2 = [
    np.array(bits, dtype=bool).reshape(2, 2)
    for n in range(1, 15)
    for bits in ([[(n >> k) & 1 for k in range(4)]])
]


def _make_pattern(colors: list[int], rows: int, cols: int) -> np.ndarray:
    mode = random.choice(PATTERN_MODES)
    if mode == "rows":
        row_colors = [random.choice(colors) for _ in range(rows)]
        pattern = np.array([[row_colors[r]] * cols for r in range(rows)], dtype=int)
    elif mode == "cols":
        col_colors = [random.choice(colors) for _ in range(cols)]
        pattern = np.array([col_colors for _ in range(rows)], dtype=int)
    elif mode == "checker":
        order = random.sample(colors, len(colors))
        offset = random.randrange(len(order))
        pattern = np.array(
            [[order[(r + c + offset) % len(order)] for c in range(cols)] for r in range(rows)],
            dtype=int,
        )
    elif mode == "tile":
        tile_h = random.randint(1, min(3, rows))
        tile_w = random.randint(2, min(4, cols))
        tile = np.array(
            [[random.choice(colors) for _c in range(tile_w)] for _r in range(tile_h)],
            dtype=int,
        )
        reps = ((rows + tile_h - 1) // tile_h, (cols + tile_w - 1) // tile_w)
        pattern = np.tile(tile, reps)[:rows, :cols]
    else:
        pattern = np.array(
            [[random.choice(colors) for _c in range(cols)] for _r in range(rows)],
            dtype=int,
        )

    missing = [color for color in colors if color not in set(map(int, pattern.ravel()))]
    if missing:
        cells = random.sample(range(rows * cols), len(missing))
        for idx, color in zip(cells, missing):
            pattern[idx // cols, idx % cols] = color
    return pattern


def _build_legend(colors: list[int], masks: list[np.ndarray], gap: int) -> np.ndarray:
    width = len(colors) * 4 + (len(colors) - 1) * gap
    legend = np.full((2, width), AZURE, dtype=int)
    c = 0
    for color, mask in zip(colors, masks):
        legend[:, c : c + 2] = color
        legend[:, c + 2 : c + 4] = np.where(mask, GREY, AZURE)
        c += 4 + gap
    return legend


def _expand(pattern: np.ndarray, colors: list[int], masks: list[np.ndarray]) -> np.ndarray:
    mapping = dict(zip(colors, masks))
    rows, cols = pattern.shape
    out = np.full((rows * 2, cols * 2), AZURE, dtype=int)
    for r in range(rows):
        for c in range(cols):
            color = int(pattern[r, c])
            out[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2][mapping[color]] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(AVAILABLE_COLORS, pair_count)
    masks = random.sample(MASKS_2X2, pair_count)
    gap = random.randint(*PAIR_GAP_RANGE)
    legend = _build_legend(colors, masks, gap)

    rows = random.randint(*LOWER_ROW_RANGE)
    cols = random.randint(*LOWER_COL_RANGE)
    pattern = _make_pattern(colors, rows, cols)

    min_width = max(legend.shape[1], cols)
    width = random.randint(min_width, min(30, min_width + random.randint(*EXTRA_WIDTH_RANGE)))
    legend_c0 = random.randint(0, width - legend.shape[1])
    pattern_c0 = random.randint(0, width - cols)

    sep_rows = random.randint(*SEPARATOR_ROW_RANGE)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    height = 2 + sep_rows + rows + bottom_margin
    if height > 30 or rows * 2 > 30 or cols * 2 > 30:
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    grid[0:2, legend_c0 : legend_c0 + legend.shape[1]] = legend
    lower_r0 = 2 + sep_rows
    grid[lower_r0 : lower_r0 + rows, pattern_c0 : pattern_c0 + cols] = pattern

    expected = _expand(pattern, colors, masks)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed2047415201/8/2 input")
