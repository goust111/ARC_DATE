from __future__ import annotations

from collections import Counter
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


PALETTE = (
    Color.BLACK,
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
NON_SEPARATOR_COLORS = tuple(color for color in PALETTE if color != Color.YELLOW)
MAX_RETRIES = 400
LEGEND_HEIGHT = 3
SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _most_common(values: np.ndarray) -> int:
    flat = [int(x) for x in values.ravel()]
    return int(Counter(flat).most_common(1)[0][0])


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == Color.YELLOW)]
    assert len(rows) == 1, "expected exactly one full-width yellow separator row"
    return int(rows[0])


def _parse_legend(legend: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    h, w = legend.shape
    mapping: dict[int, np.ndarray] = {}

    for r0 in range(h - 1):
        for c0 in range(w - 2):
            patch = legend[r0 : r0 + 2, c0 : c0 + 3]
            left = patch[:, 0]
            key_rows = np.flatnonzero(left != bg)
            if len(key_rows) != 1:
                continue

            pattern = patch[:, 1:3]
            if np.all(pattern == bg):
                continue

            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + 3)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + 4)
            halo = legend[rr0:rr1, cc0:cc1].copy()
            halo[(r0 - rr0) : (r0 - rr0 + 2), (c0 - cc0) : (c0 - cc0 + 3)] = bg
            if np.any(halo != bg):
                continue

            key_color = int(left[int(key_rows[0])])
            assert key_color != Color.YELLOW, "legend key color cannot be separator yellow"
            assert key_color not in mapping, "duplicate legend key"
            mapping[key_color] = pattern.copy()

    assert mapping, "expected at least one legend entry"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rewrite each whitelisted main-area pixel into its legend-defined 2x2 pattern and erase everything else.

    Input structure:
    - One full-width yellow separator row splits the grid into an upper main region and a lower legend region.
    - Each legend entry is a single key pixel with a touching 2x2 pattern block immediately to its right; entries are separated by background.
    - The main region contains background plus colored pixels, some of which may be absent from the legend.

    Rule:
    - Ignore the separator row and the legend in the output, and double the main region in both height and width.
    - Replace each main-region pixel whose color appears in the legend with that color's 2x2 legend pattern.
    - Replace background pixels and unregistered colors with a 2x2 background block.

    Key insight:
    - The lower block is a data-driven rewrite table and whitelist: only colors explicitly listed there survive the upscale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_row(g)
    assert 0 < sep < g.shape[0] - 1, "separator must split non-empty main and legend regions"
    main_region = g[:sep]
    legend = g[sep + 1 :]
    assert legend.shape[0] >= 2

    bg = _most_common(main_region)
    mapping = _parse_legend(legend, bg)

    h, w = main_region.shape
    out = np.full((2 * h, 2 * w), bg, dtype=int)
    bg_block = np.full((2, 2), bg, dtype=int)

    for r in range(h):
        for c in range(w):
            color = int(main_region[r, c])
            block = mapping.get(color, bg_block)
            out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = block

    return out


def _random_pattern(bg: int) -> np.ndarray:
    colors = list(NON_SEPARATOR_COLORS)
    if bg in colors:
        colors.remove(bg)

    mode = random.choice(("checker", "stripe_h", "stripe_v", "solid", "mixed"))
    if mode == "checker":
        a, b = random.sample(colors, 2)
        return np.array([[a, b], [b, a]], dtype=int)
    if mode == "stripe_h":
        a, b = random.sample(colors, 2)
        return np.array([[a, a], [b, b]], dtype=int)
    if mode == "stripe_v":
        a, b = random.sample(colors, 2)
        return np.array([[a, b], [a, b]], dtype=int)
    if mode == "solid":
        a = random.choice(colors)
        pattern = np.full((2, 2), a, dtype=int)
    else:
        choices = colors + [bg]
        pattern = np.array(random.choices(choices, k=4), dtype=int).reshape(2, 2)

    if np.all(pattern == bg):
        pattern[random.randrange(2), random.randrange(2)] = random.choice(colors)
    return pattern.astype(int)


def _legend_layout(count: int, width: int) -> list[int] | None:
    min_width = 3 * count + (count - 1)
    if min_width > width:
        return None

    extra = width - min_width
    gaps = [1] * (count - 1)
    while extra > 0:
        idx = random.randrange(count + len(gaps))
        if idx == count + len(gaps) - 1:
            break
        if idx < count - 1:
            gaps[idx] += 1
        extra -= 1

    starts = [0]
    for gap in gaps:
        starts.append(starts[-1] + 3 + gap)
    return starts


def _paint_legend(grid: np.ndarray, row: int, starts: list[int], entries: list[tuple[int, np.ndarray]]) -> None:
    for start, (key_color, pattern) in zip(starts, entries):
        key_row = row + random.randrange(2)
        grid[key_row, start] = int(key_color)
        grid[row : row + 2, start + 1 : start + 3] = pattern


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _can_place(mask: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    h, w = mask.shape
    for dr, dc in shape:
        rr = top + dr
        cc = left + dc
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if np.any(mask[r0:r1, c0:c1]):
            return False
    return True


def _place_shape(grid: np.ndarray, mask: np.ndarray, color: int, shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = grid.shape
    sh, sw = _shape_bbox(shape)
    anchors = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    random.shuffle(anchors)

    for top, left in anchors:
        if not _can_place(mask, shape, top, left):
            continue
        for dr, dc in shape:
            rr = top + dr
            cc = left + dc
            grid[rr, cc] = int(color)
            mask[rr, cc] = True
        return True
    return False


def _build_expected(main_region: np.ndarray, mapping: dict[int, np.ndarray], bg: int) -> np.ndarray:
    h, w = main_region.shape
    out = np.full((2 * h, 2 * w), bg, dtype=int)
    bg_block = np.full((2, 2), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            block = mapping.get(int(main_region[r, c]), bg_block)
            out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = block
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(NON_SEPARATOR_COLORS)

    total_legend = random.randint(2, 4)
    main_used_legend = random.randint(1, total_legend - 1) if total_legend > 1 else 1
    extra_main_colors = random.randint(1, 2)

    available = [color for color in NON_SEPARATOR_COLORS if color != bg]
    if total_legend + extra_main_colors > len(available):
        return None

    random.shuffle(available)
    legend_colors = available[:total_legend]
    unmapped_colors = available[total_legend : total_legend + extra_main_colors]
    used_legend_colors = legend_colors[:main_used_legend]

    mapping = {color: _random_pattern(bg) for color in legend_colors}
    legend_entries = [(color, mapping[color]) for color in legend_colors]
    random.shuffle(legend_entries)

    min_width = 3 * total_legend + (total_legend - 1)
    width = random.randint(max(7, min_width), 15)
    starts = _legend_layout(total_legend, width)
    if starts is None:
        return None

    main_h = random.randint(4, 12)
    grid = np.full((main_h + 1 + LEGEND_HEIGHT, width), bg, dtype=int)
    main_region = grid[:main_h]
    grid[main_h, :] = Color.YELLOW
    legend_row = main_h + random.randint(0, LEGEND_HEIGHT - 2)
    _paint_legend(grid, legend_row, starts, legend_entries)

    occupancy = np.zeros(main_region.shape, dtype=bool)
    required_colors = used_legend_colors + unmapped_colors
    random.shuffle(required_colors)

    for color in required_colors:
        shape = random.choice(SHAPES)
        if not _place_shape(main_region, occupancy, color, shape):
            return None

    extra_objects = random.randint(0, 2)
    palette = used_legend_colors + unmapped_colors
    for _ in range(extra_objects):
        color = random.choice(palette)
        shape = random.choice(SHAPES)
        _place_shape(main_region, occupancy, color, shape)

    if not any(np.any(main_region == color) for color in used_legend_colors):
        return None
    if not any(np.any(main_region == color) for color in unmapped_colors):
        return None
    if set(used_legend_colors) == set(legend_colors):
        return None

    expected = _build_expected(main_region, mapping, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-rewrite input")
