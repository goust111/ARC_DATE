from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, get_bg


BG = Color.BLACK
PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
MAX_ATTEMPTS = 400


def _legend_order_from_pairs(pairs: list[tuple[int, int]]) -> list[int]:
    """Return the legend's unique color order from bottom to top."""
    above_of: dict[int, int] = {}
    indegree: dict[int, int] = {}
    colors: set[int] = set()

    for above, below in pairs:
        colors.add(int(above))
        colors.add(int(below))
        indegree.setdefault(int(above), 0)
        indegree.setdefault(int(below), 0)
        if int(below) in above_of and above_of[int(below)] != int(above):
            raise AssertionError("legend chain branches upward")
        above_of[int(below)] = int(above)
        indegree[int(above)] += 1

    bottoms = [color for color in colors if indegree[color] == 0]
    assert len(bottoms) == 1, "legend must define one bottom color"

    order: list[int] = []
    seen: set[int] = set()
    cur = int(bottoms[0])
    while cur not in seen:
        seen.add(cur)
        order.append(cur)
        if cur not in above_of:
            break
        cur = int(above_of[cur])

    assert len(order) == len(colors), "legend must be one simple chain"
    return order


def _extract_legend(grid: np.ndarray, bg: int, ground: int) -> tuple[int, int, list[tuple[int, int]]]:
    """Locate the left-side 2-column legend and read its pair rows."""
    mask = (grid[:-1] != int(bg)) & (grid[:-1] != int(ground))
    active_cols = np.flatnonzero(mask.any(axis=0))
    assert active_cols.size >= 2, "expected a 2-column legend"
    c0, c1 = int(active_cols[0]), int(active_cols[1])

    legend_rows = np.flatnonzero(mask[:, c0] | mask[:, c1])
    assert legend_rows.size > 0, "legend must occupy at least one row"
    assert np.all(mask[legend_rows, c0] & mask[legend_rows, c1]), "legend rows must fill both legend columns"

    pairs = [(int(grid[r, c0]), int(grid[r, c1])) for r in legend_rows]
    return c0, c1, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each scene column onto the ground, but sort its stack by the order encoded in the left legend chain.

    Input structure:
    - The bottom row is a solid ground color.
    - On the left sits a 2-column legend; each legend row contains two colors, and the rows together form one unbranched cover chain.
    - To the right of the legend, isolated colored scene pixels float above the ground on a background field.

    Rule:
    - Each legend row states that its left color must be above its right color.
    - Chain all legend rows into the unique total order of legend colors from bottom to top.
    - Process each scene column independently: keep only pixels whose colors appear in the legend, then drop them to just above the ground in the derived bottom-to-top order, preserving multiplicity.
    - Delete all non-legend scene pixels and leave the ground and legend unchanged.

    Key insight:
    - The legend rows are not a lookup table; they are adjacent cover relations that must be linked into one full vertical ordering before gravity is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D ARC grid"

    ground = int(g[-1, 0])
    assert np.all(g[-1] == ground), "bottom row must be a solid ground"
    bg = int(get_bg(g[:-1]))

    _c0, c1, legend_pairs = _extract_legend(g, bg, ground)
    order = _legend_order_from_pairs(legend_pairs)
    legend_colors = set(order)

    out = g.copy()
    h, w = g.shape
    for c in range(c1 + 1, w):
        counts = {color: int(np.count_nonzero(g[:-1, c] == color)) for color in order}
        out[:-1, c] = bg
        row = h - 2
        for color in order:
            count = counts[color]
            if count:
                out[row - count + 1 : row + 1, c] = int(color)
                row -= count

    return out


def _ordered_multiset(order: list[int]) -> list[int]:
    subset_size = random.randint(2, min(4, len(order)))
    indices = sorted(random.sample(range(len(order)), subset_size))
    colors = [int(order[i]) for i in indices]
    out: list[int] = []
    for color in colors:
        out.extend([color] * random.randint(1, 2))
    return out


def _place_column(grid: np.ndarray, col: int, colors: list[int], mode: str) -> None:
    """Place one scene column using the requested arrangement mode."""
    h = grid.shape[0]
    rows = sorted(random.sample(range(h - 1), len(colors)))
    rows_bottom_to_top = rows[::-1]

    if mode == "sorted":
        placed = list(colors)
    elif mode == "scrambled":
        placed = list(colors)
        while True:
            random.shuffle(placed)
            if placed != colors or len(set(placed)) == 1:
                break
    else:
        placed = list(colors)
        random.shuffle(placed)

    for row, color in zip(rows_bottom_to_top, placed):
        grid[int(row), int(col)] = int(color)


def _expected_from_latent(
    grid: np.ndarray,
    bg: int,
    scene_start: int,
    order: list[int],
) -> np.ndarray:
    out = grid.copy()
    legend_colors = set(order)
    h, w = grid.shape
    for c in range(scene_start, w):
        kept = [int(grid[r, c]) for r in range(h - 1) if int(grid[r, c]) in legend_colors]
        out[: h - 1, c] = bg
        row = h - 2
        for color in order:
            count = kept.count(int(color))
            for _ in range(count):
                out[row, c] = int(color)
                row -= 1
    return out


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.randint(3, 5)
    legend_height = legend_count - 1
    legend_c0 = random.randint(0, 2)
    gap = random.randint(2, 4)
    right_margin = random.randint(0, 2)
    min_scene_width = 6
    max_scene_width = 30 - (legend_c0 + 2 + gap + right_margin)
    if max_scene_width < min_scene_width:
        return None
    scene_width = random.randint(min_scene_width, min(16, max_scene_width))

    top_margin = random.randint(0, 3)
    lower_space = random.randint(3, 8)
    height = top_margin + legend_height + lower_space + 1
    if height > 30:
        return None

    width = legend_c0 + 2 + gap + scene_width + right_margin
    if width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)

    ground = random.choice(PALETTE)
    legend_pool = [color for color in PALETTE if color != ground]
    order = random.sample(legend_pool, legend_count)
    noise_pool = [color for color in legend_pool if color not in order]
    if len(noise_pool) < 2:
        return None
    noise_colors = random.sample(noise_pool, random.randint(1, min(2, len(noise_pool))))

    grid[-1, :] = int(ground)

    legend_top = random.randint(0, height - 2 - legend_height)
    legend_pairs = [(int(order[i + 1]), int(order[i])) for i in range(legend_count - 1)]
    placed_pairs = legend_pairs[:]
    random.shuffle(placed_pairs)
    for offset, (left_color, right_color) in enumerate(placed_pairs):
        r = legend_top + offset
        grid[r, legend_c0] = int(left_color)
        grid[r, legend_c0 + 1] = int(right_color)

    scene_start = legend_c0 + 2 + gap
    scene_cols = list(range(scene_start, width))
    if len(scene_cols) < 5:
        return None

    mandatory_cols = random.sample(scene_cols, 5)
    remaining_cols = [c for c in scene_cols if c not in mandatory_cols]

    sorted_colors = _ordered_multiset(order)
    scrambled_colors = _ordered_multiset(order)
    mixed_colors = _ordered_multiset(order) + [random.choice(noise_colors) for _ in range(random.randint(1, 2))]
    noise_only = [random.choice(noise_colors) for _ in range(random.randint(2, min(4, height - 1)))]
    repeat_color = random.choice(order)
    repeat_colors = [int(repeat_color)] * random.randint(2, min(4, height - 1))

    mandatory_specs = [
        ("sorted", mandatory_cols[0], sorted_colors),
        ("scrambled", mandatory_cols[1], scrambled_colors),
        ("random", mandatory_cols[2], mixed_colors),
        ("random", mandatory_cols[3], noise_only),
        ("random", mandatory_cols[4], repeat_colors),
    ]

    for mode, col, colors in mandatory_specs:
        if len(colors) > height - 1:
            return None
        _place_column(grid, int(col), list(colors), mode)

    extra_count = random.randint(0, min(4, len(remaining_cols)))
    extra_cols = random.sample(remaining_cols, extra_count)
    extra_modes = ["sorted", "scrambled", "random", "random"]
    for col in extra_cols:
        mode = random.choice(extra_modes)
        if mode == "sorted":
            colors = _ordered_multiset(order)
        elif mode == "scrambled":
            colors = _ordered_multiset(order)
        else:
            if random.random() < 0.35:
                colors = [random.choice(noise_colors) for _ in range(random.randint(1, min(3, height - 1)))]
            else:
                colors = _ordered_multiset(order)
                if random.random() < 0.6:
                    colors += [random.choice(noise_colors)]
        if len(colors) > height - 1:
            return None
        _place_column(grid, int(col), list(colors), mode)

    expected = _expected_from_latent(grid, BG, scene_start, order)

    if int(get_bg(grid[:-1])) != int(BG):
        return None

    parsed_c0, parsed_c1, parsed_pairs = _extract_legend(grid, BG, int(ground))
    if parsed_c0 != legend_c0 or parsed_c1 != legend_c0 + 1:
        return None
    if _legend_order_from_pairs(parsed_pairs) != [int(color) for color in order]:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-legend ARC sample")
