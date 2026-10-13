from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, get_bg, paste


BG = Color.BLACK
STENCIL_COLOR = Color.GREY
AVAILABLE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

TEMPLATE_COUNT_RANGE = (3, 4)
TEMPLATE_H_RANGE = (3, 5)
TEMPLATE_W_RANGE = (3, 5)
INDEX_H_RANGE = (2, 6)
INDEX_W_RANGE = (3, 6)
LEGEND_GAP_RANGE = (1, 2)
ROW_GAP_RANGE = (1, 3)
SIDE_MARGIN_RANGE = (0, 3)
RIGHT_MARGIN_RANGE = (0, 3)
STENCIL_FILL_RANGE = (0.40, 0.78)
STENCIL_RETRIES = 120
GEN_RETRIES = 400


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(values.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _extract_templates(legend: np.ndarray, bg: int) -> list[np.ndarray]:
    cols = _runs(np.any(legend != bg, axis=0))
    assert cols, "legend must contain at least one stencil"

    non_bg = set(map(int, np.unique(legend))) - {int(bg)}
    assert len(non_bg) == 1, "legend must use exactly one neutral stencil color"
    neutral = next(iter(non_bg))

    templates: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    dims: tuple[int, int] | None = None
    for c0, c1 in cols:
        mask = legend[:, c0 : c1 + 1] == int(neutral)
        assert mask.any(), "empty stencil block"
        assert bool(np.all(mask.any(axis=0))), "stencil must span every legend column in its block"
        assert bool(np.all(mask.any(axis=1))), "stencil must span every legend row in its block"
        if dims is None:
            dims = mask.shape
        else:
            assert mask.shape == dims, "all stencils must have the same size"
        key = tuple(int(v) for v in mask.ravel())
        assert key not in seen, "stencils must be pairwise distinct"
        seen.add(key)
        templates.append(mask)
    return templates


def _rank_colors_by_frequency(pattern: np.ndarray) -> list[int]:
    counts = Counter(map(int, pattern.ravel()))
    freq_values = list(counts.values())
    assert len(freq_values) == len(set(freq_values)), "pattern color counts must be pairwise distinct"
    return sorted(counts, key=lambda color: counts[color], reverse=True)


def _first_appearance_order(pattern: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for color in map(int, pattern.ravel()):
        if color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _expand(pattern: np.ndarray, template_order: list[int], templates: list[np.ndarray], bg: int) -> np.ndarray:
    assert len(template_order) == len(templates)
    template_by_color = {int(color): templates[i] for i, color in enumerate(template_order)}
    block_h, block_w = templates[0].shape
    out = np.full((pattern.shape[0] * block_h, pattern.shape[1] * block_w), int(bg), dtype=int)
    for r in range(pattern.shape[0]):
        for c in range(pattern.shape[1]):
            color = int(pattern[r, c])
            stencil = template_by_color[color]
            patch = np.full(stencil.shape, int(bg), dtype=int)
            patch[stencil] = color
            paste(out, patch, r * block_h, c * block_w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each index-grid color with a left-to-right grey stencil by frequency rank, then expand the grid blockwise.

    Input structure:
    - The top band contains several same-size grey stencil masks laid out left to right on the background.
    - After one or more blank rows, a lower dense index grid appears; it uses exactly as many colors as there are stencils.
    - The lower colors have pairwise-distinct pixel counts, and the legend contains no color swatches.

    Rule:
    - Split the input into the top stencil legend and the lower dense index grid.
    - Count each lower-grid color and sort the colors from most frequent to least frequent.
    - Pair the k-th most frequent color with the k-th stencil from the left.
    - Replace each lower-grid pixel by one stencil-sized block: recolor stencil cells to that pixel's color and keep stencil holes as background.

    Key insight:
    - The legend has only neutral masks, so the matching key is not first appearance or color identity; it is the global frequency rank of each lower-grid color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    row_runs = _runs(np.any(g != bg, axis=1))
    assert len(row_runs) == 2, "expected a legend band and an index-grid band"

    (legend_r0, legend_r1), (index_r0, index_r1) = row_runs
    legend = g[legend_r0 : legend_r1 + 1, :]
    index_band = g[index_r0 : index_r1 + 1, :]

    templates = _extract_templates(legend, bg)
    index_box = bbox(index_band != bg)
    assert index_box is not None, "missing lower index grid"
    pattern = crop_bbox(index_band, index_box)
    assert bool(np.all(pattern != bg)), "index grid must be dense"

    colors = sorted(set(map(int, pattern.ravel())))
    assert len(colors) == len(templates), "stencil count must match index-grid color count"
    ranked_colors = _rank_colors_by_frequency(pattern)
    return _expand(pattern, ranked_colors, templates, bg)


def _grow_connected_mask(height: int, width: int, target: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                frontier.add((rr, cc))

    add_frontier(*start)
    while int(mask.sum()) < target:
        if not frontier:
            return None
        options = sorted(
            frontier,
            key=lambda rc: sum(
                0 <= rc[0] + dr < height and 0 <= rc[1] + dc < width and mask[rc[0] + dr, rc[1] + dc]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ),
            reverse=True,
        )
        top_k = min(4, len(options))
        cell = random.choice(options[:top_k]) if random.random() < 0.7 else random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        add_frontier(*cell)
    return mask


def _sample_templates(count: int, height: int, width: int) -> list[np.ndarray] | None:
    templates: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    area = height * width
    min_fill = max(height + width - 1, math.ceil(STENCIL_FILL_RANGE[0] * area))
    max_fill = min(area - 1, math.floor(STENCIL_FILL_RANGE[1] * area))
    if min_fill > max_fill:
        return None

    for _ in range(count):
        for _retry in range(STENCIL_RETRIES):
            target = random.randint(min_fill, max_fill)
            mask = _grow_connected_mask(height, width, target)
            if mask is None:
                continue
            if not bool(np.all(mask.any(axis=0))) or not bool(np.all(mask.any(axis=1))):
                continue
            key = tuple(int(v) for v in mask.ravel())
            if key in seen:
                continue
            seen.add(key)
            templates.append(mask)
            break
        else:
            return None
    return templates


def _sample_dimensions(template_count: int) -> tuple[int, int, int, int, list[int], int, int, int, int] | None:
    for _ in range(200):
        block_h = random.randint(*TEMPLATE_H_RANGE)
        block_w = random.randint(*TEMPLATE_W_RANGE)

        max_index_h = min(INDEX_H_RANGE[1], 30 // block_h)
        max_index_w = min(INDEX_W_RANGE[1], 30 // block_w)
        if max_index_h < INDEX_H_RANGE[0] or max_index_w < INDEX_W_RANGE[0]:
            continue

        index_h = random.randint(INDEX_H_RANGE[0], max_index_h)
        index_w = random.randint(INDEX_W_RANGE[0], max_index_w)
        if index_h * index_w < template_count * (template_count + 1) // 2:
            continue

        legend_gaps = [random.randint(*LEGEND_GAP_RANGE) for _ in range(template_count - 1)]
        left_legend = random.randint(*SIDE_MARGIN_RANGE)
        left_index = random.randint(*SIDE_MARGIN_RANGE)
        right_margin = random.randint(*RIGHT_MARGIN_RANGE)
        row_gap = random.randint(*ROW_GAP_RANGE)

        legend_width = template_count * block_w + sum(legend_gaps)
        total_w = max(left_legend + legend_width, left_index + index_w) + right_margin
        total_h = block_h + row_gap + index_h
        if total_h <= 30 and total_w <= 30:
            return block_h, block_w, index_h, index_w, legend_gaps, left_legend, left_index, row_gap, total_w
    return None


def _sample_distinct_counts(color_count: int, area: int) -> list[int]:
    for _ in range(200):
        cuts = sorted(random.sample(range(1, area), color_count - 1))
        parts = [cuts[0], *[cuts[i] - cuts[i - 1] for i in range(1, len(cuts))], area - cuts[-1]]
        if len(set(parts)) != color_count:
            continue
        counts = sorted((int(part) for part in parts), reverse=True)
        if all(counts[i] > counts[i + 1] for i in range(len(counts) - 1)):
            return counts
    raise RuntimeError("failed to sample pairwise-distinct color counts")


def _different_first_order(rank_colors: list[int]) -> list[int]:
    for _ in range(50):
        order = rank_colors[:]
        random.shuffle(order)
        if order != rank_colors and order[0] != rank_colors[0]:
            return order
    return rank_colors[1:] + rank_colors[:1]


def _render_pattern(index_h: int, index_w: int, counts_by_color: dict[int, int], first_order: list[int]) -> np.ndarray:
    flat: list[int] = []
    remaining = counts_by_color.copy()
    for color in first_order:
        flat.append(int(color))
        remaining[int(color)] -= 1
        assert remaining[int(color)] >= 0

    rest: list[int] = []
    for color, count in remaining.items():
        rest.extend([int(color)] * int(count))
    random.shuffle(rest)
    flat.extend(rest)

    pattern = np.array(flat, dtype=int).reshape(index_h, index_w)
    assert _first_appearance_order(pattern) == [int(color) for color in first_order]
    return pattern


def _render_input(
    templates: list[np.ndarray],
    pattern: np.ndarray,
    legend_gaps: list[int],
    left_legend: int,
    left_index: int,
    row_gap: int,
    total_w: int,
) -> np.ndarray:
    block_h, block_w = templates[0].shape
    total_h = block_h + row_gap + pattern.shape[0]
    grid = np.full((total_h, total_w), BG, dtype=int)

    c = left_legend
    for i, stencil in enumerate(templates):
        patch = np.where(stencil, STENCIL_COLOR, BG).astype(int)
        paste(grid, patch, 0, c)
        c += block_w
        if i < len(legend_gaps):
            c += legend_gaps[i]

    paste(grid, pattern, block_h + row_gap, left_index)
    return grid


def _numeric_orders(colors: list[int]) -> set[tuple[int, ...]]:
    asc = tuple(sorted(int(c) for c in colors))
    desc = tuple(reversed(asc))
    return {asc, desc}


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, templates: list[np.ndarray], pattern: np.ndarray) -> bool:
    fg_colors = [int(color) for color in np.unique(inp) if int(color) != BG]
    if not fg_colors:
        return False
    if int(np.count_nonzero(inp == BG)) <= max(int(np.count_nonzero(inp == color)) for color in fg_colors):
        return False

    counts = Counter(map(int, pattern.ravel()))
    if len(counts) != len(templates):
        return False
    if len(set(counts.values())) != len(counts):
        return False

    rank_order = _rank_colors_by_frequency(pattern)
    first_order = _first_appearance_order(pattern)
    if first_order == rank_order:
        return False
    if tuple(rank_order) in _numeric_orders(rank_order):
        return False

    template_keys = {tuple(int(v) for v in stencil.ravel()) for stencil in templates}
    if len(template_keys) != len(templates):
        return False

    wrong_first = _expand(pattern, first_order, templates, BG)
    if np.array_equal(wrong_first, expected):
        return False

    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    dims = _sample_dimensions(template_count)
    if dims is None:
        return None
    block_h, block_w, index_h, index_w, legend_gaps, left_legend, left_index, row_gap, total_w = dims

    templates = _sample_templates(template_count, block_h, block_w)
    if templates is None:
        return None

    rank_colors = random.sample(AVAILABLE_COLORS, template_count)
    if tuple(rank_colors) in _numeric_orders(rank_colors):
        return None

    counts = _sample_distinct_counts(template_count, index_h * index_w)
    counts_by_color = {int(color): int(count) for color, count in zip(rank_colors, counts)}
    first_order = _different_first_order(rank_colors)
    pattern = _render_pattern(index_h, index_w, counts_by_color, first_order)

    expected = _expand(pattern, rank_colors, templates, BG)
    inp = _render_input(templates, pattern, legend_gaps, left_legend, left_index, row_gap, total_w)
    if not _semantic_checks(inp, expected, templates, pattern):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frequency-rank stencil input")
