from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste as _gen_paste
import random as _gen_random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four quadrant masks split by a separator cross into one smaller grid.

    Input structure:
    - A 2D grid has one solid separator row and one solid separator column of the
      same non-background color.
    - The cross divides the grid into four equal-size regions.
    - Each region is a mask made from background plus one distinct foreground color.

    Rule:
    - Compare corresponding pixels of the four regions.
    - For each output pixel, keep the first non-background pixel in row-major
      region order: top-left, top-right, bottom-left, bottom-right.
    - If all four corresponding pixels are background, output background.

    Key insight:
    - The separator cross is not part of the answer; it presents four aligned
      layers whose visible top layer is determined by quadrant position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = get_bg(g)

    solid_rows = [
        (r, int(g[r, 0]))
        for r in range(height)
        if np.all(g[r, :] == int(g[r, 0])) and int(g[r, 0]) != bg
    ]
    solid_cols = [
        (c, int(g[0, c]))
        for c in range(width)
        if np.all(g[:, c] == int(g[0, c])) and int(g[0, c]) != bg
    ]
    crosses = [
        (r, c, row_color)
        for r, row_color in solid_rows
        for c, col_color in solid_cols
        if row_color == col_color
    ]
    assert len(crosses) == 1, "expected exactly one solid same-color separator cross"
    cross_r, cross_c, sep = crosses[0]
    assert sep != bg, "separator must be foreground"

    top_h = cross_r
    bottom_h = height - cross_r - 1
    left_w = cross_c
    right_w = width - cross_c - 1
    assert top_h > 0 and left_w > 0, "separator must split nonempty regions"
    assert top_h == bottom_h, "top and bottom regions must have equal height"
    assert left_w == right_w, "left and right regions must have equal width"

    regions = [
        g[:cross_r, :cross_c],
        g[:cross_r, cross_c + 1 :],
        g[cross_r + 1 :, :cross_c],
        g[cross_r + 1 :, cross_c + 1 :],
    ]
    layer_colors = []
    for region in regions:
        colors = [int(color) for color in np.unique(region) if int(color) != bg]
        assert len(colors) == 1, "each region must contain one foreground mask color"
        assert sep not in colors, "separator color must not appear inside a mask region"
        layer_colors.append(colors[0])
    assert len(set(layer_colors)) == len(layer_colors), "mask colors must be distinct"

    out = np.full((top_h, left_w), bg, dtype=int)
    empty = np.ones_like(out, dtype=bool)
    for region in regions:
        mask = (region != bg) & empty
        out[mask] = region[mask]
        empty &= ~mask
    return out

# --------------- generator (o5x) ---------------
# Canvas/object size bounds.
_GEN_Q_MIN, _GEN_Q_MAX = 3, 6
_GEN_BG, _GEN_TOP_LAYER, _GEN_EMPTY_WINNER = 0, 0, -1
# Color roles.
_GEN_COLOR_POOL = tuple(range(1, 10))
_GEN_LAYER_COUNT = 4
# Pixel-count constraints.
_GEN_REGION_FG_MIN, _GEN_REGION_FG_MAX = 4, 11
_GEN_OTHER_VISIBLE_MIN, _GEN_MIN_BG_CELLS, _GEN_CROSS_CENTER_CELLS = 1, 1, 1
_GEN_DENSE_OUTPUT_PERCENT, _GEN_PERCENT_SCALE = 65, 100
# Sampling/retry controls.
_GEN_VISIBLE_LAYER_COUNTS = (3, 4)
_GEN_OPTIONAL_VISIBLE_LAYERS = (1, 2, 3)
_GEN_RETRY_LIMIT = 300

def _gen_bounded_counts(lower, upper, total):
    """Sample per-layer counts satisfying inclusive bounds and a fixed total."""
    counts = list(map(int, lower))
    upper = list(map(int, upper))
    total = int(total)
    remaining = total - sum(counts)
    if remaining < 0 or total > sum(upper):
        raise ValueError("infeasible count bounds")

    order = _gen_random.sample(range(len(counts)), len(counts))
    for pos, idx in enumerate(order):
        later = order[pos + 1 :]
        max_later = sum(upper[j] - counts[j] for j in later)
        min_add = max(0, remaining - max_later)
        max_add = min(upper[idx] - counts[idx], remaining)
        add = _gen_random.randint(min_add, max_add)
        counts[idx] += add
        remaining -= add
    return counts


def _gen_visible_output(q, max_fg, fg_colors):
    """Sample the visible output and its winning layer per cell."""
    n_cells = q * q
    visible_n = _gen_random.choice(_GEN_VISIBLE_LAYER_COUNTS)
    visible_layers = [_GEN_TOP_LAYER] + sorted(_gen_random.sample(_GEN_OPTIONAL_VISIBLE_LAYERS, visible_n - 1))
    visible_lower = [_GEN_REGION_FG_MIN if layer == _GEN_TOP_LAYER else _GEN_OTHER_VISIBLE_MIN for layer in visible_layers]
    visible_upper = [max_fg for _ in visible_layers]
    dense_min = (_GEN_DENSE_OUTPUT_PERCENT * n_cells + _GEN_PERCENT_SCALE - 1) // _GEN_PERCENT_SCALE
    target_min = max(sum(visible_lower), dense_min)
    target_max = min(n_cells, sum(visible_upper))
    if target_min > target_max:
        return None, None

    target_non_bg = _gen_random.randint(target_min, target_max)
    visible_counts = _gen_bounded_counts(visible_lower, visible_upper, target_non_bg)

    winner = np.full(n_cells, _GEN_EMPTY_WINNER, dtype=int)
    positions = _gen_random.sample(range(n_cells), target_non_bg)
    offset = 0
    for layer, count in zip(visible_layers, visible_counts):
        winner[positions[offset : offset + count]] = layer
        offset += count
    winner = winner.reshape((q, q))

    out = np.zeros((q, q), dtype=int)
    for layer, color in enumerate(fg_colors):
        out[winner == layer] = color
    return winner, out


def _gen_layer_masks(winner, max_fg):
    """Reverse visible winners into masks, sampling hidden lower-priority overlap."""
    q, n_cells = int(winner.shape[0]), int(winner.size)
    masks = [winner == layer for layer in range(_GEN_LAYER_COUNT)]
    layer_lower, layer_upper, hidden_choices = [], [], []
    for layer, mask in enumerate(masks):
        visible_count = int(np.count_nonzero(mask))
        hidden = np.argwhere((winner >= _GEN_TOP_LAYER) & (winner < layer))
        hidden_choices.append(hidden)
        layer_lower.append(max(_GEN_REGION_FG_MIN, visible_count))
        layer_upper.append(min(max_fg, visible_count + len(hidden)))
        if layer_lower[-1] > layer_upper[-1]:
            return None

    sep_count = _GEN_LAYER_COUNT * q + _GEN_CROSS_CENTER_CELLS
    total_fg_cap = _GEN_LAYER_COUNT * n_cells - sep_count - _GEN_MIN_BG_CELLS
    desired_total_min, desired_total_max = sum(layer_lower), min(sum(layer_upper), total_fg_cap)
    if desired_total_min > desired_total_max:
        return None

    desired_total = _gen_random.randint(desired_total_min, desired_total_max)
    desired_counts = _gen_bounded_counts(layer_lower, layer_upper, desired_total)

    for layer in range(_GEN_TOP_LAYER + 1, _GEN_LAYER_COUNT):
        visible_count = int(np.count_nonzero(masks[layer]))
        extra = desired_counts[layer] - visible_count
        if extra:
            chosen = _gen_random.sample(range(len(hidden_choices[layer])), extra)
            mask = masks[layer].copy()
            rows, cols = hidden_choices[layer][chosen].T
            mask[rows, cols] = True
            masks[layer] = mask
    return masks


def _gen_assemble_grid(q, sep, fg_colors, masks):
    """Render quadrant masks around the separator cross."""
    regions = []
    grid = np.zeros((2 * q + _GEN_CROSS_CENTER_CELLS, 2 * q + _GEN_CROSS_CENTER_CELLS), dtype=int)
    grid[q, :] = sep
    grid[:, q] = sep
    for layer, (row, col) in enumerate(((0, 0), (0, q + _GEN_CROSS_CENTER_CELLS), (q + _GEN_CROSS_CENTER_CELLS, 0), (q + _GEN_CROSS_CENTER_CELLS, q + _GEN_CROSS_CENTER_CELLS))):
        region = np.zeros((q, q), dtype=int)
        region[masks[layer]] = fg_colors[layer]
        regions.append(region)
        _gen_paste(grid, region, row, col, transparent=None)
    return grid, regions


def _gen_attempt() -> np.ndarray | None:
    q = _gen_random.randint(_GEN_Q_MIN, _GEN_Q_MAX)
    max_fg = min(_GEN_REGION_FG_MAX, q * q)
    palette = list(_GEN_COLOR_POOL)
    _gen_random.shuffle(palette)
    sep, fg_colors = palette[0], palette[1 : 1 + _GEN_LAYER_COUNT]

    winner, out = _gen_visible_output(q, max_fg, fg_colors)
    if winner is None:
        return None
    masks = _gen_layer_masks(winner, max_fg)
    if masks is None:
        return None
    grid, regions = _gen_assemble_grid(q, sep, fg_colors, masks)

    sep_count = _GEN_LAYER_COUNT * q + _GEN_CROSS_CENTER_CELLS
    if np.count_nonzero(grid == _GEN_BG) <= sep_count:
        return None
    if any(not (_GEN_REGION_FG_MIN <= int(np.count_nonzero(region)) <= _GEN_REGION_FG_MAX) for region in regions):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
