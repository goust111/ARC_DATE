from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
SEP = 5
LAYER_COUNT = 3
TOP_PAD = 4
AREA_W_CHOICES = (5, 6)
AREA_H_RANGE = (12, 21)
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _separator_columns(grid: np.ndarray) -> tuple[int, list[int]]:
    cols: list[int] = []
    colors: list[int] = []
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != BG:
            cols.append(int(c))
            colors.append(int(grid[0, c]))
    assert cols, "expected full-height separator columns"
    assert len(set(colors)) == 1, "separator columns must share one color"
    return colors[0], cols


def _spans(width: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for col in separators:
        spans.append((start, col - 1))
        start = col + 1
    spans.append((start, width - 1))
    assert all(c0 <= c1 for c0, c1 in spans), "empty span found"
    return spans


def _legend_data(panel: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int]]]:
    legend_rows = [r for r in range(panel.shape[0]) if int(np.count_nonzero(panel[r] != BG)) == 3]
    assert len(legend_rows) == 3, "expected a three-row legend"

    pair_to_mix: dict[frozenset[int], int] = {}
    input_counts: Counter[int] = Counter()
    for r in legend_rows:
        cols = list(map(int, np.flatnonzero(panel[r] != BG)))
        assert len(cols) == 3
        a = int(panel[r, cols[0]])
        b = int(panel[r, cols[1]])
        mix = int(panel[r, cols[2]])
        pair = frozenset((a, b))
        assert len(pair) == 2, "legend inputs must use two distinct colors"
        pair_to_mix[pair] = mix
        input_counts[a] += 1
        input_counts[b] += 1

    base_colors = [color for color, count in input_counts.items() if count == 2]
    assert len(base_colors) == 3, "legend must describe three base colors"
    mix_to_pair = {mix: tuple(pair) for pair, mix in pair_to_mix.items()}
    return base_colors, mix_to_pair


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "empty layer mask"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return r0, c0, r1, c1


def _interval_overlapping(base0: int, base1: int, max_size: int) -> tuple[int, int] | None:
    width = int(_RNG.integers(2, max_size + 1))
    lo = max(0, base0 - width + 1)
    hi = min(max_size - width, base1)
    if lo > hi:
        return None
    start = int(_RNG.integers(lo, hi + 1))
    return start, start + width - 1


def _vertical_chain(area_h: int, area_w: int) -> list[tuple[int, int, int, int]] | None:
    usable = area_h - TOP_PAD
    for _ in range(80):
        h1 = int(_RNG.integers(2, min(7, usable - 2)))
        h3 = int(_RNG.integers(2, min(7, usable - 2)))
        remaining = usable - h1 - h3
        if remaining < 2:
            continue
        gap = int(_RNG.integers(1, remaining))
        slack = remaining - gap
        top_slack = int(_RNG.integers(0, slack + 1))
        a0 = TOP_PAD + top_slack
        a1 = a0 + h1 - 1
        c0 = a1 + gap + 1
        c1 = c0 + h3 - 1
        if c1 >= area_h:
            continue

        b0 = int(_RNG.integers(a0, a1 + 1))
        b1 = int(_RNG.integers(c0, c1 + 1))
        if b1 - b0 + 1 < 2:
            continue

        wb = int(_RNG.integers(3, area_w + 1))
        bc0 = int(_RNG.integers(0, area_w - wb + 1))
        bc1 = bc0 + wb - 1

        a_cols = _interval_overlapping(bc0, bc1, area_w)
        c_cols = _interval_overlapping(bc0, bc1, area_w)
        if a_cols is None or c_cols is None:
            continue

        rects = [
            (a0, a_cols[0], a1, a_cols[1]),
            (b0, bc0, b1, bc1),
            (c0, c_cols[0], c1, c_cols[1]),
        ]
        return rects
    return None


def _horizontal_chain(area_h: int, area_w: int) -> list[tuple[int, int, int, int]] | None:
    usable = area_h - TOP_PAD
    for _ in range(80):
        w1 = int(_RNG.integers(2, min(4, area_w)))
        w3 = int(_RNG.integers(2, min(4, area_w)))
        remaining = area_w - w1 - w3
        if remaining < 1:
            continue
        gap = int(_RNG.integers(1, remaining + 1))
        slack = remaining - gap
        left_slack = int(_RNG.integers(0, slack + 1))
        a_c0 = left_slack
        a_c1 = a_c0 + w1 - 1
        c_c0 = a_c1 + gap + 1
        c_c1 = c_c0 + w3 - 1
        if c_c1 >= area_w:
            continue

        b_c0 = int(_RNG.integers(a_c0, a_c1 + 1))
        b_c1 = int(_RNG.integers(c_c0, c_c1 + 1))
        if b_c1 - b_c0 + 1 < 2:
            continue

        hb = int(_RNG.integers(3, usable + 1))
        br0 = int(_RNG.integers(TOP_PAD, area_h - hb + 1))
        br1 = br0 + hb - 1

        a_rows = _interval_overlapping(br0 - TOP_PAD, br1 - TOP_PAD, usable)
        c_rows = _interval_overlapping(br0 - TOP_PAD, br1 - TOP_PAD, usable)
        if a_rows is None or c_rows is None:
            continue

        rects = [
            (TOP_PAD + a_rows[0], a_c0, TOP_PAD + a_rows[1], a_c1),
            (br0, b_c0, br1, b_c1),
            (TOP_PAD + c_rows[0], c_c0, TOP_PAD + c_rows[1], c_c1),
        ]
        return rects
    return None


def _rect_mask(area_h: int, area_w: int, rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((area_h, area_w), dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _pair_key(a: int, b: int) -> frozenset[int]:
    return frozenset((int(a), int(b)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate overlapped transparent rectangle layers into sorted side panels.

    Input structure:
    - Black background with one left workspace and three equally sized right panels, separated by full-height grey columns.
    - The workspace contains three solid axis-aligned color layers whose pairwise overlaps are shown by third "mix" colors instead of the original colors.
    - The top-right panel holds a three-row legend; in each row, the first two colored pixels are source colors and the rightmost colored pixel is their overlap result.

    Rule:
    - Read the legend to recover which mix color corresponds to each unordered pair of original colors.
    - In the workspace, treat every pure original-color pixel as belonging to one layer and every mix-color pixel as belonging to both source layers from the legend.
    - Reconstruct each layer as the full bounding rectangle of all pixels assigned to it, then draw those monochrome rectangles into the three right panels at the same relative positions.
    - Order the filled panels from left to right by descending rectangle area; keep the workspace, legend, and grey separators unchanged.

    Key insight:
    - The mix colors are not independent objects: once the legend turns each overlap color back into a pair of layer memberships, each hidden layer is exactly the rectangle spanning all of its supporting pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep, sep_cols = _separator_columns(g)
    assert sep == SEP, "expected grey separators"
    spans = _spans(g.shape[1], sep_cols)
    widths = {c1 - c0 + 1 for c0, c1 in spans}
    assert len(widths) == 1, "workspace and panels must have equal widths"

    workspace_c0, workspace_c1 = spans[0]
    workspace = g[:, workspace_c0 : workspace_c1 + 1]
    base_colors, mix_to_pair = _legend_data(g[:, spans[-1][0] : spans[-1][1] + 1])
    assert len(spans) == len(base_colors) + 1

    layer_masks = {color: workspace == color for color in base_colors}
    for mix_color, pair in mix_to_pair.items():
        mix_mask = workspace == mix_color
        for color in pair:
            layer_masks[color] |= mix_mask

    layers: list[tuple[int, int, int, int, int, int]] = []
    for color in base_colors:
        r0, c0, r1, c1 = _bbox(layer_masks[color])
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        layers.append((area, r0, c0, r1, c1, color))

    ordered = sorted(layers, key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    out = g.copy()
    for panel_idx, layer in enumerate(ordered, start=1):
        _area, r0, c0, r1, c1, color = layer
        panel_c0, _panel_c1 = spans[panel_idx]
        out[r0 : r1 + 1, panel_c0 + c0 : panel_c0 + c1 + 1] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    area_w = int(_RNG.choice(AREA_W_CHOICES))
    area_h = int(_RNG.integers(*AREA_H_RANGE))

    if bool(_RNG.integers(0, 2)):
        rects = _vertical_chain(area_h, area_w)
    else:
        rects = _horizontal_chain(area_h, area_w)
    if rects is None:
        return None

    sampled = list(map(int, _RNG.choice(PALETTE, size=6, replace=False)))
    base_colors = sampled[:LAYER_COUNT]
    mix_colors = sampled[LAYER_COUNT:]
    role_colors = list(base_colors)
    _RNG.shuffle(role_colors)

    pair_rows = [((0, 1), mix_colors[0]), ((0, 2), mix_colors[1]), ((1, 2), mix_colors[2])]
    pair_to_mix = {_pair_key(base_colors[i], base_colors[j]): mix for (i, j), mix in pair_rows}

    layer_masks = np.stack([_rect_mask(area_h, area_w, rect) for rect in rects], axis=0)
    counts = layer_masks.sum(axis=0)
    if int(counts.max()) > 2:
        return None
    if np.any(layer_masks[0] & layer_masks[2]):
        return None

    work = np.zeros((area_h, area_w), dtype=int)
    for idx, color in enumerate(role_colors):
        pure = layer_masks[idx] & (counts == 1)
        if not pure.any():
            return None
        work[pure] = color

    edge_pairs = [(0, 1), (1, 2)]
    for i, j in edge_pairs:
        overlap = layer_masks[i] & layer_masks[j]
        if not overlap.any():
            return None
        mix = pair_to_mix[_pair_key(role_colors[i], role_colors[j])]
        work[overlap] = mix

    areas = [int(mask.sum()) for mask in layer_masks]
    if len(set(areas)) != LAYER_COUNT:
        return None

    total_w = (LAYER_COUNT + 1) * area_w + LAYER_COUNT
    if max(area_h, total_w) > 30:
        return None

    grid = np.zeros((area_h, total_w), dtype=int)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for idx in range(LAYER_COUNT + 1):
        spans.append((cursor, cursor + area_w - 1))
        cursor += area_w
        if idx < LAYER_COUNT:
            grid[:, cursor] = SEP
            cursor += 1

    grid[:, spans[0][0] : spans[0][1] + 1] = work

    legend_rows = [(_pair_key(base_colors[0], base_colors[1]), mix_colors[0]),
                   (_pair_key(base_colors[0], base_colors[2]), mix_colors[1]),
                   (_pair_key(base_colors[1], base_colors[2]), mix_colors[2])]
    _RNG.shuffle(legend_rows)
    last_panel_c0, _last_panel_c1 = spans[-1]
    legend_cols = [area_w - 5, area_w - 3, area_w - 1]
    for row, (pair, mix) in enumerate(legend_rows):
        a, b = tuple(pair)
        if bool(_RNG.integers(0, 2)):
            a, b = b, a
        grid[row, last_panel_c0 + legend_cols[0]] = a
        grid[row, last_panel_c0 + legend_cols[1]] = b
        grid[row, last_panel_c0 + legend_cols[2]] = mix

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
