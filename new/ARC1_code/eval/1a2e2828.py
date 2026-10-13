from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, get_bg
import random


Bar = tuple[str, tuple[int, int, int, int]]


def _bar_info(g: np.ndarray, color: int) -> Bar:
    h, w = g.shape
    b = bbox(g == color)
    assert b is not None, "each foreground color must be visible"
    r0, c0, r1, c1 = b

    spans_rows = r0 == 0 and r1 == h - 1
    spans_cols = c0 == 0 and c1 == w - 1
    assert spans_rows != spans_cols, "each color must be exactly one full-height or full-width bar"

    orientation = "vertical" if spans_rows else "horizontal"
    return orientation, b


def _overlap_edges(g: np.ndarray, bars: dict[int, Bar]) -> set[tuple[int, int]]:
    colors = list(bars)
    edges: set[tuple[int, int]] = set()

    for i, color_a in enumerate(colors):
        orient_a, bbox_a = bars[color_a]
        for color_b in colors[i + 1 :]:
            orient_b, bbox_b = bars[color_b]
            if orient_a == orient_b:
                continue

            if orient_a == "horizontal":
                horizontal_color, horizontal_bbox = color_a, bbox_a
                vertical_color, vertical_bbox = color_b, bbox_b
            else:
                horizontal_color, horizontal_bbox = color_b, bbox_b
                vertical_color, vertical_bbox = color_a, bbox_a

            hr0, _hc0, hr1, _hc1 = horizontal_bbox
            _vr0, vc0, _vr1, vc1 = vertical_bbox
            visible = np.unique(g[hr0 : hr1 + 1, vc0 : vc1 + 1])
            assert visible.size == 1, "a bar crossing must show a single top color"

            top = int(visible[0])
            assert top in {horizontal_color, vertical_color}, "a crossing must show one of its two bars"
            if top == horizontal_color:
                edges.add((vertical_color, horizontal_color))
            else:
                edges.add((horizontal_color, vertical_color))

    return edges


def _unique_top_color(colors: list[int], edges: set[tuple[int, int]]) -> int:
    higher: dict[int, set[int]] = {color: set() for color in colors}
    for lower, upper in edges:
        assert lower != upper, "layer constraint cannot point to itself"
        higher[lower].add(upper)

    top_colors = [color for color in colors if not higher[color]]
    assert len(top_colors) == 1, "crossings must determine a unique top bar"
    return top_colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the color of the unique topmost bar in a stack of overlaid horizontal and vertical bars.

    Input structure:
    - The background is the dominant color.
    - Every foreground color is one straight bar whose visible bbox spans either
      the full grid height or the full grid width.
    - Perpendicular bars overlap, and each overlap block shows whichever bar is
      higher in the stack.

    Rule:
    - Reconstruct each foreground color as a full-height or full-width bar from
      the full-span direction of its visible bbox.
    - At every horizontal/vertical crossing, add a lower-to-higher constraint
      toward the visible color.
    - Output the single foreground color with no higher bar above it as a 1x1 grid.

    Key insight:
    - The answer is not the largest or most frequent bar; the crossings encode a
      partial layer order, and the requested color is the unique maximal layer.
    """
    g = np.asarray(grid, dtype=int)
    colors = colors_present(g, exclude={get_bg(g)})
    assert colors, "input must contain foreground bars"

    bars = {color: _bar_info(g, color) for color in colors}
    edges = _overlap_edges(g, bars)
    assert edges, "foreground bars must cross to reveal layer constraints"

    top = _unique_top_color(colors, edges)
    return np.array([[top]], dtype=int)

# --------------- generator (o5x) ---------------

# Canvas and palette sampling.
GRID_ROWS, GRID_COLS = (3, 11), (3, 13)
BAR_COUNT_RANGE, FOREGROUND_COLORS = range(2, 6), range(1, 10)
GENERATION_RETRY_LIMIT = 300

# Bar orientation, thickness, and cross-axis spacing.
HORIZONTAL, VERTICAL = "h", "v"
HORIZONTAL_THICKNESSES, VERTICAL_THICKNESSES = (1, 2, 3), (1, 2)
EDGE_PADDING, SAME_ORIENT_GAP = 1, 1

GenBar = dict[str, int | str]


def _gen_thickness_options(length: int, count: int, choices: tuple[int, ...]) -> list[tuple[int, ...]]:
    avail = length - 2 * EDGE_PADDING
    out: list[tuple[int, ...]] = []

    def rec(prefix: tuple[int, ...]) -> None:
        if len(prefix) == count:
            if sum(prefix) + SAME_ORIENT_GAP * (count - 1) <= avail:
                out.append(prefix)
            return
        for t in choices:
            used = sum(prefix) + t + SAME_ORIENT_GAP * len(prefix)
            if used <= avail:
                rec(prefix + (t,))

    rec(())
    return out


def _gen_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    vals: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        vals.append(cut - prev - 1)
        prev = cut
    return vals


def _gen_intervals(length: int, count: int, choices: tuple[int, ...]) -> list[tuple[int, int]]:
    options = _gen_thickness_options(length, count, choices)
    if not options:
        return []
    thicknesses = list(random.choice(options))
    random.shuffle(thicknesses)
    extra = length - 2 * EDGE_PADDING - sum(thicknesses) - SAME_ORIENT_GAP * (count - 1)
    gaps = _gen_composition(extra, count + 1)

    pos = EDGE_PADDING + gaps[0]
    intervals: list[tuple[int, int]] = []
    for i, t in enumerate(thicknesses):
        intervals.append((pos, pos + t - 1))
        pos += t
        if i < count - 1:
            pos += SAME_ORIENT_GAP + gaps[i + 1]
    return intervals


def _gen_valid_splits(h: int, w: int) -> list[tuple[int, int]]:
    splits: list[tuple[int, int]] = []
    for n in BAR_COUNT_RANGE:
        for h_count in range(1, n):
            v_count = n - h_count
            if _gen_thickness_options(h, h_count, HORIZONTAL_THICKNESSES) and _gen_thickness_options(w, v_count, VERTICAL_THICKNESSES):
                splits.append((h_count, v_count))
    return splits


def _gen_layer_order(bars: list[GenBar]) -> list[GenBar]:
    h_bars = [b for b in bars if b["orient"] == HORIZONTAL]
    v_bars = [b for b in bars if b["orient"] == VERTICAL]
    top_orient = random.choice((HORIZONTAL, VERTICAL))
    same = h_bars if top_orient == HORIZONTAL else v_bars
    opp = v_bars if top_orient == HORIZONTAL else h_bars

    top = random.choice(same)
    lower_same = [b for b in same if b is not top]
    lower_opp = opp[:]
    random.shuffle(lower_opp)

    if lower_same:
        witness_count = random.randint(1, len(lower_opp))
        upper_opp = lower_opp[:witness_count]
        lower = lower_opp[witness_count:] + lower_same
    else:
        upper_opp = []
        lower = lower_opp

    random.shuffle(lower)
    random.shuffle(upper_opp)
    return lower + upper_opp + [top]


def _gen_paint_bar(grid: np.ndarray, bar: GenBar) -> None:
    color = int(bar["color"])
    if bar["orient"] == HORIZONTAL:
        grid[int(bar["lo"]) : int(bar["hi"]) + 1, :] = color
    else:
        grid[:, int(bar["lo"]) : int(bar["hi"]) + 1] = color


def _gen_bar_layout(h: int, w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    splits = _gen_valid_splits(h, w)
    if not splits:
        return None

    h_count, v_count = random.choice(splits)
    h_intervals = _gen_intervals(h, h_count, HORIZONTAL_THICKNESSES)
    v_intervals = _gen_intervals(w, v_count, VERTICAL_THICKNESSES)
    if not h_intervals or not v_intervals:
        return None
    return h_intervals, v_intervals


def _gen_bars(h_intervals: list[tuple[int, int]], v_intervals: list[tuple[int, int]]) -> list[GenBar]:
    colors = random.sample(FOREGROUND_COLORS, len(h_intervals) + len(v_intervals))
    bars: list[GenBar] = []
    for lo, hi in h_intervals:
        bars.append({"orient": HORIZONTAL, "lo": lo, "hi": hi, "color": colors.pop()})
    for lo, hi in v_intervals:
        bars.append({"orient": VERTICAL, "lo": lo, "hi": hi, "color": colors.pop()})
    return bars


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    layout = _gen_bar_layout(h, w)
    if layout is None:
        return None

    h_intervals, v_intervals = layout
    bars = _gen_bars(h_intervals, v_intervals)
    order = _gen_layer_order(bars)
    grid = np.zeros((h, w), dtype=int)
    for bar in order:
        _gen_paint_bar(grid, bar)

    expected = int(order[-1]["color"])
    try:
        if np.array_equal(main(grid), np.array([[expected]], dtype=int)):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_RETRY_LIMIT} attempts")
