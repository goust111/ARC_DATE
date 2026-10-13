from __future__ import annotations

import numpy as np

from arc_utils import components_mask, get_bg, paste


def _find_legends_and_markers(g: np.ndarray, bg: int):
    height, width = map(int, g.shape)
    legends = []
    markers = []

    for coords in components_mask(g != int(bg), connectivity=4):
        if len(coords) == 1:
            r, c = map(int, coords[0])
            markers.append((r, c))
            continue

        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        assert r0 == r1, "multi-pixel foreground components must be horizontal legends"
        assert len(coords) == c1 - c0 + 1, "legend components must be contiguous horizontal runs"
        assert r0 in (0, height - 1), "legend components must sit on the top or bottom edge"
        assert 0 <= c0 <= c1 < width

        seq = tuple(int(g[r0, c]) for c in range(c0, c1 + 1))
        assert len(seq) >= 2, "legend must contain at least two colors"
        assert all(color != int(bg) for color in seq), "legend colors must be foreground"
        legends.append(seq)

    assert legends, "expected at least one border legend"
    assert markers, "expected at least one isolated marker"
    return legends, markers


def _legend_occurrence(legends: list[tuple[int, ...]], color: int) -> tuple[tuple[int, ...], int]:
    choices = []
    for seq in legends:
        for idx, value in enumerate(seq):
            if int(value) == int(color):
                choices.append((seq, idx))
    assert len(choices) == 1, "each marker color must occur in exactly one legend position"
    return choices[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each isolated marker into the edge legend string containing its color, aligned so the marker keeps that legend position.

    Input structure
    - The background is the dominant color.
    - One or more top- or bottom-edge horizontal foreground runs are legend strings.
    - All other foreground components are single-pixel markers whose colors appear in exactly one legend position.

    Rule
    - For each marker, find the unique legend string containing the marker color.
    - Copy that string horizontally on the marker row, shifted so the matching legend entry lands on the marker pixel.
    - Clip the copied string at the left or right grid boundary when needed.

    Key insight
    - The isolated pixel is not just recolored; its color gives both the legend string to copy and the anchor index inside that string.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    height, width = map(int, g.shape)

    legends, markers = _find_legends_and_markers(g, bg)
    out = g.copy()

    for r, c in markers:
        color = int(g[r, c])
        seq, anchor_idx = _legend_occurrence(legends, color)
        start_c = int(c) - int(anchor_idx)

        for offset, value in enumerate(seq):
            cc = start_c + offset
            if 0 <= cc < width:
                existing = int(out[r, cc])
                assert existing in (int(bg), int(value)), "legend copies must not conflict"

        patch = np.asarray([seq], dtype=int)
        paste(out, patch, int(r), start_c, clip=True)

    assert out.shape == (height, width)
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds.
CANVAS_HEIGHT_RANGE = (12, 18)
CANVAS_WIDTH_RANGE = (13, 19)
BACKGROUND_COLOR = 0

# Legend counts, lengths, colors, and edge spacing.
LEGEND_COUNT_RANGE = range(1, 4)
LEGEND_LEN_RANGE = range(3, 7)
TOTAL_LEGEND_CELLS_RANGE = (6, 12)
MAX_LEGENDS_PER_EDGE = 2
EDGE_LEGEND_GAP = 7
LEGEND_COLOR_POOL = np.arange(1, 10, dtype=int)

# Marker counts and clipping frequency.
MARKER_COUNT_RANGE = (3, 7)
CLIPPED_SAMPLE_PROB = 0.35

# Generation retry budget.
MAX_GENERATION_ATTEMPTS = 300

def _gen_pick(options):
    return options[int(np.random.randint(len(options)))]

def _gen_edge_layouts(indices, lengths, width, row):
    if not indices:
        return [()]
    if len(indices) == 1:
        idx = indices[0]
        length = lengths[idx]
        return [((idx, row, c0),) for c0 in range(width - length + 1)]
    if len(indices) > MAX_LEGENDS_PER_EDGE:
        return []

    layouts = []
    a, b = indices
    for first, second in ((a, b), (b, a)):
        first_len = lengths[first]
        second_len = lengths[second]
        max_first = width - first_len - EDGE_LEGEND_GAP - second_len
        for first_c0 in range(max_first + 1):
            min_second_c0 = first_c0 + first_len + EDGE_LEGEND_GAP
            for second_c0 in range(min_second_c0, width - second_len + 1):
                layouts.append(((first, row, first_c0), (second, row, second_c0)))
    return layouts


def _gen_legend_layouts(lengths, height, width):
    count = len(lengths)
    layouts = []
    for mask in range(1 << count):
        top = [i for i in range(count) if (mask >> i) & 1]
        bottom = [i for i in range(count) if not ((mask >> i) & 1)]
        if len(top) > MAX_LEGENDS_PER_EDGE or len(bottom) > MAX_LEGENDS_PER_EDGE:
            continue
        top_layouts = _gen_edge_layouts(top, lengths, width, 0)
        bottom_layouts = _gen_edge_layouts(bottom, lengths, width, height - 1)
        for top_layout in top_layouts:
            for bottom_layout in bottom_layouts:
                layout = {}
                for idx, row, c0 in top_layout + bottom_layout:
                    layout[idx] = (row, c0)
                layouts.append(layout)
    return layouts


def _gen_length_tuples(count, width, height):
    options = []

    def rec(prefix):
        min_total, max_total = TOTAL_LEGEND_CELLS_RANGE
        if len(prefix) == count:
            total = sum(prefix)
            if (
                min_total <= total <= max_total
                and _gen_legend_layouts(prefix, height, width)
            ):
                options.append(tuple(prefix))
            return
        for length in LEGEND_LEN_RANGE:
            remaining = count - len(prefix) - 1
            if sum(prefix) + length + LEGEND_LEN_RANGE.start * remaining > max_total:
                continue
            rec(prefix + [length])

    rec([])
    return options


def _gen_make_legends(lengths):
    slots = [(legend_i, pos) for legend_i, length in enumerate(lengths) for pos in range(length)]
    order = list(np.random.permutation(len(slots)))
    colors = list(np.random.permutation(LEGEND_COLOR_POOL))
    legends = [[None for _ in range(length)] for length in lengths]
    used_by_legend = [set() for _ in lengths]
    color_counts = {int(color): 0 for color in LEGEND_COLOR_POOL}

    base_count = min(len(slots), len(LEGEND_COLOR_POOL))
    for order_i in order[:base_count]:
        legend_i, pos = slots[order_i]
        color = int(colors.pop())
        legends[legend_i][pos] = color
        used_by_legend[legend_i].add(color)
        color_counts[color] += 1

    repeated = set()
    for order_i in order[base_count:]:
        legend_i, pos = slots[order_i]
        choices = [
            color
            for color, count in color_counts.items()
            if count == 1 and color not in repeated and color not in used_by_legend[legend_i]
        ]
        if not choices:
            return None
        color = int(choices[int(np.random.randint(len(choices)))])
        legends[legend_i][pos] = color
        used_by_legend[legend_i].add(color)
        color_counts[color] += 1
        repeated.add(color)

    return [tuple(int(v) for v in seq) for seq in legends]


def _gen_choose_legend_plan(height, width):
    length_options_by_count = {
        count: _gen_length_tuples(count, width, height)
        for count in LEGEND_COUNT_RANGE
    }
    count_options = [count for count in LEGEND_COUNT_RANGE if length_options_by_count[count]]
    legend_count = _gen_pick(count_options)
    lengths = _gen_pick(length_options_by_count[legend_count])
    layout = _gen_pick(_gen_legend_layouts(lengths, height, width))
    return lengths, layout


def _gen_unique_occurrences(legends):
    positions = {}
    counts = {}
    for legend_i, seq in enumerate(legends):
        for pos, color in enumerate(seq):
            color = int(color)
            counts[color] = counts.get(color, 0) + 1
            positions[color] = (legend_i, pos)
    return {color: positions[color] for color, count in counts.items() if count == 1}


def _gen_visible_interval(width, length, anchor, marker_c):
    start = marker_c - anchor
    return max(0, start), min(width - 1, start + length - 1)


def _gen_marker_columns(
    width,
    seq,
    anchor,
    *,
    clipped,
    forbidden_cols,
    occupied_intervals,
):
    length = len(seq)
    if clipped:
        candidates = []
        if anchor > 0:
            candidates.append(anchor - 1)
        if anchor < length - 1:
            candidates.append(width - length + anchor + 1)
    else:
        candidates = list(range(anchor, width - length + anchor + 1))

    good = []
    for marker_c in candidates:
        if not (0 <= marker_c < width) or marker_c in forbidden_cols:
            continue
        vis0, vis1 = _gen_visible_interval(width, length, anchor, marker_c)
        if any(not (vis1 < old0 or vis0 > old1) for old0, old1 in occupied_intervals):
            continue
        good.append(marker_c)
    return good


def _gen_render_edge_legends(height, width, legends, layout):
    out = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    edge_cells = {0: set(), height - 1: set()}
    for legend_i, seq in enumerate(legends):
        row, c0 = layout[legend_i]
        paste(out, np.asarray([seq], dtype=int), row, c0, clip=False)
        edge_cells[row].update(range(c0, c0 + len(seq)))
    return out, edge_cells


def _gen_forbidden_marker_cols(row, height, edge_cells, marker_positions):
    forbidden = set()
    if row == 1:
        forbidden.update(edge_cells[0])
    if row == height - 2:
        forbidden.update(edge_cells[height - 1])
    for old_row, old_c in marker_positions:
        if row == old_row:
            forbidden.update((old_c - 1, old_c, old_c + 1))
        elif abs(row - old_row) == 1:
            forbidden.add(old_c)
    return forbidden


def _gen_place_marker_expansions(out, legends, unique, marker_colors, edge_cells):
    height, width = map(int, out.shape)
    rows = list(np.random.permutation(np.arange(1, height - 1, dtype=int)))[: len(marker_colors)]
    clipped_i = -1
    if np.random.random() < CLIPPED_SAMPLE_PROB:
        clipped_i = int(np.random.randint(len(marker_colors)))
    placements = []
    occupied_by_row = {}
    marker_positions = []

    for i, color in enumerate(marker_colors):
        legend_i, anchor = unique[int(color)]
        seq = legends[legend_i]
        row = int(rows[i])
        forbidden = _gen_forbidden_marker_cols(row, height, edge_cells, marker_positions)
        columns = _gen_marker_columns(
            width,
            seq,
            anchor,
            clipped=(i == clipped_i),
            forbidden_cols=forbidden,
            occupied_intervals=occupied_by_row.get(row, []),
        )
        if not columns:
            return None
        marker_c = int(_gen_pick(columns))
        start = marker_c - anchor
        paste(out, np.asarray([seq], dtype=int), row, start, clip=True)
        occupied_by_row.setdefault(row, []).append(
            _gen_visible_interval(width, len(seq), anchor, marker_c)
        )
        placements.append((row, marker_c, int(color)))
        marker_positions.append((row, marker_c))

    return placements


def _gen_attempt():
    height = int(np.random.randint(*CANVAS_HEIGHT_RANGE))
    width = int(np.random.randint(*CANVAS_WIDTH_RANGE))
    lengths, layout = _gen_choose_legend_plan(height, width)
    legends = _gen_make_legends(lengths)
    if legends is None:
        return None

    unique = _gen_unique_occurrences(legends)
    marker_count = int(np.random.randint(*MARKER_COUNT_RANGE))
    if len(unique) < marker_count:
        return None
    marker_colors = list(np.random.permutation(list(unique)))[:marker_count]

    out, edge_cells = _gen_render_edge_legends(height, width, legends, layout)
    placements = _gen_place_marker_expansions(
        out, legends, unique, marker_colors, edge_cells
    )
    if placements is None:
        return None

    inp = out.copy()
    inp[1:-1, :] = BACKGROUND_COLOR
    for row, marker_c, color in placements:
        inp[row, marker_c] = color

    try:
        solved = main(inp)
    except AssertionError:
        return None

    if np.array_equal(solved, out):
        return inp
    return None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
