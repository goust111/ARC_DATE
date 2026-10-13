from __future__ import annotations

import random

import numpy as np


GREY = 5
WHITE = 10
MARKER_COLORS = {GREY, WHITE}
LEGEND_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_LEN_RANGE = (4, 7)
CONTENT_WIDTH_RANGE = (12, 24)
MIN_REGION_SIDE = 3
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _detect_legend_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for c in (0, w - 1):
        column = list(map(int, grid[:, c]))
        if len(set(column)) == h and not any(color in MARKER_COLORS for color in column):
            candidates.append(c)
    assert len(candidates) == 1, "expected exactly one edge legend column"
    return int(candidates[0])


def _restore_region_colors(content: np.ndarray) -> np.ndarray:
    restored = np.array(content, copy=True)
    marker_coords = np.argwhere(np.isin(content, list(MARKER_COLORS)))
    for r, c in marker_coords:
        neighbor_colors = {
            int(content[nr, nc])
            for nr, nc in _neighbors4(content.shape, int(r), int(c))
            if int(content[nr, nc]) not in MARKER_COLORS
        }
        assert len(neighbor_colors) == 1, "each marker must sit strictly inside one region"
        restored[int(r), int(c)] = neighbor_colors.pop()
    return restored


def _color_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(grid.shape, cr, cc):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append((color, coords))
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each rectangular color region along the legend cycle by its net grey-minus-white marker count.

    Input structure:
    - One edge column is a vertical legend whose top-to-bottom colors are all distinct.
    - The remaining area is tiled by solid rectangular color regions whose base colors all appear in the legend.
    - Grey and white marker pixels are embedded inside those rectangles and may appear in different counts per region.

    Rule:
    - Read the legend top-to-bottom as a cyclic color ring.
    - For each rectangular region, compute net steps = number of grey markers minus number of white markers inside it.
    - Move the region's current legend color forward by that many steps on the cycle, wrapping around for large or negative steps, and recolor the entire region including its markers to the reached color.
    - Keep the legend unchanged.

    Key insight:
    - Grey and white markers are opposite-direction ticks on the same color dial, so only their net difference matters before rotating each region around the legend cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_col = _detect_legend_column(g)
    legend = list(map(int, g[:, legend_col]))
    ring_index = {color: idx for idx, color in enumerate(legend)}
    assert len(ring_index) == len(legend), "legend colors must be distinct"

    if legend_col == 0:
        content = g[:, 1:]
        left_legend = True
    else:
        content = g[:, :-1]
        left_legend = False

    restored = _restore_region_colors(content)
    out_content = np.array(content, copy=True)

    for color, coords in _color_components(restored):
        assert color in ring_index, "every region color must appear in the legend"
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        region_mask = restored[r0 : r1 + 1, c0 : c1 + 1] == color
        assert region_mask.all(), "each connected region must be a solid rectangle"

        grey_count = 0
        white_count = 0
        for r, c in coords:
            if int(content[r, c]) == GREY:
                grey_count += 1
            elif int(content[r, c]) == WHITE:
                white_count += 1

        shift = grey_count - white_count
        target = legend[(ring_index[color] + shift) % len(legend)]
        for r, c in coords:
            out_content[r, c] = target

    if left_legend:
        return np.concatenate([g[:, :1], out_content], axis=1)
    return np.concatenate([out_content, g[:, -1:]], axis=1)


def _partition_width(total: int, parts: int, min_width: int) -> list[int]:
    remaining = total - parts * min_width
    widths = [min_width] * parts
    for i in range(parts - 1):
        add = random.randint(0, remaining)
        widths[i] += add
        remaining -= add
    widths[-1] += remaining
    random.shuffle(widths)
    return widths


def _build_rectangles(height: int, width: int) -> list[tuple[int, int, int, int]]:
    if height >= 6 and random.random() < 0.55:
        split = random.randint(MIN_REGION_SIDE, height - MIN_REGION_SIDE)
        band_heights = [split, height - split]
    else:
        band_heights = [height]

    max_cols = width // MIN_REGION_SIDE
    n_cols = random.randint(2, min(5, max_cols))
    widths = _partition_width(width, n_cols, MIN_REGION_SIDE)

    rects: list[tuple[int, int, int, int]] = []
    row_start = 0
    for band_height in band_heights:
        col_start = 0
        for band_width in widths:
            rects.append((row_start, col_start, row_start + band_height, col_start + band_width))
            col_start += band_width
        row_start += band_height
    return rects


def _adjacent(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    vertical_touch = ac1 == bc0 or bc1 == ac0
    row_overlap = max(ar0, br0) < min(ar1, br1)
    horizontal_touch = ar1 == br0 or br1 == ar0
    col_overlap = max(ac0, bc0) < min(ac1, bc1)
    return (vertical_touch and row_overlap) or (horizontal_touch and col_overlap)


def _assign_region_colors(rects: list[tuple[int, int, int, int]], legend: list[int]) -> list[int] | None:
    neighbors = [[] for _ in rects]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _adjacent(rects[i], rects[j]):
                neighbors[i].append(j)
                neighbors[j].append(i)

    order = sorted(range(len(rects)), key=lambda idx: -len(neighbors[idx]))
    colors = [-1] * len(rects)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        used = {colors[nbr] for nbr in neighbors[idx] if colors[nbr] != -1}
        options = [color for color in legend if color not in used]
        random.shuffle(options)
        for color in options:
            colors[idx] = color
            if backtrack(pos + 1):
                return True
        colors[idx] = -1
        return False

    if backtrack(0):
        return [int(color) for color in colors]
    return None


def _interior_marker_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = []
    for r in range(r0 + 1, r1 - 1):
        for c in range(c0 + 1, c1 - 1):
            if (r + c) % 2 == 0:
                cells.append((r, c))
    return cells


def _sample_counts(capacity: int, mode: str) -> tuple[int, int]:
    assert capacity >= 1
    if mode == "zero":
        if capacity >= 2 and random.random() < 0.7:
            white = random.randint(1, capacity // 2)
            grey = white
            return grey, white
        return 0, 0

    if mode == "pos":
        net = random.randint(1, min(3, capacity))
        max_white = max(0, (capacity - net) // 2)
        white = random.randint(0, max_white)
        grey = white + net
        return grey, white

    net = random.randint(1, min(3, capacity))
    max_grey = max(0, (capacity - net) // 2)
    grey = random.randint(0, max_grey)
    white = grey + net
    return grey, white


def _marker_modes(count: int) -> list[str]:
    base = ["pos", "neg", "zero"]
    extra = ["pos", "neg", "zero"]
    while len(base) < count:
        base.append(random.choice(extra))
    random.shuffle(base)
    return base[:count]


def _paint_regions(
    legend: list[int],
    rects: list[tuple[int, int, int, int]],
    region_colors: list[int],
    legend_on_left: bool,
) -> np.ndarray | None:
    h = len(legend)
    content_w = max(c1 for _, _, _, c1 in rects)
    content = np.zeros((h, content_w), dtype=int)
    out_counts: list[tuple[int, int]] = []

    capacities = [_interior_marker_cells(rect) for rect in rects]
    if any(len(cells) == 0 for cells in capacities):
        return None

    modes = _marker_modes(len(rects))
    positive = negative = zero = 0
    for rect, color, cells, mode in zip(rects, region_colors, capacities, modes):
        r0, c0, r1, c1 = rect
        content[r0:r1, c0:c1] = color
        grey_count, white_count = _sample_counts(len(cells), mode)
        if grey_count > white_count:
            positive += 1
        elif grey_count < white_count:
            negative += 1
        else:
            zero += 1
        marker_cells = random.sample(cells, grey_count + white_count)
        for r, c in marker_cells[:grey_count]:
            content[r, c] = GREY
        for r, c in marker_cells[grey_count:]:
            content[r, c] = WHITE
        out_counts.append((grey_count, white_count))

    if len(rects) >= 3 and min(positive, negative, zero) == 0:
        return None

    legend_col = np.array(legend, dtype=int).reshape(-1, 1)
    if legend_on_left:
        grid = np.concatenate([legend_col, content], axis=1)
    else:
        grid = np.concatenate([content, legend_col], axis=1)

    solved = main(grid)
    if np.array_equal(solved, grid):
        return None

    # Reject degenerate cases where every region lands on the same target color.
    if len(set(map(int, solved[:, 1:].flatten() if legend_on_left else solved[:, :-1].flatten()))) < 2:
        return None
    return grid


def _generate_attempt() -> np.ndarray | None:
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    legend = random.sample(LEGEND_PALETTE, legend_len)
    content_w = random.randint(*CONTENT_WIDTH_RANGE)
    rects = _build_rectangles(legend_len, content_w)
    region_colors = _assign_region_colors(rects, legend)
    if region_colors is None:
        return None
    legend_on_left = random.random() < 0.5
    return _paint_regions(legend, rects, region_colors, legend_on_left)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-dial ARC input")
