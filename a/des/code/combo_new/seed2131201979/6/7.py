from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

COLOR_ORDER = (RED, ORANGE, YELLOW, GREEN, BLUE, MAGENTA, MAROON)
COLOR_RANK = {color: idx for idx, color in enumerate(COLOR_ORDER)}
MARKER_COLORS = (RED, YELLOW, GREEN, BLUE, MAGENTA, MAROON)
TICK_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, 8, MAROON)

_RNG = np.random.default_rng()
_LAYOUTS = tuple((rows, cols) for rows in range(1, 4) for cols in range(1, 4) if 2 <= rows * cols <= 6)
_REGION_SIZE_RANGE = (4, 9)
_MAX_SHAPE_CELLS = 7
_GEN_RETRIES = 300


def _partition_spans(ticks: list[int], limit: int) -> list[tuple[int, int]]:
    ticks = sorted(set(int(t) for t in ticks))
    starts = [1] + ticks
    ends = ticks + [int(limit - 1)]
    spans = []
    for start, end in zip(starts, ends, strict=False):
        assert 1 <= start < end <= limit - 1, "tick positions must define nonempty interior regions"
        spans.append((int(start), int(end)))
    return spans


def _extract_regions(grid: np.ndarray) -> tuple[list[np.ndarray], list[int], int, int]:
    top_ticks = [c for c in range(1, grid.shape[1] - 1) if int(grid[0, c]) not in (BLACK, GREY)]
    left_ticks = [r for r in range(1, grid.shape[0] - 1) if int(grid[r, 0]) not in (BLACK, GREY)]

    row_spans = _partition_spans(left_ticks, int(grid.shape[0]))
    col_spans = _partition_spans(top_ticks, int(grid.shape[1]))

    regions: list[np.ndarray] = []
    markers: list[int] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = np.array(grid[r0:r1, c0:c1], copy=True, dtype=int)
            marker_pixels = region[(region != BLACK) & (region != ORANGE)]
            unique_markers = np.unique(marker_pixels)
            assert marker_pixels.size == 1 and unique_markers.size == 1, "each region must contain exactly one non-orange marker pixel"
            marker = int(unique_markers[0])
            assert marker in COLOR_RANK, "marker color must belong to the visible-spectrum ordering"
            regions.append(region)
            markers.append(marker)

    assert len(markers) == len(set(markers)), "region markers must have unique colors"
    return regions, markers, len(row_spans), len(col_spans)


def _pack_regions(regions: list[np.ndarray], markers: list[int], rows: int, cols: int) -> np.ndarray:
    assert len(regions) == rows * cols
    order = sorted(range(len(regions)), key=lambda idx: COLOR_RANK[int(markers[idx])])
    max_h = max(int(region.shape[0]) for region in regions)
    max_w = max(int(region.shape[1]) for region in regions)

    out_h = rows * max_h + (rows - 1)
    out_w = cols * max_w + (cols - 1)
    out = np.full((out_h, out_w), BLACK, dtype=int)

    for r in range(1, rows):
        out[r * (max_h + 1) - 1, :] = GREY
    for c in range(1, cols):
        out[:, c * (max_w + 1) - 1] = GREY

    for slot, idx in enumerate(order):
        block_r = slot // cols
        block_c = slot % cols
        r0 = block_r * (max_h + 1)
        c0 = block_c * (max_w + 1)
        region = regions[idx]
        h, w = region.shape
        out[r0 : r0 + h, c0 : c0 + w] = region

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden partition from border ticks and reorder the partition cells by marker color.

    Input structure:
    - The background is black, and the whole grid is enclosed by a 1-pixel grey frame.
    - The top border contains colored single-pixel ticks whose column positions mark vertical partition starts.
    - The left border contains colored single-pixel ticks whose row positions mark horizontal partition starts.
    - Each implied rectangular region inside the frame contains an orange shape and exactly one non-orange marker pixel.

    Rule:
    - Read the top and left tick positions as the hidden cut layout that partitions the interior into rectangular regions.
    - Extract every region as its full rectangle, keeping both the orange shape and the single marker pixel.
    - Sort the regions by marker color using the fixed visible order red, orange, yellow, green, blue, magenta, maroon.
    - Rebuild the output as a compact grid with the same number of partition rows and columns, placing the sorted regions in row-major order.
    - Pad smaller regions with black inside max-sized slots and separate adjacent slots with 1-pixel grey lines.

    Key insight:
    - The colored border ticks are not decoration: they implicitly define the region boundaries, and the marker color inside each region determines the new placement order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 4 and g.shape[1] >= 4, "grid must contain a framed interior"

    regions, markers, rows, cols = _extract_regions(g)
    return _pack_regions(regions, markers, rows, cols)


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[int(_RNG.integers(0, len(seq)))]


def _sample_without_replacement(seq, k: int) -> list[int]:
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [int(seq[int(i)]) for i in idx]


def _grow_blob(height: int, width: int, size: int) -> set[tuple[int, int]]:
    cells = {(int(_RNG.integers(0, height)), int(_RNG.integers(0, width)))}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(frontier[int(_RNG.integers(0, len(frontier)))])
    return cells


def _shape_rect(height: int, width: int, max_cells: int) -> set[tuple[int, int]] | None:
    candidates = [
        (h, w)
        for h in range(1, height + 1)
        for w in range(1, width + 1)
        if 2 <= h * w <= max_cells
    ]
    if not candidates:
        return None
    h, w = _choice(candidates)
    r0 = _randint(0, height - h)
    c0 = _randint(0, width - w)
    return {(r0 + r, c0 + c) for r in range(h) for c in range(w)}


def _shape_line(height: int, width: int, max_cells: int) -> set[tuple[int, int]] | None:
    options = []
    if width >= 2:
        for length in range(2, min(width, max_cells) + 1):
            options.append(("h", length))
    if height >= 2:
        for length in range(2, min(height, max_cells) + 1):
            options.append(("v", length))
    if not options:
        return None
    orient, length = _choice(options)
    if orient == "h":
        r = _randint(0, height - 1)
        c0 = _randint(0, width - length)
        return {(r, c0 + c) for c in range(length)}
    c = _randint(0, width - 1)
    r0 = _randint(0, height - length)
    return {(r0 + r, c) for r in range(length)}


def _shape_l(height: int, width: int, max_cells: int) -> set[tuple[int, int]] | None:
    options = []
    for arm_h in range(2, height + 1):
        for arm_w in range(2, width + 1):
            cells = arm_h + arm_w - 1
            if cells <= max_cells:
                options.append((arm_h, arm_w))
    if not options:
        return None
    arm_h, arm_w = _choice(options)
    base = {(r, 0) for r in range(arm_h)} | {(arm_h - 1, c) for c in range(arm_w)}
    variants = []
    shape = set(base)
    h0, w0 = arm_h, arm_w
    for _ in range(4):
        if h0 <= height and w0 <= width:
            variants.append((set(shape), h0, w0))
        shape = {(c, h0 - 1 - r) for r, c in shape}
        h0, w0 = w0, h0
    if not variants:
        return None
    shape, h0, w0 = _choice(variants)
    rr = _randint(0, height - h0)
    cc = _randint(0, width - w0)
    return {(rr + r, cc + c) for r, c in shape}


def _sample_orange_mask(height: int, width: int) -> np.ndarray:
    inner_h = height - 2
    inner_w = width - 2
    max_cells = min(_MAX_SHAPE_CELLS, inner_h * inner_w - 1)
    assert max_cells >= 2

    builders = (_shape_rect, _shape_line, _shape_l)
    mask = None
    if _RNG.random() < 0.35:
        size = _randint(2, max_cells)
        cells = _grow_blob(inner_h, inner_w, size)
        if len(cells) >= 2:
            mask = np.zeros((inner_h, inner_w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
    if mask is None:
        for _ in range(6):
            builder = _choice(builders)
            cells = builder(inner_h, inner_w, max_cells)
            if cells is None:
                continue
            mask = np.zeros((inner_h, inner_w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            if int(mask.sum()) >= 2:
                break
    if mask is None or int(mask.sum()) < 2:
        size = _randint(2, max_cells)
        cells = _grow_blob(inner_h, inner_w, size)
        mask = np.zeros((inner_h, inner_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

    full = np.zeros((height, width), dtype=bool)
    full[1:-1, 1:-1] = mask
    return full


def _make_region(height: int, width: int, marker_color: int) -> np.ndarray:
    region = np.full((height, width), BLACK, dtype=int)
    orange_mask = _sample_orange_mask(height, width)
    region[orange_mask] = ORANGE

    marker_candidates = np.argwhere(region[1:-1, 1:-1] == BLACK)
    assert marker_candidates.size > 0
    mr, mc = marker_candidates[int(_RNG.integers(0, len(marker_candidates)))]
    region[int(mr) + 1, int(mc) + 1] = int(marker_color)
    return region


def _build_expected(regions: list[np.ndarray], markers: list[int], rows: int, cols: int) -> np.ndarray:
    return _pack_regions(regions, markers, rows, cols)


def _gen_attempt() -> np.ndarray | None:
    rows, cols = _choice(_LAYOUTS)
    region_count = rows * cols

    heights = [_randint(*_REGION_SIZE_RANGE) for _ in range(rows)]
    widths = [_randint(*_REGION_SIZE_RANGE) for _ in range(cols)]
    grid_h = sum(heights) + 2
    grid_w = sum(widths) + 2
    if max(grid_h, grid_w) > 30:
        return None

    max_h = max(heights)
    max_w = max(widths)
    out_h = rows * max_h + (rows - 1)
    out_w = cols * max_w + (cols - 1)
    if max(out_h, out_w) > 30:
        return None

    marker_colors = _sample_without_replacement(MARKER_COLORS, region_count)
    _RNG.shuffle(marker_colors)
    if all(COLOR_RANK[int(marker_colors[i])] <= COLOR_RANK[int(marker_colors[i + 1])] for i in range(region_count - 1)):
        return None

    regions: list[np.ndarray] = []
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    top_ticks = []
    c = 1
    for width in widths[:-1]:
        c += width
        top_ticks.append(c)
    left_ticks = []
    r = 1
    for height in heights[:-1]:
        r += height
        left_ticks.append(r)

    if top_ticks:
        top_colors = _RNG.choice(TICK_COLORS, size=len(top_ticks), replace=True)
        for tick, color in zip(top_ticks, top_colors, strict=False):
            grid[0, int(tick)] = int(color)
    if left_ticks:
        left_colors = _RNG.choice(TICK_COLORS, size=len(left_ticks), replace=True)
        for tick, color in zip(left_ticks, left_colors, strict=False):
            grid[int(tick), 0] = int(color)

    idx = 0
    r0 = 1
    for height in heights:
        c0 = 1
        for width in widths:
            region = _make_region(int(height), int(width), int(marker_colors[idx]))
            grid[r0 : r0 + height, c0 : c0 + width] = region
            regions.append(region)
            idx += 1
            c0 += width
        r0 += height

    expected = _build_expected(regions, marker_colors, rows, cols)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
