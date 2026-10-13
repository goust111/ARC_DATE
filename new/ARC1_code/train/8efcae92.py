import numpy as np

from arc_utils import crop_bbox, find_components
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Crop out the foreground rectangle that contains the most marker pixels.

    Input structure
    - The upper-left cell is background.
    - The foreground consists of multiple 4-connected solid rectangles made from
      a common fill color, with a rarer marker color replacing some cells.

    Rule
    - Treat the rarer of the two foreground colors as the marker color.
    - Count marker pixels inside each foreground rectangle.
    - Return the exact bounding-box crop of the unique rectangle with the largest
      marker count.

    Key insight
    - The output is not redrawn or recolored; it is one existing input rectangle
      selected by the count of its marker pixels.
    """
    g = np.asarray(grid, dtype=int)

    bg = int(g[0, 0])
    fg_pixels = g[g != bg]
    colors, counts = np.unique(fg_pixels, return_counts=True)
    assert len(colors) == 2

    min_count = int(counts.min())
    marker_candidates = colors[counts == min_count]
    assert len(marker_candidates) == 1
    marker = int(marker_candidates[0])
    foreground_colors = set(map(int, colors))

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) >= 2

    scored = []
    for comp in components:
        box = comp["bbox"]
        crop = crop_bbox(g, box)
        assert np.all(crop != bg)
        assert set(map(int, np.unique(crop))).issubset(foreground_colors)
        marker_count = int(np.count_nonzero(crop == marker))
        assert marker_count > 0
        scored.append((marker_count, box))

    best_count = max(marker_count for marker_count, _box in scored)
    winners = [box for marker_count, box in scored if marker_count == best_count]
    assert len(winners) == 1
    return crop_bbox(g, winners[0])

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_SIZE = 20
BG_COLOR, FILL_COLOR, MARKER_COLOR = 0, 1, 2

# Rectangle layout and sizing.
RECT_COUNT_RANGE = (3, 4)
RECT_SIDE_RANGE = (3, 18)
RECT_GAP, GRID_LAYOUT_PARTS = 2, 2
TOTAL_RECT_AREA_RANGE = (120, 240)
LAYOUTS = ("vertical_stack", "horizontal_stack", "grid")

# Marker count constraints.
MARKER_FRACTION_RANGE = (0.05, 0.25)
MIN_MARKERS_PER_RECT = 2
TOTAL_MARKER_RANGE = (13, 25)
WINNER_MARKER_MARGIN = 2

# Rejection sampling limit.
MAX_GENERATION_ATTEMPTS = 300


def _gen_split_bands(total, parts):
    min_band, max_gap = RECT_SIDE_RANGE[0], RECT_GAP
    spare = total - max_gap * (parts - 1) - min_band * parts
    if spare < 0:
        return None

    slots = [0] * (parts + 2)
    for _ in range(spare):
        slots[random.randrange(len(slots))] += 1

    pos = slots[0]
    bands = []
    for i in range(parts):
        size = min_band + slots[i + 1]
        bands.append((pos, pos + size))
        pos += size + max_gap
    return bands


def _gen_rect_in_cell(r0, r1, c0, c1):
    cell_h = r1 - r0
    cell_w = c1 - c0
    min_side, max_side = RECT_SIDE_RANGE
    if cell_h < min_side or cell_w < min_side:
        return None

    h = random.randint(min_side, min(max_side, cell_h))
    w = random.randint(min_side, min(max_side, cell_w))
    starts = [
        (r, c)
        for r in range(r0, r1 - h + 1)
        for c in range(c0, c1 - w + 1)
        if not (r == 0 and c == 0)
    ]
    if not starts:
        return None
    sr, sc = random.choice(starts)
    return sr, sc, sr + h - 1, sc + w - 1


def _gen_stack_cells(axis, n_rects):
    bands = _gen_split_bands(GRID_SIZE, n_rects)
    if bands is None:
        return None
    if axis == "row":
        return [(r0, r1, 0, GRID_SIZE) for r0, r1 in bands]
    return [(0, GRID_SIZE, c0, c1) for c0, c1 in bands]


def _gen_grid_cells(n_rects):
    row_bands = _gen_split_bands(GRID_SIZE, GRID_LAYOUT_PARTS)
    col_bands = _gen_split_bands(GRID_SIZE, GRID_LAYOUT_PARTS)
    if row_bands is None or col_bands is None:
        return None
    all_cells = [
        (row_bands[ri][0], row_bands[ri][1], col_bands[ci][0], col_bands[ci][1])
        for ri in range(GRID_LAYOUT_PARTS)
        for ci in range(GRID_LAYOUT_PARTS)
    ]
    return random.sample(all_cells, n_rects)


def _gen_boxes(n_rects):
    layout = random.choice(LAYOUTS)
    if layout == "vertical_stack":
        cells = _gen_stack_cells("row", n_rects)
    elif layout == "horizontal_stack":
        cells = _gen_stack_cells("col", n_rects)
    else:
        cells = _gen_grid_cells(n_rects)
    if cells is None:
        return None

    boxes = []
    for cell in cells:
        box = _gen_rect_in_cell(*cell)
        if box is None:
            return None
        boxes.append(box)
    return boxes


def _gen_marker_ranges(areas):
    ranges = []
    min_frac, max_frac = MARKER_FRACTION_RANGE
    for area in areas:
        lo = max(MIN_MARKERS_PER_RECT, math.ceil(area * min_frac))
        hi = math.floor(area * max_frac)
        if lo > hi:
            return None
        ranges.append((lo, hi))
    return ranges


def _gen_allocate_counts(ranges):
    total_lo = sum(lo for lo, _hi in ranges)
    total_hi = sum(hi for _lo, hi in ranges)
    target_lo = max(TOTAL_MARKER_RANGE[0], total_lo)
    target_hi = min(TOTAL_MARKER_RANGE[1], total_hi)
    if target_lo > target_hi:
        target_lo, target_hi = total_lo, total_hi

    total = random.randint(target_lo, target_hi)
    possible = []
    for winner in range(len(ranges)):
        win_lo, win_hi = ranges[winner]
        other_los = [lo for i, (lo, _hi) in enumerate(ranges) if i != winner]
        min_runner = max(other_los)
        for win_count in range(max(win_lo, min_runner + WINNER_MARKER_MARGIN), win_hi + 1):
            rem = total - win_count
            other_ranges = [
                (lo, min(hi, win_count - WINNER_MARKER_MARGIN))
                for i, (lo, hi) in enumerate(ranges)
                if i != winner
            ]
            if all(lo <= hi for lo, hi in other_ranges):
                rem_lo = sum(lo for lo, _hi in other_ranges)
                rem_hi = sum(hi for _lo, hi in other_ranges)
                if rem_lo <= rem <= rem_hi:
                    possible.append((winner, win_count, other_ranges, rem))

    if not possible:
        return None, None

    winner, win_count, other_ranges, rem = random.choice(possible)
    other_counts = []
    for idx, (lo, hi) in enumerate(other_ranges):
        rest = other_ranges[idx + 1 :]
        rest_lo = sum(x for x, _y in rest)
        rest_hi = sum(y for _x, y in rest)
        pick_lo = max(lo, rem - rest_hi)
        pick_hi = min(hi, rem - rest_lo)
        count = random.randint(pick_lo, pick_hi)
        other_counts.append(count)
        rem -= count

    counts = []
    other_iter = iter(other_counts)
    for i in range(len(ranges)):
        counts.append(win_count if i == winner else next(other_iter))
    return counts, winner


def _gen_scatter_patch(h, w, marker_count):
    patch = np.full((h, w), FILL_COLOR, dtype=int)
    marker_positions = random.sample(range(h * w), marker_count)
    patch.reshape(-1)[marker_positions] = MARKER_COLOR
    return patch


def _render_scene(boxes, counts, winner):
    """Render all rectangles and return the winner crop expected by main()."""
    grid = np.full((GRID_SIZE, GRID_SIZE), BG_COLOR, dtype=int)
    expected = None
    for i, ((r0, c0, r1, c1), marker_count) in enumerate(zip(boxes, counts)):
        patch = _gen_scatter_patch(r1 - r0 + 1, c1 - c0 + 1, marker_count)
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
        if i == winner:
            expected = patch.copy()
    return grid, expected


def _passes_oracle(grid, expected):
    if grid[0, 0] != BG_COLOR:
        return False
    if np.count_nonzero(grid == MARKER_COLOR) >= np.count_nonzero(grid == FILL_COLOR):
        return False
    try:
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _gen_attempt():
    n_rects = random.randint(*RECT_COUNT_RANGE)
    boxes = _gen_boxes(n_rects)
    if boxes is None:
        return None

    areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in boxes]
    total_area = sum(areas)
    if total_area < TOTAL_RECT_AREA_RANGE[0] or total_area > TOTAL_RECT_AREA_RANGE[1]:
        return None

    ranges = _gen_marker_ranges(areas)
    if ranges is None:
        return None

    counts, winner = _gen_allocate_counts(ranges)
    if counts is None:
        return None

    grid, expected = _render_scene(boxes, counts, winner)
    return grid if _passes_oracle(grid, expected) else None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 8efcae92 input")
