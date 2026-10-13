import numpy as np

from arc_utils import colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten the single foreground mask into a centered horizontal segment of the same color and length.

    Input structure:
    - The grid has one dominant background color.
    - All non-background pixels share one foreground color.
    - The number of foreground pixels fits in one centered row of the grid.

    Rule:
    - Erase the original foreground mask.
    - Count its pixels, keep its color, and paint that many pixels as a horizontal segment.
    - Place the segment on the middle row with equal background margin on the left and right.

    Key insight:
    - The original foreground shape is irrelevant except for its color and area; the output is the area normalized into the grid's central horizontal line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    color = fg_colors[0]
    count = int(np.count_nonzero(g == color))
    assert 0 < count <= width, "foreground pixels must fit in one row"
    assert height % 2 == 1, "expected a unique middle row"
    assert (width - count) % 2 == 0, "foreground segment must center with equal side margins"

    out = np.full(g.shape, bg, dtype=int)
    row = height // 2
    start = (width - count) // 2
    out[row, start : start + count] = color
    return out

# --------------- generator (o5x) ---------------
# Canvas, palette, and foreground-count controls.
HEIGHT_MIN, HEIGHT_MAX, HEIGHT_STEP = 5, 11, 2
WIDTH_MIN, WIDTH_MAX = 10, 28
BACKGROUND_COLOR = 8
FOREGROUND_COLORS = tuple(c for c in range(10) if c != BACKGROUND_COLOR)
COUNT_MIN, COUNT_MAX = 6, 15
# Sparse bbox shape, blob growth, and retry controls.
BBOX_SPAN_MIN, BBOX_SPAN_MAX = 3, 8
BBOX_FILL_MIN, BBOX_FILL_MAX = 0.29, 0.67
COVER_PATTERN_COUNT = 3
ADJACENT_GROW_PROB = 0.85
GENERATION_ATTEMPTS = 300


def _gen_covering_cells(box_h, box_w):
    cells = set()
    mode = random.randrange(COVER_PATTERN_COUNT)

    if mode == 0:
        length = max(box_h, box_w)
        flip_r = random.choice([False, True])
        flip_c = random.choice([False, True])
        for i in range(length):
            r = round(i * (box_h - 1) / (length - 1))
            c = round(i * (box_w - 1) / (length - 1))
            if flip_r:
                r = box_h - 1 - r
            if flip_c:
                c = box_w - 1 - c
            cells.add((r, c))
    elif box_h >= box_w:
        cols = list(range(box_w)) + [random.randrange(box_w) for _ in range(box_h - box_w)]
        random.shuffle(cols)
        cells.update((r, cols[r]) for r in range(box_h))
    else:
        rows = list(range(box_h)) + [random.randrange(box_h) for _ in range(box_w - box_h)]
        random.shuffle(rows)
        cells.update((rows[c], c) for c in range(box_w))

    return cells


def _gen_adjacent_empty(cells, all_cells):
    out = []
    for r, c in all_cells:
        if (r, c) in cells:
            continue
        for rr, cc in cells:
            if max(abs(r - rr), abs(c - cc)) == 1:
                out.append((r, c))
                break
    return out


def _gen_blob_cells(box_h, box_w, count):
    cells = _gen_covering_cells(box_h, box_w)
    all_cells = [(r, c) for r in range(box_h) for c in range(box_w)]

    while len(cells) < count:
        empty = [p for p in all_cells if p not in cells]
        frontier = _gen_adjacent_empty(cells, all_cells)
        pool = frontier if frontier and random.random() < ADJACENT_GROW_PROB else empty
        cells.add(random.choice(pool))

    return cells


def _gen_bbox_shape(height, width, count):
    max_h = min(BBOX_SPAN_MAX, height)
    max_w = min(BBOX_SPAN_MAX, width)
    shapes = []
    for box_h in range(BBOX_SPAN_MIN, max_h + 1):
        for box_w in range(BBOX_SPAN_MIN, max_w + 1):
            area = box_h * box_w
            fill = count / area
            if max(box_h, box_w) < count and BBOX_FILL_MIN <= fill <= BBOX_FILL_MAX:
                shapes.append((box_h, box_w))
    if not shapes:
        for box_h in range(BBOX_SPAN_MIN, max_h + 1):
            for box_w in range(BBOX_SPAN_MIN, max_w + 1):
                if max(box_h, box_w) < count and box_h * box_w > count:
                    shapes.append((box_h, box_w))
    return random.choice(shapes)


def _draw_centered_segment(height, width, count, bg, color):
    grid = np.full((height, width), bg, dtype=int)
    row = height // 2
    start = (width - count) // 2
    grid[row, start : start + count] = color
    return grid


def _draw_scattered_blob(height, width, count, bg, color):
    grid = np.full((height, width), bg, dtype=int)
    box_h, box_w = _gen_bbox_shape(height, width, count)
    r0 = random.randint(0, height - box_h)
    c0 = random.randint(0, width - box_w)
    for rr, cc in _gen_blob_cells(box_h, box_w, count):
        grid[r0 + rr, c0 + cc] = color
    return grid


def _gen_attempt():
    height = random.randrange(HEIGHT_MIN, HEIGHT_MAX + 1, HEIGHT_STEP)
    width = random.randint(WIDTH_MIN, WIDTH_MAX)
    count_max = min(COUNT_MAX, width)
    counts = [n for n in range(COUNT_MIN, count_max + 1) if (width - n) % 2 == 0]
    if not counts:
        return None

    count = random.choice(counts)
    bg = BACKGROUND_COLOR
    color = random.choice(FOREGROUND_COLORS)

    output = _draw_centered_segment(height, width, count, bg, color)
    inp = _draw_scattered_blob(height, width, count, bg, color)
    if np.array_equal(main(inp), output):
        return inp
    return None


def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dce56571 input")
