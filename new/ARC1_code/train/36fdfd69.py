import numpy as np

from arc_utils import Color
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the non-red filler pixels inside solid foreground rectangles bounded by red markers to yellow.

    Input structure
    - Black is the empty background.
    - Red pixels mark one or more solid rectangular foreground regions.
    - Each marked region is otherwise filled with the single non-red foreground color, and yellow is absent from the input.

    Rule
    - Find every non-black rectangle, at least 2 by 2, whose red pixels touch its top, bottom, left, and right sides.
    - Recolor the non-red pixels inside those rectangles to yellow.
    - Preserve black, red, and all foreground pixels outside such rectangles.

    Key insight
    - The target cells are not a connected object; they are the foreground cells inside red-tight rectangular extents.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = Color.BLACK
    marker = Color.RED
    fill = Color.YELLOW

    colors = set(map(int, np.unique(g)))
    assert bg in colors, "expected black background"
    assert marker in colors, "expected red marker pixels"
    assert fill not in colors, "yellow is reserved for the output fill"
    assert len(colors - {bg, marker}) == 1, "expected exactly one non-red foreground color"

    foreground = g != bg
    red = g == marker
    height, width = map(int, g.shape)

    out = g.copy()
    found_region = False
    for r0 in range(height - 1):
        for r1 in range(r0 + 1, height):
            solid_cols = np.all(foreground[r0 : r1 + 1, :], axis=0)
            c = 0
            while c < width:
                if not solid_cols[c]:
                    c += 1
                    continue

                run_start = c
                while c < width and solid_cols[c]:
                    c += 1
                run_end = c - 1

                for c0 in range(run_start, run_end):
                    for c1 in range(c0 + 1, run_end + 1):
                        red_region = red[r0 : r1 + 1, c0 : c1 + 1]
                        red_count = int(np.count_nonzero(red_region))
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        if red_count < 2 or red_count == area:
                            continue
                        if not (
                            np.any(red_region[0, :])
                            and np.any(red_region[-1, :])
                            and np.any(red_region[:, 0])
                            and np.any(red_region[:, -1])
                        ):
                            continue

                        region = g[r0 : r1 + 1, c0 : c1 + 1]
                        out_region = out[r0 : r1 + 1, c0 : c1 + 1]
                        out_region[(region != bg) & (region != marker)] = fill
                        found_region = True

    assert found_region, "expected at least one red-tight foreground rectangle"
    assert np.any(out != g), "expected at least one recolored filler pixel"
    return out

# --------------- generator (o5x) ---------------

# Palette choices for the single non-red foreground color.
_GEN_FG_COLORS = [
    Color.BLUE, Color.GREEN, Color.GREY, Color.PINK,
    Color.ORANGE, Color.TEAL, Color.MAROON,
]

# Canvas, rectangle count, and rectangle geometry.
_GEN_CANVAS_MIN, _GEN_CANVAS_MAX = 14, 18
_GEN_RECT_COUNT_MIN, _GEN_RECT_COUNT_MAX = 2, 4
_GEN_RECT_HEIGHTS = (2, 3)
_GEN_RECT_WIDTHS = (2, 3, 4, 5, 6, 7)
_GEN_RECT_BORDER_MARGIN = 1
_GEN_RECT_GAP = 1

# Red marker density inside each target rectangle.
_GEN_RED_MIN_CELLS = 2
_GEN_RED_MIN_FRACTION, _GEN_RED_MAX_FRACTION = 0.38, 0.88

# Foreground density and random blob sizes.
_GEN_FG_MIN_DENSITY, _GEN_FG_MAX_DENSITY = 0.40, 0.48
_GEN_CLUSTER_SMALL_PROB = 0.76
_GEN_CLUSTER_MEDIUM_CONDITIONAL_PROB = 0.90
_GEN_CLUSTER_SMALL_MAX, _GEN_CLUSTER_MEDIUM_MAX, _GEN_CLUSTER_LARGE_MAX = 8, 25, 50
_GEN_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Connector and retry limits.
_GEN_CONNECTOR_MAX_LEN, _GEN_RETRIES = 2, 300


def _gen_place_boxes(height, width, count):
    boxes = []
    blocked = np.zeros((height, width), dtype=bool)
    for _ in range(count):
        by_size = {}
        for h in _GEN_RECT_HEIGHTS:
            for w in _GEN_RECT_WIDTHS:
                locs = []
                for r0 in range(_GEN_RECT_BORDER_MARGIN, height - h):
                    for c0 in range(_GEN_RECT_BORDER_MARGIN, width - w):
                        rr0 = max(0, r0 - _GEN_RECT_GAP)
                        cc0 = max(0, c0 - _GEN_RECT_GAP)
                        rr1 = min(height, r0 + h + _GEN_RECT_GAP)
                        cc1 = min(width, c0 + w + _GEN_RECT_GAP)
                        if not blocked[rr0:rr1, cc0:cc1].any():
                            locs.append((r0, c0))
                if locs:
                    by_size[(h, w)] = locs
        if not by_size:
            return None

        h, w = random.choice(list(by_size))
        r0, c0 = random.choice(by_size[(h, w)])
        r1, c1 = r0 + h - 1, c0 + w - 1
        boxes.append((r0, c0, r1, c1))
        blocked[
            max(0, r0 - _GEN_RECT_GAP) : min(height, r1 + _GEN_RECT_GAP + 1),
            max(0, c0 - _GEN_RECT_GAP) : min(width, c1 + _GEN_RECT_GAP + 1),
        ] = True
    return boxes


def _gen_red_mask(h, w):
    area = h * w
    low = max(_GEN_RED_MIN_CELLS, math.ceil(_GEN_RED_MIN_FRACTION * area))
    high = min(area - 1, math.floor(_GEN_RED_MAX_FRACTION * area))
    if low > high:
        return None

    target = random.randint(low, high)
    cells = set()
    if target <= 3:
        cells.update(random.choice([[(0, 0), (h - 1, w - 1)], [(0, w - 1), (h - 1, 0)]]))
    else:
        cells.add((0, random.randrange(w)))
        cells.add((h - 1, random.randrange(w)))
        cells.add((random.randrange(h), 0))
        cells.add((random.randrange(h), w - 1))

    remaining = [(r, c) for r in range(h) for c in range(w) if (r, c) not in cells]
    random.shuffle(remaining)
    for cell in remaining:
        if len(cells) >= target:
            break
        cells.add(cell)

    mask = np.zeros((h, w), dtype=bool)
    rows, cols = zip(*cells)
    mask[rows, cols] = True
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return None
    if mask.all():
        return None
    return mask

def _gen_connector_cells(box, height, width):
    r0, c0, r1, c1 = box
    candidates = []
    if r0 > 0:
        c = random.randint(c0, c1)
        cells = [(r0 - 1, c)]
        if r0 >= _GEN_CONNECTOR_MAX_LEN:
            cells.append((r0 - 2, c))
        candidates.append(cells)
    if r1 + 1 < height:
        c = random.randint(c0, c1)
        cells = [(r1 + 1, c)]
        if r1 + _GEN_CONNECTOR_MAX_LEN < height:
            cells.append((r1 + 2, c))
        candidates.append(cells)
    if c0 > 0:
        r = random.randint(r0, r1)
        cells = [(r, c0 - 1)]
        if c0 >= _GEN_CONNECTOR_MAX_LEN:
            cells.append((r, c0 - 2))
        candidates.append(cells)
    if c1 + 1 < width:
        r = random.randint(r0, r1)
        cells = [(r, c1 + 1)]
        if c1 + _GEN_CONNECTOR_MAX_LEN < width:
            cells.append((r, c1 + 2))
        candidates.append(cells)
    random.shuffle(candidates)
    return candidates

def _gen_cluster_size(needed):
    if random.random() < _GEN_CLUSTER_SMALL_PROB:
        max_size = _GEN_CLUSTER_SMALL_MAX
    elif random.random() < _GEN_CLUSTER_MEDIUM_CONDITIONAL_PROB:
        max_size = _GEN_CLUSTER_MEDIUM_MAX
    else:
        max_size = _GEN_CLUSTER_LARGE_MAX
    return random.randint(1, min(max_size, needed))

def _gen_fill_scatter(out, fg, needed):
    if needed <= 0:
        return True

    available = {tuple(p) for p in np.argwhere(out == Color.BLACK)}
    while needed > 0 and available:
        cluster_size = _gen_cluster_size(needed)
        seed = random.choice(tuple(available))
        available.remove(seed)
        cluster = {seed}
        frontier = [seed]
        while len(cluster) < cluster_size and frontier:
            base = random.choice(frontier)
            r, c = base
            nbrs = [
                (r + dr, c + dc)
                for dr, dc in _GEN_NEIGHBORS_4
                if (r + dr, c + dc) in available
            ]
            if not nbrs:
                frontier.remove(base)
                continue
            nxt = random.choice(nbrs)
            available.remove(nxt)
            cluster.add(nxt)
            frontier.append(nxt)

        rows, cols = zip(*cluster)
        out[rows, cols] = fg
        needed -= len(cluster)

    return needed == 0

def _gen_render_targets(out, boxes):
    """Paint intended output rectangles; return False if any red mask is invalid."""
    for r0, c0, r1, c1 in boxes:
        red_mask = _gen_red_mask(r1 - r0 + 1, c1 - c0 + 1)
        if red_mask is None:
            return False
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[:, :] = Color.YELLOW
        region[red_mask] = Color.RED
    return True

def _gen_attach_connectors(out, fg, boxes):
    """Add at least one adjacent foreground path for every target rectangle."""
    height, width = out.shape
    for box in boxes:
        placed = False
        for cells in _gen_connector_cells(box, height, width):
            if all(out[r, c] in (Color.BLACK, fg) for r, c in cells):
                rows, cols = zip(*cells)
                out[rows, cols] = fg
                placed = True
                break
        if not placed:
            return False
    return True

def _gen_fill_to_density(out, fg):
    """Scatter foreground until the output reaches the configured density range."""
    total = out.size
    fg_low = math.ceil(_GEN_FG_MIN_DENSITY * total)
    fg_high = math.floor(_GEN_FG_MAX_DENSITY * total)
    current_fg = int(np.count_nonzero((out == Color.YELLOW) | (out == fg)))
    if current_fg > fg_high:
        return False

    desired_fg = random.randint(max(fg_low, current_fg), fg_high)
    return _gen_fill_scatter(out, fg, desired_fg - current_fg)

def _gen_solver_matches(inp, out):
    try:
        solved = main(inp)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(solved, out)

def _gen_attempt():
    height = random.randint(_GEN_CANVAS_MIN, _GEN_CANVAS_MAX)
    width = random.randint(_GEN_CANVAS_MIN, _GEN_CANVAS_MAX)
    fg = random.choice(_GEN_FG_COLORS)
    rect_count = random.randint(_GEN_RECT_COUNT_MIN, _GEN_RECT_COUNT_MAX)

    boxes = _gen_place_boxes(height, width, rect_count)
    if boxes is None:
        return None

    out = np.full((height, width), Color.BLACK, dtype=int)
    if not _gen_render_targets(out, boxes):
        return None
    if not _gen_attach_connectors(out, fg, boxes):
        return None
    if not _gen_fill_to_density(out, fg):
        return None

    inp = out.copy()
    inp[inp == Color.YELLOW] = fg
    if not _gen_solver_matches(inp, out):
        return None
    return inp

def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 36fdfd69 input")
