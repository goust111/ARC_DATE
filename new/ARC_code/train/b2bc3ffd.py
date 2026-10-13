import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Move every object above the solid bottom stripe upward by its own pixel count.

    Input structure
    The grid has a dominant background, exactly one full-width non-background row
    that acts as a fixed stripe, and one or more 4-connected monochrome objects
    above that stripe.

    Rule
    Clear each object from its original cells, then translate it straight upward
    by the number of pixels in that object, preserving its shape and color. Leave
    the full-width stripe unchanged.

    Key insight
    Each object's area is also its vertical displacement; the full-width row is a
    fixed baseline, not a moving object.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    stripe_rows = [
        r
        for r in range(h)
        if np.unique(g[r]).size == 1 and int(g[r, 0]) != bg
    ]
    assert len(stripe_rows) == 1
    stripe_row = stripe_rows[0]
    assert stripe_row == h - 1

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    stripe_components = []
    moving_objects = []
    for component in components:
        coords = component["coords"]
        is_stripe = int(component["area"]) == w and np.all(coords[:, 0] == stripe_row)
        if is_stripe:
            stripe_components.append(component)
        else:
            assert np.all(coords[:, 0] < stripe_row)
            moving_objects.append(component)

    assert len(stripe_components) == 1
    assert moving_objects

    out = g.copy()
    for component in moving_objects:
        coords = component["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    occupied = np.zeros(g.shape, dtype=bool)
    occupied[stripe_row, :] = True

    for component in moving_objects:
        coords = component["coords"]
        target = coords.copy()
        target[:, 0] -= int(component["area"])

        assert np.all(target[:, 0] >= 0)
        assert not np.any(occupied[target[:, 0], target[:, 1]])

        occupied[target[:, 0], target[:, 1]] = True
        out[target[:, 0], target[:, 1]] = int(component["color"])

    return out

# --------------- generator (att2) ---------------

# CFG: canvas/colors, counts/spacing, shape/motion, biased sampling, retries.
_GEN_GRID_SIZE = 8
_GEN_BG = 7
_GEN_STRIPE = 8
_GEN_COLORS = [1, 2, 3, 4, 5, 6, 9]

_GEN_OBJECT_COUNTS = [2, 3, 4]
_GEN_MIN_INTER_OBJECT_GAP = 1

_GEN_MAX_HEIGHT = 3
_GEN_MAX_AREA = 6
_GEN_SUPPORT_ROW_OFFSET = 2

_GEN_TWO_OBJECT_WIDE_SPAN_PROB = 0.7
_GEN_WIDE_WIDTH_CHOICES = (6, 5)
_GEN_WIDEST_WIDTH_PROB = 0.55
_GEN_SPAN6_AREA6_PROB = 0.75
_GEN_SPAN5_AREA5_PROB = 0.65

_GEN_MAX_ATTEMPTS = 300


def _gen_partition_spans(n):
    available = _GEN_GRID_SIZE - _GEN_MIN_INTER_OBJECT_GAP * (n - 1)
    if n == 2 and random.random() < _GEN_TWO_OBJECT_WIDE_SPAN_PROB:
        wide = _GEN_WIDE_WIDTH_CHOICES[0] if random.random() < _GEN_WIDEST_WIDTH_PROB else _GEN_WIDE_WIDTH_CHOICES[1]
        widths = [wide, available - wide]
        random.shuffle(widths)
    else:
        total = random.randint(n, available)
        widths = [1] * n
        for _ in range(total - n):
            widths[random.randrange(n)] += 1
        random.shuffle(widths)

    extra = _GEN_GRID_SIZE - sum(widths) - _GEN_MIN_INTER_OBJECT_GAP * (n - 1)
    gap_extra = [0] * (n + 1)
    for _ in range(extra):
        gap_extra[random.randrange(n + 1)] += 1

    spans = []
    c = gap_extra[0]
    for i, width in enumerate(widths):
        spans.append((c, c + width - 1))
        c += width
        if i < n - 1:
            c += _GEN_MIN_INTER_OBJECT_GAP + gap_extra[i + 1]
    return spans


def _gen_shape_params(span_width):
    pairs = [
        (area, height)
        for height in range(1, _GEN_MAX_HEIGHT + 1)
        for area in range(1, _GEN_MAX_AREA + 1)
        if height <= area
        and area + height <= _GEN_GRID_SIZE - 1
        and area <= height * span_width
    ]
    areas = sorted({area for area, _ in pairs})
    if span_width >= 6 and random.random() < _GEN_SPAN6_AREA6_PROB:
        area = 6
    elif span_width >= 5 and random.random() < _GEN_SPAN5_AREA5_PROB:
        area = 5
    else:
        area = random.choice(areas)
    heights = [height for cand_area, height in pairs if cand_area == area]
    return area, random.choice(heights)


def _gen_connected_mask(area, height, width):
    backbone_col = random.randrange(width)
    cells = {(r, backbone_col) for r in range(height)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _gen_paste_shape(grid, mask, top, left, color):
    patch = np.full(mask.shape, _GEN_BG, dtype=int)
    patch[mask] = color
    paste(grid, patch, top, left, transparent=_GEN_BG)


def _gen_build_objects(n_objects):
    spans = _gen_partition_spans(n_objects)
    colors = random.sample(_GEN_COLORS, n_objects)
    objects = []
    for color, (c0, c1) in zip(colors, spans):
        span_width = c1 - c0 + 1
        area, height = _gen_shape_params(span_width)
        mask = _gen_connected_mask(area, height, span_width)
        bottom = _GEN_GRID_SIZE - _GEN_SUPPORT_ROW_OFFSET - area
        top = bottom - height + 1
        if top < 0:
            return None
        objects.append((mask, top, c0, area, color))
    return objects


def _gen_render(objects, shift_down=0):
    grid = np.full((_GEN_GRID_SIZE, _GEN_GRID_SIZE), _GEN_BG, dtype=int)
    grid[-1, :] = _GEN_STRIPE
    for mask, top, c0, area, color in objects:
        _gen_paste_shape(grid, mask, top + shift_down * area, c0, color)
    return grid


def _gen_attempt():
    n_objects = random.choice(_GEN_OBJECT_COUNTS)
    objects = _gen_build_objects(n_objects)
    if objects is None:
        return None

    out = _gen_render(objects)
    inp = _gen_render(objects, shift_down=1)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b2bc3ffd input")
