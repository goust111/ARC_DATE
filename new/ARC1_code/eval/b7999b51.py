import numpy as np

from arc_utils import bbox, bbox_h, bbox_w, colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize each foreground color as a top-aligned vertical bar ordered by bbox height.

    Input structure
    The grid has a dominant background color and several foreground color masks.
    The foreground colors have distinct bounding-box heights.

    Rule
    Measure the bounding-box height of each foreground color mask. Sort the
    colors from tallest to shortest, make one output column per color, and fill
    the top cells of that column with the color for its measured height.

    Key insight
    The input positions and horizontal extents are discarded; only each color's
    vertical span determines the histogram height and column order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    specs: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg}):
        mask = g == color
        b = bbox(mask)
        assert b is not None, "foreground colors must be present"
        specs.append((int(color), bbox_h(b)))

    assert specs, "expected at least one foreground color"
    heights = [height for _, height in specs]
    assert len(heights) == len(set(heights)), "bbox heights must uniquely determine column order"

    specs.sort(key=lambda item: -item[1])
    out = np.full((max(heights), len(specs)), bg, dtype=int)
    for col, (color, height) in enumerate(specs):
        out[:height, col] = color
    return out

# --------------- generator (o5x) ---------------
# Color and bbox sampling.
N_COLORS_MIN, N_COLORS_MAX = 3, 5
COLOR_MIN, COLOR_MAX = 1, 9
BBOX_HEIGHT_MIN, BBOX_HEIGHT_MAX = 1, 6
BBOX_WIDTH_MIN, BBOX_WIDTH_MAX = 4, 12

# Canvas sizing.
GRID_ROWS_MIN, GRID_ROWS_MAX, GRID_COLS_MIN, GRID_COLS_MAX = 10, 19, 15, 18
HEIGHT_PADDING, COLOR_COUNT_PADDING = 4, 8

# Mask density, overlap, and retry behavior.
BACKGROUND = 0
MIN_FILL_RATIO = 0.57
EDGE_HOLE_SEED_PROB, OVERLAP_POSITION_PROB = 0.8, 0.4
NEIGHBOR_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300


def _gen_output(color_height_pairs):
    """Build the expected histogram output from sampled color heights."""
    ordered = sorted(color_height_pairs, key=lambda item: -item[1])
    out = np.full((max(height for _, height in ordered), len(ordered)), BACKGROUND, dtype=int)
    for col, (color, height) in enumerate(ordered):
        out[:height, col] = color
    return out


def _gen_dense_mask(height, width):
    """Return a dense mask whose occupied bbox spans the requested size."""
    area = height * width
    if height == 1:
        return np.ones((height, width), dtype=bool)

    min_count = int(np.ceil(MIN_FILL_RATIO * area))
    target = random.randint(min_count, area)

    mask = np.ones((height, width), dtype=bool)
    required = {
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
    }
    available = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in required
    }
    holes = area - target
    if holes == 0:
        return mask

    edge_cells = [
        (r, c)
        for r, c in available
        if r in (0, height - 1) or c in (0, width - 1)
    ]
    if edge_cells and random.random() < EDGE_HOLE_SEED_PROB:
        seed = random.choice(edge_cells)
    else:
        seed = random.choice(tuple(available))

    removed = {seed}
    frontier = {
        (seed[0] + dr, seed[1] + dc)
        for dr, dc in NEIGHBOR_DELTAS
        if (seed[0] + dr, seed[1] + dc) in available
    }
    while len(removed) < holes:
        if frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
        else:
            cell = random.choice(tuple(available - removed))
        removed.add(cell)
        r, c = cell
        frontier.update(
            (r + dr, c + dc)
            for dr, dc in NEIGHBOR_DELTAS
            if (r + dr, c + dc) in available and (r + dr, c + dc) not in removed
        )

    rr, cc = zip(*removed)
    mask[rr, cc] = False
    return mask


def _gen_position(rows, cols, height, width, placed):
    """Pick either an overlapping or unconstrained top-left placement."""
    if placed and random.random() < OVERLAP_POSITION_PROB:
        pr0, pc0, ph, pw = random.choice(placed)
        r0 = random.randint(max(0, pr0 - height + 1), min(rows - height, pr0 + ph - 1))
        c0 = random.randint(max(0, pc0 - width + 1), min(cols - width, pc0 + pw - 1))
        return r0, c0
    return random.randint(0, rows - height), random.randint(0, cols - width)


def _gen_validate(inp, color_height_pairs, widths, expected):
    """Reject attempts whose occlusion changed the intended solver output."""
    if get_bg(inp) != BACKGROUND:
        return False
    colors = colors_present(inp, exclude={BACKGROUND})
    if set(colors) != {color for color, _ in color_height_pairs}:
        return False

    heights = []
    for color, wanted_height in color_height_pairs:
        mask = inp == color
        b = bbox(mask)
        if b is None:
            return False
        found_height = bbox_h(b)
        found_width = bbox_w(b)
        if found_height != wanted_height or found_width != widths[color]:
            return False
        fill_ratio = int(mask.sum()) / (found_height * found_width)
        if fill_ratio < MIN_FILL_RATIO:
            return False
        heights.append(found_height)

    if len(heights) != len(set(heights)):
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _sample_layout():
    """Sample intended output, canvas shape, and target bbox widths."""
    n_colors = random.randint(N_COLORS_MIN, N_COLORS_MAX)
    colors = random.sample(range(COLOR_MIN, COLOR_MAX + 1), n_colors)
    heights = random.sample(range(BBOX_HEIGHT_MIN, BBOX_HEIGHT_MAX + 1), n_colors)
    color_height_pairs = list(zip(colors, heights))
    expected = _gen_output(color_height_pairs)

    max_height = max(heights)
    min_side = max(max_height + HEIGHT_PADDING, n_colors + COLOR_COUNT_PADDING)
    rows = random.randint(max(GRID_ROWS_MIN, min_side), GRID_ROWS_MAX)
    cols = random.randint(max(GRID_COLS_MIN, min_side), GRID_COLS_MAX)
    widths = {color: random.randint(BBOX_WIDTH_MIN, min(BBOX_WIDTH_MAX, cols)) for color in colors}
    return color_height_pairs, expected, rows, cols, widths


def _render_input(rows, cols, color_height_pairs, widths):
    """Place shuffled masks; later colors may overwrite earlier colors."""
    inp = np.full((rows, cols), BACKGROUND, dtype=int)
    placed = []

    placement_order = color_height_pairs[:]
    random.shuffle(placement_order)
    for color, height in placement_order:
        width = widths[color]
        mask = _gen_dense_mask(height, width)
        r0, c0 = _gen_position(rows, cols, height, width, placed)
        rr, cc = np.nonzero(mask)
        inp[r0 + rr, c0 + cc] = color
        placed.append((r0, c0, height, width))
    return inp


def _gen_attempt() -> np.ndarray | None:
    color_height_pairs, expected, rows, cols, widths = _sample_layout()
    inp = _render_input(rows, cols, color_height_pairs, widths)
    return inp if _gen_validate(inp, color_height_pairs, widths, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid b7999b51 input")
