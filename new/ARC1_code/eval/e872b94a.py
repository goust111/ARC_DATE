import numpy as np

from arc_utils import colors_present, get_bg, label_mask
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the 4-connected background regions and return one background pixel per region as a single column.

    Input structure:
    - The grid has one dominant background color and one foreground color.
    - Foreground path-like barriers split the background into separate 4-connected regions.

    Rule:
    - Infer the background as the most common color.
    - Count the 4-connected components of the background mask.
    - Output a one-column grid with one background pixel for each background component.

    Key insight:
    - The foreground paths act as separators; the output encodes the number of background spaces they create.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    _labels, region_count = label_mask(g == bg, connectivity=4)
    assert region_count > 0, "expected at least one background region"

    return np.full((region_count, 1), bg, dtype=int)

# --------------- generator (o5x) ---------------

# Color selection.
PREFERRED_COLOR_PROB = 0.75
PREFERRED_COLOR_PAIR, ARC_COLOR_COUNT = (0, 5), 10
# Shared validation and retry limits.
FG_DENSITY_PCT_RANGE = (30, 45)
RANDOM_FLIP_PROB, GENERATION_RETRIES = 0.5, 300
# Vertical barrier layouts.
BARRIER_PATH_COUNT_RANGE, BARRIER_SPAN_RANGE = (1, 3), (3, 12)
BARRIER_WIDTH_MAX, BARRIER_LANE_WIDTH, BARRIER_MIN_GAP = 12, 2, 1
BARRIER_MIN_MOVES_PER_PATH, BARRIER_MAX_MOVES_PER_PATH = 1, 5
# Four-barrier special case.
FOUR_BARRIER_HEIGHT_RANGE, FOUR_BARRIER_WIDTH = (8, 12), 12
FOUR_BARRIER_COLUMNS = (1, 4, 7, 10)
FOUR_BARRIER_NUB_DIRECTIONS, FOUR_BARRIER_REGION_COUNT = (-1, 1), 5
# Pocket layouts.
POCKET_PATH_COUNT_RANGE = (1, 4)
POCKET_SIZE_BY_PATH_COUNT = ((0, 0), (5, 6), (6, 8), (8, 10), (9, 12))
POCKET_EDGES, NEIGHBORS4 = ("top", "bottom", "left", "right"), ((1, 0), (-1, 0), (0, 1), (0, -1))
POCKET_MIN_RUN, POCKET_MAX_RUN, POCKET_MAX_DEPTH = 3, 5, 4
POCKET_BORDER_MARGIN, POCKET_CENTER_JITTER = 1, 2
# Long single-pocket special case.
LONG_POCKET_HEIGHT_RANGE, LONG_POCKET_WIDTH_RANGE = (9, 12), (5, 7)
LONG_POCKET_RUN_RANGE, LONG_POCKET_FG_CELLS_RANGE = (3, 5), (18, 23)
LONG_POCKET_REGION_COUNT = 2
# Top-level strategy thresholds.
FOUR_BARRIER_CUTOFF, LONG_POCKET_CUTOFF, POCKET_CUTOFF = 0.12, 0.22, 0.40


def _gen_color_pair():
    if random.random() < PREFERRED_COLOR_PROB:
        return PREFERRED_COLOR_PAIR
    bg = random.randrange(ARC_COLOR_COUNT)
    fg = random.randrange(ARC_COLOR_COUNT - 1)
    if fg >= bg:
        fg += 1
    return bg, fg


def _gen_nonadjacent_positions(n, k):
    if k <= 0:
        return []
    base = sorted(random.sample(range(n - k + 1), k))
    return [v + i for i, v in enumerate(base)]


def _gen_lane_intervals(width, count):
    gaps = [BARRIER_MIN_GAP] * (count + 1)
    min_width = BARRIER_LANE_WIDTH * count + BARRIER_MIN_GAP * (count + 1)
    extra = width - min_width
    for _ in range(extra):
        gaps[random.randrange(count + 1)] += 1
    out = []
    pos = gaps[0]
    for i in range(count):
        out.append((pos, pos + BARRIER_LANE_WIDTH - 1))
        pos += BARRIER_LANE_WIDTH + gaps[i + 1]
    return out


def _gen_wall_path(span, lo_col, move_count):
    moves = set(_gen_nonadjacent_positions(span - 1, move_count))
    col = lo_col + random.randrange(BARRIER_LANE_WIDTH)
    coords = [(0, col)]
    for r in range(span - 1):
        if r in moves:
            col = lo_col + (BARRIER_LANE_WIDTH - 1 - (col - lo_col))
            coords.append((r, col))
        coords.append((r + 1, col))
    return coords


def _gen_spread(total, count, lo, hi):
    vals = [lo] * count
    remaining = total - lo * count
    caps = [hi - lo] * count
    while remaining > 0:
        choices = [i for i, cap in enumerate(caps) if cap > 0]
        i = random.choice(choices)
        vals[i] += 1
        caps[i] -= 1
        remaining -= 1
    random.shuffle(vals)
    return vals


def _density_bounds(area):
    lo_pct, hi_pct = FG_DENSITY_PCT_RANGE
    return (lo_pct * area + 99) // 100, (hi_pct * area) // 100


def _density_ok(fg_cells, area):
    lo_pct, hi_pct = FG_DENSITY_PCT_RANGE
    return lo_pct * area <= 100 * fg_cells <= hi_pct * area


def _paint_coords(grid, coords, color):
    if coords:
        rows, cols = zip(*coords)
        grid[rows, cols] = color


def _apply_random_flips(grid):
    if random.random() < RANDOM_FLIP_PROB:
        grid = np.flipud(grid).copy()
    if random.random() < RANDOM_FLIP_PROB:
        grid = np.fliplr(grid).copy()
    return grid


def _gen_accept_grid(grid, bg, expected_regions):
    area = int(grid.size)
    fg_cells = int(np.count_nonzero(grid != bg))
    if not _density_ok(fg_cells, area):
        return None
    _, region_count = label_mask(grid == bg, connectivity=4)
    if region_count != expected_regions:
        return None
    expected = np.full((expected_regions, 1), bg, dtype=int)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def _build_barrier_combos():
    combos = []
    for path_count in range(BARRIER_PATH_COUNT_RANGE[0], BARRIER_PATH_COUNT_RANGE[1] + 1):
        min_width = BARRIER_LANE_WIDTH * path_count + BARRIER_MIN_GAP * (path_count + 1)
        for span in range(BARRIER_SPAN_RANGE[0], BARRIER_SPAN_RANGE[1] + 1):
            max_moves_per_path = min(span // 2, BARRIER_MAX_MOVES_PER_PATH)
            if max_moves_per_path < BARRIER_MIN_MOVES_PER_PATH:
                continue
            for width in range(min_width, BARRIER_WIDTH_MAX + 1):
                area = span * width
                min_fg, max_fg = _density_bounds(area)
                move_lo = max(path_count * BARRIER_MIN_MOVES_PER_PATH, min_fg - path_count * span)
                move_hi = min(path_count * max_moves_per_path, max_fg - path_count * span)
                if move_lo <= move_hi:
                    combos.append((path_count, span, width, move_lo, move_hi, max_moves_per_path))
    return combos


def _gen_barrier_attempt():
    path_count, span, width, move_lo, move_hi, max_moves_per_path = random.choice(BARRIER_COMBOS)
    total_moves = random.randint(move_lo, move_hi)
    moves_by_path = _gen_spread(total_moves, path_count, BARRIER_MIN_MOVES_PER_PATH, max_moves_per_path)
    bg, fg = _gen_color_pair()
    grid = np.full((span, width), bg, dtype=int)
    for (lo_col, _), move_count in zip(_gen_lane_intervals(width, path_count), moves_by_path):
        _paint_coords(grid, _gen_wall_path(span, lo_col, move_count), fg)
    grid = _apply_random_flips(grid)
    return _gen_accept_grid(grid, bg, path_count + 1)


def _gen_four_barrier_attempt():
    height = random.randint(*FOUR_BARRIER_HEIGHT_RANGE)
    width = FOUR_BARRIER_WIDTH
    bg, fg = _gen_color_pair()
    grid = np.full((height, width), bg, dtype=int)
    nub_dir = random.choice(FOUR_BARRIER_NUB_DIRECTIONS)
    for col in FOUR_BARRIER_COLUMNS:
        grid[0, col + nub_dir] = fg
        grid[:, col] = fg
    grid = _apply_random_flips(grid)
    return _gen_accept_grid(grid, bg, FOUR_BARRIER_REGION_COUNT)


def _gen_pocket_coords(height, width, edge, start, run, depth):
    end = start + run - 1
    coords = [(r, start) for r in range(depth + 1)]
    coords.extend((depth, c) for c in range(start + 1, end + 1))
    coords.extend((r, end) for r in range(depth - 1, -1, -1))
    if edge == "top":
        return coords
    if edge == "bottom":
        return [(height - 1 - r, c) for r, c in coords]
    if edge == "left":
        return [(c, r) for r, c in coords]
    return [(c, width - 1 - r) for r, c in coords]


def _gen_segment_start(long_dim, run):
    lo = POCKET_BORDER_MARGIN
    hi = long_dim - run - POCKET_BORDER_MARGIN
    if hi < lo:
        return None
    center = (long_dim - run) // 2
    lo = max(lo, center - POCKET_CENTER_JITTER)
    hi = min(hi, center + POCKET_CENTER_JITTER)
    return random.randint(lo, hi)


def _gen_pocket_params(height, width, edge):
    long_dim, inward_dim = (width, height) if edge in ("top", "bottom") else (height, width)
    max_run = min(POCKET_MAX_RUN, long_dim - 2 * POCKET_BORDER_MARGIN)
    max_depth = min(POCKET_MAX_DEPTH, max(1, (inward_dim - (2 * POCKET_BORDER_MARGIN + 1)) // 2))
    if max_run < POCKET_MIN_RUN or max_depth < 1:
        return None
    run = random.randint(POCKET_MIN_RUN, max_run)
    depth = random.randint(1, max_depth)
    start = _gen_segment_start(long_dim, run)
    if start is None:
        return None
    return start, run, depth


def _available_coord_set(coords, occupied):
    coord_set = set(coords)
    if len(coord_set) != len(coords):
        return None
    for r, c in coord_set:
        if (r, c) in occupied or any((r + dr, c + dc) in occupied for dr, dc in NEIGHBORS4):
            return None
    return coord_set


def _gen_pocket_attempt():
    path_count = random.randint(*POCKET_PATH_COUNT_RANGE)
    size_lo, size_hi = POCKET_SIZE_BY_PATH_COUNT[path_count]
    height = random.randint(size_lo, size_hi)
    width = random.randint(size_lo, size_hi)
    bg, fg = _gen_color_pair()
    grid = np.full((height, width), bg, dtype=int)
    occupied = set()
    for edge in random.sample(POCKET_EDGES, path_count):
        params = _gen_pocket_params(height, width, edge)
        if params is None:
            return None
        start, run, depth = params
        coords = _gen_pocket_coords(height, width, edge, start, run, depth)
        coord_set = _available_coord_set(coords, occupied)
        if coord_set is None:
            return None
        occupied.update(coord_set)
        _paint_coords(grid, coords, fg)
    return _gen_accept_grid(grid, bg, path_count + 1)


def _build_long_pocket_combos():
    combos = []
    min_fg_cells, max_fg_cells = LONG_POCKET_FG_CELLS_RANGE
    for height in range(LONG_POCKET_HEIGHT_RANGE[0], LONG_POCKET_HEIGHT_RANGE[1] + 1):
        for width in range(LONG_POCKET_WIDTH_RANGE[0], LONG_POCKET_WIDTH_RANGE[1] + 1):
            max_run = min(LONG_POCKET_RUN_RANGE[1], width)
            for run in range(LONG_POCKET_RUN_RANGE[0], max_run + 1):
                for depth in range(1, height - 1):
                    fg_cells = 2 * depth + run
                    area = height * width
                    if min_fg_cells <= fg_cells <= max_fg_cells and _density_ok(fg_cells, area):
                        combos.append((height, width, run, depth))
    return combos


BARRIER_COMBOS = _build_barrier_combos()
LONG_POCKET_COMBOS = _build_long_pocket_combos()


def _gen_long_pocket_attempt():
    height, width, run, depth = random.choice(LONG_POCKET_COMBOS)
    start = random.randint(0, width - run)
    bg, fg = _gen_color_pair()
    grid = np.full((height, width), bg, dtype=int)
    _paint_coords(grid, _gen_pocket_coords(height, width, "top", start, run, depth), fg)
    grid = _apply_random_flips(grid)
    return _gen_accept_grid(grid, bg, LONG_POCKET_REGION_COUNT)


def _gen_attempt():
    choice = random.random()
    if choice < FOUR_BARRIER_CUTOFF:
        return _gen_four_barrier_attempt()
    if choice < LONG_POCKET_CUTOFF:
        return _gen_long_pocket_attempt()
    if choice < POCKET_CUTOFF:
        return _gen_pocket_attempt()
    return _gen_barrier_attempt()


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e872b94a input")
