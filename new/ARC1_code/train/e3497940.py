import numpy as np

from arc_utils import get_bg


def _separator_col(g: np.ndarray, bg: int) -> int:
    width = int(g.shape[1])
    cols = []
    for c in range(width):
        color = int(g[0, c])
        if color != bg and np.all(g[:, c] == color):
            cols.append(c)

    assert len(cols) == 1, "expected exactly one solid non-background separator column"
    sep = cols[0]
    sep_color = int(g[0, sep])
    assert not np.any(g[:, :sep] == sep_color), "separator color must not appear in the left region"
    assert not np.any(g[:, sep + 1 :] == sep_color), "separator color must not appear in the right region"
    return sep


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the grid across its solid vertical separator, overlaying the reflected right side onto the left side.

    Input structure
    A non-background vertical separator column spans the full grid height. It
    splits the remaining cells into equal-width left and right regions on the
    background. Reflected foreground overlaps, when present, have the same color.

    Rule
    Remove the separator column, mirror the right region horizontally, and
    overlay it onto the left region. For each output pixel, keep the foreground
    color from whichever folded side is non-background; if both sides are
    foreground, their colors must agree.

    Key insight
    The two sides are complementary halves around the separator, and the output
    is the separator-free left-side coordinate system after folding the right
    side inward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    bg = get_bg(g)
    height, width = map(int, g.shape)
    sep = _separator_col(g, bg)
    assert 2 * sep + 1 == width, "separator must split equal-width regions"

    left = g[:, :sep]
    right = np.fliplr(g[:, sep + 1 :])
    assert left.shape == right.shape == (height, sep), "folded regions must align"

    overlap = (left != bg) & (right != bg)
    assert np.all(left[overlap] == right[overlap]), "folded foreground overlaps must agree in color"

    out = left.copy()
    right_mask = right != bg
    out[right_mask] = right[right_mask]
    return out

# --------------- generator (o5x) ---------------

# Canvas, colors, and retry budget.
BG_COLOR, SEPARATOR_COLOR = 0, 5
HEIGHT_MIN, HEIGHT_MAX = 8, 14
HALF_WIDTH_MIN, HALF_WIDTH_MAX = 3, 6
FOREGROUND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300

# Output density and color counts.
PRIMARY_COUNT_MIN, PRIMARY_COUNT_MAX = 7, 11
SECONDARY_COUNT_MIN, SECONDARY_COUNT_MAX = 3, 5
FG_AREA_PERCENT_MIN, FG_AREA_PERCENT_MAX = 25, 40

# Fold split constraints.
SIDE_COUNT_MIN, SIDE_COUNT_MAX = 5, 13
MAX_OVERLAP = 6
MIN_SIDE_COLOR_COUNT = 1

# Sparse blob growth.
INTERIOR_ROW_START = 1
MAX_COMPONENTS_SMALL, MAX_COMPONENTS_LARGE = 2, 3
LARGE_COMPONENT_THRESHOLD = 6
GROWTH_MODE_COUNT, STRAIGHT_GROWTH_MODE = 3, 2
STRAIGHT_TURN_PROB = 0.25
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _gen_pick(seq):
    return seq[int(np.random.randint(len(seq)))]


def _gen_partition(total, parts):
    if parts <= 1:
        return [total]
    cuts = sorted(np.random.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _gen_neighbors(cell, height, width):
    r, c = cell
    out = []
    for dr, dc in ORTHOGONAL_STEPS:
        nr, nc = r + dr, c + dc
        if INTERIOR_ROW_START <= nr < height - 1 and 0 <= nc < width:
            out.append((nr, nc))
    return out


def _gen_grow_cells(height, width, count, occupied):
    free = {
        (r, c)
        for r in range(INTERIOR_ROW_START, height - 1)
        for c in range(width)
        if (r, c) not in occupied
    }
    if len(free) < count:
        return None

    component_limit = MAX_COMPONENTS_LARGE if count >= LARGE_COMPONENT_THRESHOLD else MAX_COMPONENTS_SMALL
    max_components = min(count, component_limit)
    component_count = int(np.random.randint(1, max_components + 1))
    sizes = _gen_partition(count, component_count)
    cells = []

    for size in sizes:
        if not free:
            return None
        seed = _gen_pick(tuple(free))
        free.remove(seed)
        cells.append(seed)
        component = [seed]
        frontier = {nb for nb in _gen_neighbors(seed, height, width) if nb in free}
        mode = int(np.random.randint(GROWTH_MODE_COUNT))
        head = seed
        direction = _gen_pick(ORTHOGONAL_STEPS)

        while len(component) < size:
            if mode == 1:
                candidates = [nb for nb in _gen_neighbors(head, height, width) if nb in free]
            elif mode == STRAIGHT_GROWTH_MODE:
                nr, nc = head[0] + direction[0], head[1] + direction[1]
                candidates = [(nr, nc)] if (nr, nc) in free else []
            else:
                candidates = list(frontier)

            if not candidates:
                candidates = list(frontier)
            if not candidates:
                if not free:
                    return None
                nxt = _gen_pick(tuple(free))
            else:
                nxt = _gen_pick(candidates)

            free.remove(nxt)
            cells.append(nxt)
            component.append(nxt)
            head = nxt
            if mode == STRAIGHT_GROWTH_MODE and np.random.random() < STRAIGHT_TURN_PROB:
                direction = _gen_pick(ORTHOGONAL_STEPS)
            frontier.discard(nxt)
            for nb in _gen_neighbors(nxt, height, width):
                if nb in free:
                    frontier.add(nb)

    return cells


def _gen_feasible_layout():
    primary_count = int(np.random.randint(PRIMARY_COUNT_MIN, PRIMARY_COUNT_MAX + 1))
    secondary_count = int(np.random.randint(SECONDARY_COUNT_MIN, SECONDARY_COUNT_MAX + 1))
    total = primary_count + secondary_count
    feasible = []
    for height in range(HEIGHT_MIN, HEIGHT_MAX + 1):
        for width in range(HALF_WIDTH_MIN, HALF_WIDTH_MAX + 1):
            area = height * width
            if (total * 100 >= FG_AREA_PERCENT_MIN * area
                    and total * 100 <= FG_AREA_PERCENT_MAX * area
                    and total <= (height - 2) * width):
                feasible.append((height, width))
    if not feasible:
        return None
    height, width = _gen_pick(feasible)
    return height, width, primary_count, secondary_count


def _gen_split_counts(primary_count, secondary_count):
    total = primary_count + secondary_count
    overlap = int(np.random.randint(0, min(MAX_OVERLAP, total) + 1))
    options = []
    for left_count in range(SIDE_COUNT_MIN, SIDE_COUNT_MAX + 1):
        right_count = total + overlap - left_count
        if not SIDE_COUNT_MIN <= right_count <= SIDE_COUNT_MAX:
            continue
        left_only = left_count - overlap
        right_only = right_count - overlap
        if left_only < 0 or right_only < 0:
            continue
        for primary_overlap in range(max(0, overlap - secondary_count), min(primary_count, overlap) + 1):
            secondary_overlap = overlap - primary_overlap
            primary_remaining = primary_count - primary_overlap
            secondary_remaining = secondary_count - secondary_overlap
            for primary_left_only in range(max(0, left_only - secondary_remaining),
                                           min(primary_remaining, left_only) + 1):
                secondary_left_only = left_only - primary_left_only
                primary_right_only = primary_remaining - primary_left_only
                secondary_right_only = secondary_remaining - secondary_left_only
                if primary_right_only < 0 or secondary_right_only < 0:
                    continue
                primary_left = primary_overlap + primary_left_only
                primary_right = primary_overlap + primary_right_only
                secondary_left = secondary_overlap + secondary_left_only
                secondary_right = secondary_overlap + secondary_right_only
                if min(primary_left, primary_right, secondary_left, secondary_right) < MIN_SIDE_COLOR_COUNT:
                    continue
                options.append((
                    overlap,
                    left_count,
                    right_count,
                    (primary_overlap, primary_left_only, primary_right_only),
                    (secondary_overlap, secondary_left_only, secondary_right_only),
                ))
    if not options:
        return None
    return _gen_pick(options)


def _gen_choose_palette():
    palette = list(FOREGROUND_COLORS)
    np.random.shuffle(palette)
    return int(palette[0]), int(palette[1])


def _gen_paint_cells(grid, cells, color):
    for r, c in cells:
        grid[r, c] = color


def _gen_apply_split(cells, color, split_counts, left, right):
    overlap_count, left_only_count, right_only_count = split_counts
    shuffled = list(cells)
    np.random.shuffle(shuffled)
    both = shuffled[:overlap_count]
    left_only = shuffled[overlap_count:overlap_count + left_only_count]
    right_only = shuffled[overlap_count + left_only_count:
                          overlap_count + left_only_count + right_only_count]
    _gen_paint_cells(left, both, color)
    _gen_paint_cells(right, both, color)
    _gen_paint_cells(left, left_only, color)
    _gen_paint_cells(right, right_only, color)


def _gen_attempt():
    layout = _gen_feasible_layout()
    if layout is None:
        return None
    height, width, primary_count, secondary_count = layout

    primary_color, secondary_color = _gen_choose_palette()
    occupied = set()
    primary_cells = _gen_grow_cells(height, width, primary_count, occupied)
    if primary_cells is None:
        return None
    occupied.update(primary_cells)
    secondary_cells = _gen_grow_cells(height, width, secondary_count, occupied)
    if secondary_cells is None:
        return None

    out = np.full((height, width), BG_COLOR, dtype=int)
    _gen_paint_cells(out, primary_cells, primary_color)
    _gen_paint_cells(out, secondary_cells, secondary_color)

    split = _gen_split_counts(primary_count, secondary_count)
    if split is None:
        return None
    _, left_count, right_count, primary_split, secondary_split = split
    if not (SIDE_COUNT_MIN <= left_count <= SIDE_COUNT_MAX
            and SIDE_COUNT_MIN <= right_count <= SIDE_COUNT_MAX):
        return None

    left = np.full((height, width), BG_COLOR, dtype=int)
    right = np.full((height, width), BG_COLOR, dtype=int)
    _gen_apply_split(primary_cells, primary_color, primary_split, left, right)
    _gen_apply_split(secondary_cells, secondary_color, secondary_split, left, right)

    grid = np.full((height, 2 * width + 1), BG_COLOR, dtype=int)
    grid[:, :width] = left
    grid[:, width] = SEPARATOR_COLOR
    grid[:, width + 1:] = np.fliplr(right)

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
