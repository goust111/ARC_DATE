import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, label_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey pixel that has a same-row horizontal mirror partner to blue.

    Input structure:
    - The grid has a black background and one grey foreground mask.
    - The grid width is even, so horizontal mirror partners are distinct cells.
    - Blue is absent from the input and is introduced only in the output.

    Rule:
    - For each grey pixel, look at the cell reflected across the vertical midline.
    - If the reflected cell is also grey, recolor the original pixel blue.
    - Leave unpaired grey pixels and all background pixels unchanged.

    Key insight:
    - The output marks the horizontally symmetric grey pairs in each row; grey pixels without a matching mirror partner are noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _, width = map(int, g.shape)
    assert width % 2 == 0, "expected an even-width grid with distinct mirror partners"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    assert Color.BLUE not in colors_present(g), "blue should be absent from the input"

    foreground = colors_present(g, exclude={bg})
    assert foreground == [Color.GREY], "expected a single grey foreground color"

    grey = g == Color.GREY
    paired = grey & np.fliplr(grey)

    out = g.copy()
    out[paired] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

# Canvas.
GEN_HEIGHT = GEN_WIDTH = 10

# Symmetric core placement.
MIRROR_LAST_COL, CORE_START_COL, CORE_START_ROW_RANGE = GEN_WIDTH - 1, 4, (1, 9)
INNER_CORE_COLS, ALL_LEFT_CORE_COLS, ISOLATED_PAIR_COLS = range(1, 5), range(0, 5), range(0, 4)
INNER_CORE_COL_PROB, ORTHOGONAL_STEPS = 0.7, ((1, 0), (-1, 0), (0, 1), (0, -1))

# Cluster growth weighting.
ROW_BIAS_CENTER, COL_BIAS_CENTER = 5, 4
ROW_BIAS_SCALE, COL_BIAS_SCALE = 0.35, 0.45

# Count sampling and final validation.
TARGET_COMPONENT_RANGE, PAIR_COUNT_RANGE = (4, 14), (4, 11)
UNPAIRED_COUNTS_FOR_MIN_PAIRS, UNPAIRED_COUNTS_DEFAULT = (6, 7), (5, 6, 7)
MIN_CORE_PAIR_CELLS, MAX_ISOLATED_PAIR_CELLS = 3, 4
BASE_CORE_COMPONENTS, ISOLATED_PAIR_COMPONENTS = 1, 2
GREY_COUNT_RANGE, PAIRED_PIXEL_RANGE, UNPAIRED_PIXEL_RANGE = (14, 27), (8, 20), (5, 7)

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_mirror_col(c):
    return MIRROR_LAST_COL - int(c)


def _gen_grow_pair_cluster(rng, size, start, allowed_cols):
    allowed_cols = set(int(c) for c in allowed_cols)
    cells = {tuple(start)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for dr, dc in ORTHOGONAL_STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GEN_HEIGHT and nc in allowed_cols and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return cells
        options = sorted(frontier)
        weights = []
        for r, c in options:
            center_bias = 1.0 / (1.0 + abs(r - ROW_BIAS_CENTER) * ROW_BIAS_SCALE + abs(c - COL_BIAS_CENTER) * COL_BIAS_SCALE)
            weights.append(center_bias)
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        cells.add(options[int(rng.choice(len(options), p=weights))])
    return cells


def _gen_place_mirror_pair(grid, r, c):
    grid[r, c] = grid[r, _gen_mirror_col(c)] = Color.GREY


def _gen_neighbors4(r, c):
    for dr, dc in ORTHOGONAL_STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GEN_HEIGHT and 0 <= nc < GEN_WIDTH:
            yield nr, nc


def _gen_can_place_unpaired(grid, r, c):
    return grid[r, c] == Color.BLACK and grid[r, _gen_mirror_col(c)] == Color.BLACK


def _gen_is_isolated_from_grey(grid, cells):
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in _gen_neighbors4(r, c):
            if (nr, nc) not in cell_set and grid[nr, nc] == Color.GREY:
                return False
    return True


def _gen_unpaired_single_options(grid, require_isolated=True):
    opts = []
    for r in range(GEN_HEIGHT):
        for c in range(GEN_WIDTH):
            if not _gen_can_place_unpaired(grid, r, c):
                continue
            if require_isolated and not _gen_is_isolated_from_grey(grid, [(r, c)]):
                continue
            opts.append((r, c))
    return opts


def _gen_isolated_pair_options(grid):
    opts = []
    for r in range(GEN_HEIGHT):
        for c in ISOLATED_PAIR_COLS:
            cells = [(r, c), (r, _gen_mirror_col(c))]
            if all(grid[rr, cc] == Color.BLACK for rr, cc in cells) and _gen_is_isolated_from_grey(grid, cells):
                opts.append((r, c))
    return opts


def _gen_place_isolated_pair_cells(rng, grid, count):
    for _ in range(count):
        opts = _gen_isolated_pair_options(grid)
        if not opts:
            return False
        r, c = opts[int(rng.integers(len(opts)))]
        _gen_place_mirror_pair(grid, r, c)
    return True


def _gen_unpaired_attached_options(grid):
    labels, _ = label_mask(grid == Color.GREY, connectivity=4)
    opts = []
    for r in range(GEN_HEIGHT):
        for c in range(GEN_WIDTH):
            if not _gen_can_place_unpaired(grid, r, c):
                continue
            neighbor_labels = {int(labels[nr, nc]) for nr, nc in _gen_neighbors4(r, c) if labels[nr, nc] > 0}
            if len(neighbor_labels) == 1:
                opts.append((r, c))
    return opts


def _gen_add_unpaired_noise(rng, grid, count, component_count):
    for _ in range(component_count):
        opts = _gen_unpaired_single_options(grid, require_isolated=True)
        if not opts:
            return False
        r, c = opts[int(rng.integers(len(opts)))]
        grid[r, c] = Color.GREY

    for _ in range(count - component_count):
        opts = _gen_unpaired_attached_options(grid)
        if not opts:
            return False
        r, c = opts[int(rng.integers(len(opts)))]
        grid[r, c] = Color.GREY
    return True


def _gen_choose_count_plan(rng):
    """Sample paired/unpaired counts that can realize the target component count."""
    target_components = int(rng.integers(*TARGET_COMPONENT_RANGE))
    combos = []
    for candidate_pair_count in range(*PAIR_COUNT_RANGE):
        unpaired_choices = UNPAIRED_COUNTS_FOR_MIN_PAIRS if candidate_pair_count == PAIR_COUNT_RANGE[0] else UNPAIRED_COUNTS_DEFAULT
        for candidate_unpaired in unpaired_choices:
            max_isolated = min(candidate_pair_count - MIN_CORE_PAIR_CELLS, MAX_ISOLATED_PAIR_CELLS)
            for candidate_isolated_pairs in range(max_isolated + 1):
                candidate_unpaired_components = target_components - (BASE_CORE_COMPONENTS + ISOLATED_PAIR_COMPONENTS * candidate_isolated_pairs)
                if 1 <= candidate_unpaired_components <= candidate_unpaired:
                    combos.append((candidate_pair_count, candidate_unpaired, candidate_isolated_pairs, candidate_unpaired_components))
    if not combos:
        return None
    feasible_pair_counts = sorted({combo[0] for combo in combos})
    pair_count = int(rng.choice(feasible_pair_counts))
    pair_combos = [combo for combo in combos if combo[0] == pair_count]
    pair_count, unpaired_count, isolated_pair_count, unpaired_component_count = pair_combos[int(rng.integers(len(pair_combos)))]
    return target_components, pair_count, unpaired_count, isolated_pair_count, unpaired_component_count


def _gen_place_paired_structure(rng, grid, pair_count, isolated_pair_count):
    """Place one connected mirrored core plus optional isolated mirrored pairs."""
    start_row = int(rng.integers(*CORE_START_ROW_RANGE))
    central_pair_count = pair_count - isolated_pair_count
    allowed = INNER_CORE_COLS if rng.random() < INNER_CORE_COL_PROB else ALL_LEFT_CORE_COLS
    pair_cells = _gen_grow_pair_cluster(rng, central_pair_count, (start_row, CORE_START_COL), allowed)

    if len(pair_cells) != central_pair_count:
        return False
    for r, c in pair_cells:
        _gen_place_mirror_pair(grid, r, c)
    return _gen_place_isolated_pair_cells(rng, grid, isolated_pair_count)


def _gen_grid_stats(grid):
    grey = grid == Color.GREY
    paired = grey & np.fliplr(grey)
    grey_count = int(grey.sum())
    paired_count = int(paired.sum())
    unpaired_actual = grey_count - paired_count
    comp_count = len(find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True))
    return paired, grey_count, paired_count, unpaired_actual, comp_count


def _gen_validate_grid(grid, target_components):
    """Return the grid only if counts and the solver oracle match."""
    paired, grey_count, paired_count, unpaired_actual, comp_count = _gen_grid_stats(grid)
    if not (GREY_COUNT_RANGE[0] <= grey_count <= GREY_COUNT_RANGE[1]):
        return None
    if not (PAIRED_PIXEL_RANGE[0] <= paired_count <= PAIRED_PIXEL_RANGE[1] and UNPAIRED_PIXEL_RANGE[0] <= unpaired_actual <= UNPAIRED_PIXEL_RANGE[1]):
        return None
    if comp_count != target_components:
        return None

    out = grid.copy()
    out[paired] = Color.BLUE
    if not np.array_equal(main(grid), out):
        return None
    return grid


def _gen_attempt():
    rng = np.random.default_rng()
    plan = _gen_choose_count_plan(rng)
    if plan is None:
        return None
    target_components, pair_count, unpaired_count, isolated_pair_count, unpaired_component_count = plan

    grid = np.zeros((GEN_HEIGHT, GEN_WIDTH), dtype=int)
    if not _gen_place_paired_structure(rng, grid, pair_count, isolated_pair_count):
        return None
    if not _gen_add_unpaired_noise(rng, grid, unpaired_count, unpaired_component_count):
        return None
    return _gen_validate_grid(grid, target_components)


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate ce039d91 input")
