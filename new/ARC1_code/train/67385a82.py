import numpy as np

from arc_utils import Color, components_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every non-isolated green object on the black canvas to teal.

    Input structure
    - The grid is a black background containing green foreground pixels.
    - Green pixels form one or more 4-connected objects; some objects may be single pixels.

    Rule
    - Keep black pixels unchanged.
    - Keep each single-pixel green object unchanged.
    - Recolor every green object with area greater than one to teal.

    Key insight
    - The deciding property is 4-connected object size: isolated green pixels are markers to preserve, while larger green objects are recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.GREEN}, "expected only black and green in the input"
    assert np.any(g == Color.BLACK), "expected a black background"
    assert np.any(g == Color.GREEN), "expected at least one green foreground object"

    out = g.copy()
    large_components = 0
    for comp in components_mask(g == Color.GREEN, connectivity=4):
        if len(comp) > 1:
            out[comp[:, 0], comp[:, 1]] = Color.TEAL
            large_components += 1

    assert large_components > 0, "expected at least one non-isolated green object"
    return out

# --------------- generator (o5x) ---------------
# Canvas bounds.
CANVAS_SIDE_MIN = 3
CANVAS_SIDE_MAX = 6

# Component counts and object sizes.
COMPONENT_COUNT_MIN = 3
COMPONENT_COUNT_MAX = 5
MULTI_COMPONENT_COUNTS = (1, 2)
SINGLE_COMPONENT_COUNTS = (1, 2, 3)
BLOB_AREA_MIN = 2
BLOB_AREA_MAX = 5

# Green-cell density and retry policy.
GREEN_DENSITY_MIN = 0.20
GREEN_DENSITY_MAX = 0.60
GENERATION_RETRIES = 300

# 4-connected adjacency.
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_neighbors4(r, c, h, w):
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _gen_has_green_neighbor(grid, r, c):
    h, w = grid.shape
    for rr, cc in _gen_neighbors4(r, c, h, w):
        if grid[rr, cc] == Color.GREEN:
            return True
    return False


def _gen_green_count_bounds(area):
    return (
        int(np.ceil(area * GREEN_DENSITY_MIN)),
        int(np.floor(area * GREEN_DENSITY_MAX)),
    )


def _gen_size_options(num_multi, num_single, min_green, max_green):
    size_options = []
    if num_multi == 1:
        for a in range(BLOB_AREA_MIN, BLOB_AREA_MAX + 1):
            total_green = a + num_single
            if min_green <= total_green <= max_green:
                size_options.append((a,))
    else:
        for a in range(BLOB_AREA_MIN, BLOB_AREA_MAX + 1):
            for b in range(BLOB_AREA_MIN, BLOB_AREA_MAX + 1):
                total_green = a + b + num_single
                if min_green <= total_green <= max_green:
                    size_options.append((a, b))
    return size_options


def _gen_pick_structure(h, w):
    min_green, max_green = _gen_green_count_bounds(h * w)
    options = []

    for num_multi in MULTI_COMPONENT_COUNTS:
        for num_single in SINGLE_COMPONENT_COUNTS:
            if not (COMPONENT_COUNT_MIN <= num_multi + num_single <= COMPONENT_COUNT_MAX):
                continue
            size_options = _gen_size_options(num_multi, num_single, min_green, max_green)
            if size_options:
                options.append((num_single, size_options))

    if not options:
        return None

    num_single, size_options = options[np.random.randint(len(options))]
    sizes = list(size_options[np.random.randint(len(size_options))])
    np.random.shuffle(sizes)
    return num_single, sizes


def _gen_empty_isolated_cells(grid):
    h, w = grid.shape
    return [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == Color.BLACK and not _gen_has_green_neighbor(grid, r, c)
    ]


def _gen_grow_blob(grid, size):
    h, w = grid.shape
    seeds = _gen_empty_isolated_cells(grid)
    if not seeds:
        return None

    blob = {seeds[np.random.randint(len(seeds))]}
    while len(blob) < size:
        frontier = set()
        for r, c in blob:
            for rr, cc in _gen_neighbors4(r, c, h, w):
                if (
                    (rr, cc) not in blob
                    and grid[rr, cc] == Color.BLACK
                    and not _gen_has_green_neighbor(grid, rr, cc)
                ):
                    frontier.add((rr, cc))
        if not frontier:
            return None
        choices = list(frontier)
        blob.add(choices[np.random.randint(len(choices))])

    return list(blob)


def _gen_place_single(grid):
    cells = _gen_empty_isolated_cells(grid)
    if not cells:
        return False
    r, c = cells[np.random.randint(len(cells))]
    grid[r, c] = Color.GREEN
    return True


def _gen_expected_output(grid):
    out = grid.copy()
    for comp in components_mask(grid == Color.GREEN, connectivity=4):
        if len(comp) > 1:
            out[comp[:, 0], comp[:, 1]] = Color.TEAL
    return out


def _gen_place_multi_blobs(grid, multi_sizes):
    for size in multi_sizes:
        blob = _gen_grow_blob(grid, size)
        if blob is None:
            return False
        for r, c in blob:
            grid[r, c] = Color.GREEN
    return True


def _gen_valid_grid(grid):
    green_count = int(np.count_nonzero(grid == Color.GREEN))
    density = green_count / grid.size
    if not (GREEN_DENSITY_MIN <= density <= GREEN_DENSITY_MAX):
        return False

    sizes = [len(comp) for comp in components_mask(grid == Color.GREEN, connectivity=4)]
    singles = sum(size == 1 for size in sizes)
    multis = sum(size > 1 for size in sizes)
    if not (
        COMPONENT_COUNT_MIN <= len(sizes) <= COMPONENT_COUNT_MAX
        and SINGLE_COMPONENT_COUNTS[0] <= singles <= SINGLE_COMPONENT_COUNTS[-1]
        and MULTI_COMPONENT_COUNTS[0] <= multis <= MULTI_COMPONENT_COUNTS[-1]
        and all(size <= BLOB_AREA_MAX for size in sizes)
    ):
        return False
    return np.array_equal(main(grid), _gen_expected_output(grid))


def _gen_attempt():
    h = int(np.random.randint(CANVAS_SIDE_MIN, CANVAS_SIDE_MAX + 1))
    w = int(np.random.randint(CANVAS_SIDE_MIN, CANVAS_SIDE_MAX + 1))
    grid = np.full((h, w), Color.BLACK, dtype=int)

    structure = _gen_pick_structure(*grid.shape)
    if structure is None:
        return None
    num_single, multi_sizes = structure

    if not _gen_place_multi_blobs(grid, multi_sizes):
        return None
    for _ in range(num_single):
        if not _gen_place_single(grid):
            return None
    return grid if _gen_valid_grid(grid) else None


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 67385a82 input")
