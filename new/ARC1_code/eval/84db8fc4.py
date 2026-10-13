import numpy as np

from arc_utils import components_mask


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each black region red if it reaches the grid edge and grey if it is enclosed.

    Input structure
    - The grid contains black background gaps plus blue and green foreground barriers.
    - Black pixels form multiple 4-connected regions; some touch the outer border and some are enclosed.
    Rule
    - Preserve every non-black pixel.
    - Split the black pixels into 4-connected regions.
    - Recolor an edge-touching black region red, and recolor an enclosed black region grey.
    Key insight
    - The output colors distinguish outside-connected background from topological holes; the blue and green barriers are unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)) == {0, 1, 3}

    out = g.copy()
    components = components_mask(g == 0, connectivity=4)
    assert components

    edge_components = 0
    enclosed_components = 0
    for coords in components:
        assert coords.ndim == 2 and coords.shape[1] == 2 and len(coords) > 0
        touches_edge = _touches_edge(coords, g.shape)
        fill = 2 if touches_edge else 5
        out[coords[:, 0], coords[:, 1]] = fill
        edge_components += int(touches_edge)
        enclosed_components += int(not touches_edge)

    assert edge_components > 0
    assert enclosed_components > 0
    return out

# --------------- generator (o5x) ---------------
# Canvas, colors, and retry budget.
GRID_SHAPE, GRID_CELLS = (10, 10), 10 * 10
BLUE, GREEN, EDGE_FILL, ENCLOSED_FILL = 1, 3, 2, 5
MAX_GENERATION_ATTEMPTS = 300
# Barrier sampling and component constraints.
BARRIER_COUNT_RANGE = (54, 66)  # np.random.randint upper bound is exclusive.
BLUE_COUNT_RANGE, GREEN_COUNT_RANGE = (24, 36), (23, 39)
BARRIER_COMPONENT_RANGE, MIN_LARGEST_BARRIER = (3, 10), 24
# Black-region count and size constraints.
BLACK_COMPONENT_RANGE = (11, 18)
EDGE_REGION_RANGE, ENCLOSED_REGION_RANGE = (7, 9), (2, 10)
MAX_EDGE_REGION_SIZE, MAX_ENCLOSED_REGION_SIZE = 12, 14


def _gen_touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))


def _range_ok(n: int, bounds: tuple[int, int]) -> bool:
    lo, hi = bounds
    return lo <= n <= hi


def _split_regions_by_edge(regions: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Partition component coords into edge-touching and enclosed regions."""
    edge_regions = []
    enclosed_regions = []
    for coords in regions:
        target = edge_regions if _gen_touches_edge(coords, GRID_SHAPE) else enclosed_regions
        target.append(coords)
    return edge_regions, enclosed_regions


def _gen_attempt() -> np.ndarray | None:
    barrier_count = int(np.random.randint(*BARRIER_COUNT_RANGE))
    flat = np.zeros(GRID_CELLS, dtype=bool)
    flat[np.random.choice(GRID_CELLS, size=barrier_count, replace=False)] = True
    barrier = flat.reshape(GRID_SHAPE)

    black_regions = components_mask(~barrier, connectivity=4)
    if not _range_ok(len(black_regions), BLACK_COMPONENT_RANGE):
        return None
    edge_regions, enclosed_regions = _split_regions_by_edge(black_regions)
    if not (_range_ok(len(edge_regions), EDGE_REGION_RANGE) and _range_ok(len(enclosed_regions), ENCLOSED_REGION_RANGE)):
        return None
    if any(len(coords) > MAX_EDGE_REGION_SIZE for coords in edge_regions) or any(len(coords) > MAX_ENCLOSED_REGION_SIZE for coords in enclosed_regions):
        return None

    barrier_regions = components_mask(barrier, connectivity=4)
    if not _range_ok(len(barrier_regions), BARRIER_COMPONENT_RANGE) or max(len(coords) for coords in barrier_regions) < MIN_LARGEST_BARRIER:
        return None

    blue_min = max(BLUE_COUNT_RANGE[0], barrier_count - GREEN_COUNT_RANGE[1])
    blue_max = min(BLUE_COUNT_RANGE[1], barrier_count - GREEN_COUNT_RANGE[0])
    if blue_min > blue_max:
        return None
    blue_count = int(np.random.randint(blue_min, blue_max + 1))

    out = np.zeros(GRID_SHAPE, dtype=int)
    barrier_coords = np.argwhere(barrier)
    order = np.random.permutation(len(barrier_coords))
    blue = barrier_coords[order[:blue_count]]
    green = barrier_coords[order[blue_count:]]
    out[blue[:, 0], blue[:, 1]] = BLUE
    out[green[:, 0], green[:, 1]] = GREEN

    for regions, color in ((edge_regions, EDGE_FILL), (enclosed_regions, ENCLOSED_FILL)):
        for coords in regions:
            out[coords[:, 0], coords[:, 1]] = color

    inp = out.copy()
    inp[(inp == EDGE_FILL) | (inp == ENCLOSED_FILL)] = 0
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise ValueError("failed to generate valid 84db8fc4 input")
