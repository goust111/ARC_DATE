import numpy as np


GRID_SIZE = 5
BACKGROUND = 0
FOREGROUND = 1
INVERTED_FILL = 2
MIN_CELLS = 8
MAX_CELLS = 15
GENERATION_RETRIES = 2000


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(r: int, c: int, size: int = GRID_SIZE) -> bool:
    return 0 <= r < size and 0 <= c < size


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        stack = [(rr, cc)]
        visited[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(nr, nc, mask.shape[0]) and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _clockwise_rotations(grid: np.ndarray) -> list[np.ndarray]:
    return [
        grid.copy(),
        np.rot90(grid, -1).copy(),
        np.rot90(grid, 2).copy(),
        np.rot90(grid, 1).copy(),
    ]


def _invert_occupancy(grid: np.ndarray) -> np.ndarray:
    out = np.full(grid.shape, INVERTED_FILL, dtype=int)
    out[grid == FOREGROUND] = BACKGROUND
    return out


def _processed_variant(grid: np.ndarray) -> np.ndarray:
    top_count = int(np.count_nonzero(grid[0] == FOREGROUND))
    if top_count % 2 == 1:
        return _invert_occupancy(grid)
    return grid.copy()


def _compose_windmill(quadrants: list[np.ndarray]) -> np.ndarray:
    top = np.hstack([quadrants[0], quadrants[1]])
    bottom = np.hstack([quadrants[3], quadrants[2]])
    return np.vstack([top, bottom])


def _signature(grid: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in grid.tolist())


def _edge_counts(grid: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.count_nonzero(grid[0] == FOREGROUND)),
        int(np.count_nonzero(grid[:, 0] == FOREGROUND)),
        int(np.count_nonzero(grid[-1] == FOREGROUND)),
        int(np.count_nonzero(grid[:, -1] == FOREGROUND)),
    )


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if not np.isin(grid, [BACKGROUND, FOREGROUND]).all():
        return False

    cell_count = int(np.count_nonzero(grid == FOREGROUND))
    if not (MIN_CELLS <= cell_count <= MAX_CELLS):
        return False

    if len(_components_mask(grid == FOREGROUND)) != 1:
        return False

    rotations = _clockwise_rotations(grid)
    if len({_signature(rot) for rot in rotations}) != 4:
        return False
    if np.array_equal(rotations[0], rotations[2]):
        return False

    edge_counts = _edge_counts(grid)
    touched_edges = sum(count > 0 for count in edge_counts)
    if touched_edges < 2:
        return False

    parities = [int(np.count_nonzero(rot[0] == FOREGROUND)) % 2 for rot in rotations]
    if set(parities) != {0, 1}:
        return False

    processed = [_processed_variant(rot) for rot in rotations]
    mixed_output = _compose_windmill(processed)
    if np.array_equal(mixed_output, _compose_windmill(rotations)):
        return False
    if np.array_equal(
        mixed_output,
        _compose_windmill([_invert_occupancy(rot) for rot in rotations]),
    ):
        return False
    return True


def _random_connected_shape(cell_count: int) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    start = tuple(np.random.randint(0, GRID_SIZE, size=2))
    cells = {start}

    while len(cells) < cell_count:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(nr, nc) and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return grid

        unique_frontier = list(dict.fromkeys(frontier))
        if np.random.rand() < 0.5:
            weights = np.array(
                [1 + (r in (0, GRID_SIZE - 1)) + (c in (0, GRID_SIZE - 1)) for r, c in unique_frontier],
                dtype=float,
            )
            weights /= weights.sum()
            idx = int(np.random.choice(len(unique_frontier), p=weights))
        else:
            idx = int(np.random.randint(len(unique_frontier)))
        cells.add(unique_frontier[idx])

    for r, c in cells:
        grid[r, c] = FOREGROUND
    return grid


def generate_input() -> np.ndarray:
    target_odd_counts = (1, 2, 3)
    target_touched_edges = (2, 3, 4)

    for _ in range(GENERATION_RETRIES):
        cell_count = int(np.random.randint(MIN_CELLS, MAX_CELLS + 1))
        grid = _random_connected_shape(cell_count)

        if int(np.count_nonzero(grid == FOREGROUND)) != cell_count:
            continue

        odd_count_target = int(np.random.choice(target_odd_counts))
        touched_edges_target = int(np.random.choice(target_touched_edges))
        edge_counts = _edge_counts(grid)
        if sum(count % 2 for count in edge_counts) != odd_count_target:
            continue
        if sum(count > 0 for count in edge_counts) < touched_edges_target:
            continue
        if _validate_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a 2x2 windmill from the input sprite's four rotations, inverting only rotations whose top row has odd foreground count.

    Input structure:
    - The input is a square black-background sprite made of black(0) and blue(1) cells.
    - The intended training distribution uses a 5x5 sprite with about 8 to 15 blue cells.
    - The sprite is not 90-degree or 180-degree rotation-symmetric, and the four rotations do not all share the same top-row parity.

    Rule:
    - Generate the four rotations R0, R1, R2, R3 of the input, where R1/R3 are clockwise 90/270 degrees.
    - For each rotated variant, count blue cells in its top row.
    - If that top-row count is odd, replace blue with black and replace black with red(2); if it is even, leave the variant unchanged.
    - Assemble the processed variants into a 2x2 output: top-left R0, top-right R1, bottom-left R3, bottom-right R2.

    Key insight:
    - The inversion condition is tied to each rotated copy's own top row, so rotation changes both appearance and whether that quadrant is inverted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    assert max(g.shape) * 2 <= 30, "output would exceed ARC size limit"
    assert np.isin(g, [BACKGROUND, FOREGROUND]).all(), "expected a binary black/blue sprite"

    rotations = _clockwise_rotations(g)
    processed = [_processed_variant(rot) for rot in rotations]
    return _compose_windmill(processed)
