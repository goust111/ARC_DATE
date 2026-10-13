import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

SEED_COLORS = (BLUE, RED, GREEN, ORANGE)
SEED_GRID_TO_X = {
    (0, 0): (0, 0),
    (0, 1): (1, 1),
    (0, 2): (0, 4),
    (1, 0): (3, 1),
    (1, 1): (2, 2),
    (1, 2): (1, 3),
    (2, 0): (4, 0),
    (2, 1): (3, 3),
    (2, 2): (4, 4),
}
X_TO_SEED_GRID = {x: seed for seed, x in SEED_GRID_TO_X.items()}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _find_separator_index(grid: np.ndarray, axis: int) -> int:
    limit = grid.shape[axis]
    for idx in range(limit):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if np.all(line == GREY):
            return idx
    raise AssertionError("expected a full grey separator line")


def _fold_x_to_seed_grid(x_grid: np.ndarray) -> np.ndarray:
    seed_grid = np.zeros((3, 3), dtype=int)
    for x_pos, seed_pos in X_TO_SEED_GRID.items():
        seed_grid[seed_pos] = int(x_grid[x_pos])
    return seed_grid


def _seed_grid_to_x(seed_grid: np.ndarray) -> np.ndarray:
    x_grid = np.zeros((5, 5), dtype=int)
    for seed_pos, x_pos in SEED_GRID_TO_X.items():
        x_grid[x_pos] = int(seed_grid[seed_pos])
    return x_grid


def _wall_lines(region: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = region.shape
    wall_rows = [r for r in range(h) if int(np.count_nonzero(region[r] == TEAL)) > w // 2]
    wall_cols = [c for c in range(w) if int(np.count_nonzero(region[:, c] == TEAL)) > h // 2]
    assert len(wall_rows) == 4 and len(wall_cols) == 4, "expected a 3x3 partition wall lattice"
    return wall_rows, wall_cols


def _room_centers(region: np.ndarray) -> list[list[tuple[int, int]]]:
    wall_rows, wall_cols = _wall_lines(region)
    centers: list[list[tuple[int, int]]] = []
    for i in range(3):
        row_centers = []
        r0, r1 = wall_rows[i] + 1, wall_rows[i + 1] - 1
        for j in range(3):
            c0, c1 = wall_cols[j] + 1, wall_cols[j + 1] - 1
            assert r0 <= r1 and c0 <= c1, "room interior must be non-empty"
            row_centers.append(((r0 + r1) // 2, (c0 + c1) // 2))
        centers.append(row_centers)
    return centers


def _bfs_distances(walkable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = walkable.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = start
    assert walkable[sr, sc], "seed must lie on a walkable cell"
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _seed_specs_from_fold(seed_grid: np.ndarray, centers: list[list[tuple[int, int]]]):
    specs = []
    for r in range(3):
        for c in range(3):
            color = int(seed_grid[r, c])
            if color != BLACK:
                sr, sc = centers[r][c]
                specs.append((sr, sc, color, r, c))
    assert 2 <= len(specs) <= 4, "folded seed count must be between 2 and 4"
    return specs


def _fill_corridor(region: np.ndarray, seed_grid: np.ndarray):
    walkable = region != TEAL
    centers = _room_centers(region)
    seed_specs = _seed_specs_from_fold(seed_grid, centers)
    dist_stack = np.full((len(seed_specs),) + region.shape, -1, dtype=int)
    for idx, (sr, sc, _color, _gr, _gc) in enumerate(seed_specs):
        dist_stack[idx] = _bfs_distances(walkable, (sr, sc))

    flat = dist_stack.copy()
    flat[flat < 0] = 10**6
    best = flat.min(axis=0)
    tie = np.sum(flat == best, axis=0) > 1
    owner = flat.argmin(axis=0)

    out = region.copy()
    ownership = [0] * len(seed_specs)
    odd_ownership = [0] * len(seed_specs)
    for r in range(region.shape[0]):
        for c in range(region.shape[1]):
            if not walkable[r, c]:
                continue
            if tie[r, c]:
                out[r, c] = GREY
                continue
            idx = int(owner[r, c])
            distance = int(best[r, c])
            seed_color = int(seed_specs[idx][2])
            out[r, c] = seed_color if distance % 2 == 0 else YELLOW
            ownership[idx] += 1
            if distance % 2 == 1:
                odd_ownership[idx] += 1

    info = {
        "tie_count": int(np.count_nonzero(tie & walkable)),
        "ownership": ownership,
        "odd_ownership": odd_ownership,
        "seed_specs": seed_specs,
        "centers": centers,
    }
    return out, info


def _connected_walkable(region: np.ndarray) -> bool:
    walkable = region != TEAL
    cells = np.argwhere(walkable)
    if len(cells) == 0:
        return False
    start = tuple(map(int, cells[0]))
    dist = _bfs_distances(walkable, start)
    return int(np.count_nonzero(dist >= 0)) == int(np.count_nonzero(walkable))


def _compose_grid(seed_grid: np.ndarray, region: np.ndarray) -> np.ndarray:
    x_grid = _seed_grid_to_x(seed_grid)
    h, w = region.shape
    grid = np.zeros((h + 6, w + 6), dtype=int)
    grid[:5, :5] = x_grid
    grid[5, :] = GREY
    grid[:, 5] = GREY
    grid[6:, 6:] = region
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the corner X into a 3x3 seed map, then use it to alternately flood-color the corridor lattice.

    Input structure:
    - A 5x5 diagonal-X pattern sits in the upper-left corner, using only diagonal cells.
    - A full grey separator row and a full grey separator column isolate that corner pattern from the corridor area.
    - The lower-right area is a 3x3 partitioned corridor grid: teal walls form the room lattice and black cells are walkable corridors.

    Rule:
    - Fold the 5x5 X into a 3x3 grid by mapping the northwest arm to the top edge, the northeast arm to the right edge, the southwest arm to the left edge, the southeast arm to the bottom edge, and the center to the center.
    - Every non-black cell in that folded 3x3 grid becomes a seed whose room position is the center of the corresponding corridor partition.
    - Flood the corridor by 4-neighbor BFS from all seeds: even distances take the seed color, odd distances become yellow, shorter paths win, and equal-distance conflicts become grey.
    - Leave the teal walls, the original X pattern, and the separator lines unchanged.

    Key insight:
    - The visible X is an encoded seed-configuration file: only after folding it into the correct 3x3 room-coordinate system do the corridor seed colors and positions become known.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_r = _find_separator_index(g, axis=0)
    sep_c = _find_separator_index(g, axis=1)
    assert sep_r == 5 and sep_c == 5, "expected the 5x5 X to be separated by one grey row and one grey column"

    x_grid = g[:sep_r, :sep_c]
    assert x_grid.shape == (5, 5), "expected a 5x5 corner X pattern"
    region = g[sep_r + 1 :, sep_c + 1 :]
    assert region.size > 0, "expected a corridor region below-right of the separators"

    seed_grid = _fold_x_to_seed_grid(x_grid)
    filled_region, _info = _fill_corridor(region, seed_grid)

    out = g.copy()
    out[sep_r + 1 :, sep_c + 1 :] = filled_region
    return out


def _build_region(row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    h = sum(row_heights) + 4
    w = sum(col_widths) + 4
    region = np.full((h, w), TEAL, dtype=int)

    row_ranges = []
    cursor = 1
    for room_h in row_heights:
        row_ranges.append((cursor, cursor + room_h - 1))
        cursor += room_h + 1

    col_ranges = []
    cursor = 1
    for room_w in col_widths:
        col_ranges.append((cursor, cursor + room_w - 1))
        cursor += room_w + 1

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            region[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    for i in range(3):
        for j in range(2):
            wall_c = col_ranges[j][1] + 1
            door_r = random.randint(row_ranges[i][0], row_ranges[i][1])
            region[door_r, wall_c] = BLACK

    for i in range(2):
        for j in range(3):
            wall_r = row_ranges[i][1] + 1
            door_c = random.randint(col_ranges[j][0], col_ranges[j][1])
            region[wall_r, door_c] = BLACK

    return region


def _sample_seed_grid() -> np.ndarray:
    seed_grid = np.zeros((3, 3), dtype=int)
    seed_count = random.randint(2, 4)
    positions = random.sample([(r, c) for r in range(3) for c in range(3)], seed_count)
    colors = [random.choice(SEED_COLORS) for _ in range(seed_count)]
    if len(set(colors)) == 1:
        colors[-1] = random.choice([c for c in SEED_COLORS if c != colors[0]])
    for (r, c), color in zip(positions, colors):
        seed_grid[r, c] = color
    return seed_grid


def _validate_instance(grid: np.ndarray) -> bool:
    sep_r = _find_separator_index(grid, axis=0)
    sep_c = _find_separator_index(grid, axis=1)
    x_grid = grid[:sep_r, :sep_c]
    region = grid[sep_r + 1 :, sep_c + 1 :]
    folded = _fold_x_to_seed_grid(x_grid)

    if not _connected_walkable(region):
        return False

    try:
        filled_region, info = _fill_corridor(region, folded)
    except AssertionError:
        return False

    if info["tie_count"] < 2:
        return False
    if any(count < 6 for count in info["ownership"]):
        return False
    if any(count == 0 for count in info["odd_ownership"]):
        return False

    out = main(grid)
    if not np.array_equal(out[sep_r + 1 :, sep_c + 1 :], filled_region):
        return False
    if not np.array_equal(out[: sep_r + 1, :], grid[: sep_r + 1, :]):
        return False
    if not np.array_equal(out[:, : sep_c + 1], grid[:, : sep_c + 1]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        seed_grid = _sample_seed_grid()
        row_heights = [random.choice((3, 5)) for _ in range(3)]
        col_widths = [random.choice((3, 5)) for _ in range(3)]
        region = _build_region(row_heights, col_widths)
        grid = _compose_grid(seed_grid, region)

        if max(grid.shape) > 30:
            continue
        if not np.array_equal(_fold_x_to_seed_grid(grid[:5, :5]), seed_grid):
            continue
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
