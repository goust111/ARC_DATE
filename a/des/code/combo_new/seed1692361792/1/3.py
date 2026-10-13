from __future__ import annotations

from collections import deque

import numpy as np
import random


CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_PRIORITY = {direction: idx for idx, direction in enumerate(CARDINAL_DIRS)}
PERPENDICULAR_DIRS = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}


def _in_bounds(shape: tuple[int, int], row: int, col: int) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _find_divider_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = grid.shape
    candidates: list[tuple[int, int, list[int], list[int]]] = []
    for color in map(int, np.unique(grid)):
        full_rows = [row for row in range(height) if np.all(grid[row, :] == color)]
        full_cols = [col for col in range(width) if np.all(grid[:, col] == color)]
        if full_rows and full_cols:
            candidates.append((len(full_rows) + len(full_cols), color, full_rows, full_cols))
    assert candidates, "expected a divider color spanning full rows and columns"
    _score, color, rows, cols = max(candidates, key=lambda item: item[0])
    return color, rows, cols


def _partition_spans(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in cuts:
        spans.append((start, cut))
        start = cut + 1
    spans.append((start, length))
    assert all(lo < hi for lo, hi in spans), "divider lines must separate non-empty regions"
    return spans


def _infer_background_and_obstacle(grid: np.ndarray, divider: int) -> tuple[int, int]:
    values, counts = np.unique(grid[grid != divider], return_counts=True)
    assert values.size >= 3, "expected background, obstacle, and seed colors"

    bg_count = int(counts.max())
    bg_candidates = values[counts == bg_count]
    assert bg_candidates.size == 1, "background color must be unique"
    background = int(bg_candidates[0])

    fg_values = values[values != background]
    fg_counts = counts[values != background]
    obstacle_count = int(fg_counts.max())
    obstacle_candidates = fg_values[fg_counts == obstacle_count]
    assert obstacle_candidates.size == 1, "obstacle color must be uniquely most frequent non-background color"
    return background, int(obstacle_candidates[0])


def _find_seed(region: np.ndarray, background: int, obstacle: int) -> tuple[tuple[int, int], int]:
    mask = (region != background) & (region != obstacle)
    coords = np.argwhere(mask)
    assert len(coords) == 1, "each region must contain exactly one seed pixel"
    row, col = map(int, coords[0])
    return (row, col), int(region[row, col])


def _choose_turn(
    current: tuple[int, int],
    seed: tuple[int, int],
    visited: set[tuple[int, int]],
    candidates: list[tuple[int, int]],
) -> tuple[tuple[int, int], str, bool]:
    if len(candidates) == 1:
        return candidates[0], "unique", False

    unvisited = []
    for direction in candidates:
        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        if (nr, nc) not in visited:
            unvisited.append(direction)

    if len(unvisited) == 1:
        return unvisited[0], "visited", False

    pool = unvisited if len(unvisited) == 2 else candidates
    distances = []
    for direction in pool:
        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        distances.append(abs(nr - seed[0]) + abs(nc - seed[1]))

    best_distance = max(distances)
    best_dirs = [direction for direction, distance in zip(pool, distances) if distance == best_distance]
    ambiguous = len(best_dirs) > 1
    chosen = min(best_dirs, key=lambda direction: DIR_PRIORITY[direction])
    return chosen, "away", ambiguous


def _trace_ray(
    painted: np.ndarray,
    obstacle_mask: np.ndarray,
    background: int,
    seed: tuple[int, int],
    seed_color: int,
    start_dir: tuple[int, int],
    stats: dict[str, int | bool] | None,
) -> None:
    height, width = painted.shape
    current = seed
    direction = start_dir
    visited = {seed}
    bounces = 0

    while True:
        next_row = current[0] + direction[0]
        next_col = current[1] + direction[1]
        if not _in_bounds((height, width), next_row, next_col):
            return

        if obstacle_mask[next_row, next_col]:
            if bounces >= 3:
                return

            candidates = []
            for new_dir in PERPENDICULAR_DIRS[direction]:
                turn_row = current[0] + new_dir[0]
                turn_col = current[1] + new_dir[1]
                if _in_bounds((height, width), turn_row, turn_col) and not obstacle_mask[turn_row, turn_col]:
                    candidates.append(new_dir)

            if not candidates:
                return

            direction, reason, ambiguous = _choose_turn(current, seed, visited, candidates)
            bounces += 1
            if stats is not None:
                stats[reason] = int(stats[reason]) + 1
                stats["ambiguous"] = bool(stats["ambiguous"]) or ambiguous
                stats["max_bounces"] = max(int(stats["max_bounces"]), bounces)
            continue

        current = (next_row, next_col)
        visited.add(current)
        if painted[current] == background:
            painted[current] = seed_color


def _fill_enclosed_black(painted: np.ndarray, background: int, seed_color: int) -> int:
    height, width = painted.shape
    seen = np.zeros((height, width), dtype=bool)
    filled = 0

    for row in range(height):
        for col in range(width):
            if painted[row, col] != background or seen[row, col]:
                continue

            component: list[tuple[int, int]] = []
            queue = deque([(row, col)])
            seen[row, col] = True
            enclosed = True

            while queue:
                cur_row, cur_col = queue.popleft()
                component.append((cur_row, cur_col))
                for dr, dc in CARDINAL_DIRS:
                    nr, nc = cur_row + dr, cur_col + dc
                    if not _in_bounds((height, width), nr, nc):
                        continue
                    if painted[nr, nc] == background:
                        if not seen[nr, nc]:
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                    elif painted[nr, nc] != seed_color:
                        enclosed = False

            if enclosed:
                for cur_row, cur_col in component:
                    painted[cur_row, cur_col] = seed_color
                filled += len(component)

    return filled


def _solve_region(
    region: np.ndarray,
    background: int,
    obstacle: int,
    collect_stats: bool = False,
) -> tuple[np.ndarray, dict[str, int | bool] | None]:
    seed, seed_color = _find_seed(region, background, obstacle)
    painted = region.copy()
    obstacle_mask = region == obstacle

    stats: dict[str, int | bool] | None = None
    if collect_stats:
        stats = {"unique": 0, "visited": 0, "away": 0, "ambiguous": False, "max_bounces": 0, "filled": 0}

    for direction in CARDINAL_DIRS:
        _trace_ray(painted, obstacle_mask, background, seed, seed_color, direction, stats)

    filled = _fill_enclosed_black(painted, background, seed_color)
    if stats is not None:
        stats["filled"] = filled
    return painted, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each region's seed by casting bouncing orthogonal rays and then fill any black areas enclosed by the resulting path.

    Input structure:
    - The background is black.
    - Full-width and full-height maroon divider lines split the grid into equal rectangular regions.
    - Inside each region there is exactly one colored seed pixel and several grey obstacle pixels.

    Rule:
    - Process every region independently, using the divider lines and the outer grid edge as region boundaries.
    - From the seed, emit four rays in the cardinal directions; a ray paints every background cell it traverses.
    - If the next cell is a grey obstacle, the ray turns 90 degrees into an available perpendicular direction, preferring an unvisited side and otherwise the side that moves away from the seed; each ray can bounce at most three times.
    - After all rays finish, fill every 4-connected black region whose entire boundary consists only of the seed color or the region boundary.

    Key insight:
    - The divider lines first isolate independent subproblems, then each seed behaves like a local ray-tracer whose reflected path becomes the wall for a final enclosure fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider, divider_rows, divider_cols = _find_divider_color(g)
    background, obstacle = _infer_background_and_obstacle(g, divider)

    row_spans = _partition_spans(g.shape[0], divider_rows)
    col_spans = _partition_spans(g.shape[1], divider_cols)
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected multiple regions in both axes"

    row_heights = {hi - lo for lo, hi in row_spans}
    col_widths = {hi - lo for lo, hi in col_spans}
    assert len(row_heights) == 1 and len(col_widths) == 1, "regions must be equal-sized rectangles"

    out = g.copy()
    for row_lo, row_hi in row_spans:
        for col_lo, col_hi in col_spans:
            solved, _stats = _solve_region(g[row_lo:row_hi, col_lo:col_hi], background, obstacle, collect_stats=False)
            out[row_lo:row_hi, col_lo:col_hi] = solved

    return out


# --------------- generator ---------------

BACKGROUND = 0
OBSTACLE = 5
DIVIDER = 9
SEED_PALETTE = (1, 2, 3, 4, 6, 7, 8)
REGION_ROWS_OPTIONS = (2, 3)
REGION_COLS_OPTIONS = (2, 3)
REGION_H_RANGE = (7, 9)
REGION_W_RANGE = (7, 9)
GEN_RETRIES = 300

BASE_TEMPLATES: dict[str, set[tuple[int, int]]] = {
    "unique": {(0, 0), (0, 1), (0, 3), (1, 2)},
    "away": {(0, 0), (0, 1), (3, 0), (4, 1)},
    "visited": {(1, 3), (1, 4), (2, 2), (2, 5), (3, 6), (4, 4), (4, 6), (5, 4)},
}


def _transform_point(point: tuple[int, int], kind: int, size: int = 7) -> tuple[int, int]:
    row, col = point
    if kind == 0:
        return row, col
    if kind == 1:
        return col, size - 1 - row
    if kind == 2:
        return size - 1 - row, size - 1 - col
    if kind == 3:
        return size - 1 - col, row
    if kind == 4:
        return row, size - 1 - col
    if kind == 5:
        return size - 1 - row, col
    if kind == 6:
        return col, row
    if kind == 7:
        return size - 1 - col, size - 1 - row
    raise ValueError(f"unknown symmetry kind: {kind}")


def _region_template_cells(kind: str, symmetry: int, offset_row: int, offset_col: int) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    seed = _transform_point((3, 3), symmetry)
    obstacles = {_transform_point(point, symmetry) for point in BASE_TEMPLATES[kind]}
    shifted_seed = (seed[0] + offset_row, seed[1] + offset_col)
    shifted_obstacles = {(row + offset_row, col + offset_col) for row, col in obstacles}
    return shifted_seed, shifted_obstacles


def _assign_seed_colors(region_rows: int, region_cols: int) -> list[list[int]]:
    colors: list[list[int]] = [[BACKGROUND] * region_cols for _ in range(region_rows)]
    for row in range(region_rows):
        for col in range(region_cols):
            blocked = set()
            if row > 0:
                blocked.add(colors[row - 1][col])
            if col > 0:
                blocked.add(colors[row][col - 1])
            choices = [color for color in SEED_PALETTE if color not in blocked]
            colors[row][col] = random.choice(choices)
    return colors


def _build_region(kind: str, seed_color: int, region_h: int, region_w: int) -> tuple[np.ndarray, dict[str, int | bool]]:
    symmetry = random.randrange(8)
    offset_row = random.randint(0, region_h - 7)
    offset_col = random.randint(0, region_w - 7)
    seed, obstacles = _region_template_cells(kind, symmetry, offset_row, offset_col)

    region = np.full((region_h, region_w), BACKGROUND, dtype=int)
    for row, col in obstacles:
        region[row, col] = OBSTACLE
    region[seed] = seed_color

    _solved, stats = _solve_region(region, BACKGROUND, OBSTACLE, collect_stats=True)
    assert stats is not None
    return region, stats


def _target_kinds(region_count: int) -> list[str]:
    kinds = ["unique", "away", "visited"]
    while len(kinds) < region_count:
        kinds.append(random.choice(("unique", "away", "visited")))
    random.shuffle(kinds)
    return kinds


def _stats_match(kind: str, stats: dict[str, int | bool]) -> bool:
    if bool(stats["ambiguous"]) or int(stats["filled"]) <= 0:
        return False
    if kind == "unique":
        return int(stats["unique"]) >= 1 and int(stats["away"]) == 0 and int(stats["visited"]) == 0
    if kind == "away":
        return int(stats["away"]) >= 1
    if kind == "visited":
        return int(stats["visited"]) >= 1
    return False


def _build_sample() -> np.ndarray | None:
    region_rows = random.choice(REGION_ROWS_OPTIONS)
    region_cols = random.choice(REGION_COLS_OPTIONS)
    region_h = random.randint(*REGION_H_RANGE)
    region_w = random.randint(*REGION_W_RANGE)

    height = region_rows * region_h + (region_rows - 1)
    width = region_cols * region_w + (region_cols - 1)
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for row in range(1, region_rows):
        divider_row = row * region_h + (row - 1)
        grid[divider_row, :] = DIVIDER
    for col in range(1, region_cols):
        divider_col = col * region_w + (col - 1)
        grid[:, divider_col] = DIVIDER

    seed_colors = _assign_seed_colors(region_rows, region_cols)
    kinds = _target_kinds(region_rows * region_cols)
    total_stats = {"unique": 0, "away": 0, "visited": 0}

    idx = 0
    for region_row in range(region_rows):
        row_lo = region_row * (region_h + 1)
        row_hi = row_lo + region_h
        for region_col in range(region_cols):
            col_lo = region_col * (region_w + 1)
            col_hi = col_lo + region_w

            kind = kinds[idx]
            seed_color = seed_colors[region_row][region_col]
            region, stats = _build_region(kind, seed_color, region_h, region_w)
            if not _stats_match(kind, stats):
                return None

            grid[row_lo:row_hi, col_lo:col_hi] = region
            total_stats["unique"] += int(stats["unique"])
            total_stats["away"] += int(stats["away"])
            total_stats["visited"] += int(stats["visited"])
            idx += 1

    if min(total_stats.values()) <= 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_sample()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
