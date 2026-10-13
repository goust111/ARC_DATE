from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color


BACKGROUND = Color.BLACK
MARKER = Color.BLUE
TERRAIN = Color.GREEN
HIGHLIGHT = Color.YELLOW
OBSTACLE = Color.GREY

GRID_HEIGHT_RANGE = (15, 20)
GRID_WIDTH_RANGE = (15, 20)
MARKER_COUNT_RANGE = (3, 5)
TERRAIN_HEIGHT_RANGE = (4, 8)
GEN_RETRIES = 500


def _terrain_surface_rows(grid: np.ndarray) -> np.ndarray:
    h, w = map(int, grid.shape)
    surfaces = np.empty(w, dtype=int)
    for c in range(w):
        terrain_rows = np.flatnonzero(grid[:, c] == TERRAIN)
        assert terrain_rows.size > 0, "each column must contain terrain"
        surface = int(terrain_rows[0])
        assert surface > 0, "terrain surface must leave room for the marker"
        assert np.all(grid[surface:, c] == TERRAIN), "terrain must be a contiguous column suffix"
        assert not np.any(grid[:surface, c] == TERRAIN), "terrain cannot float above its surface"
        surfaces[c] = surface
    assert np.all(grid[h - 1, :] == TERRAIN), "bottom row must be all terrain"
    return surfaces


def _marker_rows_by_col(grid: np.ndarray, surfaces: np.ndarray) -> dict[int, int]:
    markers: dict[int, int] = {}
    for r0, c0 in np.argwhere(grid == MARKER):
        r, c = int(r0), int(c0)
        assert c not in markers, "expected at most one marker per column"
        assert r == int(surfaces[c]) - 1, "each marker must sit exactly one cell above its terrain surface"
        markers[c] = r

    cols = sorted(markers)
    assert 3 <= len(cols) <= 5, "expected 3-5 marker columns"
    assert all(cols[i + 1] - cols[i] >= 3 for i in range(len(cols) - 1)), "marker columns must be well separated"
    surface_rows = [int(surfaces[c]) for c in cols]
    assert len(set(surface_rows)) == len(surface_rows), "marker columns must have distinct surface heights"
    return markers


def _obstacle_positions(grid: np.ndarray, surfaces: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    positions = [(int(r), int(c)) for r, c in np.argwhere(grid == OBSTACLE)]
    for r, c in positions:
        assert 0 < r < h - 1, "obstacles cannot sit on the top or bottom border"
        assert 0 < c < w - 1, "obstacles cannot sit on the left or right border"
        assert r < int(surfaces[c]), "obstacles must stay above the terrain surface"
    for i, (r0, c0) in enumerate(positions):
        for r1, c1 in positions[i + 1 :]:
            assert max(abs(r0 - r1), abs(c0 - c1)) > 1, "obstacles must be mutually non-adjacent"
    return positions


def _deepest_marker_col(marker_rows: dict[int, int], surfaces: np.ndarray) -> int:
    cols = sorted(marker_rows)
    surface_rows = [int(surfaces[c]) for c in cols]
    deepest_surface = max(surface_rows)
    deepest_cols = [c for c in cols if int(surfaces[c]) == deepest_surface]
    assert len(deepest_cols) == 1, "expected a unique deepest marker column"
    return int(deepest_cols[0])


def _simulate_rays(grid: np.ndarray) -> tuple[np.ndarray, dict[int, dict[str, object]], dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    surfaces = _terrain_surface_rows(g)
    marker_rows = _marker_rows_by_col(g, surfaces)
    _obstacle_positions(g, surfaces)

    out = g.copy()
    occupied: set[tuple[int, int]] = set()
    ray_infos: dict[int, dict[str, object]] = {}
    obstacle_detours = 0
    ray_detours = 0
    clear_up_moves = True
    clear_landings = True

    for start_col in sorted(marker_rows):
        r = int(marker_rows[start_col])
        c = int(start_col)
        path = [(r, c)]
        detours: list[tuple[str, tuple[int, int]]] = []
        occupied.add((r, c))
        out[r, c] = MARKER

        while r > 0:
            above = (r - 1, c)
            if above in occupied:
                cause = "ray"
            elif int(g[above]) == OBSTACLE:
                cause = "obstacle"
            else:
                cause = None

            if cause is None:
                if int(g[above]) != BACKGROUND:
                    clear_up_moves = False
                r -= 1
                path.append((r, c))
                occupied.add((r, c))
                out[r, c] = MARKER
                continue

            if cause == "obstacle":
                obstacle_detours += 1
            else:
                ray_detours += 1

            if c + 1 >= w:
                break

            landing = (r, c + 1)
            if landing in occupied or int(g[landing]) != BACKGROUND:
                clear_landings = False
            c += 1
            path.append((r, c))
            detours.append((cause, (r, c)))
            occupied.add((r, c))
            out[r, c] = MARKER

        ray_infos[start_col] = {
            "start_row": int(marker_rows[start_col]),
            "path": path,
            "detours": detours,
            "obstacle_detours": sum(1 for cause, _ in detours if cause == "obstacle"),
            "ray_detours": sum(1 for cause, _ in detours if cause == "ray"),
        }

    deepest_col = _deepest_marker_col(marker_rows, surfaces)
    for r, c in ray_infos[deepest_col]["path"]:
        out[r, c] = HIGHLIGHT

    stats = {
        "surfaces": surfaces,
        "marker_rows": marker_rows,
        "deepest_col": deepest_col,
        "obstacle_detours": obstacle_detours,
        "ray_detours": ray_detours,
        "clear_up_moves": clear_up_moves,
        "clear_landings": clear_landings,
    }
    return out, ray_infos, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire upward surface rays that sidestep right around obstacles, then recolor the deepest marker ray.

    Input structure:
    - The grid has a black background with a green terrain suffix in every column, so each column has one visible surface cell.
    - A small set of blue marker pixels sits exactly one cell above selected terrain surface columns.
    - Grey single-cell obstacles float above the terrain and are separated from each other and from the side borders.

    Rule:
    - For each marker column, processed left to right, start a ray at the marker cell and repeatedly try to move one cell upward.
    - If the cell above is a grey obstacle or a previously drawn ray cell, detour one step to the right instead and continue from there; if the right step would leave the grid, stop that ray.
    - Otherwise move upward and paint the visited cell blue; after all rays are drawn, recolor the full ray from the deepest marked terrain column yellow.

    Key insight:
    - The terrain profile supplies each ray's starting height, while obstacle hits and left-to-right ray precedence decide where detours occur before the deepest marked column is highlighted.
    """
    out, _, _ = _simulate_rays(np.asarray(grid, dtype=int))
    return out


def _gen_terrain_heights(height: int, width: int) -> np.ndarray | None:
    min_h = TERRAIN_HEIGHT_RANGE[0]
    max_h = min(TERRAIN_HEIGHT_RANGE[1], height - 3)
    if max_h < min_h:
        return None

    for _ in range(120):
        heights = np.empty(width, dtype=int)
        heights[0] = random.randint(min_h, max_h)
        for c in range(1, width):
            delta = random.choices([-2, -1, 0, 1, 2], weights=[1, 3, 4, 3, 1])[0]
            heights[c] = int(np.clip(heights[c - 1] + delta, min_h, max_h))
        if len(set(map(int, heights))) >= 4 and int(heights.max() - heights.min()) >= 2:
            return heights
    return None


def _base_grid(height: int, width: int, terrain_heights: np.ndarray) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for c, terrain_height in enumerate(terrain_heights.tolist()):
        grid[height - int(terrain_height) :, c] = TERRAIN
    return grid


def _choose_marker_cols(surfaces: np.ndarray, include_collision: bool) -> tuple[int, ...] | None:
    width = len(surfaces)
    candidates = list(range(1, width - 2))
    random.shuffle(candidates)
    count_choices = list(range(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1))
    random.shuffle(count_choices)

    for marker_count in count_choices:
        valid_sets = []
        for cols in combinations(sorted(candidates), marker_count):
            if any(cols[i + 1] - cols[i] < 3 for i in range(marker_count - 1)):
                continue
            surface_rows = [int(surfaces[c]) for c in cols]
            if len(set(surface_rows)) != marker_count:
                continue
            if include_collision and not any(3 <= cols[i + 1] - cols[i] <= 5 for i in range(marker_count - 1)):
                continue
            valid_sets.append(cols)
        if valid_sets:
            return random.choice(valid_sets)
    return None


def _place_markers(grid: np.ndarray, marker_cols: tuple[int, ...]) -> None:
    surfaces = _terrain_surface_rows(grid)
    for c in marker_cols:
        grid[int(surfaces[c]) - 1, int(c)] = MARKER


def _positions_can_hold_obstacles(grid: np.ndarray, positions: list[tuple[int, int]]) -> bool:
    surfaces = _terrain_surface_rows(grid)
    occupied = {(int(r), int(c)) for r, c in np.argwhere(grid != BACKGROUND)}
    placed: list[tuple[int, int]] = []
    h, w = map(int, grid.shape)

    for r, c in positions:
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return False
        if (r, c) in occupied:
            return False
        if r >= int(surfaces[c]):
            return False
        for rr, cc in placed:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
        for rr, cc in np.argwhere(grid == OBSTACLE):
            if max(abs(r - int(rr)), abs(c - int(cc))) <= 1:
                return False
        placed.append((r, c))
    return True


def _find_collision_chain(grid: np.ndarray) -> list[tuple[int, int]] | None:
    surfaces = _terrain_surface_rows(grid)
    marker_rows = _marker_rows_by_col(grid, surfaces)
    cols = sorted(marker_rows)
    pair_order = list(range(len(cols) - 1))
    random.shuffle(pair_order)

    for idx in pair_order:
        left_col, right_col = cols[idx], cols[idx + 1]
        gap = right_col - left_col
        if not 3 <= gap <= 5:
            continue

        start_row = int(marker_rows[left_col])
        obstacles = [(start_row - (2 * step + 1), left_col + step) for step in range(gap)]
        if not _positions_can_hold_obstacles(grid, obstacles):
            continue

        trial = grid.copy()
        for r, c in obstacles:
            trial[r, c] = OBSTACLE
        try:
            _, ray_infos, stats = _simulate_rays(trial)
        except AssertionError:
            continue

        if not stats["clear_up_moves"] or not stats["clear_landings"]:
            continue
        if ray_infos[left_col]["obstacle_detours"] < gap:
            continue
        if ray_infos[right_col]["ray_detours"] < 1:
            continue
        return obstacles
    return None


def _find_single_detour_obstacle(grid: np.ndarray) -> tuple[int, int] | None:
    surfaces = _terrain_surface_rows(grid)
    marker_rows = _marker_rows_by_col(grid, surfaces)
    candidates: list[tuple[int, int]] = []

    for col in sorted(marker_rows):
        start_row = int(marker_rows[col])
        for r in range(start_row, 1, -1):
            obstacle = (r - 1, col)
            landing = (r, col + 1)
            if col + 1 >= grid.shape[1]:
                continue
            if not _positions_can_hold_obstacles(grid, [obstacle]):
                continue
            if int(grid[landing]) != BACKGROUND or landing[0] >= int(surfaces[landing[1]]):
                continue

            trial = grid.copy()
            trial[obstacle] = OBSTACLE
            try:
                _, ray_infos, stats = _simulate_rays(trial)
            except AssertionError:
                continue
            if not stats["clear_up_moves"] or not stats["clear_landings"]:
                continue
            if ray_infos[col]["obstacle_detours"] >= 1:
                candidates.append(obstacle)

    if not candidates:
        return None
    return random.choice(candidates)


def _add_distractor_obstacles(grid: np.ndarray, target_count: int) -> bool:
    out, ray_infos, _ = _simulate_rays(grid)
    del out  # only the traced cells matter here

    ray_cells = set()
    for info in ray_infos.values():
        ray_cells.update(info["path"])

    forbidden = set(ray_cells)
    for r, c in list(ray_cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((r + dr, c + dc))

    marker_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == MARKER)}
    for r, c in marker_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((r + dr, c + dc))

    surfaces = _terrain_surface_rows(grid)
    obstacle_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == OBSTACLE)}
    candidates = []
    h, w = map(int, grid.shape)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BACKGROUND:
                continue
            if r >= int(surfaces[c]):
                continue
            if (r, c) in forbidden:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in obstacle_cells):
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    for r, c in candidates:
        if len(obstacle_cells) >= target_count:
            break
        grid[r, c] = OBSTACLE
        obstacle_cells.add((r, c))

    return len(obstacle_cells) >= target_count


def _validate_case(grid: np.ndarray, require_collision: bool) -> bool:
    try:
        expected, ray_infos, stats = _simulate_rays(grid)
    except AssertionError:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if not stats["clear_up_moves"] or not stats["clear_landings"]:
        return False
    if stats["obstacle_detours"] < 1:
        return False
    if require_collision and stats["ray_detours"] < 1:
        return False
    if np.array_equal(grid, expected):
        return False
    if not np.any(expected == HIGHLIGHT):
        return False
    if not np.any(expected == MARKER):
        return False
    if not np.all(expected[grid == TERRAIN] == TERRAIN):
        return False
    if not np.all(expected[grid == OBSTACLE] == OBSTACLE):
        return False

    deepest_col = int(stats["deepest_col"])
    deepest_path = set(ray_infos[deepest_col]["path"])
    for r, c in deepest_path:
        if int(expected[r, c]) != HIGHLIGHT:
            return False
    for col, info in ray_infos.items():
        if col == deepest_col:
            continue
        for r, c in info["path"]:
            if int(expected[r, c]) != MARKER:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    terrain_heights = _gen_terrain_heights(height, width)
    if terrain_heights is None:
        return None

    grid = _base_grid(height, width, terrain_heights)
    include_collision = random.random() < 0.75

    surfaces = _terrain_surface_rows(grid)
    marker_cols = _choose_marker_cols(surfaces, include_collision)
    if marker_cols is None:
        return None
    _place_markers(grid, marker_cols)

    if include_collision:
        collision_chain = _find_collision_chain(grid)
        if collision_chain is None:
            return None
        for r, c in collision_chain:
            grid[r, c] = OBSTACLE

    if np.count_nonzero(grid == OBSTACLE) == 0:
        simple_obstacle = _find_single_detour_obstacle(grid)
        if simple_obstacle is None:
            return None
        grid[simple_obstacle] = OBSTACLE
    elif random.random() < 0.5:
        simple_obstacle = _find_single_detour_obstacle(grid)
        if simple_obstacle is not None:
            grid[simple_obstacle] = OBSTACLE

    target_obstacles = random.randint(max(5, int(np.count_nonzero(grid == OBSTACLE))), 10)
    if not _add_distractor_obstacles(grid, target_obstacles):
        return None
    if not _validate_case(grid, require_collision=include_collision):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid terrain-ray detour input")
