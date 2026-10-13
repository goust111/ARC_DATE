from __future__ import annotations

import random

import numpy as np


BG = 0
WAYPOINT = 5
SEED_PALETTE = (1, 2, 4)
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _seed_regions(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in {int(bg), WAYPOINT})
    seeds: list[dict] = []
    for color in colors:
        for coords in _components_mask(g == int(color)):
            scan = tuple(map(int, coords[np.lexsort((coords[:, 1], coords[:, 0]))][0]))
            seeds.append({"color": int(color), "coords": coords, "scan": scan})
    seeds.sort(key=lambda item: item["scan"])
    return seeds


def _segment_path(
    start: tuple[int, int], end: tuple[int, int], *, horizontal_first: bool
) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    cells: list[tuple[int, int]] = []

    if horizontal_first:
        dc = 1 if c1 >= c0 else -1
        for col in range(c0, c1 + dc, dc):
            cells.append((r0, int(col)))
        if r0 != r1:
            dr = 1 if r1 >= r0 else -1
            for row in range(r0 + dr, r1 + dr, dr):
                cells.append((int(row), c1))
    else:
        dr = 1 if r1 >= r0 else -1
        for row in range(r0, r1 + dr, dr):
            cells.append((int(row), c0))
        if c0 != c1:
            dc = 1 if c1 >= c0 else -1
            for col in range(c0 + dc, c1 + dc, dc):
                cells.append((r1, int(col)))
    return cells


def _segment_seed_distance(path: list[tuple[int, int]], seed_coords: np.ndarray) -> int:
    path_arr = np.array(path, dtype=int)
    diffs = np.abs(path_arr[:, None, :] - seed_coords[None, :, :]).sum(axis=2)
    return int(diffs.min())


def _solve_details(grid: np.ndarray, *, horizontal_first: bool) -> tuple[np.ndarray, list[dict], list[dict], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(_get_bg(g))
    waypoints = [tuple(map(int, coord)) for coord in np.argwhere(g == WAYPOINT)]
    waypoints.sort()
    assert 4 <= len(waypoints) <= 6, "expected 4-6 waypoint pixels"

    seeds = _seed_regions(g, bg)
    assert 2 <= len(seeds) <= 3, "expected 2-3 seed regions"

    seed_mask = (g != int(bg)) & (g != WAYPOINT)
    overlay: dict[tuple[int, int], tuple[int, int, int]] = {}
    infos: list[dict] = []

    for index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        path = _segment_path(start, end, horizontal_first=horizontal_first)
        choices = []
        for seed_index, seed in enumerate(seeds):
            dist = _segment_seed_distance(path, seed["coords"])
            choices.append((int(dist), int(seed_index), int(seed["color"])))
        dist, seed_index, color = min(choices, key=lambda item: (item[0], item[1]))
        infos.append(
            {
                "index": int(index),
                "path": path,
                "distance": int(dist),
                "seed_index": int(seed_index),
                "color": int(color),
            }
        )
        for cell in path:
            if seed_mask[cell]:
                continue
            cur = overlay.get(cell)
            cand = (int(dist), int(seed_index), int(color))
            if cur is None or cand[:2] < cur[:2]:
                overlay[cell] = cand

    out = g.copy()
    for (row, col), (_dist, _seed_index, color) in overlay.items():
        out[row, col] = int(color)
    out[seed_mask] = g[seed_mask]
    return out, infos, seeds, waypoints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw shortest orthogonal paths through the ordered gray waypoints, then recolor each path segment by its nearest seed region.

    Input structure:
    - The grid has black background, 4-6 isolated gray waypoint pixels, and 2-3 small colored seed regions.
    - Each seed region is a 4-connected 2-3 pixel shape using a distinct seed color.
    - Waypoints are interpreted in scan order, and the seed regions are obstacles only for the path placement constraints.

    Rule:
    - Order the gray waypoints from top to bottom, then left to right.
    - For each consecutive waypoint pair, draw the Manhattan-shortest orthogonal path that moves horizontally to the target column first and then vertically to the target row.
    - For each whole path segment, compute its minimum Manhattan distance to each seed region and recolor the entire segment with the nearest seed's color; if a pixel belongs to two segments, keep the color of the segment with the smaller segment-to-seed distance, breaking ties by earlier seed scan order.
    - Preserve every seed region exactly as it appears in the input.

    Key insight:
    - The hidden convention is the L-path orientation: using horizontal-then-vertical changes which seed is closest to each segment, so the path must be drawn before its colors can be determined.
    """
    out, _infos, _seeds, _waypoints = _solve_details(grid, horizontal_first=True)
    return out


def _spaced_positions(lo: int, hi: int, count: int, min_gap: int, rng: random.Random) -> list[int] | None:
    span = hi - lo + 1
    needed = count + (count - 1) * min_gap
    if span < needed:
        return None
    compressed = span - (count - 1) * min_gap
    picks = sorted(rng.sample(range(compressed), count))
    return [int(lo + picks[i] + i * min_gap) for i in range(count)]


def _seed_cells_ok(
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
    waypoint_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    height, width = map(int, shape)
    if len(set(cells)) != len(cells):
        return False
    for row, col in cells:
        if not (0 <= row < height and 0 <= col < width):
            return False
        if (row, col) in occupied:
            return False
        for wr, wc in waypoint_set:
            if abs(row - wr) + abs(col - wc) <= 1:
                return False
        for orow, ocol in occupied:
            if abs(row - orow) + abs(col - ocol) <= 1:
                return False
    return True


def _make_horizontal_seed(row: int, center_col: int, direction: int, rng: random.Random) -> list[tuple[int, int]]:
    if rng.random() < 0.5:
        return [(int(row), int(center_col)), (int(row), int(center_col + direction))]
    return [
        (int(row), int(center_col - direction)),
        (int(row), int(center_col)),
        (int(row), int(center_col + direction)),
    ]


def _make_vertical_seed(bottom_row: int, col: int, rng: random.Random) -> list[tuple[int, int]]:
    if rng.random() < 0.5:
        return [(int(bottom_row - 1), int(col)), (int(bottom_row), int(col))]
    return [(int(bottom_row - 2), int(col)), (int(bottom_row - 1), int(col)), (int(bottom_row), int(col))]


def _make_bridge_seed(bottom_row: int, col: int) -> list[tuple[int, int]]:
    return [(int(bottom_row - 1), int(col)), (int(bottom_row), int(col))]


def _semantic_ok(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if int(np.count_nonzero(g == WAYPOINT)) < 4:
        return False

    try:
        out_h, infos_h, seeds_h, waypoints_h = _solve_details(g, horizontal_first=True)
        out_v, infos_v, _seeds_v, _waypoints_v = _solve_details(g, horizontal_first=False)
    except Exception:
        return False

    if np.array_equal(out_h, out_v):
        return False
    if infos_h[0]["seed_index"] == infos_v[0]["seed_index"]:
        return False

    waypoint_set = set(waypoints_h)
    seed_cells = {
        tuple(map(int, coord))
        for seed in seeds_h
        for coord in seed["coords"]
    }
    for seed_cell in seed_cells:
        for waypoint in waypoint_set:
            if abs(seed_cell[0] - waypoint[0]) + abs(seed_cell[1] - waypoint[1]) <= 1:
                return False

    winners = [int(info["seed_index"]) for info in infos_h]
    if set(winners) != set(range(len(seeds_h))):
        return False

    has_conflict = False
    has_tied_conflict = False
    for i in range(1, len(waypoints_h) - 1):
        left = infos_h[i - 1]
        right = infos_h[i]
        if left["color"] != right["color"]:
            has_conflict = True
            if left["distance"] == right["distance"]:
                has_tied_conflict = True
    if not has_conflict or not has_tied_conflict:
        return False

    coverage: dict[tuple[int, int], list[int]] = {}
    for info in infos_h:
        for cell in info["path"]:
            coverage.setdefault(cell, []).append(int(info["index"]))
            if cell in seed_cells:
                return False
    for cell, ids in coverage.items():
        if len(ids) > 1 and cell not in waypoint_set:
            return False

    if len(set(int(info["color"]) for info in infos_h)) < 2:
        return False

    if np.array_equal(out_h, g):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_waypoints = rng.randint(4, 6)
    n_seeds = 2 if rng.random() < 0.45 else 3

    min_height = max(18, 4 * n_waypoints + 2)
    height = rng.randint(min_height, 26)
    width = rng.randint(16, 24)

    rows = _spaced_positions(2, height - 3, n_waypoints, 3, rng)
    if rows is None or rows[1] - rows[0] < 4:
        return None

    left_col = rng.randint(3, max(3, width // 2 - 2))
    right_lo = left_col + 6
    right_hi = width - 4
    if right_lo > right_hi:
        return None
    right_col = rng.randint(right_lo, right_hi)

    start_left = rng.random() < 0.5
    start_col = left_col if start_left else right_col
    other_col = right_col if start_left else left_col
    delta = 1 if other_col > start_col else -1
    cols = [int(start_col if i % 2 == 0 else other_col) for i in range(n_waypoints)]

    grid = np.zeros((height, width), dtype=int)
    waypoints = [(int(r), int(c)) for r, c in zip(rows, cols)]
    for row, col in waypoints:
        grid[row, col] = WAYPOINT

    waypoint_set = set(waypoints)
    occupied = set(waypoints)
    colors = list(rng.sample(SEED_PALETTE, n_seeds))
    seeds_to_place: list[tuple[int, list[tuple[int, int]]]] = []

    inner_lo = min(start_col, other_col) + 2
    inner_hi = max(start_col, other_col) - 2
    if inner_lo > inner_hi:
        return None
    a_center = rng.randint(inner_lo, inner_hi)
    a_cells = _make_horizontal_seed(rows[0] - 1, a_center, delta, rng)
    if not _seed_cells_ok(a_cells, grid.shape, waypoint_set, occupied):
        return None
    seeds_to_place.append((colors[0], a_cells))
    occupied.update(a_cells)

    b_col = start_col
    b_cells = _make_bridge_seed(rows[1] - 1, b_col)
    if not _seed_cells_ok(b_cells, grid.shape, waypoint_set, occupied):
        return None
    seeds_to_place.append((colors[1], b_cells))
    occupied.update(b_cells)

    if n_seeds == 3:
        seg_index = 2
        seg_start = cols[seg_index]
        seg_end = cols[seg_index + 1]
        seg_delta = 1 if seg_end > seg_start else -1
        c_col = seg_end + seg_delta
        c_cells = _make_vertical_seed(rows[seg_index + 1] - 1, c_col, rng)
        if not _seed_cells_ok(c_cells, grid.shape, waypoint_set, occupied):
            return None
        seeds_to_place.append((colors[2], c_cells))
        occupied.update(c_cells)

    for color, cells in seeds_to_place:
        rr, cc = np.array(cells, dtype=int).T
        grid[rr, cc] = int(color)

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-seed coloring input")
