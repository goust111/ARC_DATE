from __future__ import annotations

from collections import Counter, deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


SEED_PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
INF = 10**9
GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _extract_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    seed_cells = np.argwhere((grid != Color.BLACK) & (grid != Color.GREY))
    assert len(seed_cells) >= 2, "expected at least two colored seed pixels"

    seeds: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for r0, c0 in seed_cells:
        r = int(r0)
        c = int(c0)
        color = int(grid[r, c])
        assert color != Color.YELLOW, "yellow is reserved for output tie markers"
        assert color not in seen_colors, "seed colors must be pairwise distinct"
        seen_colors.add(color)
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc):
                neighbor = int(grid[nr, nc])
                assert neighbor in (Color.BLACK, Color.GREY), "seed pixels must stay isolated"
        seeds.append((r, c, color))
    seeds.sort()
    return seeds


def _distance_from_seed(shape: tuple[int, int], wall_mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(shape, INF, dtype=int)
    sr, sc = map(int, start)
    dist[sr, sc] = 0
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(shape, nr, nc):
                continue
            if wall_mask[nr, nc] or dist[nr, nc] != INF:
                continue
            dist[nr, nc] = nd
            queue.append((nr, nc))
    return dist


def _solve_from_distances(
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    seed_colors: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    out[wall_mask] = Color.GREY

    min_dist = distances.min(axis=0)
    reachable = min_dist < INF
    tied = reachable & ((distances == min_dist[None, :, :]).sum(axis=0) >= 2)
    out[tied] = Color.YELLOW

    unique = reachable & (~tied)
    owners = distances.argmin(axis=0)
    even = unique & ((min_dist % 2) == 0)
    out[even] = seed_colors[owners[even]]
    return out


def _analyze_grid(grid: np.ndarray):
    wall_mask = grid == Color.GREY
    seeds = _extract_seeds(grid)
    seed_colors = np.array([color for _r, _c, color in seeds], dtype=int)
    dist_stack = np.stack(
        [_distance_from_seed(grid.shape, wall_mask, (r, c)) for r, c, _color in seeds],
        axis=0,
    )
    return seeds, wall_mask, seed_colors, dist_stack


def _solve_manhattan_like(grid: np.ndarray, seeds: list[tuple[int, int, int]]) -> np.ndarray:
    rows, cols = np.indices(grid.shape)
    wall_mask = grid == Color.GREY
    seed_colors = np.array([color for _r, _c, color in seeds], dtype=int)
    distances = np.stack(
        [np.abs(rows - r) + np.abs(cols - c) for r, c, _color in seeds],
        axis=0,
    )
    distances = distances.astype(int, copy=False)
    distances[:, wall_mask] = INF
    return _solve_from_distances(grid.shape, wall_mask, seed_colors, distances)


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each seed by wall-avoiding geodesic wavefronts, revealing only even layers and marking ties yellow.

    Input structure
    - A black background contains several isolated single-pixel seeds, each in a different non-black, non-grey color.
    - One or more one-cell-thick grey wall segments act as obstacles but remain open rather than forming solid filled regions.
    - No other foreground objects are present; yellow is absent from the input.

    Rule
    - For every non-wall cell, compute its 4-connected shortest-path distance to every seed, with paths forbidden to pass through grey wall cells.
    - If two or more seeds achieve the same minimum distance, recolor that cell yellow.
    - Otherwise let d be the unique minimum distance: keep the cell in the seed's color when d is even (including 0), and keep it black when d is odd.
    - Grey wall cells stay grey, and cells unreachable from every seed stay black.

    Key insight
    - The visible pattern is not a plain Manhattan Voronoi map: walls bend the shortest-path wavefronts, and the final image overlays parity gating and yellow equal-distance boundaries on that geodesic distance field.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert _get_bg(g) == Color.BLACK, "expected black background"

    seeds, wall_mask, seed_colors, dist_stack = _analyze_grid(g)
    assert wall_mask.any(), "expected at least one grey wall segment"
    assert len(seeds) >= 2, "expected multiple seeds"
    return _solve_from_distances(g.shape, wall_mask, seed_colors, dist_stack)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _random_wall_path(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocked_mask: np.ndarray,
    min_len: int,
    max_len: int,
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    if h < 5 or w < 5:
        return None

    target_len = int(rng.integers(min_len, max_len + 1))
    for _ in range(140):
        start = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        if blocked_mask[start]:
            continue

        path = [start]
        path_set = {start}
        direction = int(rng.integers(0, 4))
        bends = 0
        while len(path) < target_len:
            r, c = path[-1]
            options: list[tuple[int, int, int]] = []
            candidate_dirs = [direction]
            if direction < 2:
                candidate_dirs.extend([2, 3])
            else:
                candidate_dirs.extend([0, 1])
            rng.shuffle(candidate_dirs)
            if direction in candidate_dirs:
                candidate_dirs.remove(direction)
                candidate_dirs.insert(0, direction)
            for next_dir in candidate_dirs:
                dr, dc = DIRS4[next_dir]
                nr, nc = r + dr, c + dc
                if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                    continue
                if blocked_mask[nr, nc] or (nr, nc) in path_set:
                    continue
                ok = True
                for rr in range(nr - 1, nr + 2):
                    for cc in range(nc - 1, nc + 2):
                        if not _in_bounds(shape, rr, cc):
                            continue
                        if blocked_mask[rr, cc]:
                            ok = False
                            break
                        if (rr, cc) in path_set and (rr, cc) != (r, c):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    options.append((next_dir, nr, nc))
            if not options:
                break
            weights = np.array([3 if next_dir == direction else 1 for next_dir, _nr, _nc in options], dtype=float)
            choice = int(rng.choice(len(options), p=weights / weights.sum()))
            next_dir, nr, nc = options[choice]
            bends += int(next_dir != direction)
            direction = next_dir
            path.append((nr, nc))
            path_set.add((nr, nc))

        if len(path) < min_len:
            continue
        if _chebyshev(path[0], path[-1]) < 3:
            continue
        if bends == 0 and rng.random() < 0.45:
            continue
        return path
    return None


def _place_seed_pair_across_main_wall(
    grid: np.ndarray,
    wall_paths: list[list[tuple[int, int]]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    main_path = max(wall_paths, key=len)
    rows = [r for r, _c in main_path]
    cols = [c for _r, c in main_path]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h_span = r1 - r0 + 1
    w_span = c1 - c0 + 1

    offsets = [2, 3, 4]
    row_jitter = [0, -1, 1, -2, 2]
    col_jitter = [0, -1, 1, -2, 2]
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if h_span >= w_span:
        mid_r = int(round(np.mean(rows)))
        for jitter in row_jitter:
            rr = mid_r + jitter
            if not (0 <= rr < grid.shape[0]):
                continue
            for left_off in offsets:
                for right_off in offsets:
                    left = (rr, c0 - left_off)
                    right = (rr, c1 + right_off)
                    if all(_in_bounds(grid.shape, *p) and grid[p] == Color.BLACK for p in (left, right)):
                        candidates.append((left, right))
    else:
        mid_c = int(round(np.mean(cols)))
        for jitter in col_jitter:
            cc = mid_c + jitter
            if not (0 <= cc < grid.shape[1]):
                continue
            for up_off in offsets:
                for down_off in offsets:
                    up = (r0 - up_off, cc)
                    down = (r1 + down_off, cc)
                    if all(_in_bounds(grid.shape, *p) and grid[p] == Color.BLACK for p in (up, down)):
                        candidates.append((up, down))

    rng.shuffle(candidates)
    for a, b in candidates:
        if _chebyshev(a, b) >= 5:
            return [a, b]
    return None


def _random_extra_seed_positions(
    grid: np.ndarray,
    rng: np.random.Generator,
    existing: list[tuple[int, int]],
    total: int,
) -> list[tuple[int, int]] | None:
    positions = list(existing)
    empties = [(int(r), int(c)) for r, c in np.argwhere(grid == Color.BLACK)]
    rng.shuffle(empties)
    for cell in empties:
        if all(abs(cell[0] - r) + abs(cell[1] - c) >= 4 for r, c in positions):
            positions.append(cell)
            if len(positions) == total:
                return positions
    return None


def _build_candidate_input(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(11, 19))
    w = int(rng.integers(11, 19))
    grid = np.full((h, w), Color.BLACK, dtype=int)

    wall_count = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
    blocked = np.zeros((h, w), dtype=bool)
    wall_paths: list[list[tuple[int, int]]] = []
    for wall_index in range(wall_count):
        min_len = 5 if wall_index == 0 else 4
        max_len = min(h + w - 6, 9 if wall_index == 0 else 8)
        path = _random_wall_path(rng, grid.shape, blocked, min_len, max_len)
        if path is None:
            return None
        wall_paths.append(path)
        for r, c in path:
            grid[r, c] = Color.GREY
            blocked[r, c] = True

    base_pair = _place_seed_pair_across_main_wall(grid, wall_paths, rng)
    if base_pair is None:
        return None

    seed_count = int(rng.choice([2, 3, 4], p=[0.3, 0.4, 0.3]))
    seed_positions = _random_extra_seed_positions(grid, rng, base_pair, seed_count)
    if seed_positions is None:
        return None

    colors = list(rng.choice(np.array(SEED_PALETTE, dtype=int), size=seed_count, replace=False))
    rng.shuffle(colors)
    for (r, c), color in zip(seed_positions, colors):
        grid[r, c] = int(color)

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        seeds, wall_mask, seed_colors, dist_stack = _analyze_grid(grid)
    except AssertionError:
        return False

    out = _solve_from_distances(grid.shape, wall_mask, seed_colors, dist_stack)
    manhattan_out = _solve_manhattan_like(grid, seeds)

    min_dist = dist_stack.min(axis=0)
    reachable = min_dist < INF
    tied = reachable & ((dist_stack == min_dist[None, :, :]).sum(axis=0) >= 2)
    unique = reachable & (~tied)
    owners = dist_stack.argmin(axis=0)

    if np.count_nonzero(wall_mask) < 4:
        return False
    if np.count_nonzero(out == Color.YELLOW) < 2:
        return False
    visible_nonseed = np.count_nonzero((out != Color.BLACK) & (out != Color.GREY)) - len(seeds)
    if visible_nonseed < 8:
        return False
    if np.count_nonzero(unique & ((min_dist % 2) == 1)) < 8:
        return False
    if np.count_nonzero(out != manhattan_out) < 6:
        return False

    for idx, (_r, _c, _color) in enumerate(seeds):
        own_mask = unique & (owners == idx)
        if np.count_nonzero(own_mask) < 4:
            return False
        if np.count_nonzero(own_mask & ((min_dist % 2) == 0)) < 2:
            return False

    # Ensure at least one cell is pushed strictly farther by a wall than by plain Manhattan distance.
    rows, cols = np.indices(grid.shape)
    manhattan_min = np.min(
        np.stack([np.abs(rows - r) + np.abs(cols - c) for r, c, _color in seeds], axis=0),
        axis=0,
    )
    if np.count_nonzero((min_dist < INF) & (min_dist > manhattan_min)) < 6:
        return False

    # Reject visually flat instances where one seed color barely appears.
    for color in seed_colors:
        if np.count_nonzero(out == int(color)) < 2:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_candidate_input(rng)
        if grid is not None and _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid geodesic parity-wave input")
