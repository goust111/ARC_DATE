import random
from collections import deque
from itertools import combinations

import numpy as np


BG = 0
FILL_BLUE = 1
RAY = 2
FILL_GREEN = 3
FILL_YELLOW = 4
WALL = 5

GEN_RETRIES = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_marker_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    markers = np.argwhere(grid == RAY)
    assert len(markers) == 1, "expected exactly one red edge marker"
    r, c = map(int, markers[0])
    h, w = grid.shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top or on_bottom or on_left or on_right, "marker must lie on the grid edge"
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "marker direction must be unique"
    if on_top:
        return (r, c), (1, 0)
    if on_bottom:
        return (r, c), (-1, 0)
    if on_left:
        return (r, c), (0, 1)
    return (r, c), (0, -1)


def _wall_scan(base: np.ndarray, hit_r: int, hit_c: int, ray_dr: int, ray_dc: int) -> tuple[tuple, list[tuple[int, int]]]:
    if ray_dr == 0:
        scan_dr, scan_dc = 1, 0
        fixed = hit_c
        axis_len = base.shape[0]

        def at(idx: int) -> int:
            return int(base[idx, fixed])

        def cell(idx: int) -> tuple[int, int]:
            return idx, fixed

        orientation = "V"
    else:
        scan_dr, scan_dc = 0, 1
        fixed = hit_r
        axis_len = base.shape[1]

        def at(idx: int) -> int:
            return int(base[fixed, idx])

        def cell(idx: int) -> tuple[int, int]:
            return fixed, idx

        orientation = "H"

    axis_idx = hit_r if scan_dr else hit_c

    def extend(idx: int, step: int) -> int:
        cur = idx
        while True:
            nxt = cur + step
            if not (0 <= nxt < axis_len):
                return cur
            value = at(nxt)
            if value == WALL:
                cur = nxt
                continue
            if value == BG:
                nxt2 = nxt + step
                if 0 <= nxt2 < axis_len and at(nxt2) == WALL:
                    cur = nxt2
                    continue
            return cur

    lo = extend(axis_idx, -1)
    hi = extend(axis_idx, 1)
    gaps = [cell(idx) for idx in range(lo, hi + 1) if at(idx) == BG]
    segment_id = (orientation, int(fixed), int(lo), int(hi))
    return segment_id, gaps


def _diffuse_rays(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    base = np.asarray(grid, dtype=int)
    out = base.copy()
    (start_r, start_c), (dr, dc) = _find_marker_and_direction(base)
    first_r = start_r + dr
    first_c = start_c + dc
    assert _in_bounds(base.shape, first_r, first_c), "marker must face the grid interior"
    assert int(base[first_r, first_c]) != WALL, "marker must have at least one free step"

    queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
    scanned_segments: set[tuple] = set()
    branching_segments: set[tuple] = set()
    hit_cells: set[tuple[int, int]] = set()
    spawned_from_gaps = 0

    while queue:
        r, c = queue.popleft()
        cur_r, cur_c = r, c
        while True:
            nr = cur_r + dr
            nc = cur_c + dc
            if not _in_bounds(base.shape, nr, nc):
                break
            value = int(out[nr, nc])
            if value == RAY:
                break
            if value == BG:
                out[nr, nc] = RAY
                cur_r, cur_c = nr, nc
                continue
            if value != WALL:
                break

            out[nr, nc] = RAY
            hit_cells.add((nr, nc))
            segment_id, gaps = _wall_scan(base, nr, nc, dr, dc)
            scanned_segments.add(segment_id)
            if len(gaps) >= 2:
                branching_segments.add(segment_id)
            for gr, gc in gaps:
                if int(out[gr, gc]) != RAY:
                    out[gr, gc] = RAY
                    queue.append((gr, gc))
                    spawned_from_gaps += 1
            break

    stats = {
        "direction": (dr, dc),
        "hit_cells": hit_cells,
        "scanned_segments": scanned_segments,
        "branching_segments": branching_segments,
        "spawned_from_gaps": spawned_from_gaps,
    }
    return out, stats


def _count_to_color(count: int) -> int:
    if count == 0:
        return BG
    if count <= 2:
        return FILL_BLUE
    if count <= 4:
        return FILL_GREEN
    return FILL_YELLOW


def _region_contacts(ray_grid: np.ndarray) -> list[dict]:
    h, w = ray_grid.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []

    for start_r in range(h):
        for start_c in range(w):
            if int(ray_grid[start_r, start_c]) != BG or seen[start_r, start_c]:
                continue

            queue = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            red_boundary: set[tuple[int, int]] = set()

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds(ray_grid.shape, nr, nc):
                        continue
                    value = int(ray_grid[nr, nc])
                    if value == BG:
                        if not seen[nr, nc]:
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                    elif value == RAY:
                        red_boundary.add((nr, nc))

            regions.append({"cells": cells, "contact_count": len(red_boundary)})

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse edge-launched rays through wall openings, then recolor each black region by how many red boundary cells it touches.

    Input structure:
    - Black (0) is the background.
    - Exactly one red (2) marker lies on a non-corner edge cell and points inward.
    - Gray (5) cells form several disjoint one-cell-thick parallel wall segments with one or two one-cell openings each.

    Rule:
    - Launch a ray from the red edge marker perpendicular to that edge; every ray keeps the same direction.
    - Rays paint traversed black cells red. When a ray would step into a gray wall cell, paint that hit wall cell red,
      scan the whole wall segment along its length, and launch new rays from every opening on that segment.
    - Rays stop when they leave the grid or would revisit an already red cell.
    - After all rays stop, flood-fill the remaining black cells into 4-connected regions and count, for each region,
      how many distinct 4-neighbor boundary cells are red.
    - Recolor each region by that count: 0 stays black, 1-2 becomes blue (1), 3-4 becomes green (3), and 5+ becomes yellow (4).
      Existing red ray cells and unreached gray wall cells stay unchanged.

    Key insight:
    - The output color of a region is not local to that region: it is determined by the global branching topology of the ray tree and
      by which openings let those rays touch the region boundary.
    """
    ray_grid, _ = _diffuse_rays(np.asarray(grid, dtype=int))
    out = ray_grid.copy()
    for region in _region_contacts(ray_grid):
        color = _count_to_color(int(region["contact_count"]))
        if color == BG:
            continue
        for r, c in region["cells"]:
            out[r, c] = color
    return out


def _sample_strip_widths(n_walls: int) -> list[int] | None:
    for _ in range(100):
        widths = [random.randint(2, 6) for _ in range(n_walls + 1)]
        total_w = sum(widths) + n_walls
        if 15 <= total_w <= 28 and len(set(widths)) >= 2:
            return widths
    return None


def _candidate_gap_sets(height: int, count: int, active_rows: set[int]) -> list[tuple[int, ...]]:
    rows = list(range(1, height - 1))
    choices = []
    for combo in combinations(rows, count):
        if count == 2 and abs(combo[0] - combo[1]) < 2:
            continue
        if active_rows.issubset(set(combo)):
            continue
        choices.append(combo)
    return choices


def _pick_gap_set(height: int, count: int, active_rows: set[int]) -> tuple[int, ...] | None:
    choices = _candidate_gap_sets(height, count, active_rows)
    if not choices:
        return None
    partial_overlap = [combo for combo in choices if 0 < len(active_rows & set(combo)) < len(active_rows)]
    if partial_overlap and random.random() < 0.6:
        return random.choice(partial_overlap)
    return random.choice(choices)


def _build_vertical_candidate() -> np.ndarray | None:
    n_walls = random.randint(3, 5)
    height = random.randint(8, 12)
    widths = _sample_strip_widths(n_walls)
    if widths is None:
        return None

    wall_cols = []
    col = widths[0]
    wall_cols.append(col)
    for width in widths[1:-1]:
        col += width + 1
        wall_cols.append(col)
    total_w = sum(widths) + n_walls

    marker_row = random.randint(1, height - 2)
    gap_counts = [2] + [random.choice([1, 2]) for _ in range(n_walls - 1)]
    if n_walls > 1 and 1 not in gap_counts[1:]:
        gap_counts[random.randint(1, n_walls - 1)] = 1

    active_rows = {marker_row}
    gap_sets = []
    for count in gap_counts:
        gaps = _pick_gap_set(height, count, active_rows)
        if gaps is None:
            return None
        gap_sets.append(gaps)
        active_rows = set(gaps)

    grid = np.zeros((height, total_w), dtype=int)
    for wall_c, gaps in zip(wall_cols, gap_sets):
        grid[:, wall_c] = WALL
        for gap_r in gaps:
            grid[gap_r, wall_c] = BG
    grid[marker_row, 0] = RAY
    return grid


def _random_transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    if random.random() < 0.5:
        g = g.T
    if random.random() < 0.5:
        g = np.flipud(g)
    if random.random() < 0.5:
        g = np.fliplr(g)
    return np.ascontiguousarray(g)


def _extract_parallel_walls(grid: np.ndarray) -> tuple[str, list[tuple[int, list[int]]]]:
    _, (dr, dc) = _find_marker_and_direction(grid)
    h, w = grid.shape
    if dr == 0:
        cols = [c for c in range(w) if np.any(grid[:, c] == WALL)]
        walls = []
        for c in cols:
            gap_rows = [r for r in range(h) if int(grid[r, c]) == BG]
            wall_pixels = int(np.sum(grid[:, c] == WALL))
            if wall_pixels >= 5:
                walls.append((c, gap_rows))
        return "V", walls

    rows = [r for r in range(h) if np.any(grid[r, :] == WALL)]
    walls = []
    for r in rows:
        gap_cols = [c for c in range(w) if int(grid[r, c]) == BG]
        wall_pixels = int(np.sum(grid[r, :] == WALL))
        if wall_pixels >= 5:
            walls.append((r, gap_cols))
    return "H", walls


def _semantic_valid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if set(map(int, np.unique(g))) - {BG, RAY, WALL}:
        return False

    try:
        (mr, mc), (dr, dc) = _find_marker_and_direction(g)
    except AssertionError:
        return False

    nr = mr + dr
    nc = mc + dc
    if not _in_bounds(g.shape, nr, nc) or int(g[nr, nc]) == WALL:
        return False

    orientation, walls = _extract_parallel_walls(g)
    if not (3 <= len(walls) <= 5):
        return False
    if orientation == "V" and dr != 0:
        return False
    if orientation == "H" and dc != 0:
        return False

    span_len = g.shape[0] if orientation == "V" else g.shape[1]
    if not (5 <= span_len <= 12):
        return False

    gap_sizes = []
    for _, gap_positions in walls:
        gaps = [idx for idx in gap_positions if 0 < idx < span_len - 1]
        if len(gaps) not in (1, 2):
            return False
        if len(gaps) == 2 and abs(gaps[0] - gaps[1]) < 2:
            return False
        gap_sizes.append(len(gaps))

    if 2 not in gap_sizes or 1 not in gap_sizes:
        return False

    try:
        ray_grid, stats = _diffuse_rays(g)
    except AssertionError:
        return False

    if len(stats["scanned_segments"]) < 2:
        return False
    if not stats["branching_segments"]:
        return False
    if stats["spawned_from_gaps"] < 3:
        return False

    red_density = float(np.mean(ray_grid == RAY))
    if red_density > 0.42:
        return False

    regions = _region_contacts(ray_grid)
    touched = [int(region["contact_count"]) for region in regions if int(region["contact_count"]) > 0]
    if len(touched) < 3:
        return False
    if len(set(touched)) < 2:
        return False
    if len(set(_count_to_color(c) for c in touched)) < 2:
        return False

    out = main(g)
    if out.shape != g.shape:
        return False
    recolored = {int(v) for v in np.unique(out)} - {BG, RAY, WALL}
    if len(recolored) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        base = _build_vertical_candidate()
        if base is None:
            continue
        grid = _random_transform(base)
        if _semantic_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid ray-branch diffusion input")
