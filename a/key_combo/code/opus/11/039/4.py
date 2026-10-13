from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
ORANGE = 7

QUADRANTS = ("TL", "TR", "BL", "BR")
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_marker_box(grid: np.ndarray) -> tuple[int, int]:
    marker_mask = grid == GREEN
    box = _bbox(marker_mask)
    assert box is not None, "expected a green 2x2 marker block"
    r0, c0, r1, c1 = box
    assert int(marker_mask.sum()) == 4, "marker must contain exactly four green cells"
    assert r1 == r0 + 1 and c1 == c0 + 1, "marker must be a 2x2 square"
    assert np.all(marker_mask[r0 : r1 + 1, c0 : c1 + 1]), "marker square must be solid"
    return r0, c0


def _quadrant_of(r: int, c: int, marker_r: int, marker_c: int) -> str | None:
    if marker_r <= r <= marker_r + 1:
        return None
    vertical = "T" if r < marker_r else "B"
    horizontal = "L" if c <= marker_c else "R"
    return vertical + horizontal


def _marker_adjacent(r: int, c: int, marker_r: int, marker_c: int) -> bool:
    return marker_r - 1 <= r <= marker_r + 2 and marker_c - 1 <= c <= marker_c + 2


def _reflection_positions(r: int, c: int, marker_r: int, marker_c: int) -> list[tuple[int, int]]:
    return [
        (r, 2 * marker_c + 1 - c),
        (2 * marker_r + 1 - r, c),
        (2 * marker_r + 1 - r, 2 * marker_c + 1 - c),
    ]


def _symmetry_stage(
    grid: np.ndarray, *, obstacles_block: bool = True
) -> tuple[np.ndarray, dict[str, object]]:
    out = np.array(grid, dtype=int, copy=True)
    marker_r, marker_c = _find_marker_box(out)
    source_reds = [tuple(map(int, rc)) for rc in np.argwhere(out == RED)]
    blocked_targets: set[tuple[int, int]] = set()
    added_targets: set[tuple[int, int]] = set()
    full_targets: set[tuple[int, int]] = set()
    full_targets_by_quadrant: dict[str, set[tuple[int, int]]] = {q: set() for q in QUADRANTS}
    blocked_by_quadrant: dict[str, set[tuple[int, int]]] = {q: set() for q in QUADRANTS}
    added_by_quadrant: dict[str, set[tuple[int, int]]] = {q: set() for q in QUADRANTS}

    for r, c in source_reds:
        for rr, cc in _reflection_positions(r, c, marker_r, marker_c):
            full_targets.add((rr, cc))
            quadrant = _quadrant_of(rr, cc, marker_r, marker_c)
            if quadrant is not None:
                full_targets_by_quadrant[quadrant].add((rr, cc))

            if obstacles_block and int(out[rr, cc]) == ORANGE:
                blocked_targets.add((rr, cc))
                if quadrant is not None:
                    blocked_by_quadrant[quadrant].add((rr, cc))
                continue
            if int(out[rr, cc]) == BG:
                out[rr, cc] = RED
                added_targets.add((rr, cc))
                if quadrant is not None:
                    added_by_quadrant[quadrant].add((rr, cc))

    return out, {
        "marker": (marker_r, marker_c),
        "blocked_targets": blocked_targets,
        "added_targets": added_targets,
        "full_targets": full_targets,
        "full_targets_by_quadrant": full_targets_by_quadrant,
        "blocked_by_quadrant": blocked_by_quadrant,
        "added_by_quadrant": added_by_quadrant,
    }


def _erase_obstacles(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[out == ORANGE] = BG
    return out


def _apply_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    out = np.full_like(g, BG)
    out[g == GREEN] = GREEN

    for c in range(w):
        red_rows = np.flatnonzero(g[:, c] == RED)
        blocker_rows = np.flatnonzero(g[:, c] == GREEN)
        prev = -1
        for blocker in list(map(int, blocker_rows)) + [h]:
            seg_start = prev + 1
            seg_end = blocker - 1
            if seg_start <= seg_end:
                count = sum(seg_start <= int(r) <= seg_end for r in red_rows)
                if count:
                    out[seg_end - count + 1 : seg_end + 1, c] = RED
            prev = blocker
    return out


def _apply_gravity_without_marker_block(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    out = np.full_like(g, BG)
    red_mask = g == RED
    for c in range(w):
        count = int(red_mask[:, c].sum())
        if count:
            out[h - count : h, c] = RED
    out[g == GREEN] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete four-way reflections around the green 2x2 center, let orange blockers delete some mirrored cells, then drop the resulting red pixels under gravity.

    Input structure:
    - Black background dominates the grid.
    - A unique green 2x2 square marks the geometric symmetry center.
    - Red source pixels lie in one or two quadrants around that center, and orange single-cell obstacles occupy other quadrants.

    Rule:
    - For every red pixel, compute its horizontal reflection, vertical reflection, and 180-degree reflection across the center of the green 2x2 block.
    - Paint each reflected location red if it is black; if that reflected location is orange, block that single reflection and leave it empty; existing red and green cells stay unchanged.
    - Erase all orange obstacle cells, then apply gravity column by column to all red pixels while keeping the green block fixed; in columns containing green cells, reds stack separately above and below the green block.

    Key insight:
    - The orange cells do not block falling; they selectively delete specific mirrored pixels before gravity, so the final falling shape depends on both incomplete symmetry completion and marker-aware settling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    reflected, _ = _symmetry_stage(g, obstacles_block=True)
    cleared = _erase_obstacles(reflected)
    return _apply_gravity(cleared)


def _abs_from_quad(
    marker_r: int, marker_c: int, quadrant: str, dr: int, dc: int
) -> tuple[int, int]:
    if quadrant == "TL":
        return marker_r - dr, marker_c - dc
    if quadrant == "TR":
        return marker_r - dr, marker_c + 1 + dc
    if quadrant == "BL":
        return marker_r + 1 + dr, marker_c - dc
    if quadrant == "BR":
        return marker_r + 1 + dr, marker_c + 1 + dc
    raise ValueError(f"unknown quadrant: {quadrant}")


def _symmetric_extents(shape: tuple[int, int], marker_r: int, marker_c: int) -> tuple[int, int]:
    h, w = shape
    row_extent = min(marker_r, h - marker_r - 2)
    col_extent = min(marker_c, w - marker_c - 2)
    return int(row_extent), int(col_extent)


def _quadrant_pool(
    shape: tuple[int, int],
    marker_r: int,
    marker_c: int,
    quadrant: str,
    row_extent: int,
    col_extent: int,
) -> set[tuple[int, int]]:
    cells = set()
    for dr in range(2, row_extent + 1):
        for dc in range(0, col_extent + 1):
            cells.add(_abs_from_quad(marker_r, marker_c, quadrant, dr, dc))
    return cells


def _grow_connected(
    pool: set[tuple[int, int]], size: int, preferred_seed: tuple[int, int] | None = None
) -> set[tuple[int, int]] | None:
    if not pool or size > len(pool):
        return None
    cells = {preferred_seed} if preferred_seed is not None else {random.choice(tuple(pool))}
    frontier = set()
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in pool and nb not in cells:
                frontier.add(nb)
    while len(cells) < size:
        if not frontier:
            return None
        picked = random.choice(tuple(frontier))
        frontier.remove(picked)
        cells.add(picked)
        r, c = picked
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in pool and nb not in cells:
                frontier.add(nb)
    return cells


def _sample_scattered(
    pool: set[tuple[int, int]], size: int, preferred_seed: tuple[int, int] | None = None
) -> set[tuple[int, int]] | None:
    if size > len(pool):
        return None
    remaining = list(pool)
    if preferred_seed is not None and preferred_seed in pool:
        remaining.remove(preferred_seed)
        return {preferred_seed, *random.sample(remaining, size - 1)}
    return set(random.sample(remaining, size))


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            cr, cc = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _place_obstacles(
    shape: tuple[int, int],
    marker_r: int,
    marker_c: int,
    source_cells: set[tuple[int, int]],
    source_quadrants: set[str],
    blocked_goal: int,
    total_goal: int,
) -> set[tuple[int, int]] | None:
    blocked_candidates: set[tuple[int, int]] = set()
    for r, c in source_cells:
        for rr, cc in _reflection_positions(r, c, marker_r, marker_c):
            quadrant = _quadrant_of(rr, cc, marker_r, marker_c)
            if quadrant is None or quadrant in source_quadrants:
                continue
            if (rr, cc) in source_cells or _marker_adjacent(rr, cc, marker_r, marker_c):
                continue
            blocked_candidates.add((rr, cc))
    if len(blocked_candidates) < blocked_goal:
        return None

    blocked = set(random.sample(tuple(blocked_candidates), blocked_goal))
    obstacles = set(blocked)

    h, w = shape
    extra_candidates = []
    for r in range(h):
        for c in range(w):
            quadrant = _quadrant_of(r, c, marker_r, marker_c)
            if quadrant is None or quadrant in source_quadrants:
                continue
            if (r, c) in source_cells or (r, c) in obstacles:
                continue
            if _marker_adjacent(r, c, marker_r, marker_c):
                continue
            extra_candidates.append((r, c))
    random.shuffle(extra_candidates)

    for cell in extra_candidates:
        if len(obstacles) >= total_goal:
            break
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in obstacles):
            obstacles.add(cell)

    if len(obstacles) != total_goal:
        return None
    return obstacles


def _build_attempt() -> tuple[np.ndarray, bool] | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    grid = np.full((h, w), BG, dtype=int)

    marker_r = random.randint(4, h - 7)
    marker_c = random.randint(3, w - 5)
    grid[marker_r : marker_r + 2, marker_c : marker_c + 2] = GREEN

    row_extent, col_extent = _symmetric_extents(grid.shape, marker_r, marker_c)
    if row_extent < 2 or col_extent < 1:
        return None

    need_green_interaction = random.random() < 0.6
    source_quadrant_count = 1 if random.random() < 0.55 else 2
    total_reds = random.randint(3, 8)
    connected_mode = random.random() < 0.6

    if need_green_interaction:
        primary = random.choice(("TL", "TR"))
    else:
        primary = random.choice(QUADRANTS)

    remaining_quadrants = [q for q in QUADRANTS if q != primary]
    second = None
    if source_quadrant_count == 2:
        second = random.choice(remaining_quadrants)
    source_quadrants = {primary} if second is None else {primary, second}

    quotas = [total_reds] if second is None else [random.randint(2, total_reds - 1), 0]
    if second is not None:
        quotas[1] = total_reds - quotas[0]

    source_cells: set[tuple[int, int]] = set()
    quadrants_in_order = [primary] if second is None else [primary, second]
    interaction_requested = False

    for quadrant, count in zip(quadrants_in_order, quotas):
        pool = _quadrant_pool(grid.shape, marker_r, marker_c, quadrant, row_extent, col_extent)
        preferred_seed = None
        if need_green_interaction and not interaction_requested and quadrant in {"TL", "TR"}:
            preferred_seed = _abs_from_quad(marker_r, marker_c, quadrant, 2, 0)
            interaction_requested = True
        placed = (
            _grow_connected(pool, count, preferred_seed=preferred_seed)
            if connected_mode
            else _sample_scattered(pool, count, preferred_seed=preferred_seed)
        )
        if placed is None:
            return None
        source_cells |= placed

    if len(source_cells) != total_reds:
        return None
    if any(_marker_adjacent(r, c, marker_r, marker_c) for r, c in source_cells):
        return None

    for r, c in source_cells:
        grid[r, c] = RED

    blocked_goal = 1 if random.random() < 0.55 else 2
    total_obstacles = random.randint(max(2, blocked_goal + 1), 4)
    obstacles = _place_obstacles(
        grid.shape,
        marker_r,
        marker_c,
        source_cells,
        source_quadrants,
        blocked_goal,
        total_obstacles,
    )
    if obstacles is None:
        return None

    for r, c in obstacles:
        grid[r, c] = ORANGE
    return grid, need_green_interaction


def _validate_input_structure(grid: np.ndarray) -> tuple[int, int, bool]:
    marker_r, marker_c = _find_marker_box(grid)
    assert grid.shape[0] - (marker_r + 2) >= 5, "marker must stay well above the bottom"

    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    orange_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == ORANGE)]
    assert 3 <= len(red_coords) <= 8, "expected 3-8 red source pixels"
    assert 2 <= len(orange_coords) <= 4, "expected 2-4 orange obstacles"
    assert all(not _marker_adjacent(r, c, marker_r, marker_c) for r, c in red_coords), (
        "red source pixels must not touch the green marker"
    )

    source_quadrants = {_quadrant_of(r, c, marker_r, marker_c) for r, c in red_coords}
    assert None not in source_quadrants
    assert 1 <= len(source_quadrants) <= 2, "source must occupy one or two quadrants"

    obstacle_quadrants = {_quadrant_of(r, c, marker_r, marker_c) for r, c in orange_coords}
    assert None not in obstacle_quadrants
    assert obstacle_quadrants.isdisjoint(source_quadrants), (
        "obstacles must live outside the source quadrants"
    )

    for i, (r0, c0) in enumerate(orange_coords):
        for r1, c1 in orange_coords[i + 1 :]:
            assert max(abs(r0 - r1), abs(c0 - c1)) > 1, "obstacles must stay isolated"

    touches_marker_column = any(c in {marker_c, marker_c + 1} and r < marker_r for r, c in red_coords)
    return marker_r, marker_c, touches_marker_column


def _validate_non_degenerate(grid: np.ndarray, expect_green_interaction: bool) -> bool:
    marker_r, marker_c, has_marker_column_red = _validate_input_structure(grid)

    full_input = grid.copy()
    full_input[full_input == ORANGE] = BG
    fully_reflected, full_meta = _symmetry_stage(full_input, obstacles_block=False)
    blocked_reflected, blocked_meta = _symmetry_stage(grid, obstacles_block=True)
    cleared = _erase_obstacles(blocked_reflected)
    out = _apply_gravity(cleared)

    if len(blocked_meta["blocked_targets"]) == 0:
        return False
    if len(blocked_meta["added_targets"]) == 0:
        return False

    selective_quadrant_seen = False
    for quadrant in QUADRANTS:
        full_count = len(full_meta["full_targets_by_quadrant"][quadrant])
        blocked_count = len(blocked_meta["blocked_by_quadrant"][quadrant])
        added_count = len(blocked_meta["added_by_quadrant"][quadrant])
        if full_count >= 2 and 0 < blocked_count < full_count and added_count > 0:
            selective_quadrant_seen = True
            break
    if not selective_quadrant_seen:
        return False

    gravity_only = _apply_gravity(_erase_obstacles(grid))
    if np.array_equal(out, gravity_only):
        return False

    full_reflect_then_gravity = _apply_gravity(_erase_obstacles(fully_reflected))
    if np.array_equal(out, full_reflect_then_gravity):
        return False

    if expect_green_interaction:
        if not has_marker_column_red:
            return False
        transparent_gravity = _apply_gravity_without_marker_block(cleared)
        if np.array_equal(out, transparent_gravity):
            return False

    if np.any(out == ORANGE):
        return False
    if not np.array_equal(out[marker_r : marker_r + 2, marker_c : marker_c + 2], np.full((2, 2), GREEN)):
        return False

    if _count_components(grid == RED) > 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, expect_green_interaction = built
        try:
            if _validate_non_degenerate(grid, expect_green_interaction):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 4")
