from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREEN = 3
PALETTE = [1, 2, 4, 5, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (17, 25)
GRID_W_RANGE = (17, 25)
REGION_COUNT_CHOICES = [1, 2, 3]
REGION_COUNT_WEIGHTS = [2, 3, 2]
REGION_SHAPE_TRIES = 40
REGION_PLACE_TRIES = 90
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_PAD = 1
EDGE_MARGIN = 1

GROUP_PATTERNS = {
    1: [[1]],
    2: [[2], [1, 1]],
    3: [[3], [2, 1], [1, 1, 1]],
    4: [[2, 2], [3, 1], [2, 1, 1], [1, 1, 1, 1]],
}


def _in_bounds(height: int, width: int, row: int, col: int) -> bool:
    return 0 <= row < height and 0 <= col < width


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIRS:
                    nr = r + dr
                    nc = c + dc
                    if _in_bounds(height, width, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _touches_edge(coords: list[tuple[int, int]], height: int, width: int) -> bool:
    return any(row == 0 or row == height - 1 or col == 0 or col == width - 1 for row, col in coords)


def _region_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    return [
        coords
        for coords in _components(grid != GREEN)
        if not _touches_edge(coords, height, width)
    ]


def _tower_direction(grid: np.ndarray, row: int, col: int) -> tuple[int, int]:
    height, width = grid.shape
    dirs = []
    for dr, dc in DIRS:
        nr = row + dr
        nc = col + dc
        if _in_bounds(height, width, nr, nc) and int(grid[nr, nc]) == GREEN:
            dirs.append((dr, dc))
    assert len(dirs) == 1, "each signal tower must touch the boundary in exactly one direction"
    return dirs[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored rays outward from green-enclosed regions.

    Each non-black, non-green tower inside a closed green boundary sends a ray
    through the adjacent boundary cell and outward. The ray length equals the
    number of towers of the same color in that region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    out = g.copy()
    for region in _region_components(g):
        towers = [(row, col) for row, col in region if int(g[row, col]) not in (BLACK, GREEN)]
        assert towers, "each enclosed region must contain at least one tower"
        color_counts = Counter(int(g[row, col]) for row, col in towers)
        for row, col in sorted(towers):
            color = int(g[row, col])
            dr, dc = _tower_direction(g, row, col)
            length = int(color_counts[color])
            ray_row = row + 2 * dr
            ray_col = col + 2 * dc
            for _ in range(length):
                if not _in_bounds(height, width, ray_row, ray_col):
                    break
                if int(out[ray_row, ray_col]) != BLACK:
                    break
                out[ray_row, ray_col] = color
                ray_row += dr
                ray_col += dc
    return out


def _rectangle_interior() -> np.ndarray:
    height = random.randint(4, 7)
    width = random.randint(4, 7)
    return np.ones((height, width), dtype=bool)


def _bump_interior() -> np.ndarray:
    base_h = random.randint(4, 7)
    base_w = random.randint(4, 7)
    side = random.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        bump_h = random.randint(1, 2)
        span = random.randint(2, base_w - 1)
        start = random.randint(0, base_w - span)
        interior = np.zeros((base_h + bump_h, base_w), dtype=bool)
        if side == "top":
            interior[bump_h:, :] = True
            interior[:bump_h, start : start + span] = True
        else:
            interior[:base_h, :] = True
            interior[base_h:, start : start + span] = True
        return interior

    bump_w = random.randint(1, 2)
    span = random.randint(2, base_h - 1)
    start = random.randint(0, base_h - span)
    interior = np.zeros((base_h, base_w + bump_w), dtype=bool)
    if side == "left":
        interior[:, bump_w:] = True
        interior[start : start + span, :bump_w] = True
    else:
        interior[:, :base_w] = True
        interior[start : start + span, base_w:] = True
    return interior


def _sample_interior() -> np.ndarray:
    if random.random() < 0.5:
        return _rectangle_interior()
    return _bump_interior()


def _build_region_patch(interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full_interior = np.zeros((interior.shape[0] + 2, interior.shape[1] + 2), dtype=bool)
    full_interior[1:-1, 1:-1] = interior
    boundary = np.zeros_like(full_interior)
    for row, col in np.argwhere(full_interior):
        row = int(row)
        col = int(col)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = row + dr
                nc = col + dc
                if not full_interior[nr, nc]:
                    boundary[nr, nc] = True
    return full_interior, boundary


def _local_tower_candidates(interior: np.ndarray, boundary: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    candidates: list[dict[str, int | tuple[int, int]]] = []
    for row, col in np.argwhere(interior):
        row = int(row)
        col = int(col)
        dirs = []
        for dr, dc in DIRS:
            if boundary[row + dr, col + dc]:
                dirs.append((dr, dc))
        if len(dirs) == 1:
            candidates.append({"row": row, "col": col, "dir": dirs[0]})
    return candidates


def _max_clear_run(
    occupied: np.ndarray,
    reserved: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
) -> int:
    height, width = occupied.shape
    run = 0
    ray_row = row + 2 * dr
    ray_col = col + 2 * dc
    while _in_bounds(height, width, ray_row, ray_col) and not occupied[ray_row, ray_col] and not reserved[ray_row, ray_col]:
        run += 1
        ray_row += dr
        ray_col += dc
    return run


def _ray_cells(
    occupied: np.ndarray,
    reserved: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    length: int,
    extra_blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    height, width = occupied.shape
    cells: list[tuple[int, int]] = []
    ray_row = row + 2 * dr
    ray_col = col + 2 * dc
    for _ in range(length):
        if not _in_bounds(height, width, ray_row, ray_col):
            return None
        if occupied[ray_row, ray_col] or reserved[ray_row, ray_col] or (ray_row, ray_col) in extra_blocked:
            return None
        cells.append((ray_row, ray_col))
        ray_row += dr
        ray_col += dc
    return cells


def _can_place_patch(grid: np.ndarray, reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    grid_h, grid_w = grid.shape
    if reserved[top : top + height, left : left + width].any():
        return False
    row0 = max(0, top - PLACEMENT_PAD)
    row1 = min(grid_h, top + height + PLACEMENT_PAD)
    col0 = max(0, left - PLACEMENT_PAD)
    col1 = min(grid_w, left + width + PLACEMENT_PAD)
    return not np.any(grid[row0:row1, col0:col1] != BLACK)


def _touches_existing_tower(row: int, col: int, chosen_cells: set[tuple[int, int]]) -> bool:
    if (row, col) in chosen_cells:
        return True
    for dr, dc in DIRS:
        if (row + dr, col + dc) in chosen_cells:
            return True
    return False


def _assign_groups(
    usable: list[dict[str, int | tuple[int, int]]],
    groups: list[int],
    occupied: np.ndarray,
    reserved: np.ndarray,
) -> list[dict[str, object]] | None:
    used: set[int] = set()
    temp_blocked: set[tuple[int, int]] = set()
    chosen_towers: set[tuple[int, int]] = set()
    chosen_groups: list[list[tuple[int, list[tuple[int, int]]]]] = []

    for group_size in sorted(groups, reverse=True):
        eligible = [
            idx
            for idx, cand in enumerate(usable)
            if idx not in used
            and int(cand["max_run"]) >= group_size
            and not _touches_existing_tower(int(cand["row"]), int(cand["col"]), chosen_towers)
        ]
        if len(eligible) < group_size:
            return None

        group_choice: list[tuple[int, list[tuple[int, int]]]] | None = None
        for _ in range(30):
            picked = random.sample(eligible, group_size)
            current: list[tuple[int, list[tuple[int, int]]]] = []
            blocked_now = set(temp_blocked)
            towers_now = set(chosen_towers)
            ok = True
            for idx in picked:
                cand = usable[idx]
                row = int(cand["row"])
                col = int(cand["col"])
                if _touches_existing_tower(row, col, towers_now):
                    ok = False
                    break
                dr, dc = cand["dir"]
                cells = _ray_cells(occupied, reserved, row, col, dr, dc, group_size, blocked_now)
                if cells is None:
                    ok = False
                    break
                current.append((idx, cells))
                blocked_now.update(cells)
                towers_now.add((row, col))
            if ok:
                group_choice = current
                break
        if group_choice is None:
            return None

        chosen_groups.append(group_choice)
        for idx, cells in group_choice:
            used.add(idx)
            temp_blocked.update(cells)
            chosen_towers.add((int(usable[idx]["row"]), int(usable[idx]["col"])))

    colors = random.sample(PALETTE, len(chosen_groups))
    placements: list[dict[str, object]] = []
    for color, group_choice in zip(colors, chosen_groups):
        for idx, ray_cells in group_choice:
            cand = usable[idx]
            dr, dc = cand["dir"]
            placements.append(
                {
                    "row": int(cand["row"]),
                    "col": int(cand["col"]),
                    "dir": (int(dr), int(dc)),
                    "color": int(color),
                    "group_size": int(len(group_choice)),
                    "ray_cells": ray_cells,
                }
            )
    return placements


def _choose_towers(
    occupied: np.ndarray,
    reserved: np.ndarray,
    candidates: list[dict[str, int | tuple[int, int]]],
) -> list[dict[str, object]] | None:
    usable = []
    for cand in candidates:
        row = int(cand["row"])
        col = int(cand["col"])
        dr, dc = cand["dir"]
        max_run = _max_clear_run(occupied, reserved, row, col, int(dr), int(dc))
        if max_run >= 1:
            usable.append(
                {
                    "row": row,
                    "col": col,
                    "dir": (int(dr), int(dc)),
                    "max_run": int(max_run),
                }
            )
    if not usable:
        return None

    tower_counts = list(range(1, min(4, len(usable)) + 1))
    random.shuffle(tower_counts)
    for tower_count in tower_counts:
        patterns = GROUP_PATTERNS[tower_count][:]
        random.shuffle(patterns)
        for groups in patterns:
            for _ in range(25):
                placements = _assign_groups(usable, groups, occupied, reserved)
                if placements is not None:
                    return placements
    return None


def _try_add_region(grid: np.ndarray, reserved: np.ndarray) -> list[dict[str, object]] | None:
    grid_h, grid_w = grid.shape
    for _ in range(REGION_SHAPE_TRIES):
        interior = _sample_interior()
        interior_mask, boundary_mask = _build_region_patch(interior)
        if len(_components(boundary_mask)) != 1:
            continue
        local_candidates = _local_tower_candidates(interior_mask, boundary_mask)
        if len(local_candidates) < 2:
            continue

        patch_h, patch_w = boundary_mask.shape
        top_lo = EDGE_MARGIN
        left_lo = EDGE_MARGIN
        top_hi = grid_h - patch_h - EDGE_MARGIN
        left_hi = grid_w - patch_w - EDGE_MARGIN
        if top_hi < top_lo or left_hi < left_lo:
            continue

        for _ in range(REGION_PLACE_TRIES):
            top = random.randint(top_lo, top_hi)
            left = random.randint(left_lo, left_hi)
            if not _can_place_patch(grid, reserved, top, left, patch_h, patch_w):
                continue

            occupied = grid != BLACK
            boundary_points = np.argwhere(boundary_mask)
            for row, col in boundary_points:
                occupied[top + int(row), left + int(col)] = True

            global_candidates = [
                {
                    "row": top + int(cand["row"]),
                    "col": left + int(cand["col"]),
                    "dir": cand["dir"],
                }
                for cand in local_candidates
            ]
            placements = _choose_towers(occupied, reserved, global_candidates)
            if placements is None:
                continue

            for row, col in boundary_points:
                grid[top + int(row), left + int(col)] = GREEN
            for tower in placements:
                grid[int(tower["row"]), int(tower["col"])] = int(tower["color"])
                for ray_row, ray_col in tower["ray_cells"]:
                    reserved[ray_row, ray_col] = True
            return placements
    return None


def _is_valid_sample(grid: np.ndarray, tower_groups: list[int], directions_seen: set[tuple[int, int]]) -> bool:
    if not tower_groups or max(tower_groups) < 2:
        return False
    if len(directions_seen) < 2:
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    if np.array_equal(out, grid):
        return False

    regions = _region_components(grid)
    if not regions:
        return False
    for region in regions:
        towers = [(row, col) for row, col in region if int(grid[row, col]) not in (BLACK, GREEN)]
        if not towers:
            return False
        for row, col in towers:
            for dr, dc in DIRS:
                nr = row + dr
                nc = col + dc
                if _in_bounds(grid.shape[0], grid.shape[1], nr, nc) and int(grid[nr, nc]) not in (BLACK, GREEN):
                    return False
            if len(
                [
                    1
                    for dr, dc in DIRS
                    if _in_bounds(grid.shape[0], grid.shape[1], row + dr, col + dc) and int(grid[row + dr, col + dc]) == GREEN
                ]
            ) != 1:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        grid = np.full((height, width), BLACK, dtype=int)
        reserved = np.zeros((height, width), dtype=bool)

        target_regions = random.choices(REGION_COUNT_CHOICES, weights=REGION_COUNT_WEIGHTS, k=1)[0]
        all_group_sizes: list[int] = []
        directions_seen: set[tuple[int, int]] = set()
        ok = True

        for _region_idx in range(target_regions):
            placements = _try_add_region(grid, reserved)
            if placements is None:
                ok = False
                break
            all_group_sizes.extend(int(tower["group_size"]) for tower in placements)
            directions_seen.update(tuple(map(int, tower["dir"])) for tower in placements)

        if not ok:
            continue
        if _is_valid_sample(grid, all_group_sizes, directions_seen):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
