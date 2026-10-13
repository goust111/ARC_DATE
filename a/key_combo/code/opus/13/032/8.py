from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
SECTOR_DIRS = [
    (0, 1),   # E
    (1, 1),   # SE
    (1, 0),   # S
    (1, -1),  # SW
    (0, -1),  # W
    (-1, -1), # NW
    (-1, 0),  # N
    (-1, 1),  # NE
]
GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield int(r + dr), int(c + dc)


def _neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield int(r + dr), int(c + dc)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _round_nearest(x: float) -> int:
    return int(math.floor(float(x) + 0.5))


def _centroid(coords: np.ndarray) -> tuple[float, float]:
    mean = coords.mean(axis=0)
    return float(mean[0]), float(mean[1])


def _anchor_from_centroid(coords: np.ndarray) -> tuple[int, int]:
    cr, cc = _centroid(coords)
    return _round_nearest(cr), _round_nearest(cc)


def _angle_clockwise(center: tuple[float, float], point: tuple[float, float]) -> float:
    dr = float(point[0] - center[0])
    dc = float(point[1] - center[1])
    ang = math.atan2(dr, dc)
    return ang if ang >= 0 else ang + 2 * math.pi


def _build_l_path(start: tuple[int, int], end: tuple[int, int], horizontal_first: bool) -> list[tuple[int, int]]:
    sr, sc = map(int, start)
    er, ec = map(int, end)
    if (sr, sc) == (er, ec):
        return [(sr, sc)]
    if sr == er:
        step = 1 if ec >= sc else -1
        return [(sr, c) for c in range(sc, ec + step, step)]
    if sc == ec:
        step = 1 if er >= sr else -1
        return [(r, sc) for r in range(sr, er + step, step)]
    if horizontal_first:
        cstep = 1 if ec >= sc else -1
        rstep = 1 if er >= sr else -1
        path = [(sr, c) for c in range(sc, ec + cstep, cstep)]
        path.extend((r, ec) for r in range(sr + rstep, er + rstep, rstep))
        return path
    rstep = 1 if er >= sr else -1
    cstep = 1 if ec >= sc else -1
    path = [(r, sc) for r in range(sr, er + rstep, rstep)]
    path.extend((er, c) for c in range(sc + cstep, ec + cstep, cstep))
    return path


def _path_hits_foreground(path: list[tuple[int, int]], obstacle_mask: np.ndarray) -> bool:
    for r, c in path[1:-1]:
        if obstacle_mask[int(r), int(c)]:
            return True
    return False


def _divider_shapes() -> list[list[tuple[int, int]]]:
    return [
        [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
        [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (0, -2), (0, 2)],
        [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0)],
    ]


def _parse_regions(grid: np.ndarray) -> list[dict]:
    comps = _components_from_mask((grid != BG) & (grid != DIVIDER))
    regions = []
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each foreground region must be monochrome"
        color = int(next(iter(colors)))
        centroid = _centroid(coords)
        regions.append(
            {
                "coords": coords,
                "color": color,
                "area": int(coords.shape[0]),
                "centroid": centroid,
                "anchor": _anchor_from_centroid(coords),
            }
        )
    return regions


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    divider_comps = _components_from_mask(g == DIVIDER)
    assert len(divider_comps) == 1, "expected one gray divider"
    divider = divider_comps[0]
    assert divider.shape[0] in {5, 7}, "divider must be a small cross"
    divider_centroid = _centroid(divider)

    regions = _parse_regions(g)
    assert 5 <= len(regions) <= 7, "expected 5-7 surrounding regions"
    areas = [region["area"] for region in regions]
    largest_area = max(areas)
    largest = [region for region in regions if region["area"] == largest_area]
    assert len(largest) == 1, "largest region must be unique"
    distractor = largest[0]
    survivors = [region for region in regions if region is not distractor]
    survivor_areas = [region["area"] for region in survivors]
    assert len(survivor_areas) == len(set(survivor_areas)), "survivor sizes must be pairwise distinct"

    for region in regions:
        angle = _angle_clockwise(divider_centroid, region["centroid"])
        region["angle"] = angle
    ordered_all = sorted(regions, key=lambda region: region["angle"])
    ordered_survivors = sorted(survivors, key=lambda region: region["angle"])

    return {
        "grid": g,
        "divider_coords": divider,
        "divider_centroid": divider_centroid,
        "regions": regions,
        "distractor": distractor,
        "ordered_all": ordered_all,
        "ordered_survivors": ordered_survivors,
    }


def _path_plan(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    info = _analyze_input(grid)
    base = np.array(info["grid"], copy=True)
    base[info["distractor"]["coords"][:, 0], info["distractor"]["coords"][:, 1]] = BG
    obstacle_mask = base != BG

    survivors = info["ordered_survivors"]
    plans = []
    n = len(survivors)
    for idx, left in enumerate(survivors):
        right = survivors[(idx + 1) % n]
        hv = _build_l_path(left["anchor"], right["anchor"], horizontal_first=True)
        vh = _build_l_path(left["anchor"], right["anchor"], horizontal_first=False)
        hv_blocked = _path_hits_foreground(hv, obstacle_mask)
        vh_blocked = _path_hits_foreground(vh, obstacle_mask)
        chosen = hv if not hv_blocked else vh
        mode = "hv" if not hv_blocked else "vh"
        plans.append(
            {
                "left": left,
                "right": right,
                "path": chosen,
                "mode": mode,
                "hv_blocked": hv_blocked,
                "vh_blocked": vh_blocked,
            }
        )
    return base, plans


def _transform(grid: np.ndarray) -> np.ndarray:
    base, plans = _path_plan(grid)
    out = np.array(base, copy=True)
    for plan in plans:
        left_color = int(plan["left"]["color"])
        right_color = int(plan["right"]["color"])
        for i, (r, c) in enumerate(plan["path"][1:-1]):
            out[int(r), int(c)] = left_color if i % 2 == 0 else right_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the largest surrounding region, then connect the remaining regions in clockwise order with alternating-color Manhattan paths.

    Input structure:
    - The grid has black background, one small gray divider near the center, and 5-7 separated monochrome 4-connected regions around it.
    - All non-gray foreground regions have distinct colors; exactly one region has strictly more pixels than every other region.
    - The divider does not touch the colored regions.

    Rule:
    - Count the pixels in every non-divider foreground region and erase the unique largest region to black.
    - Compute the gray divider centroid and sort the surviving regions by the clockwise angle of their centroids, with rightward as 0 degrees.
    - For each adjacent pair in that cyclic order, connect the two centroid-nearest grid cells with an L-shaped Manhattan path, preferring horizontal-then-vertical unless that route intersects a foreground object, in which case use vertical-then-horizontal.
    - Paint the interior path pixels by alternating between the two region colors, starting from the earlier region's color, and let later paths overwrite earlier ones on overlaps.

    Key insight:
    - The largest region is a deliberate distractor: removing it first changes which regions are angular neighbors, so the final connection topology depends on the frequency filter.
    """
    return _transform(np.asarray(grid, dtype=int))


def _slot_offsets(grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    center_r = grid_shape[0] // 2
    center_c = grid_shape[1] // 2
    radial = min(center_r, center_c) - 3
    diag = radial - 1
    return [
        (0, radial),
        (diag, diag),
        (radial, 0),
        (diag, -diag),
        (0, -radial),
        (-diag, -diag),
        (-radial, 0),
        (-diag, diag),
    ]


def _random_region_offsets(area: int, direction: tuple[int, int]) -> list[tuple[int, int]] | None:
    dir_r, dir_c = direction
    allowed = set()
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            if max(abs(dr), abs(dc)) > 3 or abs(dr) + abs(dc) > 5:
                continue
            score = dr * dir_r + dc * dir_c
            if score < 0:
                continue
            if score == 0 and abs(dr) + abs(dc) > 3:
                continue
            allowed.add((dr, dc))

    for _ in range(250):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in allowed and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            candidates = sorted(frontier)
            weights = []
            for dr, dc in candidates:
                score = dr * dir_r + dc * dir_c
                compact = abs(dr) + abs(dc)
                balance = 3 - min(3, abs(dr + dc))
                weights.append(max(1, 8 - 2 * score - compact + balance))
            pick = random.choices(candidates, weights=weights, k=1)[0]
            cells.add(pick)
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=float)
        mean_r, mean_c = coords.mean(axis=0)
        if abs(mean_r) >= 0.45 or abs(mean_c) >= 0.45:
            continue
        if max(dr * dir_r + dc * dir_c for dr, dc in cells) <= 0:
            continue
        return sorted((int(dr), int(dc)) for dr, dc in cells)
    return None


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    occupied = {(int(r), int(c)) for r, c in np.argwhere(grid != BG)}
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or grid[r, c] != BG:
            return False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _place_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _divider_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    shape = random.choice(_divider_shapes())
    return [(center[0] + dr, center[1] + dc) for dr, dc in shape]


def _touches_anything(coords: np.ndarray, grid: np.ndarray) -> bool:
    for r, c in coords:
        for nr, nc in _neighbors8(int(r), int(c)):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != BG and (nr, nc) not in {(int(r), int(c))}:
                return True
    return False


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        info = _analyze_input(grid)
    except AssertionError:
        return False

    color_set = [region["color"] for region in info["regions"]]
    if len(color_set) != len(set(color_set)):
        return False

    divider_cells = {tuple(map(int, cell)) for cell in info["divider_coords"]}
    for region in info["regions"]:
        coords = {tuple(map(int, cell)) for cell in region["coords"]}
        for r, c in coords:
            for nr, nc in _neighbors8(r, c):
                if (nr, nc) in divider_cells:
                    return False
        if not _in_bounds(grid.shape, *region["anchor"]):
            return False

    angles = [region["angle"] for region in info["ordered_survivors"]]
    for left, right in zip(angles, angles[1:]):
        if math.isclose(left, right, abs_tol=1e-6):
            return False

    full_ids = [id(region) for region in info["ordered_all"]]
    dist_id = id(info["distractor"])
    dist_pos = full_ids.index(dist_id)
    prev_id = full_ids[(dist_pos - 1) % len(full_ids)]
    next_id = full_ids[(dist_pos + 1) % len(full_ids)]
    survivor_ids = [id(region) for region in info["ordered_survivors"]]
    if prev_id not in survivor_ids or next_id not in survivor_ids:
        return False
    prev_pos = survivor_ids.index(prev_id)
    if survivor_ids[(prev_pos + 1) % len(survivor_ids)] != next_id:
        return False

    try:
        base, plans = _path_plan(grid)
    except AssertionError:
        return False

    modes = {plan["mode"] for plan in plans}
    if modes != {"hv", "vh"}:
        return False
    if any(plan["hv_blocked"] and plan["vh_blocked"] for plan in plans):
        return False

    painted_cells: dict[tuple[int, int], int] = {}
    overlap_count = 0
    for plan in plans:
        for cell in plan["path"][1:-1]:
            if base[cell] != BG:
                return False
            overlap_count += int(cell in painted_cells)
            painted_cells[cell] = painted_cells.get(cell, 0) + 1
    if overlap_count == 0 and random.random() < 0.55:
        return False

    expected = _transform(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice([17, 19, 21])
    w = random.choice([17, 19, 21])
    grid = np.full((h, w), BG, dtype=int)
    center = (h // 2, w // 2)

    divider = _divider_cells(center)
    _place_cells(grid, divider, DIVIDER)

    total_regions = random.randint(5, 7)
    slot_ids = sorted(random.sample(range(8), total_regions))
    offsets = _slot_offsets(grid.shape)
    colors = random.sample(PALETTE, total_regions)
    survivor_sizes = random.sample(list(range(3, 9)), total_regions - 1)
    largest_size = random.randint(max(survivor_sizes) + 1, 9)
    distractor_slot = random.choice(slot_ids)

    size_by_slot = {}
    shuffled_survivors = survivor_sizes[:]
    random.shuffle(shuffled_survivors)
    survivor_iter = iter(shuffled_survivors)
    for slot_id in slot_ids:
        size_by_slot[slot_id] = largest_size if slot_id == distractor_slot else next(survivor_iter)

    color_by_slot = {slot_id: color for slot_id, color in zip(slot_ids, colors)}
    for slot_id in sorted(slot_ids, key=lambda s: size_by_slot[s], reverse=True):
        anchor = (center[0] + offsets[slot_id][0], center[1] + offsets[slot_id][1])
        shape = _random_region_offsets(size_by_slot[slot_id], SECTOR_DIRS[slot_id])
        if shape is None:
            return None
        cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in shape]
        if not _can_place(grid, cells):
            return None
        _place_cells(grid, cells, color_by_slot[slot_id])

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
