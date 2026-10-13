from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator configuration.
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
ROOM_COUNT_RANGE = (1, 4)
ROOM_RETRIES = 160
PLACEMENT_RETRIES = 220
GEN_RETRIES = 400
ROOM_GAP = 1

BASE_H_RANGE = (4, 8)
BASE_W_RANGE = (4, 8)
ARM_DEPTH_RANGE = (1, 3)
ARM_COUNT_RANGE = (0, 3)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    comps = _components(grid != WALL)
    regions: list[list[tuple[int, int]]] = []
    for comp in comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        if any(int(grid[r, c]) not in (BLACK, WALL) for r, c in comp):
            regions.append(comp)
    return regions


def _region_mask(points: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _wall_distance_map(region_mask: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    h, w = region_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in np.argwhere(region_mask):
        rr = int(r)
        cc = int(c)
        if any(0 <= nr < h and 0 <= nc < w and bool(wall_mask[nr, nc]) for nr, nc in _neighbors4(rr, cc)):
            dist[rr, cc] = 0
            q.append((rr, cc))

    assert q, "every enclosed region must touch its wall"

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(region_mask[nr, nc]) and int(dist[nr, nc]) < 0:
                dist[nr, nc] = int(dist[r, c]) + 1
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed grey-walled region from its seeds in priority order.

    Priority is determined by the seed's distance to the nearest wall layer:
    smaller distance first, then row-major order for ties.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    wall_mask = g == WALL

    for region in _find_enclosed_regions(g):
        region_mask = _region_mask(region, g.shape)
        dist = _wall_distance_map(region_mask, wall_mask)

        seeds: list[tuple[int, int, int, int]] = []
        for r, c in region:
            color = int(g[r, c])
            if color not in (BLACK, WALL):
                seeds.append((int(dist[r, c]), r, c, color))

        if not seeds:
            continue

        seeds.sort(key=lambda item: (item[0], item[1], item[2]))
        for _prio, sr, sc, color in seeds:
            q = deque([(sr, sc)])
            while q:
                r, c = q.popleft()
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                        if bool(region_mask[nr, nc]) and int(out[nr, nc]) == BLACK:
                            out[nr, nc] = color
                            q.append((nr, nc))

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _rect(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _build_interior_shape() -> set[tuple[int, int]]:
    base_h = random.randint(*BASE_H_RANGE)
    base_w = random.randint(*BASE_W_RANGE)
    cells = _rect(0, 0, base_h, base_w)

    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    arm_budget = random.randint(*ARM_COUNT_RANGE)
    arms_added = 0

    for side in sides:
        if arms_added >= arm_budget:
            break
        if random.random() < 0.3:
            continue

        depth = random.randint(*ARM_DEPTH_RANGE)
        if side in ("top", "bottom"):
            span_hi = max(2, base_w - 1)
            span = random.randint(2, span_hi)
            start = random.randint(0, base_w - span)
            if side == "top":
                cells |= _rect(-depth, start, depth, span)
            else:
                cells |= _rect(base_h, start, depth, span)
        else:
            span_hi = max(2, base_h - 1)
            span = random.randint(2, span_hi)
            start = random.randint(0, base_h - span)
            if side == "left":
                cells |= _rect(start, -depth, span, depth)
            else:
                cells |= _rect(start, base_w, span, depth)
        arms_added += 1

    return _normalize(cells)


def _room_from_interior(interior: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    shifted = {(r + 1, c + 1) for r, c in interior}
    wall: set[tuple[int, int]] = set()
    for r, c in shifted:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                pos = (r + dr, c + dc)
                if pos not in shifted:
                    wall.add(pos)

    max_r = max(max(r for r, _ in shifted), max(r for r, _ in wall))
    max_c = max(max(c for _, c in shifted), max(c for _, c in wall))
    patch = np.full((max_r + 1, max_c + 1), BLACK, dtype=int)
    for r, c in wall:
        patch[r, c] = WALL

    region_mask = np.zeros_like(patch, dtype=bool)
    for r, c in shifted:
        region_mask[r, c] = True
    return patch, region_mask


def _spaced(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(abs(cell[0] - r) + abs(cell[1] - c) >= 2 for r, c in chosen)


def _search_from_groups(
    groups: list[list[tuple[int, int]]],
    idx: int,
    chosen: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if idx == len(groups):
        return chosen.copy()
    order = groups[idx][:]
    random.shuffle(order)
    for cell in order:
        if not _spaced(cell, chosen):
            continue
        chosen.append(cell)
        found = _search_from_groups(groups, idx + 1, chosen)
        if found is not None:
            return found
        chosen.pop()
    return None


def _pick_same_layer(cells: list[tuple[int, int]], k: int) -> list[tuple[int, int]] | None:
    order = cells[:]
    random.shuffle(order)
    chosen: list[tuple[int, int]] = []
    for cell in order:
        if _spaced(cell, chosen):
            chosen.append(cell)
            if len(chosen) == k:
                return chosen
    return None


def _choose_seed_positions(region_mask: np.ndarray, wall_mask: np.ndarray, mode: str) -> list[tuple[int, int]] | None:
    dist = _wall_distance_map(region_mask, wall_mask)
    by_dist: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(region_mask):
        by_dist[int(dist[r, c])].append((int(r), int(c)))

    for cells in by_dist.values():
        cells.sort()

    all_cells = [cell for cells in by_dist.values() for cell in cells]
    all_cells.sort()
    if mode == "single":
        pool = all_cells[:]
        random.shuffle(pool)
        return [pool[0]] if pool else None

    if mode == "tie":
        candidate_layers = [d for d, cells in by_dist.items() if len(cells) >= 2]
        random.shuffle(candidate_layers)
        for d in candidate_layers:
            ks = [3, 2] if len(by_dist[d]) >= 3 else [2]
            random.shuffle(ks)
            for k in ks:
                picked = _pick_same_layer(by_dist[d], k)
                if picked is not None:
                    return picked
        return None

    unique_dists = sorted(by_dist)
    if len(unique_dists) < 2:
        return None
    size_options = [k for k in (2, 3) if len(unique_dists) >= k]
    random.shuffle(size_options)
    for k in size_options:
        dist_combos = list(combinations(unique_dists, k))
        random.shuffle(dist_combos)
        for combo in dist_combos:
            groups = [by_dist[d] for d in combo]
            picked = _search_from_groups(groups, 0, [])
            if picked is not None:
                return picked
    return None


def _classify_priority_mode(grid: np.ndarray, region: list[tuple[int, int]]) -> str:
    region_mask = _region_mask(region, grid.shape)
    dist = _wall_distance_map(region_mask, grid == WALL)
    seed_dists = sorted(
        (int(dist[r, c]), r, c)
        for r, c in region
        if int(grid[r, c]) not in (BLACK, WALL)
    )
    if len(seed_dists) == 1:
        return "single"
    only_dists = [d for d, _r, _c in seed_dists]
    return "tie" if len(set(only_dists)) < len(only_dists) else "distinct"


def _build_room(mode: str) -> np.ndarray | None:
    for _ in range(ROOM_RETRIES):
        interior = _build_interior_shape()
        patch, region_mask = _room_from_interior(interior)
        seeds = _choose_seed_positions(region_mask, patch == WALL, mode)
        if seeds is None:
            continue

        colors = random.sample(SEED_COLORS, len(seeds))
        room = patch.copy()
        for color, (r, c) in zip(colors, seeds, strict=False):
            room[r, c] = color

        regions = _find_enclosed_regions(room)
        if len(regions) != 1:
            continue
        region = regions[0]
        seed_colors = [int(room[r, c]) for r, c in region if int(room[r, c]) not in (BLACK, WALL)]
        if not (1 <= len(seed_colors) <= 3):
            continue
        if len(seed_colors) != len(set(seed_colors)):
            continue
        if _classify_priority_mode(room, region) != mode:
            continue
        if int(np.count_nonzero(main(room) != room)) == 0:
            continue
        return room
    return None


def _place_room(grid: np.ndarray, reserved: np.ndarray, room: np.ndarray) -> bool:
    h, w = grid.shape
    rh, rw = room.shape
    if rh + 2 > h or rw + 2 > w:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(1, h - rh - 1)
        c0 = random.randint(1, w - rw - 1)
        r1 = r0 + rh
        c1 = c0 + rw

        rr0 = max(0, r0 - ROOM_GAP)
        cc0 = max(0, c0 - ROOM_GAP)
        rr1 = min(h, r1 + ROOM_GAP)
        cc1 = min(w, c1 + ROOM_GAP)
        if bool(reserved[rr0:rr1, cc0:cc1].any()):
            continue

        room_mask = room != BLACK
        grid[r0:r1, c0:c1][room_mask] = room[room_mask]
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def generate_input() -> np.ndarray:
    """
    Generate several closed rooms with 1-3 seeds each.

    The first room is forced to realize one of the semantic branches:
    a single seed, multiple seeds with distinct wall distances, or a tie
    resolved by row-major order.
    """
    modes = ("single", "distinct", "tie")
    for _ in range(GEN_RETRIES):
        required_mode = random.choice(modes)
        room_count = random.randint(*ROOM_COUNT_RANGE)
        room_modes = [required_mode] + [random.choice(modes) for _ in range(room_count - 1)]
        random.shuffle(room_modes)

        rooms: list[np.ndarray] = []
        ok = True
        for mode in room_modes:
            room = _build_room(mode)
            if room is None:
                ok = False
                break
            rooms.append(room)
        if not ok:
            continue

        grid_h = random.randint(*GRID_H_RANGE)
        grid_w = random.randint(*GRID_W_RANGE)
        grid = np.full((grid_h, grid_w), BLACK, dtype=int)
        reserved = np.zeros_like(grid, dtype=bool)

        placed = True
        for room in sorted(rooms, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
            if not _place_room(grid, reserved, room):
                placed = False
                break
        if not placed:
            continue

        regions = _find_enclosed_regions(grid)
        if len(regions) != room_count:
            continue
        region_modes = [_classify_priority_mode(grid, region) for region in regions]
        if required_mode not in region_modes:
            continue
        if not all(1 <= sum(int(grid[r, c]) not in (BLACK, WALL) for r, c in region) <= 3 for region in regions):
            continue
        if int(np.count_nonzero(main(grid) != grid)) == 0:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
