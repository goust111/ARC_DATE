from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
MAX_RETRIES = 300


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS8:
        yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_regions(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[int]]:
    h, w = grid.shape
    regions: list[list[tuple[int, int]]] = []
    ids = np.full((h, w), -1, dtype=int)
    for comp in _components(grid == BG, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        region_id = len(regions)
        for r, c in comp:
            ids[r, c] = region_id
        regions.append(comp)
    return regions, ids


def _build_component_region_map(
    grid: np.ndarray, regions: list[list[tuple[int, int]]], region_ids: np.ndarray
) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    grey_components = _components(grid == GREY, connectivity=8)
    cell_to_component: dict[tuple[int, int], int] = {}
    component_to_region: dict[int, int] = {}

    for comp_idx, comp in enumerate(grey_components):
        adjacent_regions: set[int] = set()
        for r, c in comp:
            cell_to_component[(r, c)] = comp_idx
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    region_id = int(region_ids[nr, nc])
                    if region_id >= 0:
                        adjacent_regions.add(region_id)
        if len(adjacent_regions) == 1:
            component_to_region[comp_idx] = next(iter(adjacent_regions))

    return component_to_region, cell_to_component


def _visible_component(
    grid: np.ndarray, seed: tuple[int, int], grey_cell_to_component: dict[tuple[int, int], int]
) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = seed
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        step = 1
        while True:
            nr, nc = r + dr * step, c + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                break
            value = int(grid[nr, nc])
            if value == BG:
                step += 1
                continue
            if value == GREY:
                comp = grey_cell_to_component.get((nr, nc))
                if comp is not None:
                    candidates.append((step, comp))
            break
    if not candidates:
        return None
    best_dist = min(dist for dist, _ in candidates)
    best_components = {comp for dist, comp in candidates if dist == best_dist}
    if len(best_components) != 1:
        return None
    return best_dist, next(iter(best_components))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed black regions using colors from external seed pixels that have a
    straight horizontal or vertical background corridor to a grey contour.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    regions, region_ids = _extract_regions(g)
    component_to_region, grey_cell_to_component = _build_component_region_map(g, regions, region_ids)

    region_to_choice: dict[int, tuple[int, int]] = {}
    seeds = [(int(r), int(c)) for r, c in np.argwhere((g != BG) & (g != GREY))]
    for seed in seeds:
        visible = _visible_component(g, seed, grey_cell_to_component)
        if visible is None:
            continue
        _dist_to_wall, comp_idx = visible
        region_id = component_to_region.get(comp_idx)
        if region_id is None:
            continue
        region_dist = min(abs(seed[0] - r) + abs(seed[1] - c) for r, c in regions[region_id])
        current = region_to_choice.get(region_id)
        if current is None or region_dist < current[0]:
            region_to_choice[region_id] = (region_dist, int(g[seed]))

    for region_id, (_dist, color) in region_to_choice.items():
        for r, c in regions[region_id]:
            out[r, c] = color
    return out


def _draw_rect_loop(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left:right + 1] = GREY
    grid[bottom, left:right + 1] = GREY
    grid[top:bottom + 1, left] = GREY
    grid[top:bottom + 1, right] = GREY


def _draw_open_rect(
    grid: np.ndarray, top: int, left: int, height: int, width: int, side: str, gap_len: int
) -> None:
    _draw_rect_loop(grid, top, left, height, width)
    bottom = top + height - 1
    right = left + width - 1
    if side in ("top", "bottom"):
        gap_start = left + 1 + (width - 2 - gap_len) // 2
        rr = top if side == "top" else bottom
        grid[rr, gap_start:gap_start + gap_len] = BG
    else:
        gap_start = top + 1 + (height - 2 - gap_len) // 2
        cc = left if side == "left" else right
        grid[gap_start:gap_start + gap_len, cc] = BG


def _bbox_clear(used: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 2) -> bool:
    h, w = used.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not used[r0:r1, c0:c1].any()


def _mark_bbox(used: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    used[top:top + height, left:left + width] = True


def _path_is_clear(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(int(grid[r, c]) == BG for r, c in cells)


def _candidate_seed_positions(
    grid: np.ndarray, rect: tuple[int, int, int, int], side: str
) -> list[tuple[tuple[int, int], int]]:
    top, left, height, width = rect
    h, w = grid.shape
    bottom = top + height - 1
    right = left + width - 1
    out: list[tuple[tuple[int, int], int]] = []
    if side == "left":
        for row in range(top + 1, bottom):
            for dist in range(2, left + 1):
                seed = (row, left - dist)
                corridor = [(row, c) for c in range(seed[1] + 1, left)]
                if _path_is_clear(grid, corridor):
                    out.append((seed, dist))
    elif side == "right":
        for row in range(top + 1, bottom):
            for dist in range(2, w - right):
                seed = (row, right + dist)
                corridor = [(row, c) for c in range(right + 1, seed[1])]
                if _path_is_clear(grid, corridor):
                    out.append((seed, dist))
    elif side == "top":
        for col in range(left + 1, right):
            for dist in range(2, top + 1):
                seed = (top - dist, col)
                corridor = [(r, col) for r in range(seed[0] + 1, top)]
                if _path_is_clear(grid, corridor):
                    out.append((seed, dist))
    else:
        for col in range(left + 1, right):
            for dist in range(2, h - bottom):
                seed = (bottom + dist, col)
                corridor = [(r, col) for r in range(bottom + 1, seed[0])]
                if _path_is_clear(grid, corridor):
                    out.append((seed, dist))
    return out


def _place_effective_seed(
    rng: np.random.Generator,
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    color: int,
    forbidden: set[tuple[int, int]],
    prefer_far: bool | None = None,
) -> tuple[int, int, int] | None:
    options: list[tuple[tuple[int, int], int]] = []
    for side in ("left", "right", "top", "bottom"):
        options.extend(_candidate_seed_positions(grid, rect, side))
    options = [(seed, dist) for seed, dist in options if seed not in forbidden and int(grid[seed]) == BG]
    if not options:
        return None
    dists = sorted({dist for _, dist in options})
    if prefer_far is True:
        target_dist = dists[-1]
        options = [(seed, dist) for seed, dist in options if dist == target_dist]
    elif prefer_far is False:
        target_dist = dists[0]
        options = [(seed, dist) for seed, dist in options if dist == target_dist]
    seed, dist = options[int(rng.integers(len(options)))]
    grid[seed] = color
    forbidden.add(seed)
    return seed[0], seed[1], dist


def _place_stray_seed(
    rng: np.random.Generator,
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    color: int,
) -> bool:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG or (r, c) in forbidden:
                continue
            visible = False
            for dr, dc in DIRS4:
                step = 1
                while True:
                    nr, nc = r + dr * step, c + dc * step
                    if not (0 <= nr < h and 0 <= nc < w):
                        break
                    value = int(grid[nr, nc])
                    if value == BG:
                        step += 1
                        continue
                    visible = value == GREY
                    break
                if visible:
                    break
            if not visible:
                candidates.append((r, c))
    if not candidates:
        return False
    pos = candidates[int(rng.integers(len(candidates)))]
    grid[pos] = color
    forbidden.add(pos)
    return True


def _place_open_seed(
    rng: np.random.Generator,
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    color: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    options: list[tuple[tuple[int, int], int]] = []
    for side in ("left", "right", "top", "bottom"):
        options.extend(_candidate_seed_positions(grid, rect, side))
    options = [(seed, dist) for seed, dist in options if seed not in forbidden and int(grid[seed]) == BG]
    if not options:
        return False
    seed, _dist = options[int(rng.integers(len(options)))]
    grid[seed] = color
    forbidden.add(seed)
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 25))
    w = int(rng.integers(16, 25))
    grid = np.zeros((h, w), dtype=int)
    used = np.zeros((h, w), dtype=bool)

    closed_rects: list[tuple[int, int, int, int]] = []
    n_closed = int(rng.integers(1, 4))
    for _ in range(n_closed):
        placed = False
        for _ in range(80):
            rh = int(rng.integers(5, 9))
            rw = int(rng.integers(5, 9))
            if rh >= h - 2 or rw >= w - 2:
                continue
            top = int(rng.integers(1, h - rh))
            left = int(rng.integers(1, w - rw))
            if not _bbox_clear(used, top, left, rh, rw, pad=2):
                continue
            _draw_rect_loop(grid, top, left, rh, rw)
            _mark_bbox(used, top, left, rh, rw)
            closed_rects.append((top, left, rh, rw))
            placed = True
            break
        if not placed:
            return None

    open_rects: list[tuple[int, int, int, int]] = []
    n_open = int(rng.integers(0, 3))
    for _ in range(n_open):
        placed = False
        for _ in range(60):
            rh = int(rng.integers(5, 8))
            rw = int(rng.integers(5, 8))
            if rh >= h - 2 or rw >= w - 2:
                continue
            top = int(rng.integers(1, h - rh))
            left = int(rng.integers(1, w - rw))
            if not _bbox_clear(used, top, left, rh, rw, pad=2):
                continue
            side = ("left", "right", "top", "bottom")[int(rng.integers(4))]
            gap_len = int(rng.integers(1, 3))
            if (side in ("left", "right") and rh - 2 <= gap_len) or (side in ("top", "bottom") and rw - 2 <= gap_len):
                continue
            _draw_open_rect(grid, top, left, rh, rw, side, gap_len)
            _mark_bbox(used, top, left, rh, rw)
            open_rects.append((top, left, rh, rw))
            placed = True
            break
        if not placed:
            return None

    forbidden: set[tuple[int, int]] = set()
    palette = list(rng.permutation(SEED_COLORS))
    color_idx = 0
    active_rects = list(rng.permutation(len(closed_rects)))
    n_active = int(rng.integers(1, len(closed_rects) + 1))
    active_set = set(active_rects[:n_active])

    for rect_idx, rect in enumerate(closed_rects):
        if rect_idx not in active_set:
            continue
        if color_idx >= len(palette):
            return None
        primary_color = int(palette[color_idx])
        color_idx += 1
        if _place_effective_seed(rng, grid, rect, primary_color, forbidden, prefer_far=False) is None:
            return None

        if rng.random() < 0.5:
            if color_idx >= len(palette):
                return None
            competitor_color = int(palette[color_idx])
            color_idx += 1
            if _place_effective_seed(rng, grid, rect, competitor_color, forbidden, prefer_far=True) is None:
                return None

    if open_rects and color_idx < len(palette) and rng.random() < 0.75:
        _place_open_seed(rng, grid, open_rects[int(rng.integers(len(open_rects)))], int(palette[color_idx]), forbidden)
        color_idx += 1

    n_strays = int(rng.integers(0, 3))
    for _ in range(n_strays):
        if color_idx >= len(palette):
            break
        if _place_stray_seed(rng, grid, forbidden, int(palette[color_idx])):
            color_idx += 1

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    regions, region_ids = _extract_regions(grid)
    component_to_region, grey_cell_to_component = _build_component_region_map(grid, regions, region_ids)
    filled_regions = 0
    seed_total = 0
    seed_hits = 0
    for seed in [(int(r), int(c)) for r, c in np.argwhere((grid != BG) & (grid != GREY))]:
        seed_total += 1
        visible = _visible_component(grid, seed, grey_cell_to_component)
        if visible is not None and visible[1] in component_to_region:
            seed_hits += 1
    for region_id, region in enumerate(regions):
        colors = {int(out[r, c]) for r, c in region}
        if len(colors) == 1 and next(iter(colors)) != BG:
            filled_regions += 1
    if filled_regions == 0 or seed_total == 0 or seed_hits == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
