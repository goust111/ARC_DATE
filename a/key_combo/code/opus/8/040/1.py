from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_shared_arc_utils_for_040_1", _ARC_UTILS_PATH)
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

get_bg = _ARC_UTILS.get_bg
find_components = _ARC_UTILS.find_components
neighbors8 = _ARC_UTILS.neighbors8


BG = 0
START = 5
GLYPH = 7
WALL_COLORS = (1, 2, 3, 4)
EDGE_NAMES = ("top", "bottom", "left", "right")

GLYPH_TEMPLATES = {
    3: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    4: np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    5: np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int),
}


def _band_mask(shape: tuple[int, int], edge: str) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    if edge == "top":
        mask[:3, :] = True
    elif edge == "bottom":
        mask[h - 3 :, :] = True
    elif edge == "left":
        mask[:, :3] = True
    else:
        mask[:, w - 3 :] = True
    return mask


def _corner_from_start(start: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, start)
    h, w = map(int, shape)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    return min(corners, key=lambda rc: abs(r - rc[0]) + abs(c - rc[1]))


def _initial_direction(start: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    cr, cc = _corner_from_start(start, shape)
    dr = 1 if cr == 0 else -1
    dc = 1 if cc == 0 else -1
    return int(dr), int(dc)


def _edge_for_component(coords: np.ndarray, shape: tuple[int, int]) -> str | None:
    h, w = map(int, shape)
    rs = coords[:, 0]
    cs = coords[:, 1]
    if np.all(rs < 3):
        return "top"
    if np.all(rs >= h - 3):
        return "bottom"
    if np.all(cs < 3):
        return "left"
    if np.all(cs >= w - 3):
        return "right"
    return None


def _parse_instructions(grid: np.ndarray) -> tuple[tuple[int, int], str, list[int], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    start_pts = np.argwhere(g == START)
    assert start_pts.shape[0] == 1
    start = tuple(map(int, start_pts[0]))

    comps = find_components(g, bg=get_bg(g), connectivity=8, monochromatic=True, colors=[GLYPH], return_format="dict")
    glyph_comps: list[dict] = []
    by_edge = {edge: [] for edge in EDGE_NAMES}
    for comp in comps:
        area = int(comp["area"])
        if not (3 <= area <= 7):
            continue
        coords = comp["coords"]
        edge = _edge_for_component(coords, (h, w))
        if edge is None:
            continue
        glyph_comps.append(comp)
        by_edge[edge].append(comp)

    preferred = None
    r0, c0 = start
    if r0 == 0:
        preferred = "top"
    elif r0 == h - 1:
        preferred = "bottom"
    elif c0 == 0:
        preferred = "left"
    elif c0 == w - 1:
        preferred = "right"

    if preferred is not None and by_edge[preferred]:
        edge = preferred
    else:
        edge = max(EDGE_NAMES, key=lambda name: len(by_edge[name]))
    glyphs = by_edge[edge]
    assert 3 <= len(glyphs) <= 4

    if edge in ("top", "bottom"):
        glyphs.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    else:
        glyphs.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
    budgets = [int(comp["area"]) for comp in glyphs]

    wall_map: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere((g != BG) & (g != START) & (g != GLYPH)):
        wall_map[(int(r), int(c))] = int(g[int(r), int(c)])
    return start, edge, budgets, wall_map


def _advance_one_counted(
    state: tuple[int, int, int, int, int],
    shape: tuple[int, int],
    wall_map: dict[tuple[int, int], int],
) -> tuple[tuple[int, int, int, int, int], tuple[int, int], bool, int]:
    r, c, dr, dc, color = map(int, state)
    h, w = map(int, shape)
    boundary_bounces = 0
    while True:
        nr = int(r + dr)
        nc = int(c + dc)
        out_r = nr < 0 or nr >= h
        out_c = nc < 0 or nc >= w
        if out_r or out_c:
            if out_r:
                dr = -int(dr)
            if out_c:
                dc = -int(dc)
            boundary_bounces += 1
            continue
        if (nr, nc) in wall_map:
            color = int(wall_map[(nr, nc)])
            return (nr, nc, -int(dr), -int(dc), color), (nr, nc), True, int(boundary_bounces)
        return (nr, nc, dr, dc, color), (nr, nc), False, int(boundary_bounces)


def _simulate(
    base_grid: np.ndarray,
    start: tuple[int, int],
    budgets: list[int],
    wall_map: dict[tuple[int, int], int],
    *,
    band_mask: np.ndarray | None = None,
    glyph_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    out = np.asarray(base_grid, dtype=int).copy()
    state = (int(start[0]), int(start[1]), *_initial_direction(start, out.shape), START)
    stats = {
        "hit_count": 0,
        "boundary_bounces": 0,
        "path_cells": [],
        "path_set": set(),
        "hit_colors": set(),
        "touched_band": False,
        "touched_glyph": False,
        "revisited_start": False,
    }

    for budget in budgets:
        for _ in range(int(budget)):
            state, landed, hit_wall, bounce_count = _advance_one_counted(state, out.shape, wall_map)
            stats["boundary_bounces"] += int(bounce_count)
            r, c = landed
            if band_mask is not None and band_mask[r, c] and landed != start:
                stats["touched_band"] = True
            if glyph_mask is not None and glyph_mask[r, c]:
                stats["touched_glyph"] = True
            if landed == start:
                stats["revisited_start"] = True
            if out[r, c] != GLYPH:
                out[r, c] = int(state[4])
            if hit_wall:
                stats["hit_count"] += 1
                stats["hit_colors"].add(int(state[4]))
            stats["path_cells"].append((int(r), int(c)))
            stats["path_set"].add((int(r), int(c)))
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace one diagonal ray whose per-stage step budgets are encoded by edge glyph areas.

    Input structure:
    - A black background contains one gray start pixel near a corner, 3-4 orange 3x3 glyphs packed along one edge, and several isolated interior wall pixels in distinct colors.
    - Each orange glyph is read only by its pixel count, not by its exact shape.
    - The start edge and the glyph edge match, while the walls stay away from the glyph band.

    Rule:
    - Read the orange glyphs along the start edge in reading order; each glyph's area gives one stage's maximum number of counted ray steps.
    - Launch a gray diagonal ray away from the nearest corner. Moving onto a wall counts as one step, flips both diagonal axes, and changes the ray color to the wall color.
    - Trying to leave the grid causes a free boundary bounce that flips only the crossed axis and does not consume stage budget; normal cells are painted with the current ray color.
    - After all stage budgets are exhausted, return the fully painted grid while leaving the orange glyphs unchanged.

    Key insight:
    - The glyphs encode only how long each stage lasts; the actual geometry comes from repeated boundary and wall reflections, with wall colors propagating forward along the ray.
    """
    g = np.asarray(grid, dtype=int)
    start, edge, budgets, wall_map = _parse_instructions(g)
    out, _stats = _simulate(g, start, budgets, wall_map, band_mask=_band_mask(g.shape, edge))
    return out


def _stamp_glyphs(grid: np.ndarray, edge: str, origins: list[int], areas: list[int]) -> None:
    if edge in ("top", "bottom"):
        row0 = 0 if edge == "top" else grid.shape[0] - 3
        for c0, area in zip(origins, areas):
            mask = GLYPH_TEMPLATES[int(area)].astype(bool)
            grid[row0 : row0 + 3, int(c0) : int(c0) + 3][mask] = GLYPH
    else:
        col0 = 0 if edge == "left" else grid.shape[1] - 3
        for r0, area in zip(origins, areas):
            mask = GLYPH_TEMPLATES[int(area)].astype(bool)
            grid[int(r0) : int(r0) + 3, col0 : col0 + 3][mask] = GLYPH


def _layout_glyphs(shape: tuple[int, int], edge: str, start: tuple[int, int], areas: list[int]) -> list[int] | None:
    h, w = map(int, shape)
    count = len(areas)
    gaps = [random.randint(1, 3) for _ in range(count - 1)]
    total_span = 3 * count + sum(gaps)

    if edge in ("top", "bottom"):
        reserve_left = 5 if start[1] <= 2 else 1
        reserve_right = 5 if start[1] >= w - 3 else 1
        lo = reserve_left
        hi = w - reserve_right - total_span
        if hi < lo:
            return None
        cur = random.randint(lo, hi)
        origins = []
        for idx in range(count):
            origins.append(cur)
            cur += 3
            if idx < count - 1:
                cur += gaps[idx]
        return origins

    reserve_top = 5 if start[0] <= 2 else 1
    reserve_bottom = 5 if start[0] >= h - 3 else 1
    lo = reserve_top
    hi = h - reserve_bottom - total_span
    if hi < lo:
        return None
    cur = random.randint(lo, hi)
    origins = []
    for idx in range(count):
        origins.append(cur)
        cur += 3
        if idx < count - 1:
            cur += gaps[idx]
    return origins


def _start_for_edge(shape: tuple[int, int], edge: str) -> tuple[int, int]:
    h, w = map(int, shape)
    dist = random.choice((1, 2))
    if edge == "top":
        return (0, dist) if random.random() < 0.5 else (0, w - 1 - dist)
    if edge == "bottom":
        return (h - 1, dist) if random.random() < 0.5 else (h - 1, w - 1 - dist)
    if edge == "left":
        return (dist, 0) if random.random() < 0.5 else (h - 1 - dist, 0)
    return (dist, w - 1) if random.random() < 0.5 else (h - 1 - dist, w - 1)


def _is_valid_wall_spot(
    cell: tuple[int, int],
    shape: tuple[int, int],
    band_mask: np.ndarray,
    reserved: set[tuple[int, int]],
    wall_map: dict[tuple[int, int], int],
    history: set[tuple[int, int]],
) -> bool:
    r, c = map(int, cell)
    h, w = map(int, shape)
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if band_mask[r, c]:
        return False
    if cell in reserved or cell in wall_map or cell in history:
        return False
    for wr, wc in wall_map:
        if max(abs(r - wr), abs(c - wc)) <= 1:
            return False
    return True


def _plan_hit_walls(
    shape: tuple[int, int],
    start: tuple[int, int],
    budgets: list[int],
    band_mask: np.ndarray,
    reserved: set[tuple[int, int]],
    colors: list[int],
) -> dict[tuple[int, int], int] | None:
    total_steps = sum(map(int, budgets))
    wall_map: dict[tuple[int, int], int] = {}
    state = (int(start[0]), int(start[1]), *_initial_direction(start, shape), START)
    counted_done = 0
    history = {start}

    for idx, color in enumerate(colors):
        remaining_steps = total_steps - counted_done
        future: list[tuple[int, tuple[int, int]]] = []
        temp_state = state
        max_lookahead = min(remaining_steps, 12)
        for dist in range(1, max_lookahead + 1):
            temp_state, landed, _hit_existing, _bounces = _advance_one_counted(temp_state, shape, wall_map)
            if _is_valid_wall_spot(landed, shape, band_mask, reserved, wall_map, history) and dist >= 2:
                future.append((dist, landed))
        if not future:
            return None

        min_future = 1 if idx == len(colors) - 1 else 2
        future = [item for item in future if total_steps - (counted_done + item[0]) >= min_future]
        if not future:
            return None

        dist, landed = random.choice(future[: min(len(future), 6)])
        wall_map[landed] = int(color)
        for _ in range(dist):
            state, step_cell, hit_wall, _bounces = _advance_one_counted(state, shape, wall_map)
            history.add(step_cell)
        if step_cell != landed or not hit_wall:
            return None
        counted_done += int(dist)

    return wall_map


def _add_distractor_walls(
    wall_map: dict[tuple[int, int], int],
    shape: tuple[int, int],
    band_mask: np.ndarray,
    reserved: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    colors: list[int],
) -> bool:
    h, w = map(int, shape)
    for color in colors:
        candidates = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                cell = (int(r), int(c))
                if band_mask[r, c] or cell in reserved or cell in forbidden or cell in wall_map:
                    continue
                ok = True
                for wr, wc in wall_map:
                    if max(abs(r - wr), abs(c - wc)) <= 1:
                        ok = False
                        break
                if ok:
                    candidates.append(cell)
        if not candidates:
            return False
        wall_map[random.choice(candidates)] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(15, 22)
    edge = random.choice(EDGE_NAMES)
    start = _start_for_edge((h, w), edge)

    glyph_count = random.choice((3, 4))
    areas = random.sample((3, 4, 5, 6, 7), glyph_count)
    budgets = list(map(int, areas))

    glyph_origins = _layout_glyphs((h, w), edge, start, areas)
    if glyph_origins is None:
        return None

    band_mask = _band_mask((h, w), edge)
    grid = np.zeros((h, w), dtype=int)
    grid[start] = START
    _stamp_glyphs(grid, edge, glyph_origins, areas)

    glyph_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == GLYPH)}
    glyph_mask = grid == GLYPH
    reserved = set(glyph_cells)
    reserved.add((int(start[0]), int(start[1])))

    total_wall_count = random.choice((3, 4))
    hit_wall_count = random.choice((2, 3))
    if hit_wall_count > total_wall_count:
        hit_wall_count = total_wall_count
    wall_colors = random.sample(list(WALL_COLORS), total_wall_count)

    hit_walls = _plan_hit_walls((h, w), start, budgets, band_mask, reserved, wall_colors[:hit_wall_count])
    if hit_walls is None:
        return None

    for cell, color in hit_walls.items():
        grid[cell] = int(color)

    provisional_out, provisional_stats = _simulate(
        grid,
        start,
        budgets,
        hit_walls,
        band_mask=band_mask,
        glyph_mask=glyph_mask,
    )
    if provisional_stats["hit_count"] < hit_wall_count:
        return None
    if provisional_stats["boundary_bounces"] < 1:
        return None
    if provisional_stats["touched_glyph"]:
        return None
    if len(provisional_stats["path_set"]) < max(7, sum(budgets) // 2):
        return None
    if len(provisional_stats["hit_colors"]) < min(2, hit_wall_count):
        return None

    if not _add_distractor_walls(
        hit_walls,
        (h, w),
        band_mask,
        reserved,
        provisional_stats["path_set"],
        wall_colors[hit_wall_count:],
    ):
        return None

    final_grid = np.zeros((h, w), dtype=int)
    final_grid[start] = START
    _stamp_glyphs(final_grid, edge, glyph_origins, areas)
    for cell, color in hit_walls.items():
        final_grid[cell] = int(color)

    final_out = main(final_grid)
    if np.array_equal(final_out, final_grid):
        return None
    if np.count_nonzero(final_out != final_grid) < max(7, sum(budgets) // 2):
        return None
    return final_grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 040/1")
