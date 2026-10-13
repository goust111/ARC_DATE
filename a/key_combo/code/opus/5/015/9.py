import random
from collections import defaultdict

import numpy as np


WALL = 9
BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _first_wall_row(grid: np.ndarray) -> int:
    rows = np.where((grid == WALL).any(axis=1))[0]
    assert rows.size > 0
    return int(rows[0])


def _find_regions(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    top = _first_wall_row(grid)
    comps = _components(grid[top:, :] != WALL)
    regions: list[np.ndarray] = []
    for comp in comps:
        if comp.size == 0:
            continue
        lifted = comp.copy()
        lifted[:, 0] += top
        regions.append(lifted)
    regions.sort(key=lambda pts: _bbox_coords(pts)[1])
    return top, regions


def _choose_wall_side(wall_mask: np.ndarray, r: int, c: int) -> tuple[int, bool]:
    h, w = wall_mask.shape
    cc = c - 1
    while cc >= 0 and not bool(wall_mask[r, cc]):
        cc -= 1
    left = c - cc if cc >= 0 else c + 1

    cc = c + 1
    while cc < w and not bool(wall_mask[r, cc]):
        cc += 1
    right = cc - c if cc < w else w - c

    if left <= right:
        return -1, left == right
    return 1, False


def _drop_one(occ: np.ndarray, wall_mask: np.ndarray, start: tuple[int, int]) -> tuple[tuple[int, int], dict]:
    h, w = occ.shape
    r, c = map(int, start)
    lateral_steps = 0
    used_tie = False

    while True:
        while r + 1 < h and int(occ[r + 1, c]) == BG:
            r += 1

        if r + 1 >= h:
            return (r, c), {"lateral_steps": lateral_steps, "used_tie": used_tie}

        down_left = c - 1 >= 0 and int(occ[r + 1, c - 1]) == BG
        down_right = c + 1 < w and int(occ[r + 1, c + 1]) == BG

        if down_left ^ down_right:
            r += 1
            c += -1 if down_left else 1
            lateral_steps += 1
            continue

        if down_left and down_right:
            side, tied = _choose_wall_side(wall_mask, r, c)
            used_tie = used_tie or tied
            r += 1
            c += side
            lateral_steps += 1
            continue

        slips: list[int] = []
        for side in (-1, 1):
            nc = c + side
            if 0 <= nc < w and int(occ[r, nc]) == BG and r + 1 < h and int(occ[r + 1, nc]) == BG:
                slips.append(side)

        if len(slips) == 1:
            c += slips[0]
            lateral_steps += 1
            continue

        if len(slips) == 2:
            side, tied = _choose_wall_side(wall_mask, r, c)
            used_tie = used_tie or tied
            c += side
            lateral_steps += 1
            continue

        return (r, c), {"lateral_steps": lateral_steps, "used_tie": used_tie}


def _simulate_settling(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int).copy()
    starts = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != BG) & (g != WALL))]
    starts.sort()
    for r, c, _color in starts:
        g[r, c] = BG

    wall_mask = g == WALL
    settled: list[dict] = []
    for r, c, color in starts:
        end, meta = _drop_one(g, wall_mask, (r, c))
        er, ec = end
        g[er, ec] = int(color)
        settled.append(
            {
                "color": int(color),
                "start": (int(r), int(c)),
                "end": (int(er), int(ec)),
                "lateral_steps": int(meta["lateral_steps"]),
                "used_tie": bool(meta["used_tie"]),
            }
        )
    return g, settled


def _simulate_vertical_only(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int).copy()
    starts = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != BG) & (g != WALL))]
    starts.sort()
    for r, c, _color in starts:
        g[r, c] = BG
    ends: list[tuple[int, int]] = []
    for r, c, color in starts:
        while r + 1 < g.shape[0] and int(g[r + 1, c]) == BG:
            r += 1
        g[r, c] = int(color)
        ends.append((int(r), int(c)))
    return ends


def _region_depths(grid: np.ndarray, coords: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    wall_mask = grid == WALL
    boundary: list[tuple[int, int]] = []
    for r, c in coords:
        rr, cc = int(r), int(c)
        if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
            boundary.append((rr, cc))
            continue
        hit_wall = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and bool(wall_mask[nr, nc]):
                    hit_wall = True
                    break
            if hit_wall:
                break
        if hit_wall:
            boundary.append((rr, cc))
    assert boundary

    depth: dict[tuple[int, int], int] = {}
    for r, c in coords:
        rr, cc = int(r), int(c)
        d = min(max(abs(rr - br), abs(cc - bc)) for br, bc in boundary)
        depth[(rr, cc)] = int(d)
    return depth


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _top, regions = _find_regions(g)
    settled_grid, settled = _simulate_settling(g)

    out = np.full_like(g, BG)
    out[g == WALL] = WALL

    region_id = np.full(g.shape, -1, dtype=int)
    region_depths: list[dict[tuple[int, int], int]] = []
    for rid, coords in enumerate(regions):
        for r, c in coords:
            region_id[int(r), int(c)] = int(rid)
        region_depths.append(_region_depths(g, coords))

    landed_by_region: dict[int, list[dict]] = defaultdict(list)
    for info in settled:
        er, ec = info["end"]
        rid = int(region_id[er, ec])
        assert rid >= 0
        landed_by_region[rid].append(info)

    for rid, coords in enumerate(regions):
        landed = landed_by_region[rid]
        assert landed
        depth_map = region_depths[rid]
        landed.sort(key=lambda item: (depth_map[item["end"]], item["end"][0], item["end"][1]))
        palette = [BG] + [int(item["color"]) for item in landed]
        plen = len(palette)
        for r, c in coords:
            out[int(r), int(c)] = int(palette[depth_map[(int(r), int(c))] % plen])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the top marker pixels into wall-bounded basins, then use their landing depths to paint concentric bands.

    Input structure:
    - A black background contains a wall network in color 9 that starts below an open top band.
    - The wall network partitions the lower part of the grid into several basin-like regions.
    - Several single colored pixels sit in the open top band above those basins.

    Rule:
    - Remove the top colored pixels and simulate their deterministic gravity settling: fall vertically through black cells, slide diagonally when only one diagonal is open, break two-way choices toward the nearer wall, and use one-step horizontal slips when resting at a drop edge.
    - After every colored pixel stops, group them by the basin region where they landed.
    - For each region, measure every landed pixel's Chebyshev depth from the region boundary layer that hugs the nearest wall or grid edge, sort the landed colors by increasing depth and then reading order, prepend black, and treat that list as a cyclic palette.
    - Repaint every cell of the region with palette[depth mod palette_length]; keep walls unchanged and clear the original top pixels.

    Key insight:
    - The walls serve two coupled roles at once: they steer the falling pixels into specific landing sites, and those landing depths become the palette order that drives the concentric region fill.
    """
    return _solve(np.asarray(grid, dtype=int))


def _paint_region(
    grid: np.ndarray,
    *,
    top: int,
    left_wall: int,
    right_wall: int,
    style: str,
    platform_row: int,
    colors: list[int],
    start_rows: list[int],
) -> None:
    assert len(colors) == 4
    inner_w = right_wall - left_wall - 1
    assert inner_w >= 9

    if style == "left":
        platform_start = left_wall + 2
        platform_end = left_wall + 6
        support_col = platform_start
        hole_col = left_wall + 4
        start_cols = [hole_col, hole_col - 1, hole_col + 1, hole_col]
    else:
        platform_start = right_wall - 6
        platform_end = right_wall - 2
        support_col = platform_end
        hole_col = right_wall - 4
        start_cols = [hole_col, hole_col + 1, hole_col - 1, hole_col]

    grid[top, hole_col] = BG
    grid[platform_row, platform_start : platform_end + 1] = WALL
    grid[platform_row : grid.shape[0] - 1, support_col] = WALL

    for row, col, color in zip(start_rows, start_cols, colors):
        grid[int(row), int(col)] = int(color)


def _build_attempt(rng: random.Random) -> np.ndarray:
    top = rng.randint(4, 5)
    h = rng.randint(15, 18)
    inner_ws = [rng.randint(9, 11), rng.randint(9, 11)]
    w = 1 + inner_ws[0] + 1 + inner_ws[1] + 1
    assert w <= 30

    grid = np.full((h, w), BG, dtype=int)
    grid[top, :] = WALL
    grid[h - 1, :] = WALL

    boundaries = [0, inner_ws[0] + 1, w - 1]
    for col in boundaries:
        grid[top:, int(col)] = WALL

    platform_row = h - rng.randint(4, 5)
    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    left_colors = colors[:4]
    right_colors = colors[4:8]
    rng.shuffle(left_colors)
    rng.shuffle(right_colors)

    start_rows = [0, 1, 2, top - 1]
    if top == 5 and rng.random() < 0.5:
        start_rows = [0, 2, 3, 4]

    _paint_region(
        grid,
        top=top,
        left_wall=0,
        right_wall=inner_ws[0] + 1,
        style=rng.choice(["left", "right"]),
        platform_row=platform_row,
        colors=left_colors,
        start_rows=start_rows,
    )
    _paint_region(
        grid,
        top=top,
        left_wall=inner_ws[0] + 1,
        right_wall=w - 1,
        style=rng.choice(["left", "right"]),
        platform_row=platform_row,
        colors=right_colors,
        start_rows=start_rows,
    )
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    top, regions = _find_regions(grid)
    if len(regions) != 2:
        return False

    settled_grid, settled = _simulate_settling(grid)
    if any(info["used_tie"] for info in settled):
        return False
    if sum(1 for info in settled if info["lateral_steps"] > 0) < 2:
        return False
    if len({info["color"] for info in settled}) != len(settled):
        return False

    naive = _simulate_vertical_only(grid)
    if all(info["end"] == naive[idx] for idx, info in enumerate(settled)):
        return False

    region_id = np.full(grid.shape, -1, dtype=int)
    depth_maps: list[dict[tuple[int, int], int]] = []
    for rid, coords in enumerate(regions):
        for r, c in coords:
            region_id[int(r), int(c)] = int(rid)
        depth_maps.append(_region_depths(grid, coords))

    landed_by_region: dict[int, list[dict]] = defaultdict(list)
    for info in settled:
        er, ec = info["end"]
        rid = int(region_id[er, ec])
        if rid < 0:
            return False
        landed_by_region[rid].append(info)

    if sorted(len(v) for v in landed_by_region.values()) != [4, 4]:
        return False

    for rid, landed in landed_by_region.items():
        depth_values = [depth_maps[rid][info["end"]] for info in landed]
        if max(depth_values) < 1:
            return False
        if len(set(depth_values)) < 2:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, settled_grid):
        return False
    if not np.all(out[:top, :] != WALL):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        try:
            if _semantic_checks(grid):
                return grid
        except AssertionError:
            continue
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
