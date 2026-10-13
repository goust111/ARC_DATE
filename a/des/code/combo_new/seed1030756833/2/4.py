from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values.astype(int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _background_color(g: np.ndarray) -> int:
    h, w = g.shape
    if h == 1 or w == 1:
        return _mode(g.ravel())
    border = np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))
    return _mode(border)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _closed_background_regions(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    regions: list[dict[str, object]] = []
    for pts in _components(g == bg):
        touches_edge = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_edge:
            continue
        rows = [r for r, _c in pts]
        cols = [c for _r, c in pts]
        regions.append(
            {
                "pts": pts,
                "area": len(pts),
                "top": min(rows),
                "left": min(cols),
            }
        )
    return regions


def _edge_runs(line: np.ndarray, bg: int, wanted: int) -> list[tuple[int, list[int]]]:
    runs: list[tuple[int, list[int]]] = []
    i = 0
    n = int(line.size)
    while i < n:
        if int(line[i]) == bg:
            i += 1
            continue
        start = i
        colors: list[int] = []
        while i < n and int(line[i]) != bg:
            colors.append(int(line[i]))
            i += 1
        if len(colors) == wanted and len(set(colors)) == wanted:
            runs.append((start, colors))
    return runs


def _color_strip(g: np.ndarray, bg: int, wanted: int) -> list[int]:
    h, w = g.shape
    candidates: list[tuple[int, int, list[int]]] = []
    for priority, col in enumerate((0, w - 1)):
        for start, colors in _edge_runs(g[:, col], bg, wanted):
            candidates.append((priority, start, colors))
    assert candidates, "expected a vertical edge color strip matching the region count"
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill enclosed wall regions with ranked colors from an edge color strip.

    Input structure:
    - The border-majority color is the background.
    - A single wall color forms one-pixel closed loops around background regions.
    - One vertical edge contains a contiguous color strip with one distinct color for each closed region.

    Rule:
    - Find all 4-connected background components that do not touch the grid edge.
    - Sort those enclosed regions by decreasing area, breaking equal areas by the topmost row.
    - Read the edge color strip from top to bottom and assign its kth color to the kth sorted region.
    - Fill each enclosed background region with its assigned color while preserving the wall and color strip.

    Key insight:
    The color strip is an ordered legend: region rank is determined by enclosed area, not by position or wall shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    regions = _closed_background_regions(g, bg)
    assert regions, "expected at least one enclosed background region"
    regions.sort(key=lambda r: (-int(r["area"]), int(r["top"]), int(r["left"])))

    colors = _color_strip(g, bg, len(regions))
    out = g.copy()
    for region, color in zip(regions, colors):
        for r, c in region["pts"]:  # type: ignore[index]
            out[int(r), int(c)] = int(color)
    return out


N_COLORS = 10
GEN_RETRIES = 500
ROOM_COUNT_RANGE = (2, 5)
AREA_MIN = 3
AREA_MAX = 18
SHAPE_DIM_MAX = 6


def _random_polyomino(area: int, rng: random.Random) -> set[tuple[int, int]] | None:
    for _ in range(200):
        cells = {(0, 0)}
        frontier = {(dr, dc) for dr, dc in DIR4}
        while len(cells) < area and frontier:
            choices = list(frontier)
            rng.shuffle(choices)
            best_score = None
            best: list[tuple[int, int]] = []
            for cand in choices:
                trial = cells | {cand}
                rows = [r for r, _c in trial]
                cols = [c for _r, c in trial]
                height = max(rows) - min(rows) + 1
                width = max(cols) - min(cols) + 1
                score = (max(height, width), height * width)
                if best_score is None or score < best_score:
                    best_score = score
                    best = [cand]
                elif score == best_score:
                    best.append(cand)
            cell = rng.choice(best)
            frontier.discard(cell)
            cells.add(cell)
            for dr, dc in DIR4:
                nb = (cell[0] + dr, cell[1] + dc)
                if nb not in cells:
                    frontier.add(nb)

        if len(cells) != area:
            continue
        min_r = min(r for r, _c in cells)
        min_c = min(c for _r, c in cells)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        height = max(r for r, _c in norm) + 1
        width = max(c for _r, c in norm) + 1
        if height <= SHAPE_DIM_MAX and width <= SHAPE_DIM_MAX:
            return norm
    return None


def _room_from_interior(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int, int]:
    wall: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8:
            nb = (r + dr, c + dc)
            if nb not in cells:
                wall.add(nb)

    all_cells = cells | wall
    min_r = min(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    interior = {(r - min_r, c - min_c) for r, c in cells}
    wall = {(r - min_r, c - min_c) for r, c in wall}
    height = max(r for r, _c in (interior | wall)) + 1
    width = max(c for _r, c in (interior | wall)) + 1
    return interior, wall, height, width


def _target_areas(n: int, rng: random.Random) -> list[int]:
    pool = list(range(AREA_MIN, AREA_MAX + 1))
    if n >= 2 and rng.random() < 0.45:
        tie_area = rng.choice(pool)
        remaining = [a for a in pool if a != tie_area]
        areas = [tie_area, tie_area] + rng.sample(remaining, n - 2)
    else:
        areas = rng.sample(pool, n)
    rng.shuffle(areas)
    return areas


def _paint_room(
    grid: np.ndarray,
    occ: np.ndarray,
    interior: set[tuple[int, int]],
    wall: set[tuple[int, int]],
    room_h: int,
    room_w: int,
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    positions = [
        (r, c)
        for r in range(1, h - room_h)
        for c in range(1, w - room_w)
        if not bool(occ[r - 1 : r + room_h + 1, c - 1 : c + room_w + 1].any())
    ]
    if not positions:
        return False

    r0, c0 = rng.choice(positions)
    for dr, dc in wall:
        grid[r0 + dr, c0 + dc] = color
    occ[r0 - 1 : r0 + room_h + 1, c0 - 1 : c0 + room_w + 1] = True

    # Keep the intended region cells reserved even though they remain background.
    for dr, dc in interior:
        occ[r0 + dr, c0 + dc] = True
    return True


def _sort_is_unambiguous(regions: list[dict[str, object]]) -> bool:
    seen: set[tuple[int, int]] = set()
    for region in regions:
        key = (int(region["area"]), int(region["top"]))
        if key in seen:
            return False
        seen.add(key)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rooms = rng.randint(*ROOM_COUNT_RANGE)
    height = rng.randint(14, 30)
    width = rng.randint(14, 30)

    all_colors = list(range(N_COLORS))
    bg = rng.choice(all_colors)
    wall_color = rng.choice([c for c in all_colors if c != bg])
    strip_colors = rng.sample([c for c in all_colors if c not in (bg, wall_color)], n_rooms)

    grid = np.full((height, width), bg, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    side = rng.choice(("left", "right"))
    bar_start = rng.randint(0, height - n_rooms)
    bar_col = 0 if side == "left" else width - 1
    for i, color in enumerate(strip_colors):
        grid[bar_start + i, bar_col] = color
    if side == "left":
        occ[:, :2] = True
    else:
        occ[:, width - 2 :] = True
    occ[bar_start : bar_start + n_rooms, bar_col] = True

    areas = _target_areas(n_rooms, rng)
    rooms = []
    for area in areas:
        cells = _random_polyomino(area, rng)
        if cells is None:
            return None
        rooms.append(_room_from_interior(cells))
    rng.shuffle(rooms)

    for interior, wall, room_h, room_w in rooms:
        if not _paint_room(grid, occ, interior, wall, room_h, room_w, wall_color, rng):
            return None

    if _background_color(grid) != bg:
        return None

    regions = _closed_background_regions(grid, bg)
    if len(regions) != n_rooms:
        return None
    if sorted(int(r["area"]) for r in regions) != sorted(areas):
        return None
    if not _sort_is_unambiguous(regions):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
