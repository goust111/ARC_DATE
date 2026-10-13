from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GENERATE_RETRIES = 400


def _tower_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    non_bg_cols = np.flatnonzero(np.any(grid != BG, axis=0))
    assert non_bg_cols.size > 0, "expected a legend column"
    legend_col = int(non_bg_cols[0])
    assert legend_col + 1 < grid.shape[1], "legend must have a bg gap on the right"
    assert np.all(grid[:, legend_col + 1] == BG), "legend gap must be all bg"

    rows = np.flatnonzero(grid[:, legend_col] != BG)
    assert rows.size >= 2, "expected at least two legend colors"
    assert np.all(np.diff(rows) == 1), "legend colors must form one vertical strip"
    colors = [int(grid[r, legend_col]) for r in rows]
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return legend_col, colors


def _plus_center(comp: list[tuple[int, int]]) -> tuple[int, int] | None:
    pts = set(comp)
    if len(pts) != 5:
        return None
    for r, c in pts:
        if pts == {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}:
            return r, c
    return None


def _extract_towers(grid: np.ndarray, legend_col: int) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    mask = grid != BG
    mask[:, legend_col] = False
    towers: list[tuple[int, int, int]] = []
    tower_mask = np.zeros_like(mask, dtype=bool)
    for comp in _components(mask):
        center = _plus_center(comp)
        assert center is not None, "all non-legend objects must be 3x3 plus towers"
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "tower must be monochromatic"
        color = int(next(iter(colors)))
        towers.append((center[0], center[1], color))
        for r, c in comp:
            tower_mask[r, c] = True
    assert towers, "expected at least one tower"
    return towers, tower_mask


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    legend_col, priority = _find_legend(g)
    priority_rank = {color: rank for rank, color in enumerate(priority)}
    towers, tower_mask = _extract_towers(g, legend_col)
    for _r, _c, color in towers:
        assert color in priority_rank, "tower color must appear in the legend"

    best_rank = np.full((h, w), len(priority) + 1, dtype=int)
    best_color = np.full((h, w), BG, dtype=int)
    covered_by: dict[tuple[int, int], set[int]] = defaultdict(set)
    tower_stop_count = 0

    for r, c, color in towers:
        rank = int(priority_rank[color])
        for dr, dc in DIR4:
            rr = r + 2 * dr
            cc = c + 2 * dc
            while 0 <= rr < h and 0 <= cc < w:
                if tower_mask[rr, cc]:
                    tower_stop_count += 1
                    break
                if g[rr, cc] == BG:
                    covered_by[(rr, cc)].add(color)
                    if rank < best_rank[rr, cc]:
                        best_rank[rr, cc] = rank
                        best_color[rr, cc] = color
                rr += dr
                cc += dc

    out = g.copy()
    paint_mask = (g == BG) & (best_rank <= len(priority))
    out[paint_mask] = best_color[paint_mask]
    out[:, legend_col] = BG

    meta = {
        "tower_count": len(towers),
        "tower_colors": {int(color) for _r, _c, color in towers},
        "ray_cells": int(paint_mask.sum()),
        "conflict_cells": sum(1 for colors in covered_by.values() if len(colors) >= 2),
        "tower_stops": int(tower_stop_count),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the left legend strip as a priority order and let tower rays claim bg cells.
    Higher-priority colors win on overlaps, the legend is erased in the output,
    and the original plus-shaped towers stay unchanged.
    """
    out, _meta = _solve_with_meta(grid)
    return out


def _can_place_tower(center: tuple[int, int], blocked: np.ndarray, h: int, w: int) -> bool:
    for r, c in _tower_cells(center):
        if not (0 <= r < h and 2 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _place_tower(
    grid: np.ndarray,
    center: tuple[int, int],
    color: int,
    occupied: np.ndarray,
    blocked: np.ndarray,
) -> None:
    h, w = grid.shape
    for r, c in _tower_cells(center):
        grid[r, c] = int(color)
        occupied[r, c] = True
    for r, c in _tower_cells(center):
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _sample_center(h: int, w: int) -> tuple[int, int]:
    return random.randint(1, h - 2), random.randint(3, w - 2)


def _place_random_tower(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    color: int,
    tries: int = 80,
) -> tuple[int, int] | None:
    h, w = grid.shape
    for _ in range(tries):
        center = _sample_center(h, w)
        if _can_place_tower(center, blocked, h, w):
            _place_tower(grid, center, color, occupied, blocked)
            return center
    return None


def _place_block_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    colors: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    for _ in range(120):
        if random.random() < 0.5:
            row = random.randint(1, h - 2)
            c1 = random.randint(3, w - 2)
            c2 = random.randint(3, w - 2)
            if abs(c1 - c2) < 4:
                continue
            a = (row, c1)
            b = (row, c2)
        else:
            col = random.randint(3, w - 2)
            r1 = random.randint(1, h - 2)
            r2 = random.randint(1, h - 2)
            if abs(r1 - r2) < 4:
                continue
            a = (r1, col)
            b = (r2, col)
        if _can_place_tower(a, blocked, h, w) and _can_place_tower(b, blocked, h, w):
            _place_tower(grid, a, colors[0], occupied, blocked)
            _place_tower(grid, b, colors[1], occupied, blocked)
            return [a, b]
    return []


def _sample_spaced(values: range, count: int, min_gap: int, tries: int = 120) -> list[int] | None:
    seq = list(values)
    if len(seq) < count:
        return None
    for _ in range(tries):
        picked = sorted(random.sample(seq, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _sample_tower_colors(legend_colors: list[int], count: int) -> list[int]:
    distinct = random.randint(2, min(len(legend_colors), count))
    chosen = random.sample(legend_colors, distinct)
    colors = chosen[:]
    while len(colors) < count:
        colors.append(random.choice(chosen))
    random.shuffle(colors)
    return colors


def _sample_conflict_layout(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    for _ in range(160):
        rows = _sample_spaced(range(1, h - 1), count, 2)
        cols = _sample_spaced(range(3, w - 1), count, 2)
        if rows is None or cols is None:
            return None
        random.shuffle(cols)
        centers = list(zip(rows, cols))
        blocked = np.zeros((h, w), dtype=bool)
        blocked[:, :2] = True
        ok = True
        for center in centers:
            if not _can_place_tower(center, blocked, h, w):
                ok = False
                break
            for r, c in _tower_cells(center):
                for rr in range(max(0, r - 1), min(h, r + 2)):
                    for cc in range(max(0, c - 1), min(w, c + 2)):
                        blocked[rr, cc] = True
        if ok:
            return centers
    return None


def _sample_block_layout(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    modes: list[str] = []
    if w >= 4 * (count - 1) + 5:
        modes.append("row")
    if h >= 4 * (count - 1) + 3:
        modes.append("col")
    if not modes:
        return None
    random.shuffle(modes)
    for mode in modes:
        if mode == "row":
            cols = _sample_spaced(range(3, w - 1), count, 4)
            if cols is None:
                continue
            row = random.randint(1, h - 2)
            return [(row, c) for c in cols]
        rows = _sample_spaced(range(1, h - 1), count, 4)
        if rows is None:
            continue
        col = random.randint(3, w - 2)
        return [(r, col) for r in rows]
    return None


def _try_generate_case() -> np.ndarray | None:
    h = random.randint(11, 19)
    w = random.randint(15, 24)
    max_colors = min(5, h - 2)
    n_colors = random.randint(2, max_colors)
    legend_colors = random.sample(PALETTE, n_colors)
    legend_top = random.randint(0, h - n_colors)

    grid = np.zeros((h, w), dtype=int)
    grid[legend_top : legend_top + n_colors, 0] = np.array(legend_colors, dtype=int)

    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, :2] = True

    scenario = random.choice(["conflict_only", "block_only", "mixed"])
    placed_centers: list[tuple[int, int]] = []

    if scenario == "conflict_only":
        max_towers = min(6, (h - 1) // 2, (w - 3) // 2)
        if max_towers < 3:
            return None
        total_towers = random.randint(3, max_towers)
        centers = _sample_conflict_layout(h, w, total_towers)
        if centers is None:
            return None
        colors = _sample_tower_colors(legend_colors, total_towers)
        for center, color in zip(centers, colors):
            _place_tower(grid, center, color, occupied, blocked)
            placed_centers.append(center)
    elif scenario == "block_only":
        row_cap = (w - 5) // 4 + 1
        col_cap = (h - 3) // 4 + 1
        max_towers = min(6, max(row_cap, col_cap))
        if max_towers < 3:
            return None
        total_towers = random.randint(3, max_towers)
        centers = _sample_block_layout(h, w, total_towers)
        if centers is None:
            return None
        colors = [random.choice(legend_colors)] * total_towers
        for center, color in zip(centers, colors):
            _place_tower(grid, center, color, occupied, blocked)
            placed_centers.append(center)
    else:
        max_towers = min(6, max(3, (h * (w - 2)) // 32))
        total_towers = random.randint(3, max_towers)
        pair_colors = tuple(random.sample(legend_colors, 2))
        centers = _place_block_pair(grid, occupied, blocked, pair_colors)
        if not centers:
            return None
        placed_centers.extend(centers)

        extra_color = random.choice(legend_colors)
        center = _place_random_tower(grid, occupied, blocked, extra_color)
        if center is None:
            return None
        placed_centers.append(center)

        while len(placed_centers) < total_towers:
            color = random.choice(legend_colors)
            center = _place_random_tower(grid, occupied, blocked, color)
            if center is None:
                return None
            placed_centers.append(center)

    if scenario != "block_only" and len({int(grid[r, c]) for r, c in placed_centers}) < 2:
        return None

    expected, meta = _solve_with_meta(grid)
    if meta["tower_count"] != len(placed_centers):
        return None
    if meta["ray_cells"] < max(4, len(placed_centers)):
        return None
    if scenario == "conflict_only" and meta["conflict_cells"] < 1:
        return None
    if scenario == "conflict_only" and meta["tower_stops"] != 0:
        return None
    if scenario == "block_only" and meta["conflict_cells"] != 0:
        return None
    if scenario == "block_only" and meta["tower_stops"] < 1:
        return None
    if scenario == "mixed" and meta["conflict_cells"] < 1:
        return None
    if scenario == "mixed" and meta["tower_stops"] < 1:
        return None
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        inp = _try_generate_case()
        if inp is not None:
            return inp
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
