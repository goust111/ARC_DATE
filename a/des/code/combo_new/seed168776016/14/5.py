from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _extract_walls(grid: np.ndarray):
    comps = _components_of_color(grid, WALL)
    wall_ids = -np.ones(grid.shape, dtype=int)
    walls: list[dict[str, object]] = []
    for idx, comp in enumerate(comps):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        orientation = "H" if len(rows) == 1 else "V" if len(cols) == 1 else "P"
        for r, c in comp:
            wall_ids[r, c] = idx
        walls.append({"id": idx, "cells": comp, "orientation": orientation})
    return walls, wall_ids


def _extract_seeds(grid: np.ndarray, wall_ids: np.ndarray):
    h, w = grid.shape
    seeds: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, WALL):
                continue
            wall_neighbors = []
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == WALL:
                    wall_neighbors.append((int(wall_ids[nr, nc]), nr, nc))
            if not wall_neighbors:
                continue
            wall_neighbors.sort()
            wall_id, wr, wc = wall_neighbors[0]
            seeds.append(
                {
                    "color": color,
                    "pos": (r, c),
                    "dir": (r - wr, c - wc),
                    "wall_id": wall_id,
                }
            )
    seeds.sort(key=lambda item: (item["pos"], item["color"]))
    return seeds


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    _, wall_ids = _extract_walls(g)
    seeds = _extract_seeds(g, wall_ids)
    if not seeds:
        return (out, []) if return_meta else out

    counts = Counter(int(seed["color"]) for seed in seeds)
    first_pos = {}
    seeds_by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for seed in seeds:
        color = int(seed["color"])
        pos = tuple(seed["pos"])
        seeds_by_color[color].append(seed)
        first_pos[color] = min(first_pos.get(color, pos), pos)

    ordered_colors = sorted(counts, key=lambda color: (counts[color], first_pos[color]))
    painted_mask = g == WALL
    meta: list[dict[str, object]] = []

    for color in ordered_colors:
        growth_len = int(counts[color])
        new_cells: set[tuple[int, int]] = set()
        color_meta: list[dict[str, object]] = []

        for seed in sorted(seeds_by_color[color], key=lambda item: item["pos"]):
            r, c = map(int, seed["pos"])
            dr, dc = map(int, seed["dir"])
            painted_len = 0
            stop_reason = None
            crosses_foreign_seed = False

            for step in range(growth_len):
                rr = r + dr * step
                cc = c + dc * step
                if not (0 <= rr < h and 0 <= cc < w):
                    stop_reason = "boundary"
                    break
                if step > 0:
                    if painted_mask[rr, cc]:
                        stop_reason = "wall" if int(g[rr, cc]) == WALL else "strip"
                        break
                    if int(g[rr, cc]) not in (BG, WALL, color):
                        crosses_foreign_seed = True
                new_cells.add((rr, cc))
                painted_len += 1

            color_meta.append(
                {
                    "color": color,
                    "count": growth_len,
                    "start": (r, c),
                    "dir": (dr, dc),
                    "painted_len": painted_len,
                    "stop": stop_reason,
                    "crosses_foreign_seed": crosses_foreign_seed,
                }
            )

        for rr, cc in new_cells:
            out[rr, cc] = color
            painted_mask[rr, cc] = True
        meta.extend(color_meta)

    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow wall-adjacent colored seeds outward into short bars whose lengths equal their global color counts.

    Input structure:
    - The background is black.
    - Grey pixels form disconnected one-cell-thick horizontal or vertical wall segments.
    - Non-grey, non-black seed pixels sit 4-neighbor-adjacent to a wall, all on one side of that wall.
    - The same color may appear beside different walls, and one wall side may host several seed colors.

    Rule:
    - Count how many seed pixels each color has in the whole grid.
    - Process colors from smaller global counts to larger global counts; ties are broken by first seed position in reading order.
    - For each seed of the current color, grow a one-pixel-wide bar along the outward wall normal for exactly that many cells, counting the seed cell itself.
    - Stop a bar before a grey wall, a previously grown bar, or the grid boundary; walls stay grey.

    Key insight:
    - The visible wall adjacency gives each seed's direction, while the globally repeated seed colors encode the growth lengths and the cross-color blocking order.
    """
    return _solve(np.asarray(grid, dtype=int), return_meta=False)


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _can_place(cells: set[tuple[int, int]], grid: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c] or int(grid[r, c]) != BG:
            return False
    return True


def _paint_group(grid: np.ndarray, blocked: np.ndarray, wall_cells: set[tuple[int, int]], seeds: list[tuple[tuple[int, int], int]]):
    for r, c in wall_cells:
        grid[r, c] = WALL
    for (r, c), color in seeds:
        grid[r, c] = color
    footprint = set(wall_cells)
    footprint.update(pos for pos, _ in seeds)
    for r, c in _expand_cells(footprint, *grid.shape, pad=1):
        blocked[r, c] = True


def _partition_seed_colors(seed_colors: list[int], rng: np.random.Generator, ensure_multi: bool) -> list[list[int]]:
    colors = list(seed_colors)
    rng.shuffle(colors)
    groups: list[list[int]] = []
    while colors:
        remaining = len(colors)
        max_size = min(3, remaining)
        if ensure_multi and not any(len(group) > 1 for group in groups) and remaining >= 2:
            size = int(rng.integers(2, max_size + 1))
        else:
            size = int(rng.integers(1, max_size + 1))
        if remaining - size == 1 and size == 3:
            size = 2
        groups.append(colors[:size])
        colors = colors[size:]
    if ensure_multi and groups and all(len(group) == 1 for group in groups) and len(groups) >= 2:
        groups[0].append(groups[1][0])
        groups.pop(1)
    return groups


def _build_group_candidate(
    h: int,
    w: int,
    group_colors: list[int],
    count_by_color: dict[int, int],
    rng: np.random.Generator,
):
    n = len(group_colors)
    orientation = rng.choice(["H", "V"])
    if orientation == "H":
        side = rng.choice(["up", "down"])
        max_len = min(9, w)
    else:
        side = rng.choice(["left", "right"])
        max_len = min(9, h)

    min_len = max(3, 2 * n - 1)
    if max_len < min_len:
        return None
    wall_len = int(rng.integers(min_len, max_len + 1))
    slots = list(range(0, wall_len, 2))
    if len(slots) < n:
        return None

    max_count = max(int(count_by_color[color]) for color in group_colors)
    if orientation == "H":
        avail_max = h - 1
    else:
        avail_max = w - 1
    if avail_max < 1:
        return None

    want_short = max_count > 1 and rng.random() < 0.5
    if want_short:
        short_hi = min(max_count - 1, avail_max)
        if short_hi < 1:
            want_short = False
        else:
            avail = int(rng.integers(1, short_hi + 1))
    if not want_short:
        avail_lo = min(max_count, avail_max)
        if avail_lo > avail_max:
            return None
        avail = int(rng.integers(avail_lo, avail_max + 1))

    seed_offsets = sorted(rng.choice(slots, size=n, replace=False).tolist())
    seed_colors = list(group_colors)
    rng.shuffle(seed_colors)

    if orientation == "H":
        if side == "up":
            seed_row = avail - 1
            wall_row = seed_row + 1
            dr, dc = -1, 0
        else:
            seed_row = h - avail
            wall_row = seed_row - 1
            dr, dc = 1, 0
        if not (0 <= wall_row < h and 0 <= seed_row < h):
            return None
        col0 = int(rng.integers(0, w - wall_len + 1))
        wall_cells = {(wall_row, col0 + k) for k in range(wall_len)}
        seeds = [((seed_row, col0 + off), color) for off, color in zip(seed_offsets, seed_colors)]
    else:
        if side == "left":
            seed_col = avail - 1
            wall_col = seed_col + 1
            dr, dc = 0, -1
        else:
            seed_col = w - avail
            wall_col = seed_col - 1
            dr, dc = 0, 1
        if not (0 <= wall_col < w and 0 <= seed_col < w):
            return None
        row0 = int(rng.integers(0, h - wall_len + 1))
        wall_cells = {(row0 + k, wall_col) for k in range(wall_len)}
        seeds = [((row0 + off, seed_col), color) for off, color in zip(seed_offsets, seed_colors)]

    return {"wall_cells": wall_cells, "seeds": seeds, "dir": (dr, dc)}


def _build_single_seed_wall(seed: tuple[int, int], direction: tuple[int, int], length: int = 3):
    r, c = seed
    dr, dc = direction
    if dr != 0:
        wall_row = r - dr
        col0 = c - length // 2
        return {(wall_row, col0 + k) for k in range(length)}
    wall_col = c - dc
    row0 = r - length // 2
    return {(row0 + k, wall_col) for k in range(length)}


def _place_interaction_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    rng: np.random.Generator,
    small_color: int,
    small_count: int,
    big_color: int,
    big_count: int,
) -> bool:
    h, w = grid.shape
    for _ in range(250):
        small_is_vertical = bool(rng.integers(0, 2))
        if small_is_vertical:
            d_small = [(-1, 0), (1, 0)][int(rng.integers(0, 2))]
            d_big = [(0, -1), (0, 1)][int(rng.integers(0, 2))]
        else:
            d_small = [(0, -1), (0, 1)][int(rng.integers(0, 2))]
            d_big = [(-1, 0), (1, 0)][int(rng.integers(0, 2))]

        step_small = int(rng.integers(0, small_count))
        step_big = int(rng.integers(1, big_count))

        ir = int(rng.integers(2, h - 2))
        ic = int(rng.integers(2, w - 2))
        small_seed = (ir - step_small * d_small[0], ic - step_small * d_small[1])
        big_seed = (ir - step_big * d_big[0], ic - step_big * d_big[1])

        small_wall = _build_single_seed_wall(small_seed, d_small, length=3)
        big_wall = _build_single_seed_wall(big_seed, d_big, length=3)
        cells = set(small_wall) | set(big_wall) | {small_seed, big_seed}
        if len(cells) != len(small_wall) + len(big_wall) + 2:
            continue
        if not _can_place(cells, grid, blocked):
            continue

        _paint_group(grid, blocked, set(small_wall), [(small_seed, small_color)])
        _paint_group(grid, blocked, set(big_wall), [(big_seed, big_color)])
        return True
    return False


def _pick_interaction_colors(count_by_color: dict[int, int], rng: np.random.Generator):
    items = sorted(count_by_color.items(), key=lambda item: item[1])
    candidates = []
    for i, (small_color, small_count) in enumerate(items[:-1]):
        for big_color, big_count in items[i + 1 :]:
            candidates.append((small_color, small_count, big_color, big_count))
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _generated_case_is_valid(grid: np.ndarray, require_strip_block: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    walls, wall_ids = _extract_walls(g)
    seeds = _extract_seeds(g, wall_ids)
    if not walls or not seeds:
        return False

    wall_dirs: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for seed in seeds:
        wall_dirs[int(seed["wall_id"])].add(tuple(seed["dir"]))
    if len(wall_dirs) != len(walls):
        return False
    if any(len(dirs) != 1 for dirs in wall_dirs.values()):
        return False

    for wall in walls:
        if wall["orientation"] == "P":
            return False

    counts = Counter(int(seed["color"]) for seed in seeds)
    if len(set(counts.values())) != len(counts):
        return False

    out, meta = _solve(g, return_meta=True)
    if not isinstance(out, np.ndarray) or out.shape != g.shape:
        return False
    if any(bool(item["crosses_foreign_seed"]) for item in meta):
        return False
    if not any(int(item["painted_len"]) > 1 for item in meta):
        return False
    if require_strip_block and not any(item["stop"] == "strip" for item in meta):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 25))
    w = int(rng.integers(12, 25))
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    num_colors = int(rng.integers(2, 5))
    colors = list(rng.choice(PALETTE, size=num_colors, replace=False))
    counts = list(rng.choice(np.arange(1, 6), size=num_colors, replace=False))
    rng.shuffle(counts)
    count_by_color = {int(color): int(count) for color, count in zip(colors, counts)}

    require_strip_block = bool(rng.integers(0, 2))
    remaining_seed_colors: list[int] = []
    for color, count in count_by_color.items():
        remaining_seed_colors.extend([color] * count)

    if require_strip_block:
        pick = _pick_interaction_colors(count_by_color, rng)
        if pick is None:
            return None
        small_color, small_count, big_color, big_count = pick
        if not _place_interaction_pair(grid, blocked, rng, small_color, small_count, big_color, big_count):
            return None
        remaining_seed_colors.remove(small_color)
        remaining_seed_colors.remove(big_color)

    ensure_multi_wall = len(remaining_seed_colors) >= 2 and rng.random() < 0.5
    groups = _partition_seed_colors(remaining_seed_colors, rng, ensure_multi=ensure_multi_wall)
    groups.sort(key=len, reverse=True)

    for group in groups:
        placed = False
        for _ in range(250):
            candidate = _build_group_candidate(h, w, group, count_by_color, rng)
            if candidate is None:
                continue
            wall_cells = set(candidate["wall_cells"])
            seeds = list(candidate["seeds"])
            cells = set(wall_cells)
            cells.update(pos for pos, _ in seeds)
            if not _can_place(cells, grid, blocked):
                continue
            _paint_group(grid, blocked, wall_cells, seeds)
            placed = True
            break
        if not placed:
            return None

    if not _generated_case_is_valid(grid, require_strip_block=require_strip_block):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-seed growth puzzle input")
