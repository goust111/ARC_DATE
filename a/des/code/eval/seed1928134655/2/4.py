from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
HORIZONTAL_DIRS = [(0, -1), (0, 1)]
VERTICAL_DIRS = [(-1, 0), (1, 0)]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 32)
ARROW_COUNT_RANGE = (2, 5)
COLOR_COUNT_RANGE = (2, 4)
GEN_RETRIES = 300
PAIR_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _expanded_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                out.add((rr, cc))
    return out


def _cells_in_bounds(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = tip
    dr, dc = direction
    if dr != 0:
        return [(r, c), (r + dr, c), (r + 2 * dr, c), (r, c - 1), (r, c + 1)]
    return [(r, c), (r, c + dc), (r, c + 2 * dc), (r - 1, c), (r + 1, c)]


def _ray_cells(tip: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r = tip[0] + 3 * dr
    c = tip[1] + 3 * dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(grid == GREY)
    assert comps, "missing grey legend frame"

    def key(comp: list[tuple[int, int]]) -> tuple[int, int, int]:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        touches_corner = int(not (min(rows) == 0 and min(cols) == 0))
        return (touches_corner, min(rows) + min(cols), len(comp))

    comp = min(comps, key=key)
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_legend(
    grid: np.ndarray,
) -> tuple[dict[frozenset[int], int], np.ndarray, tuple[int, int, int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = _legend_bbox(grid)
    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    inner_mask = np.zeros((h, w), dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        inner_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    entry_mask = inner_mask & (grid != BG) & (grid != GREY)
    entry_comps = _components(entry_mask)
    mix_map: dict[frozenset[int], int] = {}
    for comp in entry_comps:
        assert len(comp) == 3, "legend entries must be 3-cell strips"
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert len(rows) == 1 or len(cols) == 1, "legend entry must be straight"
        ordered = sorted(comp, key=(lambda p: p[1]) if len(rows) == 1 else (lambda p: p[0]))
        a, b, mix = [int(grid[r, c]) for r, c in ordered]
        assert a != BG and b != BG and mix != BG
        assert a != GREY and b != GREY and mix != GREY
        assert mix not in (a, b), "mixed color must differ from both source colors"
        mix_map[frozenset((a, b))] = mix

    assert mix_map, "legend mapping must be non-empty"
    return mix_map, legend_mask, (r0, c0, r1, c1)


def _detect_arrows(grid: np.ndarray, legend_mask: np.ndarray) -> list[dict[str, object]]:
    mask = (~legend_mask) & (grid != BG) & (grid != GREY)
    arrows: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid[mask])):
        color_mask = mask & (grid == color)
        for comp in _components(color_mask):
            assert len(comp) == 5, "every non-legend object must be a 5-cell arrow"
            cell_set = set(comp)
            deg = {
                p: sum((nr, nc) in cell_set for nr, nc in _neighbors4(*p))
                for p in comp
            }
            tips = [p for p, d in deg.items() if d == 3]
            mids = [p for p, d in deg.items() if d == 2]
            ends = [p for p, d in deg.items() if d == 1]
            assert len(tips) == 1 and len(mids) == 1 and len(ends) == 3, "arrow must be T-shaped"
            tip = tips[0]
            tail_candidates = [p for p in ends if abs(p[0] - tip[0]) + abs(p[1] - tip[1]) == 2]
            assert len(tail_candidates) == 1, "arrow must have one 2-step shaft end"
            tail = tail_candidates[0]
            direction = (int(np.sign(tail[0] - tip[0])), int(np.sign(tail[1] - tip[1])))
            assert abs(direction[0]) + abs(direction[1]) == 1, "arrow direction must be cardinal"
            assert mids[0] == (tip[0] + direction[0], tip[1] + direction[1]), "shaft midpoint mismatch"
            expected = set(_arrow_cells(tip, direction))
            assert expected == cell_set, "component must exactly match the arrow template"
            arrows.append(
                {
                    "color": color,
                    "tip": tip,
                    "direction": direction,
                    "cells": comp,
                }
            )
    assert arrows, "expected at least one arrow"
    return arrows


def _arrow_sort_key(arrow: dict[str, object], shape: tuple[int, int]) -> tuple[int, int, int]:
    h, w = shape
    r, c = arrow["tip"]
    dist = max(abs(2 * r - (h - 1)), abs(2 * c - (w - 1)))
    return dist, r, c


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mix_map, legend_mask, legend_bbox = _parse_legend(g)
    arrows = _detect_arrows(g, legend_mask)

    out = g.copy()
    owner = np.full((h, w), -1, dtype=int)
    arrow_color_at: dict[tuple[int, int], int] = {}
    for arrow in arrows:
        for cell in arrow["cells"]:
            arrow_color_at[cell] = int(arrow["color"])

    collisions: list[tuple[tuple[int, int], int, int, int]] = []
    blocked_hits: list[tuple[str, tuple[int, int], int]] = []

    for arrow in sorted(arrows, key=lambda a: _arrow_sort_key(a, g.shape)):
        color = int(arrow["color"])
        dr, dc = arrow["direction"]
        r = arrow["tip"][0] + 3 * dr
        c = arrow["tip"][1] + 3 * dc
        while 0 <= r < h and 0 <= c < w:
            pos = (r, c)
            if legend_mask[r, c]:
                blocked_hits.append(("legend", pos, color))
                break
            static_arrow = arrow_color_at.get(pos)
            if static_arrow is not None:
                if static_arrow != color:
                    blocked_hits.append(("arrow", pos, color))
                    break
                r += dr
                c += dc
                continue

            occ = int(owner[r, c])
            if occ == -1:
                if int(out[r, c]) == BG:
                    out[r, c] = color
                    owner[r, c] = color
                elif int(out[r, c]) == color:
                    owner[r, c] = color
                else:
                    blocked_hits.append(("static", pos, color))
                    break
            elif occ == color:
                pass
            elif occ >= 0:
                pair = frozenset((occ, color))
                assert pair in mix_map, "missing legend entry for ray collision"
                out[r, c] = mix_map[pair]
                owner[r, c] = -2
                collisions.append((pos, occ, color, int(out[r, c])))
                break
            else:
                blocked_hits.append(("mixed", pos, color))
                break
            r += dr
            c += dc

    return out, {
        "mix_map": mix_map,
        "legend_mask": legend_mask,
        "legend_bbox": legend_bbox,
        "arrows": arrows,
        "collisions": collisions,
        "blocked_hits": blocked_hits,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each arrow with a same-color ray, mixing at the first different-color ray collision.
    """
    out, _ = _simulate(grid)
    return out


def _legend_size(entry_count: int, horizontal: bool) -> tuple[int, int]:
    if horizontal:
        return 2 * entry_count + 1, 5
    return 5, 2 * entry_count + 1


def _paint_legend(
    grid: np.ndarray,
    entries: list[tuple[int, int, int]],
    horizontal: bool,
) -> tuple[int, int, int, int]:
    leg_h, leg_w = _legend_size(len(entries), horizontal)
    grid[:leg_h, :leg_w] = BG
    grid[0, :leg_w] = GREY
    grid[leg_h - 1, :leg_w] = GREY
    grid[:leg_h, 0] = GREY
    grid[:leg_h, leg_w - 1] = GREY
    if horizontal:
        for i, (a, b, mix) in enumerate(entries):
            r = 1 + 2 * i
            grid[r, 1:4] = [a, b, mix]
    else:
        for i, (a, b, mix) in enumerate(entries):
            c = 1 + 2 * i
            grid[1:4, c] = [a, b, mix]
    return (0, 0, leg_h - 1, leg_w - 1)


def _mix_entries(colors: list[int], rng: random.Random) -> tuple[dict[frozenset[int], int], list[tuple[int, int, int]]]:
    pair_list = list(combinations(colors, 2))
    mix_pool = [c for c in PALETTE if c not in colors]
    mix_map: dict[frozenset[int], int] = {}
    entries: list[tuple[int, int, int]] = []
    for a, b in pair_list:
        mix = rng.choice(mix_pool)
        mix_map[frozenset((a, b))] = mix
        if rng.random() < 0.5:
            a, b = b, a
        entries.append((a, b, mix))
    rng.shuffle(entries)
    return mix_map, entries


def _spec(
    color: int,
    tip: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> dict[str, object] | None:
    cells = _arrow_cells(tip, direction)
    path = _ray_cells(tip, direction, shape)
    if not _cells_in_bounds(cells, shape) or len(path) < 1:
        return None
    return {"color": color, "tip": tip, "direction": direction, "cells": cells, "path": path}


def _max_dist_to_intersection(
    inter: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]
) -> int:
    h, w = shape
    r, c = inter
    if direction == (0, 1):
        return c
    if direction == (0, -1):
        return w - 1 - c
    if direction == (1, 0):
        return r
    return h - 1 - r


def _collision_pair(
    shape: tuple[int, int],
    color_a: int,
    color_b: int,
    static_block: set[tuple[int, int]],
    padded_block: set[tuple[int, int]],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    h, w = shape
    for _ in range(PAIR_RETRIES):
        inter = (rng.randint(1, h - 2), rng.randint(1, w - 2))
        dir_a = rng.choice(HORIZONTAL_DIRS)
        dir_b = rng.choice(VERTICAL_DIRS)
        max_a = _max_dist_to_intersection(inter, dir_a, shape)
        max_b = _max_dist_to_intersection(inter, dir_b, shape)
        if max_a < 3 or max_b < 3:
            continue
        dist_a = rng.randint(3, max_a)
        dist_b = rng.randint(3, max_b)
        tip_a = (inter[0] - dir_a[0] * dist_a, inter[1] - dir_a[1] * dist_a)
        tip_b = (inter[0] - dir_b[0] * dist_b, inter[1] - dir_b[1] * dist_b)
        spec_a = _spec(color_a, tip_a, dir_a, shape)
        spec_b = _spec(color_b, tip_b, dir_b, shape)
        if spec_a is None or spec_b is None:
            continue
        cells_a = set(spec_a["cells"])
        cells_b = set(spec_b["cells"])
        if cells_a & cells_b:
            continue
        if any(cell in padded_block for cell in cells_a | cells_b):
            continue
        if any(cell in static_block for cell in spec_a["path"] + spec_b["path"]):
            continue
        shared = set(spec_a["path"]) & set(spec_b["path"])
        if shared != {inter}:
            continue
        return [spec_a, spec_b]
    return None


def _candidate_specs(
    shape: tuple[int, int],
    color: int,
    static_block: set[tuple[int, int]],
    padded_block: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    rng: random.Random,
) -> list[dict[str, object]]:
    h, w = shape
    candidates: list[dict[str, object]] = []
    for direction in DIRS:
        for r in range(h):
            for c in range(w):
                spec = _spec(color, (r, c), direction, shape)
                if spec is None:
                    continue
                cells = set(spec["cells"])
                path = spec["path"]
                if cells & padded_block:
                    continue
                if cells & reserved_paths:
                    continue
                if any(cell in static_block for cell in path):
                    continue
                if any(cell in reserved_paths for cell in path):
                    continue
                candidates.append(spec)
    rng.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    arrow_count = rng.randint(*ARROW_COUNT_RANGE)
    color_count = rng.randint(COLOR_COUNT_RANGE[0], min(COLOR_COUNT_RANGE[1], arrow_count))
    colors = rng.sample(PALETTE, color_count)
    want_collision = rng.random() < 0.5

    mix_map, legend_entries = _mix_entries(colors, rng)
    horizontal_legend = rng.random() < 0.5
    leg_h, leg_w = _legend_size(len(legend_entries), horizontal_legend)
    h_min = max(GRID_H_RANGE[0], leg_h + 6)
    w_min = max(GRID_W_RANGE[0], leg_w + 8)
    if h_min > GRID_H_RANGE[1] or w_min > GRID_W_RANGE[1]:
        return None

    h = rng.randint(h_min, GRID_H_RANGE[1])
    w = rng.randint(w_min, GRID_W_RANGE[1])
    grid = np.full((h, w), BG, dtype=int)
    legend_bbox = _paint_legend(grid, legend_entries, horizontal_legend)

    legend_cells = {
        (r, c)
        for r in range(legend_bbox[0], legend_bbox[2] + 1)
        for c in range(legend_bbox[1], legend_bbox[3] + 1)
    }
    static_block = set(legend_cells)
    padded_block = _expanded_cells(legend_cells, grid.shape, radius=1)
    reserved_paths: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []

    arrow_colors = colors[:] + [rng.choice(colors) for _ in range(arrow_count - color_count)]
    rng.shuffle(arrow_colors)

    remaining_colors = arrow_colors[:]
    if want_collision:
        pair_colors = rng.sample(colors, 2)
        pair_specs = _collision_pair(grid.shape, pair_colors[0], pair_colors[1], static_block, padded_block, rng)
        if pair_specs is None:
            return None
        for pair_color in pair_colors:
            remaining_colors.remove(pair_color)
        for spec in pair_specs:
            specs.append(spec)
            cell_set = set(spec["cells"])
            static_block.update(cell_set)
            padded_block.update(_expanded_cells(cell_set, grid.shape, radius=1))
            reserved_paths.update(spec["path"])

    for color in remaining_colors:
        candidates = _candidate_specs(grid.shape, color, static_block, padded_block, reserved_paths, rng)
        if not candidates:
            return None
        spec = candidates[0]
        specs.append(spec)
        cell_set = set(spec["cells"])
        static_block.update(cell_set)
        padded_block.update(_expanded_cells(cell_set, grid.shape, radius=1))
        reserved_paths.update(spec["path"])

    for spec in specs:
        for r, c in spec["cells"]:
            grid[r, c] = int(spec["color"])

    if mix_map != _parse_legend(grid)[0]:
        return None

    try:
        out, info = _simulate(grid)
    except AssertionError:
        return None

    if len(info["arrows"]) != arrow_count:
        return None
    if info["blocked_hits"]:
        return None
    collision_count = len(info["collisions"])
    if want_collision and collision_count != 1:
        return None
    if not want_collision and collision_count != 0:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
