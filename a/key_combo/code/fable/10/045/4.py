from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
OUTLINE_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
SIDE_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


Coord = tuple[int, int]


@dataclass(frozen=True)
class Cup:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    side: str
    open_start: int
    open_len: int
    opening: tuple[Coord, ...]
    walls: frozenset[Coord]
    interior: frozenset[Coord]
    footprint: frozenset[Coord]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_of(coords: Iterable[Coord]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def build_cup(color: int, r0: int, c0: int, h: int, w: int, side: str, open_start: int, open_len: int) -> Cup:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if side in ("top", "bottom"):
        side_len = w
        opening = tuple((r0 if side == "top" else r1, c0 + k) for k in range(open_start, open_start + open_len))
    else:
        side_len = h
        opening = tuple((r0 + k, c0 if side == "left" else c1) for k in range(open_start, open_start + open_len))
    assert 1 <= open_start <= side_len - open_len - 1
    assert open_len % 2 == 1

    border: set[Coord] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    opening_set = set(opening)
    walls = frozenset(border - opening_set)
    interior = frozenset((r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
    footprint = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    return Cup(
        color=color,
        r0=r0,
        c0=c0,
        r1=r1,
        c1=c1,
        side=side,
        open_start=open_start,
        open_len=open_len,
        opening=opening,
        walls=walls,
        interior=interior,
        footprint=footprint,
    )


def sort_opening(side: str, cells: Iterable[Coord]) -> tuple[Coord, ...]:
    if side in ("top", "bottom"):
        return tuple(sorted(cells, key=lambda rc: rc[1]))
    return tuple(sorted(cells, key=lambda rc: rc[0]))


def ray_specs(cup: Cup) -> list[tuple[Coord, Coord]]:
    opening = list(cup.opening)
    center = opening[len(opening) // 2]
    if cup.side == "top":
        return [(center, (-1, 0)), (opening[0], (-1, -1)), (opening[-1], (-1, 1))]
    if cup.side == "bottom":
        return [(center, (1, 0)), (opening[0], (1, -1)), (opening[-1], (1, 1))]
    if cup.side == "left":
        return [(center, (0, -1)), (opening[0], (-1, -1)), (opening[-1], (1, -1))]
    return [(center, (0, 1)), (opening[0], (-1, 1)), (opening[-1], (1, 1))]


def _components_for_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def detect_cups(grid: np.ndarray) -> tuple[list[Cup], int]:
    g = np.asarray(grid, dtype=int)
    cups: list[Cup] = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) not in (BLACK, YELLOW)):
        for comp in _components_for_color(g, color):
            r0, c0, r1, c1 = bbox_of(comp)
            border: set[Coord] = set()
            for c in range(c0, c1 + 1):
                border.add((r0, c))
                border.add((r1, c))
            for r in range(r0, r1 + 1):
                border.add((r, c0))
                border.add((r, c1))
            wall_set = set(comp)
            assert wall_set <= border
            opening_set = border - wall_set
            assert opening_set
            if all(r == r0 for r, _ in opening_set):
                side = "top"
                offsets = sorted(c - c0 for _, c in opening_set)
                side_len = c1 - c0 + 1
            elif all(r == r1 for r, _ in opening_set):
                side = "bottom"
                offsets = sorted(c - c0 for _, c in opening_set)
                side_len = c1 - c0 + 1
            elif all(c == c0 for _, c in opening_set):
                side = "left"
                offsets = sorted(r - r0 for r, _ in opening_set)
                side_len = r1 - r0 + 1
            else:
                assert all(c == c1 for _, c in opening_set)
                side = "right"
                offsets = sorted(r - r0 for r, _ in opening_set)
                side_len = r1 - r0 + 1
            assert offsets[0] >= 1 and offsets[-1] <= side_len - 2
            assert offsets == list(range(offsets[0], offsets[0] + len(offsets)))
            opening = sort_opening(side, opening_set)
            interior = frozenset((r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
            assert interior
            cups.append(
                Cup(
                    color=color,
                    r0=r0,
                    c0=c0,
                    r1=r1,
                    c1=c1,
                    side=side,
                    open_start=offsets[0],
                    open_len=len(offsets),
                    opening=opening,
                    walls=frozenset(wall_set),
                    interior=interior,
                    footprint=frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)),
                )
            )

    charged = []
    for idx, cup in enumerate(cups):
        inside_vals = {int(g[r, c]) for r, c in cup.interior}
        assert inside_vals <= {BLACK, YELLOW}
        if inside_vals == {YELLOW}:
            charged.append(idx)
        else:
            assert inside_vals == {BLACK}
    assert len(charged) == 1
    return cups, charged[0]


def render_input(cups: list[Cup], start_idx: int) -> np.ndarray:
    h = max(cup.r1 for cup in cups) + 1
    w = max(cup.c1 for cup in cups) + 1
    grid = np.full((h, w), BLACK, dtype=int)
    for idx, cup in enumerate(cups):
        for r, c in cup.walls:
            grid[r, c] = cup.color
        if idx == start_idx:
            for r, c in cup.interior:
                grid[r, c] = YELLOW
    return grid


def simulate(grid: np.ndarray, cups: list[Cup], start_idx: int) -> tuple[np.ndarray, dict]:
    out = np.array(grid, dtype=int, copy=True)
    wall_cells = {cell for cup in cups for cell in cup.walls}
    opening_to_cup = {}
    for idx, cup in enumerate(cups):
        for cell in cup.opening:
            opening_to_cup[cell] = idx

    charged = {start_idx}
    parent = {start_idx: None}
    depth = {start_idx: 0}
    queue = deque([start_idx])
    traces: dict[tuple[int, int], dict] = {}
    union_ray_cells: set[Coord] = set()
    wall_stops = 0
    boundary_stops = 0

    while queue:
        src = queue.popleft()
        for ray_idx, (start, step) in enumerate(ray_specs(cups[src])):
            cells = [start]
            union_ray_cells.add(start)
            if out[start] == BLACK:
                out[start] = YELLOW
            current = start
            reason = "boundary"
            target = None
            while True:
                nr = current[0] + step[0]
                nc = current[1] + step[1]
                if not in_bounds(out.shape, nr, nc):
                    boundary_stops += 1
                    reason = "boundary"
                    break
                if (nr, nc) in wall_cells:
                    wall_stops += 1
                    reason = "wall"
                    break
                current = (nr, nc)
                cells.append(current)
                union_ray_cells.add(current)
                if out[current] == BLACK:
                    out[current] = YELLOW
                cup_idx = opening_to_cup.get(current)
                if cup_idx is None:
                    continue
                nxt = (current[0] + step[0], current[1] + step[1])
                if nxt in cups[cup_idx].interior:
                    target = cup_idx
                    reason = "charge"
                    if cup_idx not in charged:
                        charged.add(cup_idx)
                        parent[cup_idx] = src
                        depth[cup_idx] = depth[src] + 1
                        for rr, cc in cups[cup_idx].interior:
                            out[rr, cc] = YELLOW
                        queue.append(cup_idx)
                    break
            traces[(src, ray_idx)] = {
                "cells": tuple(cells),
                "step": step,
                "reason": reason,
                "target": target,
            }

    return out, {
        "charged": charged,
        "parent": parent,
        "depth": depth,
        "max_depth": max(depth.values()),
        "traces": traces,
        "ray_cells": union_ray_cells,
        "wall_stops": wall_stops,
        "boundary_stops": boundary_stops,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every cup reached through opening-aligned rays and keep the emitted yellow trails.

    Input structure:
    - Black background with several axis-aligned rectangular cup outlines, each in its own color.
    - Every cup is a one-cell-thick rectangle with one contiguous opening on exactly one side.
    - Exactly one cup already has its whole interior filled yellow; all other cup interiors are black.

    Rule:
    - Each charged cup emits three yellow rays: one straight outward from the middle opening cell,
      plus two diagonals outward from the opening endpoints.
    - A ray advances cell by cell, painting yellow, and stops before leaving the grid or entering any
      cup wall.
    - If a ray lands on another cup's opening and its next step would move into that cup's interior,
      that cup is fully filled yellow, becomes charged, and emits its own three rays. Repeat until no
      new cup is charged.

    Key insight:
    - Propagation depends on exact opening alignment: only rays that pass through a cup's gap and step
      into its interior activate it; rays that meet colored walls are simply blocked.
    """
    g = np.asarray(grid, dtype=int)
    cups, start_idx = detect_cups(g)
    out, _ = simulate(g, cups, start_idx)
    return out


def boxes_with_margin_intersect(a: Cup, b: Cup, margin: int = 1) -> bool:
    return not (
        a.r1 + margin < b.r0 - margin
        or b.r1 + margin < a.r0 - margin
        or a.c1 + margin < b.c0 - margin
        or b.c1 + margin < a.c0 - margin
    )


def can_place_cup(cup: Cup, placed: list[Cup], shape: tuple[int, int], protected: set[Coord] | None = None) -> bool:
    h, w = shape
    if cup.r0 < 1 or cup.c0 < 1 or cup.r1 >= h - 1 or cup.c1 >= w - 1:
        return False
    for other in placed:
        if boxes_with_margin_intersect(cup, other, margin=1):
            return False
    if protected is not None and cup.footprint & protected:
        return False
    return True


def opening_choices(side_len: int, forced_offset: int | None = None) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for open_len in (1, 3):
        if open_len > side_len - 2:
            continue
        for start in range(1, side_len - open_len):
            if forced_offset is None or start <= forced_offset <= start + open_len - 1:
                options.append((start, open_len))
    return options


def sample_root(shape: tuple[int, int], color: int) -> Cup | None:
    h, w = shape
    for _ in range(80):
        cup_h = random.randint(5, 7)
        cup_w = random.randint(5, 7)
        side = random.choice(list(SIDE_DIR))
        side_len = cup_w if side in ("top", "bottom") else cup_h
        options = opening_choices(side_len)
        if not options:
            continue
        open_start, open_len = random.choice(options)
        r_lo = max(2, h // 3 - 2)
        r_hi = min(h - cup_h - 3, 2 * h // 3 - cup_h + 1)
        c_lo = max(2, w // 3 - 2)
        c_hi = min(w - cup_w - 3, 2 * w // 3 - cup_w + 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        return build_cup(color, r0, c0, cup_h, cup_w, side, open_start, open_len)
    return None


def compatible_target_sides(incoming: Coord) -> list[str]:
    dr, dc = incoming
    sides = []
    if dr == 1:
        sides.append("top")
    if dr == -1:
        sides.append("bottom")
    if dc == 1:
        sides.append("left")
    if dc == -1:
        sides.append("right")
    return sides


def sample_child_on_ray(shape: tuple[int, int], color: int, incoming: Coord, anchor: Coord) -> Cup | None:
    for _ in range(120):
        side = random.choice(compatible_target_sides(incoming))
        cup_h = random.randint(5, 7)
        cup_w = random.randint(5, 7)
        side_len = cup_w if side in ("top", "bottom") else cup_h
        if side in ("top", "bottom"):
            valid_offsets = [k for k in range(1, cup_w - 1) if 1 <= k + incoming[1] <= cup_w - 2]
        else:
            valid_offsets = [k for k in range(1, cup_h - 1) if 1 <= k + incoming[0] <= cup_h - 2]
        if not valid_offsets:
            continue
        offset = random.choice(valid_offsets)
        options = opening_choices(side_len, forced_offset=offset)
        if not options:
            continue
        open_start, open_len = random.choice(options)
        if side == "top":
            r0 = anchor[0]
            c0 = anchor[1] - offset
        elif side == "bottom":
            r0 = anchor[0] - cup_h + 1
            c0 = anchor[1] - offset
        elif side == "left":
            r0 = anchor[0] - offset
            c0 = anchor[1]
        else:
            r0 = anchor[0] - offset
            c0 = anchor[1] - cup_w + 1
        cup = build_cup(color, r0, c0, cup_h, cup_w, side, open_start, open_len)
        nxt = (anchor[0] + incoming[0], anchor[1] + incoming[1])
        if nxt in cup.interior:
            return cup
    return None


def path_is_clear(start: Coord, step: Coord, distance: int, placed: list[Cup], candidate: Cup) -> bool:
    occupied = set()
    for cup in placed:
        occupied |= cup.footprint
    for i in range(1, distance):
        cell = (start[0] + step[0] * i, start[1] + step[1] * i)
        if cell in occupied or cell in candidate.footprint:
            return False
    return True


def trace_result_for_ray(cups: list[Cup], start_idx: int, ray_idx: int) -> tuple[np.ndarray, dict]:
    grid = render_input(cups, start_idx)
    return simulate(grid, cups, start_idx)


def sample_wall_anchor_cup(shape: tuple[int, int], color: int, anchor: Coord, ray_cells: set[Coord]) -> Cup | None:
    for _ in range(120):
        wall_side = random.choice(["top", "bottom", "left", "right"])
        cup_h = random.randint(5, 7)
        cup_w = random.randint(5, 7)
        side_len = cup_w if wall_side in ("top", "bottom") else cup_h
        if side_len <= 2:
            continue
        offset = random.randint(1, side_len - 2)
        opening_side = random.choice([side for side in ("top", "bottom", "left", "right") if side != wall_side])
        open_side_len = cup_w if opening_side in ("top", "bottom") else cup_h
        options = opening_choices(open_side_len)
        if not options:
            continue
        open_start, open_len = random.choice(options)
        if wall_side == "top":
            r0 = anchor[0]
            c0 = anchor[1] - offset
        elif wall_side == "bottom":
            r0 = anchor[0] - cup_h + 1
            c0 = anchor[1] - offset
        elif wall_side == "left":
            r0 = anchor[0] - offset
            c0 = anchor[1]
        else:
            r0 = anchor[0] - offset
            c0 = anchor[1] - cup_w + 1
        cup = build_cup(color, r0, c0, cup_h, cup_w, opening_side, open_start, open_len)
        if anchor not in cup.walls:
            continue
        if set(cup.opening) & ray_cells:
            continue
        return cup
    return None


def sample_random_decoy(shape: tuple[int, int], color: int, ray_cells: set[Coord]) -> Cup | None:
    h, w = shape
    for _ in range(160):
        cup_h = random.randint(5, 7)
        cup_w = random.randint(5, 7)
        side = random.choice(["top", "bottom", "left", "right"])
        side_len = cup_w if side in ("top", "bottom") else cup_h
        options = opening_choices(side_len)
        if not options:
            continue
        open_start, open_len = random.choice(options)
        r0 = random.randint(1, h - cup_h - 2)
        c0 = random.randint(1, w - cup_w - 2)
        cup = build_cup(color, r0, c0, cup_h, cup_w, side, open_start, open_len)
        if set(cup.opening) & ray_cells:
            continue
        return cup
    return None


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(22, 28), random.randint(22, 28))
    total_cups = random.randint(4, 6)
    reachable_count = random.randint(3, min(4, total_cups - 1))
    colors = random.sample(OUTLINE_COLORS, total_cups)

    root = sample_root(shape, colors[0])
    if root is None:
        return None

    cups = [root]
    planned_edges: list[tuple[int, int, int]] = []
    used_rays: dict[int, set[int]] = {0: set()}

    for src_idx in [0]:
        used_rays.setdefault(src_idx, set())

    # Backbone chain: root -> a -> b so multi-step propagation is required.
    for parent_idx in [0, 1]:
        if parent_idx >= len(cups):
            return None
        placed = False
        ray_order = list(range(3))
        random.shuffle(ray_order)
        for ray_idx in ray_order:
            start, step = ray_specs(cups[parent_idx])[ray_idx]
            for distance in random.sample(list(range(4, 10)), 6):
                anchor = (start[0] + step[0] * distance, start[1] + step[1] * distance)
                child = sample_child_on_ray(shape, colors[len(cups)], step, anchor)
                if child is None:
                    continue
                if not can_place_cup(child, cups, shape):
                    continue
                if not path_is_clear(start, step, distance, cups, child):
                    continue
                child_idx = len(cups)
                cups.append(child)
                used_rays.setdefault(child_idx, set())
                used_rays[parent_idx].add(ray_idx)
                planned_edges.append((parent_idx, ray_idx, child_idx))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    while len(cups) < reachable_count:
        parent_candidates = [idx for idx in range(len(cups)) if len(used_rays.get(idx, set())) < 3]
        random.shuffle(parent_candidates)
        placed = False
        for parent_idx in parent_candidates:
            ray_order = [idx for idx in range(3) if idx not in used_rays[parent_idx]]
            random.shuffle(ray_order)
            for ray_idx in ray_order:
                start, step = ray_specs(cups[parent_idx])[ray_idx]
                for distance in random.sample(list(range(4, 10)), 6):
                    anchor = (start[0] + step[0] * distance, start[1] + step[1] * distance)
                    child = sample_child_on_ray(shape, colors[len(cups)], step, anchor)
                    if child is None:
                        continue
                    if not can_place_cup(child, cups, shape):
                        continue
                    if not path_is_clear(start, step, distance, cups, child):
                        continue
                    child_idx = len(cups)
                    cups.append(child)
                    used_rays.setdefault(child_idx, set())
                    used_rays[parent_idx].add(ray_idx)
                    planned_edges.append((parent_idx, ray_idx, child_idx))
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    start_idx = 0
    base_input = render_input(cups, start_idx)
    _, reach_stats = simulate(base_input, cups, start_idx)
    if len(reach_stats["charged"]) != reachable_count or reach_stats["max_depth"] < 2:
        return None

    protected_cells: set[Coord] = set()
    for parent_idx, ray_idx, child_idx in planned_edges:
        trace = reach_stats["traces"].get((parent_idx, ray_idx))
        if trace is None or trace["reason"] != "charge" or trace["target"] != child_idx:
            return None
        protected_cells.update(trace["cells"])

    ray_cells = set(reach_stats["ray_cells"])
    cups_so_far = list(cups)

    blocker_placed = False
    blocker_candidates = [
        (key, trace)
        for key, trace in reach_stats["traces"].items()
        if trace["reason"] == "boundary" and not (set(trace["cells"]) & protected_cells == set(trace["cells"]))
    ]
    random.shuffle(blocker_candidates)
    while len(cups_so_far) < total_cups and blocker_candidates and not blocker_placed:
        (src_idx, _ray_idx), trace = blocker_candidates.pop()
        candidates = [cell for cell in trace["cells"][2:] if cell not in protected_cells]
        random.shuffle(candidates)
        for anchor in candidates[:8]:
            cup = sample_wall_anchor_cup(shape, colors[len(cups_so_far)], anchor, ray_cells)
            if cup is None:
                continue
            if not can_place_cup(cup, cups_so_far, shape, protected=protected_cells):
                continue
            cups_so_far.append(cup)
            blocker_placed = True
            break

    if not blocker_placed:
        return None

    extra_attempts = 0
    while len(cups_so_far) < total_cups:
        extra_attempts += 1
        if extra_attempts > 240:
            return None
        cup = sample_random_decoy(shape, colors[len(cups_so_far)], ray_cells)
        if cup is None:
            return None
        if not can_place_cup(cup, cups_so_far, shape, protected=protected_cells):
            continue
        cups_so_far.append(cup)

    inp = render_input(cups_so_far, start_idx)
    expected, full_stats = simulate(inp, cups_so_far, start_idx)
    if len(full_stats["charged"]) != reachable_count:
        return None
    if full_stats["max_depth"] < 2:
        return None
    if len(full_stats["charged"]) >= len(cups_so_far):
        return None
    if full_stats["wall_stops"] < 1 or full_stats["boundary_stops"] < 1:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-relay input")
