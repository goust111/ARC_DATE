from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PINK = 6
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
AXIS_DIRS = {
    "nwse": [(-1, -1), (1, 1)],
    "nesw": [(-1, 1), (1, -1)],
}

GEN_ATTEMPTS = 400
GRID_MIN = 15
GRID_MAX = 19
MIN_REGION_COUNT = 4
MAX_REGION_COUNT = 6


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in (DIRS4 if connectivity == 4 else DIRS8)]


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _components(points: set[tuple[int, int]], connectivity: int) -> list[set[tuple[int, int]]]:
    remaining = set(points)
    out: list[set[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        comp = {start}
        stack = [start]
        remaining.remove(start)
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors(r, c, connectivity):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        out.append(comp)
    return out


def _mask_components(mask: np.ndarray, connectivity: int) -> list[set[tuple[int, int]]]:
    pts = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    return _components(pts, connectivity)


def _boundary_pixels(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in coords:
        if any((nr, nc) not in coords for nr, nc in _neighbors(r, c, 8) if _in_bounds(shape, nr, nc)):
            out.add((r, c))
            continue
        if r in (0, shape[0] - 1) or c in (0, shape[1] - 1):
            out.add((r, c))
    return out


def _diagonal_orientations_between(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> set[str]:
    orientations = set()
    for r, c in a:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            if (r + dr, c + dc) in b:
                orientations.add("nwse" if dr == dc else "nesw")
    return orientations


def _is_diagonal_chain(coords: set[tuple[int, int]]) -> bool:
    lobes = _components(coords, 4)
    if len(lobes) < 2:
        return False

    graph = {i: set() for i in range(len(lobes))}
    edge_count = 0
    for i in range(len(lobes)):
        for j in range(i + 1, len(lobes)):
            orientations = _diagonal_orientations_between(lobes[i], lobes[j])
            if not orientations:
                continue
            if len(orientations) != 1:
                return False
            graph[i].add(j)
            graph[j].add(i)
            edge_count += 1

    if any(len(nei) == 0 or len(nei) > 2 for nei in graph.values()):
        return False
    if edge_count not in {len(lobes) - 1, len(lobes)}:
        return False

    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if len(seen) != len(lobes):
        return False

    degrees = sorted(len(graph[i]) for i in range(len(lobes)))
    if edge_count == len(lobes) - 1:
        return degrees.count(1) == 2 and degrees.count(2) == len(lobes) - 2
    return all(deg == 2 for deg in degrees)


def _longest_diagonal_run(coords: set[tuple[int, int]], axis: str) -> int:
    lines: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in coords:
        lines[(r - c) if axis == "nwse" else (r + c)].append((r, c))

    best = 0
    want_dc = 1 if axis == "nwse" else -1
    for pts in lines.values():
        pts.sort()
        cur = 1
        best = max(best, cur)
        for (r0, c0), (r1, c1) in zip(pts, pts[1:], strict=False):
            if r1 - r0 == 1 and c1 - c0 == want_dc:
                cur += 1
            else:
                cur = 1
            best = max(best, cur)
    return best


def _main_axis(coords: set[tuple[int, int]]) -> str | None:
    nwse = _longest_diagonal_run(coords, "nwse")
    nesw = _longest_diagonal_run(coords, "nesw")
    if nwse == nesw:
        return None
    return "nwse" if nwse > nesw else "nesw"


def _region_infos(grid: np.ndarray) -> list[dict]:
    regions = _mask_components(grid == GREY, 8)
    infos = []
    for idx, coords in enumerate(regions):
        axis = _main_axis(coords) if _is_diagonal_chain(coords) else None
        if axis is None:
            is_seed = False
        else:
            is_seed = True
        infos.append(
            {
                "id": idx,
                "coords": coords,
                "bbox": _bbox(coords),
                "is_seed": is_seed,
                "axis": axis,
            }
        )
    return infos


def _fronts_for_seed(coords: set[tuple[int, int]], axis: str, shape: tuple[int, int]) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    boundary = _boundary_pixels(coords, shape)
    fronts = set()
    for r, c in boundary:
        for dr, dc in AXIS_DIRS[axis]:
            if (r + dr, c + dc) not in coords:
                fronts.add(((r, c), (dr, dc)))
    return fronts


def _transform(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    infos = _region_infos(g)
    info_by_id = {info["id"]: info for info in infos}

    for info in infos:
        if not info["is_seed"]:
            for r, c in info["coords"]:
                out[r, c] = RED

    growth_owner: dict[tuple[int, int], int] = {}
    seed_order = sorted((info for info in infos if info["is_seed"]), key=lambda item: (item["bbox"][0], item["bbox"][1], item["id"]))
    meta = {
        "regions": infos,
        "seed_order": [info["id"] for info in seed_order],
        "seed_meta": {},
        "stop_reasons": defaultdict(int),
    }

    for order_idx, info in enumerate(seed_order):
        axis = info["axis"]
        assert axis is not None
        fronts = _fronts_for_seed(info["coords"], axis, g.shape)
        claimed: set[tuple[int, int]] = set()
        generation = 0
        seed_meta = {
            "axis": axis,
            "bbox": info["bbox"],
            "growth_cells": [],
            "stop_reasons": defaultdict(int),
            "contest_cells": set(),
            "front_count": len(fronts),
            "order_idx": order_idx,
        }

        while fronts:
            color = PINK if generation % 2 == 0 else ORANGE
            next_fronts: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            pending: dict[tuple[int, int], tuple[int, int]] = {}
            for (r, c), (dr, dc) in sorted(fronts):
                nr = r + 2 * dr
                nc = c + 2 * dc
                if not _in_bounds(g.shape, nr, nc):
                    seed_meta["stop_reasons"]["boundary"] += 1
                    meta["stop_reasons"]["boundary"] += 1
                    continue
                if out[nr, nc] == RED:
                    seed_meta["stop_reasons"]["obstacle"] += 1
                    meta["stop_reasons"]["obstacle"] += 1
                    continue
                if (nr, nc) in growth_owner:
                    owner = growth_owner[(nr, nc)]
                    reason = "occupied_other" if owner != info["id"] else "occupied_self"
                    seed_meta["stop_reasons"][reason] += 1
                    meta["stop_reasons"][reason] += 1
                    if owner != info["id"]:
                        seed_meta["contest_cells"].add((nr, nc))
                    continue
                if (nr, nc) in info["coords"] or out[nr, nc] == GREY:
                    seed_meta["stop_reasons"]["seed_body"] += 1
                    meta["stop_reasons"]["seed_body"] += 1
                    continue
                pending[(nr, nc)] = (dr, dc)

            if not pending:
                break

            for (nr, nc), direction in sorted(pending.items()):
                if out[nr, nc] != BLACK:
                    seed_meta["stop_reasons"]["occupied_self"] += 1
                    meta["stop_reasons"]["occupied_self"] += 1
                    continue
                out[nr, nc] = color
                growth_owner[(nr, nc)] = info["id"]
                claimed.add((nr, nc))
                seed_meta["growth_cells"].append((nr, nc, color))
                next_fronts.add(((nr, nc), direction))

            fronts = next_fronts
            generation += 1

        seed_meta["claimed"] = claimed
        meta["seed_meta"][info["id"]] = seed_meta

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Classify grey regions by diagonal-chain connectivity, recolor the obstacles red, and grow alternating diagonal traces from the chain seeds.

    Input structure:
    - The grid is a black background with several grey 8-connected regions.
    - Some grey regions are multi-lobe diagonal chains whose 4-connected lobes meet only by corner contact.
    - The remaining grey regions are ordinary 4-connected blobs that act as obstacles.

    Rule:
    - Split each grey 8-connected region into 4-connected lobes.
    - A region is a seed only when those lobes form a non-branching diagonal chain or cycle; otherwise recolor the whole region red.
    - For each seed, find the unique longer diagonal run inside it to choose either the NW-SE axis or the NE-SW axis.
    - Starting from the seed's axis-facing boundary pixels, grow two cells at a time along that axis in both directions, coloring the first generation pink, the second orange, and then alternating until a step would leave the grid, hit a red obstacle, or hit another seed's already-claimed growth cell.

    Key insight:
    - The same diagonal-corner connectivity test both identifies which grey regions are seeds and encodes the diagonal axis that controls where each alternating growth trace must extend.
    """
    return _transform(grid, return_meta=False)


def _singleton_chain(length: int, axis: str) -> set[tuple[int, int]]:
    if axis == "nwse":
        pts = {(i, i) for i in range(length)}
    else:
        pts = {(i, length - 1 - i) for i in range(length)}
    return _normalize(pts)


def _block_plus_tail(axis: str, variant: int) -> set[tuple[int, int]]:
    pts = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 3)}
    if variant == 1:
        pts = {(0, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)}
    if axis == "nesw":
        pts = {(r, -c) for r, c in pts}
    return _normalize(pts)


def _two_blocks(axis: str) -> set[tuple[int, int]]:
    pts = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)}
    if axis == "nesw":
        pts = {(r, -c) for r, c in pts}
    return _normalize(pts)


def _block_single_block(axis: str, extra_tail: bool) -> set[tuple[int, int]]:
    pts = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 3), (3, 4), (4, 3), (4, 4)}
    if extra_tail:
        pts.add((5, 5))
    if axis == "nesw":
        pts = {(r, -c) for r, c in pts}
    return _normalize(pts)


def _random_seed_shape(axis: str) -> set[tuple[int, int]]:
    choice = random.choice(["chain4", "chain5", "chain6", "block_tail_a", "block_tail_b", "two_blocks", "block_single_block", "block_single_block_tail"])
    if choice == "chain4":
        return _singleton_chain(4, axis)
    if choice == "chain5":
        return _singleton_chain(5, axis)
    if choice == "chain6":
        return _singleton_chain(6, axis)
    if choice == "block_tail_a":
        return _block_plus_tail(axis, 0)
    if choice == "block_tail_b":
        return _block_plus_tail(axis, 1)
    if choice == "two_blocks":
        return _two_blocks(axis)
    if choice == "block_single_block":
        return _block_single_block(axis, False)
    return _block_single_block(axis, True)


def _staircase_obstacle(length: int) -> set[tuple[int, int]]:
    pts = {(0, 0)}
    r = c = 0
    for _ in range(length - 1):
        if random.random() < 0.5:
            c += 1
        else:
            r += 1
        pts.add((r, c))
    while len(pts) < length:
        pts.add((r + 1, c))
        r += 1
    return _normalize(pts)


def _random_obstacle_shape() -> set[tuple[int, int]]:
    kind = random.choice(["rect", "hbar", "vbar", "L", "T", "stair"])
    if kind == "rect":
        h = random.choice([2, 2, 3])
        w = 2 if h == 2 else random.choice([2, 3])
        return {(r, c) for r in range(h) for c in range(w)}
    if kind == "hbar":
        length = random.randint(4, 6)
        return {(0, c) for c in range(length)}
    if kind == "vbar":
        length = random.randint(4, 6)
        return {(r, 0) for r in range(length)}
    if kind == "L":
        h = random.randint(3, 4)
        w = random.randint(2, 3)
        pts = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
        return _normalize(pts)
    if kind == "T":
        width = random.choice([3, 4])
        mid = width // 2
        pts = {(0, c) for c in range(width)} | {(1, mid), (2, mid)}
        return _normalize(pts)
    return _staircase_obstacle(random.randint(5, 7))


def _translate(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _min_manhattan(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(r0 - r1) + abs(c0 - c1) for r0, c0 in a for r1, c1 in b)


def _can_place(coords: set[tuple[int, int]], h: int, w: int, existing: list[set[tuple[int, int]]]) -> bool:
    if not coords:
        return False
    if any(not _in_bounds((h, w), r, c) for r, c in coords):
        return False
    for other in existing:
        if coords & other:
            return False
        if _min_manhattan(coords, other) < 4:
            return False
    return True


def _place_random(shape: set[tuple[int, int]], h: int, w: int, existing: list[set[tuple[int, int]]]) -> set[tuple[int, int]] | None:
    norm = _normalize(shape)
    r0, c0, r1, c1 = _bbox(norm)
    ph = r1 - r0 + 1
    pw = c1 - c0 + 1
    choices = [(rr, cc) for rr in range(h - ph + 1) for cc in range(w - pw + 1)]
    random.shuffle(choices)
    for rr, cc in choices:
        placed = _translate(norm, rr, cc)
        if _can_place(placed, h, w, existing):
            return placed
    return None


def _covering_block(candidate: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = candidate
    dr, dc = direction
    row_offsets = [0, 1] if dr > 0 else [-1, 0]
    col_offsets = [0, 1] if dc > 0 else [-1, 0]
    return {(r + ro, c + co) for ro in row_offsets for co in col_offsets}


def _aligned_seed_pair(h: int, w: int, collision: bool) -> tuple[list[set[tuple[int, int]]], list[str]] | None:
    axis = random.choice(["nwse", "nesw"])
    len_a = random.randint(4, 5)
    len_b = random.randint(4, 5)
    shape_a = _singleton_chain(len_a, axis)
    shape_b = _singleton_chain(len_b, axis)

    if axis == "nwse":
        total_h = len_a + len_b + 6
        total_w = len_a + len_b + 6
        if total_h > h or total_w > w:
            return None
        ra = random.choice([0, 1])
        ca = random.randint(0, w - total_w)
        a = _translate(shape_a, ra, ca)
        a_end = (ra + len_a - 1, ca + len_a - 1)
        if collision:
            b_start = (a_end[0] + 4, a_end[1] + 4)
        else:
            gap = random.choice([6, 8])
            b_start = (a_end[0] + gap, a_end[1] + gap)
        b = _translate(shape_b, b_start[0], b_start[1])
        if not _can_place(a, h, w, []):
            return None
        if not _can_place(b, h, w, [a]):
            return None
        b_end = (b_start[0] + len_b - 1, b_start[1] + len_b - 1)
        obstacle = _covering_block((b_end[0] + 2, b_end[1] + 2), (1, 1))
    else:
        total_h = len_a + len_b + 6
        total_w = len_a + len_b + 6
        if total_h > h or total_w > w:
            return None
        ra = random.choice([0, 1])
        ca = random.randint(total_w - 1, w - 1)
        a = _translate(shape_a, ra, ca - (len_a - 1))
        a_sw = (ra + len_a - 1, ca - (len_a - 1))
        if collision:
            b_anchor = (a_sw[0] + 4, a_sw[1] - 4)
        else:
            gap = random.choice([6, 8])
            b_anchor = (a_sw[0] + gap, a_sw[1] - gap)
        b = _translate(shape_b, b_anchor[0], b_anchor[1] - (len_b - 1))
        if not _can_place(a, h, w, []):
            return None
        if not _can_place(b, h, w, [a]):
            return None
        b_sw = (b_anchor[0] + len_b - 1, b_anchor[1] - (len_b - 1))
        obstacle = _covering_block((b_sw[0] + 2, b_sw[1] - 2), (1, -1))

    if not _can_place(obstacle, h, w, [a, b]):
        return None
    return [a, b, obstacle], [axis, axis, "obstacle"]


def _render_grid(regions: list[set[tuple[int, int]]], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for coords in regions:
        for r, c in coords:
            grid[r, c] = GREY
    return grid


def _semantic_ok(grid: np.ndarray, want_collision: bool) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(np.unique(grid)) - {BLACK, GREY}:
        return False

    infos = _region_infos(grid)
    if not (MIN_REGION_COUNT <= len(infos) <= MAX_REGION_COUNT):
        return False
    if not any(info["is_seed"] for info in infos):
        return False
    if not any(not info["is_seed"] for info in infos):
        return False
    if any(info["is_seed"] and info["axis"] is None for info in infos):
        return False
    if any(_min_manhattan(a["coords"], b["coords"]) < 4 for i, a in enumerate(infos) for b in infos[i + 1 :]):
        return False

    out, meta = _transform(grid, return_meta=True)
    if np.array_equal(out, grid):
        return False
    if RED not in out or PINK not in out:
        return False
    if meta["stop_reasons"]["boundary"] == 0:
        return False
    if meta["stop_reasons"]["obstacle"] == 0:
        return False
    if want_collision and meta["stop_reasons"]["occupied_other"] == 0:
        return False
    if sum(len(seed_meta["growth_cells"]) for seed_meta in meta["seed_meta"].values()) < 3:
        return False

    for info in infos:
        region_vals = {int(out[r, c]) for r, c in info["coords"]}
        if info["is_seed"]:
            if region_vals != {GREY}:
                return False
            axis = info["axis"]
            assert axis is not None
            if _longest_diagonal_run(info["coords"], axis) <= _longest_diagonal_run(info["coords"], "nesw" if axis == "nwse" else "nwse"):
                return False
        else:
            if region_vals != {RED}:
                return False

    return True


def _attempt_grid() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    want_collision = random.random() < 0.4
    want_third_seed = random.random() < 0.45

    base = _aligned_seed_pair(h, w, want_collision)
    if base is None:
        return None
    regions, tags = base
    existing = list(regions)

    total_regions = random.randint(MIN_REGION_COUNT, MAX_REGION_COUNT)
    target_seeds = 3 if want_third_seed and total_regions >= 5 else 2
    target_obstacles = total_regions - target_seeds
    current_seed_count = 2
    current_obstacle_count = 1

    if current_seed_count < target_seeds:
        for _ in range(30):
            axis = random.choice(["nwse", "nesw"])
            shape = _random_seed_shape(axis)
            placed = _place_random(shape, h, w, existing)
            if placed is not None:
                regions.append(placed)
                existing.append(placed)
                current_seed_count += 1
                break
        if current_seed_count < target_seeds:
            return None

    while current_obstacle_count < target_obstacles:
        shape = _random_obstacle_shape()
        placed = _place_random(shape, h, w, existing)
        if placed is None:
            return None
        regions.append(placed)
        existing.append(placed)
        current_obstacle_count += 1

    grid = _render_grid(regions, h, w)
    if not _semantic_ok(grid, want_collision):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-chain growth input")
