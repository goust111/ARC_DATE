from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS4 = (UP, RIGHT, DOWN, LEFT)

PRIMITIVE_OFFSETS = {
    "square": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "hbar": ((0, 0), (0, 1), (0, 2)),
    "vbar": ((0, 0), (1, 0), (2, 0)),
}

RAY_DIRS = {
    "square": (UP, DOWN, LEFT, RIGHT),
    "hbar": (LEFT, RIGHT),
    "vbar": (UP, DOWN),
}

RAY_COLOR = {
    "square": TEAL,
    "hbar": RED,
    "vbar": RED,
}

OBSTACLE_BASES = [
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 1), (2, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)},
]

MAX_GEN_RETRIES = 500


def add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def turn_right(d: tuple[int, int]) -> tuple[int, int]:
    return DIRS4[(DIRS4.index(d) + 1) % 4]


def turn_left(d: tuple[int, int]) -> tuple[int, int]:
    return DIRS4[(DIRS4.index(d) - 1) % 4]


def turn_back(d: tuple[int, int]) -> tuple[int, int]:
    return DIRS4[(DIRS4.index(d) + 2) % 4]


def in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def neighbors4(cell: tuple[int, int]):
    r, c = cell
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def normalize_cells(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def bbox_of(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def connected_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def label_color_components(grid: np.ndarray, color: int):
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=int)
    comps = {}
    next_id = 1
    for comp in connected_components_of_color(grid, color):
        for r, c in comp:
            labels[r, c] = next_id
        comps[next_id] = set(comp)
        next_id += 1
    return labels, comps


def primitive_cells(kind: str, anchor: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0 = anchor
    return tuple(sorted((r0 + dr, c0 + dc) for dr, dc in PRIMITIVE_OFFSETS[kind]))


def primitive_center(kind: str, anchor: tuple[int, int]) -> tuple[int, int]:
    r, c = anchor
    if kind == "square":
        return r, c
    if kind == "hbar":
        return r, c + 1
    return r + 1, c


def enumerate_primitive_placements(cells: set[tuple[int, int]]):
    placements = []
    for r, c in sorted(cells):
        for kind in ("square", "hbar", "vbar"):
            p_cells = primitive_cells(kind, (r, c))
            if set(p_cells).issubset(cells):
                placements.append(
                    {
                        "kind": kind,
                        "anchor": (r, c),
                        "center": primitive_center(kind, (r, c)),
                        "cells": p_cells,
                        "cellset": set(p_cells),
                    }
                )
    placements.sort(key=lambda p: (p["center"], p["kind"], p["anchor"]))
    return placements


def unique_decomposition(coords: list[tuple[int, int]]):
    cellset = set(coords)
    placements = enumerate_primitive_placements(cellset)
    by_cell = defaultdict(list)
    for idx, placement in enumerate(placements):
        for cell in placement["cells"]:
            by_cell[cell].append(idx)

    solutions = []

    def search(remaining: set[tuple[int, int]], chosen: list[int]):
        if len(solutions) >= 2:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return
        pivot = None
        pivot_options = None
        for cell in sorted(remaining):
            options = [idx for idx in by_cell[cell] if placements[idx]["cellset"].issubset(remaining)]
            if pivot is None or len(options) < len(pivot_options):
                pivot = cell
                pivot_options = options
            if not pivot_options:
                return
        for idx in pivot_options:
            search(remaining - placements[idx]["cellset"], chosen + [idx])

    search(set(cellset), [])
    if len(solutions) != 1:
        return None
    return [placements[idx] for idx in solutions[0]]


def ray_budget(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = start
    h, w = shape
    if direction == UP:
        return r
    if direction == DOWN:
        return h - 1 - r
    if direction == LEFT:
        return c
    return w - 1 - c


def touches_component(cell: tuple[int, int], comp: set[tuple[int, int]]) -> bool:
    return any(n in comp for n in neighbors4(cell))


def trace_ray(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    obstacle_labels: np.ndarray,
    obstacle_comps: dict[int, set[tuple[int, int]]],
):
    budget = ray_budget(start, direction, shape)
    path = []
    if budget <= 0:
        return path, False

    pos = start
    hugging = False
    obstacle_id = 0
    heading = direction
    rerouted = False
    safety = shape[0] * shape[1] * 8

    while len(path) < budget and safety > 0:
        safety -= 1
        if not hugging:
            nxt = add(pos, direction)
            if not in_bounds(shape, nxt):
                break
            label = int(obstacle_labels[nxt])
            if label:
                hugging = True
                obstacle_id = label
                heading = turn_left(direction)
                rerouted = True
                continue
            pos = nxt
            path.append(pos)
            continue

        forward = add(pos, direction)
        if in_bounds(shape, forward) and int(obstacle_labels[forward]) == 0:
            pos = forward
            path.append(pos)
            hugging = False
            continue

        comp = obstacle_comps[obstacle_id]
        moved = False
        for cand in (turn_right(heading), heading, turn_left(heading), turn_back(heading)):
            nxt = add(pos, cand)
            if not in_bounds(shape, nxt):
                continue
            if int(obstacle_labels[nxt]) != 0:
                continue
            if touches_component(nxt, comp):
                pos = nxt
                path.append(pos)
                heading = cand
                moved = True
                break
        if not moved:
            raise AssertionError("wall-following failed on a valid obstacle sprite")

    assert len(path) == budget, "ray must consume its full length budget"
    return path, rerouted


def analyze_grid(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    grey_comps = connected_components_of_color(g, GREY)
    primitives = []
    for comp_id, comp in enumerate(grey_comps):
        solution = unique_decomposition(comp)
        assert solution is not None, "each gray component must have a unique valid decomposition"
        for primitive in solution:
            item = dict(primitive)
            item["component_id"] = comp_id
            primitives.append(item)

    primitives.sort(key=lambda p: (p["center"][0], p["center"][1], p["kind"]))

    obstacle_labels, obstacle_comps = label_color_components(g, ORANGE)
    rays = []
    for prim_index, primitive in enumerate(primitives):
        for direction in RAY_DIRS[primitive["kind"]]:
            path, rerouted = trace_ray(primitive["center"], direction, g.shape, obstacle_labels, obstacle_comps)
            rays.append(
                {
                    "primitive_index": prim_index,
                    "component_id": primitive["component_id"],
                    "kind": primitive["kind"],
                    "start": primitive["center"],
                    "direction": direction,
                    "color": RAY_COLOR[primitive["kind"]],
                    "budget": ray_budget(primitive["center"], direction, g.shape),
                    "path": path,
                    "rerouted": rerouted,
                    "primitive_cells": set(primitive["cells"]),
                }
            )

    out = g.copy()
    for ray in rays:
        for r, c in ray["path"]:
            if int(out[r, c]) in (BLACK, GREY):
                out[r, c] = ray["color"]
    out[out == GREY] = BLACK
    return out, primitives, rays, grey_comps, obstacle_comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose gray composite blocks, let shape type choose ray directions, and bend those rays clockwise around orange obstacles.

    Input structure:
    - The background is black, with 2-3 gray 4-connected composite blocks and 2-4 orange obstacle sprites.
    - Every gray block is made from non-overlapping 2x2 squares and 1x3 or 3x1 bars, with a unique valid decomposition.
    - Orange sprites are separate 4-connected irregular components that do not touch the gray blocks.

    Rule:
    - Decompose each gray block into its primitive pieces.
    - A 2x2 square emits teal rays in the four cardinal directions from its designated source cell; a 1x3 bar emits red rays left/right from its center cell; a 3x1 bar emits red rays up/down from its center cell.
    - Each ray has a fixed length equal to its unobstructed distance to the grid edge. When its next step would enter an orange obstacle, it follows that obstacle's outer boundary clockwise until it can resume the original direction, and it still consumes exactly the same length budget.
    - Paint rays in primitive-center order from top to bottom then left to right, keep earlier ray pixels on conflicts, and finally erase any gray pixels that remain.

    Key insight:
    - The decomposition is needed to recover shape type, because shape type decides which directions are legal; obstacle placement then changes the ray path without changing its length.
    """
    out, _primitives, _rays, _grey_comps, _obstacle_comps = analyze_grid(grid)
    return out


def rotate_cells(cells):
    return {(-c, r) for r, c in cells}


def flip_cells(cells):
    return {(r, -c) for r, c in cells}


def random_obstacle_shape():
    cells = set(random.choice(OBSTACLE_BASES))
    for _ in range(random.randrange(4)):
        cells = rotate_cells(cells)
    if random.random() < 0.5:
        cells = flip_cells(cells)
    return normalize_cells(cells)


def primitive_placement_candidates(occupied, kind_pool):
    if not occupied:
        return [(kind, (0, 0), set(primitive_cells(kind, (0, 0)))) for kind in kind_pool]

    min_r, min_c, max_r, max_c = bbox_of(occupied)
    candidates = []
    seen = set()
    for kind in kind_pool:
        for r in range(min_r - 3, max_r + 4):
            for c in range(min_c - 3, max_c + 4):
                cells = set(primitive_cells(kind, (r, c)))
                key = (kind, tuple(sorted(cells)))
                if key in seen:
                    continue
                seen.add(key)
                if cells & occupied:
                    continue
                if not any(n in occupied for cell in cells for n in neighbors4(cell)):
                    continue
                merged = occupied | cells
                nr0, nc0, nr1, nc1 = bbox_of(merged)
                if nr1 - nr0 + 1 > 7 or nc1 - nc0 + 1 > 7:
                    continue
                candidates.append((kind, (r, c), cells))
    random.shuffle(candidates)
    return candidates


def generate_component_patch(primitive_count: int):
    for _ in range(250):
        occupied = set()
        chosen = []
        target_kinds = ["square", "hbar", "vbar"]
        while len(chosen) < primitive_count:
            candidates = primitive_placement_candidates(occupied, target_kinds if len(chosen) == 0 else ("square", "hbar", "vbar"))
            if not candidates:
                break
            kind, anchor, cells = random.choice(candidates)
            occupied |= cells
            chosen.append((kind, anchor))
        if len(chosen) != primitive_count:
            continue
        norm = normalize_cells(occupied)
        solution = unique_decomposition(sorted(norm))
        if solution is None or len(solution) != primitive_count:
            continue
        return norm
    return None


def can_place_patch(grid: np.ndarray, patch: set[tuple[int, int]], top: int, left: int, pad: int = 2) -> bool:
    h, w = grid.shape
    global_cells = {(top + r, left + c) for r, c in patch}
    if not all(in_bounds(grid.shape, cell) for cell in global_cells):
        return False
    for r, c in global_cells:
        if int(grid[r, c]) != BLACK:
            return False
    for r, c in global_cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == GREY and (rr, cc) not in global_cells:
                    return False
    return True


def stamp_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int):
    for r, c in cells:
        grid[r, c] = color


def line_cells_from(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]):
    cells = []
    pos = start
    while True:
        pos = add(pos, direction)
        if not in_bounds(shape, pos):
            break
        cells.append(pos)
    return cells


def obstacle_cells_valid(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    for r, c in cells:
        if not in_bounds(grid.shape, (r, c)):
            return False
        if r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1):
            return False
        if int(grid[r, c]) != BLACK:
            return False
    for r, c in cells:
        for nr, nc in neighbors4((r, c)):
            if not in_bounds(grid.shape, (nr, nc)):
                return False
            if int(grid[nr, nc]) == GREY:
                return False
            if int(grid[nr, nc]) == ORANGE and (nr, nc) not in cells:
                return False
    return True


def try_place_obstacle_for_ray(grid: np.ndarray, ray_key: tuple[tuple[int, int], tuple[int, int], str]):
    _, primitives, rays, _grey_comps, _obstacle_comps = analyze_grid(grid)
    target = None
    for ray in rays:
        key = (ray["start"], ray["direction"], ray["kind"])
        if key == ray_key:
            target = ray
            break
    if target is None or target["rerouted"] or target["budget"] < 4:
        return None

    line = line_cells_from(target["start"], target["direction"], grid.shape)
    candidate_hits = [cell for cell in line[:-1] if int(grid[cell]) == BLACK]
    random.shuffle(candidate_hits)
    for hit in candidate_hits[:8]:
        for _ in range(12):
            shape = list(random_obstacle_shape())
            random.shuffle(shape)
            for anchor in shape:
                cells = {(hit[0] + r - anchor[0], hit[1] + c - anchor[1]) for r, c in shape}
                if not obstacle_cells_valid(grid, cells):
                    continue
                test = np.array(grid, copy=True)
                stamp_cells(test, cells, ORANGE)
                try:
                    _out, _prims, test_rays, _grey, _obs = analyze_grid(test)
                except AssertionError:
                    continue
                target_now = None
                for ray in test_rays:
                    key = (ray["start"], ray["direction"], ray["kind"])
                    if key == ray_key:
                        target_now = ray
                if target_now is not None and target_now["rerouted"]:
                    return cells
    return None


def semantic_checks(grid: np.ndarray) -> bool:
    try:
        out, primitives, rays, grey_comps, obstacle_comps = analyze_grid(grid)
    except AssertionError:
        return False

    if not (2 <= len(grey_comps) <= 3):
        return False
    if not (2 <= len(obstacle_comps) <= 4):
        return False

    comp_sizes = defaultdict(int)
    for primitive in primitives:
        comp_sizes[primitive["component_id"]] += 1
    if not all(2 <= count <= 4 for count in comp_sizes.values()):
        return False

    kinds = {primitive["kind"] for primitive in primitives}
    if not {"square", "hbar", "vbar"}.issubset(kinds):
        return False

    rerouted = [ray for ray in rays if ray["rerouted"]]
    straight = [ray for ray in rays if not ray["rerouted"]]
    if len(rerouted) < 2 or not straight:
        return False

    grey_owner = {}
    for comp_id, comp in enumerate(grey_comps):
        for cell in comp:
            grey_owner[cell] = comp_id

    seen = {}
    overlap_count = 0
    total_cells = 0
    for ray in rays:
        if len(ray["path"]) != ray["budget"]:
            return False
        for cell in ray["path"]:
            total_cells += 1
            if int(grid[cell]) == ORANGE:
                return False
            if int(grid[cell]) == GREY and grey_owner.get(cell) != ray["component_id"]:
                return False
            owner = seen.get(cell)
            key = (ray["start"], ray["direction"], ray["kind"])
            if owner is not None and owner != key:
                overlap_count += 1
            seen[cell] = key

    if overlap_count > max(12, total_cells // 4):
        return False

    if TEAL not in out or RED not in out:
        return False

    if np.array_equal(out, grid):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        h = random.randint(17, 22)
        w = random.randint(17, 22)
        grid = np.zeros((h, w), dtype=int)

        comp_count = random.randint(2, 3)
        primitive_counts = [random.randint(2, 4) for _ in range(comp_count)]
        patches = []
        for count in primitive_counts:
            patch = generate_component_patch(count)
            if patch is None:
                break
            patches.append(patch)
        if len(patches) != comp_count:
            continue

        placed = []
        ok = True
        for patch in sorted(patches, key=lambda p: (bbox_of(p)[2] - bbox_of(p)[0], bbox_of(p)[3] - bbox_of(p)[1]), reverse=True):
            r0, c0, r1, c1 = bbox_of(patch)
            ph = r1 - r0 + 1
            pw = c1 - c0 + 1
            found = False
            for _try in range(200):
                top = random.randint(1, h - ph - 2)
                left = random.randint(1, w - pw - 2)
                if can_place_patch(grid, patch, top, left):
                    global_cells = {(top + r, left + c) for r, c in patch}
                    stamp_cells(grid, global_cells, GREY)
                    placed.append(global_cells)
                    found = True
                    break
            if not found:
                ok = False
                break
        if not ok:
            continue

        try:
            _out, primitives, rays, _grey_comps, _obs = analyze_grid(grid)
        except AssertionError:
            continue

        candidate_keys = [
            (ray["start"], ray["direction"], ray["kind"])
            for ray in rays
            if ray["budget"] >= 4 and not ray["rerouted"]
        ]
        random.shuffle(candidate_keys)
        obstacle_target = 2
        placed_obstacles = 0
        for ray_key in candidate_keys:
            cells = try_place_obstacle_for_ray(grid, ray_key)
            if cells is None:
                continue
            stamp_cells(grid, cells, ORANGE)
            placed_obstacles += 1
            if placed_obstacles >= obstacle_target:
                break
        if placed_obstacles < 2:
            continue

        if semantic_checks(grid):
            return grid

    raise RuntimeError("failed to generate a valid ARC instance")
