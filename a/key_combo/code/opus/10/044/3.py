from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BACKGROUND = 0
SEED = 2
SPRITE = 5
WALL = 8

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ORTHOGONAL_SEED_SIDES = {
    "top": ("left", "right"),
    "bottom": ("left", "right"),
    "left": ("top", "bottom"),
    "right": ("top", "bottom"),
}


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def flip_side(side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[side]


def mirror_local(coords: tuple[tuple[int, int], ...], box_h: int, box_w: int, side: str) -> tuple[tuple[int, int], ...]:
    if side in {"top", "bottom"}:
        return tuple(sorted((box_h - 1 - r, c) for r, c in coords))
    return tuple(sorted((r, box_w - 1 - c) for r, c in coords))


def side_portals(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    portals = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if side == "top" and r != 0:
                continue
            if side == "bottom" and r != h - 1:
                continue
            if side == "left" and c != 0:
                continue
            if side == "right" and c != w - 1:
                continue
            if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in neighbors4(r, c)):
                portals.append((r, c))
    return portals


def build_shape_catalog() -> list[dict]:
    catalog = []
    seen = set()
    for h in range(2, 5):
        for w in range(2, 5):
            area = h * w
            for n in range(3, min(6, area) + 1):
                for combo in combinations(range(area), n):
                    mask = np.zeros((h, w), dtype=bool)
                    mask.ravel()[list(combo)] = True
                    coords = [(idx // w, idx % w) for idx in combo]
                    norm = normalize_coords(coords)
                    if norm in seen:
                        continue
                    seen.add(norm)
                    comp = components_of_color(mask.astype(int), 1)
                    if len(comp) != 1:
                        continue
                    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                        continue
                    local = tuple(sorted(coords))
                    if mirror_local(local, h, w, "top") == local:
                        continue
                    if mirror_local(local, h, w, "left") == local:
                        continue
                    portals = {side: side_portals(mask, side) for side in SIDE_STEP}
                    if not any(portals.values()):
                        continue
                    catalog.append(
                        {
                            "h": h,
                            "w": w,
                            "mask": mask,
                            "coords": local,
                            "portals": portals,
                        }
                    )
    return catalog


SHAPE_CATALOG = build_shape_catalog()


def mirror_global(coords: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side in {"top", "bottom"}:
        return [(int(r0 + r1 - r), int(c)) for r, c in coords]
    return [(int(r), int(c0 + c1 - c)) for r, c in coords]


def inside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def crossing_side(parent: tuple[int, int], child: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    pr, pc = parent
    cr, cc = child
    if cr == r0 and pr == r0 - 1 and c0 <= cc <= c1:
        return "top"
    if cr == r1 and pr == r1 + 1 and c0 <= cc <= c1:
        return "bottom"
    if cc == c0 and pc == c0 - 1 and r0 <= cr <= r1:
        return "left"
    if cc == c1 and pc == c1 + 1 and r0 <= cr <= r1:
        return "right"
    raise AssertionError("invalid bbox crossing")


def analyze_seed(grid: np.ndarray, seed: tuple[int, int], components: list[np.ndarray]):
    owner = -np.ones(grid.shape, dtype=int)
    boxes = []
    for idx, comp in enumerate(components):
        owner[comp[:, 0], comp[:, 1]] = idx
        boxes.append(bbox_coords(comp))

    visited: set[tuple[int, int]] = {seed}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {seed: None}
    q = deque([seed])
    touched = set()
    while q:
        cell = q.popleft()
        for nr, nc in neighbors4(*cell):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if grid[nr, nc] == SPRITE:
                touched.add(int(owner[nr, nc]))
            elif grid[nr, nc] == BACKGROUND and (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = cell
                q.append((nr, nc))

    assert len(touched) == 1, "each seed must reach exactly one sprite"
    comp_idx = next(iter(touched))
    bbox = boxes[comp_idx]
    sides = set()
    for cell, prev in parent.items():
        if prev is None:
            continue
        if inside_bbox(cell, bbox) and not inside_bbox(prev, bbox):
            sides.add(crossing_side(prev, cell, bbox))
    assert len(sides) == 1, "entry side must be unique"
    return comp_idx, next(iter(sides)), visited


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each gray sprite according to the corridor side from which its red seed reaches it.

    Input structure:
    - Black cells are open corridor/background cells and teal cells are maze walls.
    - The grid contains 2-3 irregular gray 4-connected sprites and the same number of red seed pixels.
    - Each seed can reach exactly one gray sprite through black 4-connected corridor cells, and the path enters that sprite's bounding box from one unique side.

    Rule:
    - Keep all teal walls and red seeds unchanged.
    - For each seed, flood through black corridor cells to identify the unique gray sprite it reaches.
    - Determine which side of that sprite's bounding box the corridor enters: top/bottom means flip the sprite vertically inside its own bounding box; left/right means flip it horizontally inside its own bounding box.
    - Clear the original gray sprite, then draw its mirrored version in gray, skipping any target cell occupied by a teal wall or red seed.

    Key insight:
    - The red seed does not encode the mirror directly; the deciding signal is the topology of the corridor and specifically which side of the sprite's bounding box that corridor enters.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    components = components_of_color(g, SPRITE)
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(g == SEED)]
    assert components and len(components) == len(seeds)

    out = g.copy()
    for comp in components:
        out[comp[:, 0], comp[:, 1]] = BACKGROUND

    placements = []
    for seed in seeds:
        comp_idx, side, _visited = analyze_seed(g, seed, components)
        placements.append((comp_idx, side))

    used = [comp_idx for comp_idx, _side in placements]
    assert sorted(used) == list(range(len(components)))

    for comp_idx, side in placements:
        comp = components[comp_idx]
        bbox = bbox_coords(comp)
        for r, c in mirror_global(comp, bbox, side):
            if in_bounds(out.shape, r, c) and out[r, c] not in {WALL, SEED}:
                out[r, c] = SPRITE

    return out


def shortest_path(start: tuple[int, int], goal: tuple[int, int], allowed: set[tuple[int, int]]):
    q = deque([start])
    parent = {start: None}
    while q:
        cell = q.popleft()
        if cell == goal:
            break
        for nxt in neighbors4(*cell):
            if nxt in allowed and nxt not in parent:
                parent[nxt] = cell
                q.append(nxt)
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def quadrant_slots(h: int, w: int) -> list[tuple[int, int, int, int]]:
    row_cut = random.randint(9, h - 10)
    col_cut = random.randint(9, w - 10)
    slots = [
        (1, 1, row_cut - 2, col_cut - 2),
        (1, col_cut + 2, row_cut - 2, w - 2),
        (row_cut + 2, 1, h - 2, col_cut - 2),
        (row_cut + 2, col_cut + 2, h - 2, w - 2),
    ]
    assert all(r1 - r0 + 1 >= 7 and c1 - c0 + 1 >= 7 for r0, c0, r1, c1 in slots)
    return slots


def boundary_cells(slot: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = slot
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def choose_shape(side: str, used_shapes: set[tuple[tuple[int, int], ...]]):
    candidates = []
    for shape in SHAPE_CATALOG:
        if shape["coords"] in used_shapes:
            continue
        if not shape["portals"][side]:
            continue
        mirrored = set(mirror_local(shape["coords"], shape["h"], shape["w"], side))
        target_only = mirrored - set(shape["coords"])
        if not target_only:
            continue
        candidates.append(shape)
    return random.choice(candidates) if candidates else None


def black_pockets_for_shape(shape: dict, side: str, portal: tuple[int, int]) -> set[tuple[int, int]]:
    mirrored = set(mirror_local(shape["coords"], shape["h"], shape["w"], side))
    target_only = list(mirrored - set(shape["coords"]))
    chosen: set[tuple[int, int]] = set()
    for cell in target_only:
        if abs(cell[0] - portal[0]) + abs(cell[1] - portal[1]) == 1:
            continue
        if any(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1 for other in chosen):
            continue
        chosen.add(cell)
        if len(chosen) == 2:
            break
    return chosen


def bbox_cells(top: int, left: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + h) for c in range(left, left + w)}


def moat_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    around = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in cells:
                around.add((nr, nc))
    return around


def try_build_object(
    grid: np.ndarray,
    slot: tuple[int, int, int, int],
    side: str,
    used_shapes: set[tuple[tuple[int, int], ...]],
):
    shape = choose_shape(side, used_shapes)
    if shape is None:
        return None

    slot_r0, slot_c0, slot_r1, slot_c1 = slot
    top_min = slot_r0 + 1
    left_min = slot_c0 + 1
    top_max = slot_r1 - shape["h"]
    left_max = slot_c1 - shape["w"]
    if top_min > top_max or left_min > left_max:
        return None

    portals = shape["portals"][side][:]
    random.shuffle(portals)
    placements = [(top, left) for top in range(top_min, top_max + 1) for left in range(left_min, left_max + 1)]
    random.shuffle(placements)

    for top, left in placements:
        box = bbox_cells(top, left, shape["h"], shape["w"])
        if any(grid[r, c] != WALL for r, c in box):
            continue
        for portal_local in portals:
            pr, pc = top + portal_local[0], left + portal_local[1]
            dr, dc = SIDE_STEP[side]
            outside = (pr + dr, pc + dc)
            if not (slot_r0 <= outside[0] <= slot_r1 and slot_c0 <= outside[1] <= slot_c1):
                continue
            if outside in box:
                continue
            moat = moat_cells(box)
            allowed = {
                (r, c)
                for r in range(slot_r0, slot_r1 + 1)
                for c in range(slot_c0, slot_c1 + 1)
                if (r, c) not in box and ((r, c) not in moat or (r, c) == outside)
            }
            seed_side = random.choice(ORTHOGONAL_SEED_SIDES[side])
            goals = boundary_cells(slot, seed_side)
            random.shuffle(goals)
            for goal in goals:
                if goal not in allowed:
                    continue
                if goal == outside:
                    continue
                if side in {"top", "bottom"} and goal[0] == outside[0]:
                    continue
                if side in {"left", "right"} and goal[1] == outside[1]:
                    continue
                path = shortest_path(outside, goal, allowed)
                if path is None or len(path) < 4:
                    continue

                pockets = black_pockets_for_shape(shape, side, portal_local)
                sprite_cells = {(top + r, left + c) for r, c in shape["coords"]}
                pocket_cells = {(top + r, left + c) for r, c in pockets}
                if any(abs(br - pr) + abs(bc - pc) == 1 for br, bc in pocket_cells):
                    continue

                for r, c in sprite_cells:
                    grid[r, c] = SPRITE
                for r, c in pocket_cells:
                    grid[r, c] = BACKGROUND
                grid[pr, pc] = BACKGROUND
                for r, c in path[:-1]:
                    grid[r, c] = BACKGROUND
                seed = path[-1]
                grid[seed] = SEED

                bbox = bbox_coords(np.array(sorted(sprite_cells), dtype=int))
                return {
                    "shape_key": shape["coords"],
                    "side": side,
                    "seed": seed,
                    "bbox": bbox,
                    "sprite_cells": sprite_cells,
                }
    return None


def expected_output_from_plan(grid: np.ndarray, plans: list[dict]) -> np.ndarray:
    out = grid.copy()
    for plan in plans:
        for r, c in plan["sprite_cells"]:
            out[r, c] = BACKGROUND
    for plan in plans:
        coords = np.array(sorted(plan["sprite_cells"]), dtype=int)
        for r, c in mirror_global(coords, plan["bbox"], plan["side"]):
            if in_bounds(out.shape, r, c) and out[r, c] not in {WALL, SEED}:
                out[r, c] = SPRITE
    return out


def semantic_check(inp: np.ndarray, plans: list[dict]) -> bool:
    components = components_of_color(inp, SPRITE)
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(inp == SEED)]
    if len(components) != len(plans) or len(seeds) != len(plans):
        return False

    seen_axes = set()
    used_components = set()
    changed_per_object = []
    for plan in plans:
        comp_idx, side, visited = analyze_seed(inp, plan["seed"], components)
        if side != plan["side"]:
            return False
        used_components.add(comp_idx)
        seen_axes.add("ud" if side in {"top", "bottom"} else "lr")
        comp = components[comp_idx]
        before = {tuple(map(int, rc)) for rc in comp}
        after = set(mirror_global(comp, bbox_coords(comp), side))
        visible_after = {cell for cell in after if inp[cell] != WALL and inp[cell] != SEED}
        changed_per_object.append(before != visible_after)
        if sum(1 for cell in visited if inside_bbox(cell, bbox_coords(comp))) != 1:
            return False

    if used_components != set(range(len(components))):
        return False
    if seen_axes != {"ud", "lr"}:
        return False
    if not all(changed_per_object):
        return False

    expected = expected_output_from_plan(inp, plans)
    actual = main(inp)
    return np.array_equal(actual, expected)


def build_input() -> tuple[np.ndarray, list[dict]]:
    h = random.randint(20, 26)
    w = random.randint(20, 26)
    grid = np.full((h, w), WALL, dtype=int)

    slots = quadrant_slots(h, w)
    obj_count = random.choice([2, 3])
    chosen_slots = random.sample(slots, obj_count)

    side_pool = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    if obj_count == 3:
        side_pool.append(random.choice(list(SIDE_STEP)))
    random.shuffle(side_pool)

    plans = []
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for slot, side in zip(chosen_slots, side_pool):
        plan = try_build_object(grid, slot, side, used_shapes)
        if plan is None:
            raise RuntimeError("failed to place object")
        used_shapes.add(plan["shape_key"])
        plans.append(plan)

    return grid, plans


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, plans = build_input()
        except Exception:
            continue
        if semantic_check(inp, plans):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
