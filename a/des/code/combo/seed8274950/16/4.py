from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
DIR_NORM = {d: np.array(d, dtype=float) / np.hypot(d[0], d[1]) for d in DIRS8}

GEN_ATTEMPTS = 500
GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
OBJECT_COUNT_RANGE = (3, 6)
MISS_PROBABILITY = 0.5
MIN_RAY_STEPS = 2
MAX_ALIGN_ATTEMPTS = 120
MAX_RANDOM_ATTEMPTS = 160

CARDINAL_ROTATIONS = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
DIAGONAL_ROTATIONS = {(-1, 1): 0, (1, 1): 1, (1, -1): 2, (-1, -1): 3}
CARDINAL_BASES = (
    ((0, 0), (0, -1), (0, -2), (1, -2)),
    ((0, 0), (0, -1), (0, -2), (-1, -2), (-1, -3)),
    ((0, 0), (0, -1), (0, -2), (1, -2), (1, -1)),
)
DIAGONAL_BASES = (
    ((0, 0), (1, 0), (1, -1), (2, -1), (2, -2), (3, -1)),
    ((0, 0), (1, 0), (1, -1), (2, -1), (2, -2), (3, -2), (3, -1)),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _quantize_direction(vec: np.ndarray) -> tuple[int, int]:
    best_dir = DIRS8[0]
    best_score = float("-inf")
    for direction in DIRS8:
        score = float(vec @ DIR_NORM[direction])
        if score > best_score:
            best_score = score
            best_dir = direction
    return best_dir


def _tip_and_direction(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], bool]:
    centroid = coords.mean(axis=0)
    delta = coords.astype(float) - centroid
    dist2 = np.einsum("ij,ij->i", delta, delta)
    max_dist2 = float(np.max(dist2))
    candidates = np.flatnonzero(np.abs(dist2 - max_dist2) <= 1e-9)
    best = min(candidates, key=lambda idx: (int(coords[idx, 0]), int(coords[idx, 1])))
    tip = (int(coords[best, 0]), int(coords[best, 1]))
    direction = _quantize_direction(delta[best])
    return tip, direction, len(candidates) == 1


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    owner = np.full(g.shape, -1, dtype=int)
    objects: list[dict] = []
    for obj_id, comp in enumerate(_components(g != bg)):
        coords = np.array(comp, dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        colors = np.unique(g[rows, cols])
        assert colors.size == 1, "every object must be monochrome"
        tip, direction, unique_tip = _tip_and_direction(coords)
        owner[rows, cols] = obj_id
        objects.append(
            {
                "id": obj_id,
                "coords": coords,
                "bbox": (
                    int(np.min(rows)),
                    int(np.min(cols)),
                    int(np.max(rows)),
                    int(np.max(cols)),
                ),
                "tip": tip,
                "direction": direction,
                "unique_tip": unique_tip,
            }
        )
    return bg, objects, owner


def _trace_ray(
    tip: tuple[int, int],
    direction: tuple[int, int],
    source_id: int,
    owner: np.ndarray,
) -> tuple[list[tuple[int, int]], int | None]:
    h, w = owner.shape
    dr, dc = direction
    r, c = tip
    traversed: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return traversed, None
        hit = int(owner[r, c])
        if hit == source_id:
            continue
        if hit >= 0:
            return traversed, hit
        traversed.append((r, c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each foreground object's pointing direction from its centroid-to-tip
    vector, then recolor the first other object hit along that ray.
    """
    g = np.asarray(grid, dtype=int)
    _, objects, owner = _extract_objects(g)
    out = g.copy()

    order = sorted(objects, key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    for obj in order:
        source_coords = obj["coords"]
        source_color = int(out[int(source_coords[0, 0]), int(source_coords[0, 1])])
        _path, target_id = _trace_ray(obj["tip"], obj["direction"], int(obj["id"]), owner)
        if target_id is None:
            continue
        target_coords = objects[target_id]["coords"]
        out[target_coords[:, 0], target_coords[:, 1]] = source_color

    return out


def _rotate90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((c, -r) for r, c in coords)


def _rotate(coords: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    out = coords
    for _ in range(turns):
        out = _rotate90(out)
    return out


def _shape_library() -> dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], ...]]:
    library: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {direction: [] for direction in DIRS8}
    for direction, turns in CARDINAL_ROTATIONS.items():
        for base in CARDINAL_BASES:
            shape = _rotate(base, turns)
            coords = np.array(shape, dtype=int)
            tip, inferred_dir, unique_tip = _tip_and_direction(coords)
            if tip == (0, 0) and inferred_dir == direction and unique_tip:
                library[direction].append(shape)
    for direction, turns in DIAGONAL_ROTATIONS.items():
        for base in DIAGONAL_BASES:
            shape = _rotate(base, turns)
            coords = np.array(shape, dtype=int)
            tip, inferred_dir, unique_tip = _tip_and_direction(coords)
            if tip == (0, 0) and inferred_dir == direction and unique_tip:
                library[direction].append(shape)
    return {direction: tuple(shapes) for direction, shapes in library.items()}


SHAPE_LIBRARY = _shape_library()


def _expand_cells(cells: list[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                expanded.add((rr, cc))
    return expanded


def _place_shape(shape: tuple[tuple[int, int], ...], tip: tuple[int, int]) -> list[tuple[int, int]]:
    return [(tip[0] + r, tip[1] + c) for r, c in shape]


def _trace_candidate(
    tip: tuple[int, int],
    direction: tuple[int, int],
    source_cells: set[tuple[int, int]],
    occupied_owner: dict[tuple[int, int], int],
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], int | None]:
    dr, dc = direction
    r, c = tip
    traversed: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return traversed, None
        if (r, c) in source_cells:
            continue
        hit = occupied_owner.get((r, c))
        if hit is not None:
            return traversed, int(hit)
        traversed.append((r, c))


def _tip_ranges(shape: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[int, int, int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return -min(rows), h - 1 - max(rows), -min(cols), w - 1 - max(cols)


def _try_place_miss(
    h: int,
    w: int,
    obj_id: int,
    forbidden: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    occupied_owner: dict[tuple[int, int], int],
) -> dict | None:
    directions = random.sample(DIRS8, len(DIRS8))
    for direction in directions:
        shapes = random.sample(list(SHAPE_LIBRARY[direction]), len(SHAPE_LIBRARY[direction]))
        for shape in shapes:
            r_lo, r_hi, c_lo, c_hi = _tip_ranges(shape, h, w)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            for _ in range(MAX_RANDOM_ATTEMPTS):
                tip = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
                cells = _place_shape(shape, tip)
                if any(cell in forbidden or cell in reserved for cell in cells):
                    continue
                cell_set = set(cells)
                traversed, hit = _trace_candidate(tip, direction, cell_set, occupied_owner, h, w)
                if hit is None and len(traversed) >= MIN_RAY_STEPS:
                    return {
                        "id": obj_id,
                        "tip": tip,
                        "direction": direction,
                        "cells": cells,
                        "ray_cells": traversed,
                        "hit_id": None,
                    }
    return None


def _try_place_hit(
    h: int,
    w: int,
    obj_id: int,
    forbidden: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    occupied_owner: dict[tuple[int, int], int],
    placed: list[dict],
) -> dict | None:
    targets = random.sample(placed, len(placed))
    directions = random.sample(DIRS8, len(DIRS8))
    for target in targets:
        impact_cells = random.sample(target["cells"], len(target["cells"]))
        for direction in directions:
            shapes = random.sample(list(SHAPE_LIBRARY[direction]), len(SHAPE_LIBRARY[direction]))
            for shape in shapes:
                for impact in impact_cells:
                    for _ in range(MAX_ALIGN_ATTEMPTS):
                        distance = random.randint(MIN_RAY_STEPS + 1, max(h, w))
                        tip = (impact[0] - distance * direction[0], impact[1] - distance * direction[1])
                        cells = _place_shape(shape, tip)
                        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                            continue
                        if any(cell in forbidden or cell in reserved for cell in cells):
                            continue
                        cell_set = set(cells)
                        traversed, hit = _trace_candidate(tip, direction, cell_set, occupied_owner, h, w)
                        if hit == target["id"] and len(traversed) >= MIN_RAY_STEPS:
                            return {
                                "id": obj_id,
                                "tip": tip,
                                "direction": direction,
                                "cells": cells,
                                "ray_cells": traversed,
                                "hit_id": int(hit),
                            }
    return None


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], list[int | None]]:
    _bg, objects, owner = _extract_objects(grid)
    by_id = {int(obj["id"]): obj for obj in objects}
    targets: list[int | None] = []
    for obj in objects:
        _path, target = _trace_ray(obj["tip"], obj["direction"], int(obj["id"]), owner)
        targets.append(None if target is None else int(target))
    return [by_id[i] for i in range(len(objects))], targets


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(range(1, 10), n_objects)

    placed: list[dict] = []
    occupied_owner: dict[tuple[int, int], int] = {}
    forbidden: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    hit_count = 0

    for obj_id in range(n_objects):
        must_hit = obj_id > 0 and hit_count == 0 and obj_id == n_objects - 1
        prefer_miss = obj_id == 0 or random.random() < MISS_PROBABILITY

        candidate = None
        if obj_id == 0:
            candidate = _try_place_miss(h, w, obj_id, forbidden, reserved, occupied_owner)
        elif must_hit:
            candidate = _try_place_hit(h, w, obj_id, forbidden, reserved, occupied_owner, placed)
        else:
            if prefer_miss:
                candidate = _try_place_miss(h, w, obj_id, forbidden, reserved, occupied_owner)
                if candidate is None:
                    candidate = _try_place_hit(h, w, obj_id, forbidden, reserved, occupied_owner, placed)
            else:
                candidate = _try_place_hit(h, w, obj_id, forbidden, reserved, occupied_owner, placed)
                if candidate is None:
                    candidate = _try_place_miss(h, w, obj_id, forbidden, reserved, occupied_owner)
        if candidate is None:
            return None

        candidate["color"] = colors[obj_id]
        placed.append(candidate)
        if candidate["hit_id"] is not None:
            hit_count += 1

        for cell in candidate["cells"]:
            occupied_owner[cell] = obj_id
        forbidden.update(_expand_cells(candidate["cells"], h, w, radius=1))
        reserved.update(candidate["ray_cells"])

    grid = np.zeros((h, w), dtype=int)
    for obj in placed:
        for r, c in obj["cells"]:
            grid[r, c] = int(obj["color"])

    objects, targets = _analyze_grid(grid)
    if len(objects) != n_objects or not all(obj["unique_tip"] for obj in objects):
        return None

    actual_hits = sum(target is not None for target in targets)
    actual_misses = n_objects - actual_hits
    if actual_hits == 0 or actual_misses == 0:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 4")
