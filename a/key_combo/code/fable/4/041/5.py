from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
KING_DIRS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in ORTHO_DIRS:
        yield r + dr, c + dc


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in (BG, RED):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "coord_set": set(coords),
                    "area": int(len(coords)),
                }
            )
    return objects


def _halo(coord_set: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in coord_set:
        for dr, dc in KING_DIRS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.add((rr, cc))
    return out


def _spark_coord(grid: np.ndarray) -> tuple[int, int]:
    spark = np.argwhere(np.asarray(grid, dtype=int) == RED)
    if len(spark) != 1:
        raise ValueError("input must contain exactly one red spark pixel")
    return int(spark[0, 0]), int(spark[0, 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every object reached by size-gated fire propagation from a single spark.

    Input structure:
    - A black background contains separated 4-connected monochrome objects of various colors and sizes.
    - Exactly one red singleton spark is present, and red does not belong to any regular object.

    Rule:
    - Any object whose cells are within Chebyshev distance 1 of the spark ignites first.
    - An ignited object with area at least 4 relays the fire to every still-unlit object within Chebyshev distance 1.
    - Ignited objects with area at most 3 do not relay further; after propagation stabilizes, recolor every ignited object to red and keep all other pixels unchanged.

    Key insight:
    - The contact graph is based on one-step Chebyshev adjacency, but only large enough objects can continue the chain, so a nearby object may stay untouched if every path to it passes through a small burned component.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    spark = _spark_coord(g)
    objects = _find_objects(g)
    halos = [_halo(obj["coord_set"], g.shape) for obj in objects]

    adjacency = [set() for _ in objects]
    initial = set()
    for i, obj in enumerate(objects):
        if spark in halos[i]:
            initial.add(i)
        for j in range(i + 1, len(objects)):
            if halos[i] & objects[j]["coord_set"]:
                adjacency[i].add(j)
                adjacency[j].add(i)

    burned = set(initial)
    q = deque(i for i in initial if int(objects[i]["area"]) >= 4)
    while q:
        idx = q.popleft()
        for nxt in adjacency[idx]:
            if nxt in burned:
                continue
            burned.add(nxt)
            if int(objects[nxt]["area"]) >= 4:
                q.append(nxt)

    for idx in burned:
        coords = objects[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = RED
    return out


def _rotate_local(cells: set[tuple[int, int]], vertical: bool) -> set[tuple[int, int]]:
    if not vertical:
        return set(cells)
    return {(c, r) for r, c in cells}


def _big_line_shape(left_neighbor: bool, right_neighbor: bool) -> set[tuple[int, int]]:
    if left_neighbor and right_neighbor:
        options = [
            {(0, -1), (0, 0), (0, 1), (1, 0)},
            {(0, -1), (0, 0), (0, 1), (-1, 0)},
            {(0, -1), (0, 0), (0, 1), (-1, 0), (1, 0)},
        ]
    elif right_neighbor:
        options = [
            {(0, 0), (0, 1), (1, 0), (1, 1)},
            {(-1, 0), (-1, 1), (0, 0), (0, 1)},
            {(-1, 0), (0, 0), (1, 0), (0, 1)},
        ]
    else:
        options = [
            {(0, -1), (0, 0), (1, -1), (1, 0)},
            {(-1, -1), (-1, 0), (0, -1), (0, 0)},
            {(-1, 0), (0, 0), (1, 0), (0, -1)},
        ]
    return set(random.choice(options))


def _small_line_shape(left_neighbor: bool, right_neighbor: bool) -> set[tuple[int, int]]:
    if left_neighbor and right_neighbor:
        return {(0, -1), (0, 0), (0, 1)}
    if right_neighbor:
        return random.choice([{(0, 0), (0, 1)}, {(0, 0), (1, 0), (0, 1)}])
    return random.choice([{(0, -1), (0, 0)}, {(0, -1), (0, 0), (1, 0)}])


def _generic_distractor_shape(kind: str) -> set[tuple[int, int]]:
    if kind == "big":
        choices = [
            {(0, 0), (0, 1), (1, 0), (1, 1)},
            {(0, 0), (1, 0), (2, 0), (2, 1)},
            {(0, 1), (1, 1), (2, 0), (2, 1)},
            {(0, 0), (0, 1), (0, 2), (1, 1)},
            {(0, 0), (1, 0), (1, 1), (1, 2)},
        ]
    else:
        choices = [
            {(0, 0)},
            {(0, 0), (0, 1)},
            {(0, 0), (1, 0)},
            {(0, 0), (0, 1), (1, 0)},
            {(0, 0), (0, 1), (1, 1)},
        ]
    return set(random.choice(choices))


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _all_far_enough(candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _build_chain_scene() -> tuple[list[set[tuple[int, int]]], list[str], tuple[int, int]]:
    left_len = random.randint(1, 2)
    relay_len = random.randint(1, 2)
    blocked_len = random.randint(1, 2)

    kinds = ["big"] * left_len
    start_idx = len(kinds)
    kinds.append("big")
    kinds.extend(["big"] * relay_len)
    kinds.append("small")
    kinds.extend(["big"] + random.choices(["big", "small"], k=max(0, blocked_len - 1)))

    centers = [(0, idx * 3) for idx in range(len(kinds))]
    objects: list[set[tuple[int, int]]] = []
    for idx, kind in enumerate(kinds):
        left_neighbor = idx > 0
        right_neighbor = idx + 1 < len(kinds)
        if kind == "big":
            local = _big_line_shape(left_neighbor, right_neighbor)
        else:
            local = _small_line_shape(left_neighbor, right_neighbor)
        cr, cc = centers[idx]
        objects.append({(cr + dr, cc + dc) for dr, dc in local})

    start_cells = objects[start_idx]
    _start_r, start_c = centers[start_idx]
    spark_options = []
    if any(r == -1 and c in (start_c - 1, start_c, start_c + 1) for r, c in start_cells):
        spark_options.append((-2, start_c))
    if any(r == 1 and c in (start_c - 1, start_c, start_c + 1) for r, c in start_cells):
        spark_options.append((2, start_c))
    if not spark_options:
        spark_options.append((-2, centers[start_idx][1]))
    spark = random.choice(spark_options)
    return objects, kinds, spark


def _place_scene() -> tuple[np.ndarray, list[dict]]:
    vertical = bool(random.getrandbits(1))
    chain_cells, chain_kinds, spark = _build_chain_scene()
    if vertical:
        chain_cells = [_rotate_local(cells, True) for cells in chain_cells]
        spark = (spark[1], spark[0])

    all_chain = set().union(*chain_cells)
    all_chain.add(spark)
    min_r = min(r for r, _ in all_chain)
    max_r = max(r for r, _ in all_chain)
    min_c = min(c for _, c in all_chain)
    max_c = max(c for _, c in all_chain)
    chain_h = max_r - min_r + 1
    chain_w = max_c - min_c + 1

    extra_h = random.randint(5, 10)
    extra_w = random.randint(5, 10)
    h = min(30, chain_h + extra_h)
    w = min(30, chain_w + extra_w)
    top = random.randint(2 - min_r, h - 3 - max_r)
    left = random.randint(2 - min_c, w - 3 - max_c)

    placed_chain = [_translate(cells, top, left) for cells in chain_cells]
    spark = (spark[0] + top, spark[1] + left)
    occupied = set().union(*placed_chain)
    occupied.add(spark)

    node_info = [{"cells": cells, "kind": kind, "chain": True} for cells, kind in zip(placed_chain, chain_kinds)]

    max_distractors = min(2, len(PALETTE) - len(node_info))
    distractor_count = random.randint(0, max_distractors)
    for _ in range(distractor_count):
        for _attempt in range(200):
            kind = random.choice(["big", "small"])
            local = _normalize(_generic_distractor_shape(kind))
            dh = max(r for r, _ in local) + 1
            dw = max(c for _, c in local) + 1
            base_r = random.randint(1, h - dh - 2)
            base_c = random.randint(1, w - dw - 2)
            candidate = _translate(local, base_r, base_c)
            if _all_far_enough(candidate, occupied):
                node_info.append({"cells": candidate, "kind": kind, "chain": False})
                occupied |= candidate
                break

    random.shuffle(PALETTE)
    colors = iter(PALETTE)
    grid = np.zeros((h, w), dtype=int)
    for obj in node_info:
        color = next(colors)
        for r, c in obj["cells"]:
            grid[r, c] = color
        obj["color"] = color
    grid[spark] = RED
    return grid, node_info


def _validate_scene(grid: np.ndarray, node_info: list[dict]) -> bool:
    out = main(grid)
    if np.array_equal(grid, out):
        return False

    burned_small = []
    burned_big = 0
    for obj in node_info:
        burned = all(int(out[r, c]) == RED for r, c in obj["cells"])
        if burned and obj["kind"] == "small":
            burned_small.append(obj)
        if burned and obj["kind"] == "big":
            burned_big += 1

    if burned_big < 2 or not burned_small:
        return False

    for small in burned_small:
        halo = _halo(set(small["cells"]), grid.shape)
        for other in node_info:
            if other is small or other["kind"] != "big":
                continue
            if any(int(out[r, c]) != RED for r, c in other["cells"]) and halo & set(other["cells"]):
                return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid, node_info = _place_scene()
        if _validate_scene(grid, node_info):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
