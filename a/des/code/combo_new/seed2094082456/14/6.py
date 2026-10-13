from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
LOCK_SIZE = 5
LOCK_AREA_RANGE = {21, 22, 23}
KEY_AREAS = {2, 3, 4}
SIDES = ("top", "bottom", "left", "right")
GRID_LIMIT = 30
PAIR_COUNT_RANGE = (1, 4)
GAP_RANGE = (1, 4)
PLACEMENT_MARGIN = 1
GENERATION_ATTEMPTS = 300
MIN_GRID_BY_PAIR_COUNT = {1: 10, 2: 16, 3: 21, 4: 25}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _shape_signature(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _component_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": _component_bbox(coords),
                    "area": len(coords),
                }
            )
    return components


def _lock_from_component(comp: dict, grid: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 + 1 != LOCK_SIZE or c1 - c0 + 1 != LOCK_SIZE:
        return None
    if comp["area"] not in LOCK_AREA_RANGE:
        return None

    color = int(comp["color"])
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if not np.all((patch == BG) | (patch == color)):
        return None

    cavity_mask = patch == BG
    cavity_area = int(np.count_nonzero(cavity_mask))
    if cavity_area not in KEY_AREAS:
        return None
    if not _mask_connected(cavity_mask):
        return None
    if not _mask_connected(~cavity_mask):
        return None

    touched = []
    if bool(cavity_mask[0, :].any()):
        touched.append("top")
    if bool(cavity_mask[-1, :].any()):
        touched.append("bottom")
    if bool(cavity_mask[:, 0].any()):
        touched.append("left")
    if bool(cavity_mask[:, -1].any()):
        touched.append("right")
    if len(touched) != 1:
        return None

    cavity_rel = [tuple(map(int, rc)) for rc in np.argwhere(cavity_mask)]
    cavity_abs = [(r0 + rr, c0 + cc) for rr, cc in cavity_rel]
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "side": touched[0],
        "cavity_coords": tuple(sorted(cavity_abs)),
        "signature": _shape_signature(cavity_abs),
    }


def _key_from_component(comp: dict) -> dict | None:
    if comp["area"] not in KEY_AREAS:
        return None
    return {
        "color": int(comp["color"]),
        "bbox": comp["bbox"],
        "coords": tuple(sorted(comp["coords"])),
        "signature": _shape_signature(comp["coords"]),
    }


def _match_possible(lock: dict, key: dict) -> bool:
    if lock["signature"] != key["signature"]:
        return False

    cavity = list(lock["cavity_coords"])
    key_coords = list(key["coords"])
    cavity_rows = [r for r, _ in cavity]
    cavity_cols = [c for _, c in cavity]
    key_rows = [r for r, _ in key_coords]
    key_cols = [c for _, c in key_coords]

    dr = min(cavity_rows) - min(key_rows)
    dc = min(cavity_cols) - min(key_cols)
    shifted = {(r + dr, c + dc) for r, c in key_coords}
    if shifted != set(cavity):
        return False

    r0, c0, r1, c1 = lock["bbox"]
    kr0, kc0, kr1, kc1 = key["bbox"]
    side = lock["side"]
    if side == "top":
        return dc == 0 and dr > 0 and kr1 < r0
    if side == "bottom":
        return dc == 0 and dr < 0 and kr0 > r1
    if side == "left":
        return dr == 0 and dc > 0 and kc1 < c0
    return dr == 0 and dc < 0 and kc0 > c1


def _solve_matching(locks: list[dict], keys: list[dict]) -> dict[int, int]:
    edges = {li: [ki for ki, key in enumerate(keys) if _match_possible(lock, key)] for li, lock in enumerate(locks)}
    assert all(edges.values()), "every lock must have at least one compatible key"

    order = sorted(range(len(locks)), key=lambda li: len(edges[li]))
    assignment: dict[int, int] = {}
    used: set[int] = set()

    def dfs(i: int) -> bool:
        if i == len(order):
            return True
        li = order[i]
        for ki in edges[li]:
            if ki in used:
                continue
            used.add(ki)
            assignment[li] = ki
            if dfs(i + 1):
                return True
            used.remove(ki)
            del assignment[li]
        return False

    assert dfs(0), "expected a complete key-lock matching"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """Match each small key shape to the uniquely compatible 5x5 lock cavity and slide it in to repair the block.

    Input structure:
    - The background is black.
    - Each lock is a monochrome 5x5 block whose color is unique among locks, with one connected black cavity of 2-4 cells cut out from exactly one side.
    - Each key is a separate connected 2-4 cell object in another color, positioned outside its lock on the cavity-opening side.

    Rule:
    - Detect every lock by its 5x5 bounding box and extract the black cavity cells inside it.
    - Detect every key as a small connected component, then pair keys to locks by exact shape equality under translation while respecting the lock's opening direction.
    - Clear each key from its starting location and fill its matched cavity with the lock color, producing a solid 5x5 square for every lock.

    Key insight:
    - Colors are distractors: the real correspondence is between the cavity shape inside a lock and the separate key shape that can reach it by a straight translation along the opening axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g, bg=BG)
    assert components, "expected at least one foreground object"

    locks: list[dict] = []
    keys: list[dict] = []
    for comp in components:
        lock = _lock_from_component(comp, g)
        if lock is not None:
            locks.append(lock)
            continue
        key = _key_from_component(comp)
        assert key is not None, "foreground objects must be either locks or keys"
        keys.append(key)

    assert locks and len(locks) == len(keys), "expected one key for each lock"

    matching = _solve_matching(locks, keys)
    out = g.copy()

    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = BG

    for lock in locks:
        r0, c0, r1, c1 = lock["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(lock["color"])

    return out


def _enumerate_cavities() -> dict[str, dict[int, list[dict]]]:
    by_side = {side: {size: [] for size in KEY_AREAS} for side in SIDES}
    all_cells = [(r, c) for r in range(LOCK_SIZE) for c in range(LOCK_SIZE)]
    for size in sorted(KEY_AREAS):
        for cells in itertools.combinations(all_cells, size):
            mask = np.zeros((LOCK_SIZE, LOCK_SIZE), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            if not _mask_connected(mask):
                continue
            if not _mask_connected(~mask):
                continue

            touched = []
            if any(r == 0 for r, _ in cells):
                touched.append("top")
            if any(r == LOCK_SIZE - 1 for r, _ in cells):
                touched.append("bottom")
            if any(c == 0 for _, c in cells):
                touched.append("left")
            if any(c == LOCK_SIZE - 1 for _, c in cells):
                touched.append("right")
            if len(touched) != 1:
                continue

            side = touched[0]
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            by_side[side][size].append(
                {
                    "side": side,
                    "size": size,
                    "cells": tuple(sorted((int(r), int(c)) for r, c in cells)),
                    "signature": _shape_signature(cells),
                    "min_r": min(rows),
                    "max_r": max(rows),
                    "min_c": min(cols),
                    "max_c": max(cols),
                }
            )
    return by_side


CAVITIES = _enumerate_cavities()


def _key_relative_cells(cavity: dict, gap: int) -> tuple[tuple[int, int], ...]:
    cells = cavity["cells"]
    side = cavity["side"]
    if side == "top":
        dr = -(cavity["max_r"] + 1 + gap)
        dc = 0
    elif side == "bottom":
        dr = LOCK_SIZE - cavity["min_r"] + gap
        dc = 0
    elif side == "left":
        dr = 0
        dc = -(cavity["max_c"] + 1 + gap)
    else:
        dr = 0
        dc = LOCK_SIZE - cavity["min_c"] + gap
    return tuple(sorted((r + dr, c + dc) for r, c in cells))


def _pair_geometry(cavity: dict, gap: int) -> dict:
    key_cells = _key_relative_cells(cavity, gap)
    rel_rows = [0, LOCK_SIZE - 1] + [r for r, _ in key_cells]
    rel_cols = [0, LOCK_SIZE - 1] + [c for _, c in key_cells]
    return {
        "cavity": cavity,
        "gap": gap,
        "key_rel": key_cells,
        "bbox_rel": (min(rel_rows), min(rel_cols), max(rel_rows), max(rel_cols)),
    }


def _sample_pair_specs(pair_count: int) -> list[dict] | None:
    used_signatures: set[tuple[tuple[int, int], ...]] = set()
    specs: list[dict] = []
    for _ in range(pair_count):
        side_order = random.sample(list(SIDES), len(SIDES))
        size_order = random.sample(sorted(KEY_AREAS), len(KEY_AREAS))
        choices: list[dict] = []
        for side in side_order:
            for size in size_order:
                candidates = [c for c in CAVITIES[side][size] if c["signature"] not in used_signatures]
                random.shuffle(candidates)
                for cavity in candidates:
                    gap = random.randint(*GAP_RANGE)
                    choices.append(_pair_geometry(cavity, gap))
        if not choices:
            return None
        spec = random.choice(choices)
        used_signatures.add(spec["cavity"]["signature"])
        specs.append(spec)
    return specs


def _candidate_lock_positions(shape: tuple[int, int], occupied: np.ndarray, spec: dict) -> list[tuple[int, int]]:
    h, w = shape
    min_rr, min_cc, max_rr, max_cc = spec["bbox_rel"]
    row_lo = -min_rr
    row_hi = h - 1 - max_rr
    col_lo = -min_cc
    col_hi = w - 1 - max_cc
    positions: list[tuple[int, int]] = []
    for r0 in range(row_lo, row_hi + 1):
        for c0 in range(col_lo, col_hi + 1):
            pr0 = r0 + min_rr
            pc0 = c0 + min_cc
            pr1 = r0 + max_rr
            pc1 = c0 + max_cc
            er0 = max(0, pr0 - PLACEMENT_MARGIN)
            ec0 = max(0, pc0 - PLACEMENT_MARGIN)
            er1 = min(h - 1, pr1 + PLACEMENT_MARGIN)
            ec1 = min(w - 1, pc1 + PLACEMENT_MARGIN)
            if not occupied[er0 : er1 + 1, ec0 : ec1 + 1].any():
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _place_pairs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    ordered = sorted(specs, key=lambda spec: (spec["bbox_rel"][2] - spec["bbox_rel"][0], spec["bbox_rel"][3] - spec["bbox_rel"][1]), reverse=True)
    for spec in ordered:
        positions = _candidate_lock_positions(shape, occupied, spec)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        min_rr, min_cc, max_rr, max_cc = spec["bbox_rel"]
        pr0 = r0 + min_rr
        pc0 = c0 + min_cc
        pr1 = r0 + max_rr
        pc1 = c0 + max_cc
        er0 = max(0, pr0 - PLACEMENT_MARGIN)
        ec0 = max(0, pc0 - PLACEMENT_MARGIN)
        er1 = min(h - 1, pr1 + PLACEMENT_MARGIN)
        ec1 = min(w - 1, pc1 + PLACEMENT_MARGIN)
        occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True

        cavity_abs = tuple(sorted((r0 + rr, c0 + cc) for rr, cc in spec["cavity"]["cells"]))
        key_abs = tuple(sorted((r0 + rr, c0 + cc) for rr, cc in spec["key_rel"]))
        placed.append(
            {
                "lock_r0": r0,
                "lock_c0": c0,
                "cavity_abs": cavity_abs,
                "key_abs": key_abs,
                "side": spec["cavity"]["side"],
                "signature": spec["cavity"]["signature"],
            }
        )
    return placed


def _build_grids(shape: tuple[int, int], placed: list[dict], lock_colors: list[int], key_colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    for pair, lock_color, key_color in zip(placed, lock_colors, key_colors):
        r0 = pair["lock_r0"]
        c0 = pair["lock_c0"]
        inp[r0 : r0 + LOCK_SIZE, c0 : c0 + LOCK_SIZE] = int(lock_color)
        out[r0 : r0 + LOCK_SIZE, c0 : c0 + LOCK_SIZE] = int(lock_color)
        for r, c in pair["cavity_abs"]:
            inp[r, c] = BG
        for r, c in pair["key_abs"]:
            inp[r, c] = int(key_color)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    specs = _sample_pair_specs(pair_count)
    if specs is None:
        return None

    min_side = MIN_GRID_BY_PAIR_COUNT[pair_count]
    h = random.randint(min_side, GRID_LIMIT)
    w = random.randint(min_side, GRID_LIMIT)
    placed = _place_pairs((h, w), specs)
    if placed is None:
        return None

    colors = random.sample(list(range(1, 10)), 2 * pair_count)
    lock_colors = colors[:pair_count]
    key_colors = colors[pair_count:]
    inp, expected = _build_grids((h, w), placed, lock_colors, key_colors)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-key ARC input")
