from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
STAMP_COLORS = tuple(c for c in range(10) if c not in (BG, GREY))
MAX_RETRIES = 400

BASE_SIZE = 5
BASE_TIP_RIGHT = (2, 4)
BASE_ARROW_RIGHT = {
    (1, 4),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 4),
}
BASE_SLOT_SHAPES = {
    "top": {
        1: {(0, 4)},
        2: {(0, 3), (0, 4)},
        3: {(0, 2), (0, 3), (0, 4)},
    },
    "mid": {
        1: {(2, 1)},
        2: {(2, 0), (2, 1)},
        3: {(1, 0), (2, 0), (2, 1)},
    },
    "bottom": {
        1: {(4, 4)},
        2: {(4, 3), (4, 4)},
        3: {(4, 2), (4, 3), (4, 4)},
    },
}
ROTATIONS = {"right": 0, "down": 1, "left": 2, "up": 3}
DIRS = tuple(ROTATIONS)
DELTAS = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
ARROW_TEMPLATES = {
    "right": {(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)},
    "down": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    "left": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)},
    "up": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
}
ARROW_TIPS = {
    "right": (1, 2),
    "down": (2, 1),
    "left": (1, 0),
    "up": (0, 1),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _rotate_coord(coord: tuple[int, int], turns: int, size: int = BASE_SIZE) -> tuple[int, int]:
    r, c = coord
    for _ in range(turns % 4):
        r, c = c, size - 1 - r
    return r, c


def _rotate_coords(coords: Iterable[tuple[int, int]], turns: int, size: int = BASE_SIZE) -> set[tuple[int, int]]:
    return {_rotate_coord(coord, turns, size=size) for coord in coords}


def _normalize_sets(
    arrow: set[tuple[int, int]],
    body_components: list[set[tuple[int, int]]],
    tip: tuple[int, int],
) -> dict:
    all_cells = set(arrow)
    for comp in body_components:
        all_cells |= set(comp)
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)

    def shift(coords: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r - min_r, c - min_c) for r, c in coords}

    arrow_n = shift(arrow)
    body_n = [shift(comp) for comp in body_components]
    cells_n = set(arrow_n)
    for comp in body_n:
        cells_n |= set(comp)
    max_r = max(r for r, _ in cells_n)
    max_c = max(c for _, c in cells_n)
    tip_n = (tip[0] - min_r, tip[1] - min_c)
    return {
        "arrow": arrow_n,
        "body_components": body_n,
        "cells": cells_n,
        "tip": tip_n,
        "shape": (max_r + 1, max_c + 1),
    }


def _build_stamp(direction: str, slots: list[str], sizes: list[int]) -> dict:
    turns = ROTATIONS[direction]
    arrow = _rotate_coords(BASE_ARROW_RIGHT, turns)
    body_components = [_rotate_coords(BASE_SLOT_SHAPES[slot][size], turns) for slot, size in zip(slots, sizes)]
    tip = _rotate_coord(BASE_TIP_RIGHT, turns)
    return _normalize_sets(arrow, body_components, tip)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _components_from_mask(grid == color)


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = list(coords)
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _background_color(grid: np.ndarray) -> int:
    vals = Counter(int(v) for v in grid.ravel())
    return vals.most_common(1)[0][0]


def _direction_from_target(
    bbox: tuple[int, int, int, int], target: tuple[int, int]
) -> str | None:
    r0, c0, r1, c1 = bbox
    tr, tc = target
    if c0 <= tc <= c1:
        if tr < r0:
            return "up"
        if tr > r1:
            return "down"
    if r0 <= tr <= r1:
        if tc < c0:
            return "left"
        if tc > c1:
            return "right"
    return None


def _tip_local(
    direction: str, bbox: tuple[int, int, int, int], target: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    tr, tc = target
    if direction == "right":
        return tr - r0, c1 - c0
    if direction == "left":
        return tr - r0, 0
    if direction == "down":
        return r1 - r0, tc - c0
    return 0, tc - c0


def _arrow_cells_from_tip(direction: str, tip_local: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = shape
    tip_r, tip_c = tip_local
    tmpl_tip = ARROW_TIPS[direction]
    anchor = (tip_r - tmpl_tip[0], tip_c - tmpl_tip[1])
    cells = {(anchor[0] + r, anchor[1] + c) for r, c in ARROW_TEMPLATES[direction]}
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cells):
        return None
    return cells


def _sort_components(components: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    def key(comp: list[tuple[int, int]]) -> tuple:
        return (len(comp), _bbox(comp), tuple(comp))

    return sorted(components, key=key)


def _decode_stamp(
    grid: np.ndarray, color: int, stamp_coords: list[tuple[int, int]], target: tuple[int, int], bg: int
) -> dict | None:
    bbox = _bbox(stamp_coords)
    direction = _direction_from_target(bbox, target)
    if direction is None:
        return None

    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local_mask = np.zeros((h, w), dtype=bool)
    for r, c in stamp_coords:
        local_mask[r - r0, c - c0] = True

    tip = _tip_local(direction, bbox, target)
    arrow = _arrow_cells_from_tip(direction, tip, local_mask.shape)
    if arrow is None:
        return None
    if not all(local_mask[r, c] for r, c in arrow):
        return None

    body_mask = local_mask.copy()
    for r, c in arrow:
        body_mask[r, c] = False
    if not body_mask.any():
        return None

    body_components_local = _components_from_mask(body_mask)
    if not body_components_local:
        return None

    arrow_set = set(arrow)
    for comp in body_components_local:
        if not any((nr, nc) in arrow_set for r, c in comp for nr, nc in _neighbors4(r, c)):
            return None

    steps = {
        "right": target[1] - c1 - 1,
        "left": c0 - target[1] - 1,
        "down": target[0] - r1 - 1,
        "up": r0 - target[0] - 1,
    }[direction]
    if steps < 1:
        return None

    tip_global = (r0 + tip[0], c0 + tip[1])
    dr, dc = DELTAS[direction]
    for t in range(1, steps + 1):
        rr = tip_global[0] + dr * t
        cc = tip_global[1] + dc * t
        if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            return None
        if (rr, cc) != target and int(grid[rr, cc]) != bg:
            return None

    body_components = [
        [(r0 + r, c0 + c) for r, c in comp]
        for comp in _sort_components(body_components_local)
    ]
    return {
        "color": color,
        "direction": direction,
        "steps": steps,
        "stamp_coords": sorted(stamp_coords),
        "target": target,
        "body_components": body_components,
        "tip": tip_global,
        "bbox": bbox,
    }


def _extract_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    specs: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (bg, GREY)):
        comps = _components_of_color(g, color)
        big = [comp for comp in comps if len(comp) > 1]
        singles = [comp[0] for comp in comps if len(comp) == 1]
        if len(big) != 1 or len(singles) != 1:
            continue
        spec = _decode_stamp(g, color, big[0], singles[0], bg)
        if spec is not None:
            specs.append(spec)
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each monochrome stamp along the direction encoded by its internal arrow.

    The same-color singleton target fixes the direction line and the projection length.
    Each shifted copy contains only the non-arrow part of the stamp; progressively
    larger prefixes of its body-components-by-area are recolored to grey.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    specs = _extract_specs(g)

    for spec in specs:
        dr, dc = DELTAS[spec["direction"]]
        body_components = spec["body_components"]
        for step in range(1, spec["steps"] + 1):
            grey_count = min(step, len(body_components))
            for idx, comp in enumerate(body_components):
                paint = GREY if idx < grey_count else spec["color"]
                for r, c in comp:
                    rr = r + dr * step
                    cc = c + dc * step
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr, cc] = paint

    # The original stamps and their targets stay visible even if a projection crosses them.
    out[g != _background_color(g)] = g[g != _background_color(g)]
    return out


def _make_object_spec(rng: np.random.Generator, direction: str, color: int) -> dict:
    slot_names = list(BASE_SLOT_SHAPES)
    component_count = int(rng.integers(2, 4))
    slots = list(rng.choice(slot_names, size=component_count, replace=False))
    sizes = list(rng.permutation([1, 2, 3])[:component_count])
    stamp = _build_stamp(direction, slots, sizes)
    steps = int(rng.integers(1, 7))
    dr, dc = DELTAS[direction]
    tip = stamp["tip"]
    target = (tip[0] + dr * (steps + 1), tip[1] + dc * (steps + 1))

    corridor = {tip}
    for t in range(1, steps + 1):
        corridor.add((tip[0] + dr * t, tip[1] + dc * t))
    corridor.add(target)

    projected = set()
    for step in range(1, steps + 1):
        for comp in stamp["body_components"]:
            for r, c in comp:
                projected.add((r + dr * step, c + dc * step))

    h, w = stamp["shape"]
    bbox_rect = {(r, c) for r in range(h) for c in range(w)}
    reserve = bbox_rect | corridor | projected
    return {
        "color": color,
        "direction": direction,
        "steps": steps,
        "stamp": stamp,
        "target": target,
        "projected": projected,
        "reserve": reserve,
    }


def _halo(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((r + dr, c + dc))
    return expanded


def _translate(cells: Iterable[tuple[int, int]], origin: tuple[int, int]) -> set[tuple[int, int]]:
    orow, ocol = origin
    return {(orow + r, ocol + c) for r, c in cells}


def _place_objects(rng: np.random.Generator, objects: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    placed: list[dict] = []
    for obj in sorted(objects, key=lambda x: (-len(x["reserve"]), -x["steps"])):
        reserve = obj["reserve"]
        min_r = min(r for r, _ in reserve)
        max_r = max(r for r, _ in reserve)
        min_c = min(c for _, c in reserve)
        max_c = max(c for _, c in reserve)
        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        origin = None
        for _ in range(120):
            cand = (
                int(rng.integers(row_lo, row_hi + 1)),
                int(rng.integers(col_lo, col_hi + 1)),
            )
            reserve_global = _translate(reserve, cand)
            if reserve_global & blocked:
                continue
            origin = cand
            blocked |= _halo(reserve_global)
            break
        if origin is None:
            return None
        placed.append({**obj, "origin": origin})
    return placed


def _render_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in placed:
        orow, ocol = obj["origin"]
        color = obj["color"]
        for r, c in obj["stamp"]["cells"]:
            grid[orow + r, ocol + c] = color
        tr, tc = obj["target"]
        grid[orow + tr, ocol + tc] = color
    return grid


def _validate_generated(grid: np.ndarray, placed: list[dict]) -> bool:
    specs = _extract_specs(grid)
    if len(specs) != len(placed):
        return False

    placed_by_color = {
        obj["color"]: {
            "direction": obj["direction"],
            "steps": obj["steps"],
            "areas": sorted(len(comp) for comp in obj["stamp"]["body_components"]),
        }
        for obj in placed
    }
    for spec in specs:
        want = placed_by_color.get(spec["color"])
        if want is None:
            return False
        if spec["direction"] != want["direction"] or spec["steps"] != want["steps"]:
            return False
        if sorted(len(comp) for comp in spec["body_components"]) != want["areas"]:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.any(out == GREY):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    object_count = int(rng.integers(1, 5))
    colors = list(rng.choice(STAMP_COLORS, size=object_count, replace=False))
    objects = []
    for color in colors:
        direction = DIRS[int(rng.integers(0, len(DIRS)))]
        objects.append(_make_object_spec(rng, direction, int(color)))

    placed = _place_objects(rng, objects, (h, w))
    if placed is None:
        return None
    grid = _render_input((h, w), placed)
    if not _validate_generated(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
