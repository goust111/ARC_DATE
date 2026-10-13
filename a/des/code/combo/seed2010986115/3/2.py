from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == BG or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIR_VECS.values():
                    nr = r + dr
                    nc = c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                }
            )

    return comps


def _coords_to_mask(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _notch_index(line: np.ndarray) -> int | None:
    if line.size < 3:
        return None
    gaps = np.flatnonzero(~line)
    if gaps.size != 1:
        return None
    idx = int(gaps[0])
    if idx == 0 or idx == line.size - 1:
        return None
    if bool(line[idx - 1]) and bool(line[idx + 1]):
        return idx
    return None


def _detect_notch_from_mask(mask: np.ndarray) -> tuple[str, tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[str, tuple[int, int]]] = []

    idx = _notch_index(mask[0, :])
    if idx is not None:
        candidates.append(("up", (0, idx)))

    idx = _notch_index(mask[h - 1, :])
    if idx is not None:
        candidates.append(("down", (h - 1, idx)))

    idx = _notch_index(mask[:, 0])
    if idx is not None:
        candidates.append(("left", (idx, 0)))

    idx = _notch_index(mask[:, w - 1])
    if idx is not None:
        candidates.append(("right", (idx, w - 1)))

    assert len(candidates) == 1, "expected exactly one boundary notch direction"
    return candidates[0]


def _detect_component_notch(comp: dict[str, object]) -> tuple[str, tuple[int, int]]:
    coords = comp["coords"]
    bbox = comp["bbox"]
    assert isinstance(coords, np.ndarray)
    assert isinstance(bbox, tuple)
    direction, (lr, lc) = _detect_notch_from_mask(_coords_to_mask(coords, bbox))
    r0, c0, _r1, _c1 = bbox
    return direction, (r0 + lr, c0 + lc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored object from the direction encoded by its own notch.

    The notch is the unique boundary-side gap on the object's bounding box.
    For an object of color c, the ray length equals the number of independent
    4-connected objects of color c in the whole grid. The ray starts at the
    notch cell itself, keeps the original object intact, and stops early if the
    next cell is already non-background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_components(g)
    if not comps:
        return g.copy()

    color_counts = Counter(int(comp["color"]) for comp in comps)
    out = g.copy()
    ordered = sorted(comps, key=lambda comp: tuple(int(x) for x in comp["bbox"]))

    for comp in ordered:
        color = int(comp["color"])
        length = int(color_counts[color])
        direction, (r, c) = _detect_component_notch(comp)
        dr, dc = DIR_VECS[direction]

        for _ in range(length):
            if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
                break
            if int(out[r, c]) != BG:
                break
            out[r, c] = color
            r += dr
            c += dc

    return out


_BASE_TEMPLATES = [
    ("101", "111"),
    ("101", "111", "010"),
    ("101", "111", "100"),
    ("101", "111", "001"),
    ("1011", "1110"),
    ("1101", "0111"),
]

_GROUP_COUNT_OPTIONS = [
    (2,),
    (3,),
    (4,),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 3),
    (2, 2, 2),
    (2, 2, 3),
    (3, 4),
]

_GRID_MIN = 10
_GRID_MAX = 19
_PLACE_TRIES = 160
_GENERATE_ATTEMPTS = 500
_BLOCKED_PROB = 0.35


def _template_to_mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _build_shape_variants() -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()

    for rows in _BASE_TEMPLATES:
        mask = _template_to_mask(rows)
        for _ in range(4):
            key = tuple(tuple(int(x) for x in row) for row in mask.astype(int))
            if key not in seen:
                direction, notch = _detect_notch_from_mask(mask)
                variants.append(
                    {
                        "mask": mask.copy(),
                        "direction": direction,
                        "notch": notch,
                    }
                )
                seen.add(key)
            mask = np.rot90(mask)

    return variants


_SHAPE_VARIANTS = _build_shape_variants()


def _variants_for_direction(direction: str) -> list[dict[str, object]]:
    return [variant for variant in _SHAPE_VARIANTS if variant["direction"] == direction]


def _rectangles_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _position_ranges(
    grid_shape: tuple[int, int],
    variant: dict[str, object],
    ray_len: int,
) -> tuple[int, int, int, int] | None:
    h, w = grid_shape
    mask = variant["mask"]
    assert isinstance(mask, np.ndarray)
    mh, mw = mask.shape
    nr, nc = variant["notch"]
    direction = str(variant["direction"])
    dr, dc = DIR_VECS[direction]

    r_lo = 0
    r_hi = h - mh
    c_lo = 0
    c_hi = w - mw

    if dr < 0:
        r_lo = max(r_lo, ray_len - 1 - int(nr))
    elif dr > 0:
        r_hi = min(r_hi, h - ray_len - int(nr))

    if dc < 0:
        c_lo = max(c_lo, ray_len - 1 - int(nc))
    elif dc > 0:
        c_hi = min(c_hi, w - ray_len - int(nc))

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return r_lo, r_hi, c_lo, c_hi


def _materialize_object(
    color: int,
    ray_len: int,
    variant: dict[str, object],
    r0: int,
    c0: int,
) -> dict[str, object]:
    mask = variant["mask"]
    assert isinstance(mask, np.ndarray)
    coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
    cells = {tuple(map(int, coord)) for coord in coords.tolist()}
    notch_r, notch_c = variant["notch"]
    return {
        "color": int(color),
        "ray_len": int(ray_len),
        "direction": str(variant["direction"]),
        "bbox": (int(r0), int(c0), int(r0 + mask.shape[0] - 1), int(c0 + mask.shape[1] - 1)),
        "cells": cells,
        "notch": (int(r0 + notch_r), int(c0 + notch_c)),
    }


def _occupied_cells(objects: list[dict[str, object]]) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for obj in objects:
        occupied.update(obj["cells"])
    return occupied


def _can_place_object(
    obj: dict[str, object],
    placed: list[dict[str, object]],
    occupied: set[tuple[int, int]],
) -> bool:
    for other in placed:
        if _rectangles_intersect(obj["bbox"], other["bbox"]):
            return False

    for r, c in obj["cells"]:
        if (r, c) in occupied:
            return False
        for dr, dc in DIR_VECS.values():
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _place_random_object(
    color: int,
    ray_len: int,
    grid_shape: tuple[int, int],
    placed: list[dict[str, object]],
) -> dict[str, object] | None:
    occupied = _occupied_cells(placed)
    variants = _SHAPE_VARIANTS[:]
    random.shuffle(variants)

    for _ in range(_PLACE_TRIES):
        variant = random.choice(variants)
        ranges = _position_ranges(grid_shape, variant, ray_len)
        if ranges is None:
            continue
        r_lo, r_hi, c_lo, c_hi = ranges
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        obj = _materialize_object(color, ray_len, variant, r0, c0)
        if _can_place_object(obj, placed, occupied):
            return obj

    return None


def _ray_prefix_clear(
    obj: dict[str, object],
    occupied: set[tuple[int, int]],
    blocker_step: int,
) -> bool:
    r, c = obj["notch"]
    dr, dc = DIR_VECS[obj["direction"]]
    for step in range(blocker_step):
        cell = (r + dr * step, c + dc * step)
        if cell in occupied:
            return False
    return True


def _position_is_valid(
    grid_shape: tuple[int, int],
    variant: dict[str, object],
    ray_len: int,
    r0: int,
    c0: int,
) -> bool:
    ranges = _position_ranges(grid_shape, variant, ray_len)
    if ranges is None:
        return False
    r_lo, r_hi, c_lo, c_hi = ranges
    return r_lo <= r0 <= r_hi and c_lo <= c0 <= c_hi


def _place_blocked_object(
    color: int,
    ray_len: int,
    grid_shape: tuple[int, int],
    placed: list[dict[str, object]],
) -> dict[str, object] | None:
    occupied = _occupied_cells(placed)
    blockers = placed[:]
    random.shuffle(blockers)

    direction_order = list(DIR_VECS)
    random.shuffle(direction_order)
    steps = list(range(1, ray_len))
    random.shuffle(steps)

    for blocker in blockers:
        blocker_cells = list(blocker["cells"])
        random.shuffle(blocker_cells)
        for direction in direction_order:
            variants = _variants_for_direction(direction)
            random.shuffle(variants)
            dr, dc = DIR_VECS[direction]
            for variant in variants:
                for step in steps:
                    for br, bc in blocker_cells:
                        notch_r = br - dr * step
                        notch_c = bc - dc * step
                        r0 = notch_r - int(variant["notch"][0])
                        c0 = notch_c - int(variant["notch"][1])
                        if not _position_is_valid(grid_shape, variant, ray_len, r0, c0):
                            continue
                        obj = _materialize_object(color, ray_len, variant, r0, c0)
                        if not _can_place_object(obj, placed, occupied):
                            continue
                        if not _ray_prefix_clear(obj, occupied, step):
                            continue
                        return obj

    return None


def _render_grid(grid_shape: tuple[int, int], objects: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = int(obj["color"])
    return grid


def _ray_cells_on_input(
    obj: dict[str, object],
    grid: np.ndarray,
) -> list[tuple[int, int]]:
    r, c = obj["notch"]
    dr, dc = DIR_VECS[obj["direction"]]
    cells: list[tuple[int, int]] = []

    for _ in range(int(obj["ray_len"])):
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            break
        if int(grid[r, c]) != BG:
            break
        cells.append((r, c))
        r += dr
        c += dc

    return cells


def _validate_candidate(
    grid: np.ndarray,
    objects: list[dict[str, object]],
    want_blocked: bool,
) -> bool:
    comps = _find_components(grid)
    if len(comps) != len(objects):
        return False

    used: set[tuple[int, int]] = set()
    has_blocked = False
    has_full = False

    for obj in objects:
        cells = _ray_cells_on_input(obj, grid)
        if len(cells) < int(obj["ray_len"]):
            has_blocked = True
        else:
            has_full = True
        for cell in cells:
            if cell in used:
                return False
            used.add(cell)

    if want_blocked and not has_blocked:
        return False
    if not has_full:
        return False

    try:
        out = main(grid)
    except Exception:
        return False

    return bool(np.any(out != grid))


def _choose_grid_shape(total_objects: int) -> tuple[int, int]:
    base = 8 + total_objects
    h_lo = max(_GRID_MIN, base - 1)
    h_hi = min(_GRID_MAX, base + 2)
    w_lo = max(_GRID_MIN, base)
    w_hi = min(_GRID_MAX, base + 3)
    return random.randint(h_lo, h_hi), random.randint(w_lo, w_hi)


def _build_specs(group_counts: tuple[int, ...]) -> list[tuple[int, int]]:
    colors = random.sample(range(1, 10), len(group_counts))
    specs: list[tuple[int, int]] = []
    for color, count in zip(colors, group_counts):
        specs.extend([(int(color), int(count))] * int(count))
    random.shuffle(specs)
    return specs


def _generate_attempt() -> np.ndarray | None:
    group_counts = random.choice(_GROUP_COUNT_OPTIONS)
    total_objects = sum(group_counts)
    grid_shape = _choose_grid_shape(total_objects)
    specs = _build_specs(group_counts)
    want_blocked = total_objects >= 2 and random.random() < _BLOCKED_PROB

    placed: list[dict[str, object]] = []

    if want_blocked:
        blocked_index = random.randrange(len(specs))
        blocked_spec = specs.pop(blocked_index)
        for color, ray_len in specs:
            obj = _place_random_object(color, ray_len, grid_shape, placed)
            if obj is None:
                return None
            placed.append(obj)

        blocked_obj = _place_blocked_object(
            blocked_spec[0],
            blocked_spec[1],
            grid_shape,
            placed,
        )
        if blocked_obj is None:
            return None
        placed.append(blocked_obj)
    else:
        for color, ray_len in specs:
            obj = _place_random_object(color, ray_len, grid_shape, placed)
            if obj is None:
                return None
            placed.append(obj)

    random.shuffle(placed)
    grid = _render_grid(grid_shape, placed)
    if _validate_candidate(grid, placed, want_blocked):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
