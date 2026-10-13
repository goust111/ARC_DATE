from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GRAY = 5
ORANGE = 7

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        pts = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in DIRS.values():
                rr, cc = r + dr, c + dc
                if _in_bounds(mask.shape, rr, cc) and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _notch_for_component(g: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    cells = {(int(r), int(c)) for r, c in coords}
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if r1 - r0 >= 2:
        for c in range(c0 + 1, c1):
            if g[r0, c] == BG and (r0, c - 1) in cells and (r0, c + 1) in cells:
                candidates.append(((r0, c), DIRS["down"]))
            if g[r1, c] == BG and (r1, c - 1) in cells and (r1, c + 1) in cells:
                candidates.append(((r1, c), DIRS["up"]))

    if c1 - c0 >= 2:
        for r in range(r0 + 1, r1):
            if g[r, c0] == BG and (r - 1, c0) in cells and (r + 1, c0) in cells:
                candidates.append(((r, c0), DIRS["right"]))
            if g[r, c1] == BG and (r - 1, c1) in cells and (r + 1, c1) in cells:
                candidates.append(((r, c1), DIRS["left"]))

    assert len(candidates) == 1, "each object must have exactly one flanked single-cell notch"
    notch, direction = candidates[0]
    return {
        "coords": coords,
        "cells": cells,
        "bbox": (r0, c0, r1, c1),
        "notch": notch,
        "dir": direction,
        "color": int(g[coords[0, 0], coords[0, 1]]),
    }


def _exit_after_own_body(obj: dict) -> tuple[int, int]:
    r, c = obj["notch"]
    dr, dc = obj["dir"]
    r0, c0, r1, c1 = obj["bbox"]
    if dr == 1:
        return r1 + 1, c
    if dr == -1:
        return r0 - 1, c
    if dc == 1:
        return r, c1 + 1
    return r, c0 - 1


def _parse_objects(g: np.ndarray) -> list[dict]:
    assert g.ndim == 2
    assert not np.any(g == ORANGE), "orange is reserved for output rays"
    fg = (g == RED) | (g == GRAY)
    comps = _components(fg)
    assert comps, "expected at least one notched object"

    objects = [_notch_for_component(g, comp) for comp in comps]
    red_objects = [i for i, obj in enumerate(objects) if obj["color"] == RED]
    assert len(red_objects) == 1, "expected exactly one initial red fire object"
    assert all(obj["color"] in (RED, GRAY) for obj in objects)

    notches = [obj["notch"] for obj in objects]
    assert len(set(notches)) == len(notches), "notch cells must be unique"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite the chain of notched objects, keep its orange rays, and erase every unignited object.

    Input structure:
    - A black background contains several separated red or gray foreground objects.
    - Each object has exactly one flanked single-cell boundary notch; the notch normal gives its firing direction.
    - Exactly one object is red at the start, and all other objects are gray.

    Rule:
    - A lit object fires an orange ray from its notch along the notch direction through background cells.
    - If the ray's first contact with another object is that object's notch, and that notch direction matches the ray direction, the whole object becomes red and immediately fires onward.
    - If the ray first hits body pixels, reaches a mismatched notch, or leaves the grid, propagation stops.
    - After propagation, every gray object that was not ignited is erased; all traveled ray cells remain orange.

    Key insight:
    - The missing boundary cell is both the lockhole and the muzzle: touching object body is not enough to ignite it, and only the objects reachable through matching notches survive.
    """
    g = np.array(grid, dtype=int)
    objects = _parse_objects(g)
    start = next(i for i, obj in enumerate(objects) if obj["color"] == RED)

    notch_owner = {obj["notch"]: i for i, obj in enumerate(objects)}
    lit: set[int] = {start}
    ray_cells: list[tuple[int, int]] = []
    current = start

    while True:
        obj = objects[current]
        dr, dc = obj["dir"]
        ray_cells.append(obj["notch"])
        r, c = _exit_after_own_body(obj)
        next_obj = None

        while _in_bounds(g.shape, r, c):
            pos = (int(r), int(c))
            if pos in notch_owner:
                owner = notch_owner[pos]
                ray_cells.append(pos)
                if owner not in lit and objects[owner]["dir"] == (dr, dc):
                    next_obj = owner
                break
            if int(g[r, c]) != BG:
                break
            ray_cells.append(pos)
            r += dr
            c += dc

        if next_obj is None:
            break
        lit.add(next_obj)
        current = next_obj

    out = np.full(g.shape, BG, dtype=int)
    for i in lit:
        coords = objects[i]["coords"]
        out[coords[:, 0], coords[:, 1]] = RED
    for r, c in ray_cells:
        if _in_bounds(out.shape, r, c) and out[r, c] == BG:
            out[r, c] = ORANGE
    return out


# --------------- generator ---------------

MIN_SIZE = 18
MAX_SIZE = 30
GEN_ATTEMPTS = 400


def _make_mask(h: int, w: int, direction: tuple[int, int], notch_offset: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    dr, dc = direction
    if (dr, dc) == DIRS["right"]:
        mask[notch_offset, 0] = False
    elif (dr, dc) == DIRS["left"]:
        mask[notch_offset, w - 1] = False
    elif (dr, dc) == DIRS["down"]:
        mask[0, notch_offset] = False
    else:
        mask[h - 1, notch_offset] = False
    return mask


def _stamp(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    assert patch.shape == mask.shape
    assert np.all(patch[mask] == BG)
    patch[mask] = color


def _padded_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rows, cols = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rows, r0 + h + pad)
    cc1 = min(cols, c0 + w + pad)
    return bool(not occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _random_object_for_right(line_r: int) -> dict | None:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    notch = random.randint(1, h - 2)
    r0 = line_r - notch
    if r0 < 1:
        return None
    return {"h": h, "w": w, "notch": notch, "r0": r0, "mask": _make_mask(h, w, DIRS["right"], notch)}


def _random_distractor_mask() -> tuple[np.ndarray, tuple[int, int]]:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    direction = random.choice(list(DIRS.values()))
    offset_limit = w if direction[0] else h
    offset = random.randint(1, offset_limit - 2)
    return _make_mask(h, w, direction, offset), direction


def _place_distractors(grid: np.ndarray, occupied: np.ndarray, line_r: int, count: int) -> bool:
    rows, cols = grid.shape
    for _ in range(count):
        placed = False
        for _try in range(120):
            mask, _direction = _random_distractor_mask()
            h, w = mask.shape
            r0 = random.randint(1, rows - h - 1)
            c0 = random.randint(1, cols - w - 1)
            if r0 <= line_r <= r0 + h - 1:
                continue
            if not _padded_free(occupied, r0, c0, h, w):
                continue
            _stamp(grid, r0, c0, mask, GRAY)
            _mark_occupied(occupied, r0, c0, h, w)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_right_facing_scene() -> np.ndarray | None:
    rows = random.randint(MIN_SIZE, MAX_SIZE)
    cols = random.randint(MIN_SIZE, MAX_SIZE)
    chain_len = random.randint(2, 4)
    line_r = random.randint(4, rows - 5)

    chain = []
    for _ in range(chain_len):
        obj = _random_object_for_right(line_r)
        if obj is None or obj["r0"] + obj["h"] >= rows - 1:
            return None
        chain.append(obj)

    final_body_stop = random.random() < 0.7
    stopper = None
    if final_body_stop:
        for _ in range(40):
            h = random.randint(3, 5)
            w = random.randint(3, 5)
            notch = random.randint(1, h - 2)
            body_rows = [x for x in range(h) if x != notch]
            r0 = line_r - random.choice(body_rows)
            if 1 <= r0 and r0 + h < rows - 1:
                stopper = {"h": h, "w": w, "notch": notch, "r0": r0, "mask": _make_mask(h, w, DIRS["right"], notch)}
                break
        if stopper is None:
            return None

    gaps = [random.randint(2, 5) for _ in range(chain_len - 1)]
    total_w = sum(obj["w"] for obj in chain) + sum(gaps)
    if final_body_stop:
        stop_gap = random.randint(2, 5)
        total_w += stop_gap + stopper["w"]
    else:
        stop_gap = 0

    if total_w > cols - 4:
        return None
    c = random.randint(1, cols - total_w - 2)
    grid = np.full((rows, cols), BG, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    for i, obj in enumerate(chain):
        obj["c0"] = c
        _stamp(grid, obj["r0"], c, obj["mask"], RED if i == 0 else GRAY)
        _mark_occupied(occupied, obj["r0"], c, obj["h"], obj["w"])
        if i < len(gaps):
            c += obj["w"] + gaps[i]

    if final_body_stop:
        c += chain[-1]["w"] + stop_gap
        if not _padded_free(occupied, stopper["r0"], c, stopper["h"], stopper["w"]):
            return None
        _stamp(grid, stopper["r0"], c, stopper["mask"], GRAY)
        _mark_occupied(occupied, stopper["r0"], c, stopper["h"], stopper["w"])

    if not _place_distractors(grid, occupied, line_r, random.randint(1, 3)):
        return None

    return grid


def _rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k=k).copy()


def _count_red_components(grid: np.ndarray) -> int:
    return len(_components(grid == RED))


def _semantic_accept(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
        objs = _parse_objects(grid)
    except Exception:
        return False

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if not np.any(out == ORANGE):
        return False
    if _count_red_components(out) < 2:
        return False
    if len(objs) <= _count_red_components(out):
        return False
    if np.array_equal(grid, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_right_facing_scene()
        if grid is None:
            continue
        grid = _rotate_grid(grid, random.randint(0, 3))
        if _semantic_accept(grid):
            return grid
    raise RuntimeError("failed to generate a valid lockhole chain ignition input")
