import random
from typing import Iterable

import numpy as np


BG = 0
KEY_COLORS = (1, 2, 3, 4, 6, 9)
OBSTACLE = 5
LOCK = 7
TRAIL = 8
GEN_RETRIES = 400
SIDES = ("top", "bottom", "left", "right")
SLIDE_DIRS = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(pts)
    return comps


def bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def normalize_coords(coords: Iterable[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    pts = list(coords)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    norm = tuple(sorted((r - min_r, c - min_c) for r, c in pts))
    return norm, (min_r, min_c)


def shape_dims(coords: Iterable[tuple[int, int]]) -> tuple[int, int]:
    pts = list(coords)
    return max(r for r, _ in pts) + 1, max(c for _, c in pts) + 1


def rotate_normalized(norm: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    pts = list(norm)
    h, w = shape_dims(pts)
    for _ in range(turns % 4):
        pts = [(c, h - 1 - r) for r, c in pts]
        h, w = w, h
        pts, _ = normalize_coords(pts)
        pts = list(pts)
    return tuple(sorted(pts))


def canonical_shape(norm: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(rotate_normalized(norm, turns) for turns in range(4))


def translate(coords: Iterable[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def dilate(cells: Iterable[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def is_connected(coords: Iterable[tuple[int, int]]) -> bool:
    pts = set(coords)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen == pts


def _find_locks(grid: np.ndarray) -> list[dict]:
    locks: list[dict] = []
    for pts in components_mask(grid == LOCK):
        r0, c0, r1, c1 = bbox(pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 5 or width < 5:
            continue

        perim = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        comp = set(pts)
        if not comp <= perim:
            continue
        gap_cells = sorted(perim - comp)
        if len(gap_cells) != 1:
            continue

        gap_r, gap_c = gap_cells[0]
        if gap_r == r0:
            side = "top"
        elif gap_r == r1:
            side = "bottom"
        elif gap_c == c0:
            side = "left"
        else:
            side = "right"

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        obstacle_comps = components_mask(inner == OBSTACLE)
        assert len(obstacle_comps) == 1, "each lock must contain one obstacle"
        obstacle_rel = obstacle_comps[0]
        or0, oc0, or1, oc1 = bbox(obstacle_rel)
        assert len(obstacle_rel) == (or1 - or0 + 1) * (oc1 - oc0 + 1), "obstacle must be rectangular"

        usable_abs = [
            (r0 + 1 + rr, c0 + 1 + cc)
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if inner[rr, cc] != OBSTACLE
        ]
        usable_norm, anchor = normalize_coords(usable_abs)
        locks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "side": side,
                "dir": SLIDE_DIRS[side],
                "usable_norm": usable_norm,
                "anchor": anchor,
                "usable_abs": set(usable_abs),
            }
        )
    assert locks, "no valid locks found"
    return locks


def _find_keys(grid: np.ndarray) -> list[dict]:
    key_mask = ~np.isin(grid, np.array([BG, OBSTACLE, LOCK]))
    keys: list[dict] = []
    for pts in components_mask(key_mask):
        colors = {int(grid[r, c]) for r, c in pts}
        assert len(colors) == 1, "each key must be monochrome"
        color = next(iter(colors))
        assert color != TRAIL, "input must not contain trail color"
        r0, c0, _, _ = bbox(pts)
        rel = tuple(sorted((r - r0, c - c0) for r, c in pts))
        keys.append({"color": color, "bbox_tl": (r0, c0), "norm": rel, "cells": set(pts)})
    assert keys, "no keys found"
    return keys


def _match_keys(locks: list[dict], keys: list[dict]) -> list[tuple[int, int, int]]:
    candidates: list[list[tuple[int, int]]] = []
    for lock in locks:
        cur: list[tuple[int, int]] = []
        for key_idx, key in enumerate(keys):
            for turns in range(4):
                if rotate_normalized(key["norm"], turns) == lock["usable_norm"]:
                    cur.append((key_idx, turns))
                    break
        assert cur, "every lock must have a matching key"
        candidates.append(cur)

    order = sorted(range(len(locks)), key=lambda idx: len(candidates[idx]))
    assignment: list[tuple[int, int, int] | None] = [None] * len(locks)
    used_keys: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        lock_idx = order[pos]
        for key_idx, turns in candidates[lock_idx]:
            if key_idx in used_keys:
                continue
            used_keys.add(key_idx)
            assignment[lock_idx] = (lock_idx, key_idx, turns)
            if backtrack(pos + 1):
                return True
            used_keys.remove(key_idx)
            assignment[lock_idx] = None
        return False

    assert backtrack(0), "key-lock matching must be solvable"
    return [item for item in assignment if item is not None]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    locks = _find_locks(g)
    keys = _find_keys(g)
    assert len(locks) == len(keys), "number of locks and keys must match"
    assignments = _match_keys(locks, keys)

    out = g.copy()
    for key in keys:
        for r, c in key["cells"]:
            out[r, c] = BG

    for lock_idx, key_idx, turns in assignments:
        lock = locks[lock_idx]
        key = keys[key_idx]
        rotated = rotate_normalized(key["norm"], turns)
        start_r, start_c = key["bbox_tl"]
        final_r, final_c = lock["anchor"]
        delta_r = final_r - start_r
        delta_c = final_c - start_c
        assert (delta_r == 0) != (delta_c == 0), "movement must be cardinal"
        steps = abs(delta_r) + abs(delta_c)
        dir_r, dir_c = lock["dir"]
        assert (delta_r, delta_c) == (dir_r * steps, dir_c * steps), "key must slide from the gap direction"

        for step in range(1, steps):
            base_r = start_r + dir_r * step
            base_c = start_c + dir_c * step
            for rr, cc in rotated:
                r = base_r + rr
                c = base_c + cc
                if out[r, c] == BG:
                    out[r, c] = TRAIL

        for rr, cc in rotated:
            out[final_r + rr, final_c + cc] = key["color"]

    return out


def _make_shape_spec(rng: random.Random, used_shapes: set[tuple[tuple[int, int], ...]]) -> dict | None:
    for _ in range(300):
        inner_h = rng.randint(4, 7)
        inner_w = rng.randint(4, 7)
        obstacle_side = rng.choice(SIDES)
        if obstacle_side in ("top", "bottom"):
            obstacle_h = rng.randint(1, inner_h - 1)
            obstacle_w = rng.randint(1, inner_w - 1)
            offset = rng.randint(0, inner_w - obstacle_w)
            row0 = 0 if obstacle_side == "top" else inner_h - obstacle_h
            col0 = offset
        else:
            obstacle_h = rng.randint(1, inner_h - 1)
            obstacle_w = rng.randint(1, inner_w - 1)
            offset = rng.randint(0, inner_h - obstacle_h)
            row0 = offset
            col0 = 0 if obstacle_side == "left" else inner_w - obstacle_w

        obstacle = {
            (r, c)
            for r in range(row0, row0 + obstacle_h)
            for c in range(col0, col0 + obstacle_w)
        }
        usable_raw = tuple(
            sorted((r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) not in obstacle)
        )
        if len(usable_raw) < 6 or not is_connected(usable_raw):
            continue

        usable_norm, usable_origin = normalize_coords(usable_raw)
        if canonical_shape(usable_norm) in used_shapes:
            continue

        entry_side = rng.choice(SIDES)
        if entry_side == "top":
            boundary = [(r, c) for r, c in usable_raw if r == 0]
            gap_rel = (0, 1 + rng.choice(boundary)[1])
        elif entry_side == "bottom":
            boundary = [(r, c) for r, c in usable_raw if r == inner_h - 1]
            gap_rel = (inner_h + 1, 1 + rng.choice(boundary)[1])
        elif entry_side == "left":
            boundary = [(r, c) for r, c in usable_raw if c == 0]
            gap_rel = (1 + rng.choice(boundary)[0], 0)
        else:
            boundary = [(r, c) for r, c in usable_raw if c == inner_w - 1]
            gap_rel = (1 + rng.choice(boundary)[0], inner_w + 1)
        if not boundary:
            continue

        key_turns = rng.randrange(4)
        key_norm = rotate_normalized(usable_norm, key_turns)
        used_shapes.add(canonical_shape(usable_norm))
        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": inner_h + 2,
            "outer_w": inner_w + 2,
            "obstacle": obstacle,
            "usable_raw": usable_raw,
            "usable_norm": usable_norm,
            "usable_origin": usable_origin,
            "entry_side": entry_side,
            "gap_rel": gap_rel,
            "key_norm": key_norm,
        }
    return None


def _frame_cells(top: int, left: int, outer_h: int, outer_w: int, gap_rel: tuple[int, int]) -> set[tuple[int, int]]:
    cells = set()
    for r in range(outer_h):
        for c in range(outer_w):
            if r in (0, outer_h - 1) or c in (0, outer_w - 1):
                if (r, c) != gap_rel:
                    cells.add((top + r, left + c))
    return cells


def _place_spec(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    spec: dict,
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]
    key_h, key_w = shape_dims(spec["key_norm"])
    dir_r, dir_c = SLIDE_DIRS[spec["entry_side"]]
    span = max(outer_h if dir_r else outer_w, key_h if dir_r else key_w)
    distance = span + rng.randint(2, 5)

    for _ in range(400):
        top = rng.randint(0, h - outer_h)
        left = rng.randint(0, w - outer_w)

        target_anchor = (
            top + 1 + spec["usable_origin"][0],
            left + 1 + spec["usable_origin"][1],
        )
        key_anchor = (target_anchor[0] - dir_r * distance, target_anchor[1] - dir_c * distance)

        key_cells = translate(spec["key_norm"], key_anchor[0], key_anchor[1])
        if not all(0 <= r < h and 0 <= c < w for r, c in key_cells):
            continue

        frame_cells = _frame_cells(top, left, outer_h, outer_w, spec["gap_rel"])
        obstacle_cells = translate(spec["obstacle"], top + 1, left + 1)
        corridor: set[tuple[int, int]] = set()
        for step in range(distance + 1):
            corridor |= translate(spec["usable_norm"], key_anchor[0] + dir_r * step, key_anchor[1] + dir_c * step)

        all_cells = frame_cells | obstacle_cells | key_cells | corridor
        if not all(0 <= r < h and 0 <= c < w for r, c in all_cells):
            continue

        padded = dilate(all_cells, h, w, pad=1)
        if reserved & padded:
            continue

        for r, c in frame_cells:
            grid[r, c] = LOCK
        for r, c in obstacle_cells:
            grid[r, c] = OBSTACLE
        for r, c in key_cells:
            grid[r, c] = color
        reserved |= padded
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_locks = rng.choice((2, 3))
    height = rng.randint(26 + 2 * (num_locks - 2), 34 + 2 * (num_locks - 2))
    width = rng.randint(26 + 2 * (num_locks - 2), 34 + 2 * (num_locks - 2))
    grid = np.full((height, width), BG, dtype=int)

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    specs: list[dict] = []
    for _ in range(num_locks):
        spec = _make_shape_spec(rng, used_shapes)
        if spec is None:
            return None
        specs.append(spec)

    colors = list(KEY_COLORS)
    rng.shuffle(colors)
    reserved: set[tuple[int, int]] = set()
    for spec, color in zip(specs, colors):
        if not _place_spec(grid, reserved, spec, color, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_find_locks(grid)) != len(_find_keys(grid)):
        return None
    if np.count_nonzero(out == TRAIL) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
