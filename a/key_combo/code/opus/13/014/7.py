import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

HOLE_TO_COLOR = {1: BLUE, 2: RED, 3: ORANGE}

GRID_ROWS = (18, 22)
GRID_COLS = (20, 25)
OBJECT_COUNT_CHOICES = (4, 5)
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_holes_in_component(grid: np.ndarray, coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = crop.shape
    obj_mask = crop == GREY
    bg_mask = crop == BG
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bg_mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bg_mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1

    assert holes in HOLE_TO_COLOR
    assert obj_mask.any()
    return holes


def _expand_bbox(bbox: tuple[int, int, int, int], radius: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - radius),
        max(0, c0 - radius),
        min(h - 1, r1 + radius),
        min(w - 1, c1 + radius),
    )


def _choose_detour_row(blocked: np.ndarray, r: int, c: int) -> int | None:
    h, w = blocked.shape
    assert c + 1 < w

    def search(step: int) -> tuple[int, int] | None:
        rr = r
        dist = 0
        while 0 <= rr + step < h:
            rr += step
            dist += 1
            if blocked[rr, c]:
                return None
            if not blocked[rr, c + 1]:
                return rr, dist
        return None

    up = search(-1)
    down = search(1)
    if up is None and down is None:
        return None
    if up is None:
        return int(down[0])
    if down is None:
        return int(up[0])
    if up[1] <= down[1]:
        return int(up[0])
    return int(down[0])


def _trace_ray(start: tuple[int, int], blocked: np.ndarray) -> tuple[list[tuple[int, int]] | None, int]:
    h, w = blocked.shape
    r, c = map(int, start)
    if not (0 <= r < h and 0 <= c < w) or blocked[r, c]:
        return None, 0

    path = [(r, c)]
    detours = 0
    steps = 0

    while c < w - 1:
        steps += 1
        if steps > h * w * 2:
            return None, detours
        if not blocked[r, c + 1]:
            c += 1
            path.append((r, c))
            continue

        target_r = _choose_detour_row(blocked, r, c)
        if target_r is None:
            return None, detours

        detours += 1
        step = -1 if target_r < r else 1
        while r != target_r:
            r += step
            if blocked[r, c]:
                return None, detours
            path.append((r, c))

    return path, detours


def _path_directions(path: list[tuple[int, int]]) -> set[int]:
    dirs = set()
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        if c0 == c1 and r0 != r1:
            dirs.add(1 if r1 > r0 else -1)
    return dirs


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 3) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return row_gap >= gap or col_gap >= gap


def _hole_positions(height: int, width: int, n_holes: int) -> list[tuple[int, int]] | None:
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(interior)

    def backtrack(idx: int, chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == n_holes:
            return chosen[:]
        if idx >= len(interior):
            return None

        for j in range(idx, len(interior)):
            r, c = interior[j]
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            out = backtrack(j + 1, chosen)
            if out is not None:
                return out
            chosen.pop()
        return None

    return backtrack(0, [])


def _make_object(n_holes: int) -> np.ndarray:
    dim_choices = {
        1: [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)],
        2: [(4, 4), (4, 5), (5, 4), (5, 5), (5, 6)],
        3: [(5, 5), (5, 6), (6, 5), (6, 6)],
    }[int(n_holes)][:]
    random.shuffle(dim_choices)

    for height, width in dim_choices:
        holes = _hole_positions(height, width, n_holes)
        if holes is None:
            continue
        obj = np.full((height, width), GREY, dtype=int)
        for r, c in holes:
            obj[r, c] = BG
        if _count_holes_in_component(obj, np.argwhere(obj == GREY)) == n_holes:
            return obj
    raise RuntimeError(f"failed to build object with {n_holes} holes")


def _slot_candidates(width: int, obj_w: int, slot: str) -> list[int]:
    hi = width - obj_w - 2
    if hi < 4:
        return []
    if slot == "left":
        lo = 4
        hi_slot = max(lo, min(6, width // 3))
        return list(range(lo, min(hi, hi_slot) + 1))
    if slot == "center":
        center = width // 2 - obj_w // 2
        vals = [center - 1, center, center + 1]
        return [c for c in vals if 4 <= c <= hi]
    if slot == "right":
        lo = max(4, width - obj_w - 6)
        return list(range(lo, hi + 1))
    raise ValueError(slot)


def _band_range(height: int, obj_h: int, band: str) -> list[int]:
    if band == "top":
        lo, hi = 1, max(1, height // 4)
    elif band == "mid":
        center = height // 2 - obj_h // 2
        lo, hi = center - 1, center + 1
    elif band == "bottom":
        lo, hi = max(1, height - obj_h - max(2, height // 5)), height - obj_h - 1
    else:
        raise ValueError(band)
    lo = max(1, lo)
    hi = min(height - obj_h - 1, hi)
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _layout_specs(object_count: int) -> list[tuple[str, str]]:
    if object_count == 4:
        patterns = [
            [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")],
            [("top", "left"), ("mid", "center"), ("bottom", "left"), ("bottom", "right")],
            [("top", "left"), ("top", "right"), ("mid", "center"), ("bottom", "center")],
        ]
    else:
        patterns = [
            [("top", "left"), ("top", "right"), ("mid", "center"), ("bottom", "left"), ("bottom", "right")],
            [("top", "left"), ("mid", "left"), ("mid", "right"), ("bottom", "left"), ("bottom", "right")],
        ]
    return random.choice(patterns)


def _place_objects(shape: tuple[int, int], objects: list[np.ndarray], hole_counts: list[int]):
    h, w = shape
    specs = _layout_specs(len(objects))
    order = list(range(len(objects)))
    order.sort(key=lambda i: (0 if specs[i][0] == "top" else 1 if specs[i][0] == "mid" else 2, specs[i][1]))

    placements: list[dict] = []
    for idx in order:
        obj = objects[idx]
        band, slot = specs[idx]
        obj_h, obj_w = obj.shape
        row_candidates = _band_range(h, obj_h, band)
        col_candidates = _slot_candidates(w, obj_w, slot)
        random.shuffle(row_candidates)
        random.shuffle(col_candidates)
        placed = False
        for r0 in row_candidates:
            for c0 in col_candidates:
                bbox = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
                if any(not _boxes_far_enough(bbox, item["bbox"]) for item in placements):
                    continue
                placements.append(
                    {
                        "bbox": bbox,
                        "cells": obj.copy(),
                        "holes": int(hole_counts[idx]),
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return placements


def _build_input_grid(shape: tuple[int, int], placements: list[dict], seeds: list[int] | None = None) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for item in placements:
        r0, c0, r1, c1 = item["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = item["cells"]
    for row in seeds or []:
        grid[int(row), 0] = GREEN
    return grid


def _base_blocked(shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    for item in placements:
        r0, c0, r1, c1 = _expand_bbox(item["bbox"], item["holes"], shape)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    return blocked


def _row_hits_object(placements: list[dict], row: int) -> bool:
    return any(item["bbox"][0] <= row <= item["bbox"][2] for item in placements)


def _candidate_seed_paths(shape: tuple[int, int], placements: list[dict]):
    blocked = _base_blocked(shape, placements)
    h, _ = shape
    accepted = []
    for row in range(h):
        if _row_hits_object(placements, row):
            continue
        path, detours = _trace_ray((row, 0), blocked)
        if path is None or detours == 0:
            continue
        if len(path) <= blocked.shape[1]:
            continue
        accepted.append(
            {
                "row": row,
                "path": path,
                "detours": detours,
                "dirs": _path_directions(path),
            }
        )
    return accepted


def _choose_seed_rows(candidates: list[dict]) -> list[int] | None:
    if not candidates:
        return None

    random.shuffle(candidates)
    pairs = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if abs(a["row"] - b["row"]) < 3:
                continue
            if set(a["path"]) & set(b["path"]):
                continue
            diversity = len(a["dirs"] | b["dirs"])
            pairs.append((diversity, a["detours"] + b["detours"], [a["row"], b["row"]]))
    pairs.sort(reverse=True)

    if pairs and random.random() < 0.6:
        return pairs[0][2]

    candidates.sort(key=lambda item: (item["detours"], len(item["dirs"])), reverse=True)
    return [candidates[0]["row"]]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    hole_counts = [1, 2, 3]
    while len(hole_counts) < object_count:
        hole_counts.append(random.choice([1, 2, 3]))
    random.shuffle(hole_counts)

    objects = [_make_object(count) for count in hole_counts]
    placements = _place_objects((h, w), objects, hole_counts)
    if placements is None:
        return None

    candidates = _candidate_seed_paths((h, w), placements)
    seed_rows = _choose_seed_rows(candidates)
    if not seed_rows:
        return None

    grid = _build_input_grid((h, w), placements, seed_rows)
    out = main(grid)

    if int(np.count_nonzero(out == GREEN)) <= len(seed_rows):
        return None
    colors = {int(v) for v in out.ravel()}
    if not {BLUE, RED, ORANGE}.issubset(colors):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each holed gray rectangle and route green rays around hole-sized exclusion zones.

    Input structure:
    - Black background contains 4-connected gray rectangular foreground objects, each with one to three enclosed black single-cell holes.
    - One or two green seed pixels sit on the left edge, on rows outside every object's row span.
    - Objects are separated so each gray component is unambiguous.

    Rule:
    - For each gray object, count the enclosed black 4-connected regions inside its bounding box.
    - Recolor all gray pixels of that object by hole count: 1 -> blue, 2 -> red, 3 -> orange, while leaving the holes black.
    - Give each object an exclusion radius equal to its hole count, formed by expanding the object's bounding box by that Chebyshev radius.
    - From each green seed, extend a horizontal ray to the right; when the next cell would enter any exclusion zone, move vertically by the shorter route until the cell to the right becomes free, preferring upward ties, then continue right. Rays paint green only on background cells and stop at the right edge.

    Key insight:
    - The same hole count drives two coupled outputs: object recoloring and the size of the rectangular keep-out zone that bends the green ray.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    objects = []
    for coords in _components_of_color(g, GREY):
        holes = _count_holes_in_component(g, coords)
        bbox = _bbox(coords)
        rr, cc = coords.T
        out[rr, cc] = HOLE_TO_COLOR[holes]
        objects.append({"bbox": bbox, "holes": holes})

    base_blocked = out != BG
    for item in objects:
        r0, c0, r1, c1 = _expand_bbox(item["bbox"], item["holes"], g.shape)
        base_blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    seeds.sort()
    assert 1 <= len(seeds) <= 2
    assert all(c == 0 for _, c in seeds)

    for seed in seeds:
        blocked = base_blocked.copy()
        blocked[seed] = False
        path, _detours = _trace_ray(seed, blocked)
        assert path is not None
        for r, c in path:
            if out[r, c] == BG:
                out[r, c] = GREEN

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
