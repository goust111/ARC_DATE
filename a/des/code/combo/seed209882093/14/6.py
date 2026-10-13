from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ARROW = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_VECS = {
    "L": (0, -1),
    "R": (0, 1),
    "U": (-1, 0),
    "D": (1, 0),
}
ARROW_PATTERNS = {
    "R": {(0, 0), (-1, -1), (1, -1)},
    "L": {(0, 0), (-1, 1), (1, 1)},
    "U": {(0, 0), (1, -1), (1, 1)},
    "D": {(0, 0), (-1, -1), (-1, 1)},
}
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 180
GEN_RETRIES = 300


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = DIRS4
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in steps:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste_patch(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = out.shape
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            value = int(patch[r, c])
            if value == 0:
                continue
            rr = int(top + r)
            cc = int(left + c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = value


def _find_objects(grid: np.ndarray) -> list[dict]:
    objs: list[dict] = []
    for coords in _components((grid != BG) & (grid != ARROW), connectivity=4):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in coords:
            patch[int(r - r0), int(c - c0)] = int(color)
        objs.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "patch": patch,
            }
        )
    return objs


def _parse_arrow(coords: np.ndarray) -> dict | None:
    cells = {(int(r), int(c)) for r, c in coords}
    if len(cells) != 3:
        return None
    for tip_r, tip_c in cells:
        rel = {(int(r - tip_r), int(c - tip_c)) for r, c in cells}
        for direction, pattern in ARROW_PATTERNS.items():
            if rel == pattern:
                return {
                    "direction": direction,
                    "tip": (int(tip_r), int(tip_c)),
                    "cells": cells,
                }
    return None


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for coords in _components(grid == ARROW, connectivity=8):
        arrow = _parse_arrow(coords)
        if arrow is not None:
            arrows.append(arrow)
    return arrows


def _gap_for_pair(obj: dict, arrow: dict) -> int | None:
    tr, tc = arrow["tip"]
    direction = arrow["direction"]
    if direction == "R":
        if obj["r0"] <= tr <= obj["r1"] and tc > obj["c1"]:
            return int(tc - obj["c1"])
        return None
    if direction == "L":
        if obj["r0"] <= tr <= obj["r1"] and tc < obj["c0"]:
            return int(obj["c0"] - tc)
        return None
    if direction == "D":
        if obj["c0"] <= tc <= obj["c1"] and tr > obj["r1"]:
            return int(tr - obj["r1"])
        return None
    if obj["c0"] <= tc <= obj["c1"] and tr < obj["r0"]:
        return int(obj["r0"] - tr)
    return None


def _pair_objects_and_arrows(objects: list[dict], arrows: list[dict]) -> list[tuple[int, int, int]]:
    if not objects or not arrows:
        return []

    edges: list[tuple[int, int, int]] = []
    obj_best: dict[int, tuple[int, int]] = {}
    arrow_best: dict[int, tuple[int, int]] = {}

    for oi, obj in enumerate(objects):
        for ai, arrow in enumerate(arrows):
            gap = _gap_for_pair(obj, arrow)
            if gap is None:
                continue
            score = (int(gap), int(ai))
            prev = obj_best.get(int(oi))
            if prev is None or score < prev:
                obj_best[int(oi)] = score
            ascore = (int(gap), int(oi))
            aprev = arrow_best.get(int(ai))
            if aprev is None or ascore < aprev:
                arrow_best[int(ai)] = ascore
            edges.append((int(gap), int(oi), int(ai)))

    pairs: list[tuple[int, int, int]] = []
    used_objects: set[int] = set()
    used_arrows: set[int] = set()

    for oi, score in obj_best.items():
        ai = int(score[1])
        if ai not in arrow_best:
            continue
        if int(arrow_best[ai][1]) != int(oi):
            continue
        gap = int(score[0])
        pairs.append((int(oi), int(ai), int(gap)))
        used_objects.add(int(oi))
        used_arrows.add(int(ai))

    for gap, oi, ai in sorted(edges):
        if int(oi) in used_objects or int(ai) in used_arrows:
            continue
        pairs.append((int(oi), int(ai), int(gap)))
        used_objects.add(int(oi))
        used_arrows.add(int(ai))

    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each monochrome object as a stamp and use the nearby Azure arrow to place a flipped copy.

    The arrow encodes three things at once:
    - its pointing direction,
    - the gap from its tip to the source object's bbox,
    - and therefore how many bbox-sized steps to translate the copy.

    Left/right arrows produce a horizontal flip; up/down arrows produce a vertical flip.
    Original objects stay in place and all Azure arrow pixels are cleared.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[g == ARROW] = BG

    objects = _find_objects(g)
    arrows = _find_arrows(g)
    for oi, ai, gap in _pair_objects_and_arrows(objects, arrows):
        obj = objects[int(oi)]
        direction = arrows[int(ai)]["direction"]
        step = int(obj["w"] if direction in ("L", "R") else obj["h"])
        vr, vc = DIR_VECS[direction]
        dr = int(vr * gap * step)
        dc = int(vc * gap * step)
        if direction in ("L", "R"):
            stamp = np.fliplr(obj["patch"])
        else:
            stamp = np.flipud(obj["patch"])
        _paste_patch(out, stamp, int(obj["r0"] + dr), int(obj["c0"] + dc))
    return out


def _random_shape(flip_axis: str) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        raw_h = random.randint(2, 5)
        raw_w = random.randint(2, 5)
        target = random.randint(max(3, (raw_h + raw_w) // 2), raw_h * raw_w - 1)

        cells = {(random.randrange(raw_h), random.randrange(raw_w))}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or w < 2:
            continue

        patch = np.zeros((h, w), dtype=int)
        for r, c in cells:
            patch[int(r - r0), int(c - c0)] = 1

        if int(patch.sum()) == h * w:
            continue
        if flip_axis == "h" and np.array_equal(patch, np.fliplr(patch)):
            continue
        if flip_axis == "v" and np.array_equal(patch, np.flipud(patch)):
            continue
        return patch
    return None


def _arrow_cells(direction: str, h: int, w: int, gap: int) -> set[tuple[int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    if direction == "R":
        tip = (mid_r, w - 1 + gap)
    elif direction == "L":
        tip = (mid_r, -gap)
    elif direction == "D":
        tip = (h - 1 + gap, mid_c)
    else:
        tip = (-gap, mid_c)
    return {(int(tip[0] + dr), int(tip[1] + dc)) for dr, dc in ARROW_PATTERNS[direction]}


def _copy_offset(direction: str, h: int, w: int, gap: int) -> tuple[int, int]:
    if direction == "R":
        return (0, int(gap * w))
    if direction == "L":
        return (0, int(-gap * w))
    if direction == "D":
        return (int(gap * h), 0)
    return (int(-gap * h), 0)


def _lane_cells(direction: str, h: int, w: int, gap: int) -> set[tuple[int, int]]:
    off_r, off_c = _copy_offset(direction, h, w, gap)
    if direction == "R":
        return {(r, c) for r in range(h) for c in range(0, off_c + w)}
    if direction == "L":
        return {(r, c) for r in range(h) for c in range(off_c, w)}
    if direction == "D":
        return {(r, c) for r in range(0, off_r + h) for c in range(w)}
    return {(r, c) for r in range(off_r, h) for c in range(w)}


def _bounds_of_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _make_spec(color: int, direction: str) -> dict | None:
    flip_axis = "h" if direction in ("L", "R") else "v"
    patch = _random_shape(flip_axis)
    if patch is None:
        return None

    h, w = patch.shape
    step = int(w if direction in ("L", "R") else h)
    gap_choices = [2, 3] if step >= 4 else [2, 3, 4]
    gap = int(random.choice(gap_choices))
    copy_dr, copy_dc = _copy_offset(direction, h, w, gap)

    object_cells = {(int(r), int(c)) for r, c in np.argwhere(patch != 0)}
    copy_cells = {(int(r + copy_dr), int(c + copy_dc)) for r, c in object_cells}
    arrow_cells = _arrow_cells(direction, h, w, gap)
    lane_cells = _lane_cells(direction, h, w, gap)
    reserved_cells = set(lane_cells) | set(arrow_cells)
    min_r, max_r, min_c, max_c = _bounds_of_cells(reserved_cells)

    return {
        "color": int(color),
        "direction": direction,
        "patch": patch.astype(int),
        "h": int(h),
        "w": int(w),
        "gap": int(gap),
        "copy_dr": int(copy_dr),
        "copy_dc": int(copy_dc),
        "object_cells": object_cells,
        "copy_cells": copy_cells,
        "arrow_cells": arrow_cells,
        "reserved_cells": reserved_cells,
        "min_r": int(min_r),
        "max_r": int(max_r),
        "min_c": int(min_c),
        "max_c": int(max_c),
        "reserved_area": int(len(reserved_cells)),
    }


def _dilate8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_specs(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    forbidden: set[tuple[int, int]] = set()

    ordered = sorted(specs, key=lambda spec: (-spec["reserved_area"], spec["direction"], spec["color"]))
    for spec in ordered:
        row_lo = int(-spec["min_r"])
        row_hi = int(h - 1 - spec["max_r"])
        col_lo = int(-spec["min_c"])
        col_hi = int(w - 1 - spec["max_c"])
        if row_lo > row_hi or col_lo > col_hi:
            return None

        success = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            reserved_abs = {(int(top + r), int(left + c)) for r, c in spec["reserved_cells"]}
            if any(cell in forbidden for cell in reserved_abs):
                continue

            placed_spec = dict(spec)
            placed_spec["top"] = int(top)
            placed_spec["left"] = int(left)
            placed.append(placed_spec)
            forbidden |= _dilate8(reserved_abs, h, w)
            success = True
            break
        if not success:
            return None

    return placed


def _paint_bool_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            if int(patch[r, c]) != 0:
                grid[int(top + r), int(left + c)] = int(color)


def _build_grids(h: int, w: int, placed_specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for spec in placed_specs:
        top = int(spec["top"])
        left = int(spec["left"])
        color = int(spec["color"])
        patch = spec["patch"]

        _paint_bool_patch(inp, patch, top, left, color)
        _paint_bool_patch(out, patch, top, left, color)

        for r, c in spec["arrow_cells"]:
            inp[int(top + r), int(left + c)] = ARROW

        if spec["direction"] in ("L", "R"):
            flipped = np.fliplr(patch)
        else:
            flipped = np.flipud(patch)
        _paint_bool_patch(out, flipped, int(top + spec["copy_dr"]), int(left + spec["copy_dc"]), color)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    max_objects = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(PALETTE, k=max_objects)

    directions = random.sample(list(DIR_VECS.keys()), k=min(max_objects, 4))
    while len(directions) < max_objects:
        directions.append(random.choice(list(DIR_VECS.keys())))
    random.shuffle(directions)

    specs: list[dict] = []
    for color, direction in zip(colors, directions):
        spec = _make_spec(int(color), direction)
        if spec is None:
            return None
        specs.append(spec)

    min_h = max(spec["max_r"] - spec["min_r"] + 1 for spec in specs)
    min_w = max(spec["max_c"] - spec["min_c"] + 1 for spec in specs)
    low_h = max(GRID_H_RANGE[0], int(min_h + 2), int(10 + 4 * max_objects))
    low_w = max(GRID_W_RANGE[0], int(min_w + 2), int(10 + 4 * max_objects))
    if low_h > GRID_H_RANGE[1] or low_w > GRID_W_RANGE[1]:
        return None

    h = random.randint(low_h, GRID_H_RANGE[1])
    w = random.randint(low_w, GRID_W_RANGE[1])
    placed_specs = _place_specs(h, w, specs)
    if placed_specs is None:
        return None

    inp, expected = _build_grids(h, w, placed_specs)
    if len(_find_objects(inp)) != len(placed_specs):
        return None
    if len(_find_arrows(inp)) != len(placed_specs):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
