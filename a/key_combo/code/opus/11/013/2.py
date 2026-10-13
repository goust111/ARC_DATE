from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BG = 0
UNKNOWN = 8
OBJECT_COLORS = (1, 2, 3, 4)
PLUS_OFFSETS = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _same_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in (BG, UNKNOWN):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _exact_parallel_line_fill(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    changed = True
    while changed:
        changed = False
        for axis in (0, 1):
            n_lines = h if axis == 0 else w
            lines = [out[i, :].copy() if axis == 0 else out[:, i].copy() for i in range(n_lines)]
            for i, line in enumerate(lines):
                for pos in np.flatnonzero(line == UNKNOWN):
                    best_support = -1
                    best_distance = n_lines + 1
                    best_values: set[int] = set()
                    for j, other in enumerate(lines):
                        if i == j or int(other[pos]) == UNKNOWN:
                            continue
                        known = (line != UNKNOWN) & (other != UNKNOWN)
                        if not np.any(known) or np.any(line[known] != other[known]):
                            continue
                        support = int(np.sum(known))
                        distance = abs(i - j)
                        value = int(other[pos])
                        if support > best_support or (support == best_support and distance < best_distance):
                            best_support = support
                            best_distance = distance
                            best_values = {value}
                        elif support == best_support and distance == best_distance:
                            best_values.add(value)
                    if len(best_values) == 1:
                        value = next(iter(best_values))
                        if axis == 0:
                            out[i, pos] = value
                        else:
                            out[pos, i] = value
                        changed = True
    return out


def _opposite_pair_fill(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    changed = True
    while changed:
        changed = False
        for r, c in map(tuple, np.argwhere(out == UNKNOWN)):
            candidates: list[int] = []
            if 0 < r < h - 1 and int(out[r - 1, c]) == int(out[r + 1, c]) != UNKNOWN:
                candidates.append(int(out[r - 1, c]))
            if 0 < c < w - 1 and int(out[r, c - 1]) == int(out[r, c + 1]) != UNKNOWN:
                candidates.append(int(out[r, c - 1]))
            if candidates and len(set(candidates)) == 1:
                out[r, c] = candidates[0]
                changed = True
    return out


def _best_line_color(grid: np.ndarray, r: int, c: int) -> int | None:
    candidates: dict[int, tuple[int, int, int]] = {}
    # value -> (best_distance, support_count, axis_hits)
    for cc in range(grid.shape[1]):
        value = int(grid[r, cc])
        if value in (BG, UNKNOWN) or cc == c:
            continue
        dist = abs(cc - c)
        best_dist, support, axis_hits = candidates.get(value, (10**9, 0, 0))
        candidates[value] = (min(best_dist, dist), support + 1, axis_hits + 1)
    for rr in range(grid.shape[0]):
        value = int(grid[rr, c])
        if value in (BG, UNKNOWN) or rr == r:
            continue
        dist = abs(rr - r)
        best_dist, support, axis_hits = candidates.get(value, (10**9, 0, 0))
        candidates[value] = (min(best_dist, dist), support + 1, axis_hits + 1)
    if not candidates:
        return None

    best_color = None
    best_key = None
    tie = False
    for color, (dist, support, axis_hits) in candidates.items():
        key = (dist, -axis_hits, -support)
        if best_key is None or key < best_key:
            best_key = key
            best_color = color
            tie = False
        elif key == best_key:
            tie = True
    return None if tie else best_color


def _nearest_line_fill(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    changed = True
    while changed and np.any(out == UNKNOWN):
        changed = False
        for r, c in map(tuple, np.argwhere(out == UNKNOWN)):
            color = _best_line_color(out, int(r), int(c))
            if color is not None:
                out[r, c] = int(color)
                changed = True
    return out


def _repair_masks(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    while np.any(out == UNKNOWN):
        prev = out.copy()
        out = _exact_parallel_line_fill(out)
        out = _opposite_pair_fill(out)
        out = _nearest_line_fill(out)
        if np.array_equal(out, prev):
            break
    if np.any(out == UNKNOWN):
        raise ValueError("could not infer all masked cells")
    return out


def _component_sort_key(comp: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return (-int(comp["area"]), r0, c0, r1, c1)


def _grow_component(grid: np.ndarray, comp: dict, layers: int) -> np.ndarray:
    out = grid.copy()
    if layers <= 0:
        return out
    color = int(comp["color"])
    visited: set[tuple[int, int]] = set()
    q: deque[tuple[int, int, int]] = deque()
    for r, c in comp["coords"]:
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BG and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc, 1))
    while q:
        r, c, depth = q.popleft()
        out[r, c] = color
        if depth >= layers:
            continue
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BG and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc, depth + 1))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair gray cross masks, recompute true object areas, and grow the repaired largest object by the area gap.

    Input structure:
    - The grid uses black as background and contains 3-4 monochromatic 8-connected colored objects.
    - Some objects are partially hidden by gray 3x3 plus masks whose five gray cells lie inside the object.
    - The visible object pixels form horizontal and vertical color runs, so each masked cell can be inferred from its row or column.

    Rule:
    - Treat every gray cell as hidden damage and replace it using parallel row/column matches first, then nearest visible row/column continuations.
    - After repairing the masks, compute the areas of all colored 8-connected objects.
    - Let M be the largest object and S be the second largest; compute D = area(M) - area(S).
    - Grow M outward through black cells for D BFS layers using 8-connectivity, coloring the newly reached cells with M's color.

    Key insight:
    - The masks corrupt the true area ranking, so the hidden cells must be restored before choosing both the winning object and its growth amount.
    """
    repaired = _repair_masks(np.asarray(grid, dtype=int))
    comps = _same_color_components(repaired)
    if len(comps) < 2:
        return repaired
    comps.sort(key=_component_sort_key)
    winner = comps[0]
    runner_up = comps[1]
    layers = int(winner["area"]) - int(runner_up["area"])
    return _grow_component(repaired, winner, layers)


def _arm_splits_for_area(area: int) -> list[tuple[int, int, int, int]]:
    parts = []
    total = area - 1
    for left in range(2, 5):
        for right in range(2, 5):
            for up in range(2, 5):
                down = total - left - right - up
                if 2 <= down <= 4:
                    parts.append((left, right, up, down))
    return parts


def _make_cross(area: int) -> dict:
    options = _arm_splits_for_area(area)
    if not options:
        raise ValueError(f"unsupported area {area}")
    left, right, up, down = random.choice(options)
    coords = {(0, 0)}
    for dc in range(-left, right + 1):
        coords.add((0, dc))
    for dr in range(-up, down + 1):
        coords.add((dr, 0))
    rel = sorted(coords)
    return {
        "coords": rel,
        "area": len(rel),
        "left": left,
        "right": right,
        "up": up,
        "down": down,
    }


def _translate(coords: list[tuple[int, int]], center_r: int, center_c: int) -> set[tuple[int, int]]:
    return {(center_r + dr, center_c + dc) for dr, dc in coords}


def _touches_existing(candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _choose_area_set(n_objects: int) -> list[int]:
    pool = []
    area_range = range(9, 14) if n_objects == 4 else range(9, 16)
    for combo in combinations(area_range, n_objects):
        if 1 <= combo[-1] - combo[-2] <= 4:
            pool.append(combo)
    chosen = list(random.choice(pool))
    prefix = chosen[:-1]
    random.shuffle(prefix)
    return prefix + [chosen[-1]]


def _feasible_centers(shape: dict, height: int, width: int, extra_margin: int = 0) -> tuple[int, int, int, int]:
    r_lo = shape["up"] + extra_margin
    r_hi = height - 1 - shape["down"] - extra_margin
    c_lo = shape["left"] + extra_margin
    c_hi = width - 1 - shape["right"] - extra_margin
    return r_lo, r_hi, c_lo, c_hi


def _pick_center_in_box(shape: dict, height: int, width: int, box: tuple[int, int, int, int], extra_margin: int = 0):
    r_lo, r_hi, c_lo, c_hi = _feasible_centers(shape, height, width, extra_margin)
    br0, br1, bc0, bc1 = box
    rr0 = max(r_lo, br0)
    rr1 = min(r_hi, br1)
    cc0 = max(c_lo, bc0)
    cc1 = min(c_hi, bc1)
    if rr0 > rr1 or cc0 > cc1:
        return None
    return random.randint(rr0, rr1), random.randint(cc0, cc1)


def _line_safe_for_mask(candidate: dict, others: list[dict]) -> bool:
    row, col = candidate["center"]
    # The masked plus needs the object's own visible arm continuations at distance 2
    # to remain the nearest line evidence.
    for obj in others:
        for r, c in obj["cells"]:
            if r == row and abs(c - col) <= 2:
                return False
            if c == col and abs(r - row) <= 2:
                return False
    return True


def _new_growth_stays_clear(base: np.ndarray, grown: np.ndarray, winner_color: int) -> bool:
    new_cells = {(int(r), int(c)) for r, c in np.argwhere((base == BG) & (grown == winner_color))}
    foreign = {(int(r), int(c)) for r, c in np.argwhere((base != BG) & (base != winner_color))}
    for r, c in new_cells:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in foreign:
                return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(15, 19)
    width = random.randint(15, 19)
    n_objects = 4 if random.random() < 0.75 else 3
    colors = random.sample(list(OBJECT_COLORS), n_objects)
    areas = _choose_area_set(n_objects)
    shapes = [_make_cross(area) for area in areas]
    winner_idx = n_objects - 1
    winner_gap = areas[winner_idx] - sorted(areas)[-2]

    center_box = (height // 3, max(height // 3, (2 * height) // 3), width // 3, max(width // 3, (2 * width) // 3))
    corner_boxes = [
        (1, height // 2 - 3, 1, width // 2 - 3),
        (1, height // 2 - 3, width // 2 + 2, width - 2),
        (height // 2 + 2, height - 2, 1, width // 2 - 3),
        (height // 2 + 2, height - 2, width // 2 + 2, width - 2),
    ]
    random.shuffle(corner_boxes)

    objects: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    winner_shape = shapes[winner_idx]
    winner_center = _pick_center_in_box(winner_shape, height, width, center_box, extra_margin=winner_gap)
    if winner_center is None:
        return None
    winner_cells = _translate(winner_shape["coords"], winner_center[0], winner_center[1])
    if _touches_existing(winner_cells, occupied):
        return None
    winner = {
        "color": colors[winner_idx],
        "area": areas[winner_idx],
        "center": winner_center,
        "cells": winner_cells,
        "shape": winner_shape,
    }
    objects.append(winner)
    occupied |= winner_cells

    for idx in range(n_objects - 1):
        shape = shapes[idx]
        placed = False
        box_order = corner_boxes[:]
        random.shuffle(box_order)
        for box in box_order:
            for _ in range(40):
                center = _pick_center_in_box(shape, height, width, box, extra_margin=0)
                if center is None:
                    break
                cells = _translate(shape["coords"], center[0], center[1])
                if _touches_existing(cells, occupied):
                    continue
                obj = {
                    "color": colors[idx],
                    "area": areas[idx],
                    "center": center,
                    "cells": cells,
                    "shape": shape,
                }
                objects.append(obj)
                occupied |= cells
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    random.shuffle(objects)
    base = np.zeros((height, width), dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            base[r, c] = int(obj["color"])

    clean_components = _same_color_components(base)
    if len(clean_components) != n_objects:
        return None
    clean_components.sort(key=_component_sort_key)
    if len({int(comp["area"]) for comp in clean_components}) != n_objects:
        return None
    if clean_components[0]["area"] - clean_components[1]["area"] != winner_gap:
        return None

    eligible = [obj for obj in objects if _line_safe_for_mask(obj, [other for other in objects if other is not obj])]
    if not eligible:
        return None
    mask_count = 2 if len(eligible) >= 2 and random.random() < 0.6 else 1
    masked = random.sample(eligible, mask_count)

    inp = base.copy()
    for obj in masked:
        cr, cc = obj["center"]
        for dr, dc in PLUS_OFFSETS:
            inp[cr + dr, cc + dc] = UNKNOWN

    try:
        expected = main(base)
        actual = main(inp)
    except ValueError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _new_growth_stays_clear(base, expected, int(clean_components[0]["color"])):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
