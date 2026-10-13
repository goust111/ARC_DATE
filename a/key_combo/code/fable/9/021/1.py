from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return tuple(out)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if (nr, nc) in seen or int(grid[nr, nc]) != color:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                    "cells": set(cells),
                }
            )
    return out


def _is_straight_bar(component: dict) -> tuple[bool, tuple[int, int], str | None]:
    if component["area"] != 3:
        return False, (-1, -1), None
    coords = component["coords"]
    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    if len(rows) == 1 and len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
        return True, (rows[0], cols[1]), "horizontal"
    if len(cols) == 1 and len(rows) == 3 and rows == list(range(rows[0], rows[0] + 3)):
        return True, (rows[1], cols[0]), "vertical"
    return False, (-1, -1), None


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    markers = []
    singletons = [comp for comp in comps if comp["area"] == 1]
    for comp in comps:
        ok, center, axis = _is_straight_bar(comp)
        if not ok:
            continue
        controls = []
        for singleton in singletons:
            if singleton["color"] == comp["color"]:
                continue
            control = tuple(map(int, singleton["coords"][0]))
            if abs(control[0] - center[0]) + abs(control[1] - center[1]) != 1:
                continue
            dr = center[0] - control[0]
            dc = center[1] - control[1]
            if axis == "horizontal" and dc != 0:
                continue
            if axis == "vertical" and dr != 0:
                continue
            controls.append((singleton, (dr, dc)))
        if len(controls) == 1:
            markers.append((comp, center, axis, controls[0][0], controls[0][1]))

    assert len(markers) == 1, "expected exactly one marker bar with one adjacent control pixel"
    bar_comp, center, axis, control_comp, step = markers[0]

    targets = [comp for comp in comps if comp not in (bar_comp, control_comp)]
    assert targets, "expected at least one target object"
    target_colors = [comp["color"] for comp in targets]
    assert len(target_colors) == len(set(target_colors)), "target objects must have distinct colors"

    ray = []
    r = center[0] + step[0]
    c = center[1] + step[1]
    while _in_bounds(g.shape, r, c):
        ray.append((r, c))
        r += step[0]
        c += step[1]
    assert ray, "expected a non-empty marker ray"

    ray_index = {cell: idx for idx, cell in enumerate(ray)}
    hit = []
    miss = []
    for comp in targets:
        hits = sorted(ray_index[cell] for cell in comp["cells"] if cell in ray_index)
        if hits:
            hit.append((hits[0], comp))
        else:
            miss.append(comp)
    hit.sort(key=lambda item: item[0])
    assert hit, "expected at least one hit target"

    return {
        "bg": bg,
        "bar": bar_comp,
        "control": control_comp,
        "center": center,
        "axis": axis,
        "step": step,
        "ray": ray,
        "hit": [comp for _idx, comp in hit],
        "miss": miss,
        "targets": targets,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift colors along the marker ray's hit order, with the marker bar closing the cycle.

    Input structure:
    - The background is the dominant color.
    - One three-pixel straight marker bar has a singleton control pixel attached to the bar center, and those two colors are unique.
    - Looking from the bar center away from the control, the ray intersects several 4-connected solid target objects with distinct colors; other target objects sit off the ray as distractors.

    Rule:
    - Cast a straight ray from the marker-bar center in the direction opposite the control pixel.
    - Order every target object touched by that ray by first hit.
    - Recolor the first hit object to the bar color, recolor each later hit object to the previous hit object's original color, and recolor the whole bar to the last hit object's original color.
    - Keep the control pixel, every unhit target object, and the background unchanged.

    Key insight:
    - The ray carries color state through a whole sequence: the complete hit order must be known before any new color is determined, because the colors rotate one step along the chain and wrap back onto the bar.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    hit = scene["hit"]
    out = g.copy()

    original_hit_colors = [comp["color"] for comp in hit]
    bar_color = scene["bar"]["color"]
    shifted_colors = [bar_color] + original_hit_colors[:-1]

    for comp, new_color in zip(hit, shifted_colors):
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(new_color)

    bar_coords = scene["bar"]["coords"]
    out[bar_coords[:, 0], bar_coords[:, 1]] = int(original_hit_colors[-1])
    return out


def _bar_cells(center: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    if step[0] != 0:
        return [(r, c - 1), (r, c), (r, c + 1)]
    return [(r - 1, c), (r, c), (r + 1, c)]


def _sample_marker() -> tuple[np.ndarray, dict]:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    grid = np.zeros((h, w), dtype=int)
    step = random.choice(CARDINALS)

    if step == (-1, 0):
        ray_len = random.randint(7, h - 2)
        center = (ray_len, random.randint(1, w - 2))
    elif step == (1, 0):
        ray_len = random.randint(7, h - 2)
        center = (h - 1 - ray_len, random.randint(1, w - 2))
    elif step == (0, -1):
        ray_len = random.randint(7, w - 2)
        center = (random.randint(1, h - 2), ray_len)
    else:
        ray_len = random.randint(7, w - 2)
        center = (random.randint(1, h - 2), w - 1 - ray_len)

    control = (center[0] - step[0], center[1] - step[1])
    bar = _bar_cells(center, step)
    ray = []
    r = center[0] + step[0]
    c = center[1] + step[1]
    while _in_bounds(grid.shape, r, c):
        ray.append((r, c))
        r += step[0]
        c += step[1]

    return grid, {"center": center, "control": control, "bar": bar, "step": step, "ray": ray}


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _sample_hit_indices(ray_len: int, count: int) -> list[int]:
    slack = ray_len - 2 * count
    assert slack >= 0
    gaps = _random_composition(slack, count + 1)
    idxs = [1 + gaps[0]]
    for i in range(1, count):
        idxs.append(idxs[-1] + 2 + gaps[i])
    return idxs


def _candidate_cells(
    shape: tuple[int, int],
    current: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = []
    seen = set()
    for r, c in current:
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(shape, nr, nc):
                continue
            if (nr, nc) in seen or (nr, nc) in current or (nr, nc) in blocked:
                continue
            if any((ar, ac) in occupied for ar, ac in _neighbors8(nr, nc)):
                continue
            seen.add((nr, nc))
            candidates.append((nr, nc))
    random.shuffle(candidates)
    return candidates


def _grow_object(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    size: int,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    if anchor in blocked or anchor in occupied:
        return None
    current = {anchor}
    for _ in range(size - 1):
        candidates = _candidate_cells(shape, current, blocked, occupied)
        if not candidates:
            return None
        current.add(candidates[0])
    return current


def _place_object(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    color: int,
    occupied: set[tuple[int, int]],
) -> None:
    for r, c in cells:
        grid[r, c] = int(color)
        occupied.add((r, c))


def _sizes_for_hits(count: int) -> list[int]:
    sizes = [1, random.choice((4, 5))]
    while len(sizes) < count:
        sizes.append(random.choice((1, 2, 4, 5)))
    random.shuffle(sizes)
    return sizes


def _perp_dirs(step: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if step[0] != 0:
        return ((0, -1), (0, 1))
    return ((-1, 0), (1, 0))


def _place_hit_objects(
    grid: np.ndarray,
    ray: list[tuple[int, int]],
    step: tuple[int, int],
    colors: list[int],
    occupied: set[tuple[int, int]],
) -> list[set[tuple[int, int]]] | None:
    max_hits = min(5, len(ray) // 2, len(colors) - 1)
    count = random.randint(3, max_hits)
    sizes = _sizes_for_hits(count)
    anchors = _sample_hit_indices(len(ray), count)
    ray_set = set(ray)
    placed: list[set[tuple[int, int]]] = []

    for idx, size, color in zip(anchors, sizes, colors[:count]):
        anchor = ray[idx]
        blocked = (ray_set - {anchor}) | occupied
        cells = None
        for _ in range(40):
            cells = _grow_object(grid.shape, anchor, size, occupied, blocked)
            if cells is not None:
                break
        if cells is None:
            return None
        _place_object(grid, cells, color, occupied)
        placed.append(cells)
    return placed


def _find_near_ray_anchor(
    grid: np.ndarray,
    ray: list[tuple[int, int]],
    step: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    dirs = _perp_dirs(step)
    choices = list(range(len(ray)))
    random.shuffle(choices)
    for idx in choices:
        rr, cc = ray[idx]
        side_dirs = list(dirs)
        random.shuffle(side_dirs)
        for dr, dc in side_dirs:
            anchor = (rr + dr, cc + dc)
            if not _in_bounds(grid.shape, anchor[0], anchor[1]):
                continue
            if anchor in occupied or anchor in set(ray):
                continue
            if any(cell in occupied for cell in _neighbors8(anchor[0], anchor[1])):
                continue
            return anchor
    return None


def _place_miss_objects(
    grid: np.ndarray,
    ray: list[tuple[int, int]],
    step: tuple[int, int],
    colors: list[int],
    occupied: set[tuple[int, int]],
    color_offset: int,
) -> list[set[tuple[int, int]]] | None:
    ray_set = set(ray)
    placed: list[set[tuple[int, int]]] = []

    near_anchor = _find_near_ray_anchor(grid, ray, step, occupied)
    if near_anchor is None:
        return None
    near_size = random.choice((1, 2, 4, 5))
    near = None
    for _ in range(50):
        near = _grow_object(grid.shape, near_anchor, near_size, occupied, ray_set)
        if near is not None and not any(cell in ray_set for cell in near):
            break
    if near is None:
        return None
    _place_object(grid, near, colors[color_offset], occupied)
    placed.append(near)

    extra_count = random.randint(0, max(0, len(colors) - color_offset - 1))
    for extra_idx in range(extra_count):
        size = random.choice((1, 2, 4, 5))
        cells = None
        for _ in range(80):
            anchor = (random.randrange(grid.shape[0]), random.randrange(grid.shape[1]))
            cells = _grow_object(grid.shape, anchor, size, occupied, ray_set)
            if cells is not None and not any(cell in ray_set for cell in cells):
                break
        if cells is None:
            return None
        _place_object(grid, cells, colors[color_offset + 1 + extra_idx], occupied)
        placed.append(cells)
    return placed


def _distance_to_ray(cells: set[tuple[int, int]], ray: list[tuple[int, int]]) -> int:
    best = 10**9
    for r, c in cells:
        for rr, cc in ray:
            best = min(best, abs(r - rr) + abs(c - cc))
    return best


def _validate_instance(grid: np.ndarray) -> bool:
    scene = _parse_scene(grid)
    hit = scene["hit"]
    miss = scene["miss"]
    if len(hit) < 3 or len(miss) < 1:
        return False
    hit_areas = [comp["area"] for comp in hit]
    if not any(area == 1 for area in hit_areas):
        return False
    if not any(area > 1 for area in hit_areas):
        return False
    if not any(_distance_to_ray(comp["cells"], scene["ray"]) == 1 for comp in miss):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    original_bar_color = scene["bar"]["color"]
    original_hit_colors = [comp["color"] for comp in hit]
    for idx, comp in enumerate(hit):
        expected = original_bar_color if idx == 0 else original_hit_colors[idx - 1]
        coords = comp["coords"]
        if len(set(map(int, out[coords[:, 0], coords[:, 1]]))) != 1:
            return False
        if int(out[coords[0, 0], coords[0, 1]]) != expected:
            return False

    bar_coords = scene["bar"]["coords"]
    if int(out[bar_coords[0, 0], bar_coords[0, 1]]) != original_hit_colors[-1]:
        return False

    control_coords = scene["control"]["coords"]
    if not np.array_equal(
        out[control_coords[:, 0], control_coords[:, 1]],
        grid[control_coords[:, 0], control_coords[:, 1]],
    ):
        return False

    for comp in miss:
        coords = comp["coords"]
        if not np.array_equal(out[coords[:, 0], coords[:, 1]], grid[coords[:, 0], coords[:, 1]]):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, marker = _sample_marker()
        total_target_count = random.randint(5, 7)
        colors = list(PALETTE)
        random.shuffle(colors)
        bg = colors.pop()
        bar_color = colors.pop()
        control_color = colors.pop()
        target_colors = colors[:total_target_count]
        grid.fill(bg)

        occupied = set()
        for r, c in marker["bar"]:
            grid[r, c] = bar_color
            occupied.add((r, c))
        grid[marker["control"]] = control_color
        occupied.add(marker["control"])

        hits = _place_hit_objects(grid, marker["ray"], marker["step"], target_colors, occupied)
        if hits is None:
            continue
        misses = _place_miss_objects(grid, marker["ray"], marker["step"], target_colors, occupied, len(hits))
        if misses is None:
            continue
        if not _validate_instance(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid ARC instance")
