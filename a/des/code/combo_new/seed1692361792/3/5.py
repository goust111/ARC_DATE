from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

# Generator bounds.
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
TARGET_COUNT_RANGE = (2, 5)
ARROW_COUNT_RANGE = (1, 4)
TARGET_AREA_RANGE = (3, 7)
TARGET_BBOX_MAX = 5
TARGET_PLACE_TRIES = 120
ARROW_PLACE_TRIES = 200
GENERATE_RETRIES = 500
HIT_GAP_MIN = 2
MISS_RAY_MIN = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS8]


def _components(
    grid: np.ndarray,
    *,
    connectivity: int,
    excluded: np.ndarray | None = None,
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    if excluded is None:
        excluded = np.zeros((h, w), dtype=bool)
    steps = DIRS8 if connectivity == 8 else DIRS4
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or excluded[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if not _in_bounds(g.shape, nr, nc) or seen[nr, nc] or excluded[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": np.asarray(coords, dtype=int),
                }
            )
    return comps


def _arrow_cells(
    origin: tuple[int, int],
    step: tuple[int, int],
    *,
    style: str,
) -> set[tuple[int, int]]:
    r, c = origin
    dr, dc = step
    orr, occ = -dc, dr
    cells = {
        (r, c),
        (r - dr, c - dc),
        (r - 2 * dr, c - 2 * dc),
    }
    if style == "diag":
        cells.add((r + dr + orr, c + dc + occ))
        cells.add((r + dr - orr, c + dc - occ))
    else:
        cells.add((r + orr, c + occ))
        cells.add((r - orr, c - occ))
    return cells


def _match_arrow_component(coords: set[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 5:
        return None

    matches: list[dict[str, object]] = []
    for origin in coords:
        for step in DIRS4:
            for style in ("diag", "broad"):
                if coords == _arrow_cells(origin, step, style=style):
                    matches.append(
                        {
                            "origin": origin,
                            "direction": step,
                            "style": style,
                        }
                    )

    if len(matches) != 1:
        return None
    return matches[0]


def _detect_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _components(grid, connectivity=8):
        coords = {tuple(map(int, cell)) for cell in np.asarray(comp["coords"], dtype=int)}
        match = _match_arrow_component(coords)
        if match is None:
            continue
        arrows.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "origin": match["origin"],
                "direction": match["direction"],
                "style": match["style"],
            }
        )
    arrows.sort(key=lambda item: (item["origin"][0], item["origin"][1], item["direction"]))
    return arrows


def _target_objects(
    grid: np.ndarray,
    arrow_cells: set[tuple[int, int]],
) -> tuple[list[np.ndarray], dict[tuple[int, int], int]]:
    excluded = np.zeros_like(grid, dtype=bool)
    for r, c in arrow_cells:
        excluded[r, c] = True
    objects = _components(grid, connectivity=4, excluded=excluded)
    coords_list = [np.asarray(obj["coords"], dtype=int) for obj in objects]
    index: dict[tuple[int, int], int] = {}
    for i, coords in enumerate(coords_list):
        for r, c in coords:
            index[(int(r), int(c))] = i
    return coords_list, index


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a same-color ray from each arrow and recolor the first struck non-arrow object.

    Input structure:
    - The background is black.
    - The foreground consists of separated monochrome objects, some of which are 5-pixel arrows that encode a cardinal direction.
    - Each arrow has a 3-pixel shaft plus two arrowhead pixels at one end; the remaining monochrome objects are non-arrow targets of arbitrary 4-connected shapes.

    Rule:
    - Detect every arrow and read its pointing direction from the arrowhead end of the shaft.
    - Starting just in front of that tip-center shaft pixel, extend a straight ray in the arrow's color through background cells until the first original non-background pixel or the grid edge.
    - Paint all traversed background cells with the arrow color; if the first struck object is a non-arrow target, recolor that entire target object to the arrow color, while arrow objects themselves stay unchanged.

    Key insight:
    - The arrow shape is a symbolic direction marker: once its pointing end is decoded, the puzzle becomes a ray-cast whose first collision selects an entire object for recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrows = _detect_arrows(g)
    assert arrows, "expected at least one arrow object"

    arrow_cells: set[tuple[int, int]] = set()
    arrow_owner: dict[tuple[int, int], int] = {}
    for arrow_id, arrow in enumerate(arrows):
        for cell in arrow["coords"]:
            assert cell not in arrow_owner, "arrows must be disjoint"
            arrow_cells.add(cell)
            arrow_owner[cell] = arrow_id

    targets, target_index = _target_objects(g, arrow_cells)
    out = g.copy()

    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = arrow["direction"]
        r, c = arrow["origin"]
        rr, cc = r + dr, c + dc

        while _in_bounds(g.shape, rr, cc):
            if int(g[rr, cc]) == BG:
                out[rr, cc] = color
                rr += dr
                cc += dc
                continue

            hit = (rr, cc)
            if hit not in arrow_cells:
                target_id = target_index.get(hit)
                assert target_id is not None, "every non-arrow foreground pixel must belong to a target object"
                coords = targets[target_id]
                out[coords[:, 0], coords[:, 1]] = color
            break

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_polyomino(area: int) -> set[tuple[int, int]] | None:
    cells = {(0, 0)}
    for _ in range(area * 8):
        if len(cells) >= area:
            break
        base = random.choice(tuple(cells))
        nbrs = _neighbors4(*base)
        random.shuffle(nbrs)
        for nbr in nbrs:
            if nbr not in cells:
                cells.add(nbr)
                break
    if len(cells) != area:
        return None
    cells = _normalize(cells)
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    if max_r + 1 > TARGET_BBOX_MAX or max_c + 1 > TARGET_BBOX_MAX:
        return None
    if _match_arrow_component(cells) is not None:
        return None
    return cells


def _make_target_patch() -> set[tuple[int, int]] | None:
    for _ in range(80):
        area = random.randint(*TARGET_AREA_RANGE)
        cells = _random_polyomino(area)
        if cells is None:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if area >= 4 and (len(rows) == 1 or len(cols) == 1) and random.random() < 0.7:
            continue
        return cells
    return None


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _object_fits(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in occupied or (r, c) in reserved_rays:
            return False
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in occupied and (nr, nc) not in cells:
                return False
    return True


def _front_cells(cells: set[tuple[int, int]], step: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = step
    front: dict[int, tuple[int, int]] = {}
    if dr == 0 and dc == 1:
        for r, c in cells:
            if r not in front or c < front[r][1]:
                front[r] = (r, c)
    elif dr == 0 and dc == -1:
        for r, c in cells:
            if r not in front or c > front[r][1]:
                front[r] = (r, c)
    elif dr == 1 and dc == 0:
        for r, c in cells:
            if c not in front or r < front[c][0]:
                front[c] = (r, c)
    else:
        for r, c in cells:
            if c not in front or r > front[c][0]:
                front[c] = (r, c)
    out = list(front.values())
    random.shuffle(out)
    return out


def _ray_to_edge(origin: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = step
    r = origin[0] + dr
    c = origin[1] + dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _find_hit_arrow(
    target_cells: set[tuple[int, int]],
    step: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
) -> dict[str, object] | None:
    gaps = list(range(HIT_GAP_MIN, max(shape) + 1))
    random.shuffle(gaps)
    for hit in _front_cells(target_cells, step):
        hr, hc = hit
        for gap in gaps:
            origin = (hr - step[0] * (gap + 1), hc - step[1] * (gap + 1))
            arrow_cells = _arrow_cells(origin, step, style="diag")
            if not _object_fits(arrow_cells, shape, occupied, reserved_rays):
                continue
            ray_cells = [(origin[0] + step[0] * k, origin[1] + step[1] * k) for k in range(1, gap + 1)]
            if any(not _in_bounds(shape, r, c) for r, c in ray_cells):
                continue
            if any((r, c) in occupied or (r, c) in reserved_rays for r, c in ray_cells):
                continue
            return {
                "origin": origin,
                "direction": step,
                "cells": arrow_cells,
                "ray_cells": ray_cells,
            }
    return None


def _find_miss_arrow(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    directions = DIRS4[:]
    random.shuffle(directions)
    for step in directions:
        for _ in range(ARROW_PLACE_TRIES):
            origin = (random.randrange(h), random.randrange(w))
            arrow_cells = _arrow_cells(origin, step, style="diag")
            if not _object_fits(arrow_cells, shape, occupied, reserved_rays):
                continue
            ray_cells = _ray_to_edge(origin, step, shape)
            if len(ray_cells) < MISS_RAY_MIN:
                continue
            if any((r, c) in occupied or (r, c) in reserved_rays for r, c in ray_cells):
                continue
            return {
                "origin": origin,
                "direction": step,
                "cells": arrow_cells,
                "ray_cells": ray_cells,
            }
    return None


def _paint(grid: np.ndarray, cells: set[tuple[int, int]] | list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    arrow_count = random.randint(*ARROW_COUNT_RANGE)
    if arrow_count == 1:
        hit_flags = [random.random() < 0.5]
    else:
        hit_flags = [random.random() < 0.5 for _ in range(arrow_count)]
        if (all(hit_flags) or not any(hit_flags)) and random.random() < 0.75:
            hit_flags[random.randrange(arrow_count)] = not hit_flags[0]
    hit_count = sum(hit_flags)

    target_count = random.randint(max(TARGET_COUNT_RANGE[0], hit_count), TARGET_COUNT_RANGE[1])
    if target_count + arrow_count > len(COLORS):
        return None

    palette = random.sample(COLORS, target_count + arrow_count)
    target_colors = palette[:target_count]
    arrow_colors = palette[target_count:]

    grid = np.zeros(shape, dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved_rays: set[tuple[int, int]] = set()
    target_cells_list: list[set[tuple[int, int]]] = []

    for color in target_colors:
        placed = False
        patch = _make_target_patch()
        if patch is None:
            return None
        local_h = max(r for r, _ in patch) + 1
        local_w = max(c for _, c in patch) + 1
        for _ in range(TARGET_PLACE_TRIES):
            r0 = random.randint(1, h - local_h - 1)
            c0 = random.randint(1, w - local_w - 1)
            abs_cells = _translate(patch, r0, c0)
            if not _object_fits(abs_cells, shape, occupied, reserved_rays):
                continue
            _paint(grid, abs_cells, color)
            occupied.update(abs_cells)
            target_cells_list.append(abs_cells)
            placed = True
            break
        if not placed:
            return None

    assigned_targets = random.sample(range(target_count), hit_count) if hit_count else []
    hit_target_iter = iter(assigned_targets)

    for color, should_hit in zip(arrow_colors, hit_flags):
        placement: dict[str, object] | None = None
        if should_hit:
            target_id = next(hit_target_iter)
            directions = DIRS4[:]
            random.shuffle(directions)
            for step in directions:
                placement = _find_hit_arrow(
                    target_cells_list[target_id],
                    step,
                    shape,
                    occupied,
                    reserved_rays,
                )
                if placement is not None:
                    break
        else:
            placement = _find_miss_arrow(shape, occupied, reserved_rays)

        if placement is None:
            return None

        arrow_cells = placement["cells"]
        ray_cells = placement["ray_cells"]
        _paint(grid, arrow_cells, color)
        occupied.update(arrow_cells)
        reserved_rays.update(ray_cells)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_detect_arrows(grid)) != arrow_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
