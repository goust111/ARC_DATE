from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
DOT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SIZE_MIN = 2
SIZE_MAX = 6
GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
OBJECT_COUNT_CHOICES = (2, 3, 4)
MAX_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _square_bbox(corner: tuple[int, int], direction: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    cr, cc = corner
    dr, dc = direction
    if dr > 0:
        r0, r1 = cr, cr + size - 1
    else:
        r0, r1 = cr - size + 1, cr
    if dc > 0:
        c0, c1 = cc, cc + size - 1
    else:
        c0, c1 = cc - size + 1, cc
    return r0, c0, r1, c1


def _bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _expand8(cells: Iterable[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
        out.add((r, c))
    return out


def _line_info(coords: list[tuple[int, int]], shape: tuple[int, int]) -> dict:
    cells = set(coords)
    endpoints = []
    for r, c in cells:
        degree = sum((nr, nc) in cells for nr, nc in _neighbors8(r, c))
        if degree == 1:
            endpoints.append((r, c))
        elif degree != 2:
            raise AssertionError("grey component must be a simple diagonal segment")

    assert len(endpoints) == 2, "grey component must have exactly two endpoints"
    border_endpoints = [pt for pt in endpoints if _is_border(shape, pt[0], pt[1])]
    assert len(border_endpoints) == 1, "each grey segment must start from exactly one border endpoint"

    outer = border_endpoints[0]
    inner = endpoints[1] if endpoints[0] == outer else endpoints[0]
    dr = int(np.sign(inner[0] - outer[0]))
    dc = int(np.sign(inner[1] - outer[1]))
    assert abs(dr) == 1 and abs(dc) == 1, "grey segment must be diagonal"
    size = len(cells)
    expected = {(outer[0] + i * dr, outer[1] + i * dc) for i in range(size)}
    assert expected == cells, "grey component must be a straight 45-degree segment"
    return {
        "outer": outer,
        "inner": inner,
        "direction": (dr, dc),
        "size": size,
        "line_cells": tuple(sorted(cells)),
    }


def _extract_tasks(grid: np.ndarray) -> list[dict]:
    grey_components = _component_coords(grid == GREY)
    assert grey_components, "expected at least one grey diagonal segment"

    tasks: list[dict] = []
    claimed_square_cells: set[tuple[int, int]] = set()
    for comp in grey_components:
        info = _line_info(comp, grid.shape)
        dr, dc = info["direction"]
        corner = (info["inner"][0] + dr, info["inner"][1] + dc)
        bbox = _square_bbox(corner, info["direction"], info["size"])
        r0, c0, r1, c1 = bbox
        assert _in_bounds(grid.shape, r0, c0) and _in_bounds(grid.shape, r1, c1), "square must stay in bounds"
        square_cells = set(_bbox_cells(bbox))
        assert not square_cells & claimed_square_cells, "generated squares must not overlap"
        claimed_square_cells |= square_cells

        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        dot_local = np.argwhere((region != BG) & (region != GREY))
        assert dot_local.shape[0] == 1, "each generated square must contain exactly one color seed dot"
        rr, cc = map(int, dot_local[0])
        dot_r, dot_c = r0 + rr, c0 + cc
        tasks.append(
            {
                "bbox": bbox,
                "color": int(grid[dot_r, dot_c]),
                "direction": info["direction"],
                "size": info["size"],
                "line_cells": info["line_cells"],
                "dot": (dot_r, dot_c),
                "square_cells": tuple(sorted(square_cells)),
            }
        )

    return tasks


def _render_output(shape: tuple[int, int], tasks: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for task in tasks:
        r0, c0, r1, c1 = task["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(task["color"])
    for task in tasks:
        for r, c in task["line_cells"]:
            out[r, c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a color square past each grey diagonal ruler using the ruler's length and direction.

    Input structure:
    - The grid has a black background with several grey 45-degree line segments.
    - Each grey segment starts on the grid border and points inward along one diagonal direction.
    - Colored singleton dots are scattered on the canvas; each target square region contains exactly one such dot.

    Rule:
    - For each grey segment, let L be its pixel length and let d be the direction from its border endpoint to its inner endpoint.
    - Step one more cell from the inner endpoint along d; that cell is the square corner touching the segment, and the square extends in the same diagonal quadrant as d.
    - Fill the resulting LxL square with the color of the unique dot inside that square region, keep the grey segment unchanged, and erase every dot not covered by any square.

    Key insight:
    - The grey segment acts as both a ruler and a pointer: its length gives the square size, its inward diagonal fixes which corner is attached, and only that derived square reveals which dot supplies the color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    tasks = _extract_tasks(g)
    return _render_output(g.shape, tasks)


def _enumerate_candidates(h: int, w: int, direction: tuple[int, int]) -> list[dict]:
    border_cells = {(0, c) for c in range(w)}
    border_cells |= {(h - 1, c) for c in range(w)}
    border_cells |= {(r, 0) for r in range(h)}
    border_cells |= {(r, w - 1) for r in range(h)}

    candidates: list[dict] = []
    seen_keys: set[tuple] = set()
    dr, dc = direction
    for size in range(SIZE_MIN, min(SIZE_MAX, min(h, w) // 2) + 1):
        for outer in border_cells:
            line = [(outer[0] + i * dr, outer[1] + i * dc) for i in range(size)]
            if not all(_in_bounds((h, w), r, c) for r, c in line):
                continue
            inner = line[-1]
            if _is_border((h, w), inner[0], inner[1]):
                continue
            corner = (inner[0] + dr, inner[1] + dc)
            if not _in_bounds((h, w), corner[0], corner[1]):
                continue
            bbox = _square_bbox(corner, direction, size)
            r0, c0, r1, c1 = bbox
            if not (_in_bounds((h, w), r0, c0) and _in_bounds((h, w), r1, c1)):
                continue
            square = tuple(_bbox_cells(bbox))
            footprint = set(line) | set(square)
            if len(footprint) != len(line) + len(square):
                continue
            key = (tuple(line), bbox)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "direction": direction,
                    "size": size,
                    "line_cells": tuple(line),
                    "bbox": bbox,
                    "square_cells": square,
                    "footprint": footprint,
                    "buffer": _expand8(footprint, (h, w)),
                }
            )
    return candidates


def _sample_directions(rng: random.Random, count: int) -> list[tuple[int, int]]:
    if count == 4:
        dirs = list(DIRECTIONS)
        rng.shuffle(dirs)
        return dirs
    if rng.random() < 0.7:
        dirs = rng.sample(list(DIRECTIONS), count)
    else:
        dirs = [rng.choice(DIRECTIONS) for _ in range(count)]
    rng.shuffle(dirs)
    return dirs


def _near_square_candidates(objects: list[dict], shape: tuple[int, int], blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    near: set[tuple[int, int]] = set()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        for r in range(max(0, r0 - 1), min(h, r1 + 2)):
            for c in range(max(0, c0 - 1), min(w, c1 + 2)):
                if (r, c) in blocked:
                    continue
                if r0 <= r <= r1 and c0 <= c <= c1:
                    continue
                near.add((r, c))
    return list(near)


def _free_dot_candidates(shape: tuple[int, int], blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    return [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]


def _validate_generated_pair(inp: np.ndarray, out: np.ndarray, objects: list[dict], extra_dots: list[tuple[int, int]]) -> None:
    solved = main(inp)
    assert np.array_equal(solved, out), "generated sample does not match main()"

    square_union = set()
    for obj in objects:
        square_cells = set(obj["square_cells"])
        square_union |= square_cells
        dot = obj["dot"]
        assert dot in square_cells, "seed dot must lie inside its own square"
        region_colors = inp[[r for r, _ in obj["square_cells"]], [c for _, c in obj["square_cells"]]]
        assert np.count_nonzero((region_colors != BG) & (region_colors != GREY)) == 1
        r0, c0, r1, c1 = obj["bbox"]
        assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == obj["color"]), "output square must be solid"

    for dot in extra_dots:
        assert dot not in square_union, "extra dots must stay outside all target squares"
        assert out[dot] == BG, "uncovered dots must be erased"

    assert len(_component_coords(inp == GREY)) == len(objects), "grey segments must stay separable"


def _attempt_generate(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    object_count = rng.choice(OBJECT_COUNT_CHOICES)
    directions = _sample_directions(rng, object_count)

    occupied_buffer: set[tuple[int, int]] = set()
    objects: list[dict] = []
    for direction in directions:
        candidates = _enumerate_candidates(h, w, direction)
        rng.shuffle(candidates)
        placed = False
        for cand in candidates:
            if cand["footprint"] & occupied_buffer:
                continue
            objects.append(dict(cand))
            occupied_buffer |= cand["buffer"]
            placed = True
            break
        if not placed:
            return None

    seed_colors = rng.sample(list(DOT_COLORS), len(objects))
    dot_blocked = {cell for obj in objects for cell in obj["line_cells"]}
    dot_blocked |= {cell for obj in objects for cell in _expand8(obj["line_cells"], (h, w))}

    for obj, color in zip(objects, seed_colors):
        dot_candidates = [cell for cell in obj["square_cells"]]
        rng.shuffle(dot_candidates)
        dot = dot_candidates[0]
        obj["dot"] = dot
        obj["color"] = int(color)

    seed_and_footprint = {cell for obj in objects for cell in obj["footprint"]}
    dot_taken = {obj["dot"] for obj in objects}
    extra_blocked = set(seed_and_footprint)
    extra_blocked |= _expand8(dot_taken, (h, w))
    extra_blocked |= _expand8({cell for obj in objects for cell in obj["line_cells"]}, (h, w))

    near_candidates = _near_square_candidates(objects, (h, w), extra_blocked)
    rng.shuffle(near_candidates)
    extra_dots: list[tuple[int, int]] = []
    target_extra = rng.randint(max(1, len(objects) - 1), len(objects) + 2)

    while near_candidates and len(extra_dots) < min(len(objects), target_extra):
        cell = near_candidates.pop()
        if cell in extra_blocked:
            continue
        extra_dots.append(cell)
        extra_blocked |= _expand8([cell], (h, w))

    free_candidates = _free_dot_candidates((h, w), extra_blocked)
    rng.shuffle(free_candidates)
    while free_candidates and len(extra_dots) < target_extra:
        cell = free_candidates.pop()
        if cell in extra_blocked:
            continue
        extra_dots.append(cell)
        extra_blocked |= _expand8([cell], (h, w))

    if not extra_dots:
        return None

    inp = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c in obj["line_cells"]:
            inp[r, c] = GREY
        dr, dc = obj["dot"]
        inp[dr, dc] = obj["color"]

    distractor_palette = list(DOT_COLORS)
    for cell in extra_dots:
        inp[cell] = rng.choice(distractor_palette)

    out = _render_output((h, w), objects)
    _validate_generated_pair(inp, out, objects, extra_dots)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        seed = random.randrange(1 << 30)
        pair = _attempt_generate(random.Random(seed))
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid diagonal-ruler sample")
