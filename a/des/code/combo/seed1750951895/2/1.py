from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_SIZE_RANGE = (14, 22)
OBJECT_COUNT_RANGE = (2, 5)
AREA_CHOICES = list(range(2, 13))
GEN_RETRIES = 400


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            objects.append({"color": color, "cells": cells, "area": len(cells)})
    return objects


def _neighbor_color_count(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> int:
    h, w = grid.shape
    neighbors = set()
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                value = int(grid[nr, nc])
                if value not in (0, color):
                    neighbors.add(value)
    return len(neighbors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort foreground objects by area and redraw them as nested rectangular rings.

    Each object's thickness equals the number of distinct other foreground colors
    it touches in the input under 4-neighborhood adjacency, with a minimum of 1.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"

    objects = _components(g)
    assert objects, "expected at least one foreground object"

    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas), "expected distinct object areas"

    objects.sort(key=lambda obj: -obj["area"])
    thicknesses = [max(1, _neighbor_color_count(g, obj["cells"], obj["color"])) for obj in objects]
    side = 2 * sum(thicknesses) - 1
    out = np.zeros((side, side), dtype=int)

    offset = 0
    for obj, thickness in zip(objects[:-1], thicknesses[:-1], strict=False):
        color = int(obj["color"])
        lo = offset
        hi = side - offset
        out[lo : lo + thickness, lo:hi] = color
        out[hi - thickness : hi, lo:hi] = color
        out[lo:hi, lo : lo + thickness] = color
        out[lo:hi, hi - thickness : hi] = color
        offset += thickness

    out[offset : side - offset, offset : side - offset] = int(objects[-1]["color"])
    return out


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, area + 1):
        if area % h == 0:
            w = area // h
            if h <= 8 and w <= 12:
                pairs.append((h, w))
    return pairs


def _choose_shapes(n: int) -> list[tuple[int, int]]:
    areas = random.sample(AREA_CHOICES, n)
    random.shuffle(areas)
    shapes = []
    for area in areas:
        pairs = _factor_pairs(area)
        assert pairs
        pairs.sort(key=lambda hw: (max(hw), abs(hw[0] - hw[1])))
        top = pairs[: min(4, len(pairs))]
        shapes.append(random.choice(top))
    return shapes


def _contact_colors(grid: np.ndarray, r0: int, c0: int, h0: int, w0: int) -> set[int]:
    h, w = grid.shape
    contacts = set()
    for r in range(r0, r0 + h0):
        for c in range(c0, c0 + w0):
            if int(grid[r, c]) != 0:
                return {-1}
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    value = int(grid[nr, nc])
                    if value != 0:
                        contacts.add(value)
    return contacts


def _place_rect(grid: np.ndarray, color: int, shape: tuple[int, int], required: set[int]) -> bool:
    gh, gw = grid.shape
    h0, w0 = shape
    candidates = []
    for r0 in range(gh - h0 + 1):
        for c0 in range(gw - w0 + 1):
            contacts = _contact_colors(grid, r0, c0, h0, w0)
            if contacts == required:
                candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    grid[r0 : r0 + h0, c0 : c0 + w0] = color
    return True


def _build_parent_list(n: int) -> list[int | None]:
    mode = random.choice(["chain", "star", "forest"])
    parents: list[int | None] = [None]
    if mode == "chain":
        for i in range(1, n):
            parents.append(i - 1)
    elif mode == "star":
        for i in range(1, n):
            parents.append(0)
    else:
        for i in range(1, n):
            if random.random() < 0.35:
                parents.append(None)
            else:
                parents.append(random.randrange(i))
        if all(p is None for p in parents[1:]):
            parents[random.randrange(1, n)] = 0
    return parents


def _degree_profile(parents: list[int | None]) -> list[int]:
    deg = [0] * len(parents)
    for i, p in enumerate(parents):
        if i == 0 or p is None:
            continue
        deg[i] += 1
        deg[p] += 1
    return deg


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, n)
    shapes = _choose_shapes(n)
    parents = _build_parent_list(n)
    degrees = _degree_profile(parents)

    gh = random.randint(*GRID_SIZE_RANGE)
    gw = random.randint(*GRID_SIZE_RANGE)
    grid = np.zeros((gh, gw), dtype=int)

    if not _place_rect(grid, colors[0], shapes[0], set()):
        return None

    for idx in range(1, n):
        parent = parents[idx]
        required = set() if parent is None else {colors[parent]}
        if not _place_rect(grid, colors[idx], shapes[idx], required):
            return None

    objs = _components(grid)
    if len(objs) != n:
        return None

    actual_areas = sorted((obj["area"] for obj in objs), reverse=True)
    if len(set(actual_areas)) != n:
        return None

    color_to_k = {obj["color"]: _neighbor_color_count(grid, obj["cells"], obj["color"]) for obj in objs}
    actual_degrees = sorted(color_to_k.values())
    expected_degrees = sorted(degrees)
    if actual_degrees != expected_degrees:
        return None

    if max(color_to_k.values()) == 0 or min(color_to_k.values()) == max(color_to_k.values()):
        if random.random() < 0.6:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape[0] != out.shape[1] or out.shape[0] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
