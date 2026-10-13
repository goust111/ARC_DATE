import random
from collections import Counter

import numpy as np


DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

CORNERS = ("TL", "TR", "BL", "BR")
L_OFFSETS = {
    "TL": ((0, 1), (1, 0), (1, 1)),
    "TR": ((0, 0), (1, 0), (1, 1)),
    "BL": ((0, 0), (0, 1), (1, 1)),
    "BR": ((0, 0), (0, 1), (1, 0)),
}
MISSING_TO_CORNER = {
    (0, 0): "TL",
    (0, 1): "TR",
    (1, 0): "BL",
    (1, 1): "BR",
}

BG_COLOR = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
MAX_MARKER_DISTANCE = 5
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR8:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _min_chebyshev(pos: tuple[int, int], cells: list[tuple[int, int]]) -> int:
    return min(_chebyshev(pos, cell) for cell in cells)


def _l_corner(cells: list[tuple[int, int]]) -> str | None:
    if len(cells) != 3:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if max(rows) - min(rows) != 1 or max(cols) - min(cols) != 1:
        return None
    r0 = min(rows)
    c0 = min(cols)
    rel = {(r - r0, c - c0) for r, c in cells}
    full = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing = full - rel
    if len(missing) != 1:
        return None
    return MISSING_TO_CORNER[next(iter(missing))]


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue
        comps = _components8(grid == color)
        singleton_positions = [comp[0] for comp in comps if len(comp) == 1]
        for comp in comps:
            corner = _l_corner(comp)
            if corner is None:
                continue
            marker_count = sum(
                1 for pos in singleton_positions if _min_chebyshev(pos, comp) <= MAX_MARKER_DISTANCE
            )
            objects.append(
                {
                    "color": color,
                    "corner": corner,
                    "markers": marker_count,
                    "cells": comp,
                }
            )
    return objects


def _paint_corner_triangle(out: np.ndarray, corner: str, side: int, color: int) -> None:
    h, w = out.shape
    side = max(0, int(side))
    if corner == "TL":
        for dr in range(min(h, side)):
            width = min(w, side - dr)
            if width > 0:
                out[dr, :width] = color
    elif corner == "TR":
        for dr in range(min(h, side)):
            width = min(w, side - dr)
            if width > 0:
                out[dr, w - width : w] = color
    elif corner == "BL":
        for dr in range(min(h, side)):
            width = min(w, side - dr)
            if width > 0:
                out[h - 1 - dr, :width] = color
    else:
        for dr in range(min(h, side)):
            width = min(w, side - dr)
            if width > 0:
                out[h - 1 - dr, w - width : w] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = np.full_like(g, bg)
    for obj in _extract_objects(g, bg):
        _paint_corner_triangle(
            out,
            str(obj["corner"]),
            int(obj["markers"]) + 2,
            int(obj["color"]),
        )
    return out


def _triangle_cells(shape: tuple[int, int], corner: str, side: int) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    if corner == "TL":
        for dr in range(min(h, side)):
            for dc in range(min(w, side - dr)):
                cells.add((dr, dc))
    elif corner == "TR":
        for dr in range(min(h, side)):
            width = min(w, side - dr)
            for dc in range(width):
                cells.add((dr, w - width + dc))
    elif corner == "BL":
        for dr in range(min(h, side)):
            row = h - 1 - dr
            for dc in range(min(w, side - dr)):
                cells.add((row, dc))
    else:
        for dr in range(min(h, side)):
            row = h - 1 - dr
            width = min(w, side - dr)
            for dc in range(width):
                cells.add((row, w - width + dc))
    return cells


def _triangles_disjoint(shape: tuple[int, int], specs: list[tuple[str, int]]) -> bool:
    occupied: set[tuple[int, int]] = set()
    for corner, side in specs:
        cells = _triangle_cells(shape, corner, side)
        if occupied & cells:
            return False
        occupied |= cells
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _pick_marker_slots(
    slots: list[tuple[int, int]], marker_count: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    for _ in range(24):
        order = slots[:]
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for pos in order:
            if all(_chebyshev(pos, other) >= 2 for other in chosen):
                chosen.append(pos)
                if len(chosen) == marker_count:
                    return chosen
    return None


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    corner: str,
    color: int,
    marker_count: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    anchors = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    rng.shuffle(anchors)
    for r0, c0 in anchors:
        l_cells = [(r0 + dr, c0 + dc) for dr, dc in L_OFFSETS[corner]]
        if any(blocked[r, c] for r, c in l_cells):
            continue

        candidate_slots: list[tuple[int, int]] = []
        rr0 = max(0, r0 - MAX_MARKER_DISTANCE)
        rr1 = min(h - 1, r0 + 1 + MAX_MARKER_DISTANCE)
        cc0 = max(0, c0 - MAX_MARKER_DISTANCE)
        cc1 = min(w - 1, c0 + 1 + MAX_MARKER_DISTANCE)
        for r in range(rr0, rr1 + 1):
            for c in range(cc0, cc1 + 1):
                if blocked[r, c] or (r, c) in l_cells:
                    continue
                dist = _min_chebyshev((r, c), l_cells)
                if 2 <= dist <= MAX_MARKER_DISTANCE:
                    candidate_slots.append((r, c))

        markers = _pick_marker_slots(candidate_slots, marker_count, rng)
        if markers is None:
            continue

        all_cells = l_cells + markers
        for r, c in all_cells:
            grid[r, c] = color
        _mark_blocked(blocked, all_cells)
        return True
    return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(1, 4)
    corners = rng.sample(CORNERS, n_objects)
    colors = rng.sample(PALETTE, n_objects)
    marker_counts = [rng.randint(1, 4) for _ in range(n_objects)]
    sides = [count + 2 for count in marker_counts]

    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    specs = list(zip(corners, sides))
    if not _triangles_disjoint((h, w), specs):
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    plan = list(zip(corners, colors, marker_counts))
    rng.shuffle(plan)
    plan.sort(key=lambda item: -item[2])

    for corner, color, marker_count in plan:
        if not _place_object(grid, blocked, corner, color, marker_count, rng):
            return None

    objects = _extract_objects(grid, BG_COLOR)
    if len(objects) != n_objects:
        return None

    by_color = {int(obj["color"]): obj for obj in objects}
    if len(by_color) != n_objects:
        return None

    for corner, color, marker_count in plan:
        obj = by_color.get(color)
        if obj is None:
            return None
        if str(obj["corner"]) != corner or int(obj["markers"]) != marker_count:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
