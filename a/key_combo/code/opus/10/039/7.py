from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

GENERATION_ATTEMPTS = 400


def _pinwheel(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    top = np.hstack((g, np.rot90(g, 1)))
    bottom = np.hstack((np.rot90(g, -1), np.rot90(g, 2)))
    return np.vstack((top, bottom))


def _collision_info(pinwheel: np.ndarray) -> dict[str, object]:
    out = np.asarray(pinwheel, dtype=int)
    side = int(out.shape[0] // 2)
    levels = np.zeros_like(out, dtype=int)
    ninety_cells: set[tuple[int, int]] = set()
    one_eighty_cells: set[tuple[int, int]] = set()
    ninety_pairs = 0
    one_eighty_pairs = 0

    for r in range(2 * side):
        a = (r, side - 1)
        b = (r, side)
        if int(out[a]) == GREEN and int(out[b]) == GREEN:
            ninety_pairs += 1
            ninety_cells.update((a, b))
            levels[a] = max(levels[a], 1)
            levels[b] = max(levels[b], 1)

    for c in range(2 * side):
        a = (side - 1, c)
        b = (side, c)
        if int(out[a]) == GREEN and int(out[b]) == GREEN:
            ninety_pairs += 1
            ninety_cells.update((a, b))
            levels[a] = max(levels[a], 1)
            levels[b] = max(levels[b], 1)

    # Opposite quadrants can only meet at the central 2x2 saddle.
    for a, b in (((side - 1, side - 1), (side, side)), ((side - 1, side), (side, side - 1))):
        if int(out[a]) == GREEN and int(out[b]) == GREEN:
            one_eighty_pairs += 1
            one_eighty_cells.update((a, b))
            levels[a] = 2
            levels[b] = 2

    return {
        "levels": levels,
        "ninety_pairs": ninety_pairs,
        "one_eighty_pairs": one_eighty_pairs,
        "priority_overlap_cells": len(ninety_cells & one_eighty_cells),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Arrange the input into a four-rotation pinwheel and recolor emergent boundary collisions.

    Input structure:
    - A square black-background grid whose only foreground color is green.
    - The green pixels form one asymmetric 4-connected sprite placed near an edge or corner.

    Rule:
    - Build a 2x2 pinwheel: original in the upper-left, 90-degree counterclockwise in the upper-right, 90-degree clockwise in the lower-left, and 180-degree rotation in the lower-right.
    - Scan the vertical and horizontal center seams; whenever opposite seam cells are 4-neighbor green pixels from different quadrants, mark them as 90-degree collisions and recolor them red.
    - At the central 2x2 meeting point, diagonally opposite occupied cells from opposite quadrants count as 180-degree collisions and are recolored yellow.
    - Non-collision foreground stays green, background stays black, and yellow overrides red if a center pixel belongs to both collision types.

    Key insight:
    - The crossing signal is not drawn explicitly in the input: it appears only after the four rotated copies are assembled, and the recolor depends on which rotated quadrants meet at the seam.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h > 0, "expected a non-empty square grid"

    out = _pinwheel(g)
    info = _collision_info(out)
    levels = info["levels"]

    result = out.copy()
    result[(result == GREEN) & (levels == 1)] = RED
    result[(result == GREEN) & (levels == 2)] = YELLOW
    return result


def _neighbors4(cell: tuple[int, int], n: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            out.append((nr, nc))
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        cell = q.popleft()
        for nxt in _neighbors4(cell, mask.shape[0]):
            if bool(mask[nxt]) and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == int(mask.sum())


def _grow_connected(
    n: int,
    seed_cells: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    target: int,
) -> set[tuple[int, int]] | None:
    cells = set(seed_cells)
    while len(cells) < target:
        frontier = sorted(
            {
                nxt
                for cell in cells
                for nxt in _neighbors4(cell, n)
                if nxt in allowed and nxt not in cells
            }
        )
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _corner_box_cells(n: int, corner: str) -> set[tuple[int, int]]:
    span_r = random.randint(3, min(4, n))
    span_c = random.randint(3, min(4, n))
    if corner == "tr":
        rows = range(span_r)
        cols = range(n - span_c, n)
    elif corner == "bl":
        rows = range(n - span_r, n)
        cols = range(span_c)
    else:
        raise AssertionError(corner)
    return {(r, c) for r in rows for c in cols}


def _make_corner_mode(n: int, corner: str) -> np.ndarray | None:
    anchor = (0, n - 1) if corner == "tr" else (n - 1, 0)
    allowed = _corner_box_cells(n, corner)
    target = random.randint(3, min(8, len(allowed)))
    cells = _grow_connected(n, {anchor}, allowed, target)
    if cells is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    for cell in cells:
        mask[cell] = True
    return mask


def _make_edge_line_mode(n: int, edge: str) -> np.ndarray | None:
    if edge == "top":
        cells = {(0, c) for c in range(n)}
        inward = {(1, c) for c in range(n)}
    elif edge == "bottom":
        cells = {(n - 1, c) for c in range(n)}
        inward = {(n - 2, c) for c in range(n)}
    elif edge == "left":
        cells = {(r, 0) for r in range(n)}
        inward = {(r, 1) for r in range(n)}
    elif edge == "right":
        cells = {(r, n - 1) for r in range(n)}
        inward = {(r, n - 2) for r in range(n)}
    else:
        raise AssertionError(edge)

    extras_max = min(8 - len(cells), len(inward))
    if extras_max > 0:
        for cell in random.sample(sorted(inward), random.randint(0, extras_max)):
            cells.add(cell)

    mask = np.zeros((n, n), dtype=bool)
    for cell in cells:
        mask[cell] = True
    return mask


def _make_priority_mode() -> np.ndarray:
    n = 4
    steps = ["D"] * (n - 1) + ["R"] * (n - 1)
    random.shuffle(steps)
    r = c = 0
    cells = {(r, c)}
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        cells.add((r, c))

    frontier = sorted(
        {
            nxt
            for cell in cells
            for nxt in _neighbors4(cell, n)
            if nxt not in cells and nxt not in {(0, n - 1), (n - 1, 0)}
        }
    )
    if frontier and random.random() < 0.6:
        cells.add(random.choice(frontier))

    mask = np.zeros((n, n), dtype=bool)
    for cell in cells:
        mask[cell] = True
    return mask


def _build_grid(mask: np.ndarray) -> np.ndarray:
    grid = np.zeros(mask.shape, dtype=int)
    grid[mask] = GREEN
    return grid


def _valid_sprite(mask: np.ndarray) -> bool:
    n = int(mask.shape[0])
    if n < 4 or n > 6:
        return False
    count = int(mask.sum())
    if count < 3 or count > 8:
        return False
    if np.array_equal(mask, np.rot90(mask, 1)):
        return False
    return _connected(mask)


def _semantic_ok(grid: np.ndarray, mode: str) -> bool:
    mask = grid == GREEN
    if not _valid_sprite(mask):
        return False

    info = _collision_info(_pinwheel(grid))
    ninety_pairs = int(info["ninety_pairs"])
    one_eighty_pairs = int(info["one_eighty_pairs"])
    overlap_cells = int(info["priority_overlap_cells"])
    out = main(grid)

    if ninety_pairs < 1:
        return False
    if mode == "corner90":
        if one_eighty_pairs != 0:
            return False
        if not np.any(out == RED) or np.any(out == YELLOW):
            return False
    elif mode == "edge180":
        if one_eighty_pairs < 1:
            return False
        if not np.any(out == RED) or not np.any(out == YELLOW):
            return False
    elif mode == "priority":
        if one_eighty_pairs < 2 or overlap_cells < 1:
            return False
        center = out.shape[0] // 2
        center_block = out[center - 1 : center + 1, center - 1 : center + 1]
        if np.count_nonzero(center_block == YELLOW) < 2:
            return False
    else:
        raise AssertionError(mode)

    if not np.array_equal(main(grid), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choices(
        ["corner90", "edge180", "priority"],
        weights=[4, 4, 3],
        k=1,
    )[0]

    if mode == "priority":
        mask = _make_priority_mode()
    elif mode == "corner90":
        n = random.randint(4, 6)
        mask = _make_corner_mode(n, random.choice(["tr", "bl"]))
    else:
        n = random.randint(4, 6)
        mask = _make_edge_line_mode(n, random.choice(["top", "bottom", "left", "right"]))

    if mask is None or not _valid_sprite(mask):
        return None

    grid = _build_grid(mask)
    if not _semantic_ok(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
