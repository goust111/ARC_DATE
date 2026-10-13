from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RING = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
CORNERS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 400
PLACE_RETRIES = 200
GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 28)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*points)
    return min(rows), min(cols), max(rows), max(cols)


def _ring_points(top: int, left: int) -> set[tuple[int, int]]:
    pts = set()
    for r in range(top, top + 5):
        pts.add((r, left))
        pts.add((r, left + 4))
    for c in range(left, left + 5):
        pts.add((top, c))
        pts.add((top + 4, c))
    return pts


def _full_box_points(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _pointer_cells(top: int, left: int, missing: str) -> set[tuple[int, int]]:
    cells = _full_box_points(top, left, 2, 2)
    corner_map = {
        "tl": (top, left),
        "tr": (top, left + 1),
        "bl": (top + 1, left),
        "br": (top + 1, left + 1),
    }
    cells.remove(corner_map[missing])
    return cells


def _missing_corner(points: Iterable[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 != 1 or c1 - c0 != 1:
        raise ValueError("pointer bbox must be 2x2")
    pset = set(points)
    corners = {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }
    missing = [name for name, pos in corners.items() if pos not in pset]
    if len(missing) != 1 or len(pset) != 3:
        raise ValueError("pointer must be a 2x2 block with one missing corner")
    return missing[0]


def _ring_anchor(top: int, left: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return top - 1, left + offset
    if side == "bottom":
        return top + 5, left + offset
    if side == "left":
        return top + offset, left - 1
    if side == "right":
        return top + offset, left + 5
    raise ValueError(f"unknown side: {side}")


def _adjacent_side_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0 - 1, c))
        cells.append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        cells.append((r, c0 - 1))
        cells.append((r, c1 + 1))
    return cells


def _extract_scene(grid: np.ndarray) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, list[tuple[int, int]]], dict[int, str]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    h, w = g.shape

    rings: dict[int, tuple[int, int, int, int]] = {}
    anchors: dict[int, tuple[int, int]] = {}

    for comp in _components(g == RING):
        if len(comp) != 16:
            raise ValueError("ring must contain 16 grey cells")
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0, c1 - c0) != (4, 4):
            raise ValueError("ring bbox must be 5x5")
        if set(comp) != _ring_points(r0, c0):
            raise ValueError("grey component is not a hollow 5x5 ring")

        anchor_candidates = []
        for ar, ac in _adjacent_side_cells((r0, c0, r1, c1)):
            if 0 <= ar < h and 0 <= ac < w:
                color = int(g[ar, ac])
                if color not in (BG, RING):
                    anchor_candidates.append((ar, ac, color))
        if len(anchor_candidates) != 1:
            raise ValueError("each ring must have exactly one adjacent colored anchor")
        ar, ac, color = anchor_candidates[0]
        if color in rings:
            raise ValueError("anchor colors must be unique")
        rings[color] = (r0, c0, r1, c1)
        anchors[color] = (ar, ac)

    if not rings:
        raise ValueError("expected at least one ring")

    pointers: dict[int, list[tuple[int, int]]] = {}
    directions: dict[int, str] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (BG, RING):
            continue
        comps = _components(g == color)
        if len(comps) != 2:
            raise ValueError("each pair color must form one anchor and one pointer")

        singleton = [comp for comp in comps if len(comp) == 1]
        triomino = [comp for comp in comps if len(comp) == 3]
        if len(singleton) != 1 or len(triomino) != 1:
            raise ValueError("expected one singleton anchor and one 3-cell pointer")
        if color not in anchors or singleton[0][0] != anchors[color]:
            raise ValueError("singleton must be the ring anchor of the same color")

        direction = _missing_corner(triomino[0])
        pointers[color] = triomino[0]
        directions[color] = direction

    if set(pointers) != set(rings):
        raise ValueError("ring and pointer colors must match one-to-one")
    return rings, pointers, directions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each colored pointer with the ring that has the same-colored anchor.
    The missing corner of the 3-cell 2x2 pointer indicates which ring corner
    should be recolored, then the pointer sprite is erased.
    """
    g = np.asarray(grid, dtype=int)
    rings, pointers, directions = _extract_scene(g)
    out = g.copy()

    for color, bbox in rings.items():
        r0, c0, r1, c1 = bbox
        corner_map = {
            "tl": (r0, c0),
            "tr": (r0, c1),
            "bl": (r1, c0),
            "br": (r1, c1),
        }
        rr, cc = corner_map[directions[color]]
        out[rr, cc] = color
        for pr, pc in pointers[color]:
            out[pr, pc] = BG
    return out


def _padded_ok(blocked: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if not (0 <= rr < h and 0 <= cc < w):
                    return False
                if blocked[rr, cc]:
                    return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                blocked[rr, cc] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    max_pairs = min(len(PAIR_COLORS), max(2, (h * w) // 95))
    min_pairs = 1 if max_pairs == 1 else 2
    pair_count = rng.randint(min_pairs, max_pairs)
    colors = rng.sample(PAIR_COLORS, pair_count)

    ring_specs: dict[int, tuple[int, int, str, int]] = {}
    for color in colors:
        placed = False
        for _ in range(PLACE_RETRIES):
            top = rng.randint(2, h - 7)
            left = rng.randint(2, w - 7)
            side = rng.choice(SIDES)
            offset = rng.choice((1, 2, 3))
            anchor = _ring_anchor(top, left, side, offset)
            reserved = _full_box_points(top, left, 5, 5) | {anchor}
            if not _padded_ok(blocked, reserved, pad=1):
                continue
            _mark_blocked(blocked, reserved, pad=1)
            for rr, cc in _ring_points(top, left):
                grid[rr, cc] = RING
            grid[anchor] = color
            ring_specs[color] = (top, left, side, offset)
            placed = True
            break
        if not placed:
            return None

    for color in colors:
        missing = rng.choice(CORNERS)
        placed = False
        for _ in range(PLACE_RETRIES):
            top = rng.randint(1, h - 3)
            left = rng.randint(1, w - 3)
            reserved = _full_box_points(top, left, 2, 2)
            if not _padded_ok(blocked, reserved, pad=1):
                continue
            _mark_blocked(blocked, reserved, pad=1)
            for rr, cc in _pointer_cells(top, left, missing):
                grid[rr, cc] = color
            placed = True
            break
        if not placed:
            return None

    try:
        rings, pointers, directions = _extract_scene(grid)
    except ValueError:
        return None
    if set(rings) != set(colors) or set(pointers) != set(colors) or set(directions) != set(colors):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a black-background scene with several grey rings, unique colored
    anchors, and matching 3-cell pointer sprites in random positions.
    """
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
