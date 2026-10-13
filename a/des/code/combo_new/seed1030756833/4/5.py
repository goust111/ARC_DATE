from __future__ import annotations

import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
DIRS: tuple[Coord, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _left_of(d: Coord) -> Coord:
    dr, dc = d
    return -dc, dr


def _right_of(d: Coord) -> Coord:
    dr, dc = d
    return dc, -dr


def _add(a: Coord, b: Coord) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Coord, b: Coord) -> Coord:
    return a[0] - b[0], a[1] - b[1]


def _degree(cell: Coord, cells: set[Coord]) -> int:
    r, c = cell
    return sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))


def _missing_vector(missing: Coord, r0: int, c0: int) -> Coord:
    mr, mc = missing
    return (-1 if mr == r0 else 1, -1 if mc == c0 else 1)


def _turn_side(missing_vec: Coord, forward: Coord) -> str:
    left = _left_of(forward)
    score = missing_vec[0] * left[0] + missing_vec[1] * left[1]
    assert score != 0
    return "left" if score > 0 else "right"


def _find_start_shape(cells: set[Coord]) -> tuple[set[Coord], Coord, Coord, Coord]:
    """Return shape cells, anchor cell, first path cell, and missing-corner vector."""
    assert cells
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    candidates: list[tuple[set[Coord], Coord, Coord, Coord]] = []

    for r0 in range(min(rows) - 1, max(rows) + 1):
        for c0 in range(min(cols) - 1, max(cols) + 1):
            block = {
                (r0, c0),
                (r0 + 1, c0),
                (r0, c0 + 1),
                (r0 + 1, c0 + 1),
            }
            present = block & cells
            if len(present) != 3:
                continue

            outside: set[Coord] = set()
            touching_shape_cells: set[Coord] = set()
            for cell in present:
                for nbr in _neighbors4(*cell):
                    if nbr in cells and nbr not in present:
                        outside.add(nbr)
                        touching_shape_cells.add(cell)

            if len(outside) != 1 or len(touching_shape_cells) != 1:
                continue

            missing = next(iter(block - present))
            anchor = next(iter(touching_shape_cells))
            first_path = next(iter(outside))
            miss_vec = _missing_vector(missing, r0, c0)
            candidates.append((set(present), anchor, first_path, miss_vec))

    assert len(candidates) == 1, f"expected one 2x2 L start shape, found {len(candidates)}"
    return candidates[0]


def _choose_next(
    current: Coord,
    previous: Coord,
    candidates: list[Coord],
    missing_vec: Coord,
) -> Coord:
    forward = _sub(current, previous)
    desired = _left_of(forward) if _turn_side(missing_vec, forward) == "left" else _right_of(forward)
    for cand in candidates:
        if _sub(cand, current) == desired:
            return cand
    assert False, "branch must offer the direction indicated by the missing corner"


def _trace_path(cells: set[Coord], anchor: Coord, first_path: Coord, missing_vec: Coord) -> tuple[list[Coord], Coord]:
    path_cells = set(cells)
    assert first_path in path_cells

    previous = anchor
    current = first_path
    visited: list[Coord] = []
    seen: set[Coord] = set()

    for _ in range(len(path_cells) + 5):
        assert current in path_cells
        assert current not in seen, "path traversal should not loop"
        visited.append(current)
        seen.add(current)

        candidates = [
            nbr
            for nbr in _neighbors4(*current)
            if nbr in path_cells and nbr != previous and nbr not in seen
        ]

        if len(candidates) == 1 and _degree(candidates[0], path_cells) == 1:
            # The single leaf beyond the path end is the same-color endpoint marker.
            break
        if not candidates:
            break
        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            nxt = _choose_next(current, previous, candidates, missing_vec)
        previous, current = current, nxt
    else:
        raise AssertionError("path traversal exceeded the path size")

    return visited, current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored L shape along its matching path, choosing fork turns from the missing corner.

    Input structure:
    - The background is black. Each foreground color owns one orthogonal one-pixel-wide path network.
    - At the start of each color network is a same-color 2x2 L triomino: a 2x2 square with one missing corner.
    - At the far path end is a same-color one-pixel endpoint marker; optional side branches may remain unchosen.

    Rule:
    - For each color, remove the starting L shape and follow the same-color path from the adjacent start cell.
    - Traversed path cells become black. At a fork with two onward choices, compare the missing corner to the current direction: a left-side missing corner turns left, and a right-side missing corner turns right.
    - Stop at the path cell before the endpoint marker, preserve the marker, and stamp the same L shape there without rotating it.

    Key insight:
    - Color matching identifies which path belongs to each shape, while the missing corner encodes the hidden left/right turn instruction at a fork.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    for color in [int(c) for c in np.unique(g) if int(c) != 0]:
        color_cells = {(int(r), int(c)) for r, c in np.argwhere(g == color)}
        shape_cells, anchor, first_path, missing_vec = _find_start_shape(color_cells)
        path_and_marker = color_cells - shape_cells
        shape_offsets = [_sub(cell, anchor) for cell in shape_cells]

        visited, final_anchor = _trace_path(path_and_marker, anchor, first_path, missing_vec)

        for r, c in shape_cells:
            out[r, c] = 0
        for r, c in visited:
            out[r, c] = 0

        for dr, dc in shape_offsets:
            rr, cc = final_anchor[0] + dr, final_anchor[1] + dc
            assert 0 <= rr < h and 0 <= cc < w
            assert int(out[rr, cc]) in (0, color)
            out[rr, cc] = color

    return out


# ----------------------------- generator -----------------------------

_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_GEN_RETRIES = 1000
_PLACE_RETRIES = 200


def _shape_options(forward: Coord, turn_side: str) -> list[tuple[list[Coord], Coord]]:
    opts: list[tuple[list[Coord], Coord]] = []
    block = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for missing in block:
        cells = block - {missing}
        miss_vec = _missing_vector(missing, 0, 0)
        if _turn_side(miss_vec, forward) != turn_side:
            continue
        for anchor in cells:
            first_path = _add(anchor, forward)
            if first_path in cells:
                continue
            touching = [cell for cell in cells if abs(cell[0] - first_path[0]) + abs(cell[1] - first_path[1]) == 1]
            if touching != [anchor]:
                continue
            offsets = [_sub(cell, anchor) for cell in cells]
            opts.append((offsets, miss_vec))
    return opts


def _make_local_object() -> tuple[set[Coord], set[Coord], Coord]:
    forward = random.choice(DIRS)
    turn_side = random.choice(("left", "right"))
    options = _shape_options(forward, turn_side)
    assert options
    shape_offsets, _miss_vec = random.choice(options)

    side = _left_of(forward) if turn_side == "left" else _right_of(forward)
    other_side = _right_of(forward) if turn_side == "left" else _left_of(forward)
    stem_len = random.randint(3, 5)
    chosen_len = random.randint(2, 5)
    final_len = random.randint(3, 6)
    dead_len = random.randint(2, 5)

    anchor = (0, 0)
    shape = {_add(anchor, off) for off in shape_offsets}
    path: set[Coord] = set()

    junction = anchor
    for i in range(1, stem_len + 1):
        junction = (anchor[0] + forward[0] * i, anchor[1] + forward[1] * i)
        path.add(junction)

    branch_end = junction
    for i in range(1, chosen_len + 1):
        cell = (junction[0] + side[0] * i, junction[1] + side[1] * i)
        path.add(cell)
        branch_end = cell

    final_end = branch_end
    for i in range(1, final_len + 1):
        cell = (branch_end[0] + forward[0] * i, branch_end[1] + forward[1] * i)
        path.add(cell)
        final_end = cell

    for i in range(1, dead_len + 1):
        path.add((junction[0] + other_side[0] * i, junction[1] + other_side[1] * i))

    marker = _add(final_end, forward)
    path.add(marker)

    final_shape = {_add(final_end, off) for off in shape_offsets}
    if marker in final_shape:
        raise ValueError("endpoint marker would be covered by final shape")
    if final_shape & (path - {final_end}):
        raise ValueError("final shape would cover unconsumed path")

    return shape | path, shape, final_end


def _normalize(cells: set[Coord], pad: int = 1) -> tuple[set[Coord], int, int]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    shifted = {(r - min_r + pad, c - min_c + pad) for r, c in cells}
    height = max(r for r, _c in shifted) + pad + 1
    width = max(c for _r, c in shifted) + pad + 1
    return shifted, height, width


def _can_place(cells: set[Coord], top: int, left: int, h: int, w: int, occupied: set[Coord]) -> bool:
    placed = {(top + r, left + c) for r, c in cells}
    if any(r < 0 or c < 0 or r >= h or c >= w for r, c in placed):
        return False
    for r, c in placed:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 30)
    w = random.randint(15, 30)
    max_objects = 3 if min(h, w) >= 22 else 2
    n_objects = random.randint(1, max_objects)
    colors = random.sample(_COLORS, n_objects)
    grid = np.zeros((h, w), dtype=int)
    occupied: set[Coord] = set()

    for color in colors:
        placed = False
        for _ in range(_PLACE_RETRIES):
            try:
                local, _shape, _final_anchor = _make_local_object()
            except ValueError:
                continue
            cells, obj_h, obj_w = _normalize(local, pad=1)
            if obj_h > h or obj_w > w:
                continue
            top = random.randint(0, h - obj_h)
            left = random.randint(0, w - obj_w)
            if not _can_place(cells, top, left, h, w, occupied):
                continue
            for r, c in cells:
                rr, cc = top + r, left + c
                grid[rr, cc] = color
                occupied.add((rr, cc))
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for seed1030756833/4/5")
