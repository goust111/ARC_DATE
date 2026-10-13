from __future__ import annotations

import random

import numpy as np


BG = 0
NEUTRAL = 5
ARROW_COLORS = (1, 2, 3, 7)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

Coord = tuple[int, int]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_arrows(grid: np.ndarray) -> tuple[list[dict], dict[Coord, int]]:
    arrows: list[dict] = []
    cell_to_arrow: dict[Coord, int] = {}

    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, NEUTRAL)):
        assert color in ARROW_COLORS, f"unexpected foreground color {color}"
        for comp in _components(grid == color):
            assert len(comp) == 2, f"color {color} must form 2-cell arrows"
            (r0, c0), (r1, c1) = sorted(comp)
            if r0 == r1 and abs(c0 - c1) == 1:
                tail = (r0, min(c0, c1))
                head = (r0, max(c0, c1))
                direction = (0, 1)
            elif c0 == c1 and abs(r0 - r1) == 1:
                tail = (min(r0, r1), c0)
                head = (max(r0, r1), c0)
                direction = (1, 0)
            else:
                raise AssertionError(f"invalid arrow cells {sorted(comp)}")

            arrow_id = len(arrows)
            cells = {tail, head}
            for cell in cells:
                assert cell not in cell_to_arrow, f"overlapping arrows at {cell}"
                cell_to_arrow[cell] = arrow_id
            arrows.append(
                {
                    "id": arrow_id,
                    "color": color,
                    "tail": tail,
                    "head": head,
                    "dir": direction,
                    "cells": cells,
                }
            )

    assert arrows, "expected at least one arrow"
    return arrows, cell_to_arrow


def _trace_wave(
    grid: np.ndarray,
    arrow: dict,
    cell_to_arrow: dict[Coord, int],
    current_id: int,
) -> tuple[list[Coord], int | None]:
    h, w = grid.shape
    dr, dc = arrow["dir"]
    r, c = arrow["head"]
    r += dr
    c += dc

    painted: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value == NEUTRAL:
            painted.append((r, c))
            r += dr
            c += dc
            continue
        if value == BG:
            return painted, None
        hit = cell_to_arrow.get((r, c))
        if hit is not None and hit != current_id:
            return painted, hit
        return painted, None
    return painted, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey corridors wave by wave from domino arrows, waking downstream arrows when a wave reaches them.

    Input structure:
    - The background is black.
    - Grey cells form neutral straight or crossing corridor segments.
    - Every colored object is a same-color 2-cell domino arrow. A horizontal domino points right; a vertical domino points down.
    - Some arrows lie directly on another arrow's straight sight line through grey corridor cells, so they can be activated by an earlier wave.

    Rule:
    - First determine which arrows are first-wave emitters: these are the arrows that are not directly reached by any other arrow's straight ray in the untouched input.
    - Process the scene round by round. In one round, every active arrow emits a straight wave from its front cell on a frozen snapshot of the grid.
    - On that snapshot, the wave keeps moving through grey cells, scheduling them to be recolored to the arrow's color. It stops at black or at any already colored cell.
    - If the stopping cell belongs to another arrow, that arrow becomes active for the next round. Apply all recolorings from the round simultaneously, and never recolor the arrow dominoes themselves.

    Key insight:
    - Grey corridors are consumable medium: once an earlier round recolors a corridor cell, that colored cell becomes a blocker for later rounds, so the activation order of the arrows matters.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows, cell_to_arrow = _parse_arrows(g)

    indegree = [0] * len(arrows)
    for arrow in arrows:
        _painted, target = _trace_wave(g, arrow, cell_to_arrow, arrow["id"])
        if target is not None:
            indegree[target] += 1

    active = sorted(i for i, deg in enumerate(indegree) if deg == 0)
    assert active, "expected at least one first-wave arrow"

    out = g.copy()
    seen = set(active)
    fired: set[int] = set()

    while active:
        snapshot = out.copy()
        pending_paint: dict[Coord, int] = {}
        next_active: list[int] = []

        for arrow_id in active:
            if arrow_id in fired:
                continue
            fired.add(arrow_id)
            arrow = arrows[arrow_id]
            painted, target = _trace_wave(snapshot, arrow, cell_to_arrow, arrow_id)

            for cell in painted:
                prev = pending_paint.get(cell)
                if prev is None:
                    pending_paint[cell] = int(arrow["color"])
                else:
                    assert prev == int(arrow["color"]), "same-round waves cannot conflict on one corridor cell"

            if target is not None and target not in seen:
                seen.add(target)
                next_active.append(target)

        for (r, c), color in pending_paint.items():
            out[r, c] = color

        active = sorted(next_active)

    return out


def _set_cell(cells: dict[Coord, int], coord: Coord, color: int) -> None:
    prev = cells.get(coord)
    assert prev is None or prev == color, f"conflicting local bundle cell at {coord}: {prev} vs {color}"
    cells[coord] = color


def _paint_h_arrow(cells: dict[Coord, int], row: int, col: int, color: int) -> None:
    _set_cell(cells, (row, col), color)
    _set_cell(cells, (row, col + 1), color)


def _paint_v_arrow(cells: dict[Coord, int], row: int, col: int, color: int) -> None:
    _set_cell(cells, (row, col), color)
    _set_cell(cells, (row + 1, col), color)


def _paint_h_corridor(cells: dict[Coord, int], row: int, c0: int, c1: int) -> None:
    for c in range(c0, c1 + 1):
        _set_cell(cells, (row, c), NEUTRAL)


def _paint_v_corridor(cells: dict[Coord, int], col: int, r0: int, r1: int) -> None:
    for r in range(r0, r1 + 1):
        _set_cell(cells, (r, col), NEUTRAL)


def _cells_to_array(cells: dict[Coord, int]) -> np.ndarray:
    max_r = max(r for r, _c in cells)
    max_c = max(c for _r, c in cells)
    arr = np.full((max_r + 1, max_c + 1), BG, dtype=int)
    for (r, c), color in cells.items():
        arr[r, c] = color
    return arr


def _maybe_transpose(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        return arr.T.copy()
    return arr


def _choose_color(rng: random.Random, exclude: set[int] | None = None) -> int:
    options = [c for c in ARROW_COLORS if exclude is None or c not in exclude]
    return int(rng.choice(options))


def _make_turn_bundle(rng: random.Random) -> np.ndarray:
    len1 = rng.randint(1, 5)
    len2 = rng.randint(1, 5)
    color_a = _choose_color(rng)
    color_b = _choose_color(rng)

    cells: dict[Coord, int] = {}
    _paint_h_arrow(cells, 0, 0, color_a)
    _paint_h_corridor(cells, 0, 2, 1 + len1)
    hit_col = 2 + len1
    _paint_v_arrow(cells, 0, hit_col, color_b)
    _paint_v_corridor(cells, hit_col, 2, 1 + len2)
    return _maybe_transpose(_cells_to_array(cells), rng)


def _make_long_bundle(rng: random.Random) -> np.ndarray:
    len1 = rng.randint(1, 4)
    len2 = rng.randint(1, 4)
    len3 = rng.randint(1, 5)
    color_a = _choose_color(rng)
    color_b = _choose_color(rng)
    color_c = _choose_color(rng)

    cells: dict[Coord, int] = {}
    _paint_h_arrow(cells, 0, 0, color_a)
    _paint_h_corridor(cells, 0, 2, 1 + len1)
    mid_col = 2 + len1
    _paint_v_arrow(cells, 0, mid_col, color_b)
    _paint_v_corridor(cells, mid_col, 2, 1 + len2)
    bottom_row = 2 + len2
    _paint_h_arrow(cells, bottom_row, mid_col, color_c)
    _paint_h_corridor(cells, bottom_row, mid_col + 2, mid_col + 1 + len3)
    return _maybe_transpose(_cells_to_array(cells), rng)


def _make_blocked_bundle(rng: random.Random) -> np.ndarray:
    len1 = rng.randint(1, 4)
    len2 = rng.randint(2, 5)
    color_a = _choose_color(rng)
    color_b = _choose_color(rng)
    color_c = _choose_color(rng, exclude={color_b})
    block_row = rng.randint(2, 1 + len2)
    blocker_len = len1 + 1 + rng.randint(0, 3)

    cells: dict[Coord, int] = {}
    _paint_h_arrow(cells, 0, 0, color_a)
    _paint_h_corridor(cells, 0, 2, 1 + len1)
    hit_col = 2 + len1
    _paint_v_arrow(cells, 0, hit_col, color_b)
    _paint_v_corridor(cells, hit_col, 2, 1 + len2)
    _paint_h_arrow(cells, block_row, 0, color_c)
    _paint_h_corridor(cells, block_row, 2, 1 + blocker_len)
    return _maybe_transpose(_cells_to_array(cells), rng)


def _place_bundle(
    grid: np.ndarray,
    occupied: np.ndarray,
    bundle: np.ndarray,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    bh, bw = bundle.shape
    positions: list[Coord] = []

    for r0 in range(h - bh + 1):
        for c0 in range(w - bw + 1):
            r_lo = max(0, r0 - 1)
            r_hi = min(h, r0 + bh + 1)
            c_lo = max(0, c0 - 1)
            c_hi = min(w, c0 + bw + 1)
            if occupied[r_lo:r_hi, c_lo:c_hi].any():
                continue
            positions.append((r0, c0))

    if not positions:
        return False

    r0, c0 = rng.choice(positions)
    mask = bundle != BG
    grid[r0 : r0 + bh, c0 : c0 + bw][mask] = bundle[mask]
    occupied[max(0, r0 - 1) : min(h, r0 + bh + 1), max(0, c0 - 1) : min(w, c0 + bw + 1)] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bundle_count = rng.randint(1, 3)
    bundle_kinds = [rng.choice(("blocked", "long"))]
    for _ in range(bundle_count - 1):
        bundle_kinds.append(rng.choice(("turn", "blocked", "long")))
    rng.shuffle(bundle_kinds)

    bundles: list[np.ndarray] = []
    for kind in bundle_kinds:
        if kind == "turn":
            bundles.append(_make_turn_bundle(rng))
        elif kind == "long":
            bundles.append(_make_long_bundle(rng))
        else:
            bundles.append(_make_blocked_bundle(rng))

    bundles.sort(key=lambda arr: int(arr.shape[0] * arr.shape[1]), reverse=True)

    min_side = max(12, max(max(bundle.shape) for bundle in bundles) + 2)
    max_side = 26
    if min_side > max_side:
        return None

    h = rng.randint(min_side, max_side)
    w = rng.randint(min_side, max_side)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for bundle in bundles:
        if not _place_bundle(grid, occupied, bundle, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(grid == NEUTRAL):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
