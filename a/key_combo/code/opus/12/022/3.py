from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
TEAL = 8

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_CROSS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_RETRIES = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[tuple[list[tuple[int, int]], bool]]:
    """Return 4-connected components of one color together with their edge-touch flag."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_edge = True
                for dr, dc in _N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((coords, touches_edge))
    return comps


def _cross_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if all(int(grid[r + dr, c + dc]) == BLACK for dr, dc in _CROSS):
                centers.append((r, c))
    return centers


def _after_recolor(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    mask = np.zeros(out.shape, dtype=bool)
    for r, c in _cross_centers(out):
        for dr, dc in _CROSS:
            mask[r + dr, c + dc] = True
    out[mask] = GREEN
    return out


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _candidate_clear(
    occupied: np.ndarray,
    cells: set[tuple[int, int]],
    padding: int = 1,
) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - padding, r + padding + 1):
            for cc in range(c - padding, c + padding + 1):
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc] and (rr, cc) not in cells:
                    return False
    return True


def _gate_structure(
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], dict[str, object]] | None:
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        for _ in range(60):
            if side in {"top", "bottom"}:
                room_h = random.randint(3, 5)
                room_w = 2
                c0 = random.randint(1, w - room_w - 2)
                if side == "top":
                    if h < 7:
                        continue
                    r0 = random.randint(3, h - room_h - 2)
                    door_c = random.choice([c0, c0 + 1])
                    center = (r0 - 1, door_c)
                    cells = {(r, c) for r in range(r0, r0 + room_h) for c in range(c0, c0 + room_w)}
                    cells |= {(r, door_c) for r in range(0, r0 - 1)}
                    cells.add(center)
                    cells.add((r0 - 1, door_c - 1))
                    cells.add((r0 - 1, door_c + 1))
                    meta = {
                        "room_anchor": (r0 + room_h - 1, c0 + (1 if door_c == c0 else 0)),
                        "plus_center": center,
                    }
                    return cells, meta

                if h < 7:
                    continue
                r0 = random.randint(1, h - room_h - 4)
                r1 = r0 + room_h - 1
                door_c = random.choice([c0, c0 + 1])
                center = (r1 + 1, door_c)
                cells = {(r, c) for r in range(r0, r0 + room_h) for c in range(c0, c0 + room_w)}
                cells |= {(r, door_c) for r in range(r1 + 2, h)}
                cells.add(center)
                cells.add((r1 + 1, door_c - 1))
                cells.add((r1 + 1, door_c + 1))
                meta = {
                    "room_anchor": (r0, c0 + (1 if door_c == c0 else 0)),
                    "plus_center": center,
                }
                return cells, meta

            room_h = 2
            room_w = random.randint(3, 5)
            r0 = random.randint(1, h - room_h - 2)
            if side == "left":
                if w < 7:
                    continue
                c0 = random.randint(3, w - room_w - 2)
                door_r = random.choice([r0, r0 + 1])
                center = (door_r, c0 - 1)
                cells = {(r, c) for r in range(r0, r0 + room_h) for c in range(c0, c0 + room_w)}
                cells |= {(door_r, c) for c in range(0, c0 - 1)}
                cells.add(center)
                cells.add((door_r - 1, c0 - 1))
                cells.add((door_r + 1, c0 - 1))
                meta = {
                    "room_anchor": (r0 + (1 if door_r == r0 else 0), c0 + room_w - 1),
                    "plus_center": center,
                }
                return cells, meta

            if w < 7:
                continue
            c0 = random.randint(1, w - room_w - 4)
            c1 = c0 + room_w - 1
            door_r = random.choice([r0, r0 + 1])
            center = (door_r, c1 + 1)
            cells = {(r, c) for r in range(r0, r0 + room_h) for c in range(c0, c0 + room_w)}
            cells |= {(door_r, c) for c in range(c1 + 2, w)}
            cells.add(center)
            cells.add((door_r - 1, c1 + 1))
            cells.add((door_r + 1, c1 + 1))
            meta = {
                "room_anchor": (r0 + (1 if door_r == r0 else 0), c0),
                "plus_center": center,
            }
            return cells, meta
    return None


def _closed_room_structure(h: int, w: int) -> tuple[set[tuple[int, int]], dict[str, object]]:
    if random.random() < 0.5:
        room_h = 2
        room_w = random.randint(3, 6)
    else:
        room_h = random.randint(3, 6)
        room_w = 2
    r0 = random.randint(1, h - room_h - 2)
    c0 = random.randint(1, w - room_w - 2)
    cells = {(r, c) for r in range(r0, r0 + room_h) for c in range(c0, c0 + room_w)}
    meta = {"room_anchor": (r0 + room_h // 2, c0 + room_w // 2)}
    return cells, meta


def _open_cross_structure(
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], dict[str, object]]:
    center_r = random.randint(3, h - 4)
    center_c = random.randint(2, w - 3)
    left_len = random.randint(1, min(4, center_c - 1))
    right_len = random.randint(1, min(4, w - center_c - 2))
    cells = {(r, center_c) for r in range(h)}
    cells |= {(center_r, c) for c in range(center_c - left_len, center_c + right_len + 1)}
    meta = {
        "plus_center": (center_r, center_c),
        "open_anchor": (0, center_c),
    }
    return cells, meta


def _semantic_check(grid: np.ndarray, meta: dict[str, object]) -> bool:
    if set(map(int, np.unique(grid))) != {BLACK, TEAL}:
        return False

    recolored = _after_recolor(grid)
    solved = main(grid)

    centers = set(_cross_centers(grid))
    if meta["gate_plus"] not in centers or meta["open_plus"] not in centers:
        return False
    if len(centers) < 2:
        return False

    gate_anchor = meta["gate_anchor"]
    closed_anchor = meta["closed_anchor"]
    open_anchor = meta["open_anchor"]

    if int(solved[gate_anchor]) != YELLOW:
        return False
    if int(solved[closed_anchor]) != YELLOW:
        return False
    if int(solved[open_anchor]) != BLACK:
        return False
    if int(solved[meta["gate_plus"]]) != GREEN or int(solved[meta["open_plus"]]) != GREEN:
        return False

    gate_reaches_edge_before = False
    gate_enclosed_after = False
    open_touches_edge_after = False
    for coords, touches_edge in _components_of_color(grid, BLACK):
        coord_set = set(coords)
        if gate_anchor in coord_set:
            gate_reaches_edge_before = touches_edge
    for coords, touches_edge in _components_of_color(recolored, BLACK):
        coord_set = set(coords)
        if gate_anchor in coord_set:
            gate_enclosed_after = not touches_edge
        if open_anchor in coord_set:
            open_touches_edge_after = touches_edge

    if not gate_reaches_edge_before or not gate_enclosed_after or not open_touches_edge_after:
        return False
    if gate_anchor == closed_anchor or gate_anchor == open_anchor or closed_anchor == open_anchor:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every full black orthogonal cross, then flood-fill the black regions that become fully enclosed.

    Input structure:
    - The input is a 2D grid that uses black (0) and teal (8).
    - Teal pixels act as fixed walls or blockers around narrow black rooms, corridors, and crossings.
    - Some black pixels are centers of complete 5-cell orthogonal crosses, while other black regions are narrow enough that they contain no full cross.

    Rule:
    - Detect every black cell whose four orthogonal neighbors are also black.
    - Recolor each such center together with its four orthogonal neighbors to green (3), merging overlaps.
    - Recompute 4-connected black components after recoloring.
    - Any remaining black component that does not touch the grid edge is fully enclosed by non-black pixels and is flood-filled to yellow (4); edge-touching black components stay black.

    Key insight:
    - The green cross repaint must happen first, because it can sever the only outlet of a black region and turn an open component into an enclosed one that should then be filled.
    """
    g = np.asarray(grid, dtype=int)
    out = _after_recolor(g)
    for coords, touches_edge in _components_of_color(out, BLACK):
        if touches_edge:
            continue
        for r, c in coords:
            out[r, c] = YELLOW
    return out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        h = random.randint(16, 24)
        w = random.randint(16, 24)
        grid = np.full((h, w), TEAL, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        gate = _gate_structure(h, w)
        if gate is None:
            continue
        gate_cells, gate_meta = gate
        if not _candidate_clear(occupied, gate_cells, padding=1):
            continue
        _paint_cells(grid, gate_cells, BLACK)
        _paint_cells(occupied, gate_cells, True)

        closed_ok = False
        for _inner in range(80):
            closed_cells, closed_meta = _closed_room_structure(h, w)
            if _candidate_clear(occupied, closed_cells, padding=2):
                _paint_cells(grid, closed_cells, BLACK)
                _paint_cells(occupied, closed_cells, True)
                closed_ok = True
                break
        if not closed_ok:
            continue

        open_ok = False
        for _inner in range(80):
            open_cells, open_meta = _open_cross_structure(h, w)
            if _candidate_clear(occupied, open_cells, padding=1):
                _paint_cells(grid, open_cells, BLACK)
                _paint_cells(occupied, open_cells, True)
                open_ok = True
                break
        if not open_ok:
            continue

        meta = {
            "gate_anchor": gate_meta["room_anchor"],
            "closed_anchor": closed_meta["room_anchor"],
            "open_anchor": open_meta["open_anchor"],
            "gate_plus": gate_meta["plus_center"],
            "open_plus": open_meta["plus_center"],
        }
        if _semantic_check(grid, meta):
            return grid

    raise RuntimeError("failed to generate a valid input")
