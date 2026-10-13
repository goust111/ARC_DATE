from __future__ import annotations

import random

import numpy as np


BG = 0
PRISM = 5
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
CARDINALS = (UP, DOWN, LEFT, RIGHT)
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        deltas = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in deltas:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if _in_bounds(h, w, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _prism_orientation(cells: list[tuple[int, int]]) -> str:
    diffs = {r - c for r, c in cells}
    sums = {r + c for r, c in cells}
    if len(diffs) == 1:
        return "\\"
    if len(sums) == 1:
        return "/"
    raise AssertionError("grey component must be a diagonal prism")


def _sorted_prism_cells(cells: list[tuple[int, int]], orientation: str) -> list[tuple[int, int]]:
    if orientation == "\\":
        return sorted(cells)
    return sorted(cells, key=lambda p: (p[0], -p[1]))


def _reflect(step: tuple[int, int], orientation: str) -> tuple[int, int]:
    dr, dc = map(int, step)
    if orientation == "/":
        if dr == 0:
            return (-dc, dc)
        return (dr, -dr)
    assert orientation == "\\"
    if dr == 0:
        return (dc, dc)
    return (dr, dr)


def _ray_to_edge(start: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = step
    out: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(h, w, r, c):
            return out
        out.append((int(r), int(c)))


def _steps_to_edge(start: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> int:
    return len(_ray_to_edge(start, step, shape))


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _most_common_color(g)

    prisms: list[dict] = []
    prism_by_cell: dict[tuple[int, int], int] = {}
    grey_components = _components(g == PRISM, connectivity=8)

    colored: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere((g != bg) & (g != PRISM)):
        colored[(int(r), int(c))] = int(g[int(r), int(c)])

    marker_positions: set[tuple[int, int]] = set()
    for comp in grey_components:
        assert 2 <= len(comp) <= 3, "each prism must have length 2 or 3"
        orientation = _prism_orientation(comp)
        ordered = _sorted_prism_cells(comp, orientation)
        endpoints = [ordered[0], ordered[-1]]

        marker_candidates: list[tuple[int, int]] = []
        for pos in colored:
            touches = 0
            for er, ec in endpoints:
                if max(abs(pos[0] - er), abs(pos[1] - ec)) == 1:
                    touches += 1
            if touches != 1:
                continue
            if sum(max(abs(pos[0] - rr), abs(pos[1] - cc)) == 1 for rr, cc in ordered) != 1:
                continue
            marker_candidates.append(pos)

        marker_candidates = sorted(set(marker_candidates))
        assert len(marker_candidates) == 1, "each prism must have one unambiguous marker"
        marker_pos = marker_candidates[0]
        assert marker_pos not in marker_positions, "marker cannot belong to two prisms"
        marker_positions.add(marker_pos)

        prism_index = len(prisms)
        cell_set = set(ordered)
        for cell in cell_set:
            prism_by_cell[cell] = prism_index
        prisms.append(
            {
                "cells": cell_set,
                "orientation": orientation,
                "marker_pos": marker_pos,
                "marker_color": int(colored[marker_pos]),
            }
        )

    towers: dict[tuple[int, int], int] = {
        pos: color for pos, color in colored.items() if pos not in marker_positions
    }
    assert towers, "expected at least one signal tower"

    return {
        "grid": g,
        "shape": (h, w),
        "bg": int(bg),
        "prisms": prisms,
        "prism_by_cell": prism_by_cell,
        "markers": marker_positions,
        "towers": towers,
    }


def _solve(grid: np.ndarray, *, strict: bool = False, return_stats: bool = False):
    scene = _parse_scene(grid)
    g = scene["grid"]
    h, w = scene["shape"]
    bg = int(scene["bg"])
    tower_positions = set(scene["towers"])
    prisms = scene["prisms"]
    prism_by_cell = scene["prism_by_cell"]
    base_non_bg = g != bg
    out = g.copy()

    path_colors: dict[tuple[int, int], int] = {}
    orth_cells: set[tuple[int, int]] = set()
    diag_cells: set[tuple[int, int]] = set()
    stats = {
        "orth_edge_stops": 0,
        "orth_tower_stops": 0,
        "orth_prism_hits": 0,
        "diag_edge_stops": 0,
        "diag_tower_stops": 0,
        "diag_lengths": [],
    }

    def paint(cell: tuple[int, int], color: int, kind: str) -> None:
        r, c = cell
        if bool(base_non_bg[r, c]):
            if strict:
                raise AssertionError("ray crossed a static object")
            return

        prev = path_colors.get(cell)
        if strict and prev is not None and prev != int(color):
            raise AssertionError("two ray colors overlapped")
        if strict and kind == "diag" and (cell in diag_cells or cell in orth_cells):
            raise AssertionError("diagonal ray overlapped an earlier ray")
        if strict and kind == "orth" and cell in diag_cells:
            raise AssertionError("orthogonal ray overlapped a diagonal ray")

        path_colors[cell] = int(color)
        if kind == "orth":
            orth_cells.add(cell)
        else:
            diag_cells.add(cell)
        out[r, c] = int(color)

    def emit_diag(start: tuple[int, int], step: tuple[int, int], color: int) -> tuple[int, str]:
        r, c = start
        dr, dc = step
        length = 0
        while True:
            r += dr
            c += dc
            if not _in_bounds(h, w, r, c):
                return length, "edge"
            if (r, c) in tower_positions:
                return length, "tower"
            if strict and bool(base_non_bg[r, c]):
                raise AssertionError("diagonal ray hit a non-tower object")
            paint((int(r), int(c)), int(color), "diag")
            length += 1

    def emit_orth(start: tuple[int, int], step: tuple[int, int], color: int) -> None:
        r, c = start
        dr, dc = step
        while True:
            r += dr
            c += dc
            if not _in_bounds(h, w, r, c):
                stats["orth_edge_stops"] += 1
                return
            if (r, c) in tower_positions:
                stats["orth_tower_stops"] += 1
                return
            prism_index = prism_by_cell.get((int(r), int(c)))
            if prism_index is not None:
                prism = prisms[int(prism_index)]
                stats["orth_prism_hits"] += 1
                diag_step = _reflect(step, prism["orientation"])
                diag_len, stop_kind = emit_diag((int(r), int(c)), diag_step, prism["marker_color"])
                stats["diag_lengths"].append(int(diag_len))
                if stop_kind == "tower":
                    stats["diag_tower_stops"] += 1
                else:
                    stats["diag_edge_stops"] += 1
                return
            if strict and bool(base_non_bg[r, c]):
                raise AssertionError("orthogonal ray hit a non-prism object")
            paint((int(r), int(c)), int(color), "orth")

    for pos, color in sorted(scene["towers"].items()):
        for step in CARDINALS:
            emit_orth(pos, step, color)

    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Colored signal towers emit orthogonal rays. Grey diagonal prisms reflect a ray once into a
    diagonal direction and recolor it using the prism's adjacent marker pixel.
    """
    return _solve(grid, strict=False, return_stats=False)


def _add_group(
    grid: np.ndarray,
    group_cells: list[tuple[int, int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    coords = {(r, c) for r, c, _color in group_cells}
    if len(coords) != len(group_cells):
        return False
    if any(not _in_bounds(h, w, r, c) for r, c in coords):
        return False
    if any((r, c) in reserved for r, c in coords):
        return False

    for r, c, color in group_cells:
        grid[r, c] = int(color)

    for r, c in coords:
        reserved.add((r, c))
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(h, w, nr, nc):
                reserved.add((nr, nc))
    return True


def _marker_spots(
    endpoint: tuple[int, int],
    prism_cells: list[tuple[int, int]],
    orth_cells: set[tuple[int, int]],
    diag_cells: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    prism_set = set(prism_cells)
    endpoints = {prism_cells[0], prism_cells[-1]}
    out: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(*endpoint):
        if not _in_bounds(h, w, nr, nc):
            continue
        if (nr, nc) in reserved or (nr, nc) in prism_set or (nr, nc) in orth_cells or (nr, nc) in diag_cells:
            continue
        if sum(max(abs(nr - er), abs(nc - ec)) == 1 for er, ec in endpoints) != 1:
            continue
        if sum(max(abs(nr - rr), abs(nc - cc)) == 1 for rr, cc in prism_cells) != 1:
            continue
        out.append((int(nr), int(nc)))
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.full((h, w), BG, dtype=int)

    tower_color = random.choice(COLORS)
    marker_palette = [color for color in COLORS if color != tower_color]

    hub = (random.randint(4, h - 5), random.randint(4, w - 5))
    reserved: set[tuple[int, int]] = set()
    if not _add_group(grid, [(hub[0], hub[1], tower_color)], reserved):
        return None

    prism_count = random.choices([1, 2, 3], weights=[2, 4, 2])[0]
    prism_dirs = random.sample(list(CARDINALS), prism_count)
    remaining_dirs = [step for step in CARDINALS if step not in prism_dirs]

    blocker_count = 0
    if len(remaining_dirs) >= 2 and random.random() < 0.6:
        blocker_count = random.randint(1, len(remaining_dirs) - 1)
    blocker_dirs = random.sample(remaining_dirs, blocker_count) if blocker_count else []

    for step in prism_dirs:
        placed = False
        for _ in range(80):
            orientation = random.choice(("/", "\\"))
            diag_step = _reflect(step, orientation)
            length = random.choice([2, 3])
            max_dist = _steps_to_edge(hub, step, grid.shape)
            if max_dist < 3:
                return None
            dist = random.randint(2, max_dist)
            hit = (hub[0] + step[0] * dist, hub[1] + step[1] * dist)

            prism_cells = [
                (hit[0] - diag_step[0] * i, hit[1] - diag_step[1] * i)
                for i in range(length)
            ]
            if len({cell for cell in prism_cells}) != length:
                continue
            if any(not _in_bounds(h, w, r, c) for r, c in prism_cells):
                continue
            orth_cells = {
                (hub[0] + step[0] * i, hub[1] + step[1] * i)
                for i in range(1, dist)
            }
            if any(grid[r, c] != BG for r, c in orth_cells):
                continue
            if any(grid[r, c] != BG for r, c in prism_cells):
                continue

            diag_cells = set(_ray_to_edge(hit, diag_step, grid.shape))
            if len(diag_cells) < 2:
                continue

            endpoints = [prism_cells[0], prism_cells[-1]]
            endpoint_order = endpoints[:]
            random.shuffle(endpoint_order)
            marker_pos = None
            for endpoint in endpoint_order:
                candidates = _marker_spots(endpoint, prism_cells, orth_cells, diag_cells, reserved, grid.shape)
                if candidates:
                    marker_pos = random.choice(candidates)
                    break
            if marker_pos is None:
                continue

            group = [(r, c, PRISM) for r, c in prism_cells]
            group.append((marker_pos[0], marker_pos[1], random.choice(marker_palette)))
            if not _add_group(grid, group, reserved):
                continue
            placed = True
            break
        if not placed:
            return None

    for step in blocker_dirs:
        placed = False
        for _ in range(60):
            max_dist = _steps_to_edge(hub, step, grid.shape)
            if max_dist < 3:
                return None
            dist = random.randint(3, max_dist)
            tower = (hub[0] + step[0] * dist, hub[1] + step[1] * dist)
            path = {
                (hub[0] + step[0] * i, hub[1] + step[1] * i)
                for i in range(1, dist)
            }
            if any(grid[r, c] != BG for r, c in path):
                continue
            if _add_group(grid, [(tower[0], tower[1], tower_color)], reserved):
                placed = True
                break
        if not placed:
            return None

    try:
        _out, stats = _solve(grid, strict=True, return_stats=True)
    except AssertionError:
        return None

    if stats["orth_prism_hits"] < prism_count:
        return None
    if not stats["diag_lengths"] or min(stats["diag_lengths"]) < 2:
        return None
    if blocker_dirs and (stats["orth_tower_stops"] + stats["diag_tower_stops"] == 0):
        return None
    if stats["orth_edge_stops"] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
