import random
from collections import deque
from itertools import combinations

import numpy as np


BG = 0
GREY = 5
RANK_COLORS = [1, 2, 3, 4]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _rounded_center(coords: np.ndarray) -> tuple[float, float]:
    center_r = round(float(coords[:, 0].mean()) * 2.0) / 2.0
    center_c = round(float(coords[:, 1].mean()) * 2.0) / 2.0
    return center_r, center_c


def _rotate_cell(cell: tuple[int, int], center: tuple[float, float], turns: int) -> tuple[int, int]:
    r, c = cell
    cr, cc = center
    dr = r - cr
    dc = c - cc
    for _ in range(turns % 4):
        dr, dc = -dc, dr
    rr = cr + dr
    cc2 = cc + dc
    r_out = int(round(rr))
    c_out = int(round(cc2))
    if abs(rr - r_out) > 1e-6 or abs(cc2 - c_out) > 1e-6:
        raise ValueError("rotation did not land on integer coordinates")
    return r_out, c_out


def _rotation_closure(cells: set[tuple[int, int]], center: tuple[float, float]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for cell in cells:
        for turns in range(4):
            out.add(_rotate_cell(cell, center, turns))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each broken rotationally symmetric gray object, rank objects by how many pixels were restored, then recolor by rank.

    Input structure:
    - The grid has a black background and several separated gray connected components.
    - Each gray component is an incomplete version of a 90-degree rotationally symmetric shape.
    - Existing gray pixels already agree with quarter-turn symmetry: rotated counterparts are either gray or missing black cells.

    Rule:
    - For each gray component, infer its quarter-turn center from the component centroid.
    - Rotate every gray pixel by 90, 180, and 270 degrees around that center, filling missing black cells to obtain the completed shape.
    - Count how many pixels were added for each component, sort components by that count in ascending order, and recolor whole completed components as blue, red, green, then yellow.

    Key insight:
    - Symmetry completion is not only a repair step: the number of restored pixels encodes the hidden ordering that determines each component's final color.
    """
    g = np.asarray(grid, dtype=int)
    components = _components_of_color(g, GREY)
    ranked = []
    for coords in components:
        visible = {tuple(map(int, cell)) for cell in coords}
        center = _rounded_center(coords)
        completed = _rotation_closure(visible, center)
        ranked.append((len(completed) - len(visible), completed))

    ranked.sort(key=lambda item: item[0])
    out = np.zeros_like(g)
    for color, (_missing, cells) in zip(RANK_COLORS, ranked):
        for r, c in cells:
            out[r, c] = color
    return out


def _enumerate_orbits(side: int, center: tuple[float, float]) -> list[tuple[tuple[int, int], ...]]:
    unseen = {(r, c) for r in range(side) for c in range(side)}
    orbits = []
    while unseen:
        cell = unseen.pop()
        orbit = tuple(sorted({_rotate_cell(cell, center, turns) for turns in range(4)}))
        for pt in orbit:
            unseen.discard(pt)
        orbits.append(orbit)
    return orbits


def _orbit_adjacency(orbits: list[tuple[tuple[int, int], ...]]) -> list[set[int]]:
    owner = {}
    for idx, orbit in enumerate(orbits):
        for cell in orbit:
            owner[cell] = idx
    adj = [set() for _ in orbits]
    for cell, idx in owner.items():
        for nbr in _neighbors4(*cell):
            j = owner.get(nbr)
            if j is not None and j != idx:
                adj[idx].add(j)
                adj[j].add(idx)
    return adj


def _make_full_shape(
    rng: random.Random, max_side: int
) -> tuple[int, tuple[float, float], set[tuple[int, int]], list[tuple[tuple[int, int], ...]]]:
    for _ in range(300):
        side_choices = []
        for side in (4, 5, 6, 7):
            if side <= max_side:
                side_choices.append(side)
        side = rng.choice(side_choices)
        center = ((side - 1) / 2.0, (side - 1) / 2.0)
        orbits = _enumerate_orbits(side, center)
        adj = _orbit_adjacency(orbits)
        sizes = [len(orbit) for orbit in orbits]

        center_orbit = None
        for idx, orbit in enumerate(orbits):
            if len(orbit) == 1:
                center_orbit = idx
                break
        if center_orbit is not None and rng.random() < 0.75:
            chosen = {center_orbit}
        else:
            chosen = {
                min(
                    range(len(orbits)),
                    key=lambda i: min(abs(r - center[0]) + abs(c - center[1]) for r, c in orbits[i]),
                )
            }

        area = sum(sizes[i] for i in chosen)
        frontier = set().union(*(adj[i] for i in chosen)) - chosen
        target = rng.randint(14, 24)
        while frontier:
            candidates = [i for i in frontier if area + sizes[i] <= 24]
            if not candidates:
                break
            if area >= 12 and len(chosen) >= 4 and area >= target and rng.random() < 0.7:
                break
            candidate = rng.choice(candidates)
            chosen.add(candidate)
            area += sizes[candidate]
            frontier |= adj[candidate]
            frontier -= chosen

        cells = set()
        for idx in chosen:
            cells.update(orbits[idx])
        if not (12 <= len(cells) <= 24):
            continue
        if not _is_connected(cells):
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if max(rows) - min(rows) < 2 or max(cols) - min(cols) < 2:
            continue
        chosen_orbits = [orbits[i] for i in chosen]
        return side, center, cells, chosen_orbits
    raise RuntimeError("failed to build a full symmetric shape")


def _visible_options(
    full_cells: set[tuple[int, int]],
    chosen_orbits: list[tuple[tuple[int, int], ...]],
    center: tuple[float, float],
) -> dict[int, set[tuple[int, int]]]:
    options: dict[int, set[tuple[int, int]]] = {}
    packets: list[frozenset[tuple[int, int]]] = []
    twice_center = (int(round(center[0] * 2)), int(round(center[1] * 2)))

    for orbit in chosen_orbits:
        orbit_set = frozenset(orbit)
        packets.append(orbit_set)
        if len(orbit) == 4:
            for a, b in combinations(orbit, 2):
                if a[0] + b[0] == twice_center[0] and a[1] + b[1] == twice_center[1]:
                    packets.append(frozenset((a, b)))

    packets = list(dict.fromkeys(packets))
    for k in range(1, len(packets) + 1):
        for subset in combinations(range(len(packets)), k):
            removed: set[tuple[int, int]] = set()
            overlap = False
            for idx in subset:
                packet = packets[idx]
                if removed & packet:
                    overlap = True
                    break
                removed.update(packet)
            if overlap:
                continue
            visible = full_cells - removed
            missing = len(removed)
            if not (8 <= len(visible) <= 20):
                continue
            if not _is_connected(visible):
                continue
            coords = np.array(sorted(visible), dtype=int)
            recovered_center = _rounded_center(coords)
            recovered_full = _rotation_closure(visible, recovered_center)
            if len(recovered_full) - len(visible) != missing or missing <= 0:
                continue
            options.setdefault(missing, visible)
    return options


def _sector_bounds(height: int, width: int):
    row_mid = height // 2
    col_mid = width // 2
    return [
        (1, row_mid - 3, 1, col_mid - 3),
        (1, row_mid - 3, col_mid + 2, width - 2),
        (row_mid + 2, height - 2, 1, col_mid - 3),
        (row_mid + 2, height - 2, col_mid + 2, width - 2),
    ]


def _order_signature(values: list[int], reverse: bool = False) -> tuple[int, ...]:
    return tuple(sorted(range(len(values)), key=lambda i: values[i], reverse=reverse))


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        height = rng.randint(20, 24)
        width = rng.randint(20, 24)
        grid = np.zeros((height, width), dtype=int)
        sectors = _sector_bounds(height, width)
        rng.shuffle(sectors)

        objects = []
        used_missing = set()
        ok = True
        for r0, r1, c0, c1 in sectors:
            max_side = min(r1 - r0 + 1, c1 - c0 + 1)
            placed = False
            for _shape_try in range(120):
                side, center, full_cells, chosen_orbits = _make_full_shape(rng, max_side)
                options = _visible_options(full_cells, chosen_orbits, center)
                counts = [count for count in options if count not in used_missing]
                if not counts:
                    continue
                missing = rng.choice(counts)
                visible = options[missing]
                rr = rng.randint(r0, r1 - side + 1)
                cc = rng.randint(c0, c1 - side + 1)
                objects.append(
                    {
                        "missing": missing,
                        "visible_area": len(visible),
                        "full_area": len(full_cells),
                        "cells": {(rr + r, cc + c) for r, c in visible},
                    }
                )
                used_missing.add(missing)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok or len(objects) != 4:
            continue

        for obj in objects:
            for r, c in obj["cells"]:
                grid[r, c] = GREY

        comps = _components_of_color(grid, GREY)
        if len(comps) != 4:
            continue

        recovered_missing = []
        visible_values = []
        full_values = []
        for comp in comps:
            visible = {tuple(map(int, cell)) for cell in comp}
            center = _rounded_center(comp)
            full = _rotation_closure(visible, center)
            recovered_missing.append(len(full) - len(visible))
            visible_values.append(len(visible))
            full_values.append(len(full))
        if len(set(recovered_missing)) != 4 or min(recovered_missing) <= 0:
            continue

        missing_order = _order_signature(recovered_missing)
        if missing_order in {_order_signature(visible_values), _order_signature(visible_values, reverse=True)}:
            continue
        if missing_order in {_order_signature(full_values), _order_signature(full_values, reverse=True)}:
            continue

        solved = main(grid)
        if solved.shape != grid.shape or int(np.count_nonzero(solved)) == 0:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
