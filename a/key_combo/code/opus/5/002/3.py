from __future__ import annotations

import random
from collections import deque

import numpy as np

BG = 0
BAR_COLORS = (1, 2, 3)
STRIPE_CYCLE = (2, 4, 1)
WIRE = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
GEN_RETRIES = 300


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        dr1 = path[i - 1][0] - path[i - 2][0]
        dc1 = path[i - 1][1] - path[i - 2][1]
        dr2 = path[i][0] - path[i - 1][0]
        dc2 = path[i][1] - path[i - 1][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _extract_bars(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    bars: list[dict] = []
    cell_to_bar: dict[tuple[int, int], int] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, WIRE)):
        comps = _components(grid == color)
        for comp in comps:
            rows = {r for r, _ in comp}
            cols = sorted(c for _, c in comp)
            if len(rows) != 1 or cols != list(range(cols[0], cols[-1] + 1)):
                raise AssertionError("colored components must be horizontal solid bars")
            bar = {
                "color": color,
                "cells": tuple(sorted(comp)),
                "row": next(iter(rows)),
                "c0": cols[0],
                "c1": cols[-1],
                "length": len(comp),
            }
            idx = len(bars)
            bars.append(bar)
            for cell in comp:
                cell_to_bar[cell] = idx
    return bars, cell_to_bar


def _ordered_path(cells: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    cellset = set(cells)
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [p for p in neighbors4(cur[0], cur[1]) if p in cellset and p != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise AssertionError("wire must be an unbranched simple path")
        prev, cur = cur, nxts[0]
        path.append(cur)
    if len(path) != len(cells):
        raise AssertionError("wire tracing did not cover the full component")
    return path


def _parse_instance(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bars, cell_to_bar = _extract_bars(g)
    wires: list[dict] = []
    for comp in _components(g == WIRE):
        cellset = set(comp)
        deg = {
            cell: sum(1 for nxt in neighbors4(cell[0], cell[1]) if nxt in cellset)
            for cell in cellset
        }
        endpoints = [cell for cell, d in deg.items() if d == 1]
        if len(endpoints) != 2:
            raise AssertionError("each wire must have exactly two endpoints")

        source: tuple[int, int] | None = None
        target: tuple[int, int] | None = None
        bar_idx: int | None = None
        for ep in endpoints:
            touching = {cell_to_bar[n] for n in neighbors4(ep[0], ep[1]) if n in cell_to_bar}
            if touching:
                if len(touching) != 1:
                    raise AssertionError("an endpoint may touch only one bar")
                if source is not None:
                    raise AssertionError("a wire may have only one source endpoint")
                source = ep
                bar_idx = next(iter(touching))
            else:
                target = ep
        if source is None or target is None or bar_idx is None:
            raise AssertionError("each wire needs one bar-touching endpoint and one bare endpoint")

        path = _ordered_path(comp, source)
        if path[-1] != target:
            raise AssertionError("wire order must run from source to the opposite endpoint")

        wires.append(
            {
                "cells": tuple(comp),
                "path": path,
                "source": source,
                "target": target,
                "bar": bars[bar_idx],
                "energy": int(bars[bar_idx]["length"]),
            }
        )
    return wires


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a red-yellow-blue stripe pattern along each grey wire for a bar-length-limited number of cycles.

    Input structure:
    - Black background contains several disjoint grey single-pixel 4-connected wire paths with bends but no branches.
    - Exactly one endpoint of each wire is adjacent to a horizontal monochrome bar in blue, red, or green.
    - Each bar length encodes an energy value, while the opposite wire endpoint has no adjacent bar.

    Rule:
    - Identify every grey wire component and its two degree-1 endpoints.
    - Use the endpoint adjacent to a horizontal colored bar as the source, and read that bar's length L as the wire energy.
    - Traverse the wire from source to target, recoloring path cells in the repeating order red(2), yellow(4), blue(1).
    - Recolor at most 3*L path cells; once L full 3-cell stripe cycles have been painted, leave the remaining wire cells grey.
    - Colored bars themselves stay unchanged.

    Key insight:
    - The bar does not set a distance directly: its visible length gives the number of full 3-color stripe cycles that may travel along the actual winding wire path.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for wire in _parse_instance(g):
        limit = min(len(wire["path"]), 3 * wire["energy"])
        for i in range(limit):
            r, c = wire["path"][i]
            out[r, c] = STRIPE_CYCLE[i % len(STRIPE_CYCLE)]
    return out


def _sample_specs() -> list[dict]:
    partial_energy = random.choice([2, 3, 4, 5])
    partial_len = random.randint(max(10, 3 * partial_energy + 1), min(20, 3 * partial_energy + 5))

    full_energy = random.choice([4, 5])
    full_len = random.randint(10, min(20, 3 * full_energy - 1))

    boundary_energy = random.choice([4, 5])
    boundary_delta = random.choice([-1, 0, 1])
    boundary_len = 3 * boundary_energy + boundary_delta
    boundary_len = max(10, min(20, boundary_len))

    specs = [
        {"energy": partial_energy, "path_len": partial_len, "min_turns": 2},
        {"energy": full_energy, "path_len": full_len, "min_turns": 1},
        {"energy": boundary_energy, "path_len": boundary_len, "min_turns": random.choice([0, 1, 2])},
    ]
    random.shuffle(specs)
    return specs


def _clear_of_existing(cell: tuple[int, int], shape: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    if not (1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1):
        return False
    if cell in occupied:
        return False
    for nr, nc in neighbors8(r, c):
        if (nr, nc) in occupied:
            return False
    return True


def _valid_next(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    used: set[tuple[int, int]],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    if not _clear_of_existing(nxt, shape, occupied):
        return False
    if nxt in used:
        return False
    neighbor_count = sum(1 for nb in neighbors4(nxt[0], nxt[1]) if nb in used)
    return neighbor_count == 1 and cur in set(neighbors4(nxt[0], nxt[1]))


def _find_path(
    shape: tuple[int, int],
    target_len: int,
    occupied: set[tuple[int, int]],
    min_turns: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    starts = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(starts)

    def dfs(
        path: list[tuple[int, int]],
        used: set[tuple[int, int]],
        prev_dir: tuple[int, int] | None,
        turns: int,
    ) -> list[tuple[int, int]] | None:
        if len(path) == target_len:
            return list(path) if turns >= min_turns else None

        cur = path[-1]
        dirs = DIR4[:]
        random.shuffle(dirs)
        if prev_dir is not None and prev_dir in dirs:
            dirs.remove(prev_dir)
            dirs.insert(0, prev_dir)

        for dr, dc in dirs:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not _valid_next(nxt, cur, used, shape, occupied):
                continue
            next_turns = turns + int(prev_dir is not None and prev_dir != (dr, dc))
            path.append(nxt)
            used.add(nxt)
            result = dfs(path, used, (dr, dc), next_turns)
            if result is not None:
                return result
            used.remove(nxt)
            path.pop()
        return None

    for start in starts[:120]:
        if not _clear_of_existing(start, shape, occupied):
            continue
        result = dfs([start], {start}, None, 0)
        if result is not None:
            return result
    return None


def _bar_options(
    path: list[tuple[int, int]],
    energy: int,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    path_set = set(path)
    options: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for endpoint in (path[0], path[-1]):
        r, c = endpoint
        for sign in (-1, 1):
            cells = [(r, c + sign * k) for k in range(1, energy + 1)]
            if not all(in_bounds(shape, rr, cc) for rr, cc in cells):
                continue
            if any(cell in occupied or cell in path_set for cell in cells):
                continue
            ok = True
            for rr, cc in cells:
                for nr, nc in neighbors8(rr, cc):
                    if (nr, nc) in occupied:
                        ok = False
                        break
                if not ok:
                    break
                touching_path = {nb for nb in neighbors4(rr, cc) if nb in path_set}
                if touching_path - {endpoint}:
                    ok = False
                    break
            if ok:
                options.append((endpoint, cells))
    return options


def _paint_wire(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for r, c in path:
        grid[r, c] = WIRE


def _paint_bar(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    wires = _parse_instance(g)
    bars, _ = _extract_bars(g)
    assert len(wires) == 3
    assert len(bars) == 3
    assert sorted(bar["color"] for bar in bars) == [1, 2, 3]

    turns = []
    full_cover = 0
    partial_cover = 0
    near_boundary = 0
    out = main(g)
    for wire in wires:
        path = wire["path"]
        energy = wire["energy"]
        assert 10 <= len(path) <= 20
        assert 2 <= energy <= 5
        turns.append(_turn_count(path))

        limit = min(len(path), 3 * energy)
        if len(path) < 3 * energy:
            full_cover += 1
        if len(path) > 3 * energy:
            partial_cover += 1
        if abs(len(path) - 3 * energy) <= 1:
            near_boundary += 1

        for i, (r, c) in enumerate(path):
            expected = STRIPE_CYCLE[i % 3] if i < limit else WIRE
            assert int(out[r, c]) == expected

    assert sum(t > 0 for t in turns) >= 2
    assert full_cover >= 1
    assert partial_cover >= 1
    assert near_boundary >= 1


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(20, 26)
    w = random.randint(20, 26)
    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    colors = list(BAR_COLORS)
    random.shuffle(colors)

    for spec, color in zip(_sample_specs(), colors):
        placed = False
        for _ in range(160):
            path = _find_path(grid.shape, spec["path_len"], occupied, spec["min_turns"])
            if path is None:
                continue
            options = _bar_options(path, spec["energy"], grid.shape, occupied)
            if not options:
                continue
            _endpoint, bar_cells = random.choice(options)
            _paint_wire(grid, path)
            _paint_bar(grid, bar_cells, color)
            occupied.update(path)
            occupied.update(bar_cells)
            placed = True
            break
        if not placed:
            return None

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wire-stripe instance")
