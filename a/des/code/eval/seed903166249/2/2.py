import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    groups: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            groups.append(coords)
    return groups


def _square_cells(top: int, left: int) -> tuple[tuple[int, int], ...]:
    return (
        (top, left),
        (top + 1, left),
        (top, left + 1),
        (top + 1, left + 1),
    )


def _external_neighbors(
    square: tuple[tuple[int, int], ...],
    comp_set: set[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    square_set = set(square)
    ext: set[tuple[int, int]] = set()
    for r, c in square:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in comp_set and (nr, nc) not in square_set:
                ext.add((nr, nc))
    return ext


def _order_path(
    path_set: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    if len(path_set) == 1:
        only = next(iter(path_set))
        assert only == start == end
        return [only]

    degrees: dict[tuple[int, int], int] = {}
    for cell in path_set:
        r, c = cell
        degrees[cell] = sum((nr, nc) in path_set for nr, nc in _neighbors4(r, c, h, w))

    endpoints = {cell for cell, deg in degrees.items() if deg == 1}
    middles = {cell for cell, deg in degrees.items() if deg == 2}
    assert endpoints == {start, end}
    assert len(middles) + len(endpoints) == len(path_set)

    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        r, c = cur
        nxt = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if (nr, nc) in path_set and (nr, nc) != prev]
        assert len(nxt) == 1
        prev, cur = cur, nxt[0]
        ordered.append(cur)
        assert len(ordered) <= len(path_set)

    assert len(ordered) == len(path_set)
    return ordered


def _parse_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    h, w = grid.shape
    comp_set = set(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]

    tx_candidates: list[dict[str, object]] = []
    rx_candidates: list[dict[str, object]] = []
    for top in range(min(rows), max(rows)):
        for left in range(min(cols), max(cols)):
            square = _square_cells(top, left)
            if any(cell not in comp_set for cell in square):
                continue
            color = int(grid[top, left])
            if any(int(grid[r, c]) != color for r, c in square):
                continue
            ext = _external_neighbors(square, comp_set, h, w)
            if len(ext) != 1:
                continue
            cand = {
                "top": top,
                "left": left,
                "color": color,
                "cells": square,
                "port": next(iter(ext)),
            }
            if color == GREY:
                rx_candidates.append(cand)
            elif color != BG:
                tx_candidates.append(cand)

    assert len(tx_candidates) == 1
    assert len(rx_candidates) == 1
    tx = tx_candidates[0]
    rx = rx_candidates[0]
    tx_cells = set(tx["cells"])
    rx_cells = set(rx["cells"])
    assert tx_cells.isdisjoint(rx_cells)

    path_set = comp_set - tx_cells - rx_cells
    assert path_set

    start = tx["port"]
    end = rx["port"]
    assert start in path_set and end in path_set

    ordered = _order_path(path_set, start, end, h, w)
    base_color = int(grid[start])
    assert int(grid[end]) == base_color

    return {
        "tx_top": int(tx["top"]),
        "tx_left": int(tx["left"]),
        "tx_color": int(tx["color"]),
        "rx_top": int(rx["top"]),
        "rx_left": int(rx["left"]),
        "path": ordered,
        "base_color": base_color,
    }


def _parse_grid(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components(g != BG)
    assert comps
    parsed = [_parse_component(g, comp) for comp in comps]
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each signal from its solid 2x2 transmitter along a single-pixel path,
    updating the current signal color at relay pixels and repainting the grey
    2x2 receiver with the final signal color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _parse_grid(g):
        signal = int(comp["tx_color"])
        base = int(comp["base_color"])
        for r, c in comp["path"]:
            cell_color = int(g[r, c])
            if cell_color != base:
                out[r, c] = signal
                signal = cell_color
        rx_top = int(comp["rx_top"])
        rx_left = int(comp["rx_left"])
        out[rx_top : rx_top + 2, rx_left : rx_left + 2] = signal

    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
PATH_COUNT_RANGE = (1, 3)
COMPONENT_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def _port_candidates(top: int, left: int, h: int, w: int) -> list[tuple[int, int]]:
    ports: list[tuple[int, int]] = []
    if top > 0:
        ports.extend([(top - 1, left), (top - 1, left + 1)])
    if top + 2 < h:
        ports.extend([(top + 2, left), (top + 2, left + 1)])
    if left > 0:
        ports.extend([(top, left - 1), (top + 1, left - 1)])
    if left + 2 < w:
        ports.extend([(top, left + 2), (top + 1, left + 2)])
    random.shuffle(ports)
    return ports


def _make_monotone_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    assert dr != 0 or dc != 0

    steps = ["v"] * abs(dr) + ["h"] * abs(dc)
    random.shuffle(steps)

    r, c = start
    vr = 0 if dr == 0 else (1 if dr > 0 else -1)
    hc = 0 if dc == 0 else (1 if dc > 0 else -1)
    path = [start]
    for step in steps:
        if step == "v":
            r += vr
        else:
            c += hc
        path.append((r, c))
    assert path[-1] == end
    assert len(path) == len(set(path))
    return path


def _touching_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _adjacent_to_square(path_cells: set[tuple[int, int]], square_cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    hits: set[tuple[int, int]] = set()
    for r, c in square_cells:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in path_cells:
                hits.add((nr, nc))
    return hits


def _sample_relay_positions(path_len: int) -> list[int] | None:
    interior = list(range(1, path_len - 1))
    if not interior:
        return None
    max_count = min(3, (len(interior) + 1) // 2)
    if max_count == 0:
        return None
    target = random.randint(1, max_count)

    for _ in range(40):
        picks = sorted(random.sample(interior, target))
        if all(b - a > 1 for a, b in zip(picks, picks[1:])):
            return picks

    random.shuffle(interior)
    picks: list[int] = []
    for idx in interior:
        if all(abs(idx - prev) > 1 for prev in picks):
            picks.append(idx)
            if len(picks) == target:
                return sorted(picks)
    return None


def _place_component(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    base_color: int,
    with_relays: bool,
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    all_tops = [(r, c) for r in range(h - 1) for c in range(w - 1)]
    random.shuffle(all_tops)

    for _ in range(COMPONENT_ATTEMPTS):
        tx_top, tx_left = random.choice(all_tops)
        rx_top, rx_left = random.choice(all_tops)
        tx_square = set(_square_cells(tx_top, tx_left))
        rx_square = set(_square_cells(rx_top, rx_left))
        if tx_square & rx_square:
            continue
        if any(cell in blocked for cell in tx_square | rx_square):
            continue

        tx_ports = [p for p in _port_candidates(tx_top, tx_left, h, w) if p not in blocked]
        rx_ports = [p for p in _port_candidates(rx_top, rx_left, h, w) if p not in blocked]
        random.shuffle(tx_ports)
        random.shuffle(rx_ports)
        if not tx_ports or not rx_ports:
            continue

        for start in tx_ports:
            for end in rx_ports:
                if start == end:
                    continue
                path = _make_monotone_path(start, end)
                if len(path) < 2:
                    continue
                if with_relays and len(path) < 3:
                    continue

                path_set = set(path)
                if any(cell in blocked for cell in path_set):
                    continue
                if path_set & (tx_square | rx_square):
                    continue

                tx_hits = _adjacent_to_square(path_set, tx_square, h, w)
                rx_hits = _adjacent_to_square(path_set, rx_square, h, w)
                if tx_hits != {start} or rx_hits != {end}:
                    continue

                comp_cells = tx_square | rx_square | path_set
                if len(comp_cells) != 8 + len(path):
                    continue

                relay_positions: list[int] = []
                if with_relays:
                    relay_positions = _sample_relay_positions(len(path)) or []
                    if not relay_positions:
                        continue

                relay_colors = random.sample([c for c in PALETTE if c != base_color], len(relay_positions))
                for r, c in tx_square:
                    grid[r, c] = base_color
                for r, c in rx_square:
                    grid[r, c] = GREY
                for r, c in path:
                    grid[r, c] = base_color
                for idx, color in zip(relay_positions, relay_colors):
                    rr, cc = path[idx]
                    grid[rr, cc] = color

                try:
                    parsed = _parse_grid(grid)
                except AssertionError:
                    for r, c in comp_cells:
                        grid[r, c] = BG
                    continue

                if len(parsed) != len(_components(grid != BG)):
                    for r, c in comp_cells:
                        grid[r, c] = BG
                    continue

                return _touching_cells(comp_cells, h, w)

    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()

    max_paths = PATH_COUNT_RANGE[1] if h * w >= 250 else 2
    target_paths = random.randint(PATH_COUNT_RANGE[0], max_paths)
    base_colors = random.sample(PALETTE, target_paths)

    for base_color in base_colors:
        with_relays = bool(random.getrandbits(1))
        footprint = _place_component(grid, blocked, base_color, with_relays)
        if footprint is None:
            return None
        blocked |= footprint

    try:
        parsed = _parse_grid(grid)
        out = main(grid)
    except AssertionError:
        return None

    if len(parsed) != target_paths:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
