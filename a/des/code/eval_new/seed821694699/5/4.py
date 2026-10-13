import numpy as np
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

BG = 0
YELLOW = 10
GRAY = 5
MAROON = 9
ORANGE = 11
SIGNAL_COLORS = {MAROON, ORANGE}
RESERVED_COLORS = {BG, GRAY, YELLOW} | SIGNAL_COLORS


def _neighbors4(h: int, w: int, r: int, c: int) -> Iterable[Tuple[int, int]]:
    """Yield 4-neighbours for a cell inside the bounds."""
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _decode_legend(grid: np.ndarray) -> Tuple[Dict[int, int], Optional[List[Tuple[int, int]]]]:
    """Decode 2x2 legend from one corner.

    Returns
    -------
    mapping
        lamp color -> segment color.
    cells
        the 2x2 coordinates of the legend area.
    """
    h, w = grid.shape
    if h < 2 or w < 2:
        return {}, None

    corners = [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]
    candidates = []

    for r0, c0 in corners:
        ok = True
        rows = [((r0, c0), (r0, c0 + 1)), ((r0 + 1, c0), (r0 + 1, c0 + 1))]
        mapping: Dict[int, int] = {}
        for (r1, c1), (r2, c2) in rows:
            a = int(grid[r1, c1])
            b = int(grid[r2, c2])
            if a in SIGNAL_COLORS and b not in RESERVED_COLORS:
                mapping[a] = b
            elif b in SIGNAL_COLORS and a not in RESERVED_COLORS:
                mapping[b] = a
            else:
                ok = False
                break
        if not ok:
            continue
        score = sum(1 for lamp in mapping if lamp in SIGNAL_COLORS)
        if len(mapping) == 2:
            score += 1
        candidates.append((score, r0, c0, mapping))

    if not candidates:
        return {}, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, r0, c0, mapping = candidates[0]
    cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
    return mapping, cells


def _collect_components(grid: np.ndarray, path_colors: Set[int]):
    """Collect connected components of all non-reserved path colors.

    Returns
    -------
    dict[color] -> list[component]
    where component has fields:
      id: (color, index)
      color
      cells: set[(r, c)]
      endpoints: list[(r, c)]
    """
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: Dict[int, List[dict]] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in path_colors or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: Set[Tuple[int, int]] = set()

            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for nr, nc in _neighbors4(h, w, cr, cc):
                    if seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            endpoints: List[Tuple[int, int]] = []
            for cr, cc in cells:
                degree = 0
                for nr, nc in _neighbors4(h, w, cr, cc):
                    if (nr, nc) in cells:
                        degree += 1
                if degree <= 1:
                    endpoints.append((cr, cc))

            comp_id = (color, len(components.get(color, [])))
            components.setdefault(color, []).append(
                {
                    "id": comp_id,
                    "color": color,
                    "cells": cells,
                    "endpoints": endpoints,
                }
            )

    return components


def _component_of_pixel(
    components: Dict[int, List[dict]],
    r: int,
    c: int,
    color: int,
) -> Optional[dict]:
    for comp in components.get(color, []):
        if (r, c) in comp["cells"]:
            return comp
    return None


def _other_endpoint(endpoints: List[Tuple[int, int]], origin: Tuple[int, int]) -> Tuple[int, int]:
    if not endpoints:
        return origin
    if len(endpoints) == 1:
        return endpoints[0]
    if origin == endpoints[0]:
        return endpoints[1]
    if origin == endpoints[1]:
        return endpoints[0]
    return endpoints[0]


def _bfs_bridge(
    grid: np.ndarray,
    start: Tuple[int, int],
    target: Tuple[int, int],
    path_mask: np.ndarray,
    max_steps: int,
) -> Optional[List[Tuple[int, int]]]:
    """Shortest path through bg/signal cells, avoiding already used path pixels."""
    if start == target:
        return [start]

    h, w = grid.shape
    q = deque([start])
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = {start}

    while q:
        cr, cc = q.popleft()
        if len(parent) > (h * w):
            return None

        for nr, nc in _neighbors4(h, w, cr, cc):
            if (nr, nc) in visited:
                continue
            if (nr, nc) != target and path_mask[nr, nc]:
                continue
            cell = int(grid[nr, nc])
            if (nr, nc) != target and cell != BG and cell not in SIGNAL_COLORS:
                continue

            visited.add((nr, nc))
            parent[(nr, nc)] = (cr, cc)
            if (nr, nc) == target:
                path = [(nr, nc)]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return path[::-1]

            if len(parent) > max_steps:
                continue
            q.append((nr, nc))

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Solve one instance by decoding legend and tracing lamps-and-segments.

    Parameters
    ----------
    grid:
        Input ARC grid with integer colors.

    Returns
    -------
    numpy.ndarray:
        Reconstructed grid with only the linked chain and gray bridges preserved.

    Approach
    --------
    1) Decode the corner legend into lamp->path-color mapping.
    2) Find all path components by color and locate start from Yellow.
    3) Iteratively, read lamp at current endpoint, locate next component,
       bridge through bg/signal area, mark component as kept and continue.
    4) Clear the legend area and leave all unlinked path as background.

    Notes
    -----
    Input/output are integer-encoded ARC grids; implementation is self-contained.
    """
    inp = np.array(grid, dtype=int)
    h, w = inp.shape

    mapping, legend_cells = _decode_legend(inp)
    if not mapping:
        return np.zeros((h, w), dtype=int)

    path_colors = set(int(x) for x in np.unique(inp)) - RESERVED_COLORS
    components = _collect_components(inp, path_colors)

    starts = np.argwhere(inp == YELLOW)
    if len(starts) == 0:
        return np.zeros((h, w), dtype=int)
    sr, sc = map(int, starts[0])

    start_component: Optional[dict] = None
    start_attachment: Optional[Tuple[int, int]] = None
    for nr, nc in _neighbors4(h, w, sr, sc):
        color = int(inp[nr, nc])
        if color not in path_colors:
            continue
        comp = _component_of_pixel(components, nr, nc, color)
        if comp is not None:
            start_component = comp
            start_attachment = (nr, nc)
            break

    if start_component is None or start_attachment is None:
        return np.zeros((h, w), dtype=int)

    out = np.zeros((h, w), dtype=int)
    for rr, cc in start_component["cells"]:
        out[rr, cc] = start_component["color"]
    out[sr, sc] = start_component["color"]

    visited_components: Set[Tuple[int, int]] = {start_component["id"]}
    path_mask = np.zeros((h, w), dtype=bool)
    for rr, cc in start_component["cells"]:
        path_mask[rr, cc] = True
    path_mask[sr, sc] = True

    current_endpoint = _other_endpoint(start_component["endpoints"], start_attachment)
    step = 0
    max_steps = h * w

    while step < max_steps:
        step += 1

        signal_options: List[int] = []
        for nr, nc in _neighbors4(h, w, *current_endpoint):
            c = int(inp[nr, nc])
            if c in mapping:
                signal_options.append(c)

        if not signal_options:
            break

        best: Optional[Tuple[int, List[Tuple[int, int]], dict]] = None
        for lamp in signal_options:
            next_color = mapping[lamp]
            for comp in components.get(next_color, []):
                comp_id = comp["id"]
                if comp_id in visited_components:
                    continue

                for ep in comp["endpoints"]:
                    bridge = _bfs_bridge(inp, current_endpoint, ep, path_mask, max_steps)
                    if bridge is None:
                        continue
                    if best is None or len(bridge) < best[0]:
                        best = (len(bridge), bridge, comp)

        if best is None:
            break

        _, bridge, nxt_comp = best

        for rr, cc in bridge[1:-1]:
            if inp[rr, cc] == BG:
                out[rr, cc] = GRAY

        for rr, cc in nxt_comp["cells"]:
            out[rr, cc] = nxt_comp["color"]
            path_mask[rr, cc] = True

        visited_components.add(nxt_comp["id"])
        target_endpoint = bridge[-1]
        current_endpoint = _other_endpoint(nxt_comp["endpoints"], target_endpoint)

    if legend_cells is not None:
        for rr, cc in legend_cells:
            out[rr, cc] = BG

    return out


def generate_input() -> np.ndarray:
    """Generate a diverse random puzzle input matching the stated ARC format."""
    rng = np.random.default_rng()
    h = int(rng.integers(12, 25))
    w = int(rng.integers(12, 25))
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    colors = [2, 3, 4, 6, 7, 8, 12, 13, 14, 15]
    rng.shuffle(colors)
    path_a, path_b = colors[0], colors[1]

    corners = [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]
    r0, c0 = corners[int(rng.integers(4))]
    lamp_a, lamp_b = MAROON, ORANGE
    if rng.random() < 0.5:
        lamp_a, lamp_b = lamp_b, lamp_a

    if rng.random() < 0.5:
        grid[r0, c0] = lamp_a
        grid[r0, c0 + 1] = path_a
        grid[r0 + 1, c0] = lamp_b
        grid[r0 + 1, c0 + 1] = path_b
    else:
        grid[r0, c0] = path_a
        grid[r0, c0 + 1] = lamp_a
        grid[r0 + 1, c0] = path_b
        grid[r0 + 1, c0 + 1] = lamp_b

    occupied[r0 : r0 + 2, c0 : c0 + 2] = True

    # helper
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def free(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and not occupied[r, c]

    def place_segment(anchor_r: int, anchor_c: int, color: int, dir_idx: int, length: int) -> Optional[List[Tuple[int, int]]]:
        dr, dc = dirs[dir_idx]
        cells: List[Tuple[int, int]] = []
        for k in range(1, length + 1):
            rr = anchor_r + dr * k
            cc = anchor_c + dc * k
            if not free(rr, cc):
                return None
            cells.append((rr, cc))

        for rr, cc in cells:
            grid[rr, cc] = color
            occupied[rr, cc] = True
        return cells

    # build a main chain with alternating mapped colors.
    chain_colors = [path_a, path_b]
    chain_length = int(rng.integers(2, 5))
    created = False
    current_r = current_c = -1

    for _ in range(500):
        anchor_r = int(rng.integers(1, h - 1))
        anchor_c = int(rng.integers(1, w - 1))
        if not free(anchor_r, anchor_c):
            continue
        if (anchor_r, anchor_c) in ((r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)):
            continue

        # ensure at least one direction can host a short segment
        ok = False
        dir_idx = int(rng.integers(4))
        for _ in range(12):
            dir_idx = int(rng.integers(4))
            dr, dc = dirs[dir_idx]
            if all(free(anchor_r + dr * k, anchor_c + dc * k) for k in range(1, 4)):
                ok = True
                break
        if not ok:
            continue

        cells = place_segment(anchor_r, anchor_c, chain_colors[0], dir_idx, int(rng.integers(2, 6)))
        if cells is None:
            continue

        grid[anchor_r, anchor_c] = YELLOW
        occupied[anchor_r, anchor_c] = True
        created = True
        current_r, current_c = cells[-1]
        current_dir = dir_idx
        break

    if created:
        inv = {v: k for k, v in {lamp_a: path_a, lamp_b: path_b}.items()}
        color_idx = 1
        for _ in range(chain_length - 1):
            next_color = chain_colors[color_idx % len(chain_colors)]
            color_idx += 1
            # all directions except one returning to prior segment naturally blocked by occupancy
            free_neighbors = [(di, anchor_r := current_r + dirs[di][0], anchor_c := current_c + dirs[di][1]) for di in range(4) if free(current_r + dirs[di][0], current_c + dirs[di][1])]
            if len(free_neighbors) < 2:
                break

            signal_color = inv.get(next_color)
            if signal_color is None:
                break

            placed = False
            rng.shuffle(free_neighbors)
            for i in range(len(free_neighbors)):
                if placed:
                    break
                si, sr, sc2 = free_neighbors[i]
                for j in range(len(free_neighbors)):
                    if j == i:
                        continue
                    ni, nr, nc = free_neighbors[j]

                    # place signal in one free side and segment start in another
                    rr_s, cc_s = sr, sc2
                    rr_n, cc_n = nr, nc
                    grid[rr_s, cc_s] = signal_color
                    cells = place_segment(current_r, current_c, next_color, ni, int(rng.integers(2, 6)))
                    if cells is None:
                        grid[rr_s, cc_s] = BG
                        continue

                    current_r, current_c = cells[-1]
                    current_dir = ni
                    placed = True
                    break

            if not placed:
                break

    # random extra unlinked segments as distractors.
    for idx in range(2, len(colors)):
        extra_color = colors[idx]
        placed = False
        for _ in range(120):
            sr = int(rng.integers(1, h - 1))
            sc = int(rng.integers(1, w - 1))
            if not free(sr, sc):
                continue
            dir_idx = int(rng.integers(4))
            length = int(rng.integers(2, 7))
            cells = place_segment(sr, sc, extra_color, dir_idx, length)
            if cells is not None:
                placed = True
                break
        if not placed:
            continue

    return grid

