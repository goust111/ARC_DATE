import random
from collections import Counter, deque

import numpy as np


BG = 0
RING = 5
ARROW = 8
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 9]
SIZE_CONFIG = {
    5: {"max_gears": 6, "max_side": 4},
    7: {"max_gears": 5, "max_side": 3},
    9: {"max_gears": 4, "max_side": 2},
}
MAX_GEN_ATTEMPTS = 400
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_DIRS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = ORTHO_DIRS if connectivity == 4 else EIGHT_DIRS
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
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_arrow(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color, connectivity=4):
            if len(comp) != 3:
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                continue
            local = np.zeros((2, 2), dtype=bool)
            for r, c in comp:
                local[r - r0, c - c0] = True
            if local.sum() != 3:
                continue
            missing = np.argwhere(~local)
            if len(missing) != 1:
                continue
            miss_r, miss_c = map(int, missing[0])
            candidates.append(
                {
                    "color": color,
                    "cells": comp,
                    "bbox": (r0, c0, r1, c1),
                    "clockwise": miss_c == 1,
                    "missing": (miss_r, miss_c),
                }
            )
    assert len(candidates) == 1, "expected exactly one 3-pixel L-shaped arrow"
    return candidates[0]


def _trace_cycle(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_set = set(cells)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        nbrs = []
        for dr, dc in ORTHO_DIRS:
            nxt = (r + dr, c + dc)
            if nxt in cell_set:
                nbrs.append(nxt)
        assert len(nbrs) == 2, "gear ring must form a single-pixel cycle"
        adj[(r, c)] = sorted(nbrs)

    start = min(cell_set)
    prev = None
    curr = start
    seq = [start]
    while True:
        nbrs = adj[curr]
        if prev is None:
            nxt = nbrs[0]
        else:
            nxts = [n for n in nbrs if n != prev]
            assert nxts, "cycle tracing failed"
            nxt = nxts[0]
        if nxt == start:
            assert len(seq) == len(cell_set), "cycle closed too early"
            break
        prev, curr = curr, nxt
        seq.append(curr)
    return seq


def _sequence_is_clockwise(seq: list[tuple[int, int]]) -> bool:
    area = 0
    wrapped = seq[1:] + seq[:1]
    for (r1, c1), (r2, c2) in zip(seq, wrapped):
        x1, y1 = c1, -r1
        x2, y2 = c2, -r2
        area += x1 * y2 - x2 * y1
    return area < 0


def _parse_gears(grid: np.ndarray, bg: int, arrow: dict) -> tuple[int, list[dict]]:
    gear_mask = grid != bg
    for r, c in arrow["cells"]:
        gear_mask[r, c] = False
    comps = _components(gear_mask, connectivity=4)
    assert comps, "expected at least one gear"

    colors = [int(grid[r, c]) for comp in comps for r, c in comp]
    ring_color = int(Counter(colors).most_common(1)[0][0])

    gears = []
    for comp in comps:
        marker_cells = [(r, c) for r, c in comp if int(grid[r, c]) != ring_color]
        assert len(marker_cells) == 1, "each gear must contain exactly one colored tooth"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        cycle = _trace_cycle(comp)
        gears.append(
            {
                "cells": comp,
                "cell_set": set(comp),
                "bbox": bbox,
                "marker": marker_cells[0],
                "marker_color": int(grid[marker_cells[0]]),
                "cycle": cycle,
            }
        )
    return ring_color, gears


def _mesh(gear_a: dict, gear_b: dict, grid: np.ndarray, bg: int) -> bool:
    h, w = grid.shape
    other = gear_b["cell_set"]
    for r, c in gear_a["cells"]:
        for dr, dc in ORTHO_DIRS:
            mr, mc = r + dr, c + dc
            nr, nc = r + 2 * dr, c + 2 * dc
            if not (0 <= mr < h and 0 <= mc < w and 0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in other and int(grid[mr, mc]) == bg:
                return True
    return False


def _parse_instance(grid: np.ndarray) -> tuple[int, int, dict, list[dict], list[list[int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    arrow = _find_arrow(g, bg)
    ring_color, gears = _parse_gears(g, bg, arrow)
    edges = [[] for _ in gears]
    for i in range(len(gears)):
        for j in range(i + 1, len(gears)):
            if _mesh(gears[i], gears[j], g, bg):
                edges[i].append(j)
                edges[j].append(i)
    return bg, ring_color, arrow, gears, edges


def _driver_index(gears: list[dict], arrow: dict) -> int:
    distances = []
    for idx, gear in enumerate(gears):
        dist = min(
            abs(r1 - r2) + abs(c1 - c2)
            for r1, c1 in arrow["cells"]
            for r2, c2 in gear["cells"]
        )
        distances.append((dist, idx))
    distances.sort()
    return int(distances[0][1])


def _propagate_directions(edges: list[list[int]], start: int, clockwise: bool) -> list[int]:
    directions = [0] * len(edges)
    directions[start] = 1 if clockwise else -1
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in edges[node]:
            want = -directions[node]
            if directions[nxt] == 0:
                directions[nxt] = want
                q.append(nxt)
            else:
                assert directions[nxt] == want, "gear graph has inconsistent parity"
    return directions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the marked tooth on each meshed gear by one step after propagating
    alternating directions from the L-shaped driver marker.
    """
    g = np.asarray(grid, dtype=int)
    bg, ring_color, arrow, gears, edges = _parse_instance(g)
    driver = _driver_index(gears, arrow)
    directions = _propagate_directions(edges, driver, arrow["clockwise"])

    out = g.copy()
    for gear, direction in zip(gears, directions):
        if direction == 0:
            continue
        cycle = gear["cycle"]
        shift = direction if _sequence_is_clockwise(cycle) else -direction
        idx = cycle.index(gear["marker"])
        nr, nc = cycle[(idx + shift) % len(cycle)]
        mr, mc = gear["marker"]
        out[mr, mc] = ring_color
        out[nr, nc] = gear["marker_color"]

    _ = bg  # keep inferred background as an explicit parsed quantity
    return out


def _square_cycle(size: int) -> list[tuple[int, int]]:
    cycle = []
    for c in range(size):
        cycle.append((0, c))
    for r in range(1, size):
        cycle.append((r, size - 1))
    for c in range(size - 2, -1, -1):
        cycle.append((size - 1, c))
    for r in range(size - 2, 0, -1):
        cycle.append((r, 0))
    return cycle


def _grow_shape(count: int, max_r: int, max_c: int) -> set[tuple[int, int]] | None:
    cells = {(0, 0)}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if nr < max_r and nc < max_c and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(0, 0)
    while len(cells) < count and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        cells.add((r, c))
        add_frontier(r, c)
    if len(cells) != count:
        return None
    return cells


def _transform_cell(cell: tuple[int, int], corner: str, max_r: int, max_c: int) -> tuple[int, int]:
    r, c = cell
    if "bottom" in corner:
        r = max_r - r
    if "right" in corner:
        c = max_c - c
    return r, c


def _arrow_pixels(h: int, w: int, corner: str, clockwise: bool) -> list[tuple[int, int]]:
    if corner == "top-left":
        r0, c0 = 0, 0
    elif corner == "top-right":
        r0, c0 = 0, w - 2
    elif corner == "bottom-left":
        r0, c0 = h - 2, 0
    elif corner == "bottom-right":
        r0, c0 = h - 2, w - 2
    else:
        raise AssertionError(corner)

    miss_c = 1 if clockwise else 0
    miss_r = random.randint(0, 1)
    pixels = []
    for dr in range(2):
        for dc in range(2):
            if dr == miss_r and dc == miss_c:
                continue
            pixels.append((r0 + dr, c0 + dc))
    return pixels


def _is_connected(edges: list[list[int]]) -> bool:
    seen = {0}
    q = deque([0])
    while q:
        node = q.popleft()
        for nxt in edges[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(edges)


def _gen_attempt() -> np.ndarray | None:
    size = random.choice([5, 7, 9])
    cfg = SIZE_CONFIG[size]
    max_r = random.randint(2, cfg["max_side"])
    max_c = random.randint(2, cfg["max_side"])
    max_gears = min(cfg["max_gears"], max_r * max_c)
    count = random.randint(2, max_gears)
    shape = _grow_shape(count, max_r, max_c)
    if shape is None:
        return None

    corner = random.choice(["top-left", "top-right", "bottom-left", "bottom-right"])
    clockwise = random.choice([False, True])
    marker_colors = random.sample(MARKER_COLORS, count)

    raw_max_r = max(r for r, _ in shape)
    raw_max_c = max(c for _, c in shape)
    transformed = {
        cell: _transform_cell(cell, corner, raw_max_r, raw_max_c)
        for cell in shape
    }

    placed = list(transformed.values())
    max_row = max(r for r, _ in placed)
    max_col = max(c for _, c in placed)
    span = size + 1
    cluster_h = size + max_row * span
    cluster_w = size + max_col * span

    top_pad = random.randint(3, 5) if "top" in corner else random.randint(1, 4)
    bottom_pad = random.randint(3, 5) if "bottom" in corner else random.randint(1, 4)
    left_pad = random.randint(3, 5) if "left" in corner else random.randint(1, 4)
    right_pad = random.randint(3, 5) if "right" in corner else random.randint(1, 4)

    h = top_pad + cluster_h + bottom_pad
    w = left_pad + cluster_w + right_pad
    grid = np.full((h, w), BG, dtype=int)

    cycle = _square_cycle(size)
    for color, cell in zip(marker_colors, sorted(shape)):
        rr, cc = transformed[cell]
        r0 = top_pad + rr * span
        c0 = left_pad + cc * span
        grid[r0 : r0 + size, c0 : c0 + size] = BG
        grid[r0, c0 : c0 + size] = RING
        grid[r0 + size - 1, c0 : c0 + size] = RING
        grid[r0 : r0 + size, c0] = RING
        grid[r0 : r0 + size, c0 + size - 1] = RING
        mr, mc = random.choice(cycle)
        grid[r0 + mr, c0 + mc] = color

    for r, c in _arrow_pixels(h, w, corner, clockwise):
        grid[r, c] = ARROW

    try:
        bg, ring_color, arrow, gears, edges = _parse_instance(grid)
        if bg != BG or ring_color != RING or len(gears) != count:
            return None
        if not all(edges[i] for i in range(len(edges))):
            return None
        if not _is_connected(edges):
            return None
        driver = _driver_index(gears, arrow)
        dists = sorted(
            min(
                abs(r1 - r2) + abs(c1 - c2)
                for r1, c1 in arrow["cells"]
                for r2, c2 in gear["cells"]
            )
            for gear in gears
        )
        if len(dists) > 1 and dists[0] == dists[1]:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
        _ = driver
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gear-rotation input")
