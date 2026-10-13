import random
from collections import deque

import numpy as np


BG = 0
PATH = 5
FILTER = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
PATH_COUNT_CHOICES = (3, 4)
GEN_RETRIES = 700
PLACEMENT_TRIES = 180
WAYPOINT_TRIES = 24

SOURCE_ENDPOINT_OFFSETS = (
    (0, -1),
    (1, -1),
    (0, 2),
    (1, 2),
    (-1, 0),
    (-1, 1),
    (2, 0),
    (2, 1),
)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _corner_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]


def _find_table(grid: np.ndarray) -> tuple[dict[int, int], tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[dict[int, int], tuple[int, int]]] = []
    for r0, c0 in _corner_positions(h, w):
        patch = grid[r0 : r0 + 2, c0 : c0 + 2]
        if patch.shape != (2, 2):
            continue
        if bool(np.isin(patch, [BG, PATH, FILTER]).any()):
            continue
        if int(patch[0, 0]) == int(patch[1, 0]):
            continue
        mapping = {int(patch[0, 0]): int(patch[0, 1]), int(patch[1, 0]): int(patch[1, 1])}
        if len(mapping) != 2:
            continue
        candidates.append((mapping, (int(r0), int(c0))))
    assert len(candidates) == 1, "expected exactly one corner conversion table"
    return candidates[0]


def _find_source_blocks(grid: np.ndarray) -> list[dict]:
    blocks: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, PATH, FILTER):
            continue
        for comp in _components(grid == color):
            if int(comp.shape[0]) != 4:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                continue
            if not bool((grid[r0 : r1 + 1, c0 : c1 + 1] == color).all()):
                continue
            cells = {tuple(map(int, pt)) for pt in comp.tolist()}
            blocks.append({"color": color, "cells": cells, "bbox": (r0, c0, r1, c1)})
    return blocks


def _walk_path(cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    neighbors = {
        cell: [nbr for nbr in _neighbors4(cell[0], cell[1], h, w) if nbr in cells]
        for cell in cells
    }
    cur = start
    prev: tuple[int, int] | None = None
    order = [start]
    while cur != end:
        nxts = [nbr for nbr in neighbors[cur] if nbr != prev]
        assert len(nxts) == 1, "path must be a simple chain"
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color along its grey path, applying the maroon table mapping once if present.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, _ = _find_table(g)
    source_blocks = _find_source_blocks(g)
    assert source_blocks, "expected at least one 2x2 source block"

    source_lookup: dict[tuple[int, int], int] = {}
    for idx, block in enumerate(source_blocks):
        for cell in block["cells"]:
            source_lookup[cell] = idx

    out = g.copy()
    path_mask = (g == PATH) | (g == FILTER)
    path_components = _components(path_mask)
    assert path_components, "expected at least one path"

    for comp in path_components:
        cells = {tuple(map(int, pt)) for pt in comp.tolist()}
        degrees = {
            cell: sum((nbr in cells) for nbr in _neighbors4(cell[0], cell[1], h, w))
            for cell in cells
        }
        endpoints = [cell for cell, degree in degrees.items() if degree == 1]
        assert len(endpoints) == 2, "each path must have exactly two endpoints"

        source_endpoint = None
        source_idx = None
        for endpoint in endpoints:
            touching = {source_lookup[nbr] for nbr in _neighbors4(endpoint[0], endpoint[1], h, w) if nbr in source_lookup}
            if touching:
                assert len(touching) == 1, "endpoint may touch only one source"
                assert source_endpoint is None, "path may touch only one source endpoint"
                source_endpoint = endpoint
                source_idx = next(iter(touching))

        assert source_endpoint is not None and source_idx is not None, "each path needs one source endpoint"
        end_endpoint = endpoints[0] if endpoints[1] == source_endpoint else endpoints[1]
        order = _walk_path(cells, source_endpoint, end_endpoint)
        assert len(order) >= 2, "path must have positive length"

        signal = int(source_blocks[source_idx]["color"])
        if any(int(g[r, c]) == FILTER for r, c in order):
            assert signal in mapping, "filtered source color must appear in the table"
            signal = int(mapping[signal])

        dr = int(order[-1][0] - order[-2][0])
        dc = int(order[-1][1] - order[-2][1])
        target = (int(order[-1][0] + dr), int(order[-1][1] + dc))
        assert 0 <= target[0] < h and 0 <= target[1] < w, "output cell must stay in bounds"
        assert int(g[target]) == BG, "output cell must be a black empty cell"
        out[target] = signal

    return out


def _reserve(blocked: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in cells:
        rr0 = max(0, int(r - pad))
        rr1 = min(h - 1, int(r + pad))
        cc0 = max(0, int(c - pad))
        cc1 = min(w - 1, int(c + pad))
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_reserve(
    blocked: np.ndarray,
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    pad: int = 1,
    allow: set[tuple[int, int]] | None = None,
) -> bool:
    allow = allow or set()
    h, w = blocked.shape
    for r, c in cells:
        rr0 = max(0, int(r - pad))
        rr1 = min(h - 1, int(r + pad))
        cc0 = max(0, int(c - pad))
        cc1 = min(w - 1, int(c + pad))
        for rr in range(rr0, rr1 + 1):
            for cc in range(cc0, cc1 + 1):
                if blocked[rr, cc] and (rr, cc) not in allow:
                    return False
    return True


def _random_shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if blocked[start] or blocked[goal]:
        return None

    h, w = blocked.shape
    q = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dirs = DIRS[:]
    rng.shuffle(dirs)

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dr, dc in dirs:
            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc] or nxt in parents:
                continue
            parents[nxt] = cur
            q.append(nxt)

    if goal not in parents:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def _route_piecewise(
    points: list[tuple[int, int]],
    blocked: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    temp = blocked.copy()
    full = [points[0]]
    temp[points[0]] = False
    for a, b in zip(points, points[1:]):
        temp[a] = False
        temp[b] = False
        segment = _random_shortest_path(a, b, temp, rng)
        if segment is None:
            return None
        for cell in segment[:-1]:
            temp[cell] = True
        full.extend(segment[1:])
    return full


def _route_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    direct = _route_piecewise([start, goal], blocked, rng)
    if direct is None:
        return None
    if rng.random() < 0.55:
        return direct

    h, w = blocked.shape
    for _ in range(WAYPOINT_TRIES):
        mid = (rng.randrange(h), rng.randrange(w))
        if blocked[mid] or mid in (start, goal):
            continue
        if abs(mid[0] - start[0]) + abs(mid[1] - start[1]) < 2:
            continue
        if abs(mid[0] - goal[0]) + abs(mid[1] - goal[1]) < 2:
            continue
        routed = _route_piecewise([start, mid, goal], blocked, rng)
        if routed is not None:
            return routed
    return direct


def _place_table(grid: np.ndarray, blocked: np.ndarray, mapping_rows: list[tuple[int, int]], corner: str) -> None:
    h, w = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - 2
    elif corner == "bl":
        r0, c0 = h - 2, 0
    else:
        r0, c0 = h - 2, w - 2

    cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
    grid[r0, c0] = mapping_rows[0][0]
    grid[r0, c0 + 1] = mapping_rows[0][1]
    grid[r0 + 1, c0] = mapping_rows[1][0]
    grid[r0 + 1, c0 + 1] = mapping_rows[1][1]
    _reserve(blocked, cells, pad=1)


def _pick_source(
    blocked: np.ndarray,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    h, w = blocked.shape
    for _ in range(PLACEMENT_TRIES):
        r0 = rng.randrange(1, h - 2)
        c0 = rng.randrange(1, w - 2)
        dr, dc = SOURCE_ENDPOINT_OFFSETS[rng.randrange(len(SOURCE_ENDPOINT_OFFSETS))]
        block_cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
        start = (r0 + dr, c0 + dc)
        if not (0 <= start[0] < h and 0 <= start[1] < w):
            continue
        if blocked[start]:
            continue
        if not _can_reserve(blocked, block_cells, pad=1, allow={start}):
            continue
        return block_cells, start
    return None


def _pick_end_stub(
    blocked: np.ndarray,
    start: tuple[int, int],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    h, w = blocked.shape
    dirs = DIRS[:]
    rng.shuffle(dirs)
    for _ in range(PLACEMENT_TRIES):
        p = (rng.randrange(h), rng.randrange(w))
        if blocked[p]:
            continue
        for dr, dc in dirs:
            e = (int(p[0] + dr), int(p[1] + dc))
            o = (int(p[0] + 2 * dr), int(p[1] + 2 * dc))
            if not (0 <= e[0] < h and 0 <= e[1] < w and 0 <= o[0] < h and 0 <= o[1] < w):
                continue
            if blocked[e] or blocked[o]:
                continue
            if abs(p[0] - start[0]) + abs(p[1] - start[1]) < 5:
                continue
            if not _can_reserve(blocked, [e, o], pad=1, allow={p}):
                continue
            return p, e, o
    return None


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], filtered: bool, rng: random.Random) -> None:
    for r, c in path:
        grid[r, c] = PATH
    if filtered:
        idx_lo = 1
        idx_hi = len(path) - 2
        if idx_hi <= idx_lo:
            idx = 1
        else:
            idx = rng.randint(idx_lo, idx_hi)
        grid[path[idx]] = FILTER


def _place_single_path(
    grid: np.ndarray,
    blocked: np.ndarray,
    source_color: int,
    filtered: bool,
    rng: random.Random,
) -> bool:
    for _ in range(PLACEMENT_TRIES):
        picked_source = _pick_source(blocked, rng)
        if picked_source is None:
            return False
        block_cells, start = picked_source

        temp = blocked.copy()
        _reserve(temp, block_cells, pad=1)
        temp[start] = False

        picked_end = _pick_end_stub(temp, start, rng)
        if picked_end is None:
            continue
        pre_end, end, output = picked_end
        temp2 = temp.copy()
        _reserve(temp2, [end, output], pad=1)
        temp2[pre_end] = False

        path = _route_path(start, pre_end, temp2, rng)
        if path is None:
            continue
        path = path + [end]
        if len(path) < 4:
            continue
        if filtered and len(path) < 5:
            continue

        for r, c in block_cells:
            grid[r, c] = source_color
        _paint_path(grid, path, filtered, rng)

        _reserve(blocked, block_cells, pad=1)
        _reserve(blocked, path, pad=1)
        _reserve(blocked, [output], pad=1)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    n_paths = rng.choice(PATH_COUNT_CHOICES)
    unfiltered_count = n_paths - 2

    colors = list(PALETTE)
    rng.shuffle(colors)
    inputs = colors[:2]
    outputs = colors[2:4]
    extras = colors[4 : 4 + unfiltered_count]
    if len(extras) != unfiltered_count:
        return None

    mapping_rows = [(inputs[0], outputs[0]), (inputs[1], outputs[1])]
    corner = rng.choice(("tl", "tr", "bl", "br"))
    _place_table(grid, blocked, mapping_rows, corner)

    path_specs = [
        {"source_color": inputs[0], "filtered": True},
        {"source_color": inputs[1], "filtered": True},
    ] + [{"source_color": color, "filtered": False} for color in extras]
    rng.shuffle(path_specs)

    for spec in path_specs:
        if not _place_single_path(grid, blocked, int(spec["source_color"]), bool(spec["filtered"]), rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
