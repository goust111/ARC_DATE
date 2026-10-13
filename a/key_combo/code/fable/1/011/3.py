from __future__ import annotations

import random

import numpy as np


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DIR_FROM_DELTA = {delta: name for name, delta in DIRS.items()}

BG = 0
CELL = 9
GRID_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3)]
JAR_SIZE_CHOICES = [(5, 5), (5, 7), (7, 5), (7, 7)]
SINK_SIZE_CHOICES = [(7, 7), (7, 7), (7, 5), (5, 7), (5, 5)]
MAX_JARS = 4
MAX_GENERATION_ATTEMPTS = 400
COUNT_RETRIES = 200
PATH_RETRIES = 80


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert int(connectivity) == 4
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _jar_midpoint(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _interior_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _fill_order_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r1 - 1, r0, -1) for c in range(c0 + 1, c1)]


def _parse_jar(
    grid: np.ndarray,
    bg: int,
    border_color: int,
    coords: np.ndarray,
    jar_id: int,
):
    cells = {tuple(map(int, rc)) for rc in coords}
    degrees: dict[tuple[int, int], int] = {}
    spike_cells: list[tuple[int, int]] = []
    for r, c in cells:
        degree = sum((nr, nc) in cells for nr, nc in neighbors4(r, c))
        degrees[(r, c)] = int(degree)
        if degree == 1:
            spike_cells.append((int(r), int(c)))
    if len(spike_cells) != 1:
        return None

    spike = spike_cells[0]
    base_neighbors = [(nr, nc) for nr, nc in neighbors4(*spike) if (nr, nc) in cells]
    if len(base_neighbors) != 1:
        return None
    base = base_neighbors[0]

    frame_cells = set(cells)
    frame_cells.remove(spike)
    if not frame_cells:
        return None
    frame_coords = np.array(sorted(frame_cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(frame_coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    expected_frame = set()
    for rr in range(r0, r1 + 1):
        expected_frame.add((int(rr), int(c0)))
        expected_frame.add((int(rr), int(c1)))
    for cc in range(c0, c1 + 1):
        expected_frame.add((int(r0), int(cc)))
        expected_frame.add((int(r1), int(cc)))
    if frame_cells != expected_frame:
        return None

    delta = (int(spike[0] - base[0]), int(spike[1] - base[1]))
    direction = DIR_FROM_DELTA.get(delta)
    if direction is None:
        return None

    if direction == "U":
        if base[0] != r0 or spike[0] != r0 - 1 or spike[1] != base[1] or base[1] in (c0, c1):
            return None
    elif direction == "D":
        if base[0] != r1 or spike[0] != r1 + 1 or spike[1] != base[1] or base[1] in (c0, c1):
            return None
    elif direction == "L":
        if base[1] != c0 or spike[1] != c0 - 1 or spike[0] != base[0] or base[0] in (r0, r1):
            return None
    else:
        if base[1] != c1 or spike[1] != c1 + 1 or spike[0] != base[0] or base[0] in (r0, r1):
            return None

    bbox = (int(r0), int(c0), int(r1), int(c1))
    interior = _interior_cells(bbox)
    inner_colors = {
        int(grid[r, c])
        for r, c in interior
        if int(grid[r, c]) != int(bg) and int(grid[r, c]) != int(border_color)
    }
    if len(inner_colors) != 1:
        return None
    content_color = int(next(iter(inner_colors)))
    content_cells = [(r, c) for r, c in interior if int(grid[r, c]) == int(content_color)]
    if not content_cells:
        return None

    return {
        "id": int(jar_id),
        "border_color": int(border_color),
        "border_cells": cells,
        "bbox": bbox,
        "spike": spike,
        "direction": direction,
        "content_color": int(content_color),
        "content_count": int(len(content_cells)),
        "content_cells": content_cells,
        "interior": interior,
    }


def _find_jars(grid: np.ndarray):
    bg = int(get_bg(grid))
    jars = []
    next_id = 0
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for coords in components_mask(grid == int(color), connectivity=4):
            jar = _parse_jar(grid, bg, int(color), coords, next_id)
            if jar is None:
                continue
            jars.append(jar)
            next_id += 1
    assert jars, "expected at least one jar"
    return bg, jars


def _trace_successor(
    grid: np.ndarray,
    bg: int,
    jar: dict,
    coord_to_jar: dict[tuple[int, int], int],
):
    dr, dc = DIRS[jar["direction"]]
    rr = int(jar["spike"][0] + dr)
    cc = int(jar["spike"][1] + dc)
    while in_bounds(grid.shape, rr, cc):
        if int(grid[rr, cc]) != int(bg):
            hit = coord_to_jar.get((int(rr), int(cc)))
            assert hit is not None and int(hit) != int(jar["id"]), "tail ray must hit another jar border"
            return int(hit)
        rr += dr
        cc += dc
    return None


def _chain_order(grid: np.ndarray, bg: int, jars: list[dict]) -> list[int]:
    coord_to_jar: dict[tuple[int, int], int] = {}
    jar_by_id = {int(jar["id"]): jar for jar in jars}
    for jar in jars:
        for cell in jar["border_cells"]:
            coord_to_jar[cell] = int(jar["id"])

    succ: dict[int, int | None] = {}
    indegree = {int(jar["id"]): 0 for jar in jars}
    for jar in jars:
        jar_id = int(jar["id"])
        hit = _trace_successor(grid, bg, jar, coord_to_jar)
        succ[jar_id] = hit
        if hit is not None:
            indegree[int(hit)] += 1

    sinks = [jar_id for jar_id, nxt in succ.items() if nxt is None]
    assert len(sinks) == 1, "expected exactly one chain sink"
    heads = [jar_id for jar_id, deg in indegree.items() if deg == 0]
    assert len(heads) == 1, "expected exactly one chain head"
    assert sum(nxt is not None for nxt in succ.values()) == len(jars) - 1

    for jar_id in jar_by_id:
        if jar_id == heads[0]:
            continue
        assert indegree[jar_id] == 1, "every non-head jar must have one predecessor"

    order = []
    seen = set()
    cur = int(heads[0])
    while cur is not None:
        assert cur not in seen, "chain must be acyclic"
        seen.add(cur)
        order.append(int(cur))
        nxt = succ[cur]
        cur = int(nxt) if nxt is not None else None
    assert len(order) == len(jars), "chain must visit every jar"
    assert order[-1] == int(sinks[0])
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour every jar's contents along the tail-spike chain into the unique sink jar and restack them by downstream distance.

    Input structure:
    - The background contains several closed hollow rectangles, each in its own border color, and each rectangle has one one-cell spike attached outside one side.
    - Every rectangle interior contains some pixels of a single jar-specific content color.
    - Following the spike direction from each jar hits the next jar's border on a straight orthogonal ray, and exactly one final jar's spike ray exits the grid.

    Rule:
    - Treat the spike direction as a one-way link toward the sink jar whose ray leaves the grid.
    - Erase all content from every non-sink jar, and also clear the sink interior before refilling it.
    - Refill the sink interior bottom row upward and left to right within each row: first the sink's own content color, then the next upstream jar, then the next farther upstream jar, preserving every jar's original content count exactly.

    Key insight:
    - The border-plus-spike component identifies each jar and its outgoing direction; tracing those rays yields a single directed chain, and the final filling order is just the reverse distance order to the unique out-of-bounds sink.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, jars = _find_jars(g)
    order = _chain_order(g, bg, jars)
    jar_by_id = {int(jar["id"]): jar for jar in jars}
    sink_id = int(order[-1])
    sink = jar_by_id[sink_id]

    out = g.copy()
    for jar in jars:
        for r, c in jar["interior"]:
            out[r, c] = int(bg)

    fill_cells = _fill_order_cells(sink["bbox"])
    fill_sequence = [sink_id, *reversed(order[:-1])]
    total = sum(int(jar_by_id[jar_id]["content_count"]) for jar_id in fill_sequence)
    assert total <= len(fill_cells), "sink interior must fit every content pixel"

    idx = 0
    for jar_id in fill_sequence:
        jar = jar_by_id[int(jar_id)]
        color = int(jar["content_color"])
        for _ in range(int(jar["content_count"])):
            rr, cc = fill_cells[idx]
            out[rr, cc] = color
            idx += 1
    return out


def _slot_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS.values():
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _boundary_cells(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(rows) for c in range(cols) if r in (0, rows - 1) or c in (0, cols - 1)]


def _boundary_dirs(cell: tuple[int, int], rows: int, cols: int) -> list[str]:
    r, c = cell
    dirs = []
    if r == 0:
        dirs.append("U")
    if r == rows - 1:
        dirs.append("D")
    if c == 0:
        dirs.append("L")
    if c == cols - 1:
        dirs.append("R")
    return dirs


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    return DIR_FROM_DELTA[(dr, dc)]


def _random_path(rows: int, cols: int, length: int, rng: random.Random) -> list[tuple[int, int]] | None:
    boundary = _boundary_cells(rows, cols)
    rng.shuffle(boundary)

    def dfs(path_rev: list[tuple[int, int]], used: set[tuple[int, int]]):
        if len(path_rev) == length:
            return list(path_rev)
        cur = path_rev[-1]
        nxts = [nb for nb in _slot_neighbors(cur, rows, cols) if nb not in used]
        rng.shuffle(nxts)
        for nb in nxts:
            used.add(nb)
            path_rev.append(nb)
            result = dfs(path_rev, used)
            if result is not None:
                return result
            path_rev.pop()
            used.remove(nb)
        return None

    for sink in boundary:
        result = dfs([sink], {sink})
        if result is not None:
            return list(reversed(result))
    return None


def _centered_bbox(slot: tuple[int, int], dims: tuple[int, int]) -> tuple[int, int, int, int]:
    slot_r, slot_c = slot
    h, w = dims
    r0 = int(slot_r * CELL + (CELL - h) // 2)
    c0 = int(slot_c * CELL + (CELL - w) // 2)
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _spike_cell(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mid_r, mid_c = _jar_midpoint(bbox)
    if direction == "U":
        return int(r0 - 1), int(mid_c)
    if direction == "D":
        return int(r1 + 1), int(mid_c)
    if direction == "L":
        return int(mid_r), int(c0 - 1)
    return int(mid_r), int(c1 + 1)


def _paint_jar(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    border_color: int,
    spike_dir: str,
    content_color: int,
    count: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(border_color)
    grid[r1, c0 : c1 + 1] = int(border_color)
    grid[r0 : r1 + 1, c0] = int(border_color)
    grid[r0 : r1 + 1, c1] = int(border_color)
    sr, sc = _spike_cell(bbox, spike_dir)
    grid[sr, sc] = int(border_color)

    interior = _interior_cells(bbox)
    picks = rng.sample(interior, int(count))
    for r, c in picks:
        grid[r, c] = int(content_color)


def _sample_counts(
    capacities: list[int],
    sink_dims: tuple[int, int],
    rng: random.Random,
) -> list[int] | None:
    sink_cap = int(capacities[-1])
    sink_inner_w = int(sink_dims[1] - 2)
    for _ in range(COUNT_RETRIES):
        counts = []
        for i, cap in enumerate(capacities):
            upper = min(int(cap), 7 if i == len(capacities) - 1 else 6)
            counts.append(rng.randint(2, upper))
        total = sum(counts)
        min_total = max(6, sink_inner_w + 1)
        if min_total <= total < sink_cap:
            return counts
    return None


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(GRID_SHAPES)
    max_len = min(MAX_JARS, rows * cols)
    chain_len = rng.randint(2, max_len)

    path = None
    for _ in range(PATH_RETRIES):
        path = _random_path(rows, cols, chain_len, rng)
        if path is not None:
            break
    if path is None:
        return None

    out_dirs = [_step_dir(path[i], path[i + 1]) for i in range(chain_len - 1)]
    out_dirs.append(rng.choice(_boundary_dirs(path[-1], rows, cols)))

    dims = [rng.choice(JAR_SIZE_CHOICES) for _ in range(chain_len - 1)]
    dims.append(rng.choice(SINK_SIZE_CHOICES))
    capacities = [(h - 2) * (w - 2) for h, w in dims]
    counts = _sample_counts(capacities, dims[-1], rng)
    if counts is None:
        return None

    palette = list(range(1, 10))
    rng.shuffle(palette)
    border_colors = palette[:chain_len]
    content_colors = palette[chain_len : 2 * chain_len]

    grid = np.full((rows * CELL, cols * CELL), BG, dtype=int)
    for i, slot in enumerate(path):
        bbox = _centered_bbox(slot, dims[i])
        _paint_jar(
            grid,
            bbox,
            border_color=int(border_colors[i]),
            spike_dir=out_dirs[i],
            content_color=int(content_colors[i]),
            count=int(counts[i]),
            rng=rng,
        )

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tail-chain jar puzzle")
