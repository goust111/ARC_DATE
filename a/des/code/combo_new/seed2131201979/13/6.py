from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
NODE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MAX_PATHS = 3
MAX_SEGMENTS_PER_PATH = 4
MAX_GENERATION_ATTEMPTS = 500
MAX_PATH_ATTEMPTS = 200
MAX_NODE_ATTEMPTS = 120
MAX_PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(comp)

    return comps


def _ordered_path(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_set = set(cells)
    adj = {
        cell: [nbr for nbr in _neighbors4(*cell) if nbr in cell_set]
        for cell in cell_set
    }
    degrees = {cell: len(nbrs) for cell, nbrs in adj.items()}
    assert all(deg in (1, 2) for deg in degrees.values()), "path graph must have degree 1/2 only"

    endpoints = sorted(cell for cell, deg in degrees.items() if deg == 1)
    assert len(endpoints) == 2, "each path must have exactly two endpoints"

    start = endpoints[0]
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        order.append(cur)
        nxts = [nbr for nbr in adj[cur] if nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal should be unambiguous"
        prev, cur = cur, nxts[0]

    assert len(order) == len(cells), "path component must be one simple chain"
    return order


def _segments_from_order(grid: np.ndarray, order: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    values = [int(grid[r, c]) for r, c in order]
    assert all(v != BLACK for v in values), "path cells must be non-black"

    node_indices = [i for i, v in enumerate(values) if v != GREY]
    assert len(node_indices) >= 2, "each path needs at least two colored nodes"
    assert node_indices[0] == 0 and node_indices[-1] == len(order) - 1, "endpoints must be colored nodes"

    node_colors = [values[i] for i in node_indices]
    assert len(set(node_colors)) == len(node_colors), "node colors must be unique within a path"
    assert all(v == GREY for i, v in enumerate(values) if i not in set(node_indices)), "non-node path cells must be grey"

    segments: list[tuple[int, int, int]] = []
    for a, b in zip(node_indices, node_indices[1:]):
        assert b > a
        segments.append((values[a], values[b], b - a + 1))
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Straighten each colored-node-bounded section of every grey polyline path into ordered horizontal rows.

    Input structure:
    - The background is black.
    - One or more disconnected 4-connected width-1 paths are present; each path is mostly grey and may turn at right angles.
    - Colored node pixels are embedded on the path, including both endpoints; node colors are unique within each path.

    Rule:
    - Treat every non-black connected component as one path and traverse it from one colored endpoint to the other.
    - Split the traversal at each colored node, keeping each consecutive node-to-node section with its full path length.
    - For each section, output one left-aligned horizontal segment whose length matches the original section length, with the two endpoint colors preserved and grey interior pixels.
    - Keep sections from the same path in traversal order with one black row between them, and keep different paths ordered by topmost-leftmost position with two black rows between them.

    Key insight:
    - The path geometry is discarded after walking the chain: only the on-path node order and the topological length between consecutive nodes survive into the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BLACK)
    assert comps, "expected at least one path"

    paths: list[dict[str, object]] = []
    for comp in comps:
        order = _ordered_path(comp)
        segments = _segments_from_order(g, order)
        paths.append({"key": min(comp), "segments": segments})

    paths.sort(key=lambda item: item["key"])
    max_len = max(length for path in paths for _a, _b, length in path["segments"])

    rows: list[np.ndarray] = []
    blank = np.full(max_len, BLACK, dtype=int)
    for path_idx, path in enumerate(paths):
        segments = path["segments"]
        for seg_idx, (start_color, end_color, length) in enumerate(segments):
            row = blank.copy()
            row[:length] = GREY
            row[0] = start_color
            row[length - 1] = end_color
            rows.append(row)
            if seg_idx != len(segments) - 1:
                rows.append(blank.copy())
        if path_idx != len(paths) - 1:
            rows.append(blank.copy())
            rows.append(blank.copy())

    out = np.vstack(rows)
    assert max(out.shape) <= 30, "output must stay within ARC limits"
    return out


def _normalize(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _turn_choices(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return ((-dc, dr), (dc, -dr))


def _simulate_polyline(
    directions: list[tuple[int, int]],
    run_lengths: list[int],
) -> list[tuple[int, int]] | None:
    cells = [(0, 0)]
    used = {(0, 0)}

    for direction, run_len in zip(directions, run_lengths):
        dr, dc = direction
        for _ in range(run_len):
            pr, pc = cells[-1]
            nr, nc = pr + dr, pc + dc
            if (nr, nc) in used:
                return None
            for ar, ac in _neighbors4(nr, nc):
                if (ar, ac) in used and (ar, ac) != (pr, pc):
                    return None
            cells.append((nr, nc))
            used.add((nr, nc))

    return _normalize(cells)


def _turn_indices(cells: list[tuple[int, int]]) -> list[int]:
    turns: list[int] = []
    for i in range(1, len(cells) - 1):
        dr1 = cells[i][0] - cells[i - 1][0]
        dc1 = cells[i][1] - cells[i - 1][1]
        dr2 = cells[i + 1][0] - cells[i][0]
        dc2 = cells[i + 1][1] - cells[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns.append(i)
    return turns


def _sample_segment_count(path_len: int, mode: str) -> int:
    max_segments = min(MAX_SEGMENTS_PER_PATH, (path_len - 1) // 2)
    assert max_segments >= 1
    if mode == "straight" or max_segments == 1:
        return random.randint(1, max_segments)
    if random.random() < 0.75:
        return random.randint(2, max_segments)
    return 1


def _sample_node_indices(path_len: int, segment_count: int) -> list[int]:
    total_steps = path_len - 1
    extra = total_steps - 2 * segment_count
    assert extra >= 0
    extras = [0] * segment_count
    for _ in range(extra):
        extras[random.randrange(segment_count)] += 1

    indices = [0]
    cur = 0
    for bonus in extras:
        cur += 2 + bonus
        indices.append(cur)
    assert indices[-1] == path_len - 1
    return indices


def _make_path_spec() -> dict[str, object] | None:
    mode = random.choice(("straight", "corner", "wiggly"))

    for _ in range(MAX_PATH_ATTEMPTS):
        if mode == "straight":
            directions = [random.choice(DIRS4)]
            run_lengths = [random.randint(4, 11)]
        elif mode == "corner":
            run_count = random.choice((2, 3))
            directions = [random.choice(DIRS4)]
            for _ in range(run_count - 1):
                directions.append(random.choice(_turn_choices(directions[-1])))
            run_lengths = [random.randint(2, 5) for _ in range(run_count)]
        else:
            run_count = random.choice((3, 4, 5))
            directions = [random.choice(DIRS4)]
            for _ in range(run_count - 1):
                directions.append(random.choice(_turn_choices(directions[-1])))
            run_lengths = [random.randint(2, 4) for _ in range(run_count)]

        cells = _simulate_polyline(directions, run_lengths)
        if cells is None:
            continue

        height = max(r for r, _ in cells) + 1
        width = max(c for _, c in cells) + 1
        if height > 12 or width > 12 or len(cells) > 18:
            continue

        turns = _turn_indices(cells)
        if mode != "straight" and not turns:
            continue

        segment_count = _sample_segment_count(len(cells), mode)
        node_count = segment_count + 1
        if node_count > len(NODE_COLORS):
            continue

        for _ in range(MAX_NODE_ATTEMPTS):
            node_indices = _sample_node_indices(len(cells), segment_count)
            if turns and mode != "straight":
                if not any(a < t < b for a, b in zip(node_indices, node_indices[1:]) for t in turns):
                    continue
            node_colors = random.sample(NODE_COLORS, node_count)
            segment_lengths = [b - a + 1 for a, b in zip(node_indices, node_indices[1:])]
            return {
                "coords": cells,
                "height": height,
                "width": width,
                "node_indices": node_indices,
                "node_colors": node_colors,
                "segment_lengths": segment_lengths,
            }

    return None


def _can_place(occupied: np.ndarray, coords: list[tuple[int, int]], top: int, left: int) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = top + r
        cc = left + c
        if rr < 0 or cc < 0 or rr >= h or cc >= w:
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if bool(np.any(occupied[r0:r1, c0:c1])):
            return False
    return True


def _paint_path(grid: np.ndarray, spec: dict[str, object], top: int, left: int) -> None:
    coords = spec["coords"]
    node_indices = spec["node_indices"]
    node_colors = spec["node_colors"]

    for r, c in coords:
        grid[top + r, left + c] = GREY
    for idx, color in zip(node_indices, node_colors):
        r, c = coords[idx]
        grid[top + r, left + c] = color


def _gen_attempt() -> np.ndarray | None:
    path_count = random.randint(1, MAX_PATHS)
    specs: list[dict[str, object]] = []

    for _ in range(path_count):
        spec = _make_path_spec()
        if spec is None:
            return None
        specs.append(spec)

    out_height = sum(2 * len(spec["segment_lengths"]) - 1 for spec in specs) + 2 * (len(specs) - 1)
    out_width = max(max(spec["segment_lengths"]) for spec in specs)
    if out_height > 30 or out_width > 30:
        return None

    min_h = max(12, max(spec["height"] for spec in specs) + 2)
    min_w = max(12, max(spec["width"] for spec in specs) + 2)
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        placed = False
        height = spec["height"]
        width = spec["width"]
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            top = random.randint(0, grid_h - height)
            left = random.randint(0, grid_w - width)
            if not _can_place(occupied, spec["coords"], top, left):
                continue
            _paint_path(grid, spec, top, left)
            for r, c in spec["coords"]:
                occupied[top + r, left + c] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
