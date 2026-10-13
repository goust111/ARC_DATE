from __future__ import annotations

import random

import numpy as np


BG = 0
LATTICE = 1
MARKER = 2
BLOCKER = 4

NODE_SIZE = 2
NODE_STRIDE = 3
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _band_starts(full_flags: np.ndarray) -> list[int]:
    starts: list[int] = []
    i = 0
    n = len(full_flags)
    while i < n:
        if not bool(full_flags[i]):
            i += 1
            continue
        j = i
        while j < n and bool(full_flags[j]):
            j += 1
        assert j - i == NODE_SIZE, "each lattice band must have width 2"
        starts.append(i)
        i = j
    assert starts, "expected at least one lattice band"
    return starts


def _validate_stride(starts: list[int], limit: int) -> None:
    assert starts[0] == 0, "lattice bbox must start at a band"
    for a, b in zip(starts, starts[1:], strict=False):
        assert b - a == NODE_STRIDE, "lattice node spacing must be 3"
    expected_limit = starts[-1] + NODE_SIZE
    assert expected_limit == limit, "lattice bbox size must match 2-wide bands with 1-wide gaps"


def _parse_lattice(grid: np.ndarray) -> tuple[tuple[int, int], list[int], list[int], tuple[int, int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, LATTICE, MARKER, BLOCKER}, "unexpected colors"

    box = _bbox(g != BG)
    assert box is not None, "expected a non-empty lattice"
    r0, c0, r1, c1 = box
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    mask = sub != BG

    row_full = np.all(mask, axis=1)
    col_full = np.all(mask, axis=0)
    row_starts = _band_starts(row_full)
    col_starts = _band_starts(col_full)
    _validate_stride(row_starts, sub.shape[0])
    _validate_stride(col_starts, sub.shape[1])

    expected_mask = np.zeros_like(mask, dtype=bool)
    for rs in row_starts:
        expected_mask[rs : rs + NODE_SIZE, :] = True
    for cs in col_starts:
        expected_mask[:, cs : cs + NODE_SIZE] = True
    assert np.array_equal(mask, expected_mask), "foreground mask must be a regular connected lattice"

    marker_nodes: list[tuple[int, int]] = []
    blocker_nodes: set[tuple[int, int]] = set()
    for nr, rs in enumerate(row_starts):
        for nc, cs in enumerate(col_starts):
            block = sub[rs : rs + NODE_SIZE, cs : cs + NODE_SIZE]
            color = int(block[0, 0])
            assert np.all(block == color), "every lattice node must be a solid 2x2 block"
            assert color in {LATTICE, MARKER, BLOCKER}, "node colors must be blue, red, or yellow"
            if color == MARKER:
                marker_nodes.append((nr, nc))
            elif color == BLOCKER:
                blocker_nodes.add((nr, nc))

            if nc + 1 < len(col_starts):
                h_edge = sub[rs : rs + NODE_SIZE, cs + NODE_SIZE : cs + NODE_STRIDE]
                assert h_edge.shape == (NODE_SIZE, 1)
                assert np.all(h_edge == LATTICE), "horizontal connectors must stay blue"
            if nr + 1 < len(row_starts):
                v_edge = sub[rs + NODE_SIZE : rs + NODE_STRIDE, cs : cs + NODE_SIZE]
                assert v_edge.shape == (1, NODE_SIZE)
                assert np.all(v_edge == LATTICE), "vertical connectors must stay blue"

    assert len(marker_nodes) == 1, "expected exactly one red marker node"
    assert blocker_nodes, "expected at least one blocker node"
    marker_node = marker_nodes[0]
    assert marker_node not in blocker_nodes
    return (r0, c0), row_starts, col_starts, marker_node, blocker_nodes


def _trace_spiral(
    shape: tuple[int, int],
    start: tuple[int, int],
    blockers: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]], bool]:
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    total_visitable = shape[0] * shape[1] - len(blockers)
    visited = {start}
    visit_order = [start]
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    position = start
    step_len = 1
    direction_index = 0
    max_radius = max(start[0], shape[0] - 1 - start[0], start[1], shape[1] - 1 - start[1])

    while len(visited) < total_visitable and step_len <= 2 * max_radius + 2:
        for _ in range(2):
            dr, dc = directions[direction_index % 4]
            for _ in range(step_len):
                prev = position
                nr = prev[0] + dr
                nc = prev[1] + dc
                if _in_bounds(shape, nr, nc) and (nr, nc) in blockers:
                    return visit_order, edges, True

                position = (nr, nc)
                prev_in = _in_bounds(shape, prev[0], prev[1],)
                curr_in = _in_bounds(shape, nr, nc)
                if prev_in and curr_in:
                    edges.append((prev, position))
                if curr_in and position not in visited:
                    visited.add(position)
                    visit_order.append(position)
            direction_index += 1
        step_len += 1

    return visit_order, edges, False


def _paint_output(
    grid: np.ndarray,
    origin: tuple[int, int],
    row_starts: list[int],
    col_starts: list[int],
    visited_nodes: list[tuple[int, int]],
    visited_edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    base_r, base_c = origin

    for nr, nc in visited_nodes:
        pr = base_r + row_starts[nr]
        pc = base_c + col_starts[nc]
        block = out[pr : pr + NODE_SIZE, pc : pc + NODE_SIZE]
        block[block == LATTICE] = MARKER

    for (r1, c1), (r2, c2) in visited_edges:
        if r1 == r2:
            rr = base_r + row_starts[r1]
            cc = base_c + col_starts[min(c1, c2)] + NODE_SIZE
            strip = out[rr : rr + NODE_SIZE, cc : cc + 1]
        else:
            rr = base_r + row_starts[min(r1, r2)] + NODE_SIZE
            cc = base_c + col_starts[c1]
            strip = out[rr : rr + 1, cc : cc + NODE_SIZE]
        strip[strip == LATTICE] = MARKER

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the spiral-reachable portion of a connected lattice until the traversal would hit a blocker node.

    Input structure:
    - The background is black.
    - Blue pixels form one connected orthogonal lattice whose nodes are solid 2x2 blocks spaced every 3 cells.
    - Exactly one node is recolored red as the spiral start, and one or more other nodes are recolored yellow as blockers.

    Rule:
    - Interpret each 2x2 block as a lattice node and the 1-cell connectors between adjacent blocks as graph edges.
    - Starting from the red node, follow the square-spiral move sequence up, right, down, left with arm lengths 1, 1, 2, 2, 3, 3, ... on the abstract node grid.
    - Out-of-bounds spiral moves are skipped by clipping the infinite spiral to the finite lattice, but if the next in-bounds step would land on a yellow blocker node, stop before entering it.
    - Recolor every visited blue node and every blue connector between consecutive visited in-bounds nodes to red; keep the original red node and all yellow blockers unchanged.

    Key insight:
    - The red region is determined by clipped spiral order on lattice nodes rather than by geometric distance, so a nearby node can remain blue if a blocker appears earlier in the spiral sequence.
    """
    origin, row_starts, col_starts, start, blockers = _parse_lattice(grid)
    visited_nodes, visited_edges, _ = _trace_spiral((len(row_starts), len(col_starts)), start, blockers)
    return _paint_output(grid, origin, row_starts, col_starts, visited_nodes, visited_edges)


def _marker_candidates(node_rows: int, node_cols: int) -> list[tuple[int, int]]:
    interior = [
        (r, c)
        for r in range(1, node_rows - 1)
        for c in range(1, node_cols - 1)
    ]
    edge = [
        (r, c)
        for r in range(node_rows)
        for c in range(node_cols)
        if (r, c) not in interior
    ]
    if interior and random.random() < 0.7:
        random.shuffle(interior)
        return interior + edge
    all_nodes = interior + edge
    random.shuffle(all_nodes)
    return all_nodes


def _node_manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _valid_blocker_layout(
    shape: tuple[int, int],
    start: tuple[int, int],
    blockers: set[tuple[int, int]],
) -> bool:
    if any(_node_manhattan(start, blocker) <= 1 for blocker in blockers):
        return False

    visited_nodes, _, hit_blocker = _trace_spiral(shape, start, blockers)
    if not hit_blocker:
        return False

    visited_set = set(visited_nodes)
    all_nodes = {
        (r, c)
        for r in range(shape[0])
        for c in range(shape[1])
    }
    unvisited_blue = sorted(all_nodes - visited_set - blockers)
    if not unvisited_blue:
        return False

    if len(visited_nodes) < 5:
        return False

    closest_unvisited = min(_node_manhattan(start, node) for node in unvisited_blue)
    closest_blocker = min(_node_manhattan(start, node) for node in blockers)
    return closest_unvisited <= min(2, closest_blocker)


def _choose_special_nodes(node_rows: int, node_cols: int) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    all_nodes = [(r, c) for r in range(node_rows) for c in range(node_cols)]
    shape = (node_rows, node_cols)

    for start in _marker_candidates(node_rows, node_cols):
        candidates = [node for node in all_nodes if node != start]
        random.shuffle(candidates)
        blocker_count = 1 if random.random() < 0.5 else 2
        for first in candidates:
            blocker_set = {first}
            if _valid_blocker_layout(shape, start, blocker_set):
                if blocker_count == 1:
                    return start, blocker_set
                extras = [node for node in candidates if node not in blocker_set]
                random.shuffle(extras)
                for extra in extras:
                    trial = blocker_set | {extra}
                    if _valid_blocker_layout(shape, start, trial):
                        return start, trial
            if blocker_count == 2:
                extras = [node for node in candidates if node != first]
                random.shuffle(extras)
                for extra in extras:
                    trial = {first, extra}
                    if _valid_blocker_layout(shape, start, trial):
                        return start, trial
    return None


def _draw_lattice(
    grid: np.ndarray,
    top: int,
    left: int,
    node_rows: int,
    node_cols: int,
) -> None:
    for nr in range(node_rows):
        for nc in range(node_cols):
            r = top + nr * NODE_STRIDE
            c = left + nc * NODE_STRIDE
            grid[r : r + NODE_SIZE, c : c + NODE_SIZE] = LATTICE
            if nc + 1 < node_cols:
                grid[r : r + NODE_SIZE, c + NODE_SIZE : c + NODE_STRIDE] = LATTICE
            if nr + 1 < node_rows:
                grid[r + NODE_SIZE : r + NODE_STRIDE, c : c + NODE_SIZE] = LATTICE


def _build_input_grid(
    node_rows: int,
    node_cols: int,
    top: int,
    left: int,
    start: tuple[int, int],
    blockers: set[tuple[int, int]],
) -> np.ndarray:
    height = top + (node_rows - 1) * NODE_STRIDE + NODE_SIZE + random.randint(0, 3)
    width = left + (node_cols - 1) * NODE_STRIDE + NODE_SIZE + random.randint(0, 3)
    grid = np.zeros((height, width), dtype=int)
    _draw_lattice(grid, top, left, node_rows, node_cols)

    sr = top + start[0] * NODE_STRIDE
    sc = left + start[1] * NODE_STRIDE
    grid[sr : sr + NODE_SIZE, sc : sc + NODE_SIZE] = MARKER

    for br, bc in blockers:
        pr = top + br * NODE_STRIDE
        pc = left + bc * NODE_STRIDE
        grid[pr : pr + NODE_SIZE, pc : pc + NODE_SIZE] = BLOCKER

    return grid


def _gen_attempt() -> np.ndarray | None:
    node_rows = random.randint(4, 7)
    node_cols = random.randint(4, 7)
    bbox_h = (node_rows - 1) * NODE_STRIDE + NODE_SIZE
    bbox_w = (node_cols - 1) * NODE_STRIDE + NODE_SIZE
    max_top = min(3, 30 - bbox_h)
    max_left = min(3, 30 - bbox_w)
    top = random.randint(0, max_top)
    left = random.randint(0, max_left)

    special = _choose_special_nodes(node_rows, node_cols)
    if special is None:
        return None
    start, blockers = special

    grid = _build_input_grid(node_rows, node_cols, top, left, start, blockers)
    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-spiral ARC input")
