from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
NODE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500
WALK_ATTEMPTS = 300

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PORT_OFFSETS = [(-1, 1), (1, 3), (3, 1), (1, -1)]
CORNER_CELLS = [
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # top-left
    ((0, 1), (0, 2), (1, 1), (1, 2)),  # top-right
    ((1, 1), (1, 2), (2, 1), (2, 2)),  # bottom-right
    ((1, 0), (1, 1), (2, 0), (2, 1)),  # bottom-left
]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(coords)
    return comps


def _corner_index(cells: set[tuple[int, int]]) -> int | None:
    for idx, corner_cells in enumerate(CORNER_CELLS):
        if cells == set(corner_cells):
            return idx
    return None


def _paint_corner(grid: np.ndarray, r0: int, c0: int, corner: int, color: int) -> None:
    for dr, dc in CORNER_CELLS[corner]:
        grid[r0 + dr, c0 + dc] = color


def _extract_nodes(grid: np.ndarray) -> list[dict]:
    colors = sorted(int(color) for color in np.unique(grid) if int(color) not in (BLACK, GREY))
    nodes: list[dict] = []
    for color in colors:
        for coords in _components(grid == color):
            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
                continue

            grey_cells = {
                (r - r0, c - c0)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if int(grid[r, c]) == GREY
            }
            corner = _corner_index(grey_cells) if grey_cells else None

            if len(coords) == 9 and not grey_cells:
                nodes.append({"color": color, "r0": r0, "c0": c0, "corner": None})
            elif len(coords) == 5 and len(grey_cells) == 4 and corner is not None:
                nodes.append({"color": color, "r0": r0, "c0": c0, "corner": corner})
    nodes.sort(key=lambda node: (int(node["r0"]), int(node["c0"])))
    return nodes


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    nodes = _extract_nodes(g)
    assert len(nodes) >= 2
    assert len({int(node["color"]) for node in nodes}) == len(nodes)

    owner = np.full(g.shape, -1, dtype=int)
    for idx, node in enumerate(nodes):
        r0, c0 = int(node["r0"]), int(node["c0"])
        assert bool((owner[r0 : r0 + 3, c0 : c0 + 3] == -1).all())
        owner[r0 : r0 + 3, c0 : c0 + 3] = int(idx)

    h, w = g.shape
    adjacency = [set() for _ in nodes]
    for idx, node in enumerate(nodes):
        r0, c0 = int(node["r0"]), int(node["c0"])
        for dir_idx, (pr_off_r, pr_off_c) in enumerate(PORT_OFFSETS):
            pr = r0 + pr_off_r
            pc = c0 + pr_off_c
            if not (0 <= pr < h and 0 <= pc < w):
                continue
            if int(g[pr, pc]) != GREY:
                continue

            dr, dc = DIRS[dir_idx]
            rr, cc = pr, pc
            while True:
                nr = rr + dr
                nc = cc + dc
                assert 0 <= nr < h and 0 <= nc < w
                other = int(owner[nr, nc])
                if other != -1:
                    assert other != idx
                    adjacency[idx].add(other)
                    break
                assert int(g[nr, nc]) == GREY
                rr, cc = nr, nc

    degrees = [len(neis) for neis in adjacency]
    assert degrees.count(1) == 2
    assert all(degree in (1, 2) for degree in degrees)

    endpoints = [idx for idx, degree in enumerate(degrees) if degree == 1]
    start = min(endpoints, key=lambda idx: (int(nodes[idx]["r0"]), int(nodes[idx]["c0"])))

    overlay_nodes = [idx for idx, node in enumerate(nodes) if node["corner"] is not None]
    assert overlay_nodes == [start]

    order = [start]
    prev = -1
    cur = start
    while True:
        nxts = [nxt for nxt in adjacency[cur] if nxt != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        nxt = nxts[0]
        order.append(int(nxt))
        prev, cur = cur, nxt

    assert len(order) == len(nodes)
    assert len(set(order)) == len(nodes)
    return g, nodes, order, int(nodes[start]["corner"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a rotating 2x2 corner pattern along a grey-linked chain of colored 3x3 nodes.

    Input structure:
    - Black background with several distinct-color 3x3 node blocks.
    - Grey 1-pixel-wide horizontal or vertical line segments connect the nodes into one unbranched chain.
    - Exactly one endpoint node contains a grey 2x2 corner patch over its 3x3 body; all later nodes are solid.

    Rule:
    - Identify the two chain endpoints and choose the start as the endpoint with the smaller row, breaking ties by smaller column.
    - Traverse the chain from that start endpoint to the opposite endpoint.
    - Keep the start node unchanged; for each later node, place the same 2x2 patch in a corner that rotates clockwise once per step from the start corner.
    - Color each propagated patch with the body color of the previous node in traversal order, while keeping the grey links and black background unchanged.

    Key insight:
    Solving the puzzle requires combining three cues at once: endpoint-based traversal order, clockwise corner rotation from node to node, and color inheritance from the previous node rather than the current one.
    """
    g, nodes, order, start_corner = _analyze_grid(grid)
    out = g.copy()
    for idx in range(1, len(order)):
        node = nodes[order[idx]]
        prev_node = nodes[order[idx - 1]]
        corner = (start_corner + idx) % 4
        _paint_corner(out, int(node["r0"]), int(node["c0"]), corner, int(prev_node["color"]))
    return out


def _draw_connector(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        row = ar + 1
        c0 = min(ac, bc) + 3
        c1 = max(ac, bc) - 1
        assert c0 <= c1
        grid[row, c0 : c1 + 1] = GREY
    else:
        col = ac + 1
        r0 = min(ar, br) + 3
        r1 = max(ar, br) - 1
        assert r0 <= r1
        grid[r0 : r1 + 1, col] = GREY


def _walk_options(cell: tuple[int, int], rows: int, cols: int, used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    options = []
    for dr, dc in DIRS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in used:
            options.append((nr, nc))
    random.shuffle(options)
    options.sort(
        key=lambda pos: sum(
            1
            for dr, dc in DIRS
            if 0 <= pos[0] + dr < rows and 0 <= pos[1] + dc < cols and (pos[0] + dr, pos[1] + dc) not in used
        )
    )
    return options


def _extend_walk(
    rows: int,
    cols: int,
    target_len: int,
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
) -> bool:
    if len(path) == target_len:
        return True
    for nxt in _walk_options(path[-1], rows, cols, used):
        used.add(nxt)
        path.append(nxt)
        if _extend_walk(rows, cols, target_len, path, used):
            return True
        path.pop()
        used.remove(nxt)
    return False


def _build_walk(rows: int, cols: int, target_len: int, same_row_endpoints: bool) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(WALK_ATTEMPTS):
        start = random.choice(starts)
        path = [start]
        used = {start}
        if not _extend_walk(rows, cols, target_len, path, used):
            continue
        if (path[0][0] == path[-1][0]) != same_row_endpoints:
            continue
        return path
    return None


def _padding(base: int) -> tuple[int, int]:
    spare = 30 - base
    assert spare >= 2
    total = random.randint(2, spare)
    front = random.randint(1, total - 1)
    return front, total - front


def _gen_attempt() -> np.ndarray | None:
    same_row_endpoints = bool(random.getrandbits(1))
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    length = random.randint(3, min(8, rows * cols))
    coarse_path = _build_walk(rows, cols, length, same_row_endpoints)
    if coarse_path is None:
        return None

    step = random.randint(4, 6)
    min_r = min(r for r, _c in coarse_path)
    min_c = min(c for _r, c in coarse_path)
    norm_path = [(r - min_r, c - min_c) for r, c in coarse_path]
    span_r = max(r for r, _c in norm_path)
    span_c = max(c for _r, c in norm_path)
    base_h = 3 + step * span_r
    base_w = 3 + step * span_c
    if base_h > 28 or base_w > 28:
        return None

    top_pad, bottom_pad = _padding(base_h)
    left_pad, right_pad = _padding(base_w)
    h = base_h + top_pad + bottom_pad
    w = base_w + left_pad + right_pad

    positions = [(top_pad + step * r, left_pad + step * c) for r, c in norm_path]
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(list(NODE_COLORS), len(positions))
    for (r0, c0), color in zip(positions, colors):
        grid[r0 : r0 + 3, c0 : c0 + 3] = int(color)
    for a, b in zip(positions, positions[1:]):
        _draw_connector(grid, a, b)

    endpoints = [0, len(positions) - 1]
    start_idx = min(endpoints, key=lambda idx: positions[idx])
    corner = random.randrange(4)
    _paint_corner(grid, positions[start_idx][0], positions[start_idx][1], corner, GREY)

    try:
        _g, nodes, order, start_corner = _analyze_grid(grid)
        assert len(nodes) == len(positions)
        assert start_corner == corner
        assert len(order) == len(positions)
        assert (positions[0][0] == positions[-1][0]) == same_row_endpoints
        out = main(grid)
        assert out.shape == grid.shape
        assert not np.array_equal(out, grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-propagation input")
