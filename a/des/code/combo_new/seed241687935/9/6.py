from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BRANCHES = ("open_open", "open_closed", "closed_closed")
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1
    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.append((r0, c))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.append((r, c0))
    return cells


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment(points[i], points[i + 1])
        if path:
            seg = seg[1:]
        path.extend(seg)
    assert len(path) == len(set(path)), "pipe may not self-intersect"
    return path


def _on_border(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _hidden_nodes_for_color(
    g: np.ndarray, color: int, visible: set[tuple[int, int]]
) -> dict[tuple[int, int], str]:
    h, w = g.shape
    hidden: dict[tuple[int, int], str] = {}
    for r_raw, c_raw in np.argwhere((g != BG) & (g != color)):
        r = int(r_raw)
        c = int(c_raw)
        horiz = 0 < c < w - 1 and (r, c - 1) in visible and (r, c + 1) in visible
        vert = 0 < r < h - 1 and (r - 1, c) in visible and (r + 1, c) in visible
        assert not (horiz and vert), "ambiguous hidden crossing"
        if horiz:
            hidden[(r, c)] = "h"
        elif vert:
            hidden[(r, c)] = "v"
    return hidden


def _trace_pipe(
    g: np.ndarray, color: int
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    visible = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
    assert visible, "expected at least one visible pixel for every pipe"

    hidden = _hidden_nodes_for_color(g, color, visible)
    nodes = visible | set(hidden)
    adj = {node: set() for node in nodes}

    for r, c in visible:
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in visible:
                adj[(r, c)].add(nxt)

    for (r, c), orient in hidden.items():
        if orient == "h":
            neighbors = [(r, c - 1), (r, c + 1)]
        else:
            neighbors = [(r - 1, c), (r + 1, c)]
        for nxt in neighbors:
            assert nxt in visible
            adj[(r, c)].add(nxt)
            adj[nxt].add((r, c))

    degree_one = [node for node, nbrs in adj.items() if len(nbrs) == 1]
    assert len(degree_one) == 2, "each pipe must have exactly two endpoints"
    assert all(node in visible for node in degree_one), "endpoints must be visible"
    for nbrs in adj.values():
        assert len(nbrs) in {1, 2}, "pipe graph must be a simple path"

    path: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = degree_one[0]
    while True:
        path.append(cur)
        next_nodes = [node for node in adj[cur] if node != prev]
        if not next_nodes:
            break
        assert len(next_nodes) == 1, "simple path traversal must stay unambiguous"
        prev, cur = cur, next_nodes[0]

    assert len(path) == len(nodes), "pipe trace must visit every logical pixel"
    assert path[-1] == degree_one[1]
    return path, visible


def _recolor_targets(
    path: list[tuple[int, int]], visible: set[tuple[int, int]], shape: tuple[int, int]
) -> set[tuple[int, int]]:
    start_open = _on_border(path[0], shape)
    end_open = _on_border(path[-1], shape)
    if start_open and end_open:
        return set()
    if not start_open and not end_open:
        return set(visible)
    window = path[:3] if not start_open else path[-3:]
    return {cell for cell in window if cell in visible}


def main(grid: np.ndarray) -> np.ndarray:
    """Separate the colored pipes, follow them through hidden crossings, and grey out the closed ends.

    Input structure:
    - The background is black.
    - Several non-black colors each draw one single-pixel-wide orthogonal pipe.
    - A pipe may be partially hidden at crossings: one crossing pixel shows the color
      of the top pipe while the perpendicular pipe continues logically underneath it.
    - Pipe endpoints are the two degree-1 logical pixels of each pipe; an endpoint on
      the outer border is open, otherwise it is closed.

    Rule:
    - Reconstruct each colored pipe independently by tracing same-color adjacency and by
      treating a one-cell interruption by another non-black color as a straight crossing.
    - If both endpoints of a pipe are open, keep it unchanged.
    - If exactly one endpoint is closed, recolor the last three logical pixels at that
      closed end to grey; only the pipe's visible pixels can change on the final grid.
    - If both endpoints are closed, recolor every visible pixel of that pipe to grey.

    Key insight:
    The visible grid is not enough by plain connected components alone: a pipe can stay
    connected through another color's crossing pixel, so each color must be traced as a
    logical path before the endpoint types and recoloring rule can be determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    for color in colors:
        path, visible = _trace_pipe(g, color)
        for r, c in _recolor_targets(path, visible, g.shape):
            out[r, c] = GREY
    return out


def _branch_statuses(branch: str) -> tuple[str, str]:
    if branch == "open_open":
        return "open", "open"
    if branch == "closed_closed":
        return "closed", "closed"
    return random.choice([("open", "closed"), ("closed", "open")])


def _pick_h_endpoint_row(status: str, r_mid: int, h: int, forbidden: set[int]) -> int:
    if status == "open":
        choices = [0, h - 1]
    else:
        choices = [r for r in range(1, h - 1) if abs(r - r_mid) >= 2]
    choices = [r for r in choices if r not in forbidden]
    assert choices
    return random.choice(choices)


def _pick_v_endpoint_col(status: str, c_mid: int, w: int, forbidden: set[int]) -> int:
    if status == "open":
        choices = [0, w - 1]
    else:
        choices = [c for c in range(1, w - 1) if abs(c - c_mid) >= 2]
    choices = [c for c in choices if c not in forbidden]
    assert choices
    return random.choice(choices)


def _build_h_pipe(
    shape: tuple[int, int], branch: str, r_mid: int, c_left: int, c_right: int, block_rows: set[int]
) -> list[tuple[int, int]]:
    h, _ = shape
    left_status, right_status = _branch_statuses(branch)
    r_left = _pick_h_endpoint_row(left_status, r_mid, h, block_rows)
    r_right = _pick_h_endpoint_row(right_status, r_mid, h, block_rows | {r_left})
    return _polyline([(r_left, c_left), (r_mid, c_left), (r_mid, c_right), (r_right, c_right)])


def _build_v_pipe(
    shape: tuple[int, int], branch: str, c_mid: int, r_top: int, r_bottom: int, block_cols: set[int]
) -> list[tuple[int, int]]:
    _, w = shape
    top_status, bottom_status = _branch_statuses(branch)
    c_top = _pick_v_endpoint_col(top_status, c_mid, w, block_cols)
    c_bottom = _pick_v_endpoint_col(bottom_status, c_mid, w, block_cols | {c_top})
    return _polyline([(r_top, c_top), (r_top, c_mid), (r_bottom, c_mid), (r_bottom, c_bottom)])


def _corner_boxes(
    shape: tuple[int, int], r_top: int, r_bottom: int, c_left: int, c_right: int
) -> list[str]:
    h, w = shape
    boxes: list[str] = []
    if r_top >= 4 and c_left >= 4:
        boxes.append("tl")
    if r_top >= 4 and w - c_right - 2 >= 4:
        boxes.append("tr")
    if h - r_bottom - 2 >= 4 and c_left >= 4:
        boxes.append("bl")
    if h - r_bottom - 2 >= 4 and w - c_right - 2 >= 4:
        boxes.append("br")
    return boxes


def _build_corner_pipe(
    shape: tuple[int, int], branch: str, corner: str, r_top: int, r_bottom: int, c_left: int, c_right: int
) -> list[tuple[int, int]]:
    h, w = shape
    horiz_status, vert_status = _branch_statuses(branch)

    if corner == "tl":
        row_min = 2 if vert_status == "open" else 3
        col_min = 2 if horiz_status == "open" else 3
        r_elbow = random.randint(row_min, r_top - 2)
        c_elbow = random.randint(col_min, c_left - 2)
        c_end = 0 if horiz_status == "open" else random.randint(1, c_elbow - 2)
        r_end = 0 if vert_status == "open" else random.randint(1, r_elbow - 2)
        points = [(r_elbow, c_end), (r_elbow, c_elbow), (r_end, c_elbow)]
    elif corner == "tr":
        row_min = 2 if vert_status == "open" else 3
        col_max = w - 3 if horiz_status == "open" else w - 4
        r_elbow = random.randint(row_min, r_top - 2)
        c_elbow = random.randint(c_right + 2, col_max)
        c_end = w - 1 if horiz_status == "open" else random.randint(c_elbow + 2, w - 2)
        r_end = 0 if vert_status == "open" else random.randint(1, r_elbow - 2)
        points = [(r_elbow, c_end), (r_elbow, c_elbow), (r_end, c_elbow)]
    elif corner == "bl":
        row_max = h - 3 if vert_status == "open" else h - 4
        col_min = 2 if horiz_status == "open" else 3
        r_elbow = random.randint(r_bottom + 2, row_max)
        c_elbow = random.randint(col_min, c_left - 2)
        c_end = 0 if horiz_status == "open" else random.randint(1, c_elbow - 2)
        r_end = h - 1 if vert_status == "open" else random.randint(r_elbow + 2, h - 2)
        points = [(r_elbow, c_end), (r_elbow, c_elbow), (r_end, c_elbow)]
    else:
        row_max = h - 3 if vert_status == "open" else h - 4
        col_max = w - 3 if horiz_status == "open" else w - 4
        r_elbow = random.randint(r_bottom + 2, row_max)
        c_elbow = random.randint(c_right + 2, col_max)
        c_end = w - 1 if horiz_status == "open" else random.randint(c_elbow + 2, w - 2)
        r_end = h - 1 if vert_status == "open" else random.randint(r_elbow + 2, h - 2)
        points = [(r_elbow, c_end), (r_elbow, c_elbow), (r_end, c_elbow)]

    return _polyline(points)


def _orientation_at(path: list[tuple[int, int]], cell: tuple[int, int]) -> str:
    idx = path.index(cell)
    r, c = cell
    if idx == 0:
        rr, cc = path[1]
        return "h" if rr == r else "v"
    if idx == len(path) - 1:
        rr, cc = path[-2]
        return "h" if rr == r else "v"

    pr, pc = path[idx - 1]
    nr, nc = path[idx + 1]
    if pr == r == nr:
        return "h"
    if pc == c == nc:
        return "v"
    return "turn"


def _rasterize_pipes(
    shape: tuple[int, int], pipes: list[dict[str, object]]
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    cell_to_pipes: dict[tuple[int, int], list[int]] = {}
    for idx, pipe in enumerate(pipes):
        for cell in pipe["path"]:
            cell_to_pipes.setdefault(cell, []).append(idx)

    ownership: dict[tuple[int, int], int] = {}
    for cell, owners in cell_to_pipes.items():
        if len(owners) == 1:
            ownership[cell] = owners[0]
            continue

        assert len(owners) == 2, "only pairwise crossings are allowed"
        ori0 = _orientation_at(pipes[owners[0]]["path"], cell)
        ori1 = _orientation_at(pipes[owners[1]]["path"], cell)
        assert {ori0, ori1} == {"h", "v"}, "crossings must join straight orthogonal segments"
        ownership[cell] = random.choice(owners)

    grid = np.zeros(shape, dtype=int)
    for cell, owner in ownership.items():
        r, c = cell
        grid[r, c] = int(pipes[owner]["color"])
    return grid, ownership


def _expected_output(grid: np.ndarray, pipes: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for pipe in pipes:
        path = pipe["path"]
        visible = pipe["visible"]
        for r, c in _recolor_targets(path, visible, grid.shape):
            out[r, c] = GREY
    return out


def _validate_generated_case(grid: np.ndarray, pipes: list[dict[str, object]]) -> bool:
    for pipe in pipes:
        traced_path, traced_visible = _trace_pipe(grid, int(pipe["color"]))
        if set(traced_path) != set(pipe["path"]):
            return False
        if traced_visible != pipe["visible"]:
            return False

    expected = _expected_output(grid, pipes)
    actual = main(grid)
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    try:
        h = random.randint(15, 24)
        w = random.randint(15, 24)
        shape = (h, w)

        r_mid = random.randint(5, h - 6)
        c_mid = random.randint(5, w - 6)
        r_top = random.randint(2, r_mid - 2)
        r_bottom = random.randint(r_mid + 2, h - 3)
        c_left = random.randint(2, c_mid - 2)
        c_right = random.randint(c_mid + 2, w - 3)

        corners = _corner_boxes(shape, r_top, r_bottom, c_left, c_right)
        if not corners:
            return None

        branches = list(BRANCHES)
        random.shuffle(branches)
        colors = random.sample(COLOR_POOL, 3)

        pipes: list[dict[str, object]] = [
            {
                "color": colors[0],
                "path": _build_h_pipe(shape, branches[0], r_mid, c_left, c_right, {r_top, r_bottom}),
            },
            {
                "color": colors[1],
                "path": _build_v_pipe(shape, branches[1], c_mid, r_top, r_bottom, {c_left, c_right}),
            },
            {
                "color": colors[2],
                "path": _build_corner_pipe(
                    shape, branches[2], random.choice(corners), r_top, r_bottom, c_left, c_right
                ),
            },
        ]

        grid, ownership = _rasterize_pipes(shape, pipes)
        for idx, pipe in enumerate(pipes):
            pipe["visible"] = {cell for cell, owner in ownership.items() if owner == idx}
            assert pipe["visible"]

        assert np.max(grid.shape) <= 30
        if not _validate_generated_case(grid, pipes):
            return None
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-crossing ARC input")
