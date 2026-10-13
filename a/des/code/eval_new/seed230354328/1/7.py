import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

TRACK_COLORS = {YELLOW, GREY, RED, BLUE, MAROON}
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

MAX_GENERATION_ATTEMPTS = 200
BRANCH_COUNT_RANGE = (1, 4)
COL_BASE_CHOICES = (1, 2, 3)
COL_STEP_CHOICES = (4, 6)
ROW_STEP_CHOICES = (3, 4)
SOURCE_GAP_CHOICES = (2, 3, 4)
MAX_EXTRA_PADDING = 6


def main(grid: np.ndarray) -> np.ndarray:
    """Light the terminal reached by the color-directed signal in the grey circuit tree.

    Input structure:
    - The background is black.
    - Grey pixels form a single one-cell-wide orthogonal wire tree with no loops.
    - One leaf is a yellow source, several other leaves are maroon terminals, and
      every degree-3 T junction is recolored red or blue.

    Rule:
    - Start at the yellow source and follow the connected wire path.
    - When the signal reaches a red branch, continue along the outgoing branch on
      the right of the arrival direction; at a blue branch, take the outgoing
      branch on the left.
    - Repeat this step-by-step tracing until the signal reaches a maroon leaf.
    - Copy the grid and recolor only the reached terminal to green.

    Key insight:
    - The branch colors encode relative turns, so the correct terminal can only
      be found by simulating the signal with its current travel direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    reached = _trace_signal(g)
    out = g.copy()
    out[reached] = GREEN
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        try:
            out = main(grid)
        except AssertionError:
            continue
        if _valid_sample(grid, out):
            return grid
    raise RuntimeError("failed to generate a valid circuit sample")


def _trace_signal(grid: np.ndarray) -> tuple[int, int]:
    positions, adjacency = _build_adjacency(grid)
    source_cells = np.argwhere(grid == YELLOW)
    assert source_cells.shape == (1, 2)
    source = tuple(map(int, source_cells[0]))
    assert source in positions
    assert len(adjacency[source]) == 1

    _assert_connected(source, positions, adjacency)

    for pos in positions:
        color = int(grid[pos])
        degree = len(adjacency[pos])
        if color in (YELLOW, MAROON):
            assert degree == 1
        elif color in (RED, BLUE):
            assert degree == 3
        else:
            assert color == GREY
            assert degree == 2

    previous = source
    current = adjacency[source][0]
    direction = (current[0] - previous[0], current[1] - previous[1])
    visited_edges = {(source, current), (current, source)}

    while True:
        color = int(grid[current])
        if color == MAROON:
            return current

        forward = [nxt for nxt in adjacency[current] if nxt != previous]
        if color == RED:
            desired_dir = _turn_right(direction)
            desired = (current[0] + desired_dir[0], current[1] + desired_dir[1])
            assert desired in forward
            nxt = desired
        elif color == BLUE:
            desired_dir = _turn_left(direction)
            desired = (current[0] + desired_dir[0], current[1] + desired_dir[1])
            assert desired in forward
            nxt = desired
        else:
            assert color == GREY
            assert len(forward) == 1
            nxt = forward[0]

        assert (current, nxt) not in visited_edges
        visited_edges.add((current, nxt))
        visited_edges.add((nxt, current))
        previous, current = current, nxt
        direction = (current[0] - previous[0], current[1] - previous[1])


def _build_adjacency(grid: np.ndarray) -> tuple[set[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    mask = np.isin(grid, list(TRACK_COLORS))
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert coords

    height, width = grid.shape
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        neighbors = []
        for dr, dc in DIRS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) in coords:
                neighbors.append((nr, nc))
        adjacency[(r, c)] = neighbors
    return coords, adjacency


def _assert_connected(
    source: tuple[int, int],
    positions: set[tuple[int, int]],
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
) -> None:
    stack = [source]
    seen = {source}
    while stack:
        cur = stack.pop()
        for nxt in adjacency[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    assert seen == positions


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _generate_attempt() -> np.ndarray:
    branch_count = random.randint(*BRANCH_COUNT_RANGE)
    tree = _build_tree(branch_count)
    _assign_depths_and_slots(tree, depth=0, next_slot=[0])

    col_base = random.choice(COL_BASE_CHOICES)
    col_step = random.choice(COL_STEP_CHOICES)
    root_row = random.choice(SOURCE_GAP_CHOICES)
    row_step = random.choice(ROW_STEP_CHOICES)
    _assign_positions(tree, col_base, col_step, root_row, row_step)

    minimal = _render_tree(tree)
    padded = _pad_canvas(minimal)
    rotation = random.randrange(4)
    return np.rot90(padded, k=rotation).copy()


def _build_tree(branches: int) -> dict:
    if branches == 0:
        return {"kind": "leaf"}

    left_branches = random.randint(0, branches - 1)
    right_branches = branches - 1 - left_branches
    return {
        "kind": "branch",
        "color": RED if random.random() < 0.5 else BLUE,
        "left": _build_tree(left_branches),
        "right": _build_tree(right_branches),
    }


def _assign_depths_and_slots(node: dict, depth: int, next_slot: list[int]) -> None:
    node["depth"] = depth
    if node["kind"] == "leaf":
        node["slot"] = next_slot[0]
        next_slot[0] += 1
        return

    _assign_depths_and_slots(node["left"], depth + 1, next_slot)
    _assign_depths_and_slots(node["right"], depth + 1, next_slot)


def _assign_positions(node: dict, col_base: int, col_step: int, root_row: int, row_step: int) -> None:
    if node["kind"] == "leaf":
        node["col"] = col_base + node["slot"] * col_step
    else:
        _assign_positions(node["left"], col_base, col_step, root_row, row_step)
        _assign_positions(node["right"], col_base, col_step, root_row, row_step)
        node["col"] = (node["left"]["col"] + node["right"]["col"]) // 2
    node["row"] = root_row + node["depth"] * row_step


def _render_tree(tree: dict) -> np.ndarray:
    nodes: list[tuple[int, int, int]] = []
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    _collect_tree_geometry(tree, nodes, segments)

    source = (0, int(tree["col"]))
    nodes.append((source[0], source[1], YELLOW))
    segments.append((source, (int(tree["row"]), int(tree["col"]))))

    max_row = max(r for r, _c, _color in nodes)
    max_col = max(c for _r, c, _color in nodes)
    canvas = np.full((max_row + 1, max_col + 1), BLACK, dtype=int)

    for start, end in segments:
        _draw_straight(canvas, start, end, GREY)
    for r, c, color in nodes:
        canvas[r, c] = color
    return canvas


def _collect_tree_geometry(
    node: dict,
    nodes: list[tuple[int, int, int]],
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    row = int(node["row"])
    col = int(node["col"])

    if node["kind"] == "leaf":
        nodes.append((row, col, MAROON))
        return

    nodes.append((row, col, int(node["color"])))
    for child_key in ("left", "right"):
        child = node[child_key]
        child_row = int(child["row"])
        child_col = int(child["col"])
        segments.append(((row, col), (row, child_col)))
        segments.append(((row, child_col), (child_row, child_col)))
        _collect_tree_geometry(child, nodes, segments)


def _draw_straight(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo = min(c0, c1)
        hi = max(c0, c1)
        canvas[r0, lo : hi + 1] = color
        return
    assert c0 == c1
    lo = min(r0, r1)
    hi = max(r0, r1)
    canvas[lo : hi + 1, c0] = color


def _pad_canvas(grid: np.ndarray) -> np.ndarray:
    min_h, min_w = grid.shape
    final_h = min_h + random.randint(0, min(MAX_EXTRA_PADDING, 30 - min_h))
    final_w = min_w + random.randint(0, min(MAX_EXTRA_PADDING, 30 - min_w))
    top = random.randint(0, final_h - min_h)
    left = random.randint(0, final_w - min_w)
    bottom = final_h - min_h - top
    right = final_w - min_w - left
    return np.pad(grid, ((top, bottom), (left, right)), constant_values=BLACK)


def _valid_sample(grid: np.ndarray, out: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or not isinstance(out, np.ndarray):
        return False
    if grid.ndim != 2 or out.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if np.count_nonzero(grid == YELLOW) != 1:
        return False
    if np.count_nonzero(grid == MAROON) < 2:
        return False
    changed = np.argwhere(grid != out)
    if changed.shape != (1, 2):
        return False
    r, c = map(int, changed[0])
    if int(grid[r, c]) != MAROON or int(out[r, c]) != GREEN:
        return False
    return True
