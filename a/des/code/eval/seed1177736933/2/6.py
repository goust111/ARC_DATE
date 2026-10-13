import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

INPUT_NODE_COLORS = (RED, BLUE, YELLOW)
COLOR_TO_BIT = {RED: 1, BLUE: 2, YELLOW: 4}
MASK_TO_COLOR = {
    1: RED,
    2: BLUE,
    4: YELLOW,
    3: MAGENTA,
    5: ORANGE,
    6: GREEN,
    7: GREY,
}

# --------------- generator bounds ---------------

MAX_GEN_ATTEMPTS = 400
INTERNAL_COUNT_CHOICES = [2, 3, 4]
INTERNAL_COUNT_WEIGHTS = [3, 5, 2]
MAX_DEPTH_CHOICES = [2, 3]
MAX_DEPTH_WEIGHTS = [3, 2]
LEAF_GAPS = (4, 5, 6)
TOP_TRUNK_RANGE = (2, 4)
LEVEL_STEP_RANGE = (3, 5)
LEAF_DROP_RANGE = (2, 5)
LEFT_PAD_RANGE = (1, 4)
RIGHT_PAD_RANGE = (1, 4)
BOTTOM_PAD_RANGE = (1, 4)


def _neighbors(shape, r, c):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _draw_edge(grid, start, end):
    r0, c0 = start
    r1, c1 = end
    step = 0 if c1 == c0 else (1 if c1 > c0 else -1)
    c = c0
    while c != c1:
        c += step
        grid[r0, c] = GREY
    for r in range(r0 + 1, r1 + 1):
        grid[r, c1] = GREY


def _new_leaf():
    return {"kind": "leaf"}


def _collect_expandable_leaves(node, depth, max_depth, out):
    if node["kind"] == "leaf":
        if depth < max_depth:
            out.append(node)
        return
    _collect_expandable_leaves(node["left"], depth + 1, max_depth, out)
    _collect_expandable_leaves(node["right"], depth + 1, max_depth, out)


def _shuffle_children(node, rng):
    if node["kind"] == "leaf":
        return
    if rng.random() < 0.5:
        node["left"], node["right"] = node["right"], node["left"]
    _shuffle_children(node["left"], rng)
    _shuffle_children(node["right"], rng)


def _inorder_leaves(node, out):
    if node["kind"] == "leaf":
        out.append(node)
        return
    _inorder_leaves(node["left"], out)
    _inorder_leaves(node["right"], out)


def _collect_branches(node, depth, out):
    if node["kind"] == "leaf":
        return
    out.append((node, depth))
    _collect_branches(node["left"], depth + 1, out)
    _collect_branches(node["right"], depth + 1, out)


def _assign_branch_columns(node, rng):
    if node["kind"] == "leaf":
        span = (int(node["col"]), int(node["col"]))
        node["span"] = span
        return span

    left_lo, left_hi = _assign_branch_columns(node["left"], rng)
    right_lo, right_hi = _assign_branch_columns(node["right"], rng)
    low = left_hi + 1
    high = right_lo - 1
    if low > high:
        raise ValueError("child subtrees are too close")
    node["col"] = rng.randint(low, high)
    node["span"] = (left_lo, right_hi)
    return node["span"]


def _assign_branch_rows(node, rng):
    branches = []
    _collect_branches(node, 0, branches)
    max_depth = max(depth for _branch, depth in branches)

    rows_by_depth = {}
    current = rng.randint(*TOP_TRUNK_RANGE)
    rows_by_depth[0] = current
    for depth in range(1, max_depth + 1):
        current += rng.randint(*LEVEL_STEP_RANGE)
        rows_by_depth[depth] = current

    for branch, depth in branches:
        branch["row"] = rows_by_depth[depth]


def _assign_leaf_rows(node, rng):
    if node["kind"] == "leaf":
        return
    parent_row = int(node["row"])
    for child in (node["left"], node["right"]):
        if child["kind"] == "leaf":
            child["row"] = parent_row + rng.randint(*LEAF_DROP_RANGE)
        else:
            _assign_leaf_rows(child, rng)


def _assign_branch_colors(node, rng):
    if node["kind"] == "leaf":
        return
    node["color"] = rng.choice(INPUT_NODE_COLORS)
    _assign_branch_colors(node["left"], rng)
    _assign_branch_colors(node["right"], rng)


def _collect_leaf_masks(node, mask, out):
    if node["kind"] == "leaf":
        out.append(mask)
        return
    next_mask = mask | COLOR_TO_BIT[int(node["color"])]
    _collect_leaf_masks(node["left"], next_mask, out)
    _collect_leaf_masks(node["right"], next_mask, out)


def _max_extent(node):
    max_row = int(node["row"])
    max_col = int(node["col"])
    if node["kind"] == "leaf":
        return max_row, max_col
    left_row, left_col = _max_extent(node["left"])
    right_row, right_col = _max_extent(node["right"])
    return max(max_row, left_row, right_row), max(max_col, left_col, right_col)


def _paint_branch_colors(grid, node, col_offset):
    if node["kind"] == "leaf":
        return
    grid[int(node["row"]), int(node["col"]) + col_offset] = int(node["color"])
    _paint_branch_colors(grid, node["left"], col_offset)
    _paint_branch_colors(grid, node["right"], col_offset)


def _draw_subtree(grid, node, col_offset):
    if node["kind"] == "leaf":
        return

    start = (int(node["row"]), int(node["col"]) + col_offset)
    for child in (node["left"], node["right"]):
        end = (int(child["row"]), int(child["col"]) + col_offset)
        _draw_edge(grid, start, end)
        _draw_subtree(grid, child, col_offset)


def _build_topology(rng):
    max_depth = rng.choices(MAX_DEPTH_CHOICES, weights=MAX_DEPTH_WEIGHTS, k=1)[0]
    max_internal = min(max(INTERNAL_COUNT_CHOICES), 2**max_depth - 1)
    count_choices = [count for count in INTERNAL_COUNT_CHOICES if count <= max_internal]
    count_weights = [
        INTERNAL_COUNT_WEIGHTS[INTERNAL_COUNT_CHOICES.index(count)]
        for count in count_choices
    ]
    target_internal = rng.choices(count_choices, weights=count_weights, k=1)[0]

    node = _new_leaf()
    for _ in range(target_internal):
        expandable = []
        _collect_expandable_leaves(node, 0, max_depth, expandable)
        if not expandable:
            return None
        chosen = rng.choice(expandable)
        chosen.clear()
        chosen["kind"] = "branch"
        chosen["left"] = _new_leaf()
        chosen["right"] = _new_leaf()

    _shuffle_children(node, rng)
    return node


def _gen_attempt(rng):
    topology = _build_topology(rng)
    if topology is None or topology["kind"] == "leaf":
        return None

    leaves = []
    _inorder_leaves(topology, leaves)
    col = 0
    for idx, leaf in enumerate(leaves):
        leaf["col"] = col
        if idx + 1 < len(leaves):
            col += rng.choice(LEAF_GAPS)

    try:
        _assign_branch_columns(topology, rng)
    except ValueError:
        return None

    _assign_branch_rows(topology, rng)
    _assign_leaf_rows(topology, rng)

    root_color = rng.choice(INPUT_NODE_COLORS)
    _assign_branch_colors(topology, rng)

    leaf_masks = []
    _collect_leaf_masks(topology, COLOR_TO_BIT[root_color], leaf_masks)
    leaf_colors = [MASK_TO_COLOR[mask] for mask in leaf_masks]
    if len(set(leaf_colors)) < 2:
        return None
    if not any(mask.bit_count() >= 2 for mask in leaf_masks):
        return None
    if not any(color != GREY for color in leaf_colors):
        return None

    max_row, max_col = _max_extent(topology)
    left_pad = rng.randint(*LEFT_PAD_RANGE)
    right_pad = rng.randint(*RIGHT_PAD_RANGE)
    bottom_pad = rng.randint(*BOTTOM_PAD_RANGE)
    height = max_row + bottom_pad + 1
    width = max_col + left_pad + right_pad + 1
    grid = np.full((height, width), BLACK, dtype=int)

    root_col = int(topology["col"]) + left_pad
    _draw_edge(grid, (0, root_col), (int(topology["row"]), root_col))
    _draw_subtree(grid, topology, left_pad)
    _paint_branch_colors(grid, topology, left_pad)
    grid[0, root_col] = root_color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each leaf of a grey tree using the combined set of node colors
    encountered from the root to that leaf.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.size > 0
    assert set(map(int, np.unique(g))) <= {BLACK, GREY, RED, BLUE, YELLOW}

    tree_cells = [tuple(map(int, rc)) for rc in np.argwhere(g != BLACK)]
    assert tree_cells, "grid must contain a tree"
    cell_set = set(tree_cells)
    top_row = min(r for r, _c in tree_cells)
    assert sum(1 for r, _c in tree_cells if r == top_row) == 1

    adjacency = {}
    degrees = {}
    for r, c in tree_cells:
        neighbors = [
            (nr, nc)
            for nr, nc in _neighbors(g.shape, r, c)
            if (nr, nc) in cell_set
        ]
        adjacency[(r, c)] = neighbors
        degrees[(r, c)] = len(neighbors)
        assert 1 <= len(neighbors) <= 3

    color_nodes = [
        cell for cell in tree_cells if int(g[cell]) in COLOR_TO_BIT
    ]
    assert color_nodes, "tree must contain colored nodes"
    root_candidates = [cell for cell in color_nodes if cell[0] == top_row]
    assert len(root_candidates) == 1
    root = root_candidates[0]
    assert degrees[root] == 1

    seen = set()
    stack = [root]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for neighbor in adjacency[cell]:
            if neighbor not in seen:
                stack.append(neighbor)
    assert len(seen) == len(tree_cells), "tree must be connected"
    assert sum(degrees.values()) // 2 == len(tree_cells) - 1, "structure must be acyclic"

    leaves = []
    for cell in tree_cells:
        value = int(g[cell])
        degree = degrees[cell]
        if cell == root:
            continue
        if value in COLOR_TO_BIT:
            assert degree == 3, "branch nodes must be T-junctions"
        else:
            assert value == GREY
            if degree == 1:
                leaves.append(cell)
            else:
                assert degree == 2, "grey non-leaf cells must be path cells"
    assert leaves, "tree must have at least one leaf"

    out = g.copy()
    walk = [(root, None, 0)]
    visited = set()
    while walk:
        cell, parent, mask = walk.pop()
        assert cell not in visited
        visited.add(cell)

        value = int(g[cell])
        if value in COLOR_TO_BIT:
            mask |= COLOR_TO_BIT[value]
        elif cell != root and degrees[cell] == 1:
            out[cell] = MASK_TO_COLOR[mask]

        for neighbor in adjacency[cell]:
            if neighbor != parent:
                walk.append((neighbor, cell, mask))

    assert len(visited) == len(tree_cells)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tree-mixing input")
