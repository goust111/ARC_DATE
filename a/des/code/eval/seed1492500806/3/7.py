from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

LEGEND_WIDTH = 2
TREE_MIN_LEFT = 4
MAX_DIM = 30
GENERATION_RETRIES = 400

DEPTH_RANGE = (2, 4)
BRANCH_LEN_RANGE = (1, 3)
TOP_MARGIN_RANGE = (0, 2)
BOTTOM_MARGIN_RANGE = (1, 4)
LEFT_MARGIN_RANGE = (TREE_MIN_LEFT, TREE_MIN_LEFT + 3)
RIGHT_MARGIN_RANGE = (1, 4)
LEGEND_TOP_RANGE = (0, 2)
EXTRA_CHILD_PROB = 0.55


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _extract_legend(grid: np.ndarray) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    assert w >= LEGEND_WIDTH

    mapping: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    r = 0
    while r < h - 1:
        block = grid[r : r + 2, :LEGEND_WIDTH]
        left = int(block[0, 0])
        right = int(block[0, 1])
        is_block = (
            left not in (BLACK, GREY)
            and right not in (BLACK, GREY)
            and bool(np.all(block[:, 0] == left))
            and bool(np.all(block[:, 1] == right))
        )
        if not is_block:
            r += 1
            continue

        assert mapping.get(left, right) == right
        mapping[left] = right
        legend_cells.update({(r, 0), (r, 1), (r + 1, 0), (r + 1, 1)})
        r += 2
        while r < h and bool(np.all(grid[r, :LEGEND_WIDTH] == BLACK)):
            r += 1

    assert mapping, "expected at least one legend block"
    return mapping, legend_cells


def _scan_child(grid: np.ndarray, start: tuple[int, int], dc: int) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    r, c = start
    rr, cc = r + 1, c + dc
    if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != GREY:
        return None

    path: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == GREY:
        path.append((rr, cc))
        rr += 1
        cc += dc

    if not _in_bounds(grid.shape, rr, cc):
        return None

    cell = int(grid[rr, cc])
    if cell in (BLACK, GREY):
        return None
    return (rr, cc), path


def _analyze_tree(
    grid: np.ndarray,
) -> tuple[dict[int, int], set[tuple[int, int]], tuple[int, int], dict[tuple[int, int], list[tuple[int, int]]], set[tuple[int, int]]]:
    mapping, legend_cells = _extract_legend(grid)
    work = np.asarray(grid, dtype=int).copy()
    for r, c in legend_cells:
        work[r, c] = BLACK

    grey_positions = [tuple(map(int, rc)) for rc in np.argwhere(work == GREY)]
    color_positions = [tuple(map(int, rc)) for rc in np.argwhere((work != BLACK) & (work != GREY))]
    assert grey_positions and color_positions, "expected a grey-rooted tree"

    top_row = min(r for r, _c in grey_positions)
    root_candidates = [pos for pos in grey_positions if pos[0] == top_row]
    assert len(root_candidates) == 1, "root must be the unique topmost grey pixel"
    root = root_candidates[0]

    colored_nodes = set(color_positions)
    parents = [root] + sorted(colored_nodes)
    children: dict[tuple[int, int], list[tuple[int, int]]] = {parent: [] for parent in parents}
    indegree = {node: 0 for node in colored_nodes}
    used_branch_cells: set[tuple[int, int]] = set()

    for parent in parents:
        for dc in (-1, 0, 1):
            scanned = _scan_child(work, parent, dc)
            if scanned is None:
                continue
            child, path = scanned
            assert child in colored_nodes, "branches must terminate at colored nodes"
            children[parent].append(child)
            indegree[child] += 1
            used_branch_cells.update(path)

        assert len(children[parent]) == len(set(children[parent])), "duplicate child detection"

    assert all(count == 1 for count in indegree.values()), "every colored node must have exactly one parent"
    assert used_branch_cells == set(grey_positions) - {root}, "all non-root grey cells must belong to branches"

    seen = {root}
    queue = [root]
    while queue:
        parent = queue.pop(0)
        for child in children[parent]:
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)

    assert seen == {root} | colored_nodes, "tree must be connected from the root"
    return mapping, legend_cells, root, children, colored_nodes


def main(grid: np.ndarray) -> np.ndarray:
    """Apply legend-based recoloring to a grey-branch tree, with child nodes yielding to a parent's new color."""
    g = np.asarray(grid, dtype=int)
    mapping, legend_cells, root, children, colored_nodes = _analyze_tree(g)

    out = g.copy()
    for r, c in legend_cells:
        out[r, c] = BLACK

    transformed_color: dict[tuple[int, int], int | None] = {root: None}
    queue = [root]
    while queue:
        parent = queue.pop(0)
        parent_color = transformed_color[parent]
        for child in sorted(children[parent]):
            input_color = int(g[child])
            assert input_color in mapping, "every node color must appear in the legend"
            new_color = input_color if parent_color == input_color else int(mapping[input_color])
            out[child] = new_color
            transformed_color[child] = new_color
            queue.append(child)

    assert set(transformed_color) == {root} | colored_nodes
    return out


def _edge_geometry(start: tuple[int, int], direction: int, branch_len: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r, c = start
    path = [(r + step, c + direction * step) for step in range(1, branch_len + 1)]
    child = (r + branch_len + 1, c + direction * (branch_len + 1))
    return path, child


def _immediate_child_starts(node_positions: list[tuple[int, int]]) -> set[tuple[int, int]]:
    starts: set[tuple[int, int]] = set()
    for r, c in node_positions:
        for dc in (-1, 0, 1):
            starts.add((r + 1, c + dc))
    return starts


def _sample_level_mappings(depth: int) -> tuple[list[int], dict[int, int]] | None:
    used_inputs: set[int] = set()
    input_colors: list[int] = []
    mapping: dict[int, int] = {}

    prev_effective: int | None = None
    prev_mapped = False
    for level in range(depth):
        if level == 0:
            input_color = random.choice(PALETTE)
            avoided = False
        else:
            assert prev_effective is not None
            can_avoid = prev_mapped and prev_effective not in used_inputs
            avoided = can_avoid and random.random() < 0.5
            if avoided:
                input_color = prev_effective
            else:
                choices = [c for c in PALETTE if c not in used_inputs and c != prev_effective]
                if not choices:
                    return None
                input_color = random.choice(choices)

        used_inputs.add(input_color)
        output_choices = [c for c in PALETTE if c != input_color]
        preferred = [c for c in output_choices if c not in used_inputs]
        output_color = random.choice(preferred if preferred else output_choices)

        input_colors.append(input_color)
        mapping[input_color] = output_color
        prev_effective = input_color if avoided else output_color
        prev_mapped = not avoided

    return input_colors, mapping


def _try_add_leaf(
    nodes: list[dict],
    edges: list[dict],
    occupied: set[tuple[int, int]],
    parent_idx: int,
    direction: int,
) -> bool:
    parent_pos = nodes[parent_idx]["pos"]
    node_positions = [node["pos"] for node in nodes]
    parent_starts = {(parent_pos[0] + 1, parent_pos[1] + dc) for dc in (-1, 0, 1)}
    forbidden_starts = _immediate_child_starts(node_positions) - parent_starts

    lengths = list(range(BRANCH_LEN_RANGE[0], BRANCH_LEN_RANGE[1] + 1))
    random.shuffle(lengths)
    for branch_len in lengths:
        path, child_pos = _edge_geometry(parent_pos, direction, branch_len)
        cells = set(path + [child_pos])
        if cells & occupied:
            continue
        if any(cell in forbidden_starts for cell in path):
            continue

        child_idx = len(nodes)
        child_level = int(nodes[parent_idx]["level"]) + 1
        nodes.append(
            {
                "pos": child_pos,
                "level": child_level,
                "parent": parent_idx,
                "children": [],
                "dir_from_parent": direction,
            }
        )
        nodes[parent_idx]["children"].append(child_idx)
        edges.append({"parent": parent_idx, "child": child_idx, "path": path})
        occupied.update(cells)
        return True
    return False


def _build_tree() -> tuple[list[dict], list[dict]] | None:
    depth = random.randint(*DEPTH_RANGE)
    nodes = [{"pos": (0, 0), "level": -1, "parent": None, "children": [], "dir_from_parent": None}]
    edges: list[dict] = []
    occupied = {(0, 0)}

    parent_idx = 0
    spine_nodes: list[int] = []
    for level in range(depth):
        direction = random.choice((-1, 0, 1))
        branch_len = random.randint(*BRANCH_LEN_RANGE)
        path, child_pos = _edge_geometry(nodes[parent_idx]["pos"], direction, branch_len)
        child_idx = len(nodes)
        nodes.append(
            {
                "pos": child_pos,
                "level": level,
                "parent": parent_idx,
                "children": [],
                "dir_from_parent": direction,
            }
        )
        nodes[parent_idx]["children"].append(child_idx)
        edges.append({"parent": parent_idx, "child": child_idx, "path": path})
        occupied.update(path)
        occupied.add(child_pos)
        parent_idx = child_idx
        spine_nodes.append(child_idx)

    extra_children = 0
    for node_idx in [0] + spine_nodes[:-1]:
        used_dirs = {
            int(nodes[child_idx]["dir_from_parent"])
            for child_idx in nodes[node_idx]["children"]
        }
        available_dirs = [dc for dc in (-1, 0, 1) if dc not in used_dirs]
        random.shuffle(available_dirs)
        for direction in available_dirs:
            if random.random() < EXTRA_CHILD_PROB and _try_add_leaf(nodes, edges, occupied, node_idx, direction):
                extra_children += 1

    if extra_children == 0 and random.random() < 0.75:
        candidates = []
        for node_idx in [0] + spine_nodes[:-1]:
            used_dirs = {
                int(nodes[child_idx]["dir_from_parent"])
                for child_idx in nodes[node_idx]["children"]
            }
            for direction in (-1, 0, 1):
                if direction not in used_dirs:
                    candidates.append((node_idx, direction))
        random.shuffle(candidates)
        for node_idx, direction in candidates:
            if _try_add_leaf(nodes, edges, occupied, node_idx, direction):
                break

    return nodes, edges


def _tree_bbox(nodes: list[dict], edges: list[dict]) -> tuple[int, int, int, int]:
    cells = [node["pos"] for node in nodes]
    for edge in edges:
        cells.extend(edge["path"])
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _render_grid(
    nodes: list[dict],
    edges: list[dict],
    input_colors: list[int],
    mapping: dict[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    legend_top = random.randint(*LEGEND_TOP_RANGE)
    legend_height = legend_top + len(input_colors) * 2 + max(0, len(input_colors) - 1)

    min_r, min_c, max_r, max_c = _tree_bbox(nodes, edges)
    assert min_r == 0

    top_margin = random.randint(*TOP_MARGIN_RANGE)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    right_margin = random.randint(*RIGHT_MARGIN_RANGE)

    row_shift = top_margin
    col_shift = left_margin - min_c

    height = max(max_r + row_shift + 1 + bottom_margin, legend_height + bottom_margin)
    width = max_c + col_shift + 1 + right_margin
    if not (8 <= height <= MAX_DIM and 8 <= width <= MAX_DIM):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    expected = np.full((height, width), BLACK, dtype=int)

    legend_cells: set[tuple[int, int]] = set()
    legend_row = legend_top
    for input_color in input_colors:
        output_color = int(mapping[input_color])
        grid[legend_row : legend_row + 2, 0] = input_color
        grid[legend_row : legend_row + 2, 1] = output_color
        legend_cells.update({(legend_row, 0), (legend_row, 1), (legend_row + 1, 0), (legend_row + 1, 1)})
        legend_row += 3

    for node in nodes:
        r, c = node["pos"]
        rr, cc = r + row_shift, c + col_shift
        color = GREY if node["level"] < 0 else int(input_colors[int(node["level"])])
        grid[rr, cc] = color
        expected[rr, cc] = color

    for edge in edges:
        for r, c in edge["path"]:
            rr, cc = r + row_shift, c + col_shift
            grid[rr, cc] = GREY
            expected[rr, cc] = GREY

    transformed: dict[int, int | None] = {0: None}
    queue = [0]
    while queue:
        parent_idx = queue.pop(0)
        parent_color = transformed[parent_idx]
        for child_idx in nodes[parent_idx]["children"]:
            input_color = int(input_colors[int(nodes[child_idx]["level"])])
            new_color = input_color if parent_color == input_color else int(mapping[input_color])
            rr, cc = nodes[child_idx]["pos"]
            expected[rr + row_shift, cc + col_shift] = new_color
            transformed[child_idx] = new_color
            queue.append(child_idx)

    for r, c in legend_cells:
        expected[r, c] = BLACK

    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    built = _build_tree()
    if built is None:
        return None
    nodes, edges = built

    depth = max(int(node["level"]) for node in nodes)
    sampled = _sample_level_mappings(depth + 1)
    if sampled is None:
        return None
    input_colors, mapping = sampled

    rendered = _render_grid(nodes, edges, input_colors, mapping)
    if rendered is None:
        return None
    grid, expected = rendered

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random legend-plus-tree input that matches the solver's structural assumptions."""
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
