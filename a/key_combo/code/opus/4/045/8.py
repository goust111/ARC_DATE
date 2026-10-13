import random
from itertools import combinations, product

import numpy as np


BG = 0
WIRE = 7
TOKEN_COLORS = (1, 2, 3)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS)}
RELATIVE_DIRS = ("R", "S", "L")
SHAPE_ORDER = ("solid", "x", "diamond")
SHAPE_MASKS = {
    "solid": np.ones((3, 3), dtype=bool),
    "x": np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    "diamond": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
}
PAIR_B2 = [(1, 3), (2, 2), (2, 3)]
PAIR_B3 = [(1, 1, 3), (1, 2, 2), (2, 1, 2), (1, 2, 3), (2, 1, 3), (2, 2, 2)]
GEN_ATTEMPTS = 400


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _scale(d, k):
    return d[0] * k, d[1] * k


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r, c):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _right(d):
    return DIRS[(DIR_TO_INDEX[d] + 1) % 4]


def _left(d):
    return DIRS[(DIR_TO_INDEX[d] - 1) % 4]


def _reverse(d):
    return -d[0], -d[1]


def _relative_dir(incoming, rel):
    if rel == "R":
        return _right(incoming)
    if rel == "S":
        return incoming
    return _left(incoming)


def _shape_to_index(name):
    return SHAPE_ORDER.index(name)


def _index_to_shape(idx):
    return SHAPE_ORDER[int(idx) % 3]


def _draw_shape(out, center, shape_name, color):
    mask = SHAPE_MASKS[shape_name]
    cr, cc = center
    r0 = cr - 1
    c0 = cc - 1
    out[r0 : r0 + 3, c0 : c0 + 3][mask] = int(color)


def _match_shape(mask):
    for name, ref in SHAPE_MASKS.items():
        if np.array_equal(mask, ref):
            return name
    return None


def _wire_neighbors(cell, wire_set):
    r, c = cell
    return [nb for nb in _neighbors4(r, c) if nb in wire_set]


def _find_token(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            if np.any(sub == WIRE):
                continue
            colors = set(int(v) for v in sub.ravel() if int(v) not in (BG, WIRE))
            if len(colors) != 1:
                continue
            color = next(iter(colors))
            if not np.all((sub == BG) | (sub == color)):
                continue
            mask = sub == color
            shape_name = _match_shape(mask)
            if shape_name is None:
                continue
            token_cells = {(r0 + r, c0 + c) for r, c in np.argwhere(mask)}
            adjacent_wires = set()
            adjacent_token_cells = []
            for tr, tc in token_cells:
                for nr, nc in _neighbors4(tr, tc):
                    if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == WIRE:
                        adjacent_wires.add((nr, nc))
                        adjacent_token_cells.append((tr, tc))
            if len(adjacent_wires) != 1:
                continue
            root = next(iter(adjacent_wires))
            touching = [cell for cell in token_cells if root in set(_neighbors4(*cell))]
            if len(touching) != 1:
                continue
            candidates.append(
                {
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "color": color,
                    "shape": shape_name,
                    "cells": token_cells,
                    "root": root,
                    "touch_cell": touching[0],
                    "center": (r0 + 1, c0 + 1),
                }
            )
    assert len(candidates) == 1, "expected exactly one valid token"
    return candidates[0]


def _analyze_grid(grid):
    g = np.asarray(grid, dtype=int)
    token = _find_token(g)
    wire_set = {tuple(map(int, cell)) for cell in np.argwhere(g == WIRE)}
    assert token["root"] in wire_set, "root must be an orange wire cell"

    seen = set()
    stack = [token["root"]]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        stack.extend(nb for nb in _wire_neighbors(cell, wire_set) if nb not in seen)
    assert seen == wire_set, "wire network must be connected"

    degrees = {cell: len(_wire_neighbors(cell, wire_set)) for cell in wire_set}
    entry_dir = (
        token["root"][0] - token["touch_cell"][0],
        token["root"][1] - token["touch_cell"][1],
    )
    branches = {cell for cell, deg in degrees.items() if deg >= 3 and cell != token["root"]}
    terminals = {cell for cell, deg in degrees.items() if deg == 1 and cell != token["root"]}
    return {
        "grid": g,
        "token": token,
        "wire_set": wire_set,
        "degrees": degrees,
        "root": token["root"],
        "entry_dir": entry_dir,
        "branches": branches,
        "terminals": terminals,
    }


def _ordered_neighbors(node, parent, incoming, wire_set):
    nbs = [nb for nb in _wire_neighbors(node, wire_set) if nb != parent]
    if incoming is None:
        return nbs
    priority = [_right(incoming), incoming, _left(incoming), _reverse(incoming)]
    rank = {d: i for i, d in enumerate(priority)}
    return sorted(
        nbs,
        key=lambda nb: (
            rank.get((nb[0] - node[0], nb[1] - node[1]), 99),
            nb[0],
            nb[1],
        ),
    )


def _paint_terminal(out, terminal, parent, shape_name, color):
    direction = (terminal[0] - parent[0], terminal[1] - parent[1])
    center = _add(terminal, _scale(direction, 2))
    cr, cc = center
    assert 1 <= cr < out.shape[0] - 1 and 1 <= cc < out.shape[1] - 1
    _draw_shape(out, center, shape_name, color)


def _solve(grid, return_trace=False):
    info = _analyze_grid(grid)
    out = np.asarray(grid, dtype=int).copy()
    start_shape_idx = _shape_to_index(info["token"]["shape"])
    color = info["token"]["color"]
    visited = set()
    trace = {}

    def dfs(node, parent, incoming, shape_idx):
        if node in visited:
            return
        visited.add(node)

        current_shape_idx = shape_idx
        if node in info["branches"]:
            current_shape_idx = (current_shape_idx + 1) % 3

        if node in info["terminals"]:
            shape_name = _index_to_shape(current_shape_idx)
            _paint_terminal(out, node, parent, shape_name, color)
            trace[node] = shape_name
            return

        for nb in _ordered_neighbors(node, parent, incoming, info["wire_set"]):
            dfs(nb, node, (nb[0] - node[0], nb[1] - node[1]), current_shape_idx)

    dfs(info["root"], None, info["entry_dir"], start_shape_idx)
    if return_trace:
        return out, info, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transmit a 3x3 token through an orange wire tree, cycling its shape at every branch before drawing it at each terminal.

    Input structure:
    - Black background contains a connected 1-cell-thick orange wire network with one root wire pixel adjacent to a colored 3x3 token.
    - The token is monochrome blue, red, or green and is exactly one of three 3x3 masks: solid, X, or diamond.
    - Orange wire pixels with degree at least 3 are branch nodes, and degree-1 wire pixels not touching the token are terminal nodes.
    - Each terminal has a neighboring empty 3x3 black slot reserved for the output shape.

    Rule:
    - Keep the orange wire network and the original token unchanged.
    - Start from the root and traverse the wire network depth-first.
    - At each branch node, cycle the carried token shape once in the order Solid -> X -> Diamond -> Solid.
    - When choosing among outgoing branches, use the direction priority relative to the incoming direction: turn right first, then go straight, then turn left.
    - When a terminal is first reached, draw the carried 3x3 shape in its reserved slot using the token's original color.

    Key insight:
    - A terminal's output depends on how many branch nodes lie on the root-to-terminal path modulo 3, while the geometry of the wire network determines both those depths and the DFS transmission order.
    """
    return _solve(grid)


def _macro_to_grid(cell):
    return 3 + 4 * cell[0], 3 + 4 * cell[1]


def _macro_path(start, direction, length):
    return [
        (start[0] + direction[0] * step, start[1] + direction[1] * step)
        for step in range(1, length + 1)
    ]


def _within_macro(shape, cell):
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _reserve_path(occupied, shape, start, direction, length):
    path = _macro_path(start, direction, length)
    if not path:
        return None
    if not all(_within_macro(shape, cell) for cell in path):
        return None
    if any(cell in occupied for cell in path):
        return None
    return path


def _render_wire_segment(grid, start, end):
    r0, c0 = _macro_to_grid(start)
    r1, c1 = _macro_to_grid(end)
    if r0 == r1:
        grid[r0, min(c0, c1) : max(c0, c1) + 1] = WIRE
    else:
        grid[min(r0, r1) : max(r0, r1) + 1, c0] = WIRE


def _slot_bbox(center):
    cr, cc = center
    return cr - 1, cc - 1, cr + 1, cc + 1


def _boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_root(mshape):
    mh, mw = mshape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return (0, random.randint(1, mw - 2)), (1, 0), (-1, 0)
    if side == "bottom":
        return (mh - 1, random.randint(1, mw - 2)), (-1, 0), (1, 0)
    if side == "left":
        return (random.randint(1, mh - 2), 0), (0, 1), (0, -1)
    return (random.randint(1, mh - 2), mw - 1), (0, -1), (0, 1)


def _nonfinal_branch_options(side_count):
    for trunk_rel in RELATIVE_DIRS:
        remaining = [rel for rel in RELATIVE_DIRS if rel != trunk_rel]
        if side_count == 1:
            for rel in remaining:
                yield trunk_rel, (rel,)
        else:
            yield trunk_rel, tuple(remaining)


def _final_branch_options(leaf_count):
    rels = list(RELATIVE_DIRS)
    if leaf_count == 3:
        return [tuple(rels)]
    return list(combinations(rels, leaf_count))


def _choose_assignment(occupied, mshape, start, direction_map):
    rels = list(direction_map.keys())
    assignments = [dict(zip(rels, lengths)) for lengths in product((1, 2), repeat=len(rels))]
    random.shuffle(assignments)

    for length_map in assignments:
        trial_occ = set(occupied)
        placed = {}
        ok = True
        for rel, direction in direction_map.items():
            path = _reserve_path(trial_occ, mshape, start, direction, length_map[rel])
            if path is None:
                ok = False
                break
            trial_occ.update(path)
            placed[rel] = (path[-1], length_map[rel])
        if ok:
            return placed, trial_occ
    return None, None


def _sample_layout():
    branch_count = random.choice((2, 3))
    if branch_count == 2:
        side1, final_count = random.choice(PAIR_B2)
        side_counts = [side1]
        total_terminals = side1 + final_count
    else:
        side1, side2, final_count = random.choice(PAIR_B3)
        side_counts = [side1, side2]
        total_terminals = side1 + side2 + final_count

    mh = random.choice((5, 6))
    mw = random.choice((5, 6))
    mshape = (mh, mw)
    root, trunk_dir, token_dir = _choose_root(mshape)
    occupied = {root}
    segments = []
    leaves = []
    branch_nodes = []

    first_len = random.choice((1, 2))
    first_path = _reserve_path(occupied, mshape, root, trunk_dir, first_len)
    if first_path is None:
        return None
    occupied.update(first_path)
    first_branch = first_path[-1]
    segments.append((root, first_branch))
    branch_nodes.append(first_branch)

    current = first_branch
    incoming = trunk_dir

    for idx, side_count in enumerate(side_counts):
        options = list(_nonfinal_branch_options(side_count))
        random.shuffle(options)
        placed = None
        next_occ = None
        for trunk_rel, side_rels in options:
            direction_map = {rel: _relative_dir(incoming, rel) for rel in (trunk_rel,) + side_rels}
            placed, next_occ = _choose_assignment(occupied, mshape, current, direction_map)
            if placed is None:
                continue
            trunk_direction = direction_map[trunk_rel]
            next_node = placed[trunk_rel][0]
            occupied = next_occ
            for rel in side_rels:
                end = placed[rel][0]
                leaf_dir = direction_map[rel]
                segments.append((current, end))
                leaves.append((end, leaf_dir, idx + 1))
            segments.append((current, next_node))
            branch_nodes.append(next_node)
            current = next_node
            incoming = trunk_direction
            break
        if placed is None:
            return None

    final_options = _final_branch_options(final_count)
    random.shuffle(final_options)
    final_ok = False
    for leaf_rels in final_options:
        direction_map = {rel: _relative_dir(incoming, rel) for rel in leaf_rels}
        placed, next_occ = _choose_assignment(occupied, mshape, current, direction_map)
        if placed is None:
            continue
        occupied = next_occ
        for rel in leaf_rels:
            end = placed[rel][0]
            leaf_dir = direction_map[rel]
            segments.append((current, end))
            leaves.append((end, leaf_dir, branch_count))
        final_ok = True
        break
    if not final_ok:
        return None

    assert len(leaves) == total_terminals
    return {
        "mshape": mshape,
        "root": root,
        "trunk_dir": trunk_dir,
        "token_dir": token_dir,
        "segments": segments,
        "branch_nodes": branch_nodes,
        "leaves": leaves,
        "branch_count": branch_count,
    }


def _render_layout(layout):
    mh, mw = layout["mshape"]
    h = 4 * mh + 3
    w = 4 * mw + 3
    grid = np.zeros((h, w), dtype=int)
    for start, end in layout["segments"]:
        _render_wire_segment(grid, start, end)

    token_shape = random.choice(SHAPE_ORDER)
    token_color = random.choice(TOKEN_COLORS)
    root_center = _macro_to_grid(layout["root"])
    token_center = _add(root_center, _scale(layout["token_dir"], 2))
    _draw_shape(grid, token_center, token_shape, token_color)

    terminal_slots = []
    for cell, direction, depth in layout["leaves"]:
        center = _add(_macro_to_grid(cell), _scale(direction, 2))
        terminal_slots.append((center, depth))
    return grid, token_shape, token_color, terminal_slots


def _expected_output(inp, token_shape, token_color, terminal_slots):
    out = inp.copy()
    start_idx = _shape_to_index(token_shape)
    for center, depth in terminal_slots:
        shape_name = _index_to_shape(start_idx + depth)
        _draw_shape(out, center, shape_name, token_color)
    return out


def _validate_generated(inp, expected, layout, token_shape, terminal_slots):
    solved, info, trace = _solve(inp, return_trace=True)
    assert np.array_equal(solved, expected), "solver output must match constructed target"
    assert len(info["branches"]) == layout["branch_count"], "branch count mismatch"
    assert 4 <= len(info["terminals"]) <= 6, "terminal count out of range"
    assert info["root"] == _macro_to_grid(layout["root"]), "root mismatch"
    assert info["token"]["shape"] == token_shape, "token shape mismatch"

    terminal_centers = []
    for center, _depth in terminal_slots:
        r0, c0, r1, c1 = _slot_bbox(center)
        assert 0 <= r0 and 0 <= c0 and r1 < inp.shape[0] and c1 < inp.shape[1], "slot out of bounds"
        assert np.all(inp[r0 : r1 + 1, c0 : c1 + 1] == BG), "terminal slot must be blank in the input"
        terminal_centers.append(center)

    token_box = _slot_bbox(info["token"]["center"])
    for center in terminal_centers:
        box = _slot_bbox(center)
        assert not _boxes_overlap(token_box, box), "token and terminal slots must not overlap"

    for i in range(len(terminal_centers)):
        for j in range(i + 1, len(terminal_centers)):
            assert not _boxes_overlap(
                _slot_bbox(terminal_centers[i]), _slot_bbox(terminal_centers[j])
            ), "terminal slots must be disjoint"

    depths = [depth for _center, depth in terminal_slots]
    assert len(set(depths)) >= 2, "must expose at least two branch depths"
    if layout["branch_count"] == 3:
        assert len(set(depths)) == 3, "three-branch layouts must expose all three depths"

    output_shapes = {
        _index_to_shape(_shape_to_index(token_shape) + depth)
        for _center, depth in terminal_slots
    }
    target_shape_variety = 3 if layout["branch_count"] == 3 else 2
    assert len(output_shapes) >= target_shape_variety, "output shape variety is too low"

    root_rc = _macro_to_grid(layout["root"])
    for branch in layout["branch_nodes"]:
        cell = _macro_to_grid(branch)
        assert cell != root_rc

    for terminal in info["terminals"]:
        parent = _wire_neighbors(terminal, info["wire_set"])[0]
        center = _add(terminal, _scale((terminal[0] - parent[0], terminal[1] - parent[1]), 2))
        box = _slot_bbox(center)
        assert np.all(inp[box[0] : box[2] + 1, box[1] : box[3] + 1] == BG), "solver-facing slot must be blank"

    for term, shape_name in trace.items():
        assert shape_name in SHAPE_ORDER


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        layout = _sample_layout()
        if layout is None:
            continue
        inp, token_shape, token_color, terminal_slots = _render_layout(layout)
        expected = _expected_output(inp, token_shape, token_color, terminal_slots)
        try:
            _validate_generated(inp, expected, layout, token_shape, terminal_slots)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid recursive transmit puzzle")
