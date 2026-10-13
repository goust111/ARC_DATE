import random
from collections import defaultdict

import numpy as np


BG = 0
PATH = 5
NODE_SIZE = 3
NODE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_solid_3x3(comp):
    if len(comp) != NODE_SIZE * NODE_SIZE:
        return False
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != NODE_SIZE - 1 or c1 - c0 != NODE_SIZE - 1:
        return False
    comp_set = set(comp)
    return all((r, c) in comp_set for r in range(r0, r0 + NODE_SIZE) for c in range(c0, c0 + NODE_SIZE))


def _find_nodes(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    node_mask = np.zeros_like(g, dtype=bool)
    nodes = []
    colors = [int(v) for v in np.unique(g) if int(v) not in (BG, PATH)]
    for color in colors:
        node_comp = None
        for comp in _components(g == color):
            if _is_solid_3x3(comp):
                if node_comp is not None:
                    raise AssertionError(f"color {color} has multiple node blocks")
                rows = [r for r, _ in comp]
                cols = [c for _, c in comp]
                top = min(rows)
                left = min(cols)
                node_comp = {
                    "color": color,
                    "top": top,
                    "left": left,
                    "cells": [(r, c) for r in range(top, top + NODE_SIZE) for c in range(left, left + NODE_SIZE)],
                }
        if node_comp is None:
            raise AssertionError(f"color {color} has no 3x3 node block")
        for r, c in node_comp["cells"]:
            node_mask[r, c] = True
        nodes.append(node_comp)
    if len(nodes) < 3:
        raise AssertionError("expected at least three nodes")
    nodes.sort(key=lambda node: (node["top"], node["left"], node["color"]))
    cell_to_node = {}
    for idx, node in enumerate(nodes):
        node["id"] = idx
        for cell in node["cells"]:
            cell_to_node[cell] = idx
    return nodes, node_mask, cell_to_node


def _parse_path_component(grid: np.ndarray, comp, cell_to_node):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}

    if len(comp) == 1:
        r, c = comp[0]
        left = cell_to_node.get((r, c - 1))
        right = cell_to_node.get((r, c + 1))
        up = cell_to_node.get((r - 1, c))
        down = cell_to_node.get((r + 1, c))
        if left is not None and right is not None and left != right:
            orientation = "h"
            ordered = [(r, c)]
            node_a, node_b = left, right
        elif up is not None and down is not None and up != down:
            orientation = "v"
            ordered = [(r, c)]
            node_a, node_b = up, down
        else:
            raise AssertionError("single-cell path must bridge two opposite node edges")
    elif len(rows) == 1:
        orientation = "h"
        ordered = sorted(comp, key=lambda p: p[1])
        row = ordered[0][0]
        node_a = cell_to_node.get((row, ordered[0][1] - 1))
        node_b = cell_to_node.get((row, ordered[-1][1] + 1))
    elif len(cols) == 1:
        orientation = "v"
        ordered = sorted(comp, key=lambda p: p[0])
        col = ordered[0][1]
        node_a = cell_to_node.get((ordered[0][0] - 1, col))
        node_b = cell_to_node.get((ordered[-1][0] + 1, col))
    else:
        raise AssertionError("path component must be straight")

    if node_a is None or node_b is None or node_a == node_b:
        raise AssertionError("path must connect two distinct nodes")

    axis_values = [p[1] if orientation == "h" else p[0] for p in ordered]
    if axis_values != list(range(axis_values[0], axis_values[0] + len(axis_values))):
        raise AssertionError("path component must be contiguous")

    touched = set()
    for r, c in comp:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                node_id = cell_to_node.get((nr, nc))
                if node_id is not None:
                    touched.add(node_id)
    if touched != {node_a, node_b}:
        raise AssertionError("path touched unexpected nodes")

    marker_cells = [cell for cell in ordered if int(g[cell]) != PATH]
    marker_colors = [int(g[cell]) for cell in marker_cells]
    return {
        "orientation": orientation,
        "ordered": ordered,
        "nodes": (node_a, node_b),
        "marker_cells": marker_cells,
        "marker_colors": marker_colors,
    }


def _analyze_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")
    nodes, node_mask, cell_to_node = _find_nodes(g)
    path_mask = (g != BG) & (~node_mask)
    path_components = []
    votes = defaultdict(list)
    for comp in _components(path_mask):
        parsed = _parse_path_component(g, comp, cell_to_node)
        path_components.append(parsed)
        if parsed["marker_colors"]:
            node_a, node_b = parsed["nodes"]
            votes[node_a].append(parsed["marker_colors"][0])
            votes[node_b].append(parsed["marker_colors"][-1])
    if not path_components:
        raise AssertionError("expected at least one path")
    return {
        "nodes": nodes,
        "node_mask": node_mask,
        "path_mask": path_mask,
        "paths": path_components,
        "votes": votes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)

    out = g.copy()
    out[info["path_mask"]] = PATH

    for node in info["nodes"]:
        node_votes = info["votes"].get(node["id"], [])
        if node_votes and len(set(node_votes)) == 1:
            color = int(node_votes[0])
            out[node["top"] : node["top"] + NODE_SIZE, node["left"] : node["left"] + NODE_SIZE] = color
    return out


def _random_tree_slots(rng: random.Random):
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 5)
    capacity = rows * cols
    node_count = rng.randint(4, min(8, capacity))

    start = (rng.randrange(rows), rng.randrange(cols))
    occupied = [start]
    occupied_set = {start}
    edges = []

    while len(occupied) < node_count:
        candidates = []
        for r, c in occupied:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in occupied_set:
                    candidates.append(((r, c), (nr, nc)))
        if not candidates:
            return None
        parent, child = rng.choice(candidates)
        occupied.append(child)
        occupied_set.add(child)
        edges.append((parent, child))
    return rows, cols, occupied, edges


def _axis_positions(count: int, rng: random.Random):
    pos = [rng.randint(1, 3)]
    for _ in range(1, count):
        pos.append(pos[-1] + rng.randint(6, 9))
    return pos


def _build_layout(rng: random.Random):
    tree = _random_tree_slots(rng)
    if tree is None:
        return None
    rows, cols, slots, edges = tree
    row_pos = _axis_positions(rows, rng)
    col_pos = _axis_positions(cols, rng)
    height = row_pos[-1] + NODE_SIZE + rng.randint(1, 3)
    width = col_pos[-1] + NODE_SIZE + rng.randint(1, 3)
    grid = np.zeros((height, width), dtype=int)

    colors = list(rng.sample(NODE_COLORS, len(slots)))
    slot_to_id = {}
    nodes = []
    for idx, slot in enumerate(slots):
        slot_to_id[slot] = idx
        top = row_pos[slot[0]]
        left = col_pos[slot[1]]
        color = colors[idx]
        grid[top : top + NODE_SIZE, left : left + NODE_SIZE] = color
        nodes.append({"slot": slot, "top": top, "left": left, "color": color})

    path_specs = []
    for slot_a, slot_b in edges:
        node_a = nodes[slot_to_id[slot_a]]
        node_b = nodes[slot_to_id[slot_b]]
        if slot_a[0] == slot_b[0]:
            if node_a["left"] > node_b["left"]:
                node_a, node_b = node_b, node_a
            row = node_a["top"] + 1
            cells = [(row, c) for c in range(node_a["left"] + NODE_SIZE, node_b["left"])]
        else:
            if node_a["top"] > node_b["top"]:
                node_a, node_b = node_b, node_a
            col = node_a["left"] + 1
            cells = [(r, col) for r in range(node_a["top"] + NODE_SIZE, node_b["top"])]
        if len(cells) < 2:
            return None
        for r, c in cells:
            grid[r, c] = PATH
        path_specs.append({"node_a": node_a, "node_b": node_b, "cells": cells})
    return grid, nodes, path_specs


def _pick_marker_indices(length: int, count: int, rng: random.Random):
    if count == 0:
        return []
    for _ in range(60):
        indices = sorted(rng.sample(range(length), count))
        if all(b - a >= 2 for a, b in zip(indices, indices[1:])):
            return indices
    return None


def _paint_markers(base_grid: np.ndarray, nodes, path_specs, rng: random.Random):
    grid = base_grid.copy()
    all_colors = [node["color"] for node in nodes]
    weights = {0: 4, 1: 5, 2: 4, 3: 2}
    for spec in path_specs:
        length = len(spec["cells"])
        max_markers = min(3, (length + 1) // 2)
        choices = list(range(max_markers + 1))
        probs = [weights[k] for k in choices]
        marker_count = rng.choices(choices, weights=probs, k=1)[0]
        indices = _pick_marker_indices(length, marker_count, rng)
        if indices is None:
            return None
        allowed = [
            color
            for color in all_colors
            if color not in (spec["node_a"]["color"], spec["node_b"]["color"])
        ]
        if marker_count > 0 and not allowed:
            return None
        marker_colors = [rng.choice(allowed) for _ in indices]
        for idx, color in zip(indices, marker_colors):
            r, c = spec["cells"][idx]
            grid[r, c] = color
    return grid


def _apply_transform(grid: np.ndarray, variant: int):
    if variant == 0:
        return grid
    if variant == 1:
        return np.rot90(grid, 1)
    if variant == 2:
        return np.rot90(grid, 2)
    if variant == 3:
        return np.rot90(grid, 3)
    if variant == 4:
        return np.fliplr(grid)
    if variant == 5:
        return np.flipud(grid)
    if variant == 6:
        return np.transpose(grid)
    return np.fliplr(np.transpose(grid))


def _stats_for_generated(grid: np.ndarray):
    info = _analyze_grid(grid)
    recolored = 0
    source = 0
    conflict = 0
    stable_same_color = 0
    for node in info["nodes"]:
        node_votes = info["votes"].get(node["id"], [])
        if not node_votes:
            source += 1
            continue
        if len(set(node_votes)) == 1:
            if node_votes[0] == node["color"]:
                stable_same_color += 1
            else:
                recolored += 1
        else:
            conflict += 1

    marked_edges = 0
    unmarked_edges = 0
    multi_marker_edges = 0
    for path in info["paths"]:
        marker_count = len(path["marker_colors"])
        if marker_count == 0:
            unmarked_edges += 1
        else:
            marked_edges += 1
        if marker_count >= 2:
            multi_marker_edges += 1

    return {
        "recolored": recolored,
        "source": source,
        "conflict": conflict,
        "stable_same_color": stable_same_color,
        "marked_edges": marked_edges,
        "unmarked_edges": unmarked_edges,
        "multi_marker_edges": multi_marker_edges,
        "node_count": len(info["nodes"]),
        "path_count": len(info["paths"]),
    }


def _gen_attempt(rng: random.Random):
    layout = _build_layout(rng)
    if layout is None:
        return None
    base_grid, nodes, path_specs = layout
    mode = rng.choice(
        [
            {"source": True, "conflict": False, "multi": False, "unmarked": True},
            {"source": False, "conflict": True, "multi": True, "unmarked": False},
            {"source": True, "conflict": True, "multi": False, "unmarked": True},
            {"source": False, "conflict": False, "multi": True, "unmarked": True},
        ]
    )

    for _ in range(40):
        grid = _paint_markers(base_grid, nodes, path_specs, rng)
        if grid is None:
            continue
        grid = _apply_transform(grid, rng.randrange(8))
        try:
            stats = _stats_for_generated(grid)
        except AssertionError:
            continue
        if stats["recolored"] < 1:
            continue
        if stats["marked_edges"] < 1:
            continue
        if mode["source"] and stats["source"] < 1:
            continue
        if mode["conflict"] and stats["conflict"] < 1:
            continue
        if mode["multi"] and stats["multi_marker_edges"] < 1:
            continue
        if mode["unmarked"] and stats["unmarked_edges"] < 1:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
