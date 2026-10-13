from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


BG = Color.BLACK
TERMINAL_COLOR = Color.YELLOW
PATH_NODE_COLOR = Color.GREEN
PATH_EDGE_COLOR = Color.GREY
NODE_SIZE = 2
NODE_COLORS = [Color.BLUE, Color.RED, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

MIN_GROUPS = 3
MAX_GROUPS = 5
MIN_GROUP_GAP = 4
MAX_GROUP_GAP = 5
MIN_MARGIN = 1
MAX_MARGIN = 3
MIN_NODES = 6
MAX_NODES = 8
MAX_GEN_ATTEMPTS = 500


def _detect_nodes(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    nodes = []
    for comp in comps:
        box = comp["bbox"]
        if int(comp["area"]) != 4 or bbox_h(box) != 2 or bbox_w(box) != 2:
            continue
        r0, c0, r1, c1 = box
        nodes.append(
            {
                "id": len(nodes),
                "color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    nodes.sort(key=lambda node: node["bbox"])
    for idx, node in enumerate(nodes):
        node["id"] = int(idx)
    return nodes


def _occupancy(nodes: list[dict], shape: tuple[int, int]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for node in nodes:
        r0, c0, r1, c1 = node["bbox"]
        occ[r0 : r1 + 1, c0 : c1 + 1] = int(node["id"])
    return occ


def _trace_hits(
    occ: np.ndarray, start_r: int, start_c: int, dr: int, dc: int, source_id: int
) -> list[int]:
    hits = []
    last = None
    r, c = int(start_r), int(start_c)
    h, w = occ.shape
    while 0 <= r < h and 0 <= c < w:
        nid = int(occ[r, c])
        if nid != -1 and nid != int(source_id) and nid != last:
            hits.append(nid)
            last = nid
        r += int(dr)
        c += int(dc)
    return hits


def _build_graph(nodes: list[dict], shape: tuple[int, int]) -> tuple[dict[int, set[int]], dict[int, list[list[int]]]]:
    occ = _occupancy(nodes, shape)
    adj = {int(node["id"]): set() for node in nodes}
    rays: dict[int, list[list[int]]] = {int(node["id"]): [] for node in nodes}
    for node in nodes:
        nid = int(node["id"])
        r0, c0, r1, c1 = node["bbox"]
        ray_specs = [
            (r0, c0 - 1, 0, -1),
            (r0, c1 + 1, 0, 1),
            (r0 - 1, c0, -1, 0),
            (r1 + 1, c0, 1, 0),
        ]
        for start_r, start_c, dr, dc in ray_specs:
            hits = _trace_hits(occ, start_r, start_c, dr, dc, nid)
            rays[nid].append(hits)
            if hits:
                first = int(hits[0])
                adj[nid].add(first)
                adj[first].add(nid)
    return adj, rays


def _segment_between(a: dict, b: dict) -> list[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ac1 < bc0:
        if br0 <= ar0 <= br1:
            return [(int(ar0), c) for c in range(int(ac1 + 1), int(bc0))]
        if ar0 <= br0 <= ar1:
            return [(int(br0), c) for c in range(int(ac1 + 1), int(bc0))]
    if bc1 < ac0:
        if br0 <= ar0 <= br1:
            return [(int(ar0), c) for c in range(int(bc1 + 1), int(ac0))]
        if ar0 <= br0 <= ar1:
            return [(int(br0), c) for c in range(int(bc1 + 1), int(ac0))]
    if ar1 < br0:
        if bc0 <= ac0 <= bc1:
            return [(r, int(ac0)) for r in range(int(ar1 + 1), int(br0))]
        if ac0 <= bc0 <= ac1:
            return [(r, int(bc0)) for r in range(int(ar1 + 1), int(br0))]
    if br1 < ar0:
        if bc0 <= ac0 <= bc1:
            return [(r, int(ac0)) for r in range(int(br1 + 1), int(ar0))]
        if ac0 <= bc0 <= ac1:
            return [(r, int(bc0)) for r in range(int(br1 + 1), int(ar0))]
    raise AssertionError("path edge must be horizontal or vertical line of sight")


def _shortest_path(adj: dict[int, set[int]], start: int, goal: int) -> tuple[list[int], int, int]:
    q = deque([int(start)])
    dist = {int(start): 0}
    ways = {int(start): 1}
    parent = {int(start): None}

    while q:
        u = int(q.popleft())
        for v in sorted(adj[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                ways[v] = ways[u]
                parent[v] = u
                q.append(v)
            elif dist[v] == dist[u] + 1:
                ways[v] += ways[u]

    assert int(goal) in dist, "terminals must be connected"
    path = []
    cur = int(goal)
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, int(dist[goal]), int(ways[goal])


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    nodes = _detect_nodes(g)
    assert len(nodes) >= 2, "expected multiple 2x2 node blocks"
    adj, rays = _build_graph(nodes, g.shape)
    return {
        "grid": g,
        "nodes": nodes,
        "adj": adj,
        "rays": rays,
    }


def _analyze_with_terminals(grid: np.ndarray) -> dict:
    info = _analyze(grid)
    terminals = [int(node["id"]) for node in info["nodes"] if int(node["color"]) == TERMINAL_COLOR]
    assert len(terminals) == 2, "expected exactly two terminal nodes"
    path, distance, ways = _shortest_path(info["adj"], terminals[0], terminals[1])
    info.update(
        {
            "terminals": terminals,
            "path": path,
            "distance": distance,
            "ways": ways,
        }
    )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a hidden line-of-sight graph between 2x2 nodes and recolor the unique shortest terminal path.

    Input structure:
    - The grid is mostly black background.
    - Foreground consists of 6-8 solid 2x2 node squares, with exactly two yellow terminal nodes and the rest all different non-black, non-grey colors.
    - No edges are pre-drawn: connectivity must be inferred from unobstructed horizontal or vertical line of sight between node bounding boxes.

    Rule:
    - For each node, cast four 1-cell-wide rays from its top row and left column: left, right, up, and down.
    - A ray creates an undirected graph edge to the first other 2x2 node whose bounding box it hits; any nearer node blocks farther nodes on that ray.
    - Find the unique shortest path between the two yellow terminals in this inferred graph.
    - Recolor every non-terminal node on that path to green and recolor every black cell strictly between consecutive path nodes along their connecting sightline to grey.

    Key insight:
    - The graph is implicit in visibility, not drawn explicitly: the node layout and blockers determine which edges exist, and only then can the shortest path be identified.
    """
    info = _analyze_with_terminals(np.asarray(grid, dtype=int))
    nodes = {int(node["id"]): node for node in info["nodes"]}
    path = list(info["path"])

    out = np.asarray(grid, dtype=int).copy()
    terminal_set = set(map(int, info["terminals"]))

    for nid in path[1:-1]:
        r0, c0, r1, c1 = nodes[int(nid)]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = PATH_NODE_COLOR

    for a_id, b_id in zip(path, path[1:], strict=False):
        for r, c in _segment_between(nodes[int(a_id)], nodes[int(b_id)]):
            out[int(r), int(c)] = PATH_EDGE_COLOR

    for nid in terminal_set:
        r0, c0, r1, c1 = nodes[int(nid)]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = TERMINAL_COLOR

    return out


def _sample_axis(group_count: int) -> tuple[list[int], int]:
    while True:
        pos = [random.randint(MIN_MARGIN, MAX_MARGIN)]
        for _ in range(group_count - 1):
            pos.append(pos[-1] + random.randint(MIN_GROUP_GAP, MAX_GROUP_GAP))
        size = pos[-1] + NODE_SIZE + random.randint(MIN_MARGIN, MAX_MARGIN)
        if size <= 30:
            return pos, int(size)


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _sample_cells(rows: int, cols: int, count: int) -> set[tuple[int, int]]:
    start = (random.randrange(rows), random.randrange(cols))
    selected = {start}
    while len(selected) < int(count):
        frontier = sorted({nbr for cell in selected for nbr in _neighbors(cell, rows, cols) if nbr not in selected})
        weights = []
        for r, c in frontier:
            same_row = sum(1 for rr, cc in selected if rr == r)
            same_col = sum(1 for rr, cc in selected if cc == c)
            touching = sum(1 for nbr in _neighbors((r, c), rows, cols) if nbr in selected)
            weight = 1.0 + 1.3 * same_row + 1.3 * same_col + 0.8 * touching
            weights.append(weight)
        selected.add(random.choices(frontier, weights=weights, k=1)[0])
    return selected


def _rows_cols(selected: set[tuple[int, int]]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in sorted(selected):
        by_row.setdefault(int(r), []).append(int(c))
        by_col.setdefault(int(c), []).append(int(r))
    for values in by_row.values():
        values.sort()
    for values in by_col.values():
        values.sort()
    return by_row, by_col


def _pick_offset_plan(
    selected: set[tuple[int, int]], rows: int, cols: int
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]] | None:
    by_row, by_col = _rows_cols(selected)
    blocked_choices = [("row", key) for key, vals in by_row.items() if len(vals) >= 3]
    blocked_choices += [("col", key) for key, vals in by_col.items() if len(vals) >= 3]
    if not blocked_choices:
        return None

    random.shuffle(blocked_choices)
    for blocked_kind, blocked_key in blocked_choices:
        row_offsets = {cell: random.randint(0, 1) for cell in selected}
        col_offsets = {cell: random.randint(0, 1) for cell in selected}

        if blocked_kind == "row":
            forced = random.randint(0, 1)
            for c in by_row[int(blocked_key)]:
                row_offsets[(int(blocked_key), int(c))] = forced
        else:
            forced = random.randint(0, 1)
            for r in by_col[int(blocked_key)]:
                col_offsets[(int(r), int(blocked_key))] = forced

        candidate_pairs: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
        if blocked_kind == "row":
            for c, rows_here in by_col.items():
                for r0, r1 in zip(rows_here, rows_here[1:], strict=False):
                    candidate_pairs.append(("vertical", (int(r0), int(c)), (int(r1), int(c))))
        else:
            for r, cols_here in by_row.items():
                for c0, c1 in zip(cols_here, cols_here[1:], strict=False):
                    candidate_pairs.append(("horizontal", (int(r), int(c0)), (int(r), int(c1))))

        random.shuffle(candidate_pairs)
        for orient, a, b in candidate_pairs:
            if orient == "horizontal":
                if blocked_kind == "row" and a[0] == int(blocked_key):
                    continue
                row_offsets[a] = 0
                row_offsets[b] = 1
            else:
                if blocked_kind == "col" and a[1] == int(blocked_key):
                    continue
                col_offsets[a] = 0
                col_offsets[b] = 1
            return row_offsets, col_offsets
    return None


def _render_grid(
    row_bases: list[int],
    col_bases: list[int],
    selected: set[tuple[int, int]],
    row_offsets: dict[tuple[int, int], int],
    col_offsets: dict[tuple[int, int], int],
    node_colors: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for cell in sorted(selected):
        rr = int(row_bases[cell[0]] + row_offsets[cell])
        cc = int(col_bases[cell[1]] + col_offsets[cell])
        grid[rr : rr + NODE_SIZE, cc : cc + NODE_SIZE] = int(node_colors[cell])
    return grid


def _choose_terminals(adj: dict[int, set[int]]) -> tuple[list[int], list[int]] | None:
    node_ids = sorted(adj)
    candidates = []
    for i, start in enumerate(node_ids):
        for goal in node_ids[i + 1 :]:
            if goal in adj[start]:
                continue
            path, distance, ways = _shortest_path(adj, start, goal)
            if distance < 2 or ways != 1:
                continue
            if len(path) >= len(node_ids):
                continue
            candidates.append((path, distance, start, goal))
    if not candidates:
        return None
    random.shuffle(candidates)
    candidates.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    path, _distance, start, goal = candidates[0]
    return [int(start), int(goal)], [int(x) for x in path]


def _blocked_ray_exists(info: dict) -> bool:
    return any(len(hits) >= 2 for hit_lists in info["rays"].values() for hits in hit_lists)


def _misaligned_edge_exists(info: dict) -> bool:
    nodes = {int(node["id"]): node for node in info["nodes"]}
    for a, nbrs in info["adj"].items():
        for b in nbrs:
            if a >= b:
                continue
            ar0, ac0, _ar1, _ac1 = nodes[a]["bbox"]
            br0, bc0, _br1, _bc1 = nodes[b]["bbox"]
            if abs(int(ar0) - int(br0)) == 1 or abs(int(ac0) - int(bc0)) == 1:
                return True
    return False


def _long_edge_exists(info: dict) -> bool:
    nodes = {int(node["id"]): node for node in info["nodes"]}
    for a, nbrs in info["adj"].items():
        for b in nbrs:
            if a >= b:
                continue
            if len(_segment_between(nodes[a], nodes[b])) >= 2:
                return True
    return False


def _validate_semantics(grid: np.ndarray) -> dict | None:
    try:
        info = _analyze_with_terminals(grid)
    except Exception:
        return None

    nodes = info["nodes"]
    if not (MIN_NODES <= len(nodes) <= MAX_NODES):
        return None
    colors = [int(node["color"]) for node in nodes]
    if colors.count(TERMINAL_COLOR) != 2:
        return None
    if PATH_NODE_COLOR in colors or PATH_EDGE_COLOR in colors:
        return None

    nonterminal_colors = [color for color in colors if color != TERMINAL_COLOR]
    if len(nonterminal_colors) != len(set(nonterminal_colors)):
        return None
    if any(color not in NODE_COLORS for color in nonterminal_colors):
        return None
    if not all(info["adj"][int(node["id"])] for node in nodes):
        return None
    if info["distance"] < 2 or info["ways"] != 1:
        return None
    if not _blocked_ray_exists(info):
        return None
    if not _misaligned_edge_exists(info):
        return None
    if not _long_edge_exists(info):
        return None
    if len(info["path"]) >= len(nodes):
        return None

    expected = np.array(grid, copy=True)
    node_lookup = {int(node["id"]): node for node in nodes}
    for nid in info["path"][1:-1]:
        r0, c0, r1, c1 = node_lookup[int(nid)]["bbox"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = PATH_NODE_COLOR
    for a_id, b_id in zip(info["path"], info["path"][1:], strict=False):
        segment = _segment_between(node_lookup[int(a_id)], node_lookup[int(b_id)])
        if not segment:
            return None
        if any(int(grid[r, c]) != BG for r, c in segment):
            return None
        for r, c in segment:
            expected[int(r), int(c)] = PATH_EDGE_COLOR

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return info


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        group_rows = random.randint(MIN_GROUPS, MAX_GROUPS)
        group_cols = random.randint(MIN_GROUPS, MAX_GROUPS)
        node_count = random.randint(MIN_NODES, MAX_NODES)
        if group_rows * group_cols < node_count:
            continue

        row_bases, height = _sample_axis(group_rows)
        col_bases, width = _sample_axis(group_cols)
        selected = _sample_cells(group_rows, group_cols, node_count)

        offset_plan = _pick_offset_plan(selected, group_rows, group_cols)
        if offset_plan is None:
            continue
        row_offsets, col_offsets = offset_plan

        placeholder_colors = {cell: NODE_COLORS[idx % len(NODE_COLORS)] for idx, cell in enumerate(sorted(selected))}
        placeholder = _render_grid(
            row_bases,
            col_bases,
            selected,
            row_offsets,
            col_offsets,
            placeholder_colors,
            (height, width),
        )

        layout_info = _analyze(placeholder)
        if not all(layout_info["adj"][int(node["id"])] for node in layout_info["nodes"]):
            continue
        if not _blocked_ray_exists(layout_info):
            continue
        if not _misaligned_edge_exists(layout_info):
            continue
        if not _long_edge_exists(layout_info):
            continue

        terminals, path = _choose_terminals(layout_info["adj"]) or (None, None)
        if terminals is None or path is None:
            continue
        if len(path) < 3:
            continue

        cell_order = sorted(selected)
        node_lookup = {int(node["id"]): cell_order[idx] for idx, node in enumerate(layout_info["nodes"])}
        colors = random.sample(NODE_COLORS, len(selected) - 2)
        node_colors: dict[tuple[int, int], int] = {}
        color_idx = 0
        for node in layout_info["nodes"]:
            cell = node_lookup[int(node["id"])]
            if int(node["id"]) in terminals:
                node_colors[cell] = TERMINAL_COLOR
            else:
                node_colors[cell] = int(colors[color_idx])
                color_idx += 1

        grid = _render_grid(
            row_bases,
            col_bases,
            selected,
            row_offsets,
            col_offsets,
            node_colors,
            (height, width),
        )

        final_info = _validate_semantics(grid)
        if final_info is None:
            continue
        if sorted(final_info["terminals"]) != sorted(terminals):
            continue
        return grid

    raise RuntimeError("failed to generate a valid implicit-graph ARC input")
