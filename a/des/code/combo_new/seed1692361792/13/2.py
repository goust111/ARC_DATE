import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SOURCE_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 300
PLACE_ATTEMPTS = 200


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(line):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(line) - 1))
    return runs


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            q = deque([(r, c)])
            labels[r, c] = comp_id
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def _extract_horizontal_strips(mask: np.ndarray) -> list[dict]:
    row_runs = [_find_runs(mask[r]) for r in range(mask.shape[0])]
    run_sets = [set(runs) for runs in row_runs]
    strips: list[dict] = []
    for r, runs in enumerate(row_runs):
        for c0, c1 in runs:
            if c1 - c0 + 1 < 3:
                continue
            if r > 0 and (c0, c1) in run_sets[r - 1]:
                continue
            r1 = r
            if r + 1 < mask.shape[0] and (c0, c1) in run_sets[r + 1]:
                r1 = r + 1
            strips.append(
                {
                    "id": len(strips),
                    "ori": "h",
                    "r0": int(r),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "width": int(r1 - r + 1),
                }
            )
    return strips


def _extract_vertical_strips(mask: np.ndarray) -> list[dict]:
    col_runs = [_find_runs(mask[:, c]) for c in range(mask.shape[1])]
    run_sets = [set(runs) for runs in col_runs]
    strips: list[dict] = []
    for c, runs in enumerate(col_runs):
        for r0, r1 in runs:
            if r1 - r0 + 1 < 3:
                continue
            if c > 0 and (r0, r1) in run_sets[c - 1]:
                continue
            c1 = c
            if c + 1 < mask.shape[1] and (r0, r1) in run_sets[c + 1]:
                c1 = c + 1
            strips.append(
                {
                    "id": len(strips),
                    "ori": "v",
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c),
                    "c1": int(c1),
                    "width": int(c1 - c + 1),
                }
            )
    return strips


def _build_graph(mask: np.ndarray) -> tuple[dict[int, dict], dict[int, dict], dict[int, list[tuple[int, int, tuple[int, int]]]]]:
    h_strips = _extract_horizontal_strips(mask)
    v_strips = _extract_vertical_strips(mask)

    nodes: dict[int, dict] = {}
    node_by_bbox: dict[tuple[int, int, int, int], int] = {}
    attached_h: dict[int, list[tuple[int, int, int]]] = {strip["id"]: [] for strip in h_strips}
    attached_v: dict[int, list[tuple[int, int, int]]] = {strip["id"]: [] for strip in v_strips}

    def add_node(r0: int, r1: int, c0: int, c1: int, kind: str) -> int:
        key = (int(r0), int(r1), int(c0), int(c1))
        node_id = node_by_bbox.get(key)
        if node_id is None:
            node_id = len(nodes)
            cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            nodes[node_id] = {"bbox": key, "cells": cells, "kind": kind}
            node_by_bbox[key] = node_id
        return int(node_id)

    for h_strip in h_strips:
        for v_strip in v_strips:
            r0 = max(h_strip["r0"], v_strip["r0"])
            r1 = min(h_strip["r1"], v_strip["r1"])
            c0 = max(h_strip["c0"], v_strip["c0"])
            c1 = min(h_strip["c1"], v_strip["c1"])
            if r0 <= r1 and c0 <= c1:
                node_id = add_node(r0, r1, c0, c1, "junction")
                attached_h[h_strip["id"]].append((node_id, c0, c1))
                attached_v[v_strip["id"]].append((node_id, r0, r1))

    for h_strip in h_strips:
        entries = attached_h[h_strip["id"]]
        if not any(span0 <= h_strip["c0"] <= span1 for _, span0, span1 in entries):
            node_id = add_node(h_strip["r0"], h_strip["r1"], h_strip["c0"], h_strip["c0"], "terminal")
            entries.append((node_id, h_strip["c0"], h_strip["c0"]))
        if not any(span0 <= h_strip["c1"] <= span1 for _, span0, span1 in entries):
            node_id = add_node(h_strip["r0"], h_strip["r1"], h_strip["c1"], h_strip["c1"], "terminal")
            entries.append((node_id, h_strip["c1"], h_strip["c1"]))

    for v_strip in v_strips:
        entries = attached_v[v_strip["id"]]
        if not any(span0 <= v_strip["r0"] <= span1 for _, span0, span1 in entries):
            node_id = add_node(v_strip["r0"], v_strip["r0"], v_strip["c0"], v_strip["c1"], "terminal")
            entries.append((node_id, v_strip["r0"], v_strip["r0"]))
        if not any(span0 <= v_strip["r1"] <= span1 for _, span0, span1 in entries):
            node_id = add_node(v_strip["r1"], v_strip["r1"], v_strip["c0"], v_strip["c1"], "terminal")
            entries.append((node_id, v_strip["r1"], v_strip["r1"]))

    edges: dict[int, dict] = {}
    adj: dict[int, list[tuple[int, int, tuple[int, int]]]] = {node_id: [] for node_id in nodes}

    def add_edge(
        node_a: int,
        node_b: int,
        cells: list[tuple[int, int]],
        width: int,
        dir_ab: tuple[int, int],
        dir_ba: tuple[int, int],
    ) -> None:
        if not cells:
            return
        edge_id = len(edges)
        edges[edge_id] = {
            "cells": cells,
            "width": int(width),
            "a": int(node_a),
            "b": int(node_b),
        }
        adj[node_a].append((edge_id, node_b, dir_ab))
        adj[node_b].append((edge_id, node_a, dir_ba))

    for h_strip in h_strips:
        entries = sorted(attached_h[h_strip["id"]], key=lambda x: (x[1], x[2], x[0]))
        for (node_a, _span0, span1), (node_b, next0, _next1) in zip(entries, entries[1:]):
            c0 = span1 + 1
            c1 = next0 - 1
            cells = [(r, c) for r in range(h_strip["r0"], h_strip["r1"] + 1) for c in range(c0, c1 + 1)]
            add_edge(node_a, node_b, cells, h_strip["width"], (0, 1), (0, -1))

    for v_strip in v_strips:
        entries = sorted(attached_v[v_strip["id"]], key=lambda x: (x[1], x[2], x[0]))
        for (node_a, _span0, span1), (node_b, next0, _next1) in zip(entries, entries[1:]):
            r0 = span1 + 1
            r1 = next0 - 1
            cells = [(r, c) for r in range(r0, r1 + 1) for c in range(v_strip["c0"], v_strip["c1"] + 1)]
            add_edge(node_a, node_b, cells, v_strip["width"], (1, 0), (-1, 0))

    return nodes, edges, adj


def _choose_edge(
    options: list[tuple[int, int, tuple[int, int]]],
    edges: dict[int, dict],
    current_dir: tuple[int, int] | None,
) -> tuple[int, int, tuple[int, int]]:
    widest = max(edges[edge_id]["width"] for edge_id, _node, _direction in options)
    best = [opt for opt in options if edges[opt[0]]["width"] == widest]
    if current_dir is not None:
        straight = [opt for opt in best if opt[2] == current_dir]
        if straight:
            best = straight
    best.sort(key=lambda opt: (opt[2][0], opt[2][1], opt[1], opt[0]))
    return best[0]


def _paint_signal(
    out: np.ndarray,
    original: np.ndarray,
    nodes: dict[int, dict],
    edges: dict[int, dict],
    adj: dict[int, list[tuple[int, int, tuple[int, int]]]],
    start_node: int,
    color: int,
    blocked_edges: set[int],
) -> set[int]:
    current = int(start_node)
    current_dir: tuple[int, int] | None = None
    visited_edges: set[int] = set()

    while True:
        for r, c in nodes[current]["cells"]:
            if original[r, c] == BG or original[r, c] == color:
                out[r, c] = int(color)

        options = [
            (edge_id, next_node, direction)
            for edge_id, next_node, direction in adj.get(current, [])
            if edge_id not in visited_edges and edge_id not in blocked_edges
        ]
        if not options:
            break

        edge_id, next_node, direction = _choose_edge(options, edges, current_dir)
        visited_edges.add(int(edge_id))
        for r, c in edges[edge_id]["cells"]:
            if original[r, c] == BG or original[r, c] == color:
                out[r, c] = int(color)
        current = int(next_node)
        current_dir = direction

    return visited_edges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each colored source through the pipe interior and color the selected route.

    Input structure:
    - Black is the background outside the pipes and also the empty interior of the pipes.
    - Grey pixels are pipe walls around horizontal and vertical channels whose visible width is 1 or 2.
    - Colored single pixels sit at dead ends inside the pipe interior and act as signal sources.

    Rule:
    - Treat grey as walls and find non-grey pipe interiors that are enclosed away from the border.
    - For each source, follow the interior graph from its dead end; at every node choose the exit with larger channel width.
    - If several exits have the same width, continue straight when possible; never reuse an already colored channel.
    - Recolor every black interior cell on the traversed route to the source color and leave walls unchanged.

    Key insight:
    The task is a graph walk on the enclosed black pipe interiors, where junction decisions depend on the visible corridor thickness and only secondarily on straight-line continuation.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _most_common_color(g)
    non_bg = [int(c) for c in np.unique(g) if int(c) != bg]
    if not non_bg:
        return g.copy()
    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    source_colors = {c for c in non_bg if c != wall}

    passable = g != wall
    labels, n_comp = _label_mask(passable)
    out = g.copy()

    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labels == comp_id)
        if len(coords) == 0:
            continue
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in coords):
            continue

        component_colors = {int(g[r, c]) for r, c in coords if int(g[r, c]) in source_colors}
        if not component_colors:
            continue

        component_mask = labels == comp_id
        nodes, edges, adj = _build_graph(component_mask)
        if not nodes:
            continue

        source_nodes: list[tuple[int, int]] = []
        for node_id, node in nodes.items():
            node_colors = {int(g[r, c]) for r, c in node["cells"] if int(g[r, c]) in source_colors}
            if len(node_colors) == 1:
                source_nodes.append((node_id, next(iter(node_colors))))

        blocked_edges: set[int] = set()
        for node_id, color in sorted(source_nodes, key=lambda item: (item[1], item[0])):
            blocked_edges |= _paint_signal(out, g, nodes, edges, adj, node_id, color, blocked_edges)

    return out


def _draw_h(mask: np.ndarray, row: int, c0: int, c1: int, width: int) -> None:
    mask[row : row + width, c0 : c1 + 1] = True


def _draw_v(mask: np.ndarray, col: int, r0: int, r1: int, width: int) -> None:
    mask[r0 : r1 + 1, col : col + width] = True


def _rotate_coord(coord: tuple[int, int], shape: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    for _ in range(k % 4):
        r, c = w - 1 - c, r
        h, w = w, h
    return int(r), int(c)


def _component_array(mask: np.ndarray, source: tuple[int, int], color: int) -> np.ndarray:
    h, w = mask.shape
    comp = np.full((h + 2, w + 2), BG, dtype=int)
    interior = np.zeros_like(comp, dtype=bool)
    interior[1 : h + 1, 1 : w + 1] = mask

    wall_mask = np.zeros_like(comp, dtype=bool)
    for r, c in np.argwhere(interior):
        for dr, dc in DIRS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < comp.shape[0] and 0 <= nc < comp.shape[1] and not interior[nr, nc]:
                wall_mask[nr, nc] = True

    comp[wall_mask] = WALL
    comp[interior] = BG
    sr, sc = source
    comp[sr + 1, sc + 1] = int(color)
    return comp


def _template_straight_tie(rng: random.Random, color: int) -> np.ndarray:
    width = rng.choice([1, 2])
    left_len = rng.randint(3, 5)
    right_len = rng.randint(3, 5)
    up_len = rng.randint(2, 4)
    down_len = rng.randint(2, 4)
    h = up_len + width + down_len
    w = left_len + width + right_len
    mask = np.zeros((h, w), dtype=bool)
    _draw_h(mask, up_len, 0, w - 1, width)
    _draw_v(mask, left_len, 0, h - 1, width)
    source = (up_len + rng.randrange(width), 0)
    comp = _component_array(mask, source, color)
    return np.rot90(comp, rng.randrange(4)).copy()


def _template_turn_wider(rng: random.Random, color: int) -> np.ndarray:
    left_len = rng.randint(3, 5)
    right_len = rng.randint(3, 5)
    up_len = rng.randint(3, 5)
    h = up_len + 1
    w = left_len + 2 + right_len
    mask = np.zeros((h, w), dtype=bool)
    _draw_h(mask, up_len, 0, w - 1, 1)
    _draw_v(mask, left_len, 0, h - 1, 2)
    source = (up_len, 0)
    comp = _component_array(mask, source, color)
    return np.rot90(comp, rng.randrange(4)).copy()


def _template_straight_wider(rng: random.Random, color: int) -> np.ndarray:
    left_len = rng.randint(3, 5)
    right_len = rng.randint(3, 5)
    up_len = rng.randint(2, 4)
    h = up_len + 2
    w = left_len + 1 + right_len
    mask = np.zeros((h, w), dtype=bool)
    _draw_h(mask, up_len, 0, w - 1, 2)
    _draw_v(mask, left_len, 0, up_len + 1, 1)
    source = (up_len + rng.randrange(2), 0)
    comp = _component_array(mask, source, color)
    return np.rot90(comp, rng.randrange(4)).copy()


def _build_component(kind: str, rng: random.Random, color: int) -> np.ndarray:
    if kind == "straight_tie":
        return _template_straight_tie(rng, color)
    if kind == "turn_wider":
        return _template_turn_wider(rng, color)
    if kind == "straight_wider":
        return _template_straight_wider(rng, color)
    raise ValueError(f"unknown component kind: {kind}")


def _place_components(grid: np.ndarray, components: list[np.ndarray], rng: random.Random) -> bool:
    h, w = grid.shape
    used = np.zeros((h, w), dtype=bool)
    for comp in components:
        ch, cw = comp.shape
        if ch > h or cw > w:
            return False
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = rng.randint(0, h - ch)
            left = rng.randint(0, w - cw)
            r0 = max(0, top - 1)
            r1 = min(h, top + ch + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + cw + 1)
            if used[r0:r1, c0:c1].any():
                continue
            sub = grid[top : top + ch, left : left + cw]
            nz = comp != BG
            sub[nz] = comp[nz]
            used[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _component_kinds(rng: random.Random, count: int) -> list[str]:
    kinds = ["straight_tie", "turn_wider"]
    extras = ["straight_wider", "straight_tie", "turn_wider"]
    while len(kinds) < count:
        kinds.append(rng.choice(extras))
    rng.shuffle(kinds)
    return kinds


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    comp_count = rng.randint(2, 4)
    colors = rng.sample(SOURCE_POOL, comp_count)
    kinds = _component_kinds(rng, comp_count)
    components = [_build_component(kind, rng, color) for kind, color in zip(kinds, colors)]

    grid = np.full((h, w), BG, dtype=int)
    if not _place_components(grid, components, rng):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid pipe-network input after {GEN_ATTEMPTS} attempts")
