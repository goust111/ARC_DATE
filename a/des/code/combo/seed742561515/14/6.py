from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
GREY = 5
NODE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
STEP = 4
GEN_ATTEMPTS = 400

Coord = tuple[int, int]
GraphKey = Coord | tuple[str, int]


def _canon_edge(a: Coord, b: Coord) -> tuple[Coord, Coord]:
    return tuple(sorted((a, b)))


def _move(v: Coord, axis: int, delta: int) -> Coord:
    if axis == 0:
        return (v[0] + delta, v[1])
    return (v[0], v[1] + delta)


def _ray(start: Coord, axis: int, delta: int, length: int) -> list[Coord]:
    pts = [start]
    cur = start
    for _ in range(length):
        cur = _move(cur, axis, delta)
        pts.append(cur)
    return pts


def _in_bounds(v: Coord, rows: int, cols: int) -> bool:
    r, c = v
    return 0 <= r < rows and 0 <= c < cols


def _free_len(start: Coord, axis: int, delta: int, used_vertices: set[Coord], rows: int, cols: int) -> int:
    cur = start
    length = 0
    while True:
        cur = _move(cur, axis, delta)
        if not _in_bounds(cur, rows, cols) or cur in used_vertices:
            return length
        length += 1


def _can_add_path(
    seq: list[Coord],
    used_vertices: set[Coord],
    used_edges: set[tuple[Coord, Coord]],
    rows: int,
    cols: int,
) -> bool:
    if len(seq) < 2 or seq[0] not in used_vertices:
        return False
    for pt in seq[1:]:
        if not _in_bounds(pt, rows, cols) or pt in used_vertices:
            return False
    for a, b in zip(seq, seq[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
        if _canon_edge(a, b) in used_edges:
            return False
    return True


def _add_path(
    seq: list[Coord],
    used_vertices: set[Coord],
    used_edges: set[tuple[Coord, Coord]],
) -> None:
    for pt in seq[1:]:
        used_vertices.add(pt)
    for a, b in zip(seq, seq[1:]):
        used_edges.add(_canon_edge(a, b))


def _degree_map(edges: set[tuple[Coord, Coord]]) -> dict[Coord, int]:
    degree: dict[Coord, int] = defaultdict(int)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    return degree


def _node_pixels(top_left: Coord) -> set[Coord]:
    r, c = top_left
    return {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}


def _edge_endpoint(pixel: Coord, direction: Coord, colored: bool) -> Coord:
    if not colored:
        return pixel
    dr, dc = direction
    r, c = pixel
    if dr == -1:
        return (r - 1, c)
    if dr == 1:
        return (r + 2, c)
    if dc == -1:
        return (r, c - 1)
    assert dc == 1
    return (r, c + 2)


def _draw_segment(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = a
    r1, c1 = b
    pts: list[Coord] = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            pts.append((r0, c))
        return pts

    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step, step):
        pts.append((r, c0))
    return pts


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    g = np.asarray(grid, dtype=int)
    rows, cols = g.shape
    seen = np.zeros((rows, cols), dtype=bool)
    components: list[list[Coord]] = []
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _detect_nodes(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    nodes: list[dict[str, object]] = []
    for color in sorted(set(map(int, g.ravel())) - {BLACK, GREY}):
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each node color must appear exactly twice"
        for coords in comps:
            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            assert len(coords) == 4, "each node must be a 2x2 block"
            r0, c0 = min(rows), min(cols)
            assert max(rows) == r0 + 1 and max(cols) == c0 + 1, "node bbox must be 2x2"
            cells = _node_pixels((r0, c0))
            assert set(coords) == cells, "node block must be solid"
            nodes.append({"color": color, "cells": cells})
    assert nodes, "expected at least one color pair"
    return nodes


def _build_graph(grid: np.ndarray, nodes: list[dict[str, object]]) -> dict[GraphKey, set[GraphKey]]:
    g = np.asarray(grid, dtype=int)
    rows, cols = g.shape
    graph: dict[GraphKey, set[GraphKey]] = defaultdict(set)
    grey_pixels = {(r, c) for r in range(rows) for c in range(cols) if int(g[r, c]) == GREY}
    for r, c in grey_pixels:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in grey_pixels:
                graph[(r, c)].add(nb)

    for idx, node in enumerate(nodes):
        key: GraphKey = ("node", idx)
        for r, c in node["cells"]:  # type: ignore[index]
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in grey_pixels:
                    graph[key].add(nb)
                    graph[nb].add(key)
    return graph


def _bfs_path(graph: dict[GraphKey, set[GraphKey]], start: GraphKey, goal: GraphKey) -> list[GraphKey]:
    q = deque([start])
    parent: dict[GraphKey, GraphKey | None] = {start: None}
    while q and goal not in parent:
        cur = q.popleft()
        for nb in graph.get(cur, set()):
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)
    assert goal in parent, "paired nodes must be connected"

    path: list[GraphKey] = []
    cur: GraphKey | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a tree-like grey line network, recolor the unique grey path between each pair of same-colored 2x2 nodes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == GREY), "expected a grey network"

    nodes = _detect_nodes(g)
    graph = _build_graph(g, nodes)
    out = g.copy()

    by_color: dict[int, list[int]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        by_color[int(node["color"])].append(idx)

    for color, ids in by_color.items():
        assert len(ids) == 2, "each color must have exactly two nodes"
        path = _bfs_path(graph, ("node", ids[0]), ("node", ids[1]))
        for vertex in path:
            if isinstance(vertex, tuple) and len(vertex) == 2 and isinstance(vertex[0], int):
                r, c = vertex
                if int(out[r, c]) == GREY:
                    out[r, c] = color

    return out


def _make_trunk(rng: random.Random, rows: int, cols: int, axis: int) -> list[Coord]:
    if axis == 1:
        trunk_row = rng.randint(1, rows - 2)
        return [(trunk_row, c) for c in range(1, cols - 1)]
    trunk_col = rng.randint(1, cols - 2)
    return [(r, trunk_col) for r in range(1, rows - 1)]


def _try_add_split_pair(
    rng: random.Random,
    root: Coord,
    axis: int,
    perp: int,
    side: int,
    color: int,
    rows: int,
    cols: int,
    used_vertices: set[Coord],
    used_edges: set[tuple[Coord, Coord]],
    node_colors: dict[Coord, int],
    pair_paths: list[tuple[int, list[Coord]]],
) -> bool:
    stem_cap = _free_len(root, perp, side, used_vertices, rows, cols)
    if stem_cap < 1:
        return False
    stem = _ray(root, perp, side, rng.randint(1, min(3, stem_cap)))
    if not _can_add_path(stem, used_vertices, used_edges, rows, cols):
        return False
    split = stem[-1]
    neg_cap = _free_len(split, axis, -1, used_vertices, rows, cols)
    pos_cap = _free_len(split, axis, 1, used_vertices, rows, cols)
    if neg_cap < 1 or pos_cap < 1:
        return False

    arm_neg = _ray(split, axis, -1, rng.randint(1, min(3, neg_cap)))
    arm_pos = _ray(split, axis, 1, rng.randint(1, min(3, pos_cap)))

    tmp_vertices = set(used_vertices)
    tmp_edges = set(used_edges)
    _add_path(stem, tmp_vertices, tmp_edges)
    if not _can_add_path(arm_neg, tmp_vertices, tmp_edges, rows, cols):
        return False
    _add_path(arm_neg, tmp_vertices, tmp_edges)
    if not _can_add_path(arm_pos, tmp_vertices, tmp_edges, rows, cols):
        return False

    _add_path(stem, used_vertices, used_edges)
    _add_path(arm_neg, used_vertices, used_edges)
    _add_path(arm_pos, used_vertices, used_edges)

    node_colors[arm_neg[-1]] = color
    node_colors[arm_pos[-1]] = color
    pair_paths.append((color, arm_neg[::-1] + arm_pos[1:]))

    tail_cap = _free_len(split, perp, side, used_vertices, rows, cols)
    if tail_cap >= 1 and rng.random() < 0.55:
        tail = _ray(split, perp, side, rng.randint(1, min(2, tail_cap)))
        if _can_add_path(tail, used_vertices, used_edges, rows, cols):
            _add_path(tail, used_vertices, used_edges)

    return True


def _try_add_junction_pair(
    rng: random.Random,
    root: Coord,
    axis: int,
    perp: int,
    side: int,
    color: int,
    rows: int,
    cols: int,
    used_vertices: set[Coord],
    used_edges: set[tuple[Coord, Coord]],
    node_colors: dict[Coord, int],
    pair_paths: list[tuple[int, list[Coord]]],
) -> bool:
    stem_cap = _free_len(root, perp, side, used_vertices, rows, cols)
    if stem_cap < 1:
        return False
    stem = _ray(root, perp, side, rng.randint(1, min(3, stem_cap)))
    if not _can_add_path(stem, used_vertices, used_edges, rows, cols):
        return False
    junction = stem[-1]

    arm_signs = [delta for delta in (-1, 1) if _free_len(junction, axis, delta, used_vertices, rows, cols) >= 1]
    if not arm_signs:
        return False
    arm_sign = rng.choice(arm_signs)
    arm_cap = _free_len(junction, axis, arm_sign, used_vertices, rows, cols)
    arm = _ray(junction, axis, arm_sign, rng.randint(1, min(4, arm_cap)))

    tmp_vertices = set(used_vertices)
    tmp_edges = set(used_edges)
    _add_path(stem, tmp_vertices, tmp_edges)
    if not _can_add_path(arm, tmp_vertices, tmp_edges, rows, cols):
        return False

    _add_path(stem, used_vertices, used_edges)
    _add_path(arm, used_vertices, used_edges)

    other_sign = -arm_sign
    other_cap = _free_len(junction, axis, other_sign, used_vertices, rows, cols)
    if other_cap >= 1 and rng.random() < 0.75:
        distractor = _ray(junction, axis, other_sign, rng.randint(1, min(3, other_cap)))
        if _can_add_path(distractor, used_vertices, used_edges, rows, cols):
            _add_path(distractor, used_vertices, used_edges)

    node_colors[junction] = color
    node_colors[arm[-1]] = color
    pair_paths.append((color, arm))
    return True


def _add_extra_branch(
    rng: random.Random,
    rows: int,
    cols: int,
    used_vertices: set[Coord],
    used_edges: set[tuple[Coord, Coord]],
    node_colors: dict[Coord, int],
) -> bool:
    degree = _degree_map(used_edges)
    attachments = [v for v in used_vertices if v not in node_colors and degree.get(v, 0) < 4]
    rng.shuffle(attachments)
    for attach in attachments:
        directions = [(0, -1), (0, 1), (1, -1), (1, 1)]
        rng.shuffle(directions)
        for axis, delta in directions:
            cap = _free_len(attach, axis, delta, used_vertices, rows, cols)
            if cap < 1:
                continue
            branch = _ray(attach, axis, delta, rng.randint(1, min(3, cap)))
            if _can_add_path(branch, used_vertices, used_edges, rows, cols):
                _add_path(branch, used_vertices, used_edges)
                return True
    return False


def _render_case(
    rng: random.Random,
    used_vertices: set[Coord],
    used_edges: set[tuple[Coord, Coord]],
    node_colors: dict[Coord, int],
    pair_paths: list[tuple[int, list[Coord]]],
) -> tuple[np.ndarray, np.ndarray]:
    base_pixels = {v: (v[0] * STEP, v[1] * STEP) for v in used_vertices}
    grey_pixels: set[Coord] = set()
    for v in used_vertices:
        if v not in node_colors:
            grey_pixels.add(base_pixels[v])

    for a, b in used_edges:
        dr = 0 if b[0] == a[0] else (1 if b[0] > a[0] else -1)
        dc = 0 if b[1] == a[1] else (1 if b[1] > a[1] else -1)
        start = _edge_endpoint(base_pixels[a], (dr, dc), a in node_colors)
        stop = _edge_endpoint(base_pixels[b], (-dr, -dc), b in node_colors)
        grey_pixels.update(_draw_segment(start, stop))

    all_pixels = set(grey_pixels)
    for vertex in node_colors:
        all_pixels.update(_node_pixels(base_pixels[vertex]))

    min_r = min(r for r, _c in all_pixels)
    max_r = max(r for r, _c in all_pixels)
    min_c = min(c for _r, c in all_pixels)
    max_c = max(c for _r, c in all_pixels)

    pad_top = rng.randint(1, 4)
    pad_bottom = rng.randint(1, 4)
    pad_left = rng.randint(1, 4)
    pad_right = rng.randint(1, 4)
    height = max_r - min_r + 1 + pad_top + pad_bottom
    width = max_c - min_c + 1 + pad_left + pad_right
    shift_r = pad_top - min_r
    shift_c = pad_left - min_c

    inp = np.full((height, width), BLACK, dtype=int)
    for r, c in grey_pixels:
        inp[r + shift_r, c + shift_c] = GREY
    for vertex, color in node_colors.items():
        br, bc = base_pixels[vertex]
        rr = br + shift_r
        cc = bc + shift_c
        inp[rr : rr + 2, cc : cc + 2] = color

    out = inp.copy()
    for color, path in pair_paths:
        for vertex in path:
            if vertex not in node_colors:
                br, bc = base_pixels[vertex]
                rr = br + shift_r
                cc = bc + shift_c
                if int(out[rr, cc]) == GREY:
                    out[rr, cc] = color
        for a, b in zip(path, path[1:]):
            dr = 0 if b[0] == a[0] else (1 if b[0] > a[0] else -1)
            dc = 0 if b[1] == a[1] else (1 if b[1] > a[1] else -1)
            start = _edge_endpoint(base_pixels[a], (dr, dc), a in node_colors)
            stop = _edge_endpoint(base_pixels[b], (-dr, -dc), b in node_colors)
            for r, c in _draw_segment(start, stop):
                rr = r + shift_r
                cc = c + shift_c
                if int(out[rr, cc]) == GREY:
                    out[rr, cc] = color

    return inp, out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    rows = rng.randint(6, 8)
    cols = rng.randint(7, 10)
    axis = rng.choice((0, 1))
    perp = 1 - axis

    trunk = _make_trunk(rng, rows, cols, axis)
    used_vertices: set[Coord] = set(trunk)
    used_edges: set[tuple[Coord, Coord]] = {_canon_edge(a, b) for a, b in zip(trunk, trunk[1:])}
    node_colors: dict[Coord, int] = {}
    pair_paths: list[tuple[int, list[Coord]]] = []

    target_pairs = rng.randint(1, min(4, len(NODE_COLORS), max(1, len(trunk) - 1)))
    color_pool = list(NODE_COLORS)
    rng.shuffle(color_pool)
    gadget_roots: set[Coord] = set()

    for color in color_pool[:target_pairs]:
        placed = False
        for _ in range(300):
            root = rng.choice(trunk)
            if root in gadget_roots:
                continue
            side = rng.choice((-1, 1))
            builders = (_try_add_split_pair, _try_add_junction_pair)
            builder = rng.choice(builders)
            if builder(
                rng,
                root,
                axis,
                perp,
                side,
                color,
                rows,
                cols,
                used_vertices,
                used_edges,
                node_colors,
                pair_paths,
            ):
                gadget_roots.add(root)
                placed = True
                break
        if not placed:
            raise AssertionError("failed to place requested color pairs")

    extra_targets = rng.randint(0, 3)
    for _ in range(extra_targets):
        _add_extra_branch(rng, rows, cols, used_vertices, used_edges, node_colors)

    inp, out = _render_case(rng, used_vertices, used_edges, node_colors, pair_paths)
    assert np.any(inp == GREY), "generator must leave some grey pixels"
    assert np.any(inp != out), "output should recolor at least one path"
    assert np.any(out == GREY), "some grey distractor pixels should remain"
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        try:
            inp, expected = _generate_case(rng)
        except AssertionError:
            continue
        try:
            got = main(inp)
        except AssertionError:
            continue
        if np.array_equal(got, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
