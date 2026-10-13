from __future__ import annotations

import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
LINE_COLORS = (RED, BLUE, GREEN)
PATTERN_COLORS = (4, 6, 7, 8, 9)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _make_frame_mask() -> np.ndarray:
    frame = np.zeros((5, 5), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return frame


FRAME_MASK = _make_frame_mask()
SIDE_SPECS = (
    ("up", (0, 2), (-1, 2), (-1, 0)),
    ("down", (4, 2), (5, 2), (1, 0)),
    ("left", (2, 0), (2, -1), (0, -1)),
    ("right", (2, 4), (2, 5), (0, 1)),
)


def _detect_nodes(grid: np.ndarray) -> list[dict]:
    nodes: list[dict] = []
    for coords in _find_components(grid == GREY):
        if len(coords) != int(FRAME_MASK.sum()):
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.array_equal(patch == GREY, FRAME_MASK):
            continue
        nodes.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "interior": patch[1:4, 1:4].copy(),
            }
        )
    nodes.sort(key=lambda node: (node["bbox"][0], node["bbox"][1]))
    assert nodes, "expected at least one framed node"
    return nodes


def _follow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    delta: tuple[int, int],
    border_to_node: dict[tuple[int, int], int],
) -> tuple[int, int]:
    h, w = grid.shape
    r, c = start
    color = int(grid[r, c])
    assert color in LINE_COLORS, "line must start on a connector color"
    dr, dc = delta
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == color:
        r += dr
        c += dc
    assert 0 <= r < h and 0 <= c < w, "connector line must terminate at another node"
    target = border_to_node.get((int(r), int(c)))
    assert target is not None, "connector line must end on a node border center"
    return int(target), color


def _extract_graph(grid: np.ndarray, nodes: list[dict]) -> tuple[int, dict[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    border_to_node: dict[tuple[int, int], int] = {}
    source_nodes: list[int] = []
    for idx, node in enumerate(nodes):
        r0, c0, r1, c1 = node["bbox"]
        interior = node["interior"]
        if np.any((interior != BLACK) & (interior != GREY)):
            source_nodes.append(idx)
        else:
            assert np.all(interior == BLACK), "non-source nodes must be empty"
        for _name, border_off, _start_off, _delta in SIDE_SPECS:
            br = r0 + border_off[0]
            bc = c0 + border_off[1]
            border_to_node[(int(br), int(bc))] = idx

    assert len(source_nodes) == 1, "expected exactly one source node"
    source = int(source_nodes[0])

    graph: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(nodes))}
    seen_edges: set[tuple[int, int]] = set()
    for idx, node in enumerate(nodes):
        r0, c0, _r1, _c1 = node["bbox"]
        for _name, _border_off, start_off, delta in SIDE_SPECS:
            sr = r0 + start_off[0]
            sc = c0 + start_off[1]
            if not (0 <= sr < h and 0 <= sc < w):
                continue
            if int(grid[sr, sc]) not in LINE_COLORS:
                continue
            target, color = _follow_line(grid, (int(sr), int(sc)), delta, border_to_node)
            assert target != idx, "node cannot connect to itself"
            edge = tuple(sorted((idx, target)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            graph[idx].append((target, color))
            graph[target].append((idx, color))

    assert len(seen_edges) == len(nodes) - 1, "connectors must form a tree"
    return source, graph


def _apply_transform(pattern: np.ndarray, color: int) -> np.ndarray:
    if color == RED:
        return np.fliplr(pattern)
    if color == BLUE:
        return np.flipud(pattern)
    assert color == GREEN, "unexpected connector color"
    return np.rot90(pattern, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the source node's 3x3 pattern through the connector tree, transforming it by connector color at each step.

    Input structure:
    - The background is black.
    - The grid contains several identical 5x5 square nodes, each drawn as a one-pixel-thick grey frame with a 3x3 interior.
    - Exactly one node interior contains a non-black pattern; every other node interior is black.
    - Nodes are linked by one-pixel-wide horizontal or vertical connector lines colored red, blue, or green, and the connectors form a tree.

    Rule:
    - Treat the unique non-empty node as the source and traverse the connector tree outward from it.
    - Copy the parent's 3x3 pattern into each child after applying the transform indicated by the connecting line color.
    - Red means horizontal flip, blue means vertical flip, and green means a 90-degree clockwise rotation.
    - Keep all connector pixels, grey frames, and background cells unchanged; only fill node interiors.

    Key insight:
    - The connectors define a rooted propagation tree, so distant nodes receive the source pattern after a cascade of edge-wise geometric transforms rather than a single independent transform.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    nodes = _detect_nodes(g)
    source, graph = _extract_graph(g, nodes)

    patterns: dict[int, np.ndarray] = {source: nodes[source]["interior"].copy()}
    stack = [source]
    while stack:
        cur = stack.pop()
        for nxt, color in graph[cur]:
            if nxt in patterns:
                continue
            patterns[nxt] = _apply_transform(patterns[cur], color).copy()
            stack.append(nxt)

    assert len(patterns) == len(nodes), "tree must reach every node"
    out = g.copy()
    for idx, node in enumerate(nodes):
        r0, c0, r1, c1 = node["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = patterns[idx]
    return out


def _random_pattern() -> np.ndarray:
    for _ in range(200):
        pattern = np.zeros((3, 3), dtype=int)
        color_count = random.randint(1, 3)
        colors = random.sample(list(PATTERN_COLORS), color_count)
        cell_count = random.randint(3, 6)
        cells = random.sample(range(9), cell_count)
        for cell in cells:
            pattern[cell // 3, cell % 3] = random.choice(colors)
        variants = (
            pattern,
            np.fliplr(pattern),
            np.flipud(pattern),
            np.rot90(pattern, -1),
        )
        if len({variant.tobytes() for variant in variants}) == 4:
            return pattern
    raise RuntimeError("failed to sample a non-degenerate source pattern")


def _grow_tree(slot_rows: int, slot_cols: int, n_nodes: int) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    source = (random.randrange(slot_rows), random.randrange(slot_cols))
    occupied = {source}
    nodes = [source]
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(nodes) < n_nodes:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r, c in nodes:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                nxt = (int(nr), int(nc))
                if 0 <= nr < slot_rows and 0 <= nc < slot_cols and nxt not in occupied:
                    frontier.append(((int(r), int(c)), nxt))
        assert frontier, "grid growth should always have space while nodes remain"
        parent, child = random.choice(frontier)
        occupied.add(child)
        nodes.append(child)
        edges.append((parent, child))
    return nodes, edges


def _assign_edge_colors(edge_count: int) -> list[int]:
    assert edge_count >= 3, "need at least three edges to expose all transforms"
    colors = [RED, BLUE, GREEN]
    colors.extend(random.choice(LINE_COLORS) for _ in range(edge_count - 3))
    random.shuffle(colors)
    return colors


def _propagate_patterns(
    nodes: list[tuple[int, int]],
    colored_edges: list[tuple[tuple[int, int], tuple[int, int], int]],
    source: tuple[int, int],
    pattern: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {node: [] for node in nodes}
    for a, b, color in colored_edges:
        graph[a].append((b, color))
        graph[b].append((a, color))

    patterns: dict[tuple[int, int], np.ndarray] = {source: pattern.copy()}
    stack = [source]
    while stack:
        cur = stack.pop()
        for nxt, color in graph[cur]:
            if nxt in patterns:
                continue
            patterns[nxt] = _apply_transform(patterns[cur], color).copy()
            stack.append(nxt)
    assert len(patterns) == len(nodes)
    return patterns


def _draw_node(grid: np.ndarray, top: int, left: int, interior: np.ndarray) -> None:
    grid[top : top + 5, left] = GREY
    grid[top : top + 5, left + 4] = GREY
    grid[top, left : left + 5] = GREY
    grid[top + 4, left : left + 5] = GREY
    grid[top + 1 : top + 4, left + 1 : left + 4] = interior


def _draw_edge(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], step: int, color: int) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        row = ar * step + 2
        if ac < bc:
            c0 = ac * step + 5
            c1 = bc * step - 1
        else:
            c0 = bc * step + 5
            c1 = ac * step - 1
        grid[row, c0 : c1 + 1] = color
        return

    assert ac == bc
    col = ac * step + 2
    if ar < br:
        r0 = ar * step + 5
        r1 = br * step - 1
    else:
        r0 = br * step + 5
        r1 = ar * step - 1
    grid[r0 : r1 + 1, col] = color


def _crop_pair_with_padding(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BLACK)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    cropped_inp = inp[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    cropped_out = out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    h, w = cropped_inp.shape

    top_pad = random.randint(0, min(2, 30 - h))
    bottom_pad = random.randint(0, min(2, 30 - h - top_pad))
    left_pad = random.randint(0, min(2, 30 - w))
    right_pad = random.randint(0, min(2, 30 - w - left_pad))
    pads = ((top_pad, bottom_pad), (left_pad, right_pad))
    return (
        np.pad(
            cropped_inp,
            pads,
            constant_values=BLACK,
        ),
        np.pad(
            cropped_out,
            pads,
            constant_values=BLACK,
        ),
    )


def _render_case() -> tuple[np.ndarray, np.ndarray]:
    step = random.choice((6, 7))
    max_slots = 5 if step == 6 else 4
    slot_rows = random.randint(3, max_slots)
    slot_cols = random.randint(3, max_slots)
    n_nodes = random.randint(4, min(8, slot_rows * slot_cols))
    nodes, edges = _grow_tree(slot_rows, slot_cols, n_nodes)

    edge_colors = _assign_edge_colors(len(edges))
    colored_edges = [(a, b, color) for (a, b), color in zip(edges, edge_colors)]
    source = random.choice(nodes)
    source_pattern = _random_pattern()
    patterns = _propagate_patterns(nodes, colored_edges, source, source_pattern)

    canvas = np.zeros((5 + step * (slot_rows - 1), 5 + step * (slot_cols - 1)), dtype=int)
    inp = canvas.copy()
    out = canvas.copy()

    for node in nodes:
        top = node[0] * step
        left = node[1] * step
        if node == source:
            _draw_node(inp, top, left, source_pattern)
        else:
            _draw_node(inp, top, left, np.zeros((3, 3), dtype=int))
        _draw_node(out, top, left, patterns[node])

    for a, b, color in colored_edges:
        _draw_edge(inp, a, b, step, color)
        _draw_edge(out, a, b, step, color)

    return _crop_pair_with_padding(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp, out = _render_case()
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, out):
            return inp
    raise RuntimeError("failed to generate a valid propagation puzzle")
