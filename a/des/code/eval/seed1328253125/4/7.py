from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

PRIMARY_COLORS = (RED, BLUE, YELLOW)
PRIMARY_SET_OPTIONS = [frozenset({color}) for color in PRIMARY_COLORS]
PAIR_SET_OPTIONS = [
    frozenset({RED, BLUE}),
    frozenset({RED, YELLOW}),
    frozenset({BLUE, YELLOW}),
]
TRIPLE_SET = frozenset(PRIMARY_COLORS)
SIGNAL_TO_COLOR = {
    frozenset({RED}): RED,
    frozenset({BLUE}): BLUE,
    frozenset({YELLOW}): YELLOW,
    frozenset({RED, BLUE}): MAGENTA,
    frozenset({RED, YELLOW}): ORANGE,
    frozenset({BLUE, YELLOW}): GREEN,
    TRIPLE_SET: MAROON,
}

BG_CHOICES = [0, CYAN]
ROW_STEP_OPTIONS = [4, 5]
GAP_UNIT_OPTIONS = [1, 2]
GEN_RETRIES = 500

_NEXT_TOP_KIND = 0


Coord = tuple[int, int]


def _most_common_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(values).reshape(-1))
    ranked = counts.most_common(2)
    assert ranked, "expected a nonempty grid"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "expected a unique dominant background color"
    return int(ranked[0][0])


def _neighbors(shape: tuple[int, int], cell: Coord) -> list[Coord]:
    h, w = shape
    r, c = cell
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _signal_color(signal: frozenset[int]) -> int:
    return int(SIGNAL_TO_COLOR[signal])


def _component_cells(adj: dict[Coord, list[Coord]], start: Coord) -> set[Coord]:
    seen = {start}
    q = deque([start])
    while q:
        cell = q.popleft()
        for nxt in adj[cell]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _edge_cells_from_child(
    edge: dict[str, object],
    child: Coord,
) -> list[Coord]:
    path = edge["path"]
    assert isinstance(path, list)
    if edge["u"] == child:
        return list(path[:-1])
    assert edge["v"] == child
    rev = list(reversed(path))
    return rev[:-1]


def _compress_component(
    grid: np.ndarray,
    component: set[Coord],
    adj: dict[Coord, list[Coord]],
    gray_cells: set[Coord],
    source_cells: set[Coord],
) -> tuple[set[Coord], dict[Coord, list[tuple[Coord, int]]], dict[int, dict[str, object]], Coord]:
    full_degree = {cell: len([nxt for nxt in adj[cell] if nxt in component]) for cell in component}
    gray_degree = {
        cell: sum(1 for nxt in adj[cell] if nxt in gray_cells)
        for cell in component
        if cell in gray_cells
    }

    junctions = {
        cell
        for cell in component
        if cell in gray_cells and gray_degree.get(cell, 0) in (3, 4)
    }
    gray_leaves = {
        cell
        for cell in component
        if cell in gray_cells and full_degree[cell] <= 1
    }
    significant = set(source_cells & component) | junctions | gray_leaves
    assert significant, "expected at least one significant node in the component"

    gray_significant = sorted(cell for cell in significant if cell in gray_cells)
    if gray_significant:
        leaf_candidates = [cell for cell in gray_significant if full_degree[cell] <= 1]
        root = min(leaf_candidates or gray_significant)
    else:
        root = min(significant)

    comp_adj = {cell: [nxt for nxt in adj[cell] if nxt in component] for cell in component}
    graph = {cell: [] for cell in significant}
    edges: dict[int, dict[str, object]] = {}
    visited_steps: set[tuple[Coord, Coord]] = set()

    for start in sorted(significant):
        for nxt in comp_adj[start]:
            if (start, nxt) in visited_steps:
                continue
            path = [start]
            prev = start
            cur = nxt
            while cur not in significant:
                path.append(cur)
                options = [cand for cand in comp_adj[cur] if cand != prev]
                assert len(options) == 1, "non-significant cells must form simple path segments"
                prev, cur = cur, options[0]
            path.append(cur)
            end = cur

            for a, b in zip(path, path[1:]):
                visited_steps.add((a, b))
                visited_steps.add((b, a))

            edge_id = len(edges)
            edges[edge_id] = {"u": start, "v": end, "path": path}
            graph[start].append((end, edge_id))
            graph[end].append((start, edge_id))

    seen = {root}
    q = deque([root])
    while q:
        cell = q.popleft()
        for nxt, _edge_id in graph[cell]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert seen == significant, "significant-node graph must stay connected per component"
    assert len(edges) == len(significant) - 1, "expected a tree-shaped compressed network"
    return significant, graph, edges, root


def _paint_pure_gray_subtree(
    out: np.ndarray,
    node: Coord,
    color: int,
    children: dict[Coord, list[Coord]],
    parent_of: dict[Coord, Coord | None],
    edge_to_parent: dict[Coord, int],
    edges: dict[int, dict[str, object]],
) -> None:
    parent = parent_of[node]
    assert parent is not None
    cells = _edge_cells_from_child(edges[edge_to_parent[node]], node)
    for cell in cells:
        out[cell] = color
    for child in children[node]:
        _paint_pure_gray_subtree(out, child, color, children, parent_of, edge_to_parent, edges)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate red/blue/yellow source colors through a gray orthogonal network, mixing at junctions.

    Input structure:
    - The background is the dominant color.
    - Gray pixels form a 4-connected network of one-pixel-wide orthogonal segments.
    - Some leaf endpoints are replaced by red, blue, or yellow source pixels.
    - Gray junctions are the gray pixels with 3 or 4 gray neighbors.

    Rule:
    - Each source colors the gray path segment until the next junction.
    - Junctions collect the distinct primary signals arriving from already-resolved child branches.
    - One primary keeps that color, two primaries mix to magenta/orange/green, three primaries turn maroon.
    - Single- and two-primary junction colors continue toward the root; maroon stops there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    colors = {int(v) for v in np.unique(g)}
    non_bg = colors - {bg}
    gray_candidates = non_bg - set(PRIMARY_COLORS)
    assert len(gray_candidates) == 1, "expected exactly one gray network color"
    gray = next(iter(gray_candidates))
    assert gray == GRAY, "this task uses the visible gray channel for wires"

    network_mask = np.isin(g, [gray, *PRIMARY_COLORS])
    network_cells = [tuple(map(int, cell)) for cell in np.argwhere(network_mask)]
    assert network_cells, "expected a visible signal network"

    adj: dict[Coord, list[Coord]] = {}
    gray_cells: set[Coord] = set()
    source_cells: set[Coord] = set()
    for cell in network_cells:
        color = int(g[cell])
        if color == gray:
            gray_cells.add(cell)
        else:
            assert color in PRIMARY_COLORS, "only red/blue/yellow sources may replace endpoints"
            source_cells.add(cell)
        adj[cell] = [nxt for nxt in _neighbors(g.shape, cell) if network_mask[nxt]]

    for cell in source_cells:
        assert len(adj[cell]) == 1, "source pixels must sit at leaf endpoints"

    out = g.copy()
    remaining = set(network_cells)
    while remaining:
        start = next(iter(remaining))
        component = _component_cells(adj, start)
        remaining -= component

        comp_sources = source_cells & component
        if not comp_sources:
            continue

        significant, graph, edges, root = _compress_component(g, component, adj, gray_cells, source_cells)
        children = {cell: [] for cell in significant}
        parent_of: dict[Coord, Coord | None] = {root: None}
        edge_to_parent: dict[Coord, int] = {}
        order: list[Coord] = []

        q = deque([root])
        while q:
            cell = q.popleft()
            order.append(cell)
            for nxt, edge_id in sorted(graph[cell]):
                if nxt in parent_of:
                    continue
                parent_of[nxt] = cell
                edge_to_parent[nxt] = edge_id
                children[cell].append(nxt)
                q.append(nxt)

        assert len(order) == len(significant), "rooted traversal must cover the component"

        local_signal: dict[Coord, frozenset[int]] = {}
        propagated_signal: dict[Coord, frozenset[int]] = {}
        source_count: dict[Coord, int] = {}

        for cell in reversed(order):
            if cell in comp_sources:
                signal = frozenset({int(g[cell])})
                local_signal[cell] = signal
                propagated_signal[cell] = signal
                source_count[cell] = 1
                continue

            source_count[cell] = sum(source_count[ch] for ch in children[cell])
            collected: set[int] = set()
            for ch in children[cell]:
                collected.update(propagated_signal[ch])
            signal = frozenset(collected)
            local_signal[cell] = signal
            if cell == root or not signal or len(signal) == 3:
                propagated_signal[cell] = frozenset()
            else:
                propagated_signal[cell] = signal

        for cell in order:
            if cell in comp_sources:
                continue
            signal = local_signal[cell]
            if signal:
                out[cell] = _signal_color(signal)

        for cell in order[1:]:
            signal = propagated_signal[cell]
            if not signal:
                continue
            color = _signal_color(signal)
            for path_cell in _edge_cells_from_child(edges[edge_to_parent[cell]], cell):
                out[path_cell] = color

        for cell in order:
            signal = local_signal[cell]
            if not signal or len(signal) == 3:
                continue
            color = _signal_color(signal)
            for ch in children[cell]:
                if source_count[ch] == 0:
                    _paint_pure_gray_subtree(out, ch, color, children, parent_of, edge_to_parent, edges)

    return out


def _make_source(color: int) -> dict[str, object]:
    return {"kind": "source", "color": int(color), "children": []}


def _build_single_target(target: frozenset[int], depth: int, force_internal: bool = False) -> dict[str, object]:
    color = next(iter(target))
    if depth <= 0 and not force_internal:
        return _make_source(color)

    child_count = random.choice([2, 3])
    recursive_slot = random.randrange(child_count) if depth > 0 else -1
    children = []
    for idx in range(child_count):
        recurse = depth > 0 and idx == recursive_slot and (force_internal or random.random() < 0.8)
        children.append(_build_single_target(target, depth - 1, force_internal=recurse))
    random.shuffle(children)
    return {"kind": "junction", "children": children}


def _build_pair_target(target: frozenset[int], depth: int) -> dict[str, object]:
    a, b = tuple(sorted(target))
    if depth <= 0:
        children = [_make_source(a), _make_source(b)]
        if random.random() < 0.5:
            children.append(_make_source(random.choice([a, b])))
        random.shuffle(children)
        return {"kind": "junction", "children": children}

    patterns = [
        [target, frozenset({a})],
        [target, frozenset({b})],
        [frozenset({a}), frozenset({b})],
        [frozenset({a}), frozenset({b}), frozenset({random.choice([a, b])})],
        [target, frozenset({a}), frozenset({b})],
    ]
    child_targets = random.choice(patterns)
    children = []
    for child_target in child_targets:
        if len(child_target) == 2:
            children.append(_build_pair_target(child_target, depth - 1))
        else:
            force_internal = depth > 0 and random.random() < 0.25
            children.append(_build_single_target(child_target, depth - 1, force_internal=force_internal))
    random.shuffle(children)
    return {"kind": "junction", "children": children}


def _build_triple_target(depth: int) -> dict[str, object]:
    if depth <= 0:
        children = [_make_source(RED), _make_source(BLUE), _make_source(YELLOW)]
        random.shuffle(children)
        return {"kind": "junction", "children": children}

    pair = random.choice(PAIR_SET_OPTIONS)
    leftover = next(iter(TRIPLE_SET - pair))
    patterns = [
        [pair, frozenset({leftover})],
        [frozenset({RED}), frozenset({BLUE}), frozenset({YELLOW})],
        [pair, frozenset({leftover}), frozenset({random.choice(tuple(pair))})],
    ]
    child_targets = random.choice(patterns)
    children = []
    for child_target in child_targets:
        if len(child_target) == 2:
            children.append(_build_pair_target(child_target, depth - 1))
        else:
            force_internal = depth > 0 and random.random() < 0.2
            children.append(_build_single_target(child_target, depth - 1, force_internal=force_internal))
    random.shuffle(children)
    return {"kind": "junction", "children": children}


def _annotate_tree(node: dict[str, object]) -> None:
    kind = node["kind"]
    if kind == "source":
        signal = frozenset({int(node["color"])})
        node["local_signal"] = signal
        node["outgoing_signal"] = signal
        node["source_count"] = 1
        node["junction_count"] = 0
        node["height"] = 0
        return

    children = node["children"]
    assert isinstance(children, list) and children
    for child in children:
        _annotate_tree(child)

    union: set[int] = set()
    source_count = 0
    junction_count = 0
    height = 0
    for child in children:
        union.update(child["outgoing_signal"])
        source_count += int(child["source_count"])
        junction_count += int(child["junction_count"])
        height = max(height, int(child["height"]) + 1)

    signal = frozenset(union)
    node["local_signal"] = signal
    node["source_count"] = source_count
    node["junction_count"] = junction_count + (0 if kind == "root" else 1)
    node["height"] = height
    if kind == "root" or not signal or len(signal) == 3:
        node["outgoing_signal"] = frozenset()
    else:
        node["outgoing_signal"] = signal


def _set_spans(node: dict[str, object], gap_units: int) -> int:
    if node["kind"] == "source":
        node["span"] = 1
        return 1

    children = node["children"]
    spans = [_set_spans(child, gap_units) for child in children]
    span = sum(spans) + gap_units * max(0, len(spans) - 1)
    node["span"] = span
    return span


def _assign_positions(node: dict[str, object], left_unit: int, depth: int, row_step: int, gap_units: int) -> None:
    node["r"] = depth * row_step
    if node["kind"] == "source":
        node["c"] = left_unit * 4
        return

    children = node["children"]
    cur = left_unit
    for child in children:
        _assign_positions(child, cur, depth + 1, row_step, gap_units)
        cur += int(child["span"]) + gap_units

    if len(children) == 1:
        node["c"] = int(children[0]["c"])
    elif len(children) == 2:
        node["c"] = (int(children[0]["c"]) + int(children[1]["c"])) // 2
    else:
        node["c"] = int(children[1]["c"])


def _walk_nodes(node: dict[str, object]) -> list[dict[str, object]]:
    nodes = [node]
    for child in node["children"]:
        nodes.extend(_walk_nodes(child))
    return nodes


def _draw_edge(grid: np.ndarray, parent: dict[str, object], child: dict[str, object]) -> None:
    pr, pc = int(parent["r"]), int(parent["c"])
    cr, cc = int(child["r"]), int(child["c"])
    c0, c1 = sorted((pc, cc))
    grid[pr, c0 : c1 + 1] = GRAY
    r0, r1 = sorted((pr, cr))
    grid[r0 : r1 + 1, cc] = GRAY


def _render_input(root: dict[str, object], bg: int) -> tuple[np.ndarray, Coord, Coord]:
    nodes = _walk_nodes(root)
    max_r = max(int(node["r"]) for node in nodes)
    max_c = max(int(node["c"]) for node in nodes)

    canvas = np.full((max_r + 1, max_c + 1), bg, dtype=int)

    def paint(node: dict[str, object]) -> None:
        for child in node["children"]:
            _draw_edge(canvas, node, child)
            paint(child)
        if node["kind"] == "source":
            canvas[int(node["r"]), int(node["c"])] = int(node["color"])
        else:
            canvas[int(node["r"]), int(node["c"])] = GRAY

    paint(root)

    top_junction = root["children"][0]
    root_coord = (int(root["r"]), int(root["c"]))
    top_coord = (int(top_junction["r"]), int(top_junction["c"]))

    pad_top = random.randint(1, 3)
    pad_bottom = random.randint(1, 4)
    pad_left = random.randint(1, 4)
    pad_right = random.randint(1, 4)
    padded = np.pad(
        canvas,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=bg,
    )
    shifted_root = (root_coord[0] + pad_top, root_coord[1] + pad_left)
    shifted_top = (top_coord[0] + pad_top, top_coord[1] + pad_left)
    return padded, shifted_root, shifted_top


def _build_case(top_kind: int) -> tuple[np.ndarray, frozenset[int], Coord, Coord] | None:
    depth = random.randint(1, 2)
    if top_kind == 1:
        target = random.choice(PRIMARY_SET_OPTIONS)
        top = _build_single_target(target, depth, force_internal=True)
    elif top_kind == 2:
        target = random.choice(PAIR_SET_OPTIONS)
        top = _build_pair_target(target, depth)
    else:
        target = TRIPLE_SET
        top = _build_triple_target(depth)

    root = {"kind": "root", "children": [top]}
    _annotate_tree(root)
    if int(root["source_count"]) < 2:
        return None
    if top["local_signal"] != target:
        return None
    if int(root["junction_count"]) < 2 or int(root["junction_count"]) > 6:
        return None

    row_step = random.choice(ROW_STEP_OPTIONS)
    gap_units = random.choice(GAP_UNIT_OPTIONS)
    _set_spans(root, gap_units)
    _assign_positions(root, 0, 0, row_step, gap_units)

    bg = random.choice(BG_CHOICES)
    inp, root_coord, top_coord = _render_input(root, bg)
    return inp, target, root_coord, top_coord


def _valid_generated_case(
    inp: np.ndarray,
    target: frozenset[int],
    root_coord: Coord,
    top_coord: Coord,
) -> bool:
    try:
        out = main(inp)
    except AssertionError:
        return False

    if np.array_equal(inp, out):
        return False

    if target == TRIPLE_SET:
        if int(out[top_coord]) != MAROON:
            return False
        if int(out[root_coord]) != GRAY:
            return False
    else:
        if int(out[root_coord]) != _signal_color(target):
            return False
        if int(out[top_coord]) != _signal_color(target):
            return False

    bg = _most_common_color(inp)
    if int((inp == bg).sum()) <= int(inp.size * 0.55):
        return False
    if int((inp == GRAY).sum()) < 6:
        return False
    return True


def generate_input() -> np.ndarray:
    global _NEXT_TOP_KIND

    top_kind = [1, 2, 3][_NEXT_TOP_KIND]
    _NEXT_TOP_KIND = (_NEXT_TOP_KIND + 1) % 3

    for _ in range(GEN_RETRIES):
        built = _build_case(top_kind)
        if built is None:
            continue
        inp, target, root_coord, top_coord = built
        if _valid_generated_case(inp, target, root_coord, top_coord):
            return inp
    raise RuntimeError("failed to generate a valid signal-mixing network")
