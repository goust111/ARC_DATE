import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

PRIMARY_COLORS = (BLUE, RED, YELLOW)
COLOR_TO_BIT = {
    BLUE: 1,
    RED: 2,
    YELLOW: 4,
}
MASK_TO_COLOR = {
    1: BLUE,
    2: RED,
    4: YELLOW,
    3: MAGENTA,  # blue + red
    5: GREEN,    # blue + yellow
    6: ORANGE,   # red + yellow
    7: MAROON,   # blue + red + yellow
}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _touches_edge(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> bool:
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _surrounding_same_color(g: np.ndarray, r0: int, c0: int, color: int) -> bool:
    h, w = g.shape
    for rr in range(max(0, r0 - 1), min(h, r0 + 4)):
        for cc in range(max(0, c0 - 1), min(w, c0 + 4)):
            if r0 <= rr <= r0 + 2 and c0 <= cc <= c0 + 2:
                continue
            if int(g[rr, cc]) == int(color):
                return True
    return False


def _find_source_blocks(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    sources = []
    used = set()
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            bbox = (r0, c0, r0 + 2, c0 + 2)
            if bbox in used:
                continue
            if not _touches_edge(*bbox, h, w):
                continue
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            color = int(patch[0, 0])
            if color in (BLACK, GREY):
                continue
            if not bool(np.all(patch == color)):
                continue
            if _surrounding_same_color(g, r0, c0, color):
                continue
            sources.append({"bbox": bbox, "color": color})
            used.add(bbox)
    return sources


def _find_containers(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    containers = []
    used_centers = set()
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if int(patch[1, 1]) != BLACK:
                continue
            border = []
            for rr in range(3):
                for cc in range(3):
                    if rr == 1 and cc == 1:
                        continue
                    border.append(int(patch[rr, cc]))
            if all(v == GREY for v in border):
                center = (r0 + 1, c0 + 1)
                if center in used_centers:
                    continue
                frame_cells = [
                    (r0 + rr, c0 + cc)
                    for rr in range(3)
                    for cc in range(3)
                    if not (rr == 1 and cc == 1)
                ]
                containers.append(
                    {
                        "bbox": (r0, c0, r0 + 2, c0 + 2),
                        "center": center,
                        "frame_cells": frame_cells,
                    }
                )
                used_centers.add(center)
    return containers


def _adjacent_pipe_pixels_to_bbox(
    g: np.ndarray, bbox: tuple[int, int, int, int], frame_mask: np.ndarray
) -> set[tuple[int, int]]:
    h, w = g.shape
    r0, c0, r1, c1 = bbox
    pts = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            for nr, nc in _neighbors4(r, c, h, w):
                if r0 <= nr <= r1 and c0 <= nc <= c1:
                    continue
                if int(g[nr, nc]) == GREY and not bool(frame_mask[nr, nc]):
                    pts.add((nr, nc))
    return pts


def _adjacent_pipe_pixels_to_frame(
    g: np.ndarray, frame_cells: list[tuple[int, int]], frame_mask: np.ndarray
) -> set[tuple[int, int]]:
    h, w = g.shape
    frame_set = set(frame_cells)
    pts = set()
    for r, c in frame_cells:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in frame_set:
                continue
            if int(g[nr, nc]) == GREY and not bool(frame_mask[nr, nc]):
                pts.add((nr, nc))
    return pts


def _pipe_components(pipe_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = pipe_mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not bool(pipe_mask[r, c]) or bool(vis[r, c]):
                continue
            vis[r, c] = True
            comp = []
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(pipe_mask[nr, nc]) and not bool(vis[nr, nc]):
                        vis[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _build_adjacency(nodes: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    node_set = set(nodes)
    adj = {node: [] for node in nodes}
    for r, c in nodes:
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (nr, nc) in node_set:
                adj[(r, c)].append((nr, nc))
    return adj


def _orient_tree_component(
    nodes: list[tuple[int, int]],
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    source_bits: dict[tuple[int, int], int],
    sink_nodes: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], dict[tuple[int, int], int]]:
    node_set = set(nodes)
    root = nodes[0]
    parent = {root: None}
    order = [root]
    stack = [root]
    while stack:
        node = stack.pop()
        for nxt in adj[node]:
            if nxt not in node_set or nxt == parent[node]:
                continue
            if nxt in parent:
                raise AssertionError("pipe component must be acyclic")
            parent[nxt] = node
            order.append(nxt)
            stack.append(nxt)
    if len(parent) != len(nodes):
        raise AssertionError("pipe component must be connected")

    edge_count = sum(len(adj[node]) for node in nodes) // 2
    if edge_count != len(nodes) - 1:
        raise AssertionError("pipe component must be a tree")

    children = {node: [] for node in nodes}
    for node, par in parent.items():
        if par is not None:
            children[par].append(node)

    subtree_src = {}
    subtree_sink = {}
    for node in reversed(order):
        src_count = 1 if node in source_bits else 0
        sink_count = 1 if node in sink_nodes else 0
        for child in children[node]:
            src_count += subtree_src[child]
            sink_count += subtree_sink[child]
        subtree_src[node] = src_count
        subtree_sink[node] = sink_count

    total_src = subtree_src[root]
    total_sink = subtree_sink[root]
    if total_src <= 0 or total_sink <= 0:
        raise AssertionError("every pipe component must connect sources to containers")

    out_edges = {node: [] for node in nodes}
    indeg = {node: 0 for node in nodes}
    for child in order[1:]:
        par = parent[child]
        child_src = subtree_src[child]
        child_sink = subtree_sink[child]
        other_src = total_src - child_src
        other_sink = total_sink - child_sink

        if child_src > 0 and child_sink == 0:
            src_node, dst_node = child, par
        elif child_sink > 0 and child_src == 0:
            src_node, dst_node = par, child
        elif other_src > 0 and other_sink == 0:
            src_node, dst_node = par, child
        elif other_sink > 0 and other_src == 0:
            src_node, dst_node = child, par
        else:
            raise AssertionError("ambiguous flow orientation in pipe tree")

        out_edges[src_node].append(dst_node)
        indeg[dst_node] += 1

    return out_edges, indeg


def _propagate_masks(
    nodes: list[tuple[int, int]],
    out_edges: dict[tuple[int, int], list[tuple[int, int]]],
    indeg: dict[tuple[int, int], int],
    source_bits: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    indeg_work = dict(indeg)
    masks = {node: source_bits.get(node, 0) for node in nodes}
    q = deque(node for node in nodes if indeg_work[node] == 0)
    seen = 0
    while q:
        node = q.popleft()
        seen += 1
        if masks[node] == 0:
            raise AssertionError("every upstream pipe leaf must be a source")
        for nxt in out_edges[node]:
            masks[nxt] |= masks[node]
            indeg_work[nxt] -= 1
            if indeg_work[nxt] == 0:
                q.append(nxt)
    if seen != len(nodes):
        raise AssertionError("directed pipe graph must be a DAG")
    return masks


def _mask_to_output_color(mask: int) -> int:
    if mask not in MASK_TO_COLOR:
        raise AssertionError("unsupported color mixture")
    return MASK_TO_COLOR[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors along a grey pipe tree and fill each container with
    the final downstream color after merges.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sources = _find_source_blocks(g)
    if not sources:
        return g.copy()
    if not {src["color"] for src in sources}.issubset(COLOR_TO_BIT):
        raise AssertionError("source colors must be chosen from red, blue, and yellow")

    containers = _find_containers(g)
    if not containers:
        return g.copy()

    frame_mask = np.zeros((h, w), dtype=bool)
    for container in containers:
        for r, c in container["frame_cells"]:
            frame_mask[r, c] = True

    for src in sources:
        adj = _adjacent_pipe_pixels_to_bbox(g, src["bbox"], frame_mask)
        if len(adj) != 1:
            raise AssertionError("each source block must connect to exactly one pipe cell")
        src["pipe_start"] = next(iter(adj))

    center_to_terminal = {}
    for container in containers:
        adj = _adjacent_pipe_pixels_to_frame(g, container["frame_cells"], frame_mask)
        if len(adj) != 1:
            raise AssertionError("each container must connect to exactly one pipe cell")
        terminal = next(iter(adj))
        container["pipe_terminal"] = terminal
        center_to_terminal[container["center"]] = terminal

    pipe_mask = (g == GREY) & (~frame_mask)
    pipe_comps = _pipe_components(pipe_mask)
    if not pipe_comps:
        return g.copy()

    source_bits = {
        tuple(src["pipe_start"]): COLOR_TO_BIT[int(src["color"])]
        for src in sources
    }
    sink_nodes = {tuple(container["pipe_terminal"]) for container in containers}

    node_mask = {}
    for comp in pipe_comps:
        comp_set = set(comp)
        comp_source_bits = {node: bit for node, bit in source_bits.items() if node in comp_set}
        comp_sink_nodes = {node for node in sink_nodes if node in comp_set}
        if not comp_source_bits and not comp_sink_nodes:
            continue
        if not comp_source_bits or not comp_sink_nodes:
            raise AssertionError("every pipe component must connect source blocks to containers")
        adj = _build_adjacency(comp)
        out_edges, indeg = _orient_tree_component(comp, adj, comp_source_bits, comp_sink_nodes)
        comp_masks = _propagate_masks(comp, out_edges, indeg, comp_source_bits)
        node_mask.update(comp_masks)

    out = g.copy()
    for (r, c), mask in node_mask.items():
        out[r, c] = _mask_to_output_color(mask)
    for container in containers:
        center = container["center"]
        terminal = center_to_terminal[center]
        out[center] = _mask_to_output_color(node_mask[terminal])
    return out


def _sample_spaced_starts(count: int, low: int, high: int, min_gap: int) -> list[int]:
    values = list(range(low, high + 1))
    for _ in range(300):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    raise RuntimeError("failed to sample non-overlapping rows")


def _paint_pipe(grid: np.ndarray, r: int, c: int) -> None:
    val = int(grid[r, c])
    if val not in (BLACK, GREY):
        raise AssertionError("pipe overlaps another object")
    grid[r, c] = GREY


def _draw_hline(grid: np.ndarray, r: int, c0: int, c1: int) -> None:
    lo, hi = sorted((c0, c1))
    for c in range(lo, hi + 1):
        _paint_pipe(grid, r, c)


def _draw_vline(grid: np.ndarray, c: int, r0: int, r1: int) -> None:
    lo, hi = sorted((r0, r1))
    for r in range(lo, hi + 1):
        _paint_pipe(grid, r, c)


def _draw_source(grid: np.ndarray, r0: int, color: int) -> tuple[int, int]:
    grid[r0 : r0 + 3, 0:3] = int(color)
    return r0 + 1, 3


def _draw_container(grid: np.ndarray, r0: int, c0: int) -> tuple[tuple[int, int], tuple[int, int]]:
    grid[r0 : r0 + 3, c0 : c0 + 3] = GREY
    grid[r0 + 1, c0 + 1] = BLACK
    center = (r0 + 1, c0 + 1)
    terminal = (r0 + 1, c0 - 1)
    return center, terminal


def _build_base_input() -> np.ndarray:
    branch_mode = random.randrange(4)
    if branch_mode == 0:
        source_count = 1
        container_count = 1
    elif branch_mode == 1:
        source_count = random.choice((2, 3))
        container_count = 1
    elif branch_mode == 2:
        source_count = 1
        container_count = random.choice((2, 3))
    else:
        source_count = random.choice((2, 3))
        container_count = random.choice((2, 3))

    h_min = 14 + 2 * max(source_count, container_count)
    h = random.randint(h_min, h_min + 8)
    w = random.randint(22, 30)

    source_rows = _sample_spaced_starts(source_count, 1, h - 4, 4)
    container_rows = _sample_spaced_starts(container_count, 1, h - 4, 4)
    source_colors = random.sample(PRIMARY_COLORS, source_count)

    grid = np.zeros((h, w), dtype=int)

    merge_col = random.randint(5, min(8, w - 15))
    split_col = random.randint(max(merge_col + 6, w - 10), w - 7)
    trunk_row = random.randint(2, h - 3)

    source_leaves = []
    for row, color in zip(source_rows, source_colors, strict=False):
        source_leaves.append(_draw_source(grid, row, color))

    container_specs = []
    for row in container_rows:
        c0 = random.randint(split_col + 3, w - 4)
        center, terminal = _draw_container(grid, row, c0)
        container_specs.append((center, terminal))

    merge_rows = [r for r, _ in source_leaves] + [trunk_row]
    split_rows = [terminal[0] for _, terminal in container_specs] + [trunk_row]

    _draw_vline(grid, merge_col, min(merge_rows), max(merge_rows))
    _draw_hline(grid, trunk_row, merge_col, split_col)
    _draw_vline(grid, split_col, min(split_rows), max(split_rows))

    for r, c in source_leaves:
        _draw_hline(grid, r, c, merge_col)
    for _, terminal in container_specs:
        _draw_hline(grid, terminal[0], split_col, terminal[1])

    return grid


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4)).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    if random.random() < 0.5:
        out = np.flipud(out).copy()
    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp = _random_transform(_build_base_input())
        if int(np.bincount(inp.ravel(), minlength=10).argmax()) != BLACK:
            continue
        try:
            out = main(inp)
        except AssertionError:
            continue
        if not isinstance(out, np.ndarray):
            continue
        if np.array_equal(inp, out):
            continue
        return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
