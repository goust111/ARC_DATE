from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
NODE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _detect_nodes(grid: np.ndarray):
    h, w = grid.shape
    owner = np.full((h, w), -1, dtype=int)
    nodes: list[dict] = []

    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, GREY)):
        comps = _components(grid == color)
        assert len(comps) == 1, "each node color must appear in exactly one component"
        coords = comps[0]
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        area = int(coords.shape[0])
        assert (int(r1 - r0), int(c1 - c0)) == (2, 2), "nodes must have 3x3 bounding boxes"
        assert area in (8, 9), "nodes must be solid or center-hollow 3x3 blocks"
        center = (int(r0 + 1), int(c0 + 1))
        center_val = int(grid[center])
        assert center_val in (BG, color), "node center must be black or the node color"
        for rr in range(int(r0), int(r1) + 1):
            for cc in range(int(c0), int(c1) + 1):
                if (rr, cc) == center:
                    continue
                assert int(grid[rr, cc]) == color, "node ring must be monochrome"
        node_id = len(nodes)
        for rr in range(int(r0), int(r1) + 1):
            for cc in range(int(c0), int(c1) + 1):
                assert owner[rr, cc] == -1, "node areas may not overlap"
                owner[rr, cc] = node_id
        nodes.append(
            {
                "id": node_id,
                "color": color,
                "sender": center_val == BG,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "center": center,
            }
        )

    assert len(nodes) >= 2, "there must be at least two nodes"
    return nodes, owner


def _extract_paths(grid: np.ndarray, owner: np.ndarray):
    h, w = grid.shape
    paths: list[dict] = []
    for coords in _components(grid == GREY):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if coords.shape[0] == 1:
            r = int(coords[0, 0])
            c = int(coords[0, 1])
            candidates: list[tuple[str, int, int]] = []
            if c - 1 >= 0 and c + 1 < w:
                a = int(owner[r, c - 1])
                b = int(owner[r, c + 1])
                if a >= 0 and b >= 0 and a != b:
                    candidates.append(("h", a, b))
            if r - 1 >= 0 and r + 1 < h:
                a = int(owner[r - 1, c])
                b = int(owner[r + 1, c])
                if a >= 0 and b >= 0 and a != b:
                    candidates.append(("v", a, b))
            assert len(candidates) == 1, "single-cell path must connect exactly one visible direction"
            orientation, a, b = candidates[0]
            paths.append({"coords": coords.copy(), "a": a, "b": b, "orientation": orientation})
            continue
        if int(r0) == int(r1):
            ordered = coords[np.argsort(coords[:, 1])]
            row = int(r0)
            c_start = int(ordered[0, 1])
            c_end = int(ordered[-1, 1])
            assert ordered.shape[0] == c_end - c_start + 1, "horizontal paths must be contiguous"
            assert c_start - 1 >= 0 and c_end + 1 < w, "path must terminate at nodes"
            a = int(owner[row, c_start - 1])
            b = int(owner[row, c_end + 1])
            assert a >= 0 and b >= 0 and a != b, "horizontal path must connect two different nodes"
            paths.append({"coords": ordered, "a": a, "b": b, "orientation": "h"})
        else:
            assert int(c0) == int(c1), "paths must be straight"
            ordered = coords[np.argsort(coords[:, 0])]
            col = int(c0)
            r_start = int(ordered[0, 0])
            r_end = int(ordered[-1, 0])
            assert ordered.shape[0] == r_end - r_start + 1, "vertical paths must be contiguous"
            assert r_start - 1 >= 0 and r_end + 1 < h, "path must terminate at nodes"
            a = int(owner[r_start - 1, col])
            b = int(owner[r_end + 1, col])
            assert a >= 0 and b >= 0 and a != b, "vertical path must connect two different nodes"
            paths.append({"coords": ordered, "a": a, "b": b, "orientation": "v"})
    assert paths, "there must be at least one grey path"
    return paths


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    nodes, owner = _detect_nodes(g)
    paths = _extract_paths(g, owner)
    out = g.copy()

    for path in paths:
        a = nodes[path["a"]]
        b = nodes[path["b"]]
        coords = np.asarray(path["coords"], dtype=int)
        a_sender = bool(a["sender"])
        b_sender = bool(b["sender"])

        if not a_sender and not b_sender:
            continue
        if a_sender and not b_sender:
            out[coords[:, 0], coords[:, 1]] = int(a["color"])
            continue
        if b_sender and not a_sender:
            out[coords[:, 0], coords[:, 1]] = int(b["color"])
            continue

        half = coords.shape[0] // 2
        if half > 0:
            out[coords[:half, 0], coords[:half, 1]] = int(a["color"])
        out[coords[half:, 0], coords[half:, 1]] = int(b["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey link according to whether its endpoint nodes send or receive the signal.

    Input structure:
    - The background is black and the only neutral link color is grey.
    - Each node is a unique-colored 3x3 block; its center pixel is either black (sender) or still filled (receiver).
    - Grey connections are 1-cell-thick straight horizontal or vertical segments that stop just outside the 3x3 node areas.

    Rule:
    - Keep every node's 3x3 area unchanged.
    - A sender-receiver path becomes entirely the sender's color.
    - A receiver-receiver path stays grey.
    - A sender-sender path is colored from both ends toward the middle; when its length is odd, the single middle cell belongs to the right-hand sender on a horizontal link or the lower sender on a vertical link.

    Key insight:
    The tiny center-pixel change is the role marker, so each straight grey segment can be solved independently once its two endpoint nodes are classified.
    """
    return _transform(grid)


def _slot_neighbors(slot: tuple[int, int], rows: int, cols: int):
    r, c = slot
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _build_graph(rng: random.Random):
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    max_nodes = min(7, rows * cols)
    n_nodes = rng.randint(4, max_nodes)

    start = (rng.randrange(rows), rng.randrange(cols))
    slots = [start]
    slot_to_id = {start: 0}
    edges: list[tuple[int, int]] = []
    degrees = [0]

    while len(slots) < n_nodes:
        frontier: list[tuple[int, tuple[int, int]]] = []
        for slot, node_id in slot_to_id.items():
            if degrees[node_id] >= 3:
                continue
            for nbr in _slot_neighbors(slot, rows, cols):
                if nbr not in slot_to_id:
                    frontier.append((node_id, nbr))
        if not frontier:
            return None
        parent_id, new_slot = rng.choice(frontier)
        new_id = len(slots)
        slots.append(new_slot)
        slot_to_id[new_slot] = new_id
        degrees[parent_id] += 1
        degrees.append(1)
        edges.append((parent_id, new_id))

    edge_set = {tuple(sorted(edge)) for edge in edges}
    extra_budget = rng.randint(0, 2)
    for _ in range(extra_budget):
        candidates: list[tuple[int, int]] = []
        for slot, node_id in slot_to_id.items():
            if degrees[node_id] >= 3:
                continue
            for nbr in _slot_neighbors(slot, rows, cols):
                other_id = slot_to_id.get(nbr)
                if other_id is None or other_id <= node_id or degrees[other_id] >= 3:
                    continue
                edge = tuple(sorted((node_id, other_id)))
                if edge not in edge_set:
                    candidates.append(edge)
        if not candidates:
            break
        u, v = rng.choice(candidates)
        edge_set.add((u, v))
        edges.append((u, v))
        degrees[u] += 1
        degrees[v] += 1

    span_rows = len({slot[0] for slot in slots})
    span_cols = len({slot[1] for slot in slots})
    if span_rows == 1 or span_cols == 1:
        if rng.random() < 0.7:
            return None

    orientations = set()
    for u, v in edges:
        ru, cu = slots[u]
        rv, cv = slots[v]
        orientations.add("h" if ru == rv else "v")
    if len(orientations) == 1 and len(edges) >= 3 and rng.random() < 0.6:
        return None

    return rows, cols, slots, edges


def _choose_roles(n_nodes: int, edges: list[tuple[int, int]], rng: random.Random):
    total_edges = len(edges)
    assignments: list[tuple[int, float, float, list[bool]]] = []

    for mask in range(1, (1 << n_nodes) - 1):
        senders = [bool((mask >> idx) & 1) for idx in range(n_nodes)]
        counts = [0, 0, 0]  # SS, SR, RR
        for u, v in edges:
            if senders[u] and senders[v]:
                counts[0] += 1
            elif senders[u] or senders[v]:
                counts[1] += 1
            else:
                counts[2] += 1
        kinds = sum(count > 0 for count in counts)
        balance = -sum((count - total_edges / 3.0) ** 2 for count in counts)
        sender_balance = -abs(sum(senders) - n_nodes / 2.0)
        assignments.append((kinds, balance, sender_balance, senders))

    want_kinds = 3 if total_edges >= 3 and rng.random() < 0.75 else 2
    pool = [item for item in assignments if item[0] >= want_kinds]
    if not pool:
        best_kinds = max(item[0] for item in assignments)
        pool = [item for item in assignments if item[0] == best_kinds]

    pool.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    top = pool[: min(12, len(pool))]
    return list(rng.choice(top)[3])


def _make_starts(count: int, rng: random.Random):
    starts = [rng.randint(1, 2)]
    for _ in range(count - 1):
        starts.append(starts[-1] + 3 + rng.randint(1, 4))
    size = starts[-1] + 3 + rng.randint(1, 2)
    return starts, size


def _render_input(
    rows: int,
    cols: int,
    slots: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    senders: list[bool],
    rng: random.Random,
) -> np.ndarray:
    row_starts, height = _make_starts(rows, rng)
    col_starts, width = _make_starts(cols, rng)
    assert height <= 30 and width <= 30

    colors = rng.sample(NODE_COLORS, len(slots))
    grid = np.full((height, width), BG, dtype=int)
    top_lefts: dict[int, tuple[int, int]] = {}

    for node_id, (slot_r, slot_c) in enumerate(slots):
        r0 = row_starts[slot_r]
        c0 = col_starts[slot_c]
        top_lefts[node_id] = (r0, c0)
        color = colors[node_id]
        grid[r0 : r0 + 3, c0 : c0 + 3] = color
        if senders[node_id]:
            grid[r0 + 1, c0 + 1] = BG

    for u, v in edges:
        ru, cu = slots[u]
        rv, cv = slots[v]
        r0u, c0u = top_lefts[u]
        r0v, c0v = top_lefts[v]
        if ru == rv:
            if c0u > c0v:
                r0u, c0u, r0v, c0v = r0v, c0v, r0u, c0u
            row = r0u + 1
            grid[row, c0u + 3 : c0v] = GREY
        else:
            if r0u > r0v:
                r0u, c0u, r0v, c0v = r0v, c0v, r0u, c0u
            col = c0u + 1
            grid[r0u + 3 : r0v, col] = GREY

    return grid


def _path_type_summary(grid: np.ndarray):
    nodes, owner = _detect_nodes(grid)
    summary = defaultdict(int)
    for path in _extract_paths(grid, owner):
        a_sender = bool(nodes[path["a"]]["sender"])
        b_sender = bool(nodes[path["b"]]["sender"])
        length = int(path["coords"].shape[0])
        orientation = str(path["orientation"])
        if a_sender and b_sender:
            summary["SS"] += 1
            summary[f"SS_{orientation}_{'odd' if length % 2 else 'even'}"] += 1
        elif a_sender or b_sender:
            summary["SR"] += 1
        else:
            summary["RR"] += 1
    return summary


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    built = _build_graph(rng)
    if built is None:
        return None
    rows, cols, slots, edges = built
    senders = _choose_roles(len(slots), edges, rng)
    grid = _render_input(rows, cols, slots, edges, senders, rng)

    if grid.ndim != 2 or max(grid.shape) > 30:
        return None
    if not np.any(grid == GREY):
        return None

    summary = _path_type_summary(grid)
    if len(edges) >= 3 and sum(summary[k] > 0 for k in ("SS", "SR", "RR")) < 2:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None

    if not np.any(out != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid node-network puzzle")
