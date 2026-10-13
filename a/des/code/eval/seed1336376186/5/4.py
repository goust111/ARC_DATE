from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
ARROW = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)

# --------------- shared helpers ---------------


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _monochromatic_components(
    grid: np.ndarray,
    *,
    ignore_colors: set[int],
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in ignore_colors:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return comps


def _resolve_color(colors: list[int], priority: list[int]) -> int:
    assert colors
    if all(int(c) == int(colors[0]) for c in colors):
        return int(colors[0])

    rank = {int(color): idx for idx, color in enumerate(priority)}
    ordered_unique: list[int] = []
    for color in colors:
        color = int(color)
        if color not in ordered_unique:
            ordered_unique.append(color)
    return min(ordered_unique, key=lambda color: rank.get(int(color), len(rank)))


def _propagate_colors(
    nodes: Iterable[object],
    original_colors: dict[object, int],
    edges: list[tuple[object, object]],
    priority: list[int],
    order_key,
) -> dict[object, int]:
    node_list = list(nodes)
    preds = {node: [] for node in node_list}
    succs = {node: [] for node in node_list}
    indeg = {node: 0 for node in node_list}

    for src, dst in edges:
        succs[src].append(dst)
        preds[dst].append(src)
        indeg[dst] += 1

    ready = sorted([node for node in node_list if indeg[node] == 0], key=order_key)
    final: dict[object, int] = {}

    while ready:
        node = ready.pop(0)
        if preds[node]:
            final[node] = _resolve_color([final[src] for src in preds[node]], priority)
        else:
            final[node] = int(original_colors[node])

        for dst in succs[node]:
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
        ready.sort(key=order_key)

    if len(final) != len(node_list):
        raise ValueError("graph must be acyclic")

    return final


# --------------- solver ---------------


def _extract_priority_list(singletons: list[dict], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    by_col: dict[int, list[tuple[int, int]]] = {}
    for comp in singletons:
        (r, c) = comp["coords"][0]
        by_col.setdefault(int(c), []).append((int(r), int(comp["color"])))

    candidates: list[tuple[int, int, int, list[int]]] = []
    max_left_col = max(2, w // 4)
    min_bottom_row = h // 2
    for c, items in by_col.items():
        items.sort()
        rows = [row for row, _color in items]
        if len(rows) < 2:
            continue
        if c > max_left_col or rows[-1] < min_bottom_row:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        colors = [color for _row, color in items]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((int(c), -len(colors), -rows[-1], colors))

    if not candidates:
        raise ValueError("priority list not found")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [int(color) for color in candidates[0][3]]


def _parse_graph(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    color_components = _monochromatic_components(g, ignore_colors={BG, GREY, ARROW})
    nodes: list[dict] = []
    singletons: list[dict] = []
    for comp in color_components:
        r0, c0, r1, c1 = comp["bbox"]
        if int(comp["area"]) == 4 and r1 - r0 == 1 and c1 - c0 == 1:
            nodes.append(comp)
        elif int(comp["area"]) == 1:
            singletons.append(comp)

    if not nodes:
        raise ValueError("nodes not found")

    nodes.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    for idx, node in enumerate(nodes):
        node["id"] = idx

    priority = _extract_priority_list(singletons, (h, w))

    node_by_cell: dict[tuple[int, int], int] = {}
    for node in nodes:
        for r, c in node["coords"]:
            node_by_cell[(int(r), int(c))] = int(node["id"])

    grey_components = _monochromatic_components(g, ignore_colors=set(PALETTE) | {BG, ARROW})
    edges: list[tuple[int, int]] = []

    def adjacent_node_id(cell: tuple[int, int]) -> int:
        ids = {
            int(node_by_cell[(nr, nc)])
            for nr, nc in _neighbors4(cell[0], cell[1], h, w)
            if (nr, nc) in node_by_cell
        }
        if len(ids) != 1:
            raise ValueError("line endpoint must touch exactly one node")
        return ids.pop()

    for comp in grey_components:
        if int(comp["color"]) != GREY:
            continue
        r0, c0, r1, c1 = comp["bbox"]

        if r0 == r1:
            r = int(r0)
            left_node = adjacent_node_id((r, int(c0)))
            right_node = adjacent_node_id((r, int(c1)))
            arrows = {
                (rr, cc)
                for cc in range(int(c0), int(c1) + 1)
                for rr in (r - 1, r + 1)
                if 0 <= rr < h and int(g[rr, cc]) == ARROW
            }
            if len(arrows) != 1:
                raise ValueError("horizontal line must have one arrow marker")
            arrow_r, _arrow_c = next(iter(arrows))
            if int(arrow_r) < r:
                edges.append((left_node, right_node))
            else:
                edges.append((right_node, left_node))
        elif c0 == c1:
            c = int(c0)
            top_node = adjacent_node_id((int(r0), c))
            bottom_node = adjacent_node_id((int(r1), c))
            arrows = {
                (rr, cc)
                for rr in range(int(r0), int(r1) + 1)
                for cc in (c - 1, c + 1)
                if 0 <= cc < w and int(g[rr, cc]) == ARROW
            }
            if len(arrows) != 1:
                raise ValueError("vertical line must have one arrow marker")
            _arrow_r, arrow_c = next(iter(arrows))
            if int(arrow_c) < c:
                edges.append((top_node, bottom_node))
            else:
                edges.append((bottom_node, top_node))
        else:
            raise ValueError("grey component must be straight")

    return nodes, edges, priority


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate node colors through a directed circuit of grey wires.

    Input structure:
    - Black background.
    - 2x2 monochrome nodes in visible colors.
    - Straight grey line segments connect pairs of nodes.
    - A maroon side-marker on each line encodes the signal direction.
    - A vertical priority list of single colored pixels sits in the lower-left area.

    Rule:
    - Source nodes keep their original colors.
    - Every other node gathers the colors of all predecessor nodes.
    - If all predecessor colors agree, the node becomes that color.
    - Otherwise, the highest-priority color from the lower-left legend wins.
    - Wires and arrows remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    nodes, edges, priority = _parse_graph(g)

    original_colors = {int(node["id"]): int(node["color"]) for node in nodes}
    order_key = lambda node_id: (
        int(nodes[int(node_id)]["bbox"][0]),
        int(nodes[int(node_id)]["bbox"][1]),
    )
    final_colors = _propagate_colors(
        [int(node["id"]) for node in nodes],
        original_colors,
        edges,
        priority,
        order_key,
    )

    out = g.copy()
    for node in nodes:
        color = int(final_colors[int(node["id"])])
        for r, c in node["coords"]:
            out[int(r), int(c)] = color
    return out


# --------------- generator ---------------


CORE_SLOTS = {
    "s2": (0, 1),
    "s1": (1, 0),
    "u": (1, 1),
    "x": (1, 2),
    "s3": (2, 1),
    "c": (2, 2),
    "t": (2, 3),
    "i": (0, 0),
}

CORE_EDGES = [
    ("s1", "u"),
    ("s2", "u"),
    ("u", "x"),
    ("s3", "c"),
    ("x", "c"),
]

GEN_ATTEMPTS = 400
PRIORITY_COL = 1
ROW_GAP_RANGE = (5, 7)
COL_GAP_RANGE = (5, 7)
TOP_MARGIN_RANGE = (1, 3)
LEFT_MARGIN_RANGE = (4, 6)
RIGHT_MARGIN_RANGE = (2, 4)
BOTTOM_MARGIN_RANGE = (2, 4)
PRIORITY_LEN_CHOICES = (4, 5)
EXTRA_SINK_PROB = 0.55
ISOLATED_SOURCE_PROB = 0.5


def _slot_transform(slot: tuple[int, int], *, flip_v: bool, flip_h: bool) -> tuple[int, int]:
    r, c = slot
    if flip_v:
        r = 2 - r
    if flip_h:
        c = 3 - c
    return int(r), int(c)


def _render_node(grid: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r0, c0 = top_left
    grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)


def _edge_cells(
    src_pos: tuple[int, int],
    dst_pos: tuple[int, int],
    *,
    src_to_dst: bool,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    sr, sc = src_pos
    dr, dc = dst_pos

    if sr == dr:
        row = sr + rng.randint(0, 1)
        left_c = min(sc, dc)
        right_c = max(sc, dc)
        line = [(row, c) for c in range(left_c + 2, right_c)]
        if len(line) < 3:
            raise ValueError("horizontal gap too short")
        arrow_idx = rng.randint(1, len(line) - 2)
        arrow_base = line[arrow_idx]
        left_to_right = dc > sc
        forward = src_to_dst and left_to_right or (not src_to_dst and not left_to_right)
        arrow = (arrow_base[0] - 1, arrow_base[1]) if forward else (arrow_base[0] + 1, arrow_base[1])
        return line, arrow

    if sc == dc:
        col = sc + rng.randint(0, 1)
        top_r = min(sr, dr)
        bottom_r = max(sr, dr)
        line = [(r, col) for r in range(top_r + 2, bottom_r)]
        if len(line) < 3:
            raise ValueError("vertical gap too short")
        arrow_idx = rng.randint(1, len(line) - 2)
        arrow_base = line[arrow_idx]
        top_to_bottom = dr > sr
        forward = src_to_dst and top_to_bottom or (not src_to_dst and not top_to_bottom)
        arrow = (arrow_base[0], arrow_base[1] - 1) if forward else (arrow_base[0], arrow_base[1] + 1)
        return line, arrow

    raise ValueError("edge endpoints must share a row or column")


def _expected_from_layout(
    labels: list[str],
    original_colors: dict[str, int],
    edges: list[tuple[str, str]],
    priority: list[int],
    slot_positions: dict[str, tuple[int, int]],
) -> dict[str, int]:
    return _propagate_colors(
        labels,
        original_colors,
        edges,
        priority,
        order_key=lambda label: slot_positions[label],
    )


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    include_t = rng.random() < EXTRA_SINK_PROB
    include_i = rng.random() < ISOLATED_SOURCE_PROB
    flip_v = rng.random() < 0.5
    flip_h = rng.random() < 0.5

    priority_len = rng.choice(PRIORITY_LEN_CHOICES)
    priority = rng.sample(list(PALETTE), priority_len)
    rank = {int(color): idx for idx, color in enumerate(priority)}

    labels = ["s1", "s2", "u", "x", "s3", "c"]
    if include_t:
        labels.append("t")
    if include_i:
        labels.append("i")

    edges = list(CORE_EDGES)
    if include_t:
        edges.append(("c", "t"))

    mode = rng.randint(0, 1)
    source_colors: dict[str, int] = {}
    if mode == 0:
        shared = rng.choice(priority)
        other = rng.choice([color for color in priority if int(color) != int(shared)])
        source_colors["s1"] = int(shared)
        source_colors["s2"] = int(shared)
        source_colors["s3"] = int(other)
    else:
        a, b = rng.sample(priority, 2)
        resolved = int(a) if rank[int(a)] < rank[int(b)] else int(b)
        if rng.random() < 0.5:
            a, b = b, a
        source_colors["s1"] = int(a)
        source_colors["s2"] = int(b)
        source_colors["s3"] = int(resolved)

    if include_i:
        source_colors["i"] = int(rng.choice(priority))

    original_colors = {label: int(rng.choice(priority)) for label in labels}
    for label, color in source_colors.items():
        original_colors[label] = int(color)

    transformed_slots = {
        label: _slot_transform(CORE_SLOTS[label], flip_v=flip_v, flip_h=flip_h)
        for label in labels
    }

    row0 = rng.randint(*TOP_MARGIN_RANGE)
    col0 = rng.randint(*LEFT_MARGIN_RANGE)
    row_gaps = [rng.randint(*ROW_GAP_RANGE) for _ in range(2)]
    col_gaps = [rng.randint(*COL_GAP_RANGE) for _ in range(3)]
    row_coords = [row0]
    for gap in row_gaps:
        row_coords.append(row_coords[-1] + gap)
    col_coords = [col0]
    for gap in col_gaps:
        col_coords.append(col_coords[-1] + gap)

    node_positions = {
        label: (row_coords[slot_r], col_coords[slot_c])
        for label, (slot_r, slot_c) in transformed_slots.items()
    }

    final_colors = _expected_from_layout(labels, original_colors, edges, priority, node_positions)
    non_sources = [label for label in labels if label not in source_colors]
    if non_sources and all(int(original_colors[label]) == int(final_colors[label]) for label in non_sources):
        chosen = rng.choice(non_sources)
        alternatives = [color for color in priority if int(color) != int(final_colors[chosen])]
        original_colors[chosen] = int(rng.choice(alternatives))

    graph_bottom = row_coords[-1] + 2
    graph_right = col_coords[-1] + 2
    h = graph_bottom + rng.randint(*BOTTOM_MARGIN_RANGE)
    w = graph_right + rng.randint(*RIGHT_MARGIN_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    priority_bottom_margin = rng.randint(1, 2)
    priority_top = h - priority_bottom_margin - len(priority)
    if priority_top < 0:
        return None
    for idx, color in enumerate(priority):
        grid[priority_top + idx, PRIORITY_COL] = int(color)

    for label in labels:
        r0, c0 = node_positions[label]
        if r0 < 0 or c0 < 0 or r0 + 1 >= h or c0 + 1 >= w:
            return None
        if np.any(grid[r0 : r0 + 2, c0 : c0 + 2] != BG):
            return None
        _render_node(grid, (r0, c0), original_colors[label])

    for src, dst in edges:
        src_pos = node_positions[src]
        dst_pos = node_positions[dst]
        line_cells, arrow_cell = _edge_cells(src_pos, dst_pos, src_to_dst=True, rng=rng)

        if not (0 <= arrow_cell[0] < h and 0 <= arrow_cell[1] < w):
            return None
        if int(grid[arrow_cell]) != BG:
            return None
        for r, c in line_cells:
            if not (0 <= r < h and 0 <= c < w):
                return None
            if int(grid[r, c]) != BG:
                return None
        for r, c in line_cells:
            grid[r, c] = GREY
        grid[arrow_cell] = ARROW

    expected = grid.copy()
    for label in labels:
        r0, c0 = node_positions[label]
        expected[r0 : r0 + 2, c0 : c0 + 2] = int(final_colors[label])

    if np.array_equal(expected, grid):
        return None

    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
