from __future__ import annotations

import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG_COLOR = 0
GAP_RANGE = (3, 7)
GEN_RETRIES = 400
BRANCHES = ("odd", "even_diff", "even_equal")

TEMPLATES = {
    "pair": {
        "coords": [(0, 0), (0, 1)],
        "edges": [(0, 1)],
    },
    "path3": {
        "coords": [(0, 0), (0, 1), (0, 2)],
        "edges": [(0, 1), (1, 2)],
    },
    "turn3": {
        "coords": [(0, 0), (0, 1), (1, 1)],
        "edges": [(0, 1), (1, 2)],
    },
    "path4": {
        "coords": [(0, 0), (0, 1), (0, 2), (0, 3)],
        "edges": [(0, 1), (1, 2), (2, 3)],
    },
    "turn4": {
        "coords": [(0, 0), (0, 1), (1, 1), (2, 1)],
        "edges": [(0, 1), (1, 2), (2, 3)],
    },
    "zigzag4": {
        "coords": [(0, 0), (0, 1), (1, 1), (1, 2)],
        "edges": [(0, 1), (1, 2), (2, 3)],
    },
    "t4": {
        "coords": [(0, 1), (1, 0), (1, 1), (1, 2)],
        "edges": [(0, 2), (1, 2), (2, 3)],
    },
    "square4": {
        "coords": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "edges": [(0, 1), (0, 2), (1, 3), (2, 3)],
    },
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_records(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            comps.append(
                {
                    "color": int(color),
                    "cells": sorted(cells),
                    "size": len(cells),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                }
            )
    return comps


def _all_foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps.extend(_component_records(grid, color))
    for idx, comp in enumerate(comps):
        comp["id"] = int(idx)
    return comps


def _extract_source_components(components: list[dict]) -> list[dict]:
    candidates = [
        comp
        for comp in components
        if comp["size"] == 2 and comp["height"] == 1 and comp["width"] == 2
    ]
    assert candidates, "expected at least one 1x2 energy source"

    groups: dict[int, list[dict]] = {}
    for comp in candidates:
        c0 = int(comp["bbox"][1])
        groups.setdefault(c0, []).append(comp)

    best_key = None
    best_group: list[dict] | None = None
    for c0, group in groups.items():
        rows = sorted(int(comp["bbox"][0]) for comp in group)
        if len(rows) != len(set(rows)):
            continue
        score = (len(group), -int(c0))
        if best_key is None or score > best_key:
            best_key = score
            best_group = sorted(group, key=lambda comp: int(comp["bbox"][0]))

    assert best_group is not None, "failed to locate the stacked energy sources"
    return best_group


def _component_set(components: list[dict]) -> set[int]:
    return {int(comp["id"]) for comp in components}


def _choose_midpoint_index(edge: dict, nodes: list[dict], degrees: dict[int, int]) -> int:
    length = len(edge["cells"])
    assert length >= 1, "wire must contain at least one grey pixel"
    if length % 2 == 1:
        return length // 2

    u = int(edge["u"])
    v = int(edge["v"])
    du = int(degrees[u])
    dv = int(degrees[v])
    if du < dv:
        owner = u
    elif dv < du:
        owner = v
    elif edge["orientation"] == "h":
        owner = u if int(nodes[u]["c0"]) < int(nodes[v]["c0"]) else v
    else:
        owner = u if int(nodes[u]["r0"]) < int(nodes[v]["r0"]) else v
    return length // 2 - 1 if owner == u else length // 2


def _paint_output(
    grid: np.ndarray,
    nodes: list[dict],
    edges: list[dict],
    degree_to_color: dict[int, int],
    wire_color: int,
    degrees: dict[int, int],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for node in nodes:
        node_color = int(degree_to_color[int(degrees[int(node["id"])])])
        r0 = int(node["r0"])
        c0 = int(node["c0"])
        out[r0 : r0 + 2, c0 : c0 + 2] = node_color

    for edge in edges:
        u = int(edge["u"])
        v = int(edge["v"])
        color_u = int(degree_to_color[int(degrees[u])])
        color_v = int(degree_to_color[int(degrees[v])])
        midpoint = _choose_midpoint_index(edge, nodes, degrees)
        for idx, (r, c) in enumerate(edge["cells"]):
            if idx < midpoint:
                out[r, c] = color_u
            elif idx > midpoint:
                out[r, c] = color_v
            else:
                out[r, c] = int(wire_color)
    return out


def _parse_nodes_and_edges(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, int], int, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _all_foreground_components(g, bg)
    source_components = _extract_source_components(components)
    source_ids = _component_set(source_components)

    node_components = [
        comp
        for comp in components
        if comp["id"] not in source_ids
        and comp["size"] == 4
        and comp["height"] == 2
        and comp["width"] == 2
    ]
    assert len(node_components) >= 2, "expected at least two 2x2 nodes"
    node_ids = _component_set(node_components)

    leftover_colors = {
        int(comp["color"])
        for comp in components
        if comp["id"] not in source_ids and comp["id"] not in node_ids
    }
    assert len(leftover_colors) == 1, "expected a unique grey wire color"
    wire_color = int(next(iter(leftover_colors)))

    degree_to_color = {
        idx + 1: int(comp["color"])
        for idx, comp in enumerate(sorted(source_components, key=lambda comp: int(comp["bbox"][0])))
    }

    nodes: list[dict] = []
    for idx, comp in enumerate(sorted(node_components, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))):
        r0, c0, _, _ = comp["bbox"]
        nodes.append(
            {
                "id": int(idx),
                "r0": int(r0),
                "c0": int(c0),
                "cells": list(comp["cells"]),
            }
        )

    cell_to_node: dict[tuple[int, int], int] = {}
    for node in nodes:
        for cell in node["cells"]:
            cell_to_node[cell] = int(node["id"])

    h, w = g.shape
    stubs_by_node: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    degrees: dict[int, int] = {}
    for node in nodes:
        node_id = int(node["id"])
        stubs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        seen_stub_cells: set[tuple[int, int]] = set()
        node_cells = set(node["cells"])
        for r, c in node["cells"]:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in node_cells and int(g[nr, nc]) == wire_color:
                    if (nr, nc) not in seen_stub_cells:
                        seen_stub_cells.add((nr, nc))
                        stubs.append(((int(nr), int(nc)), (int(dr), int(dc))))
        stubs_by_node[node_id] = stubs
        degrees[node_id] = len(stubs)

    assert degrees, "expected nonempty node set"
    assert max(degrees.values()) <= len(degree_to_color), "missing source color for some degree"

    edges: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()
    incident_counts = {int(node["id"]): 0 for node in nodes}
    for node in nodes:
        node_id = int(node["id"])
        for (sr, sc), (dr, dc) in stubs_by_node[node_id]:
            cells: list[tuple[int, int]] = []
            cr = sr
            cc = sc
            while 0 <= cr < h and 0 <= cc < w and int(g[cr, cc]) == wire_color:
                cells.append((int(cr), int(cc)))
                cr += dr
                cc += dc
            assert (int(cr), int(cc)) in cell_to_node, "wire must terminate at another node"
            other_id = int(cell_to_node[(int(cr), int(cc))])
            assert other_id != node_id, "wire cannot loop back into the same node"
            pair = tuple(sorted((node_id, other_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            incident_counts[node_id] += 1
            incident_counts[other_id] += 1
            edges.append(
                {
                    "u": int(node_id),
                    "v": int(other_id),
                    "cells": cells,
                    "orientation": "h" if dr == 0 else "v",
                }
            )

    assert sum(incident_counts.values()) == 2 * len(edges), "bad edge accounting"
    assert incident_counts == degrees, "node degree must match the number of traced wires"
    return nodes, edges, degrees, wire_color, degree_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 2x2 node by its degree and spread that color halfway along each grey wire.
    """
    g = np.asarray(grid, dtype=int)
    nodes, edges, degrees, wire_color, degree_to_color = _parse_nodes_and_edges(g)
    return _paint_output(g, nodes, edges, degree_to_color, wire_color, degrees)


def _graph_degrees(node_count: int, edges: list[tuple[int, int]]) -> dict[int, int]:
    degrees = {idx: 0 for idx in range(node_count)}
    for a, b in edges:
        degrees[int(a)] += 1
        degrees[int(b)] += 1
    return degrees


def _transformed_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    )
    transform = random.choice(transforms)
    mapped = [transform(r, c) for r, c in coords]
    min_r = min(r for r, _ in mapped)
    min_c = min(c for _, c in mapped)
    return [(int(r - min_r), int(c - min_c)) for r, c in mapped]


def _eligible_templates(branch: str) -> list[str]:
    eligible = []
    for name, spec in TEMPLATES.items():
        degrees = _graph_degrees(len(spec["coords"]), spec["edges"])
        good = False
        for a, b in spec["edges"]:
            same = degrees[int(a)] == degrees[int(b)]
            if branch == "odd":
                good = True
            elif branch == "even_diff" and not same:
                good = True
            elif branch == "even_equal" and same:
                good = True
            if good:
                break
        if good:
            eligible.append(name)
    return eligible


def _pick_forced_edge(
    branch: str,
    coords: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    degrees: dict[int, int],
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for a, b in edges:
        same = degrees[int(a)] == degrees[int(b)]
        if branch == "odd":
            candidates.append((int(a), int(b)))
        elif branch == "even_diff" and not same:
            candidates.append((int(a), int(b)))
        elif branch == "even_equal" and same:
            candidates.append((int(a), int(b)))
    assert candidates, "selected template does not support the requested branch"
    return random.choice(candidates)


def _random_gap(parity: str | None = None) -> int:
    values = [value for value in range(GAP_RANGE[0], GAP_RANGE[1] + 1) if parity is None or value % 2 == (0 if parity == "even" else 1)]
    return int(random.choice(values))


def _actual_positions(levels: list[int], start: int, gaps: list[int]) -> dict[int, int]:
    pos = {int(levels[0]): int(start)}
    for idx in range(len(levels) - 1):
        pos[int(levels[idx + 1])] = int(pos[int(levels[idx])] + 2 + gaps[idx])
    return pos


def _build_scene(template_name: str, branch: str) -> tuple[np.ndarray, np.ndarray]:
    spec = TEMPLATES[template_name]
    coords = _transformed_coords(spec["coords"])
    edges = [(int(a), int(b)) for a, b in spec["edges"]]
    node_count = len(coords)
    degrees = _graph_degrees(node_count, edges)
    forced_edge = _pick_forced_edge(branch, coords, edges, degrees)

    row_levels = sorted({int(r) for r, _ in coords})
    col_levels = sorted({int(c) for _, c in coords})
    row_gaps = [_random_gap() for _ in range(max(0, len(row_levels) - 1))]
    col_gaps = [_random_gap() for _ in range(max(0, len(col_levels) - 1))]

    fa, fb = forced_edge
    ra, ca = coords[fa]
    rb, cb = coords[fb]
    forced_parity = "odd" if branch == "odd" else "even"
    if ra == rb:
        gap_index = min(ca, cb)
        col_gaps[gap_index] = _random_gap(forced_parity)
    else:
        gap_index = min(ra, rb)
        row_gaps[gap_index] = _random_gap(forced_parity)

    scene_top = random.randint(1, 4)
    scene_left = random.randint(6, 10)
    row_pos = _actual_positions(row_levels, scene_top, row_gaps)
    col_pos = _actual_positions(col_levels, scene_left, col_gaps)

    source_count = max(degrees.values())
    wire_color = int(random.choice([color for color in range(1, 10)]))
    palette = [color for color in range(1, 10) if color != wire_color]
    random.shuffle(palette)
    source_colors = palette[:source_count]
    node_colors = palette[source_count : source_count + node_count]

    source_top = random.randint(1, 4)
    source_col = 1
    network_bottom = max(row_pos[int(r)] + 2 for r, _ in coords)
    network_right = max(col_pos[int(c)] + 2 for _, c in coords)
    source_bottom = source_top + 2 * (source_count - 1) + 1
    height = max(network_bottom + random.randint(1, 4), source_bottom + random.randint(1, 4))
    width = network_right + random.randint(1, 4)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    degree_to_color = {degree: int(source_colors[degree - 1]) for degree in range(1, source_count + 1)}

    for degree in range(1, source_count + 1):
        sr = source_top + 2 * (degree - 1)
        grid[sr, source_col : source_col + 2] = int(degree_to_color[degree])

    nodes: list[dict] = []
    for node_id, (r_level, c_level) in enumerate(coords):
        r0 = int(row_pos[int(r_level)])
        c0 = int(col_pos[int(c_level)])
        grid[r0 : r0 + 2, c0 : c0 + 2] = int(node_colors[node_id])
        nodes.append(
            {
                "id": int(node_id),
                "r0": int(r0),
                "c0": int(c0),
                "cells": [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)],
            }
        )

    rendered_edges: list[dict] = []
    for a, b in edges:
        ra, ca = coords[a]
        rb, cb = coords[b]
        node_a = nodes[int(a)]
        node_b = nodes[int(b)]
        if ra == rb:
            line_row = int(node_a["r0"] + random.randint(0, 1))
            if int(node_a["c0"]) < int(node_b["c0"]):
                cells_lr = [(line_row, c) for c in range(int(node_a["c0"]) + 2, int(node_b["c0"]))]
                cells = cells_lr
            else:
                cells_lr = [(line_row, c) for c in range(int(node_b["c0"]) + 2, int(node_a["c0"]))]
                cells = list(reversed(cells_lr))
            for r, c in cells_lr:
                grid[r, c] = int(wire_color)
            rendered_edges.append({"u": int(a), "v": int(b), "cells": cells, "orientation": "h"})
        else:
            line_col = int(node_a["c0"] + random.randint(0, 1))
            if int(node_a["r0"]) < int(node_b["r0"]):
                cells_tb = [(r, line_col) for r in range(int(node_a["r0"]) + 2, int(node_b["r0"]))]
                cells = cells_tb
            else:
                cells_tb = [(r, line_col) for r in range(int(node_b["r0"]) + 2, int(node_a["r0"]))]
                cells = list(reversed(cells_tb))
            for r, c in cells_tb:
                grid[r, c] = int(wire_color)
            rendered_edges.append({"u": int(a), "v": int(b), "cells": cells, "orientation": "v"})

    expected = _paint_output(grid, nodes, rendered_edges, degree_to_color, wire_color, degrees)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        branch = random.choice(BRANCHES)
        template_name = random.choice(_eligible_templates(branch))
        grid, expected = _build_scene(template_name, branch)
        result = main(grid)
        if np.array_equal(result, expected):
            return grid
    raise RuntimeError("failed to generate a valid circuit scene")
