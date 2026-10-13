import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
NODE_SIZE = 2
VISIBLE_COLORS = tuple(color for color in range(1, 10) if color != GREY)
INPUT_SIDES = {"left", "top"}
OUTPUT_SIDES = {"right", "bottom"}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
ROW_COUNT_RANGE = (3, 5)
COL_COUNT_RANGE = (4, 6)


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _touch_side(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    r, c = cell
    if c == c0 - 1 and r0 <= r <= r1:
        return "left"
    if c == c1 + 1 and r0 <= r <= r1:
        return "right"
    if r == r0 - 1 and c0 <= c <= c1:
        return "top"
    if r == r1 + 1 and c0 <= c <= c1:
        return "bottom"
    raise AssertionError(f"cell {cell} does not touch bbox {bbox}")


def _adjacent_node_ids(cell: tuple[int, int], node_id_grid: np.ndarray) -> list[int]:
    h, w = node_id_grid.shape
    r, c = cell
    out: list[int] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            node_id = int(node_id_grid[nr, nc])
            if node_id >= 0 and node_id not in out:
                out.append(node_id)
    return out


def _parse_legend(grid: np.ndarray) -> tuple[dict[frozenset[int], int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    c0 = w - 5
    rows: list[int] = []
    for r in range(h - 1, -1, -1):
        row = g[r, c0:w]
        if (
            row.shape[0] == 5
            and int(row[1]) == GREY
            and int(row[3]) == GREY
            and int(row[0]) not in (BLACK, GREY)
            and int(row[2]) not in (BLACK, GREY)
            and int(row[4]) not in (BLACK, GREY)
        ):
            rows.append(r)
        elif rows:
            break
    assert rows, "legend missing"
    rows.sort()

    legend_mask = np.zeros_like(g, dtype=bool)
    mix_map: dict[frozenset[int], int] = {}
    for r in rows:
        legend_mask[r, c0:w] = True
        a = int(g[r, c0])
        b = int(g[r, c0 + 2])
        result = int(g[r, c0 + 4])
        key = frozenset((a, b))
        if key in mix_map:
            assert mix_map[key] == result
        else:
            mix_map[key] = result
    return mix_map, legend_mask


def _parse_nodes(grid: np.ndarray, legend_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    mask = (g != BLACK) & (g != GREY) & ~legend_mask
    comps = _components(mask)
    assert comps, "no nodes found"

    nodes: list[dict] = []
    node_id_grid = np.full(g.shape, -1, dtype=int)
    for comp in sorted(comps, key=lambda item: item["bbox"]):
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["area"] == 4
        assert r1 - r0 + 1 == NODE_SIZE and c1 - c0 + 1 == NODE_SIZE
        color = int(g[r0, c0])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color)
        node_id = len(nodes)
        node_id_grid[r0 : r1 + 1, c0 : c1 + 1] = node_id
        nodes.append({"bbox": (r0, c0, r1, c1), "color": color})
    return nodes, node_id_grid


def _parse_edges(grid: np.ndarray, legend_mask: np.ndarray, nodes: list[dict], node_id_grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components((g == GREY) & ~legend_mask)
    edges: list[dict] = []
    for comp in sorted(comps, key=lambda item: item["bbox"]):
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == r1:
            pixels = [(r0, c) for c in range(c0, c1 + 1)]
        else:
            assert c0 == c1
            pixels = [(r, c0) for r in range(r0, r1 + 1)]

        start = pixels[0]
        end = pixels[-1]
        start_nodes = _adjacent_node_ids(start, node_id_grid)
        end_nodes = _adjacent_node_ids(end, node_id_grid)
        assert len(start_nodes) == 1 and len(end_nodes) == 1
        start_node = start_nodes[0]
        end_node = end_nodes[0]
        assert start_node != end_node

        start_side = _touch_side(nodes[start_node]["bbox"], start)
        end_side = _touch_side(nodes[end_node]["bbox"], end)
        if start_side in OUTPUT_SIDES and end_side in INPUT_SIDES:
            src = start_node
            dst = end_node
            dir_pixels = pixels
        elif end_side in OUTPUT_SIDES and start_side in INPUT_SIDES:
            src = end_node
            dst = start_node
            dir_pixels = list(reversed(pixels))
        else:
            # Fallback for degenerate same-depth edges: keep the geometric order.
            src = start_node
            dst = end_node
            dir_pixels = pixels

        edges.append(
            {
                "start_node": start_node,
                "end_node": end_node,
                "src": src,
                "dst": dst,
                "pixels_ordered": pixels,
                "pixels_dir": dir_pixels,
            }
        )
    return edges


def _unique_colors(colors: list[int]) -> list[int]:
    out: list[int] = []
    for color in colors:
        if color not in out:
            out.append(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 5 and g.shape[1] >= 8

    mix_map, legend_mask = _parse_legend(g)
    nodes, node_id_grid = _parse_nodes(g, legend_mask)
    edges = _parse_edges(g, legend_mask, nodes, node_id_grid)

    incoming: list[list[int]] = [[] for _ in nodes]
    for edge_id, edge in enumerate(edges):
        incoming[edge["dst"]].append(edge_id)

    order = sorted(range(len(nodes)), key=lambda idx: nodes[idx]["bbox"][:2])
    rank = {node_id: pos for pos, node_id in enumerate(order)}
    for edge in edges:
        assert rank[edge["src"]] < rank[edge["dst"]]

    node_colors: list[int | None] = [None] * len(nodes)
    for node_id in order:
        if not incoming[node_id]:
            node_colors[node_id] = int(nodes[node_id]["color"])
            continue

        input_colors = [int(node_colors[edges[edge_id]["src"]]) for edge_id in incoming[node_id]]
        distinct = _unique_colors(input_colors)
        if len(distinct) == 1:
            node_colors[node_id] = distinct[0]
        else:
            assert len(distinct) == 2
            pair = frozenset(distinct)
            assert pair in mix_map
            node_colors[node_id] = int(mix_map[pair])

    out = g.copy()
    is_root = [not incoming[node_id] for node_id in range(len(nodes))]
    for node_id, node in enumerate(nodes):
        r0, c0, r1, c1 = node["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(node_colors[node_id])

    for edge in edges:
        if is_root[edge["start_node"]] and is_root[edge["end_node"]]:
            left_top_color = int(node_colors[edge["start_node"]])
            right_bottom_color = int(node_colors[edge["end_node"]])
            split = (len(edge["pixels_ordered"]) + 1) // 2
            colors = [left_top_color] * split + [right_bottom_color] * (len(edge["pixels_ordered"]) - split)
            for (r, c), color in zip(edge["pixels_ordered"], colors):
                out[r, c] = color
        else:
            color = int(node_colors[edge["src"]])
            for r, c in edge["pixels_dir"]:
                out[r, c] = color
    return out


def _build_graph() -> tuple[
    set[tuple[int, int]],
    dict[tuple[int, int], list[tuple[int, int]]],
    tuple[tuple[int, int], tuple[int, int]],
] | None:
    rows = random.randint(*ROW_COUNT_RANGE)
    cols = random.randint(*COL_COUNT_RANGE)
    if rows < 2 or cols < 3:
        return None

    base_r = random.randint(0, rows - 2)
    base_c = random.randint(0, cols - 3)
    a = (base_r, base_c)
    b = (base_r + 1, base_c)
    c = (base_r, base_c + 1)
    d = (base_r + 1, base_c + 1)
    e = (base_r + 1, base_c + 2)

    occupied = {a, b, c, d, e}
    parents: dict[tuple[int, int], list[tuple[int, int]]] = {
        a: [],
        b: [],
        c: [a],
        d: [b, c],
        e: [d],
    }

    for r in range(rows):
        for col in range(cols):
            cell = (r, col)
            if cell in occupied:
                continue
            parent_options: list[tuple[int, int]] = []
            left = (r, col - 1)
            up = (r - 1, col)
            if left in occupied:
                parent_options.append(left)
            if up in occupied:
                parent_options.append(up)

            occupy_prob = 0.55 if parent_options else 0.15
            if random.random() >= occupy_prob:
                continue

            if parent_options and random.random() < 0.78:
                if len(parent_options) == 2 and random.random() < 0.5:
                    chosen = parent_options[:]
                else:
                    chosen = [random.choice(parent_options)]
            else:
                chosen = []

            occupied.add(cell)
            parents[cell] = chosen

    if not (6 <= len(occupied) <= 13):
        return None
    return occupied, parents, (a, b)


def _assign_colors(
    occupied: set[tuple[int, int]],
    parents: dict[tuple[int, int], list[tuple[int, int]]],
    motif_roots: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
    dict[frozenset[int], int],
    int,
] | None:
    roots = [cell for cell in occupied if not parents[cell]]
    if len(roots) < 2:
        return None

    non_azure = [color for color in VISIBLE_COLORS if color != AZURE]
    signal_colors = [AZURE, random.choice(non_azure)]
    if random.random() < 0.5:
        extra_choices = [color for color in non_azure if color not in signal_colors]
        signal_colors.append(random.choice(extra_choices))

    placeholder_choices = [color for color in VISIBLE_COLORS if color not in signal_colors]
    placeholder = random.choice(placeholder_choices)
    result_pool = [color for color in VISIBLE_COLORS if color not in signal_colors and color != placeholder]

    root_colors: dict[tuple[int, int], int] = {}
    root_colors[motif_roots[0]] = signal_colors[0]
    root_colors[motif_roots[1]] = signal_colors[1]
    for cell in roots:
        if cell not in root_colors:
            root_colors[cell] = random.choice(signal_colors)

    output_colors: dict[tuple[int, int], int] = {}
    mix_rules: dict[frozenset[int], int] = {}
    has_single = False
    has_mix = False
    has_chain = False
    has_h = False
    has_v = False

    for cell in sorted(occupied):
        cell_parents = parents[cell]
        if not cell_parents:
            output_colors[cell] = root_colors[cell]
            continue

        if any(parents[parent] for parent in cell_parents):
            has_chain = True
        for parent in cell_parents:
            if parent[0] == cell[0]:
                has_h = True
            if parent[1] == cell[1]:
                has_v = True

        parent_colors = [output_colors[parent] for parent in cell_parents]
        distinct = _unique_colors(parent_colors)
        if len(distinct) == 1:
            has_single = True
            output_colors[cell] = distinct[0]
            continue

        if len(distinct) != 2:
            return None

        has_mix = True
        pair = frozenset(distinct)
        if pair not in mix_rules:
            candidates = [color for color in result_pool if color not in distinct and color not in mix_rules.values()]
            if not candidates:
                return None
            mix_rules[pair] = random.choice(candidates)
        output_colors[cell] = mix_rules[pair]

    if len(mix_rules) == 0 or len(mix_rules) > 3:
        return None
    if not (has_single and has_mix and has_chain and has_h and has_v):
        return None
    return root_colors, output_colors, mix_rules, placeholder


def _sample_positions(rows: int, cols: int) -> tuple[list[int], list[int]]:
    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)

    row_pos = [top_margin]
    for _ in range(1, rows):
        row_pos.append(row_pos[-1] + random.randint(4, 6))

    col_pos = [left_margin]
    for _ in range(1, cols):
        col_pos.append(col_pos[-1] + random.randint(4, 6))

    return row_pos, col_pos


def _build_layout(
    occupied: set[tuple[int, int]],
    parents: dict[tuple[int, int], list[tuple[int, int]]],
    mix_rules: dict[frozenset[int], int],
) -> dict:
    rows = max(cell[0] for cell in occupied) + 1
    cols = max(cell[1] for cell in occupied) + 1
    row_pos, col_pos = _sample_positions(rows, cols)

    network_bottom = row_pos[-1] + NODE_SIZE
    network_right = col_pos[-1] + NODE_SIZE
    legend_rows = len(mix_rules)
    h = network_bottom + random.randint(1, 3) + legend_rows
    w = network_right + random.randint(2, 4) + 5
    attachments: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for cell in sorted(occupied):
        for parent in parents[cell]:
            attachments[(parent, cell)] = random.randint(0, 1)

    legend_items = list(mix_rules.items())
    random.shuffle(legend_items)
    legend_rows_data: list[tuple[int, int, int]] = []
    for pair, result in legend_items:
        a, b = tuple(pair)
        if random.random() < 0.5:
            a, b = b, a
        legend_rows_data.append((a, b, result))

    return {
        "row_pos": row_pos,
        "col_pos": col_pos,
        "shape": (h, w),
        "legend_rows": legend_rows_data,
        "attachments": attachments,
    }


def _render_scene(
    occupied: set[tuple[int, int]],
    parents: dict[tuple[int, int], list[tuple[int, int]]],
    node_colors: dict[tuple[int, int], int],
    mix_rules: dict[frozenset[int], int],
    placeholder: int,
    layout: dict,
    output_mode: bool,
) -> np.ndarray:
    h, w = layout["shape"]
    row_pos = layout["row_pos"]
    col_pos = layout["col_pos"]
    legend_rows_data = layout["legend_rows"]
    attachments = layout["attachments"]
    legend_c0 = w - 5
    legend_r0 = h - len(legend_rows_data)

    grid = np.full((h, w), BLACK, dtype=int)
    for cell in occupied:
        r, c = cell
        rr = row_pos[r]
        cc = col_pos[c]
        color = node_colors[cell] if (output_mode or not parents[cell]) else placeholder
        grid[rr : rr + NODE_SIZE, cc : cc + NODE_SIZE] = color

    for child in sorted(occupied):
        for parent in parents[child]:
            pr, pc = parent
            cr, cc = child
            parent_r = row_pos[pr]
            parent_c = col_pos[pc]
            child_r = row_pos[cr]
            child_c = col_pos[cc]
            track = attachments[(parent, child)]
            edge_color = node_colors[parent] if output_mode else GREY
            if pr == cr:
                rr = parent_r + track
                grid[rr, parent_c + NODE_SIZE : child_c] = edge_color
            else:
                assert pc == cc
                cc2 = parent_c + track
                grid[parent_r + NODE_SIZE : child_r, cc2] = edge_color

    for idx, (a, b, result) in enumerate(legend_rows_data):
        r = legend_r0 + idx
        grid[r, legend_c0 : legend_c0 + 5] = np.array([a, GREY, b, GREY, result], dtype=int)
    return grid


def _gen_attempt() -> np.ndarray | None:
    graph = _build_graph()
    if graph is None:
        return None
    occupied, parents, motif_roots = graph

    colors = _assign_colors(occupied, parents, motif_roots)
    if colors is None:
        return None
    root_colors, output_colors, mix_rules, placeholder = colors

    layout = _build_layout(occupied, parents, mix_rules)
    inp = _render_scene(occupied, parents, root_colors, mix_rules, placeholder, layout, output_mode=False)
    expected = _render_scene(occupied, parents, output_colors, mix_rules, placeholder, layout, output_mode=True)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
