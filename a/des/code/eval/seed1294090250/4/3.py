import random
from collections import deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BRIDGE_COLOR = 5
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]
NODE_SIZE_CHOICES = [3, 5]
TOPO_ROWS = [2, 3, 4]
TOPO_COLS = [2, 3, 4]
MAX_NODES = 6
GEN_ATTEMPTS = 500

ORIENTATION_MODES = ["horizontal_only", "vertical_only", "mixed"]
DISTANCE_MODES = ["adjacent_only", "long_only", "mixed"]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bridge_component_info(
    grid: np.ndarray,
    bg: int,
    color: int,
    coords: list[tuple[int, int]],
) -> dict | None:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox(coords)
    coord_set = set(coords)

    if len(coords) == 1:
        r, c = coords[0]
        left = int(grid[r, c - 1]) if c > 0 else bg
        right = int(grid[r, c + 1]) if c + 1 < w else bg
        up = int(grid[r - 1, c]) if r > 0 else bg
        down = int(grid[r + 1, c]) if r + 1 < h else bg
        if left not in (bg, color) and right not in (bg, color) and up == bg and down == bg and left != right:
            return {"coords": coords, "orientation": "h", "neighbors": (left, right)}
        if up not in (bg, color) and down not in (bg, color) and left == bg and right == bg and up != down:
            return {"coords": coords, "orientation": "v", "neighbors": (up, down)}
        return None

    if r0 == r1:
        if len(coords) != c1 - c0 + 1:
            return None
        row = r0
        left = int(grid[row, c0 - 1]) if c0 > 0 else bg
        right = int(grid[row, c1 + 1]) if c1 + 1 < w else bg
        if left in (bg, color) or right in (bg, color) or left == right:
            return None
        for c in range(c0, c1 + 1):
            if (row, c) not in coord_set:
                return None
            if row > 0 and int(grid[row - 1, c]) not in (bg, color):
                return None
            if row + 1 < h and int(grid[row + 1, c]) not in (bg, color):
                return None
        return {"coords": coords, "orientation": "h", "neighbors": (left, right)}

    if c0 == c1:
        if len(coords) != r1 - r0 + 1:
            return None
        col = c0
        up = int(grid[r0 - 1, col]) if r0 > 0 else bg
        down = int(grid[r1 + 1, col]) if r1 + 1 < h else bg
        if up in (bg, color) or down in (bg, color) or up == down:
            return None
        for r in range(r0, r1 + 1):
            if (r, col) not in coord_set:
                return None
            if col > 0 and int(grid[r, col - 1]) not in (bg, color):
                return None
            if col + 1 < w and int(grid[r, col + 1]) not in (bg, color):
                return None
        return {"coords": coords, "orientation": "v", "neighbors": (up, down)}

    return None


def _detect_bridge_color(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidates: list[tuple[int, list[dict]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        infos: list[dict] = []
        for comp in _components(grid == color):
            info = _bridge_component_info(grid, bg, color, comp)
            if info is None:
                infos = []
                break
            infos.append(info)
        if infos:
            candidates.append((color, infos))
    assert len(candidates) == 1, "expected a unique bridge color"
    return candidates[0]


def _ordered_neighbors(cur: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = cur
    tr, tc = target
    order: list[tuple[int, int]] = []
    primary = []
    if tr > cr:
        primary.append((1, 0))
    elif tr < cr:
        primary.append((-1, 0))
    if tc > cc:
        primary.append((0, 1))
    elif tc < cc:
        primary.append((0, -1))
    for step in primary + DIRS4:
        if step not in order:
            order.append(step)
    return [(cr + dr, cc + dc) for dr, dc in order]


def _shortest_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nr, nc in _ordered_neighbors(cur, end):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    assert end in prev, "failed to route a bridge in the compact graph"
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _render_compact_graph(
    shape: tuple[int, int],
    bg: int,
    nodes: list[dict],
    edges: list[tuple[int, int]],
    bridge_color: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    positions = {int(node["id"]): tuple(node["pos"]) for node in nodes}
    node_cells = {tuple(node["pos"]) for node in nodes}
    for node in nodes:
        r, c = node["pos"]
        out[r, c] = int(node["color"])

    def edge_key(edge: tuple[int, int]) -> tuple[int, int, int, int]:
        a, b = edge
        pa = positions[a]
        pb = positions[b]
        return min(pa[0], pb[0]), min(pa[1], pb[1]), max(pa[0], pb[0]), max(pa[1], pb[1])

    for a, b in sorted(edges, key=edge_key):
        start = positions[a]
        end = positions[b]
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) == 1:
            continue

        if start[0] == end[0]:
            row = start[0]
            c0, c1 = sorted([start[1], end[1]])
            if all((row, c) not in node_cells for c in range(c0 + 1, c1)):
                out[row, c0 + 1 : c1] = bridge_color
                continue
        if start[1] == end[1]:
            col = start[1]
            r0, r1 = sorted([start[0], end[0]])
            if all((r, col) not in node_cells for r in range(r0 + 1, r1)):
                out[r0 + 1 : r1, col] = bridge_color
                continue

        blocked = set(node_cells)
        blocked.discard(start)
        blocked.discard(end)
        path = _shortest_path(shape, start, end, blocked)
        for r, c in path[1:-1]:
            out[r, c] = bridge_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink colored blocks into single-pixel nodes and keep grey bridge topology.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    bridge_color, bridge_infos = _detect_bridge_color(g, bg)

    node_colors = [int(color) for color in np.unique(g) if color not in (bg, bridge_color)]
    nodes: list[dict] = []
    color_to_id: dict[int, int] = {}
    for color in node_colors:
        comps = _components(g == color)
        assert len(comps) == 1, "each node color must form one component"
        comp = comps[0]
        r0, c0, r1, c1 = _bbox(comp)
        node_id = len(nodes)
        color_to_id[color] = node_id
        nodes.append(
            {
                "id": node_id,
                "color": color,
                "row_key": r0 + r1,
                "col_key": c0 + c1,
            }
        )

    row_keys = sorted({int(node["row_key"]) for node in nodes})
    col_keys = sorted({int(node["col_key"]) for node in nodes})
    row_rank = {key: idx for idx, key in enumerate(row_keys)}
    col_rank = {key: idx for idx, key in enumerate(col_keys)}
    for node in nodes:
        node["pos"] = (row_rank[int(node["row_key"])], col_rank[int(node["col_key"])])

    edge_set: set[tuple[int, int]] = set()
    for info in bridge_infos:
        color_a, color_b = info["neighbors"]
        a = color_to_id[int(color_a)]
        b = color_to_id[int(color_b)]
        edge_set.add(tuple(sorted((a, b))))

    return _render_compact_graph((len(row_keys), len(col_keys)), bg, nodes, sorted(edge_set), bridge_color)


def _visible_pairs(positions: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pos_to_id = {pos: idx for idx, pos in enumerate(positions)}
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in positions:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    horiz: list[tuple[int, int]] = []
    vert: list[tuple[int, int]] = []
    for r, cols in by_row.items():
        cols.sort()
        for c0, c1 in zip(cols, cols[1:]):
            horiz.append(tuple(sorted((pos_to_id[(r, c0)], pos_to_id[(r, c1)]))))
    for c, rows in by_col.items():
        rows.sort()
        for r0, r1 in zip(rows, rows[1:]):
            vert.append(tuple(sorted((pos_to_id[(r0, c)], pos_to_id[(r1, c)]))))
    return horiz, vert


def _edge_length(edge: tuple[int, int], positions: list[tuple[int, int]]) -> int:
    a, b = edge
    ra, ca = positions[a]
    rb, cb = positions[b]
    return abs(ra - rb) + abs(ca - cb)


def _edges_cross(
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    positions: list[tuple[int, int]],
) -> bool:
    a0, a1 = positions[edge_a[0]], positions[edge_a[1]]
    b0, b1 = positions[edge_b[0]], positions[edge_b[1]]
    if a0[0] == a1[0] and b0[1] == b1[1]:
        row = a0[0]
        col = b0[1]
        return min(b0[0], b1[0]) < row < max(b0[0], b1[0]) and min(a0[1], a1[1]) < col < max(a0[1], a1[1])
    if a0[1] == a1[1] and b0[0] == b1[0]:
        row = b0[0]
        col = a0[1]
        return min(a0[0], a1[0]) < row < max(a0[0], a1[0]) and min(b0[1], b1[1]) < col < max(b0[1], b1[1])
    return False


def _choose_edges(rng: random.Random, positions: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    horiz, vert = _visible_pairs(positions)
    orientation_mode = rng.choice(ORIENTATION_MODES)
    distance_mode = rng.choice(DISTANCE_MODES)

    if orientation_mode == "horizontal_only":
        if not horiz:
            return None
        allowed = list(horiz)
        mandatory = [rng.choice(horiz)]
    elif orientation_mode == "vertical_only":
        if not vert:
            return None
        allowed = list(vert)
        mandatory = [rng.choice(vert)]
    else:
        if not horiz or not vert:
            return None
        allowed = list(dict.fromkeys(horiz + vert))
        mandatory = [rng.choice(horiz), rng.choice(vert)]

    for _ in range(200):
        selected = set(mandatory)
        remaining = [edge for edge in allowed if edge not in selected]
        max_extra = min(4 - len(selected), len(remaining))
        extra_count = rng.randint(0, max_extra) if max_extra > 0 else 0
        if extra_count:
            selected.update(rng.sample(remaining, extra_count))
        lengths = [_edge_length(edge, positions) for edge in selected]
        has_adj = any(length == 1 for length in lengths)
        has_long = any(length > 1 for length in lengths)
        if distance_mode == "adjacent_only" and (not has_adj or has_long):
            continue
        if distance_mode == "long_only" and (not has_long or has_adj):
            continue
        if distance_mode == "mixed" and not (has_adj and has_long):
            continue

        selected_list = sorted(selected)
        ok = True
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                if _edges_cross(selected_list[i], selected_list[j], positions):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return selected_list
    return None


def _sample_positions(
    rng: random.Random,
    topo_h: int,
    topo_w: int,
    node_count: int,
) -> list[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(topo_h) for c in range(topo_w)]
    for _ in range(200):
        positions = sorted(rng.sample(all_cells, node_count))
        if len({r for r, _ in positions}) != topo_h:
            continue
        if len({c for _, c in positions}) != topo_w:
            continue
        return positions
    return None


def _node_mask(rng: random.Random, height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if rng.random() < 0.35:
        corner = rng.choice([(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)])
        mask[corner] = False
    return mask


def _render_input_from_layout(
    rng: random.Random,
    bg: int,
    positions: list[tuple[int, int]],
    edges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    topo_h = max(r for r, _ in positions) + 1
    topo_w = max(c for _, c in positions) + 1
    row_gap = rng.randint(7, 10)
    col_gap = rng.randint(7, 10)
    margin_r = rng.randint(3, 6)
    margin_c = rng.randint(3, 6)

    row_centers = [margin_r + i * row_gap for i in range(topo_h)]
    col_centers = [margin_c + j * col_gap for j in range(topo_w)]

    palette = [color for color in range(10) if color not in (bg, BRIDGE_COLOR)]
    rng.shuffle(palette)

    nodes: list[dict] = []
    for node_id, (r, c) in enumerate(positions):
        height = rng.choice(NODE_SIZE_CHOICES)
        width = rng.choice(NODE_SIZE_CHOICES)
        center_r = row_centers[r]
        center_c = col_centers[c]
        mask = _node_mask(rng, height, width)
        r0 = center_r - height // 2
        c0 = center_c - width // 2
        nodes.append(
            {
                "id": node_id,
                "color": int(palette[node_id]),
                "pos": (r, c),
                "center": (center_r, center_c),
                "bbox": (r0, c0, r0 + height - 1, c0 + width - 1),
                "mask": mask,
            }
        )

    max_row = max(node["bbox"][2] for node in nodes) + margin_r + 1
    max_col = max(node["bbox"][3] for node in nodes) + margin_c + 1
    grid = np.full((max_row, max_col), int(bg), dtype=int)

    for node in nodes:
        r0, c0, r1, c1 = node["bbox"]
        mask = node["mask"]
        color = int(node["color"])
        for dr in range(r1 - r0 + 1):
            for dc in range(c1 - c0 + 1):
                if mask[dr, dc]:
                    grid[r0 + dr, c0 + dc] = color

    id_to_node = {int(node["id"]): node for node in nodes}
    for a, b in edges:
        node_a = id_to_node[a]
        node_b = id_to_node[b]
        ra, ca = node_a["pos"]
        rb, cb = node_b["pos"]
        if ra == rb:
            if ca > cb:
                node_a, node_b = node_b, node_a
            left_r = int(node_a["center"][0])
            start = int(node_a["bbox"][3]) + 1
            end = int(node_b["bbox"][1]) - 1
            assert start <= end
            grid[left_r, start : end + 1] = BRIDGE_COLOR
        else:
            if ra > rb:
                node_a, node_b = node_b, node_a
            top_c = int(node_a["center"][1])
            start = int(node_a["bbox"][2]) + 1
            end = int(node_b["bbox"][0]) - 1
            assert start <= end
            grid[start : end + 1, top_c] = BRIDGE_COLOR

    compact_nodes = [
        {"id": int(node["id"]), "color": int(node["color"]), "pos": tuple(node["pos"])}
        for node in nodes
    ]
    expected = _render_compact_graph((topo_h, topo_w), bg, compact_nodes, edges, BRIDGE_COLOR)
    return grid, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    topo_h = rng.choice(TOPO_ROWS)
    topo_w = rng.choice(TOPO_COLS)
    node_min = max(3, topo_h, topo_w)
    node_max = min(MAX_NODES, topo_h * topo_w)
    if node_min > node_max:
        return None
    node_count = rng.randint(node_min, node_max)
    positions = _sample_positions(rng, topo_h, topo_w, node_count)
    if positions is None:
        return None

    edges = _choose_edges(rng, positions)
    if not edges:
        return None

    bg = rng.choice(BG_CHOICES)
    grid, expected = _render_input_from_layout(rng, bg, positions, edges)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
