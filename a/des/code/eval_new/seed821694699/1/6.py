import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _extract_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue

            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if _in_bounds(nr, nc, h, w) and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _find_rule_panel(grid: np.ndarray):
    orange_mask = grid == ORANGE
    comps = _extract_components(orange_mask)
    candidates = []

    for cells in comps:
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        if h < 3 or w < 3:
            continue

        expected_border = 2 * (h + w) - 4
        if len(cells) != expected_border:
            continue

        ok = True
        for c in range(c0, c1 + 1):
            if grid[r0, c] != ORANGE or grid[r1, c] != ORANGE:
                ok = False
                break
        if not ok:
            continue

        for r in range(r0 + 1, r1):
            if grid[r, c0] != ORANGE or grid[r, c1] != ORANGE:
                ok = False
                break

        if not ok:
            continue

        candidates.append((r0, r1, c0, c1, len(cells)))

    if not candidates:
        return None

    # Prefer larger panel when multiple components exist.
    candidates.sort(key=lambda x: x[4], reverse=True)
    r0, r1, c0, c1, _ = candidates[0]
    return r0, r1, c0, c1


def _parse_panel_rules(grid: np.ndarray):
    panel = _find_rule_panel(grid)
    if panel is None:
        return []

    pr0, pr1, pc0, pc1 = panel

    def _is_rule_color(v: int) -> bool:
        return v != BG and v != GREY and v != ORANGE

    rules = []
    for rr in range(pr0 + 1, pr1):
        row = grid[rr, pc0 + 1 : pc1]
        for c in range(len(row) - 3):
            a, b, gap, out = int(row[c]), int(row[c + 1]), int(row[c + 2]), int(row[c + 3])
            if not (_is_rule_color(a) and _is_rule_color(b) and _is_rule_color(out)):
                continue
            if gap != BG:
                continue
            left_ok = (c == 0) or (row[c - 1] == BG)
            right_ok = (c + 3 == len(row) - 1) or (row[c + 4] == BG)
            if not (left_ok and right_ok):
                continue
            rules.append((a, b, out))
            break
    return rules


def _extract_nodes(grid: np.ndarray, panel):
    h, w = grid.shape
    node_mask = (grid != BG) & (grid != GREY)
    if panel is not None:
        pr0, pr1, pc0, pc1 = panel
        node_mask[pr0 : pr1 + 1, pc0 : pc1 + 1] = False

    comps = _extract_components(node_mask)
    node_positions = []
    node_colors = []

    for comp in comps:
        if len(comp) != 4:
            continue
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        if r1 - r0 != 1 or c1 - c0 != 1:
            continue

        color = int(grid[r0, c0])
        if any(int(grid[r, c]) != color for r, c in comp):
            continue

        node_positions.append((r0, c0))
        node_colors.append(color)

    node_ids = {pos: i for i, pos in enumerate(node_positions)}
    return node_positions, node_colors, node_ids


def _build_graph(grid: np.ndarray, node_positions):
    h, w = grid.shape
    comp_map, _ = _extract_grey_component_map(grid)
    comp_nodes = defaultdict(set)

    for nid, (r0, c0) in enumerate(node_positions):
        r1, c1 = r0 + 1, c0 + 1
        near = []

        # top and bottom rows next to node
        if r0 - 1 >= 0:
            near.extend([(r0 - 1, c) for c in (c0, c1)])
        if r1 + 1 < h:
            near.extend([(r1 + 1, c) for c in (c0, c1)])

        # left and right columns next to node
        if c0 - 1 >= 0:
            near.extend([(r, c0 - 1) for r in (r0, r1)])
        if c1 + 1 < w:
            near.extend([(r, c1 + 1) for r in (r0, r1)])

        for rr, cc in near:
            cid = comp_map[rr, cc]
            if cid >= 0:
                comp_nodes[cid].add(nid)

    graph = [set() for _ in node_positions]
    for nodes in comp_nodes.values():
        ids = sorted(nodes)
        if len(ids) >= 2:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    u, v = ids[i], ids[j]
                    graph[u].add(v)
                    graph[v].add(u)

    return graph


def _extract_grey_component_map(grid: np.ndarray):
    h, w = grid.shape
    grey = grid == GREY
    comp_map = -np.ones((h, w), dtype=int)
    visited = np.zeros_like(grey, dtype=bool)
    comps = []

    for sr in range(h):
        for sc in range(w):
            if not grey[sr, sc] or visited[sr, sc]:
                continue

            cid = len(comps)
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells = []

            while q:
                r, c = q.popleft()
                comp_map[r, c] = cid
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if _in_bounds(nr, nc, h, w) and grey[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            comps.append(cells)

    return comp_map, comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    1) Parse
    Parse ARC grid colors, find the orange bordered rule panel and its (A,B)->C rows.

    2) Build graph
    Locate 2x2 colored node blocks and connect nodes through 4-neighbor gray components.

    3) Apply rules
    For each node, collect neighbor colors and choose the first rule whose two required colors
    are both present.

    4) Compose output
    Copy input and recolor only matched nodes; edges and panel remain unchanged.
    """

    grid = np.asarray(grid)
    panel = _find_rule_panel(grid)
    rules = _parse_panel_rules(grid)
    node_positions, node_colors, _ = _extract_nodes(grid, panel)

    if not node_positions:
        return np.array(grid, copy=True)

    graph = _build_graph(grid, node_positions)

    out = np.array(grid, copy=True)
    for idx, (r0, c0) in enumerate(node_positions):
        neigh_colors = {node_colors[nid] for nid in graph[idx]}
        new_color = node_colors[idx]

        for a, b, c in rules:
            if a in neigh_colors and b in neigh_colors:
                new_color = c
                break

        out[r0 : r0 + 2, c0 : c0 + 2] = new_color

    return out


def _draw_horizontal_edge(grid: np.ndarray, r: int, c1: int, c2: int):
    if c1 > c2:
        c1, c2 = c2, c1
    for c in range(c1, c2 + 1):
        grid[r, c] = GREY


def _draw_vertical_edge(grid: np.ndarray, c: int, r1: int, r2: int):
    if r1 > r2:
        r1, r2 = r2, r1
    for r in range(r1, r2 + 1):
        grid[r, c] = GREY


def generate_input() -> np.ndarray:
    """
    1) Layout
    Create a random compact node lattice and sample colors within non-background palette.

    2) Structure
    Draw 2x2 node blocks and random gray edges (right/down adjacency).

    3) Rules panel
    Add an orange framed panel and write multiple (A,B)->C rows.

    4) Return
    Return a valid ARC integer ndarray with height/width <= 30.
    """

    rng = random.Random()
    node_colors = [1, 2, 3, 4, 6, 8, 9, 10]

    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    gap = rng.randint(1, 3)
    rule_count = rng.randint(2, min(6, rows * cols))

    # Node geometry spans including only nodes + spaces.
    node_w = cols * 2 + (cols - 1) * gap
    node_h = rows * 2 + (rows - 1) * gap
    margin = 1

    panel_h = rule_count + 2
    panel_w = 7

    base_w = node_w + 2 * margin
    base_h = node_h + 2 * margin

    place_panel_right = (base_w + panel_w + 2 <= 30)

    if place_panel_right:
        h = max(base_h, panel_h + 2)
        w = base_w + panel_w + 2
    else:
        h = base_h + panel_h + 2
        w = max(base_w + 1, panel_w + 2)

    h = min(30, max(10, h))
    w = min(30, max(10, w))

    grid = np.zeros((h, w), dtype=np.int32)

    # Place nodes on a lattice.
    node_positions = []
    id_map = {}
    nid = 0
    for ri in range(rows):
        for ci in range(cols):
            r = margin + ri * (2 + gap)
            c = margin + ci * (2 + gap)
            if r + 1 >= h or c + 1 >= w:
                continue

            color = rng.choice(node_colors)
            grid[r : r + 2, c : c + 2] = color
            node_positions.append((ri, ci, r, c, nid))
            id_map[(ri, ci)] = nid
            nid += 1

    if not node_positions:
        return grid

    # Random adjacency on lattice; then make sure every node has at least one neighbor.
    edges = set()
    for ri, ci, r, c, i in node_positions:
        right = id_map.get((ri, ci + 1))
        down = id_map.get((ri + 1, ci))
        if right is not None and rng.random() < 0.7:
            edges.add((min(i, right), max(i, right)))
        if down is not None and rng.random() < 0.7:
            edges.add((min(i, down), max(i, down)))

    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    if len(node_positions) > 1:
        all_ids = [i for i in range(len(node_positions))]
        for _, _, _, _, i in node_positions:
            if deg[i] > 0:
                continue

            best_candidates = []
            for _, _, _, _, j in node_positions:
                if j != i:
                    best_candidates.append(j)
            if best_candidates:
                j = rng.choice(best_candidates)
                edges.add((min(i, j), max(i, j)))

    id_to_rc = {}
    for _, _, r, c, i in node_positions:
        id_to_rc[i] = (r, c)

    for u, v in edges:
        r1, c1 = id_to_rc[u]
        r2, c2 = id_to_rc[v]
        if r1 == r2:
            row = min(r1, r2) + 1
            c_start = min(c1, c2) + 2
            c_end = max(c1, c2) - 1
            _draw_horizontal_edge(grid, row, c_start, c_end)
        elif c1 == c2:
            col = c1 + 1
            r_start = min(r1, r2) + 2
            r_end = max(r1, r2) - 1
            _draw_vertical_edge(grid, col, r_start, r_end)

    # Place panel.
    if place_panel_right:
        panel_top = margin
        panel_left = margin + node_w + 1
    else:
        panel_top = margin + node_h + 1
        panel_left = margin

    if panel_top + panel_h - 1 >= h or panel_left + panel_w - 1 >= w:
        return grid

    panel_bottom = panel_top + panel_h - 1
    panel_right = panel_left + panel_w - 1

    grid[panel_top, panel_left : panel_right + 1] = ORANGE
    grid[panel_bottom, panel_left : panel_right + 1] = ORANGE
    grid[panel_top : panel_bottom + 1, panel_left] = ORANGE
    grid[panel_top : panel_bottom + 1, panel_right] = ORANGE

    for i in range(rule_count):
        rr = panel_top + 1 + i
        a = rng.choice(node_colors)
        b = rng.choice([x for x in node_colors if x != a])
        c = rng.choice(node_colors)

        grid[rr, panel_left + 1] = a
        grid[rr, panel_left + 2] = b
        grid[rr, panel_left + 3] = BG
        grid[rr, panel_left + 4] = c

    return grid


if __name__ == "__main__":
    inp = generate_input()
    out = main(inp)
    print(inp.shape, out.shape)
