from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9
USABLE_COLORS = (1, 2, 3, 6, 7, 8)
SPECIAL_COLORS = {BLACK, YELLOW, GREY, MAROON}
RNG = np.random.default_rng()
GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _edge_key(a, b):
    return tuple(sorted((tuple(a), tuple(b))))


def _find_locked_connectors(grid: np.ndarray):
    h, w = grid.shape
    locked = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            up = int(grid[r - 1, c])
            down = int(grid[r + 1, c])
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])
            if up == down and up not in SPECIAL_COLORS:
                locked[(r, c)] = {
                    "axis": "h",
                    "color": int(up),
                    "door_pixels": ((r - 1, c), (r + 1, c)),
                }
            elif left == right and left not in SPECIAL_COLORS:
                locked[(r, c)] = {
                    "axis": "v",
                    "color": int(left),
                    "door_pixels": ((r, c - 1), (r, c + 1)),
                }
    return locked


def _parse_layout(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    locked = _find_locked_connectors(g)
    door_pixels = set()
    for info in locked.values():
        door_pixels.update(info["door_pixels"])

    passable = g != MAROON
    for r, c in door_pixels:
        passable[r, c] = False

    connectors = {}
    for pos, info in locked.items():
        r, c = pos
        left_right = bool(passable[r, c - 1]) and bool(passable[r, c + 1])
        up_down = bool(passable[r - 1, c]) and bool(passable[r + 1, c])
        if info["axis"] == "h" and left_right and not bool(passable[r - 1, c]) and not bool(passable[r + 1, c]):
            connectors[pos] = {
                "axis": "h",
                "color": info["color"],
                "door_pixels": info["door_pixels"],
            }
        elif info["axis"] == "v" and up_down and not bool(passable[r, c - 1]) and not bool(passable[r, c + 1]):
            connectors[pos] = {
                "axis": "v",
                "color": info["color"],
                "door_pixels": info["door_pixels"],
            }

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK or (r, c) in connectors or not bool(passable[r, c]):
                continue
            left = bool(passable[r, c - 1])
            right = bool(passable[r, c + 1])
            up = bool(passable[r - 1, c])
            down = bool(passable[r + 1, c])
            horiz = left and right and not up and not down
            vert = up and down and not left and not right
            if horiz ^ vert:
                connectors[(r, c)] = {
                    "axis": "h" if horiz else "v",
                    "color": None,
                    "door_pixels": (),
                }

    room_mask = passable.copy()
    for r, c in connectors:
        room_mask[r, c] = False

    region = -np.ones((h, w), dtype=int)
    region_count = 0
    for r in range(h):
        for c in range(w):
            if not bool(room_mask[r, c]) or int(region[r, c]) != -1:
                continue
            q = deque([(r, c)])
            region[r, c] = region_count
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(room_mask[nr, nc]) and int(region[nr, nc]) == -1:
                        region[nr, nc] = region_count
                        q.append((nr, nc))
            region_count += 1

    edges = []
    adjacency = defaultdict(list)
    for pos, info in connectors.items():
        r, c = pos
        if info["axis"] == "h":
            ends = [(r, c - 1), (r, c + 1)]
        else:
            ends = [(r - 1, c), (r + 1, c)]
        a = int(region[ends[0]])
        b = int(region[ends[1]])
        if a < 0 or b < 0 or a == b:
            continue
        edge_id = len(edges)
        edges.append(
            {
                "a": a,
                "b": b,
                "color": info["color"],
                "door_pixels": tuple(info["door_pixels"]),
            }
        )
        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))

    start_region = None
    end_region = None
    slot_positions = []
    region_keys = defaultdict(list)
    region_gems = defaultdict(list)
    door_colors = {int(edge["color"]) for edge in edges if edge["color"] is not None}

    for r in range(h):
        for c in range(w):
            if (r, c) in connectors or (r, c) in door_pixels:
                continue
            rid = int(region[r, c])
            if rid < 0:
                continue
            value = int(g[r, c])
            if value == YELLOW:
                start_region = rid
            elif value == GREY:
                end_region = rid
                slot_positions.append((r, c))
            elif value == BLACK:
                continue
            elif value in door_colors:
                region_keys[rid].append((value, (r, c)))
            else:
                region_gems[rid].append((value, (r, c)))

    slot_positions.sort(key=lambda rc: (rc[1], rc[0]))
    assert start_region is not None
    assert end_region is not None
    return {
        "grid": g,
        "region": region,
        "edges": edges,
        "adjacency": adjacency,
        "start_region": int(start_region),
        "end_region": int(end_region),
        "slots": slot_positions,
        "region_keys": region_keys,
        "region_gems": region_gems,
    }


def _solve_region_path(layout):
    edges = layout["edges"]
    adjacency = layout["adjacency"]
    start_region = layout["start_region"]
    end_region = layout["end_region"]
    region_keys = layout["region_keys"]

    door_colors = sorted({int(edge["color"]) for edge in edges if edge["color"] is not None})
    bit_of = {color: i for i, color in enumerate(door_colors)}
    region_keymask = defaultdict(int)
    for rid, items in region_keys.items():
        mask = 0
        for color, _pos in items:
            mask |= 1 << bit_of[int(color)]
        region_keymask[int(rid)] = mask

    def collect(mask: int, rid: int) -> int:
        return int(mask | region_keymask[int(rid)])

    start_state = (int(start_region), collect(0, int(start_region)))
    pred = {start_state: None}
    pred_edge = {}
    q = deque([start_state])
    goal = None

    while q:
        state = q.popleft()
        rid, keymask = state
        if int(rid) == int(end_region):
            goal = state
            break
        for nxt, edge_id in adjacency[int(rid)]:
            color = edges[int(edge_id)]["color"]
            if color is not None and not (int(keymask) & (1 << bit_of[int(color)])):
                continue
            nxt_state = (int(nxt), collect(int(keymask), int(nxt)))
            if nxt_state in pred:
                continue
            pred[nxt_state] = state
            pred_edge[nxt_state] = int(edge_id)
            q.append(nxt_state)

    assert goal is not None

    states = []
    edge_path = []
    cur = goal
    while cur is not None:
        states.append(cur)
        if cur != start_state:
            edge_path.append(pred_edge[cur])
        cur = pred[cur]
    states.reverse()
    edge_path.reverse()
    region_path = [int(rid) for rid, _mask in states]
    return region_path, edge_path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Traverse the only legal route through the maze, using color-matched keys to open doors.

    Collected keys and gems are erased, opened doors are removed, and collected gem colors are
    written left-to-right into the grey slots in the end room.
    """
    layout = _parse_layout(np.asarray(grid, dtype=int))
    region_path, edge_path = _solve_region_path(layout)

    out = np.array(layout["grid"], copy=True)
    collected_positions = set()
    gems_in_order = []

    for rid in region_path:
        for color, pos in sorted(layout["region_keys"][int(rid)], key=lambda item: item[1]):
            if pos not in collected_positions:
                collected_positions.add(pos)
                out[pos] = BLACK
        for color, pos in sorted(layout["region_gems"][int(rid)], key=lambda item: item[1]):
            if pos not in collected_positions:
                collected_positions.add(pos)
                gems_in_order.append(int(color))
                out[pos] = BLACK

    opened = set()
    for edge_id in edge_path:
        edge = layout["edges"][int(edge_id)]
        if edge["color"] is None:
            continue
        for pos in edge["door_pixels"]:
            opened.add(pos)
    for pos in opened:
        out[pos] = BLACK

    for (r, c), color in zip(layout["slots"], gems_in_order):
        out[r, c] = int(color)

    return out


def _lattice_neighbors(node, rows: int, cols: int):
    r, c = node
    if r > 0:
        yield (r - 1, c)
    if r + 1 < rows:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < cols:
        yield (r, c + 1)


def _make_main_path(rows: int, cols: int, rng: np.random.Generator):
    start_r = int(rng.integers(rows))
    end_r = int(rng.integers(rows))
    column_rows = [int(rng.integers(rows)) for _ in range(cols)]
    column_rows[0] = start_r
    column_rows[-1] = end_r

    path = [(start_r, 0)]
    cur_r = start_r
    for c in range(cols - 1):
        target_r = int(column_rows[c + 1])
        step = 1 if target_r > cur_r else -1
        while cur_r != target_r:
            cur_r += step
            path.append((cur_r, c))
        path.append((cur_r, c + 1))
    return path


def _available_branch_starts(node, used, rows: int, cols: int):
    return [nbr for nbr in _lattice_neighbors(node, rows, cols) if nbr not in used]


def _attach_branch(anchor, used, rows: int, cols: int, rng: np.random.Generator):
    first_choices = _available_branch_starts(anchor, used, rows, cols)
    if not first_choices:
        return None
    first = first_choices[int(rng.integers(len(first_choices)))]
    branch = [first]
    want_len2 = bool(rng.integers(2))
    if want_len2:
        second_choices = [nbr for nbr in _lattice_neighbors(first, rows, cols) if nbr != anchor and nbr not in used]
        if second_choices:
            second = second_choices[int(rng.integers(len(second_choices)))]
            branch.append(second)
    return branch


def _solve_node_graph(start, end, edge_colors, key_by_node):
    adjacency = defaultdict(list)
    for ek, color in edge_colors.items():
        a, b = ek
        adjacency[a].append((b, ek, color))
        adjacency[b].append((a, ek, color))

    door_colors = sorted({int(color) for color in edge_colors.values() if color is not None})
    bit_of = {color: i for i, color in enumerate(door_colors)}
    node_keymask = {node: 1 << bit_of[int(color)] for node, color in key_by_node.items()}

    def collect(mask: int, node):
        return int(mask | node_keymask.get(node, 0))

    start_state = (tuple(start), collect(0, tuple(start)))
    pred = {start_state: None}
    pred_edge = {}
    q = deque([start_state])
    goal = None

    while q:
        node, keymask = q.popleft()
        if node == tuple(end):
            goal = (node, keymask)
            break
        for nxt, ek, color in adjacency[node]:
            if color is not None and not (int(keymask) & (1 << bit_of[int(color)])):
                continue
            nxt_state = (tuple(nxt), collect(int(keymask), tuple(nxt)))
            if nxt_state in pred:
                continue
            pred[nxt_state] = (node, keymask)
            pred_edge[nxt_state] = ek
            q.append(nxt_state)

    if goal is None:
        return None

    nodes = []
    edges = []
    cur = goal
    while cur is not None:
        nodes.append(cur[0])
        if cur != start_state:
            edges.append(pred_edge[cur])
        cur = pred[cur]
    nodes.reverse()
    edges.reverse()
    return nodes, edges


def _row_col_starts(sizes):
    starts = []
    cursor = 1
    for size in sizes:
        starts.append(cursor)
        cursor += int(size) + 1
    return starts


def _room_bbox(node, row_starts, col_starts, row_heights, col_widths):
    r, c = node
    r0 = int(row_starts[r])
    c0 = int(col_starts[c])
    return r0, r0 + int(row_heights[r]) - 1, c0, c0 + int(col_widths[c]) - 1


def _gate_position(a, b, row_starts, col_starts, row_heights, col_widths, rng: np.random.Generator):
    ar, ac = a
    br, bc = b
    if ar == br:
        r0, r1, c0, c1 = _room_bbox(a, row_starts, col_starts, row_heights, col_widths)
        wall_c = c1 + 1 if ac < bc else c0 - 1
        gate_r = int(rng.integers(r0 + 1, r1))
        return (gate_r, wall_c), "h"
    r0, r1, c0, c1 = _room_bbox(a, row_starts, col_starts, row_heights, col_widths)
    wall_r = r1 + 1 if ar < br else r0 - 1
    gate_c = int(rng.integers(c0 + 1, c1))
    return (wall_r, gate_c), "v"


def _safe_item_positions(node, row_starts, col_starts, row_heights, col_widths, reserved):
    r0, r1, c0, c1 = _room_bbox(node, row_starts, col_starts, row_heights, col_widths)
    pts = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) not in reserved:
                pts.append((r, c))
    return pts


def _bad_key_position(pos, color: int, grid: np.ndarray):
    r, c = pos
    h, w = grid.shape
    for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2)):
        rr = r + dr
        cc = c + dc
        mr = r + dr // 2
        mc = c + dc // 2
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(color) and int(grid[mr, mc]) == BLACK:
            return True
    return False


def _build_abstract_spec(rng: np.random.Generator):
    rows = int(rng.integers(4, 6))
    cols = int(rng.integers(4, 7))
    path = _make_main_path(rows, cols, rng)
    start = path[0]
    end = path[-1]
    path_set = set(path)

    eligible = [i for i in range(1, len(path) - 1) if _available_branch_starts(path[i], path_set, rows, cols)]
    if not eligible:
        return None

    max_doors = min(3, len(eligible))
    door_count = int(rng.integers(1, max_doors + 1))
    rng.shuffle(eligible)

    used = set(path)
    door_branch = {}
    for i in eligible:
        if len(door_branch) == door_count:
            break
        branch = _attach_branch(path[i], used, rows, cols, rng)
        if branch is None:
            continue
        for node in branch:
            used.add(node)
        door_branch[int(i)] = branch
    if len(door_branch) != door_count:
        return None

    edge_colors = {}
    for a, b in zip(path, path[1:]):
        edge_colors[_edge_key(a, b)] = None
    for anchor_index, branch in door_branch.items():
        prev = path[int(anchor_index)]
        for node in branch:
            edge_colors[_edge_key(prev, node)] = None
            prev = node

    door_colors = rng.choice(np.array(USABLE_COLORS), size=door_count, replace=False).tolist()
    key_by_node = {}
    for color, idx in zip(door_colors, sorted(door_branch)):
        key_by_node[door_branch[idx][-1]] = int(color)
        edge_colors[_edge_key(path[idx], path[idx + 1])] = int(color)

    route = _solve_node_graph(start, end, edge_colors, key_by_node)
    if route is None:
        return None
    route_nodes, route_edges = route

    first_visit = []
    seen = set()
    for node in route_nodes:
        if node not in seen:
            seen.add(node)
            first_visit.append(node)
    candidates = [node for node in first_visit[1:-1] if node not in key_by_node]
    free_colors = [color for color in USABLE_COLORS if color not in door_colors]
    max_gems = min(3, len(candidates), len(free_colors))
    if max_gems < 1:
        return None

    gem_count = int(rng.integers(1, max_gems + 1))
    chosen_idx = rng.choice(np.arange(len(candidates)), size=gem_count, replace=False)
    gem_nodes = [candidates[int(i)] for i in chosen_idx]
    first_index = {node: i for i, node in enumerate(first_visit)}
    gem_nodes.sort(key=lambda node: first_index[node])
    gem_colors = rng.choice(np.array(free_colors), size=gem_count, replace=False).tolist()
    gem_by_node = {node: int(color) for node, color in zip(gem_nodes, gem_colors)}

    return {
        "rows": rows,
        "cols": cols,
        "path": path,
        "start": start,
        "end": end,
        "used": used,
        "edge_colors": edge_colors,
        "key_by_node": key_by_node,
        "gem_by_node": gem_by_node,
        "route_nodes": route_nodes,
        "route_edges": route_edges,
    }


def _embed_spec(spec, rng: np.random.Generator):
    rows = spec["rows"]
    cols = spec["cols"]
    row_heights = [int(x) for x in rng.integers(4, 7, size=rows)]
    col_widths = [int(x) for x in rng.integers(4, 7, size=cols)]
    row_starts = _row_col_starts(row_heights)
    col_starts = _row_col_starts(col_widths)
    h = sum(row_heights) + rows + 1
    w = sum(col_widths) + cols + 1
    grid = np.full((h, w), MAROON, dtype=int)

    for node in spec["used"]:
        r0, r1, c0, c1 = _room_bbox(node, row_starts, col_starts, row_heights, col_widths)
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    door_pixels_by_edge = {}
    for ek, color in spec["edge_colors"].items():
        a, b = ek
        gate, axis = _gate_position(a, b, row_starts, col_starts, row_heights, col_widths, rng)
        gr, gc = gate
        grid[gr, gc] = BLACK
        if axis == "h":
            flanks = ((gr - 1, gc), (gr + 1, gc))
        else:
            flanks = ((gr, gc - 1), (gr, gc + 1))
        if color is not None:
            for fr, fc in flanks:
                grid[fr, fc] = int(color)
            door_pixels_by_edge[ek] = flanks
        else:
            door_pixels_by_edge[ek] = ()

    reserved = defaultdict(set)
    start_node = spec["start"]
    end_node = spec["end"]

    def place_single(node, color: int, *, avoid_key_pattern: bool):
        candidates = _safe_item_positions(node, row_starts, col_starts, row_heights, col_widths, reserved[node])
        rng.shuffle(candidates)
        for pos in candidates:
            if avoid_key_pattern and _bad_key_position(pos, int(color), grid):
                continue
            reserved[node].add(pos)
            grid[pos] = int(color)
            return pos
        return None

    start_pos = place_single(start_node, YELLOW, avoid_key_pattern=False)
    if start_pos is None:
        return None

    r0, r1, c0, c1 = _room_bbox(end_node, row_starts, col_starts, row_heights, col_widths)
    slot_count = len(spec["gem_by_node"])
    slot_row_choices = list(range(r0 + 1, r1))
    rng.shuffle(slot_row_choices)
    slot_positions = None
    for slot_r in slot_row_choices:
        start_cols = list(range(c0, c1 - slot_count + 2))
        rng.shuffle(start_cols)
        for slot_c0 in start_cols:
            positions = [(slot_r, slot_c0 + i) for i in range(slot_count)]
            if any(pos in reserved[end_node] for pos in positions):
                continue
            for pos in positions:
                reserved[end_node].add(pos)
                grid[pos] = GREY
            slot_positions = positions
            break
        if slot_positions is not None:
            break
    if slot_positions is None:
        return None

    key_positions = {}
    for node, color in spec["key_by_node"].items():
        pos = place_single(node, int(color), avoid_key_pattern=True)
        if pos is None:
            return None
        key_positions[node] = pos

    gem_positions = {}
    for node, color in spec["gem_by_node"].items():
        pos = place_single(node, int(color), avoid_key_pattern=False)
        if pos is None:
            return None
        gem_positions[node] = pos

    pts = np.argwhere(grid != MAROON)
    crop_r0 = max(0, int(pts[:, 0].min()) - 1)
    crop_r1 = min(h - 1, int(pts[:, 0].max()) + 1)
    crop_c0 = max(0, int(pts[:, 1].min()) - 1)
    crop_c1 = min(w - 1, int(pts[:, 1].max()) + 1)
    grid = grid[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1].copy()

    def shift(pos):
        return (int(pos[0] - crop_r0), int(pos[1] - crop_c0))

    slot_positions = [shift(pos) for pos in slot_positions]
    key_positions = {node: shift(pos) for node, pos in key_positions.items()}
    gem_positions = {node: shift(pos) for node, pos in gem_positions.items()}
    door_pixels_by_edge = {
        ek: tuple(shift(pos) for pos in positions) for ek, positions in door_pixels_by_edge.items()
    }

    spec = dict(spec)
    spec["grid"] = grid
    spec["slot_positions"] = slot_positions
    spec["key_positions"] = key_positions
    spec["gem_positions"] = gem_positions
    spec["door_pixels_by_edge"] = door_pixels_by_edge
    return spec


def _validate_spec(spec):
    grid = spec["grid"]
    out = main(grid)

    expected_gems = []
    seen_nodes = set()
    for node in spec["route_nodes"]:
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        if node in spec["gem_by_node"]:
            expected_gems.append(int(spec["gem_by_node"][node]))

    for pos in spec["key_positions"].values():
        if int(out[pos]) != BLACK:
            return False
    for pos in spec["gem_positions"].values():
        if int(out[pos]) != BLACK:
            return False
    for ek in spec["route_edges"]:
        color = spec["edge_colors"][ek]
        if color is None:
            continue
        for pos in spec["door_pixels_by_edge"][ek]:
            if int(out[pos]) != BLACK:
                return False
    for pos, color in zip(spec["slot_positions"], expected_gems):
        if int(out[pos]) != int(color):
            return False
    return True


def _gen_attempt(rng: np.random.Generator):
    spec = _build_abstract_spec(rng)
    if spec is None:
        return None
    embedded = _embed_spec(spec, rng)
    if embedded is None:
        return None
    if not _validate_spec(embedded):
        return None
    return embedded["grid"]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-door-gem sample")
