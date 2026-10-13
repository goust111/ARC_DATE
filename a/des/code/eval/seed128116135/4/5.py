import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
SWITCH_COLORS = (1, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
GRID_MIN = 17
GRID_MAX = 23
EDGE_COUNT_RANGE = (4, 7)
LEAF_COUNT_RANGE = (2, 5)
MIN_SEGMENT_DELTA = 4  # inclusive segment length is delta + 1, so >= 5 and always odd


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        step = 1 if c2 > c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    step = 1 if r2 > r1 else -1
    return [(r, c1) for r in range(r1, r2 + step, step)]


def _segment_mid(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        return r1, (c1 + c2) // 2
    return (r1 + r2) // 2, c1


def _edge_axis(a: tuple[int, int], b: tuple[int, int]) -> tuple[str, int]:
    if a[0] == b[0]:
        return ("row", a[0])
    return ("col", a[1])


def _find_switches(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    fixed = {BLACK, RED, GREEN, YELLOW, GREY}
    switches = []
    seen_colors = set()
    edge_coords = []
    for c in range(w):
        edge_coords.append((0, c))
        edge_coords.append((h - 1, c))
    for r in range(1, h - 1):
        edge_coords.append((r, 0))
        edge_coords.append((r, w - 1))

    for r, c in edge_coords:
        color = int(grid[r, c])
        if color in fixed:
            continue
        if color in seen_colors:
            continue
        seen_colors.add(color)
        if c in (0, w - 1):
            switches.append({"axis": "row", "index": r, "color": color})
        else:
            switches.append({"axis": "col", "index": c, "color": color})
    switches.sort(key=lambda item: (item["axis"], item["index"], item["color"]))
    return switches


def _row_intervals(grid: np.ndarray, row: int) -> list[tuple[int, int]]:
    w = grid.shape[1]
    mask = np.zeros(w, dtype=bool)
    for c in range(w):
        if int(grid[row, c]) == GREY:
            if (c > 0 and int(grid[row, c - 1]) == GREY) or (c + 1 < w and int(grid[row, c + 1]) == GREY):
                mask[c] = True
        elif (
            int(grid[row, c]) == BLACK
            and 0 < c < w - 1
            and int(grid[row, c - 1]) == GREY
            and int(grid[row, c + 1]) == GREY
        ):
            mask[c] = True

    intervals = []
    c = 0
    while c < w:
        if not mask[c]:
            c += 1
            continue
        start = c
        while c + 1 < w and mask[c + 1]:
            c += 1
        end = c
        if end - start + 1 >= 5 and (end - start) % 2 == 0:
            intervals.append((start, end))
        c += 1
    return intervals


def _col_intervals(grid: np.ndarray, col: int) -> list[tuple[int, int]]:
    h = grid.shape[0]
    mask = np.zeros(h, dtype=bool)
    for r in range(h):
        if int(grid[r, col]) == GREY:
            if (r > 0 and int(grid[r - 1, col]) == GREY) or (r + 1 < h and int(grid[r + 1, col]) == GREY):
                mask[r] = True
        elif (
            int(grid[r, col]) == BLACK
            and 0 < r < h - 1
            and int(grid[r - 1, col]) == GREY
            and int(grid[r + 1, col]) == GREY
        ):
            mask[r] = True

    intervals = []
    r = 0
    while r < h:
        if not mask[r]:
            r += 1
            continue
        start = r
        while r + 1 < h and mask[r + 1]:
            r += 1
        end = r
        if end - start + 1 >= 5 and (end - start) % 2 == 0:
            intervals.append((start, end))
        r += 1
    return intervals


def _toggle_wires(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for switch in _find_switches(grid):
        axis = switch["axis"]
        index = int(switch["index"])
        color = int(switch["color"])
        if axis == "row":
            for c0, c1 in _row_intervals(grid, index):
                mid_c = (c0 + c1) // 2
                out[index, mid_c] = color if int(grid[index, mid_c]) == BLACK else BLACK
        else:
            for r0, r1 in _col_intervals(grid, index):
                mid_r = (r0 + r1) // 2
                out[mid_r, index] = color if int(grid[mid_r, index]) == BLACK else BLACK
    return out


def _reachable_wires(grid: np.ndarray) -> set[tuple[int, int]]:
    red_pixels = np.argwhere(grid == RED)
    assert len(red_pixels) == 1
    start = tuple(int(v) for v in red_pixels[0])
    conductive = {RED, GREY}
    conductive.update(s["color"] for s in _find_switches(grid))

    h, w = grid.shape
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) not in conductive:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Toggle whole switched rows/columns of grey wire segments, then light the bulbs
    whose adjacent endpoints are reachable from the corner power source.
    """
    toggled = _toggle_wires(np.asarray(grid, dtype=int))
    reach = _reachable_wires(toggled)
    out = toggled.copy()
    h, w = out.shape
    for r, c in np.argwhere(out == YELLOW):
        rr = int(r)
        cc = int(c)
        if any((nr, nc) in reach for nr, nc in _neighbors4(rr, cc) if 0 <= nr < h and 0 <= nc < w):
            out[rr, cc] = GREEN
    return out


def _source_options(h: int, w: int):
    return [
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((0, w - 1), (1, w - 1)),
        ((0, w - 1), (0, w - 2)),
        ((h - 1, 0), (h - 2, 0)),
        ((h - 1, 0), (h - 1, 1)),
        ((h - 1, w - 1), (h - 2, w - 1)),
        ((h - 1, w - 1), (h - 1, w - 2)),
    ]


def _inward_dir(root: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = root
    if c == 0:
        return (0, 1)
    if c == w - 1:
        return (0, -1)
    if r == 0:
        return (1, 0)
    return (-1, 0)


def _can_add_segment(
    start: tuple[int, int],
    end: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    cells = _segment_cells(start, end)
    if len(cells) < 5:
        return False
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    if any(cell in occupied for cell in cells[1:]):
        return False

    for idx, cell in enumerate(cells[1:], start=1):
        prev = cells[idx - 1]
        for nr, nc in _neighbors4(*cell):
            if (nr, nc) in occupied and (nr, nc) != prev:
                return False
    return True


def _grow_tree(h: int, w: int):
    source, root = random.choice(_source_options(h, w))
    target_edges = random.randint(*EDGE_COUNT_RANGE)

    nodes = [root]
    degrees = {root: 0}
    occupied = {root}
    edges = []

    attempts = 0
    while len(edges) < target_edges and attempts < 500:
        attempts += 1
        candidates = []
        for node in nodes:
            cap = 1 if node == root else 3
            if degrees[node] < cap:
                candidates.append(node)
        if not candidates:
            break

        start = random.choice(candidates)
        if start == root and degrees[root] == 0:
            directions = [_inward_dir(root, h, w)]
        else:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            random.shuffle(directions)

        placed = False
        for dr, dc in directions:
            max_dist = 0
            rr, cc = start
            while True:
                rr += dr
                cc += dc
                if not (1 <= rr <= h - 2 and 1 <= cc <= w - 2):
                    break
                max_dist += 1
            dists = [d for d in range(MIN_SEGMENT_DELTA, max_dist + 1, 2)]
            random.shuffle(dists)
            for dist in dists:
                end = (start[0] + dr * dist, start[1] + dc * dist)
                if end in degrees:
                    continue
                if not _can_add_segment(start, end, occupied, h, w):
                    continue
                cells = _segment_cells(start, end)
                occupied.update(cells[1:])
                edges.append({"u": start, "v": end, "axis": _edge_axis(start, end)})
                nodes.append(end)
                degrees[start] += 1
                degrees[end] = 1
                placed = True
                break
            if placed:
                break

    if len(edges) < EDGE_COUNT_RANGE[0]:
        return None

    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {node: [] for node in degrees}
    for idx, edge in enumerate(edges):
        u = edge["u"]
        v = edge["v"]
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    leaves = [node for node, nbrs in adj.items() if len(nbrs) == 1 and node != root]
    if not (LEAF_COUNT_RANGE[0] <= len(leaves) <= LEAF_COUNT_RANGE[1]):
        return None

    rows = sorted({edge["axis"][1] for edge in edges if edge["axis"][0] == "row"})
    cols = sorted({edge["axis"][1] for edge in edges if edge["axis"][0] == "col"})
    if not rows or not cols:
        return None

    sr, sc = source
    row_side = w - 1 if sc == 0 else 0
    col_side = h - 1 if sr == 0 else 0

    bulbs = {}
    for leaf in leaves:
        parent = adj[leaf][0][0]
        dr = 0 if leaf[0] == parent[0] else (1 if leaf[0] > parent[0] else -1)
        dc = 0 if leaf[1] == parent[1] else (1 if leaf[1] > parent[1] else -1)
        candidate_dirs = [(dr, dc), (dc, dr), (-dc, -dr), (-dr, -dc)]
        bulb = None
        for br, bc in candidate_dirs:
            pos = (leaf[0] + br, leaf[1] + bc)
            if not (0 <= pos[0] < h and 0 <= pos[1] < w):
                continue
            if pos == parent:
                continue
            if pos in occupied or pos == source or pos in bulbs.values():
                continue
            if pos[1] == row_side:
                continue
            if pos[0] == col_side:
                continue
            bulb = pos
            break
        if bulb is None:
            return None
        bulbs[leaf] = bulb

    return {
        "shape": (h, w),
        "source": source,
        "root": root,
        "edges": edges,
        "adj": adj,
        "leaves": leaves,
        "bulbs": bulbs,
        "rows": rows,
        "cols": cols,
    }


def _pick_switch_spec(model: dict):
    rows = model["rows"]
    cols = model["cols"]
    row_count = random.randint(1, min(2, len(rows)))
    col_count = random.randint(1, min(2, len(cols)))
    total = row_count + col_count
    if total > len(SWITCH_COLORS):
        return None

    row_axes = random.sample(rows, row_count)
    col_axes = random.sample(cols, col_count)
    colors = random.sample(list(SWITCH_COLORS), total)

    axes = [("row", r) for r in row_axes] + [("col", c) for c in col_axes]
    random.shuffle(axes)
    broken = {axis: random.choice([False, True]) for axis in axes}
    if all(not state for state in broken.values()):
        broken[random.choice(axes)] = True
    if all(state for state in broken.values()):
        broken[random.choice(axes)] = False

    switch_colors = {axis: colors[i] for i, axis in enumerate(axes)}
    return {"broken": broken, "colors": switch_colors}


def _render_input(model: dict, switch_spec: dict) -> np.ndarray:
    h, w = model["shape"]
    grid = np.full((h, w), BLACK, dtype=int)

    for edge in model["edges"]:
        for r, c in _segment_cells(edge["u"], edge["v"]):
            grid[r, c] = GREY

    for edge in model["edges"]:
        axis = edge["axis"]
        if switch_spec["broken"].get(axis, False):
            mr, mc = _segment_mid(edge["u"], edge["v"])
            grid[mr, mc] = BLACK

    sr, sc = model["source"]
    grid[sr, sc] = RED

    for bulb in model["bulbs"].values():
        grid[bulb] = YELLOW

    row_side = w - 1 if sc == 0 else 0
    col_side = h - 1 if sr == 0 else 0
    for (axis, index), color in switch_spec["colors"].items():
        if axis == "row":
            grid[index, row_side] = color
        else:
            grid[col_side, index] = color

    return grid


def _reachable_leaves_after_toggle(model: dict, switch_spec: dict) -> set[tuple[int, int]]:
    q = deque([model["root"]])
    seen = {model["root"]}
    while q:
        node = q.popleft()
        for nxt, edge_idx in model["adj"][node]:
            edge = model["edges"][edge_idx]
            axis = edge["axis"]
            passable = True
            if axis in switch_spec["broken"]:
                passable = bool(switch_spec["broken"][axis])
            if not passable:
                continue
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return {leaf for leaf in model["leaves"] if leaf in seen}


def _expected_output(model: dict, switch_spec: dict, grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for edge in model["edges"]:
        axis = edge["axis"]
        if axis not in switch_spec["broken"]:
            continue
        mr, mc = _segment_mid(edge["u"], edge["v"])
        if switch_spec["broken"][axis]:
            out[mr, mc] = switch_spec["colors"][axis]
        else:
            out[mr, mc] = BLACK

    reached = _reachable_leaves_after_toggle(model, switch_spec)
    for leaf, bulb in model["bulbs"].items():
        if leaf in reached:
            out[bulb] = GREEN
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    model = _grow_tree(h, w)
    if model is None:
        return None

    for _ in range(40):
        switch_spec = _pick_switch_spec(model)
        if switch_spec is None:
            continue

        grid = _render_input(model, switch_spec)
        solved = main(grid)
        green_count = int(np.count_nonzero(solved == GREEN))
        yellow_count = int(np.count_nonzero(solved == YELLOW))
        if green_count == 0 or yellow_count == 0:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
