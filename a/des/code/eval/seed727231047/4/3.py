from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

PATH_COLORS = (BLUE, RED, GREEN, YELLOW)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OUTER_RING = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_nodes(grid: np.ndarray, legend_mask: np.ndarray) -> tuple[list[dict], np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    nodes: list[dict] = []
    node_of = -np.ones((h, w), dtype=int)

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if bool(legend_mask[r0 : r0 + 3, c0 : c0 + 3].any()):
                continue
            if bool((node_of[r0 : r0 + 3, c0 : c0 + 3] >= 0).any()):
                continue

            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if int(patch[1, 1]) not in (GREY, MAROON):
                continue
            if not bool((patch[OUTER_RING] == GREY).all()):
                continue
            if not bool(np.isin(patch, [GREY, MAROON]).all()):
                continue
            if int((patch == MAROON).sum()) not in (0, 1):
                continue
            if int((patch == MAROON).sum()) == 1 and int(patch[1, 1]) != MAROON:
                continue

            idx = len(nodes)
            center = (r0 + 1, c0 + 1)
            nodes.append(
                {
                    "idx": idx,
                    "r0": r0,
                    "c0": c0,
                    "center": center,
                    "source": bool(int(patch[1, 1]) == MAROON),
                }
            )
            node_of[r0 : r0 + 3, c0 : c0 + 3] = idx

    source_ids = [int(node["idx"]) for node in nodes if bool(node["source"])]
    assert len(source_ids) == 1, "must have exactly one source node"
    return nodes, node_of, source_ids[0]


def _priority_order(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[h - 2 : h, w - 2 : w] = True
    order = [int(v) for v in grid[h - 2 : h, w - 2 : w].reshape(-1).tolist()]
    assert sorted(order) == sorted(PATH_COLORS), "legend must be a permutation of the four path colors"
    return order, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a signal along colored paths, choosing the higher-priority color at orange junctions,
    and light every reached 3x3 grey node with the incoming path color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    order, legend_mask = _priority_order(g)
    prio = {int(col): i for i, col in enumerate(order)}
    nodes, node_of, source_idx = _find_nodes(g, legend_mask)

    out = g.copy()
    active_nodes: set[int] = set()
    seen_states: set[tuple[int, int, int, int]] = set()
    q: deque[tuple[tuple[int, int], tuple[int, int]]] = deque()

    def node_entries(idx: int) -> list[tuple[int, int]]:
        r0 = int(nodes[idx]["r0"])
        c0 = int(nodes[idx]["c0"])
        entries = set()
        for r in range(r0, r0 + 3):
            for c in range(c0, c0 + 3):
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    if node_of[nr, nc] >= 0 or legend_mask[nr, nc]:
                        continue
                    if int(g[nr, nc]) in PATH_COLORS or int(g[nr, nc]) == ORANGE:
                        entries.add((nr, nc))
        return sorted(entries)

    def entry_prev_inside(idx: int, entry: tuple[int, int]) -> tuple[int, int]:
        er, ec = entry
        inside = [(nr, nc) for nr, nc in _neighbors4(er, ec) if _in_bounds(h, w, nr, nc) and int(node_of[nr, nc]) == idx]
        assert inside, "entry cell must touch its node"
        return inside[0]

    def enqueue(prev: tuple[int, int], cur: tuple[int, int]) -> None:
        state = (int(prev[0]), int(prev[1]), int(cur[0]), int(cur[1]))
        if state in seen_states:
            return
        seen_states.add(state)
        q.append((prev, cur))

    def activate_node(idx: int, incoming_color: int | None, arrival_entry: tuple[int, int] | None) -> None:
        if idx in active_nodes:
            return
        active_nodes.add(idx)

        if idx != source_idx and incoming_color is not None:
            r0 = int(nodes[idx]["r0"])
            c0 = int(nodes[idx]["c0"])
            patch = out[r0 : r0 + 3, c0 : c0 + 3]
            patch[patch == GREY] = int(incoming_color)

        for entry in node_entries(idx):
            if arrival_entry is not None and entry == arrival_entry:
                continue
            enqueue(entry_prev_inside(idx, entry), entry)

    activate_node(source_idx, incoming_color=None, arrival_entry=None)

    while q:
        prev, cur = q.popleft()
        cr, cc = cur
        if not _in_bounds(h, w, cr, cc):
            continue
        if bool(legend_mask[cr, cc]) or int(node_of[cr, cc]) >= 0:
            continue

        val = int(g[cr, cc])

        if val in PATH_COLORS:
            for nr, nc in _neighbors4(cr, cc):
                if not _in_bounds(h, w, nr, nc) or (nr, nc) == prev or bool(legend_mask[nr, nc]):
                    continue
                if int(node_of[nr, nc]) >= 0:
                    activate_node(int(node_of[nr, nc]), val, cur)
                    continue
                nv = int(g[nr, nc])
                if nv == val or nv == ORANGE:
                    enqueue(cur, (nr, nc))
            continue

        if val != ORANGE:
            continue

        color_neighbors: dict[int, list[tuple[int, int]]] = {}
        for nr, nc in _neighbors4(cr, cc):
            if not _in_bounds(h, w, nr, nc) or bool(legend_mask[nr, nc]) or int(node_of[nr, nc]) >= 0:
                continue
            nv = int(g[nr, nc])
            if nv in PATH_COLORS:
                color_neighbors.setdefault(int(nv), []).append((nr, nc))

        if not color_neighbors:
            continue

        chosen = min(color_neighbors, key=lambda col: int(prio[int(col)]))
        for nxt in color_neighbors[chosen]:
            if nxt == prev:
                continue
            enqueue(cur, nxt)

    return out


def _box_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _node_box(center: tuple[int, int], pad: int = 0) -> tuple[int, int, int, int]:
    r, c = center
    return (r - 1 - pad, c - 1 - pad, r + 1 + pad, c + 1 + pad)


def _reserve_node(
    center: tuple[int, int],
    h: int,
    w: int,
    reserved_boxes: list[tuple[int, int, int, int]],
    forbidden_box: tuple[int, int, int, int],
) -> bool:
    r, c = center
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    real_box = _node_box(center, pad=0)
    if _box_intersects(real_box, forbidden_box):
        return False
    expanded = _node_box(center, pad=1)
    for box in reserved_boxes:
        if _box_intersects(expanded, box):
            return False
    reserved_boxes.append(expanded)
    return True


def _place_node(grid: np.ndarray, center: tuple[int, int], source: bool) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = GREY
    if source:
        grid[r, c] = MAROON


def _attach(center: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = center
    if side == "N":
        return (r - 2, c)
    if side == "S":
        return (r + 2, c)
    if side == "W":
        return (r, c - 2)
    if side == "E":
        return (r, c + 2)
    raise ValueError(f"unknown side {side}")


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment must be axis-aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment_cells(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _draw_polyline(
    grid: np.ndarray,
    points: list[tuple[int, int]],
    color: int,
    node_mask: np.ndarray,
    allowed_node_touch: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c in _polyline_cells(points):
        if not _in_bounds(h, w, r, c):
            return False
        cur = int(grid[r, c])
        if cur == ORANGE:
            pass
        elif cur == BLACK or cur == int(color):
            grid[r, c] = int(color)
        else:
            return False

        if (r, c) in allowed_node_touch:
            continue
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(h, w, nr, nc) and bool(node_mask[nr, nc]):
                return False
    return True


def _draw_child_path(
    grid: np.ndarray,
    node_mask: np.ndarray,
    parent: tuple[int, int],
    child: tuple[int, int],
    color: int,
) -> bool:
    pr, pc = parent
    cr, cc = child

    if pr == cr:
        start = _attach(parent, "E" if cc > pc else "W")
        end = _attach(child, "W" if cc > pc else "E")
        return _draw_polyline(grid, [start, end], color, node_mask, {start, end})

    if pc == cc:
        start = _attach(parent, "S" if cr > pr else "N")
        end = _attach(child, "N" if cr > pr else "S")
        return _draw_polyline(grid, [start, end], color, node_mask, {start, end})

    start = _attach(parent, "E" if cc > pc else "W")
    end = _attach(child, "N" if cr > pr else "S")
    bend = (start[0], end[1])
    return _draw_polyline(grid, [start, bend, end], color, node_mask, {start, end})


def _junction_columns(src_c: int, end_c: int, count: int) -> list[int] | None:
    cols: list[int] = []
    lo = src_c + 4
    for remaining in range(count, 0, -1):
        hi = end_c - 4 * remaining
        if lo > hi:
            return None
        col = random.randint(lo, hi)
        cols.append(col)
        lo = col + 4
    return cols


def _junction_rows(src_r: int, end_r: int, count: int) -> list[int] | None:
    rows: list[int] = []
    lo = src_r + 4
    for remaining in range(count, 0, -1):
        hi = end_r - 4 * remaining
        if lo > hi:
            return None
        row = random.randint(lo, hi)
        rows.append(row)
        lo = row + 4
    return rows


def _plan_child_from_horizontal_branch(
    branch_center: tuple[int, int],
    branch_sign: int,
    h: int,
    w: int,
    reserved_boxes: list[tuple[int, int, int, int]],
    forbidden_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int] | None, str | None]:
    if random.random() < 0.4:
        return None, None

    br, bc = branch_center
    modes = ["straight", "bend"]
    random.shuffle(modes)
    for mode in modes:
        delta_c = random.randint(5, 7)
        child_c = bc + delta_c
        if mode == "straight":
            child_r = br
        else:
            child_r = br + branch_sign * random.randint(4, 5)
        child = (child_r, child_c)
        if _reserve_node(child, h, w, reserved_boxes, forbidden_box):
            return child, mode
    return None, None


def _plan_child_from_vertical_branch(
    branch_center: tuple[int, int],
    branch_sign: int,
    h: int,
    w: int,
    reserved_boxes: list[tuple[int, int, int, int]],
    forbidden_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int] | None, str | None]:
    if random.random() < 0.4:
        return None, None

    br, bc = branch_center
    modes = ["straight", "bend"]
    random.shuffle(modes)
    for mode in modes:
        delta_c = random.randint(4, 6)
        child_c = bc + branch_sign * delta_c
        if mode == "straight":
            child_r = br
        else:
            child_r = br + random.randint(4, 6)
        child = (child_r, child_c)
        if _reserve_node(child, h, w, reserved_boxes, forbidden_box):
            return child, mode
    return None, None


def _oracle_non_source_colored(inp: np.ndarray, out: np.ndarray) -> int:
    changed = (inp != out) & (inp == GREY)
    return int(changed.sum())


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(19, 27)
    w = random.randint(21, 29)
    priority = random.sample(list(PATH_COLORS), k=4)
    forbidden_box = (h - 2, w - 2, h - 1, w - 1)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[h - 2 : h, w - 2 : w] = np.array(priority, dtype=int).reshape(2, 2)

    reserved_boxes: list[tuple[int, int, int, int]] = []
    src = (random.randint(6, h - 9), random.randint(5, min(8, w - 12)))
    if not _reserve_node(src, h, w, reserved_boxes, forbidden_box):
        return None

    use_vertical = random.random() < 0.6

    lanes: list[dict] = []
    nodes: list[tuple[tuple[int, int], bool]] = [(src, True)]

    # Horizontal lane is always present.
    h_junction_count = random.randint(1, 3 if w >= 26 else 2)
    h_end_min = src[1] + 8 + 4 * (h_junction_count - 1)
    if h_end_min > w - 5:
        return None
    h_end = (src[0], random.randint(h_end_min, w - 5))
    if not _reserve_node(h_end, h, w, reserved_boxes, forbidden_box):
        return None
    nodes.append((h_end, False))

    h_cols = _junction_columns(src[1], h_end[1], h_junction_count)
    if h_cols is None:
        return None

    h_lane = {
        "kind": "H",
        "color": random.choice(PATH_COLORS),
        "end": h_end,
        "junctions": [],
    }

    for cj in h_cols:
        choices = []
        for sign in random.sample([-1, 1], k=2):
            for delta in random.sample([4, 5, 6], k=3):
                branch_center = (src[0] + sign * delta, cj)
                boxes_copy = list(reserved_boxes)
                if not _reserve_node(branch_center, h, w, boxes_copy, forbidden_box):
                    continue
                child_boxes = list(boxes_copy)
                child_center, child_mode = _plan_child_from_horizontal_branch(
                    branch_center,
                    sign,
                    h,
                    w,
                    child_boxes,
                    forbidden_box,
                )
                choices.append((branch_center, sign, child_center, child_mode, child_boxes))
        if not choices:
            return None
        branch_center, sign, child_center, child_mode, chosen_boxes = random.choice(choices)
        reserved_boxes = chosen_boxes
        nodes.append((branch_center, False))
        if child_center is not None:
            nodes.append((child_center, False))
        h_lane["junctions"].append(
            {
                "pos": (src[0], cj),
                "branch_center": branch_center,
                "branch_sign": sign,
                "branch_color": random.choice([c for c in PATH_COLORS if c != h_lane["color"]]),
                "child_center": child_center,
                "child_mode": child_mode,
                "child_color": random.choice(PATH_COLORS) if child_center is not None else None,
            }
        )
    lanes.append(h_lane)

    if use_vertical:
        v_junction_count = random.randint(1, 2 if h < 25 else 3)
        v_end_min = src[0] + 8 + 4 * (v_junction_count - 1)
        if v_end_min > h - 5:
            return None
        v_end = (random.randint(v_end_min, h - 5), src[1])
        if not _reserve_node(v_end, h, w, reserved_boxes, forbidden_box):
            return None
        nodes.append((v_end, False))

        v_rows = _junction_rows(src[0], v_end[0], v_junction_count)
        if v_rows is None:
            return None

        v_lane = {
            "kind": "V",
            "color": random.choice(PATH_COLORS),
            "end": v_end,
            "junctions": [],
        }

        for rj in v_rows:
            choices = []
            sign_options = [1]
            if src[1] >= 6:
                sign_options.append(-1)
            random.shuffle(sign_options)
            for sign in sign_options:
                for delta in random.sample([4, 5, 6], k=3):
                    branch_center = (rj, src[1] + sign * delta)
                    boxes_copy = list(reserved_boxes)
                    if not _reserve_node(branch_center, h, w, boxes_copy, forbidden_box):
                        continue
                    child_boxes = list(boxes_copy)
                    child_center, child_mode = _plan_child_from_vertical_branch(
                        branch_center,
                        sign,
                        h,
                        w,
                        child_boxes,
                        forbidden_box,
                    )
                    choices.append((branch_center, sign, child_center, child_mode, child_boxes))
            if not choices:
                return None
            branch_center, sign, child_center, child_mode, chosen_boxes = random.choice(choices)
            reserved_boxes = chosen_boxes
            nodes.append((branch_center, False))
            if child_center is not None:
                nodes.append((child_center, False))
            v_lane["junctions"].append(
                {
                    "pos": (rj, src[1]),
                    "branch_center": branch_center,
                    "branch_sign": sign,
                    "branch_color": random.choice([c for c in PATH_COLORS if c != v_lane["color"]]),
                    "child_center": child_center,
                    "child_mode": child_mode,
                    "child_color": random.choice(PATH_COLORS) if child_center is not None else None,
                }
            )
        lanes.append(v_lane)

    # Render nodes first.
    for center, source in nodes:
        _place_node(grid, center, source)
    node_mask = np.isin(grid, [GREY, MAROON])

    # Render orange junctions.
    for lane in lanes:
        for junction in lane["junctions"]:
            jr, jc = junction["pos"]
            if int(grid[jr, jc]) != BLACK:
                return None
            grid[jr, jc] = ORANGE

    # Render trunks.
    for lane in lanes:
        if lane["kind"] == "H":
            points = [_attach(src, "E")] + [tuple(j["pos"]) for j in lane["junctions"]] + [_attach(lane["end"], "W")]
        else:
            points = [_attach(src, "S")] + [tuple(j["pos"]) for j in lane["junctions"]] + [_attach(lane["end"], "N")]
        if not _draw_polyline(grid, points, int(lane["color"]), node_mask, {points[0], points[-1]}):
            return None

    # Render branches and children.
    for lane in lanes:
        for junction in lane["junctions"]:
            branch_center = tuple(junction["branch_center"])
            jr, jc = junction["pos"]
            br, bc = branch_center
            if lane["kind"] == "H":
                end = _attach(branch_center, "S" if br < jr else "N")
                if not _draw_polyline(grid, [(jr, jc), end], int(junction["branch_color"]), node_mask, {end}):
                    return None
            else:
                end = _attach(branch_center, "E" if bc < jc else "W")
                if not _draw_polyline(grid, [(jr, jc), end], int(junction["branch_color"]), node_mask, {end}):
                    return None

            child_center = junction["child_center"]
            if child_center is not None:
                if not _draw_child_path(
                    grid,
                    node_mask,
                    branch_center,
                    tuple(child_center),
                    int(junction["child_color"]),
                ):
                    return None

    inp = grid
    out = main(inp)

    if _oracle_non_source_colored(inp, out) == 0:
        return None
    if int((inp == ORANGE).sum()) < 1:
        return None
    if len(nodes) < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
