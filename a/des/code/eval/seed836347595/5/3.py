from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
MAROON = 9


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _largest_plan_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    grey_comps = _components(g == GREY)
    assert grey_comps, "expected at least one grey structure"
    plan = max(grey_comps, key=len)
    return _bbox(plan)


def _wall_axes(plan: np.ndarray) -> tuple[list[int], list[int]]:
    row_counts = np.sum(plan == GREY, axis=1)
    col_counts = np.sum(plan == GREY, axis=0)
    row_thresh = int((int(row_counts.max()) + int(row_counts.min())) // 2)
    col_thresh = int((int(col_counts.max()) + int(col_counts.min())) // 2)
    wall_rows = [int(i) for i in np.flatnonzero(row_counts > row_thresh)]
    wall_cols = [int(i) for i in np.flatnonzero(col_counts > col_thresh)]

    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected outer room frame"
    assert wall_rows[0] == 0 and wall_rows[-1] == plan.shape[0] - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == plan.shape[1] - 1
    assert all(b - a > 1 for a, b in zip(wall_rows, wall_rows[1:], strict=False))
    assert all(b - a > 1 for a, b in zip(wall_cols, wall_cols[1:], strict=False))
    return wall_rows, wall_cols


def _door_cells(segment: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(segment != GREY)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify rooms separated by grey walls, close doors between rooms with different
    red-marker counts, and fill final room components by open-door connectivity.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _largest_plan_bbox(g)
    plan = g[r0 : r1 + 1, c0 : c1 + 1]
    wall_rows, wall_cols = _wall_axes(plan)

    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    room_count = room_rows * room_cols

    def room_id(rr: int, cc: int) -> int:
        return rr * room_cols + cc

    room_bounds: dict[int, tuple[int, int, int, int]] = {}
    red_counts = [0] * room_count
    for rr in range(room_rows):
        rs = wall_rows[rr] + 1
        re = wall_rows[rr + 1]
        for cc in range(room_cols):
            cs = wall_cols[cc] + 1
            ce = wall_cols[cc + 1]
            rid = room_id(rr, cc)
            room_bounds[rid] = (rs, re, cs, ce)
            red_counts[rid] = int(np.sum(plan[rs:re, cs:ce] == RED))

    dsu = _DSU(room_count)
    open_doors: list[tuple[list[tuple[int, int]], int, int]] = []

    for rr in range(room_rows):
        rs = wall_rows[rr] + 1
        re = wall_rows[rr + 1]
        for cc in range(1, len(wall_cols) - 1):
            wc = wall_cols[cc]
            gaps = _door_cells(plan[rs:re, wc])
            if not gaps:
                continue
            left_id = room_id(rr, cc - 1)
            right_id = room_id(rr, cc)
            if red_counts[left_id] == red_counts[right_id]:
                cells = [(rs + gap, wc) for gap in gaps]
                open_doors.append((cells, left_id, right_id))
                dsu.union(left_id, right_id)

    for rr in range(1, len(wall_rows) - 1):
        wr = wall_rows[rr]
        for cc in range(room_cols):
            cs = wall_cols[cc] + 1
            ce = wall_cols[cc + 1]
            gaps = _door_cells(plan[wr, cs:ce])
            if not gaps:
                continue
            top_id = room_id(rr - 1, cc)
            bottom_id = room_id(rr, cc)
            if red_counts[top_id] == red_counts[bottom_id]:
                cells = [(wr, cs + gap) for gap in gaps]
                open_doors.append((cells, top_id, bottom_id))
                dsu.union(top_id, bottom_id)

    comp_sizes: dict[int, int] = defaultdict(int)
    for rid in range(room_count):
        comp_sizes[dsu.find(rid)] += 1

    out = np.zeros_like(g)
    out_plan = out[r0 : r1 + 1, c0 : c1 + 1]
    out_plan[wall_rows, :] = GREY
    out_plan[:, wall_cols] = GREY

    room_colors = [MAROON] * room_count
    for rid in range(room_count):
        if comp_sizes[dsu.find(rid)] > 1:
            room_colors[rid] = GREEN

    for rid, color in enumerate(room_colors):
        rs, re, cs, ce = room_bounds[rid]
        out_plan[rs:re, cs:ce] = color

    for cells, ra, _rb in open_doors:
        color = room_colors[ra]
        for rr, cc in cells:
            out_plan[rr, cc] = color

    return out


# --------------- generator ---------------

ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 5)
ROOM_SIZE_RANGE = (3, 6)
LEGEND_SIZE = 7
LEGEND_GAP = 2
GRID_EXTRA_RANGE = (1, 5)
GEN_RETRIES = 500


def _wall_layout(sizes: list[int]) -> tuple[list[int], list[tuple[int, int]], int]:
    walls = [0]
    spans: list[tuple[int, int]] = []
    pos = 0
    for size in sizes:
        start = pos + 1
        end = start + size - 1
        spans.append((start, end))
        pos = end + 1
        walls.append(pos)
    return walls, spans, walls[-1] + 1


def _legend_patch() -> np.ndarray:
    patch = np.zeros((LEGEND_SIZE, LEGEND_SIZE), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[2, 1:4] = GREY
    patch[2, 2] = BLACK
    patch[2, 5] = GREEN
    patch[4, 1:4] = GREY
    patch[4, 5] = MAROON
    return patch


def _color_components(
    comp_adj: dict[int, set[int]], rng: random.Random
) -> dict[int, int] | None:
    nodes = sorted(comp_adj, key=lambda node: len(comp_adj[node]), reverse=True)
    palette = [0, 1, 2, 3, 4]
    rng.shuffle(palette)
    assignment: dict[int, int] = {}

    def dfs(idx: int) -> bool:
        if idx == len(nodes):
            return True
        node = nodes[idx]
        used = {assignment[nbr] for nbr in comp_adj[node] if nbr in assignment}
        choices = palette[:]
        rng.shuffle(choices)
        for color in choices:
            if color in used:
                continue
            assignment[node] = color
            if dfs(idx + 1):
                return True
            del assignment[node]
        return False

    if dfs(0):
        return assignment
    return None


def _sample_markers(
    rs: int, re: int, cs: int, ce: int, count: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    cells = [(r, c) for r in range(rs, re) for c in range(cs, ce)]
    for _ in range(50):
        rng.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _expected_output(
    shape: tuple[int, int],
    plan_origin: tuple[int, int],
    row_walls: list[int],
    col_walls: list[int],
    room_spans_r: list[tuple[int, int]],
    room_spans_c: list[tuple[int, int]],
    open_edges: list[dict[str, object]],
) -> np.ndarray:
    room_rows = len(room_spans_r)
    room_cols = len(room_spans_c)
    total_rooms = room_rows * room_cols

    def room_id(rr: int, cc: int) -> int:
        return rr * room_cols + cc

    dsu = _DSU(total_rooms)
    for edge in open_edges:
        dsu.union(int(edge["a"]), int(edge["b"]))

    comp_sizes: dict[int, int] = defaultdict(int)
    for rid in range(total_rooms):
        comp_sizes[dsu.find(rid)] += 1

    out = np.zeros(shape, dtype=int)
    pr0, pc0 = plan_origin
    plan_h = row_walls[-1] + 1
    plan_w = col_walls[-1] + 1
    plan = out[pr0 : pr0 + plan_h, pc0 : pc0 + plan_w]
    plan[row_walls, :] = GREY
    plan[:, col_walls] = GREY

    room_colors = [MAROON] * total_rooms
    for rid in range(total_rooms):
        if comp_sizes[dsu.find(rid)] > 1:
            room_colors[rid] = GREEN

    for rr, (rs, re) in enumerate(room_spans_r):
        for cc, (cs, ce) in enumerate(room_spans_c):
            plan[rs : re + 1, cs : ce + 1] = room_colors[room_id(rr, cc)]

    for edge in open_edges:
        gr, gc = edge["gap"]
        plan[int(gr), int(gc)] = GREEN

    return out


def _plan_origin(
    grid_h: int,
    grid_w: int,
    plan_h: int,
    plan_w: int,
    corner: str,
    rng: random.Random,
) -> tuple[int, int] | None:
    pad = LEGEND_SIZE + LEGEND_GAP
    if corner == "tl":
        r_lo, r_hi = pad, grid_h - plan_h
        c_lo, c_hi = pad, grid_w - plan_w
    elif corner == "tr":
        r_lo, r_hi = pad, grid_h - plan_h
        c_lo, c_hi = 0, grid_w - plan_w - pad
    elif corner == "bl":
        r_lo, r_hi = 0, grid_h - plan_h - pad
        c_lo, c_hi = pad, grid_w - plan_w
    else:
        r_lo, r_hi = 0, grid_h - plan_h - pad
        c_lo, c_hi = 0, grid_w - plan_w - pad

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return rng.randint(r_lo, r_hi), rng.randint(c_lo, c_hi)


def _legend_origin(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - LEGEND_SIZE
    if corner == "bl":
        return h - LEGEND_SIZE, 0
    return h - LEGEND_SIZE, w - LEGEND_SIZE


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.randint(*ROOM_ROWS_RANGE)
    room_cols = rng.randint(*ROOM_COLS_RANGE)
    heights = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    row_walls, row_spans, plan_h = _wall_layout(heights)
    col_walls, col_spans, plan_w = _wall_layout(widths)

    grid_h = plan_h + LEGEND_SIZE + LEGEND_GAP + rng.randint(*GRID_EXTRA_RANGE)
    grid_w = plan_w + LEGEND_SIZE + LEGEND_GAP + rng.randint(*GRID_EXTRA_RANGE)
    corner = rng.choice(["tl", "tr", "bl", "br"])
    origin = _plan_origin(grid_h, grid_w, plan_h, plan_w, corner, rng)
    if origin is None:
        return None

    total_rooms = room_rows * room_cols

    def room_id(rr: int, cc: int) -> int:
        return rr * room_cols + cc

    edges: list[dict[str, object]] = []
    for rr in range(room_rows):
        for cc in range(room_cols - 1):
            rs, re = row_spans[rr]
            gap_r = rng.randint(rs, re)
            edges.append(
                {
                    "a": room_id(rr, cc),
                    "b": room_id(rr, cc + 1),
                    "gap": (gap_r, col_walls[cc + 1]),
                    "same_component_only_absent": False,
                }
            )
    for rr in range(room_rows - 1):
        for cc in range(room_cols):
            cs, ce = col_spans[cc]
            gap_c = rng.randint(cs, ce)
            edges.append(
                {
                    "a": room_id(rr, cc),
                    "b": room_id(rr + 1, cc),
                    "gap": (row_walls[rr + 1], gap_c),
                    "same_component_only_absent": False,
                }
            )

    open_mask = [rng.random() < 0.35 for _ in edges]
    dsu = _DSU(total_rooms)
    open_edges: list[dict[str, object]] = []
    for edge, is_open in zip(edges, open_mask, strict=False):
        if is_open:
            dsu.union(int(edge["a"]), int(edge["b"]))
            open_edges.append(edge)

    comp_members: dict[int, list[int]] = defaultdict(list)
    for rid in range(total_rooms):
        comp_members[dsu.find(rid)].append(rid)
    comp_sizes = [len(v) for v in comp_members.values()]
    if not open_edges or max(comp_sizes) <= 1 or all(size > 1 for size in comp_sizes):
        return None
    if max(comp_sizes) == total_rooms:
        return None

    root_to_comp = {root: idx for idx, root in enumerate(comp_members)}
    room_comp = {rid: root_to_comp[dsu.find(rid)] for rid in range(total_rooms)}

    comp_adj: dict[int, set[int]] = {idx: set() for idx in range(len(comp_members))}
    closed_candidates: list[dict[str, object]] = []
    for edge in edges:
        ca = room_comp[int(edge["a"])]
        cb = room_comp[int(edge["b"])]
        if ca == cb:
            continue
        comp_adj[ca].add(cb)
        comp_adj[cb].add(ca)
        closed_candidates.append(edge)
    if not closed_candidates:
        return None

    comp_counts = _color_components(comp_adj, rng)
    if comp_counts is None:
        return None

    target_closed = min(
        len(closed_candidates),
        max(1, len(open_edges) + rng.randint(-1, 1)),
    )
    closed_edges = rng.sample(closed_candidates, target_closed)
    if not closed_edges:
        return None

    room_counts = [0] * total_rooms
    for rid in range(total_rooms):
        room_counts[rid] = comp_counts[room_comp[rid]]

    inp = np.zeros((grid_h, grid_w), dtype=int)
    pr0, pc0 = origin
    plan = inp[pr0 : pr0 + plan_h, pc0 : pc0 + plan_w]
    plan[row_walls, :] = GREY
    plan[:, col_walls] = GREY

    door_cells = {tuple(edge["gap"]) for edge in open_edges + closed_edges}
    for rr, cc in door_cells:
        plan[int(rr), int(cc)] = BLACK

    for rr, (rs, re) in enumerate(row_spans):
        for cc, (cs, ce) in enumerate(col_spans):
            rid = room_id(rr, cc)
            markers = _sample_markers(rs, re + 1, cs, ce + 1, room_counts[rid], rng)
            if markers is None:
                return None
            for mr, mc in markers:
                plan[mr, mc] = RED

    lg0, lg1 = _legend_origin(inp.shape, corner)
    legend = _legend_patch()
    inp[lg0 : lg0 + LEGEND_SIZE, lg1 : lg1 + LEGEND_SIZE] = legend

    expected = _expected_output(
        inp.shape,
        origin,
        row_walls,
        col_walls,
        row_spans,
        col_spans,
        open_edges,
    )

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door puzzle")
