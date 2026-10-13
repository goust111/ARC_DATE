import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SOURCE_COLORS = (RED, BLUE, YELLOW)
MIX_COLORS = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

_GEN_COUNTER = 0
_GEN_MAX_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points):
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rect(grid: np.ndarray, color: int, box):
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return np.all(patch == color)


def _is_grey_ring(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return (
        np.all(grid[r0, c0 : c1 + 1] == GREY)
        and np.all(grid[r1, c0 : c1 + 1] == GREY)
        and np.all(grid[r0 : r1 + 1, c0] == GREY)
        and np.all(grid[r0 : r1 + 1, c1] == GREY)
    )


def _find_containers(grid: np.ndarray):
    h, w = grid.shape
    containers = []
    seen_outer = set()

    def try_add(color: int, pts):
        box = _bbox(pts)
        r0, c0, r1, c1 = box
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            return
        if len(pts) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return
        if not _is_solid_rect(grid, color, box):
            return
        outer = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
        if outer in seen_outer:
            return
        if not _is_grey_ring(grid, outer):
            return
        seen_outer.add(outer)
        containers.append(
            {
                "inner": box,
                "outer": outer,
                "fill": color,
            }
        )

    black_mask = grid == BLACK
    for pts in _components(black_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        try_add(BLACK, pts)

    for color in sorted(int(v) for v in np.unique(grid) if v not in (BLACK, GREY)):
        for pts in _components(grid == color):
            try_add(color, pts)

    containers.sort(key=lambda item: item["outer"])
    return containers


def _paint_inner(out: np.ndarray, container, color: int):
    r0, c0, r1, c1 = container["inner"]
    out[r0 : r1 + 1, c0 : c1 + 1] = color


def _combine_colors(colors):
    if not colors:
        return None
    unique = set(colors)
    if len(unique) == 1:
        return colors[0]
    if len(unique) != 2:
        raise AssertionError("unsupported number of incoming colors")
    key = frozenset(unique)
    if key not in MIX_COLORS:
        raise AssertionError("unsupported color combination")
    return MIX_COLORS[key]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill empty rectangular containers based on connected pipes and valve colors.

    Containers are grey rectangular frames with monochrome interiors. Grey pipe
    bodies connect container walls; a non-grey pipe pixel is a valve whose color
    selects the only allowed source side. Valveless pipes send color from a
    filled container to an empty one. When two source colors enter the same
    empty container, use the fixed ARC mixing table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    containers = _find_containers(g)
    assert containers, "expected at least one container"

    h, w = g.shape
    wall_owner = np.full((h, w), -1, dtype=int)
    interior_owner = np.full((h, w), -1, dtype=int)
    for cid, container in enumerate(containers):
        ir0, ic0, ir1, ic1 = container["inner"]
        or0, oc0, or1, oc1 = container["outer"]
        interior_owner[ir0 : ir1 + 1, ic0 : ic1 + 1] = cid
        wall_owner[or0, oc0 : oc1 + 1] = cid
        wall_owner[or1, oc0 : oc1 + 1] = cid
        wall_owner[or0 : or1 + 1, oc0] = cid
        wall_owner[or0 : or1 + 1, oc1] = cid

    body_mask = (g != BLACK) & (interior_owner < 0) & (wall_owner < 0)
    edges = []
    for pts in _components(body_mask):
        touched = set()
        valve_colors = set()
        for r, c in pts:
            value = int(g[r, c])
            if value != GREY:
                valve_colors.add(value)
            for nr, nc in _neighbors4(r, c, h, w):
                cid = int(wall_owner[nr, nc])
                if cid >= 0:
                    touched.add(cid)
        if not touched:
            continue
        assert len(touched) == 2, "each pipe must touch exactly two containers"
        assert len(valve_colors) <= 1, "each pipe may have at most one valve color"
        edges.append(
            {
                "containers": tuple(sorted(touched)),
                "valve": next(iter(valve_colors)) if valve_colors else None,
            }
        )

    incoming = defaultdict(list)
    for edge in edges:
        a, b = edge["containers"]
        ca = int(containers[a]["fill"])
        cb = int(containers[b]["fill"])
        valve = edge["valve"]

        if valve is None:
            if ca != BLACK and cb == BLACK:
                incoming[b].append(ca)
            elif cb != BLACK and ca == BLACK:
                incoming[a].append(cb)
            continue

        match_a = ca != BLACK and ca == valve
        match_b = cb != BLACK and cb == valve
        if match_a and not match_b and cb == BLACK:
            incoming[b].append(ca)
        elif match_b and not match_a and ca == BLACK:
            incoming[a].append(cb)

    out = g.copy()
    for cid, colors in incoming.items():
        if int(containers[cid]["fill"]) != BLACK:
            continue
        fill = _combine_colors(colors)
        if fill is not None:
            _paint_inner(out, containers[cid], fill)
    return out


def _draw_container(grid: np.ndarray, top: int, left: int, height: int, width: int, fill: int):
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[top + 1 : bottom, left + 1 : right] = fill
    return {
        "outer": (top, left, bottom, right),
        "inner": (top + 1, left + 1, bottom - 1, right - 1),
        "fill": fill,
    }


def _slot_origin(margin: int, pitch: int, slot_r: int, slot_c: int):
    return margin + slot_r * pitch, margin + slot_c * pitch


def _place_container_in_slot(grid: np.ndarray, margin: int, pitch: int, slot_r: int, slot_c: int, fill: int, rng: random.Random):
    sr0, sc0 = _slot_origin(margin, pitch, slot_r, slot_c)
    max_outer = min(6, pitch - 2)
    height = rng.randint(4, max_outer)
    width = rng.randint(4, max_outer)
    top = rng.randint(sr0 + 1, sr0 + pitch - height - 1)
    left = rng.randint(sc0 + 1, sc0 + pitch - width - 1)
    return _draw_container(grid, top, left, height, width, fill)


def _pick_wall_cell(container, side: str, rng: random.Random):
    r0, c0, r1, c1 = container["outer"]
    if side == "top":
        c = rng.randint(c0 + 1, c1 - 1)
        return (r0, c), (r0 - 1, c)
    if side == "bottom":
        c = rng.randint(c0 + 1, c1 - 1)
        return (r1, c), (r1 + 1, c)
    if side == "left":
        r = rng.randint(r0 + 1, r1 - 1)
        return (r, c0), (r, c0 - 1)
    if side == "right":
        r = rng.randint(r0 + 1, r1 - 1)
        return (r, c1), (r, c1 + 1)
    raise ValueError(side)


def _line_cells(a, b):
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("line must be axis-aligned")


def _orthogonal_path(start, end, mode: str):
    if mode == "hv":
        bend = (start[0], end[1])
    elif mode == "vh":
        bend = (end[0], start[1])
    else:
        raise ValueError(mode)

    first = _line_cells(start, bend)
    second = _line_cells(bend, end)
    return first + second[1:]


def _draw_pipe(grid: np.ndarray, container_a, side_a: str, container_b, side_b: str, mode: str, valve_color, rng: random.Random):
    _, start = _pick_wall_cell(container_a, side_a, rng)
    _, end = _pick_wall_cell(container_b, side_b, rng)
    path = _orthogonal_path(start, end, mode)

    h, w = grid.shape
    for r, c in path:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BLACK:
            return False

    for r, c in path:
        grid[r, c] = GREY

    if valve_color is not None and len(path) >= 3:
        valve_index = rng.randint(1, len(path) - 2)
        vr, vc = path[valve_index]
        grid[vr, vc] = valve_color
    return True


def _mix_pair(a: int, b: int):
    if a == b:
        return a
    return MIX_COLORS[frozenset((a, b))]


def _transform(grid: np.ndarray, op: int):
    out = np.rot90(grid, k=op % 4)
    if op >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _crop_and_pad(inp: np.ndarray, out: np.ndarray, rng: random.Random):
    coords = np.argwhere(inp != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    inp = inp[r0 : r1 + 1, c0 : c1 + 1]
    out = out[r0 : r1 + 1, c0 : c1 + 1]
    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)
    shape = (inp.shape[0] + top + bottom, inp.shape[1] + left + right)
    padded_inp = np.full(shape, BLACK, dtype=int)
    padded_out = np.full(shape, BLACK, dtype=int)
    padded_inp[top : top + inp.shape[0], left : left + inp.shape[1]] = inp
    padded_out[top : top + out.shape[0], left : left + out.shape[1]] = out
    return padded_inp, padded_out


def _single_case(case_id: int, rng: random.Random):
    pitch = rng.randint(7, 9)
    margin = rng.randint(2, 3)
    side = margin * 2 + pitch * 3
    inp = np.full((side, side), BLACK, dtype=int)

    source_color = rng.choice(SOURCE_COLORS)
    source = _place_container_in_slot(inp, margin, pitch, 0, 0, source_color, rng)
    target = _place_container_in_slot(inp, margin, pitch, 2, 2, BLACK, rng)
    route = rng.choice(
        [
            ("right", "top", "hv"),
            ("bottom", "left", "vh"),
        ]
    )
    valve = source_color if case_id == 1 else None
    if not _draw_pipe(inp, source, route[0], target, route[1], route[2], valve, rng):
        return None

    out = inp.copy()
    _paint_inner(out, target, source_color)
    return inp, out


def _merge_case(case_id: int, rng: random.Random):
    pitch = rng.randint(7, 9)
    margin = rng.randint(2, 3)
    side = margin * 2 + pitch * 3
    inp = np.full((side, side), BLACK, dtype=int)

    if case_id == 2:
        c0 = c1 = rng.choice(SOURCE_COLORS)
    elif case_id == 3:
        c0, c1 = RED, BLUE
    elif case_id == 4:
        c0, c1 = RED, YELLOW
    elif case_id == 5:
        c0, c1 = BLUE, YELLOW
    else:
        raise ValueError(case_id)

    source_left = _place_container_in_slot(inp, margin, pitch, 0, 0, c0, rng)
    source_right = _place_container_in_slot(inp, margin, pitch, 0, 2, c1, rng)
    target = _place_container_in_slot(inp, margin, pitch, 2, 1, BLACK, rng)

    valve_left = c0 if rng.choice([False, True]) else None
    valve_right = c1 if rng.choice([False, True]) else None
    ok_left = _draw_pipe(inp, source_left, "right", target, "left", "hv", valve_left, rng)
    ok_right = _draw_pipe(inp, source_right, "left", target, "right", "hv", valve_right, rng)
    if not (ok_left and ok_right):
        return None

    out = inp.copy()
    _paint_inner(out, target, _mix_pair(c0, c1))
    return inp, out


def _gen_attempt(case_id: int, rng: random.Random):
    if case_id in (0, 1):
        result = _single_case(case_id, rng)
    else:
        result = _merge_case(case_id, rng)
    if result is None:
        return None

    inp, out = result
    op = rng.randrange(8)
    inp = _transform(inp, op)
    out = _transform(out, op)
    inp, out = _crop_and_pad(inp, out, rng)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    case_id = _GEN_COUNTER % 6
    _GEN_COUNTER += 1

    rng = random.Random()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(case_id, rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
