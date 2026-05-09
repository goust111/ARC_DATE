# Combo variant of 88bcf3b4 (index 5)
# combo_spec: D5/0+D6/1
# Name: 多障碍物串联绕行

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, paint_coords

# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior:
#    Original 88bcf3b4 routes a segment around one nearby obstacle and then exits.
#    Here each triple owns 2-3 obstacles ahead of the bar. The output path is built
#    as a sequence of obstacle-modules: approach diagonally on the current side,
#    skim forward along the obstacle's outer side, then cross diagonally past the
#    centerline to the opposite side. Repeating this makes the segment visibly snake
#    between multiple blockers before its final diagonal tail.
#    Example in local coords, side=+1:
#      start (0,1) -> diag to apex (1,2) -> obstacle anchored at (1,1)
#      -> cross to opposite side -> next apex on the left -> cross again -> exit.
#    The key delta from the base task is that "nearest obstacle" is no longer the
#    whole story; the segment consumes length over several wraps in sorted distance order.
# b) Complexity calibration:
#    A single obstacle would visually collapse back to the original task, so the
#    generator always creates 2-3 obstacle components. Approach lengths are limited
#    to {1,2} and obstacle forward spans to {0,1,2}; this keeps the cases legible
#    while still producing repeated left/right side flips that are easy to see.
# c) Hidden assumptions removed/replaced:
#    - The original solver assumed exactly one relevant obstacle component.
#    - It also assumed a generic BFS around arbitrary skins.
#    In this combo, the solver instead assumes obstacle components are arranged in
#    forward distance order and each obstacle exposes an "anchor" cell on the side
#    currently being wrapped. The generator enforces that by growing each obstacle
#    from the apex-inner anchor only inward/forward, making the anchor recoverable.
# d) Placement legality:
#    Generator placement uses output-first local construction, transforms triples to
#    each border, rejects overlaps/out-of-bounds placements, and enforces that a
#    triple's farthest owned obstacle is still closer to its terminal than any foreign
#    obstacle. That keeps obstacle ownership unambiguous when multiple triples exist.
# e/f/g) Observability and cleanliness:
#    No hidden color ordering is used. The observable cue is geometric: multiple
#    separate obstacle sprites in front of one bar, and an output segment that flips
#    side at each obstacle. Colors stay simple (bg + bar + segment + obstacle), and
#    the reserved placement padding prevents clutter between different triples.


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _is_straight_line(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        cols = sorted(set(cs))
        return cols == list(range(min(cols), max(cols) + 1))
    if len(set(cs)) == 1:
        rows = sorted(set(rs))
        return rows == list(range(min(rows), max(rows) + 1))
    return False


def _bar_endpoints(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        c = cs[0]
        return (min(rs), c), (max(rs), c)
    r = rs[0]
    return (r, min(cs)), (r, max(cs))


def _touches_border(p: tuple[int, int], h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] - b[0], a[1] - b[1])


def _mul(v: tuple[int, int], k: int) -> tuple[int, int]:
    return (v[0] * k, v[1] * k)


def _dot(v: tuple[int, int], basis: tuple[int, int]) -> int:
    return v[0] * basis[0] + v[1] * basis[1]


def _segment_main_path(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)

    def neigh(p: tuple[int, int]):
        r, c = p
        for q in neighbors4(r, c):
            if q in sset:
                yield q
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            q = (r + dr, c + dc)
            if q not in sset:
                continue
            mid1 = (r, c + dc)
            mid2 = (r + dr, c)
            if mid1 in sset or mid2 in sset:
                continue
            yield q

    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist: dict[tuple[int, int], int] = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in neigh(cur):
            if nxt in prev:
                continue
            prev[nxt] = cur
            dist[nxt] = dist[cur] + 1
            q.append(nxt)

    far = max(dist.keys(), key=lambda p: (dist[p], p[0], p[1]))
    path = []
    cur = far
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _segment_degrees(seg_pts: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    sset = set(seg_pts)
    deg = {p: 0 for p in sset}
    for r, c in sset:
        for q in neighbors4(r, c):
            if q in sset:
                deg[(r, c)] += 1
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            q = (r + dr, c + dc)
            if q not in sset:
                continue
            mid1 = (r, c + dc)
            mid2 = (r + dr, c)
            if mid1 in sset or mid2 in sset:
                continue
            deg[(r, c)] += 1
    return deg


def _bar_forward(bar_comp: list[tuple[int, int]], h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    e0, e1 = _bar_endpoints(bar_comp)
    touch0 = _touches_border(e0, h, w)
    border_ep = e0 if touch0 else e1
    terminal = e1 if touch0 else e0
    rs = [r for r, _ in bar_comp]
    cs = [c for _, c in bar_comp]
    vertical = len(set(cs)) == 1
    if vertical:
        fwd = (1, 0) if border_ep[0] == 0 else (-1, 0)
    else:
        fwd = (0, 1) if border_ep[1] == 0 else (0, -1)
    return terminal, fwd


def _infer_anchor(
    comp: list[tuple[int, int]],
    terminal: tuple[int, int],
    fwd: tuple[int, int],
    lat: tuple[int, int],
) -> tuple[int, int]:
    def key(p: tuple[int, int]) -> tuple[int, int, int, int]:
        rel = _sub(p, terminal)
        outward = _dot(rel, lat)
        forward = _dot(rel, fwd)
        return (outward, -forward, -p[0], -p[1])

    return max(comp, key=key)


def _simulate_module(
    current: tuple[int, int],
    comp: list[tuple[int, int]],
    terminal: tuple[int, int],
    fwd: tuple[int, int],
    lat: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], dict]:
    comp_set = set(comp)
    anchor = _infer_anchor(comp, terminal, fwd, lat)
    apex = _add(anchor, lat)
    delta = _sub(apex, current)
    approach = _dot(delta, fwd)
    side_delta = _dot(delta, lat)
    if approach not in (1, 2) or side_delta != approach:
        raise ValueError("invalid approach geometry")
    if any(_dot(_sub(p, anchor), lat) > 0 for p in comp):
        raise ValueError("obstacle leaks outward from anchor")
    if any(_dot(_sub(p, anchor), fwd) < 0 for p in comp):
        raise ValueError("obstacle extends behind anchor")
    span = max(_dot(_sub(p, anchor), fwd) for p in comp)
    if span not in (0, 1, 2):
        raise ValueError("unexpected forward span")

    steps: list[tuple[int, int]] = []
    cur = current
    for _ in range(approach):
        cur = _add(cur, _add(fwd, lat))
        if cur in comp_set:
            raise ValueError("approach overlaps obstacle")
        steps.append(cur)
    for _ in range(span):
        cur = _add(cur, fwd)
        if cur in comp_set:
            raise ValueError("forward skim overlaps obstacle")
        steps.append(cur)
    exit_len = approach + 2
    for _ in range(exit_len):
        cur = _add(cur, _sub(fwd, lat))
        if cur in comp_set:
            raise ValueError("exit overlaps obstacle")
        steps.append(cur)
    return steps, cur, (-lat[0], -lat[1]), {"approach": approach, "span": span}


def _route_from_obstacles(
    g: np.ndarray,
    seg_color: int,
    terminal: tuple[int, int],
    seg_start: tuple[int, int],
    bar_comp: list[tuple[int, int]],
    obstacle_comps: list[list[tuple[int, int]]],
    total_len: int,
) -> list[tuple[int, int]]:
    h, w = g.shape
    _terminal2, fwd = _bar_forward(bar_comp, h, w)
    assert _terminal2 == terminal
    lat = _sub(seg_start, terminal)
    route = [seg_start]
    cur = seg_start
    for comp in obstacle_comps:
        steps, cur, lat, _meta = _simulate_module(cur, comp, terminal, fwd, lat)
        route.extend(steps)
    step = _add(fwd, lat)
    while len(route) < total_len:
        nxt = _add(cur, step)
        nr, nc = nxt
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(g[nr, nc]) not in (get_bg(g), seg_color):
            break
        route.append(nxt)
        cur = nxt
    return route[:total_len]


def _obstacle_sort_key(
    comp: list[tuple[int, int]],
    terminal: tuple[int, int],
    fwd: tuple[int, int],
) -> tuple[int, int, int, int]:
    md = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
    min_fwd = min(_dot(_sub(p, terminal), fwd) for p in comp)
    mr = min(r for r, _ in comp)
    mc = min(c for _, c in comp)
    return (md, min_fwd, mr, mc)


def _select_obstacles_for_bar(
    g: np.ndarray,
    colors: list[int],
    bar_col: int,
    seg_col: int,
    bar_comp: list[tuple[int, int]],
    terminal: tuple[int, int],
    seg_start: tuple[int, int],
    total_len: int,
) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    _, fwd = _bar_forward(bar_comp, h, w)
    candidates: list[list[tuple[int, int]]] = []
    for col in colors:
        if col in (bar_col, seg_col):
            continue
        for comp in _components(g, col, conn8=True):
            if max(_dot(_sub(p, terminal), fwd) for p in comp) <= 0:
                continue
            candidates.append(comp)
    candidates.sort(key=lambda comp: _obstacle_sort_key(comp, terminal, fwd))

    selected: list[list[tuple[int, int]]] = []
    cur = seg_start
    lat = _sub(seg_start, terminal)
    used_len = 1
    for comp in candidates:
        if len(selected) == 3:
            break
        try:
            steps, next_cur, next_lat, _meta = _simulate_module(cur, comp, terminal, fwd, lat)
        except ValueError:
            continue
        if used_len + len(steps) > total_len:
            continue
        selected.append(comp)
        cur = next_cur
        lat = next_lat
        used_len += len(steps)
    assert len(selected) >= 2, "must find 2-3 sequential obstacles"
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-thread each attached stroke through a chain of small blockers, preserving pixel count.

    Input structure:
    - Background plus triples made of: a straight border bar, an attached 1-cell-thick segment,
      and 2-3 small obstacle sprites scattered ahead of that bar.

    Rule:
    - For every valid bar, detach the attached segment from its input shape.
    - Rebuild it as a forward-moving snake: wrap the nearest obstacle first, continue to the
      second-nearest (and possibly third), then keep the final diagonal heading until length runs out.
    - Each wrap flips which side of the forward axis the segment occupies; obstacle pixels stay fixed.

    Key insight:
    - The obstacles form an ordered corridor in front of each bar. Recover the segment length and
      root side from the input, sort the nearby obstacle components by distance, then replay the same
      small wrap module around each obstacle in sequence.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    def bar_candidates():
        for col in colors:
            for comp in _components(g, col, conn8=False):
                if len(comp) < 2 or not _is_straight_line(comp):
                    continue
                e0, e1 = _bar_endpoints(comp)
                touch0 = _touches_border(e0, h, w)
                touch1 = _touches_border(e1, h, w)
                if touch0 == touch1:
                    continue
                rs = [r for r, _ in comp]
                cs = [c for _, c in comp]
                vertical = len(set(cs)) == 1
                border_ep = e0 if touch0 else e1
                if vertical:
                    if border_ep[0] not in (0, h - 1):
                        continue
                else:
                    if border_ep[1] not in (0, w - 1):
                        continue
                terminal = e1 if touch0 else e0
                tr, tc = terminal
                if not any(
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal

    moves: list[tuple[list[tuple[int, int]], int, list[tuple[int, int]]]] = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a segment neighbor"

        best_seg = None
        for cand_col, cand_start in neigh:
            for comp in _components(g, cand_col, conn8=True):
                if cand_start not in comp:
                    continue
                if len(comp) < 2:
                    break
                deg = _segment_degrees(comp)
                if deg.get(cand_start, 0) != 1:
                    break
                path = _segment_main_path(comp, cand_start)
                L = len(path)
                key = (L, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (L, _comp_size, _col, _sr, _sc), seg_col, seg_start, seg_comp = best_seg

        obstacles = _select_obstacles_for_bar(g, colors, bar_col, seg_col, bar_comp, terminal, seg_start, L)
        new_pts = _route_from_obstacles(g, seg_col, terminal, seg_start, bar_comp, obstacles, L)
        assert len(new_pts) == L, "route must preserve segment pixel count"
        obstacle_union = {p for comp in obstacles for p in comp}
        assert not any(p in obstacle_union for p in new_pts), "route overlaps obstacle"
        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _seg_col, _new_pts in moves:
        for r, c in seg_comp:
            out[r, c] = bg
    for _seg_comp, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 15, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 6
SEG_LEN_MIN, SEG_LEN_MAX = 10, 20
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 6
MAX_RETRIES = 400
PLACEMENT_PAD = 2

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _terminal_for(edge: str, n: int, bar_len: int, pos: int) -> tuple[int, int]:
    if edge == "top":
        return (bar_len - 1, pos)
    if edge == "bottom":
        return (n - bar_len, pos)
    if edge == "left":
        return (pos, bar_len - 1)
    return (pos, n - bar_len)


def _transform(
    locals_: list[tuple[int, int]],
    edge: str,
    n: int,
    bar_len: int,
    pos: int,
) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _build_input_path(seg_len: int, side: int) -> list[tuple[int, int]]:
    coords = [(0, side)]
    remaining = seg_len - 1
    outward_push = min(2, remaining)
    for _ in range(outward_push):
        pr, pc = coords[-1]
        coords.append((pr, pc + side))
    remaining -= outward_push
    turns_budget = random.randint(INPUT_TURN_MIN, min(INPUT_TURN_MAX, max(INPUT_TURN_MIN, seg_len - 2)))
    last_diag: bool | None = None
    while remaining > 0:
        if turns_budget <= 0:
            run = remaining
        else:
            run = random.randint(1, max(1, remaining - turns_budget))
        if last_diag is None:
            use_diag = random.random() < 0.7
        else:
            use_diag = not last_diag if random.random() < 0.6 else last_diag
            turns_budget -= int(use_diag != last_diag)
        step = (1, side) if use_diag else (1, 0)
        last_diag = use_diag
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
    return coords


def _obstacle_templates(side: int) -> list[list[tuple[int, int]]]:
    inward = -side
    return [
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (1, inward)],
        [(0, 0), (0, inward), (1, inward)],
    ]


def _choose_obstacle(anchor: tuple[int, int], side: int, span_cap: int) -> list[tuple[int, int]]:
    choices = []
    for tpl in _obstacle_templates(side):
        span = max(r for r, _ in tpl)
        if span <= span_cap:
            choices.append(tpl)
    tpl = random.choice(choices)
    return [(anchor[0] + r, anchor[1] + c) for r, c in tpl]


def _build_output_and_obstacles(obs_count: int, side: int) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    route = [(0, side)]
    current = route[0]
    cur_side = side
    obstacles: list[list[tuple[int, int]]] = []
    for idx in range(obs_count):
        approach = random.randint(1, 2)
        span_cap = 1 if obs_count == 3 else 2
        for _ in range(approach):
            current = (current[0] + 1, current[1] + cur_side)
            route.append(current)
        apex = current
        anchor = (apex[0], apex[1] - cur_side)
        obstacle = _choose_obstacle(anchor, cur_side, span_cap)
        span = max(r - anchor[0] for r, _ in obstacle)
        obstacles.append(obstacle)
        for _ in range(span):
            current = (current[0] + 1, current[1])
            route.append(current)
        exit_len = approach + 2
        for _ in range(exit_len):
            current = (current[0] + 1, current[1] - cur_side)
            route.append(current)
        cur_side = -cur_side
    tail = random.randint(1, 2 if obs_count == 3 else 3)
    for _ in range(tail):
        current = (current[0] + 1, current[1] + cur_side)
        route.append(current)
    return route, obstacles


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _make_triple(n: int) -> dict | None:
    bar_len = random.randint(BAR_LEN_MIN, BAR_LEN_MAX)
    side = random.choice((-1, 1))
    obs_count = 3 if n >= 20 and random.random() < 0.55 else 2
    output_seg, obstacles = _build_output_and_obstacles(obs_count, side)
    seg_len = len(output_seg)
    if not (SEG_LEN_MIN <= seg_len <= SEG_LEN_MAX):
        return None

    input_seg = _build_input_path(seg_len, side)
    obstacle_union = [p for comp in obstacles for p in comp]
    if _is_4adj(input_seg, obstacle_union):
        return None
    if set(input_seg) == set(output_seg):
        return None

    bar = [(-k, 0) for k in range(bar_len)]
    union = bar + obstacle_union + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    if min(rs) < -(bar_len - 1):
        return None

    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacles": obstacles,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "obs_count": obs_count,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    obstacle_union = [p for comp in triple["obstacles"] for p in comp]
    union = triple["bar"] + obstacle_union + triple["input_seg"] + triple["output_seg"]
    bar_len = triple["bar_len"]
    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        return {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacles": [_transform(comp, edge, n, bar_len, pos) for comp in triple["obstacles"]],
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _ownership_is_unambiguous(placed_triples: list[dict]) -> bool:
    if len(placed_triples) <= 1:
        return True
    for i, placed in enumerate(placed_triples):
        tr, tc = placed["terminal"]
        own_max = max(
            min(abs(r - tr) + abs(c - tc) for r, c in comp)
            for comp in placed["obstacles"]
        )
        other_min = min(
            min(abs(r - tr) + abs(c - tc) for r, c in comp)
            for j, other in enumerate(placed_triples)
            if j != i
            for comp in other["obstacles"]
        )
        if own_max >= other_min:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    triple_count = 1 if n < 21 or random.random() < 0.65 else 2
    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    for edge in random.sample(list(EDGE_DATA), triple_count):
        triple = _make_triple(n)
        if triple is None:
            return None
        placed = _place_triple(triple, edge, n, reserved)
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        for comp in placed["obstacles"]:
            paint_coords(inp, comp, obstacle_color)
        paint_coords(inp, placed["input_seg"], seg_color)

        paint_coords(out, placed["bar"], bar_color)
        for comp in placed["obstacles"]:
            paint_coords(out, comp, obstacle_color)
        paint_coords(out, placed["output_seg"], seg_color)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append(placed)

    if not _ownership_is_unambiguous(placed_triples):
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
