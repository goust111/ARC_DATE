# Combo variant of 88bcf3b4 (index 11)
# combo_spec: D6/0+D7/3
# Name: 矩形围合填充

# Design analysis for the fused variant
# a) Concreteization + behavior trace:
#    The base task reroutes a thin irregular segment around a nearby obstacle and redraws only a 1-cell path.
#    Here the obstacle is always a filled axis-aligned rectangle, and the rerouted segment becomes the outer
#    boundary of a filled corridor. In terminal-local coordinates with side=+1, one representative output is:
#      route:      (0,1),(1,2),(2,3),(3,3),(4,2),(5,1)
#      rectangle: rows 1..4, cols -1..0   [near side at col 0]
#      filled gap rows: r=1 -> c=1 ; r=2 -> c=1,2 ; r=3 -> c=1,2 ; r=4 -> c=1
#    So the old output looked like a bent line, while the new output looks like a triangular/trapezoidal solid
#    region whose outer rim is the rerouted path and whose inner hole is the rectangle.
# b) Complexity calibration:
#    The rectangle is not arbitrary: its near side sits at a fixed offset from the path, and its top/bottom rows
#    align with the last approach row and first exit row. This keeps the geometry inferable from 2-4 examples.
#    We keep only a few controlled degrees of freedom: `p1` (approach length), `p2` (straight-alongside length),
#    rectangle width, and mirroring by side/edge.
# c) Implicit assumption audit:
#    The original solver assumes obstacle skins can be traversed generically with BFS. That no longer matches the
#    fused rule, because the new filled area depends on rectangle geometry, not arbitrary skin walkability. The
#    solver therefore replaces generic obstacle wrapping with rectangle-specific reconstruction in bar-local space.
#    The old "one output pixel per segment pixel" assumption also breaks; repaint now uses a computed filled region.
# d) Placement legality:
#    The generator is output-first. It first builds a legal local output route plus rectangle, derives the filled
#    region, then places the whole triple with padding. Input segments are built separately and rejected if they
#    touch the rectangle. Reserved bboxes use the filled output region, not only the thin route.
# e) Visual observability:
#    A human sees two coupled changes at once: every obstacle is a clean rectangle, and every output turns the
#    segment into a solid wedge/band around that rectangle. This is visually distinct from both the base task and
#    the single-variant 1-cell surround fill.
# f) Visual cleanliness:
#    Rectangles stay compact, bars remain separate, and triples are padded apart. The output has exactly three
#    object types per triple: bar, rectangle hole, and segment-colored filled region.
# g) Anti-degeneracy / fusion metric:
#    The generator enforces a minimum extra filled area beyond the route itself. Because the rectangle near side is
#    offset by a fixed gap, samples cannot collapse back to the thin-line original or to the 1-cell band surround.
# h) New solver assumptions checked against generator:
#    - obstacle components considered by the solver must be filled rectangles;
#    - for the chosen side, the rectangle near side determines `p1`;
#    - rectangle height determines `p2`;
#    - the route is the unique FL*F*FR* or FR*F*FL* monotone path implied by that rectangle.
#    The generator constructs exactly those relations.

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, paint_coords


ROUTE_RECT_GAP = 3


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


def _segment_main_path(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)

    def neighbors(p: tuple[int, int]):
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
        for nxt in neighbors(cur):
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


def _bar_basis(
    bar_comp: list[tuple[int, int]], h: int, w: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    e0, e1 = _bar_endpoints(bar_comp)
    touch0 = _touches_border(e0, h, w)
    touch1 = _touches_border(e1, h, w)
    assert touch0 != touch1, "bar must touch exactly one border endpoint"
    border_ep = e0 if touch0 else e1
    terminal = e1 if touch0 else e0

    rs = [r for r, _ in bar_comp]
    cs = [c for _, c in bar_comp]
    if len(set(cs)) == 1:
        fwd = (1, 0) if border_ep[0] == 0 else (-1, 0)
    else:
        fwd = (0, 1) if border_ep[1] == 0 else (0, -1)
    side = (-fwd[1], fwd[0])
    return terminal, fwd, side


def _to_local(
    p: tuple[int, int], origin: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]
) -> tuple[int, int]:
    dr = p[0] - origin[0]
    dc = p[1] - origin[1]
    return dr * fwd[0] + dc * fwd[1], dr * side[0] + dc * side[1]


def _from_local(
    origin: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int], p: tuple[int, int]
) -> tuple[int, int]:
    lf, ls = p
    return origin[0] + lf * fwd[0] + ls * side[0], origin[1] + lf * fwd[1] + ls * side[1]


def _is_filled_rectangle(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return len(comp) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _build_output_path(seg_len: int, side_sign: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    assert p1 + p2 + p3 == seg_len - 1
    coords = [(0, side_sign)]
    for phase_len, step in zip([p1, p2, p3], [(1, side_sign), (1, 0), (1, -side_sign)]):
        dr, dc = step
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
    return coords


def _rectangle_near_side_local(side_sign: int, p1: int) -> int:
    return side_sign * (p1 + 1 - ROUTE_RECT_GAP)


def _build_rectangle_local(side_sign: int, p1: int, p2: int, rect_w: int) -> list[tuple[int, int]]:
    top = p1 - 1
    height = p2 + 3
    near_side = _rectangle_near_side_local(side_sign, p1)
    if side_sign > 0:
        cols = range(near_side - rect_w + 1, near_side + 1)
    else:
        cols = range(near_side, near_side + rect_w)
    return [(r, c) for r in range(top, top + height) for c in cols]


def _build_output_region_local(
    route_local: list[tuple[int, int]], obstacle_local: list[tuple[int, int]], side_sign: int
) -> list[tuple[int, int]]:
    region = set(route_local)
    near_side = max(c for _, c in obstacle_local) if side_sign > 0 else min(c for _, c in obstacle_local)
    route_by_row = {r: c for r, c in route_local}
    top = min(r for r, _ in obstacle_local)
    bottom = max(r for r, _ in obstacle_local)
    near_signed = side_sign * near_side
    for r in range(top, bottom + 1):
        route_signed = side_sign * route_by_row[r]
        for signed_c in range(near_signed + 1, route_signed):
            region.add((r, side_sign * signed_c))
    obstacle_set = set(obstacle_local)
    return sorted(p for p in region if p not in obstacle_set)


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-anchor each irregular segment around a nearby rectangle, then fill the corridor between them.

    Input structure:
    - Background cells.
    - A bar color forming a straight border-anchored line perpendicular to its touched edge.
    - A segment color forming an irregular 1-cell-thick stroke attached to the bar terminal.
    - An obstacle color whose relevant components are filled rectangles near that terminal.

    Rule:
    - Remove the input segment from its original location, but keep the bar and rectangle unchanged.
    - For each valid bar, find the attached segment length and the nearest compatible rectangle.
    - In the bar-local frame, reconstruct the unique monotone reroute path whose approach length comes from the
      rectangle near side, whose straight-alongside length comes from rectangle height, and whose exit length
      preserves total segment length.
    - Paint both that outer reroute path and the entire background corridor between the path and the rectangle.

    Key insight:
    - The rectangle turns the old free-form wrap into a rigid geometric template: its near side fixes how far the
      path swings outward, its height fixes how long the path runs alongside, and the gap between rectangle and path
      becomes a filled triangle/trapezoid rather than a bare line.
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
                border_ep = e0 if touch0 else e1
                vertical = len({c for _, c in comp}) == 1
                if vertical and border_ep[0] not in (0, h - 1):
                    continue
                if (not vertical) and border_ep[1] not in (0, w - 1):
                    continue
                terminal = e1 if touch0 else e0
                tr, tc = terminal
                if not any(
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal

    moves = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        terminal, fwd, side_basis = _bar_basis(bar_comp, h, w)

        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

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
                key = (len(path), len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (seg_len, _, _, _, _), seg_col, seg_start, seg_comp = best_seg
        seg_set = set(seg_comp)

        side_local = _to_local(seg_start, terminal, fwd, side_basis)
        assert side_local[0] == 0 and abs(side_local[1]) == 1, "segment must attach sideways to the terminal"
        side_sign = side_local[1]

        obstacle_candidates = []
        for col in colors:
            if col in (bar_col, seg_col):
                continue
            for comp in _components(g, col, conn8=False):
                if not _is_filled_rectangle(comp):
                    continue
                local_cells = [_to_local(p, terminal, fwd, side_basis) for p in comp]
                min_f = min(r for r, _ in local_cells)
                max_f = max(r for r, _ in local_cells)
                min_s = min(c for _, c in local_cells)
                max_s = max(c for _, c in local_cells)
                near_side = max_s if side_sign > 0 else min_s
                near_signed = side_sign * near_side
                p1 = near_signed + ROUTE_RECT_GAP - 1
                if p1 < 2:
                    continue
                if min_f != p1 - 1:
                    continue
                rect_h = max_f - min_f + 1
                p2 = rect_h - 3
                if p2 < 0:
                    continue
                p3 = seg_len - 1 - p1 - p2
                if p3 < 1:
                    continue
                route_local = _build_output_path(seg_len, side_sign, p1, p2, p3)
                route = [_from_local(terminal, fwd, side_basis, p) for p in route_local]
                if any(not (0 <= r < h and 0 <= c < w) for r, c in route):
                    continue
                comp_set = set(comp)
                if any(p in comp_set for p in route):
                    continue
                if any(int(g[r, c]) != bg and (r, c) not in seg_set for r, c in route):
                    continue
                rect_w = max_s - min_s + 1
                rebuilt_rect_local = set(_build_rectangle_local(side_sign, p1, p2, rect_w))
                if rebuilt_rect_local != set(local_cells):
                    continue
                fill_local = _build_output_region_local(route_local, local_cells, side_sign)
                fill_pts = [_from_local(terminal, fwd, side_basis, p) for p in fill_local]
                if any(not (0 <= r < h and 0 <= c < w) for r, c in fill_pts):
                    continue
                if any(int(g[r, c]) not in (bg, seg_col) and (r, c) not in comp_set for r, c in fill_pts):
                    continue
                md = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                key = (md, min(r for r, _ in comp), min(c for _, c in comp))
                obstacle_candidates.append((key, fill_pts))

        assert obstacle_candidates, f"no rectangular obstacle compatible with bar at {terminal}"
        _, fill_pts = min(obstacle_candidates, key=lambda x: x[0])
        moves.append((seg_comp, seg_col, fill_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, fill_pts in moves:
        for r, c in fill_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 9, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 6
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE1_MIN = 2
PHASE_MAX = 3
RECT_W_MIN, RECT_W_MAX = 2, 4
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 500
PLACEMENT_PAD = 2
MIN_EXTRA_FILL = 4

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
    locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int
) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _build_input_path(seg_len: int, turns: int, side: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, side)]
    diag_step = (-1, side)
    orth_step = (-1, 0)
    last_was_diag = None
    turns_left = turns
    while remaining > 0:
        if turns_left <= 0:
            step = diag_step if last_was_diag else orth_step
            if last_was_diag is None:
                step = random.choice([diag_step, orth_step])
            run = remaining
        else:
            if last_was_diag is None:
                is_diag = random.random() < DIAG_START_PROB
            elif last_was_diag:
                is_diag = random.random() < DIAG_CONTINUE_PROB
            else:
                is_diag = random.random() < DIAG_START_PROB
            step = diag_step if is_diag else orth_step
            if last_was_diag is not None and is_diag != last_was_diag:
                turns_left -= 1
            max_run = remaining - turns_left
            run = random.randint(1, max(1, min(max_run, remaining)))
        last_was_diag = step == diag_step
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
    return coords


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None
    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None

    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    p1_hi = min(PHASE_MAX, seg_len - 2)
    if p1_hi < PHASE1_MIN:
        return None
    p1 = random.randint(PHASE1_MIN, p1_hi)
    rem = seg_len - 1 - p1
    p2 = random.randint(0, min(PHASE_MAX, rem - 1))
    p3 = rem - p2
    if p3 < 1:
        return None

    output_seg = _build_output_path(seg_len, side, p1, p2, p3)
    seg_set = set(output_seg)

    rect_w = random.randint(RECT_W_MIN, RECT_W_MAX)
    obstacle = _build_rectangle_local(side, p1, p2, rect_w)
    obstacle_set = set(obstacle)
    if obstacle_set & seg_set:
        return None
    if obstacle_set & bar_set:
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (2 <= nearest_md <= 6):
        return None

    output_region = _build_output_region_local(output_seg, obstacle, side)
    if len(output_region) - len(output_seg) < MIN_EXTRA_FILL:
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + obstacle + input_seg + output_region
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_region": output_region,
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
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_region"]
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
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_region": _transform(triple["output_region"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 12:
        max_triples = 1
    elif n < 16:
        max_triples = 2
    else:
        max_triples = 3
    triple_count = random.randint(1, max_triples)

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
        paint_coords(inp, placed["obstacle"], obstacle_color)
        paint_coords(inp, placed["input_seg"], seg_color)

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["obstacle"], obstacle_color)
        paint_coords(out, placed["output_region"], seg_color)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append(placed)

    if len(placed_triples) > 1:
        for i, placed in enumerate(placed_triples):
            tr, tc = placed["terminal"]
            own = min(abs(r - tr) + abs(c - tc) for r, c in placed["obstacle"])
            other = min(
                abs(r - tr) + abs(c - tc)
                for j, other_placed in enumerate(placed_triples)
                if j != i
                for r, c in other_placed["obstacle"]
            )
            if own >= other:
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
