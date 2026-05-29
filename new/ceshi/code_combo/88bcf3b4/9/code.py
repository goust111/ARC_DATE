# Combo variant of 88bcf3b4 (index 9)
# combo_spec: D3/2+D17/0
# Name: 等分相位周期涂色

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    Example with segment length L=8. Then seg_len-1=7, so the three phase step
#    lengths are (2, 2, 3). Relative to the bar terminal, the output route starts
#    at the same 4-neighbor attachment cell as the input segment, walks 2 diagonal
#    forward steps toward the obstacle, then 2 straight forward steps, then 3
#    diagonal forward steps on the opposite side. The obstacle contains the cell
#    just inside the V apex. Compared with the original solver, the route is no
#    longer "whatever the skin search yields"; it is an explicit equalized
#    three-phase template derived from L.
# b) Complexity calibration:
#    The original task allows short segments, but 1-step phases are visually weak
#    and make the equal-split rule hard to infer. The generator therefore keeps
#    segment lengths at least 7 while still allowing longer strokes, so examples
#    regularly show different equal-phase bases such as (2,2,*), (3,3,*), and
#    occasionally (4,4,*).
# c) Assumption audit:
#    The original main() depended on BFS path length and obstacle skin traversal.
#    Those assumptions break here because the phase lengths must be fixed by L and
#    not by free-form routing. The new main() instead assumes:
#    - the attached segment endpoint is still unique and 4-adjacent to the bar terminal;
#    - the nearest non-bar/non-segment component is the intended obstacle;
#    - the obstacle contains apex_inner = terminal + p1*(fwd + side_vec);
#    - the equal-phase route stays in bounds and does not hit other objects.
#    The generator constructs inputs that guarantee those properties.
# d) Placement legality:
#    The generator is still output-first. It first builds the equal-phase output
#    route, grows an obstacle from the required apex-inner cell without overlapping
#    the route, then derives a backward irregular input stroke of the same length.
#    A padded reservation mask keeps different triples separated.
# e) Visual observability:
#    The output route is recolored with a global repeating 2-pixel block rhythm
#    using the bar color and the segment color. Because the first two phases are
#    equal in step count, the bend locations line up with predictable color-block
#    boundaries instead of arbitrary offsets.
# f) Visual cleanliness:
#    Only four colors are used globally (bg, bar, segment, obstacle). The obstacle
#    color stays unchanged; the bar and route share a color by design; wide spacing
#    prevents merged components across different triples.
# g) Inferability:
#    The human-visible clues are structural: the route keeps the same length as the
#    detached stroke, bends in the same three-phase monotone pattern every time, and
#    alternates between the two already-present colors instead of introducing a new
#    color mapping that would require hidden numeric conventions.


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _is_straight_line(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        row = rs[0]
        cols = sorted(set(cs))
        return cols == list(range(min(cols), max(cols) + 1))
    if len(set(cs)) == 1:
        col = cs[0]
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


def _phase_lengths(seg_len: int) -> tuple[int, int, int]:
    steps = seg_len - 1
    base = steps // 3
    rem = steps % 3
    return base, base, base + rem


def _build_equal_phase_route(
    terminal: tuple[int, int], side_vec: tuple[int, int], fwd: tuple[int, int], phase_lengths: tuple[int, int, int]
) -> list[tuple[int, int]]:
    p1, p2, p3 = phase_lengths
    route = [(terminal[0] + side_vec[0], terminal[1] + side_vec[1])]
    cur = route[0]
    for _ in range(p1):
        cur = (cur[0] + fwd[0] + side_vec[0], cur[1] + fwd[1] + side_vec[1])
        route.append(cur)
    for _ in range(p2):
        cur = (cur[0] + fwd[0], cur[1] + fwd[1])
        route.append(cur)
    for _ in range(p3):
        cur = (cur[0] + fwd[0] - side_vec[0], cur[1] + fwd[1] - side_vec[1])
        route.append(cur)
    return route


def _paint_periodic_route(
    out: np.ndarray,
    route: list[tuple[int, int]],
    phase_lengths: tuple[int, int, int],
    bar_col: int,
    seg_col: int,
) -> None:
    for idx, cell in enumerate(route):
        out[cell] = bar_col if (idx // 2) % 2 == 0 else seg_col


def _nearest_obstacle_component(
    g: np.ndarray, colors: list[int], bar_col: int, seg_col: int, terminal: tuple[int, int]
) -> set[tuple[int, int]]:
    best = None
    obstacle_set = None
    for col in colors:
        if col in (bar_col, seg_col):
            continue
        for comp in _components(g, col, conn8=True):
            d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
            mr = min(r for r, _ in comp)
            mc = min(c for _, c in comp)
            key = (d, mr, mc)
            if best is None or key < best:
                best = key
                obstacle_set = set(comp)
    assert obstacle_set is not None, "must find an obstacle component"
    return obstacle_set


def _route_matches_template(
    route: list[tuple[int, int]], fwd: tuple[int, int], side_vec: tuple[int, int], phase_lengths: tuple[int, int, int]
) -> bool:
    p1, p2, p3 = phase_lengths
    expected_steps = [fwd[0] + side_vec[0], fwd[1] + side_vec[1]], [fwd[0], fwd[1]], [fwd[0] - side_vec[0], fwd[1] - side_vec[1]]
    cuts = [p1, p1 + p2, p1 + p2 + p3]
    for idx in range(len(route) - 1):
        step = (route[idx + 1][0] - route[idx][0], route[idx + 1][1] - route[idx][1])
        if idx < cuts[0]:
            want = tuple(expected_steps[0])
        elif idx < cuts[1]:
            want = tuple(expected_steps[1])
        else:
            want = tuple(expected_steps[2])
        if step != want:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rehang each endpoint-attached stroke into an equal-phase wrap and repaint it with a bar/segment rhythm.

    Input structure:
    - Background is the most frequent color.
    - Each active group contains a straight bar touching one border, a 1-cell-thick irregular segment attached to the bar's interior endpoint, and a nearby obstacle sprite of a third color.

    Rule:
    - Keep the bar and obstacle fixed.
    - Remove the irregular segment and rebuild it as a monotone 3-phase route around the nearest obstacle: diagonal approach, straight alongside, opposite diagonal exit.
    - If the segment has length L, split L-1 route steps into equal thirds; any remainder 1-2 steps goes only to the exit phase.
    - Repaint the rebuilt route with a repeating 2-pixel block pattern that alternates between the bar color and the original segment color along the whole route.

    Key insight:
    - The obstacle still marks where the route must wrap, but the exact route length allocation is no longer found by free-form skin tracing; it is inferred from the segment length itself, and the phase boundaries are echoed by the repeating 2-pixel color blocks.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
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
                terminal_ep = e1 if touch0 else e0
                if vertical:
                    if border_ep[0] not in (0, h - 1):
                        continue
                    fwd = (1, 0) if border_ep[0] == 0 else (-1, 0)
                else:
                    if border_ep[1] not in (0, w - 1):
                        continue
                    fwd = (0, 1) if border_ep[1] == 0 else (0, -1)
                tr, tc = terminal_ep
                if not any(
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal_ep, fwd

    moves: list[tuple[list[tuple[int, int]], int, int, list[tuple[int, int]], tuple[int, int, int]]] = []

    for bar_col, bar_comp, terminal, fwd in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
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
                L = len(path)
                key = (L, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (seg_len, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        side_vec = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
        assert abs(side_vec[0]) + abs(side_vec[1]) == 1, "segment must attach by a 4-neighbor"
        assert side_vec != fwd and side_vec != (-fwd[0], -fwd[1]), "segment must attach laterally to the bar"

        phase_lengths = _phase_lengths(seg_len)
        route = _build_equal_phase_route(terminal, side_vec, fwd, phase_lengths)
        obstacle_set = _nearest_obstacle_component(g, colors, bar_col, seg_col, terminal)

        p1, _, _ = phase_lengths
        apex = route[p1]
        apex_inner = (apex[0] - side_vec[0], apex[1] - side_vec[1])
        assert apex_inner in obstacle_set, f"obstacle must contain the inner apex cell {apex_inner}"

        assert route[0] == seg_start, "rebuilt route must start at the original attachment pixel"
        assert len(route) == seg_len, "output route must preserve segment length"
        assert _route_matches_template(route, fwd, side_vec, phase_lengths), "route must follow the three-phase template"
        assert all(0 <= r < h and 0 <= c < w for r, c in route), "route must remain inside the grid"
        assert not any((r, c) in obstacle_set for r, c in route), "route must not overlap the obstacle"
        assert not any((r, c) in set(bar_comp) for r, c in route), "route must not overlap the bar"

        moves.append((seg_comp, bar_col, seg_col, route, phase_lengths))

    for seg_comp, _, _, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, bar_col, seg_col, route, phase_lengths in moves:
        _paint_periodic_route(out, route, phase_lengths, bar_col, seg_col)

    return out


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 15, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 6
BAR_CLEARANCE = 8
SEG_LEN_MIN, SEG_LEN_MAX = 7, 13
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 7
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 4
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
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
    locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int
) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _grow_obstacle(
    start: tuple[int, int], max_size: int, forbidden: set[tuple[int, int]], min_md: int
) -> list[tuple[int, int]]:
    obstacle = {start}
    for _ in range(max_size - 1):
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle
            and (nr, nc) not in forbidden
            and abs(nr) + abs(nc) >= min_md
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _build_input_path(seg_len: int, turns: int, side: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, side)]
    # Keep the input stroke inside the grid interior while steering it away from
    # the obstacle/output side of the eventual wrap.
    diag_step = (1, -side)
    orth_step = (1, 0)
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


def _build_output_path(
    seg_len: int, side: int, p1: int, p2: int, p3: int
) -> list[tuple[int, int]]:
    coords = [(0, side)]
    phase_steps = [(1, side), (1, 0), (1, -side)]
    for phase_len, (dr, dc) in zip([p1, p2, p3], phase_steps):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
    assert len(coords) == seg_len
    return coords


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _paint_output_route(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    phase_lengths: tuple[int, int, int],
    bar_color: int,
    seg_color: int,
) -> None:
    _paint_periodic_route(grid, coords, phase_lengths, bar_color, seg_color)


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len - 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    p1, p2, p3 = _phase_lengths(seg_len)
    side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    output_seg = _build_output_path(seg_len, side, p1, p2, p3)
    seg_set = set(output_seg)

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    obs_size = random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX)
    min_md = abs(apex_inner[0]) + abs(apex_inner[1])
    obstacle = _grow_obstacle(apex_inner, obs_size, seg_set | bar_set, min_md=min_md)
    if apex_inner not in obstacle:
        return None
    if min(abs(r) + abs(c) for r, c in obstacle) != min_md:
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
    shared = set(input_seg) & seg_set
    if shared != {(0, side)}:
        return None
    if set(input_seg) & bar_set:
        return None
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "phase_lengths": (p1, p2, p3),
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
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"]
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
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "phase_lengths": triple["phase_lengths"],
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 19:
        max_triples = 1
    elif n < 24:
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
        _paint_output_route(out, placed["output_seg"], placed["phase_lengths"], bar_color, seg_color)

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
