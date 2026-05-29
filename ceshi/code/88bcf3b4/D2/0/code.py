# Variant of 88bcf3b4
# Dimension: D2 — 边界遍历方向 (path)
# Change: 沿障碍物边界的遍历方向由 bar 末端到障碍物最近点的位置关系隐式决定
#         （根据 root side 选择 FL 或 FR 入场，可能顺时针或逆时针）
#         改为始终顺时针遍历。
#
# Design analysis
# a) Concrete behavior:
#    Normalize each triple by the bar direction. In that local frame, "clockwise"
#    is fixed as (fwd+cw)* -> fwd* -> (fwd-cw)* where `cw` is the global
#    clockwise quarter-turn of `fwd`. The input segment may still attach on
#    either side of the terminal.
#    The input segment may still attach on either side of the terminal, so when
#    the root starts on the left, the output visibly differs from the original
#    task: the first wrapped diagonal still uses the same global clockwise side
#    instead of mirroring to the opposite side.
# b) Complexity calibration:
#    The grid still contains 1-3 bar+segment+obstacle triples, the segment keeps
#    its original length, and the obstacle remains a small 4-connected sprite.
#    The only controlled freedom removed is the side-dependent choice between the
#    two mirror wraps; this prevents degeneration into an unrelated simpler task.
# c) Old assumptions that break:
#    The original solver depends on `off_r/off_c` from the input root side,
#    translates the terminal->obstacle path by that offset, and then chooses a
#    shortest skin path to a far exit. Those lines assume the wrap side should
#    inherit the input root side. Under this variant that is false: the output
#    must always use the same clockwise phase order, even when the root starts on
#    the opposite side. Therefore the output route can no longer be inferred by
#    simply translating the input-side offset onto the obstacle skin.
# d) Placement legality:
#    The generator stays output-first. It builds the clockwise output path first,
#    derives an obstacle strip glued to the inner side of the straight-forward
#    phase, then constructs an input stroke of equal length that stays detached
#    from the obstacle. Final placement still uses padded bbox reservation and
#    validates `main(input) == output`.
# e) Visual observability:
#    The distinguishing branch is when the input root starts on the side that
#    would formerly induce the opposite boundary direction. Because root side
#    remains visible in the input, the fixed clockwise wrap is inferable from
#    2-4 examples.
# f) Visual cleanliness:
#    Obstacles are generated as a compact strip plus optional inward bulges on
#    the inner side of the wrapped segment. This keeps the "rope around sprite"
#    picture legible and avoids noisy skin traversals.
# g) Color reasoning:
#    Colors remain unchanged: bg, shared bar color, shared segment color, shared
#    obstacle color. No hidden ordering by color id is introduced.
# h) Variant metric used after implementation:
#    Compare this solver's output with the original solver on generated inputs.
#    Samples whose input root is on the local left side should usually differ.
# i) New solver assumptions checked against the generator:
#    - The clockwise output always has phase-1 length >= 1.
#    - The obstacle contains the cells immediately inside the apex and every
#      straight-forward "alongside" cell, so phase lengths are recoverable.
#    - Extra obstacle growth only expands inward, so it does not create false
#      side contacts that would confuse phase detection.

from collections import deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


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


def _bar_forward(bar_comp: list[tuple[int, int]], h: int, w: int) -> tuple[int, int]:
    rs = [r for r, _ in bar_comp]
    cs = [c for _, c in bar_comp]
    if len(set(cs)) == 1:
        return (1, 0) if min(rs) == 0 else (-1, 0)
    return (0, 1) if min(cs) == 0 else (0, -1)


def _cw_side(fwd: tuple[int, int]) -> tuple[int, int]:
    # Screen coordinates: rows increase downward, so this is a clockwise quarter turn.
    return (-fwd[1], fwd[0])


def _route_step_classes(
    pts: list[tuple[int, int]], fwd: tuple[int, int]
) -> list[str]:
    out = []
    for j in range(len(pts) - 1):
        dr = pts[j + 1][0] - pts[j][0]
        dc = pts[j + 1][1] - pts[j][1]
        fwd_comp = dr * fwd[0] + dc * fwd[1]
        perp_comp = dr * fwd[1] + dc * (-fwd[0])
        if abs(dr) + abs(dc) == 1:
            out.append("F" if fwd_comp > 0 else "X")
        elif fwd_comp > 0 and perp_comp > 0:
            out.append("FR")
        elif fwd_comp > 0 and perp_comp < 0:
            out.append("FL")
        else:
            out.append("X")
    return out


def _build_clockwise_route(
    start: tuple[int, int],
    fwd: tuple[int, int],
    cw: tuple[int, int],
    p1: int,
    p2: int,
    p3: int,
) -> list[tuple[int, int]]:
    steps = [
        (fwd[0] + cw[0], fwd[1] + cw[1]),
        fwd,
        (fwd[0] - cw[0], fwd[1] - cw[1]),
    ]
    route = [start]
    cur = start
    for count, step in zip([p1, p2, p3], steps):
        for _ in range(count):
            cur = (cur[0] + step[0], cur[1] + step[1])
            route.append(cur)
    return route


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant solver:
    - detect each bar and its attached irregular segment
    - find the nearest obstacle component
    - rebuild the output segment with a fixed clockwise local wrap:
      (fwd+cw)* -> fwd* -> (fwd-cw)* regardless of the input root side
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
                terminal_ep = e1 if touch0 else e0
                tr, tc = terminal_ep
                if not any(
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal_ep

    moves = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
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
        (L, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        obstacle_candidates = [c for c in colors if c not in (bar_col, seg_col)]
        assert obstacle_candidates, "must have at least one obstacle color distinct from bar and segment"

        best_obs = None
        obstacle_set = None
        for col in obstacle_candidates:
            for comp in _components(g, col, conn8=True):
                d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                mr = min(r for r, _ in comp)
                mc = min(c for _, c in comp)
                key = (d, mr, mc)
                if best_obs is None or key < best_obs:
                    best_obs = key
                    obstacle_set = set(comp)
        assert obstacle_set is not None, "must find at least one obstacle component"

        fwd = _bar_forward(bar_comp, h, w)
        cw = _cw_side(fwd)

        def walkable_cell(p: tuple[int, int]) -> bool:
            r, c = p
            if not (0 <= r < h and 0 <= c < w):
                return False
            if p in obstacle_set:
                return False
            return int(g[r, c]) in (bg, seg_col)

        best_route = None
        best_key = None
        for p1 in range(1, L):
            route = [seg_start]
            cur = seg_start
            ok = True
            for _ in range(p1):
                cur = (cur[0] + fwd[0] + cw[0], cur[1] + fwd[1] + cw[1])
                route.append(cur)
                if not walkable_cell(cur):
                    ok = False
                    break
            if not ok:
                continue

            apex = route[p1]
            apex_inner = (apex[0] - cw[0], apex[1] - cw[1])
            if apex_inner not in obstacle_set:
                continue
            if any((route[j][0] - cw[0], route[j][1] - cw[1]) in obstacle_set for j in range(p1)):
                continue

            p2 = 0
            straight_route = list(route)
            cur2 = apex
            remaining = L - 1 - p1
            while p2 < remaining:
                nxt = (cur2[0] + fwd[0], cur2[1] + fwd[1])
                if not walkable_cell(nxt):
                    break
                if (nxt[0] - cw[0], nxt[1] - cw[1]) not in obstacle_set:
                    break
                straight_route.append(nxt)
                cur2 = nxt
                p2 += 1

            p3 = L - 1 - p1 - p2
            full_route = list(straight_route)
            cur3 = cur2
            for _ in range(p3):
                cur3 = (cur3[0] + fwd[0] - cw[0], cur3[1] + fwd[1] - cw[1])
                full_route.append(cur3)
                if not walkable_cell(cur3):
                    ok = False
                    break
            if not ok:
                continue
            if len(set(full_route)) != len(full_route):
                continue
            if p3 > 0:
                first_exit = full_route[p1 + p2 + 1]
                if (first_exit[0] - cw[0], first_exit[1] - cw[1]) in obstacle_set:
                    continue

            contact_count = sum(
                1
                for p in full_route
                if any((p[0] + dr, p[1] + dc) in obstacle_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
            )
            forward_end = (full_route[-1][0] - terminal[0]) * fwd[0] + (full_route[-1][1] - terminal[1]) * fwd[1]
            key = (p2, contact_count, forward_end, -p1)
            if best_key is None or key > best_key:
                best_key = key
                best_route = full_route

        assert best_route is not None, f"no clockwise route for bar at {terminal}"
        assert not any(p in obstacle_set for p in best_route), "output segment passes through obstacle"
        moves.append((seg_comp, seg_col, best_route))

    for seg_comp, _seg_col, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 400
PLACEMENT_PAD = 1

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


def _build_output_path(
    seg_len: int, input_side: int, p1: int, p2: int, p3: int
) -> list[tuple[int, int]]:
    coords = [(0, input_side)]
    for count, step in zip([p1, p2, p3], [(1, 1), (1, 0), (1, -1)]):
        for _ in range(count):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
    assert len(coords) == seg_len
    return coords


def _grow_inward_obstacle(
    base: set[tuple[int, int]], max_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    obstacle = set(base)
    while len(obstacle) < max_size:
        cands = []
        for r, c in obstacle:
            for nr, nc in [(r, c - 1), (r - 1, c), (r + 1, c)]:
                if (nr, nc) in obstacle or (nr, nc) in forbidden:
                    continue
                # Grow only inward / inward-connected so the visible right edge
                # still marks the alongside phase unambiguously.
                if nc > c:
                    continue
                cands.append((nr, nc))
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


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
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    # The input stroke always walks one step backward in forward-depth per pixel,
    # so it must fit within the bar's interior clearance from the border.
    max_seg = min(SEG_LEN_MAX, bar_len)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    input_side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    remaining = seg_len - 1
    p1 = random.randint(1, min(PHASE_MAX, remaining))
    remaining -= p1
    p2 = random.randint(0, min(PHASE_MAX, remaining))
    p3 = remaining - p2

    output_seg = _build_output_path(seg_len, input_side, p1, p2, p3)
    seg_set = set(output_seg)

    base_obstacle = {(r, c - 1) for r, c in output_seg[p1 : p1 + p2 + 1]}
    if not base_obstacle or base_obstacle & seg_set or base_obstacle & bar_set:
        return None

    target_obs_size = random.randint(max(OBS_SIZE_MIN, len(base_obstacle)), OBS_SIZE_MAX)
    obstacle = _grow_inward_obstacle(base_obstacle, target_obs_size, seg_set | bar_set)

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, input_side)

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
        "input_side": input_side,
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


def _place_triple(
    triple: dict, edge: str, n: int, reserved: np.ndarray
) -> dict | None:
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
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
            "edge": edge,
            "input_side": triple["input_side"],
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 11:
        max_triples = 1
    elif n < 15:
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
        paint_coords(out, placed["output_seg"], seg_color)

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
