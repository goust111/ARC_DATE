# Variant of 88bcf3b4
# Dimension: D16 — 障碍物与路径的空间关系 (obstacle)
# Change: 障碍物紧贴在输出线段 V 形拐点的内侧（apex_inner），障碍物从拐点内侧向内生长；输入中线段不能 4-邻接障碍物... → 障碍物与 bar 对齐
#
# Design analysis for this variant
# - Concrete behavior:
#   The original output path formed a V-like wrap and the obstacle sat inside that V. In this variant the
#   obstacle is moved to the bar's forward axis: relative to the bar terminal (local origin), the obstacle
#   must contain an axis cell (f>0, s=0), so it is visually "in front of" the bar rather than tucked into a
#   side concavity. The output segment still preserves its pixel count, but now it first goes straight
#   forward on its original side lane, then bends diagonally outward to clear the front obstacle, then keeps
#   going forward on the outer lane.
# - Trace of the changed stage:
#   Original local output was roughly (0,s) -> diagonal -> straight -> opposite diagonal, with obstacle at
#   the inner side of the first bend. Here the local output is (0,s) -> F^pre -> outward-diagonal^diag ->
#   F^post, and the obstacle is generated from a front axis anchor at (pre+1,0) plus a 4-connected staircase
#   that intrudes into the segment's side lane. The changed logic therefore lives in both solver route
#   reconstruction and output-first obstacle generation.
# - Complexity calibration:
#   The task still uses 1-3 triples, bars on borders, irregular input strokes, 4-connected irregular
#   obstacles, and preserved segment length. The new output path has only one bend family (forward, then
#   outward, then forward), which is simpler than the original V-wrap, but the obstacle remains irregular and
#   the side choice is still driven by the segment attachment side, so the rule is still visually supported
#   without becoming ambiguous.
# - Implicit assumptions audited from the original main():
#   The original code assumed the nearest obstacle pixel could be used as a "skin entry" and that the output
#   route would have monotone classes FL*F*FR* or FR*F*FL*. It also asserted that the obstacle occupies the
#   old V-apex inner cell. None of those remain valid. The new solver instead assumes:
#   1. the obstacle has at least one forward axis cell,
#   2. it also reaches the segment's side lane at least once, and
#   3. its maximum reach on that side determines the diagonal detour depth.
#   The generator explicitly constructs obstacles to satisfy these assumptions.
# - Placement legality:
#   Generation is still output-first. The output path is built first, then a front-aligned obstacle is grown
#   inside a bounded box that excludes the bar and output path. The input stroke is built backward away from
#   the border and rejected if it is 4-adjacent to the obstacle. Triples reserve padded bounding boxes so
#   multi-object scenes remain separable.
# - Visual observability:
#   Across examples, the distinctive cue is that the obstacle sits directly ahead of the bar on its axis,
#   while the output segment peels outward on the same side as the input attachment. This is visibly
#   different from the original "obstacle in the V interior" layout.
# - Cleanliness and colors:
#   No color-dependent sorting rule is introduced; colors keep the original role separation only. The obstacle
#   and rerouted segment stay spatially separated enough to remain legible.

import random
from collections import deque

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


EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _edge_for_bar(comp: list[tuple[int, int]], h: int, w: int) -> tuple[str, tuple[int, int]]:
    e0, e1 = _bar_endpoints(comp)
    touch0 = _touches_border(e0, h, w)
    touch1 = _touches_border(e1, h, w)
    assert touch0 != touch1, "bar must touch exactly one border endpoint"
    border_ep = e0 if touch0 else e1
    terminal = e1 if touch0 else e0
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        edge = "top" if border_ep[0] == 0 else "bottom"
    else:
        edge = "left" if border_ep[1] == 0 else "right"
    return edge, terminal


def _to_local(
    p: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side_vec: tuple[int, int]
) -> tuple[int, int]:
    dr = p[0] - terminal[0]
    dc = p[1] - terminal[1]
    return dr * fwd[0] + dc * fwd[1], dr * side_vec[0] + dc * side_vec[1]


def _from_local(
    local: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side_vec: tuple[int, int]
) -> tuple[int, int]:
    lf, ls = local
    return (
        terminal[0] + lf * fwd[0] + ls * side_vec[0],
        terminal[1] + lf * fwd[1] + ls * side_vec[1],
    )


def _build_output_path(seg_len: int, side: int, pre: int, diag: int, post: int) -> list[tuple[int, int]]:
    assert pre >= 1 and diag >= 1 and post >= 1 and pre + diag + post == seg_len - 1
    coords = [(0, side)]
    for _ in range(pre):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc))
    for _ in range(diag):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc + side))
    for _ in range(post):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant D16:
    - obstacle is aligned with the bar's forward axis and sits in front of the terminal,
    - output segment goes forward on its original side lane, bends outward to clear that frontal obstacle,
      then keeps going forward, preserving pixel count.
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
        (seg_len, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        obstacle_candidates = [c for c in colors if c not in (bar_col, seg_col)]
        assert obstacle_candidates, "must have an obstacle color distinct from bar and segment"

        obstacle_set = None
        best_obs = None
        for col in obstacle_candidates:
            for comp in _components(g, col, conn8=True):
                d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                mr = min(r for r, _ in comp)
                mc = min(c for _, c in comp)
                key = (d, mr, mc)
                if best_obs is None or key < best_obs:
                    best_obs = key
                    obstacle_set = set(comp)
        assert obstacle_set is not None, "must find nearest obstacle component"

        edge, terminal = _edge_for_bar(bar_comp, h, w)
        fwd = EDGE_DATA[edge]["fwd"]
        side_vec = EDGE_DATA[edge]["side"]
        start_local = _to_local(seg_start, terminal, fwd, side_vec)
        assert start_local[0] == 0 and abs(start_local[1]) == 1, f"segment must attach laterally: {start_local}"
        side = 1 if start_local[1] > 0 else -1

        obs_local = [_to_local(p, terminal, fwd, side_vec) for p in obstacle_set]
        assert any(lf >= 1 and ls == 0 for lf, ls in obs_local), "front obstacle must contain an axis cell"

        lane_hits = [lf for lf, ls in obs_local if ls == side]
        assert lane_hits, "front obstacle must intrude into the segment side lane"
        pre = min(lane_hits) - 1
        diag = max(side * ls for lf, ls in obs_local)
        post = seg_len - 1 - pre - diag
        assert pre >= 1 and diag >= 1 and post >= 1, f"invalid reconstructed phases: {(pre, diag, post)}"

        local_route = _build_output_path(seg_len, side, pre, diag, post)
        new_pts = [_from_local(p, terminal, fwd, side_vec) for p in local_route]

        assert len(new_pts) == seg_len and len(set(new_pts)) == seg_len, "output route must be a simple path"
        assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), "output route must stay inside grid"
        assert not any(p in obstacle_set for p in new_pts), "output route overlaps obstacle"
        assert not any(p in set(bar_comp) for p in new_pts), "output route overlaps bar"

        # New implicit solver assumptions, guaranteed by generate_input():
        # - nearest lane-1 obstacle hit sets the end of the initial forward run,
        # - max obstacle reach on the segment side sets the outward diagonal length.
        moves.append((seg_comp, seg_col, new_pts))

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
PRE_MAX = 4
DIAG_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 400
PLACEMENT_PAD = 1


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


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _grow_front_obstacle(
    core: set[tuple[int, int]],
    obs_size: int,
    forbidden: set[tuple[int, int]],
    side: int,
    pre: int,
    diag: int,
    post: int,
) -> list[tuple[int, int]]:
    obstacle = set(core)
    max_f = pre + diag + max(2, post)

    def allowed(p: tuple[int, int]) -> bool:
        lf, ls = p
        if p in forbidden:
            return False
        if lf < pre + 1 or lf > max_f:
            return False
        reach = side * ls
        if reach < -2 or reach > diag:
            return False
        if reach == 1 and lf < pre + 1:
            return False
        return True

    for _ in range(max(0, obs_size - len(obstacle))):
        cands = []
        for r, c in list(obstacle):
            for nr, nc in neighbors4(r, c):
                p = (nr, nc)
                if p in obstacle or not allowed(p):
                    continue
                cands.append(p)
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


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

    max_pre = min(PRE_MAX, seg_len - 3, OBS_MD_MAX - 1)
    if max_pre < 1:
        return None
    pre = random.randint(1, max_pre)

    max_diag = min(DIAG_MAX, seg_len - 2 - pre)
    if max_diag < 1:
        return None
    diag = random.randint(1, max_diag)
    post = seg_len - 1 - pre - diag
    if post < 1:
        return None

    output_seg = _build_output_path(seg_len, side, pre, diag, post)
    seg_set = set(output_seg)

    core = {(pre + 1, 0)}
    for k in range(1, diag + 1):
        core.add((pre + k, k * side))
        if k < diag:
            core.add((pre + k + 1, k * side))

    if core & (seg_set | bar_set):
        return None

    obs_size = random.randint(max(OBS_SIZE_MIN, len(core)), OBS_SIZE_MAX)
    obstacle = _grow_front_obstacle(core, obs_size, seg_set | bar_set, side, pre, diag, post)
    if not obstacle:
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    if not any(c == 0 and r >= 1 for r, c in obstacle):
        return None
    if min(r for r, c in obstacle if c == side) != pre + 1:
        return None
    if max(side * c for r, c in obstacle) != diag:
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
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
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
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
