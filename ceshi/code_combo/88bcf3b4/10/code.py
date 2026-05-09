# Combo variant of 88bcf3b4 (index 10)
# combo_spec: D4/0+D16/0
# Name: 直线源正面障碍
#
# Design analysis for this combo variant
# - Concrete behavior:
#   Work in bar-local coordinates with the bar terminal at (0,0), negative forward values on the bar itself,
#   positive forward values going into the grid, and side = +/-1 selecting the attachment side. A small
#   example is:
#     input segment  = [(0,1), (1,1), (2,1), (3,1), (4,1), (5,1)]
#     obstacle core  = [(3,0), (4,0)]
#     output segment = [(0,1), (1,1), (2,1), (3,2), (4,3), (5,3)]
#   Relative to the original task, two things change together:
#   1. the source segment is no longer an irregular rope; it is a single straight line,
#   2. the obstacle is no longer tucked into a V-apex; it sits on the bar's forward axis.
#   The result is that the source visibly heads toward a frontal blocker, and the output peels outward near
#   the front face instead of wrapping around an obstacle hidden inside a concavity.
# - Complexity calibration:
#   The task still keeps the original object vocabulary and scene density: 1-3 triples, border-anchored
#   bars, a distinct obstacle sprite, preserved segment length, and a nontrivial reroute. The transformation
#   is simpler than the original only in one dimension: the input source is now a straight stroke. The
#   obstacle remains irregular and the output still has multiple controlled phases (`forward`, `diagonal`,
#   `forward`) so the task remains visually rich and does not collapse into a trivial translation.
# - Implicit assumptions audited from the original solver:
#   The original code relied on an irregular 8-connected segment, obstacle-contact on a V inner cell, and a
#   skin-following reconstruction around obstacle boundary pixels. All three assumptions break here.
#   A literal "source on the exact bar axis" interpretation would also collide with the original distance
#   budget, because a 5-9 px straight segment would run into any obstacle only 2-6 cells ahead. This combo
#   therefore realizes D4/0 as a straight source lane attached to the terminal, while D16/0 supplies the
#   frontal obstacle on the axis. The new solver instead assumes:
#   1. the input segment is a straight 4-connected line whose endpoint touches the terminal laterally,
#   2. the nearest obstacle has a contiguous run of axis cells directly in front of the bar,
#   3. that frontal axis run determines where the reroute starts and how many outward diagonal steps it uses.
#   The generator constructs every sample to satisfy those assumptions.
# - Placement legality:
#   Generation is still output-first. It first chooses the output's three phase lengths, then plants an
#   obstacle sprite whose core is the axis run that the output skirts around. Extra obstacle pixels are grown
#   only on the axis or the side opposite the source lane, so the obstacle never overlaps the straight input
#   or the rerouted output. Bounding boxes are padded before multi-triple placement so separate triples stay
#   disentangled.
# - Visual observability:
#   A human should be able to see the rule from a few examples because two cues repeat together:
#   1. the input source is always a clean straight stroke attached to a bar endpoint,
#   2. the obstacle is always visibly in front of that bar, on the bar's axis rather than inside a V.
#   The output then consistently bends outward near the obstacle's front face. No color ordering or hidden
#   numeric convention is used.
# - Visual cleanliness and non-degeneracy:
#   The generator forbids obstacle growth into the source side, so bars, source lines, obstacles, and outputs
#   remain separable. This also prevents the combo from collapsing back to the original V-apex layout: the
#   obstacle's defining support is a frontal axis strip, not a side-adjacent apex cell.

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, paint_coords


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _is_straight_line(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        r = rs[0]
        cols = sorted(set(cs))
        return cols == list(range(min(cols), max(cols) + 1))
    if len(set(cs)) == 1:
        c = cs[0]
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


def _line_path_from_start(comp: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]] | None:
    if not _is_straight_line(comp):
        return None
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        r = rs[0]
        path = [(r, c) for c in sorted(cs)]
    else:
        c = cs[0]
        path = [(r, c) for r in sorted(rs)]
    if path[0] == start:
        return path
    if path[-1] == start:
        return path[::-1]
    return None


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


def _build_input_path(seg_len: int, side: int) -> list[tuple[int, int]]:
    return [(k, side) for k in range(seg_len)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reroute a straight source line when a frontal obstacle blocks the bar's forward axis.

    Input structure:
    - Each scene contains 1-3 triples made of a border-perpendicular bar, a straight source segment attached
      laterally to the bar terminal, and an irregular obstacle sprite placed in front of that bar on its
      forward axis.

    Rule:
    - Keep every bar and obstacle fixed.
    - Replace each straight source line by a same-length path that runs forward on its original lane, then
      bends outward diagonally to get around the obstacle's frontal axis strip, then continues forward.
    - The obstacle's nearest axis cell tells where the bend starts, and the contiguous thickness of axis cells
      tells how many outward diagonal steps are needed.

    Key insight:
    - The source is no longer an irregular rope to trace backward from shape complexity; it is a clean lane
      aimed toward a blocker directly ahead. The output is recovered by reading the obstacle's front-face axis
      span and turning that span into a single outward detour, with total segment length preserved.
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
        edge, terminal = _edge_for_bar(bar_comp, h, w)
        fwd = EDGE_DATA[edge]["fwd"]
        side_vec = EDGE_DATA[edge]["side"]

        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

        best_seg = None
        for cand_col, cand_start in neigh:
            for comp in _components(g, cand_col, conn8=False):
                if cand_start not in comp:
                    continue
                path = _line_path_from_start(comp, cand_start)
                if path is None or len(path) < 2:
                    break
                start_local = _to_local(cand_start, terminal, fwd, side_vec)
                if start_local[0] != 0 or abs(start_local[1]) != 1:
                    break
                key = (len(path), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp, path)
                break
        assert best_seg is not None, f"no valid straight segment for bar at {terminal}"
        (seg_len, _, _, _), seg_col, seg_start, seg_comp, seg_path = best_seg

        start_local = _to_local(seg_start, terminal, fwd, side_vec)
        side = 1 if start_local[1] > 0 else -1
        expected_input = [_from_local(p, terminal, fwd, side_vec) for p in _build_input_path(seg_len, side)]
        assert seg_path == expected_input, "input segment must be a forward straight lane"

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

        obs_local = [_to_local(p, terminal, fwd, side_vec) for p in obstacle_set]
        axis_hits = sorted(lf for lf, ls in obs_local if ls == 0 and lf >= 1)
        assert axis_hits, "front obstacle must contain forward axis cells"
        assert axis_hits == list(range(axis_hits[0], axis_hits[-1] + 1)), "axis strip must be contiguous"
        assert all(side * ls <= 0 for lf, ls in obs_local), "obstacle must stay off the source side"

        pre = axis_hits[0] - 1
        diag = len(axis_hits)
        post = seg_len - 1 - pre - diag
        assert pre >= 1 and diag >= 1 and post >= 1, f"invalid reconstructed phases: {(pre, diag, post)}"

        local_route = _build_output_path(seg_len, side, pre, diag, post)
        new_pts = [_from_local(p, terminal, fwd, side_vec) for p in local_route]

        assert len(new_pts) == seg_len and len(set(new_pts)) == seg_len, "output route must be a simple path"
        assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), "output route must stay inside grid"
        assert not any(p in obstacle_set for p in new_pts), "output route overlaps obstacle"
        assert not any(p in set(bar_comp) for p in new_pts), "output route overlaps bar"

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
    max_f = pre + diag + max(1, post)
    core_axis = set(core)

    def allowed(p: tuple[int, int]) -> bool:
        lf, ls = p
        if p in forbidden:
            return False
        if lf < pre + 1 or lf > max_f:
            return False
        signed_side = side * ls
        if signed_side > 0 or signed_side < -2:
            return False
        if ls == 0 and p not in core_axis:
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

    max_pre = min(PRE_MAX, seg_len - 3)
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

    input_seg = _build_input_path(seg_len, side)
    output_seg = _build_output_path(seg_len, side, pre, diag, post)
    input_set = set(input_seg)
    output_set = set(output_seg)

    core = {(pre + k, 0) for k in range(1, diag + 1)}
    if core & (bar_set | output_set | input_set):
        return None

    obs_size = random.randint(max(OBS_SIZE_MIN, len(core)), OBS_SIZE_MAX)
    obstacle = _grow_front_obstacle(core, obs_size, bar_set | input_set | output_set, side, pre, diag, post)
    if not obstacle:
        return None

    axis_hits = sorted(r for r, c in obstacle if c == 0)
    if axis_hits != list(range(pre + 1, pre + diag + 1)):
        return None
    if any(side * c > 0 for r, c in obstacle):
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
