# Combo variant of 88bcf3b4 (index 4)
# combo_spec: D6/0+D17/2
# Name: 矩形尖角绕行

# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original 88bcf3b4 lets the detached segment approach an obstacle, follow the obstacle skin for a straight
#    alongside phase, then leave diagonally. Here that middle phase is removed entirely. In local coordinates
#    from the bar terminal, the output path is forced to be `(0,s) -> (1,2s) -> ... -> (p1,(p1+1)s)` and then
#    immediately switches to `(p1+1,p1*s) -> ...`, where `s in {-1,+1}` is the root side. The obstacle is no
#    longer an irregular sprite: it is a filled axis-aligned rectangle whose near top corner sits exactly one
#    forward step in front of the V apex. This makes the apex derivable from rectangle geometry alone.
# b) Complexity calibration:
#    Keeping rectangles at least 2x2 avoids degenerating into the original “small blob” feel, and forcing both
#    diagonal phases to be non-empty preserves a visibly sharp V. Restricting `p1` to a small set {1,2} keeps
#    the corner close enough to the terminal to read easily in 2-4 examples while still allowing repeated
#    support across different sizes and mirrorings.
# c) Implicit assumptions audited:
#    The original solver assumes there is a walkable obstacle skin and that the best route includes a boundary
#    traversal plus an extension direction taken from the skin endpoint. Those assumptions are invalid here,
#    so the solver now rejects non-rectangular obstacles, infers the relevant rectangle corner in the bar-local
#    frame, reconstructs the unique sharp-V path from that corner, and validates that no forward-straight cells
#    appear in the output phase sequence.
# d) Placement legality:
#    The generator is still output-first. It builds a legal sharp V, derives the rectangle from the apex, then
#    places the whole triple with padded non-overlap checks. The input segment is generated as a separate
#    backward irregular stroke of equal pixel count and is rejected if it touches the rectangle.
# e) Visual observability:
#    A human can read the rule from examples because the only obstacle shape class is “filled rectangle”, and
#    every output segment bends at a point directly above the rectangle's near top corner instead of sliding
#    along an edge. The absence of any straight-forward phase is therefore visible, not latent.
# f) Visual cleanliness:
#    Rectangles are kept compact and buffered by reserved placement padding, so bars, rectangles, input strokes,
#    and output V-shapes remain spatially separated. This avoids ambiguous merges between neighboring triples.
# g) Anti-degeneracy:
#    The generator enforces both “filled rectangle” and “zero straight phase”. On these samples the old oracle
#    would usually trace some rectangle skin, so comparing old-vs-new outputs is a useful sanity check that the
#    fusion is not collapsing back to the original behavior.

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


def _bar_basis(bar_comp: list[tuple[int, int]], h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
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


def _to_local(p: tuple[int, int], origin: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]) -> tuple[int, int]:
    dr = p[0] - origin[0]
    dc = p[1] - origin[1]
    return dr * fwd[0] + dc * fwd[1], dr * side[0] + dc * side[1]


def _from_local(origin: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int], p: tuple[int, int]) -> tuple[int, int]:
    lf, ls = p
    return origin[0] + lf * fwd[0] + ls * side[0], origin[1] + lf * fwd[1] + ls * side[1]


def _is_filled_rectangle(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return len(comp) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _build_sharp_v(side_sign: int, p1: int, p3: int) -> list[tuple[int, int]]:
    coords = [(0, side_sign)]
    for _ in range(p1):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc + side_sign))
    for _ in range(p3):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc - side_sign))
    return coords


def _classify_steps(new_pts: list[tuple[int, int]], fwd: tuple[int, int]) -> list[str]:
    classes = []
    for j in range(len(new_pts) - 1):
        dr = new_pts[j + 1][0] - new_pts[j][0]
        dc = new_pts[j + 1][1] - new_pts[j][1]
        fwd_comp = dr * fwd[0] + dc * fwd[1]
        perp_comp = dr * fwd[1] + dc * (-fwd[0])
        if abs(dr) + abs(dc) == 1:
            classes.append("F" if fwd_comp > 0 else "X")
        elif fwd_comp > 0 and perp_comp > 0:
            classes.append("FR")
        elif fwd_comp > 0 and perp_comp < 0:
            classes.append("FL")
        else:
            classes.append("X")
    return classes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-anchor each irregular segment so it forms a sharp two-leg V around a nearby rectangular blocker.

    Input structure:
    - Background, a bar color (straight border-anchored line perpendicular to the touched edge), a segment color
      (thin irregular stroke attached to the bar terminal), and an obstacle color whose components are filled
      rectangles placed deeper in the grid.

    Rule:
    - For each valid bar, find the attached segment endpoint and preserve that segment's total pixel count.
    - Find the nearest rectangular obstacle. In the bar-local frame, use the obstacle's top corner on the same
      side as the segment root; the output apex is the cell immediately before that corner in the forward
      direction.
    - Repaint the segment as exactly two diagonal-forward monotone phases: approach that apex, then immediately
      flip to the opposite diagonal and exit. There is no straight-middle phase and no edge-following run.

    Key insight:
    - The rectangle turns the old “wrap along the skin” motion into a predictable corner-graze: the near top
      corner fixes a unique V apex, so the whole output is determined by bar orientation, root side, rectangle
      placement, and preserved segment length.
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
                rs = [r for r, _ in comp]
                cs = [c for _, c in comp]
                vertical = len(set(cs)) == 1
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
                corner = (min_f, min_s) if side_sign > 0 else (min_f, max_s)
                apex_local = (corner[0] - 1, corner[1])
                p1 = apex_local[0]
                p3 = seg_len - 1 - p1
                if p1 < 1 or p3 < 1:
                    continue
                if apex_local[1] != side_sign * (p1 + 1):
                    continue
                route_local = _build_sharp_v(side_sign, p1, p3)
                route = [_from_local(terminal, fwd, side_basis, p) for p in route_local]
                if route[p1] != _from_local(terminal, fwd, side_basis, apex_local):
                    continue
                if any(not (0 <= r < h and 0 <= c < w) for r, c in route):
                    continue
                comp_set = set(comp)
                if any((r, c) in comp_set for r, c in route):
                    continue
                if any(int(g[r, c]) != bg and (r, c) not in seg_set for r, c in route):
                    continue
                md = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                obstacle_candidates.append(((md, min(r for r, _ in comp), min(c for _, c in comp)), comp, route, corner, apex_local))

        assert obstacle_candidates, f"no rectangular obstacle compatible with bar at {terminal}"
        _, obstacle_comp, new_pts, corner_local, apex_local = min(obstacle_candidates, key=lambda x: x[0])

        step_classes = _classify_steps(new_pts, fwd)
        assert "F" not in step_classes, f"sharp-V output may not contain straight phase: {step_classes}"
        first = step_classes[0]
        assert first in ("FL", "FR")
        split = 0
        while split < len(step_classes) and step_classes[split] == first:
            split += 1
        assert 0 < split < len(step_classes), "both diagonal phases must be non-empty"
        second = "FR" if first == "FL" else "FL"
        assert all(sc == second for sc in step_classes[split:]), f"non-monotone V: {step_classes}"
        apex = new_pts[split]
        expected_apex = _from_local(terminal, fwd, side_basis, apex_local)
        assert apex == expected_apex
        expected_corner = _from_local(terminal, fwd, side_basis, corner_local)
        assert expected_corner in set(obstacle_comp)

        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg
    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 9, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 5
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
RECT_H_MIN, RECT_H_MAX = 2, 4
RECT_W_MIN, RECT_W_MAX = 2, 4
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 400
PLACEMENT_PAD = 1
P1_CHOICES = (1, 2)

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
        last_was_diag = (step == diag_step)
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
    return coords


def _build_rectangle_from_apex(apex: tuple[int, int], side: int, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    corner = (apex[0] + 1, apex[1])
    if side > 0:
        return [(corner[0] + dr, corner[1] + dc) for dr in range(rect_h) for dc in range(rect_w)]
    return [(corner[0] + dr, corner[1] - dc) for dr in range(rect_h) for dc in range(rect_w)]


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
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None

    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    side = random.choice((-1, 1))
    p1 = random.choice([p for p in P1_CHOICES if p <= seg_len - 2])
    p3 = seg_len - 1 - p1
    if p3 < 1:
        return None

    bar = [(-k, 0) for k in range(bar_len)]
    output_seg = _build_sharp_v(side, p1, p3)
    seg_set = set(output_seg)
    bar_set = set(bar)

    apex = output_seg[p1]
    rect_h = random.randint(RECT_H_MIN, RECT_H_MAX)
    rect_w = random.randint(RECT_W_MIN, RECT_W_MAX)
    obstacle = _build_rectangle_from_apex(apex, side, rect_h, rect_w)
    obstacle_set = set(obstacle)
    if obstacle_set & seg_set:
        return None
    if obstacle_set & bar_set:
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if nearest_md not in (4, 6):
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
