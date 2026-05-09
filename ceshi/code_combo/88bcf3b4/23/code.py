# Combo variant of 88bcf3b4 (index 23)
# combo_spec: D2/1+D3/3+D17/0
# Name: 颜色编码方向等分双色绕行
#
# Design analysis
# ---------------
# a) Concrete behavior and traced example
#    - Keep the original object vocabulary: border bar, attached irregular segment, nearby obstacle.
#    - Replace the old contour-search interpretation with an explicit local-coordinate wrap:
#      the segment start is still one lateral step from the bar terminal, but the rerouted output is
#      reconstructed as three monotone phases in the bar's forward frame.
#    - Example in local coordinates, terminal at (0, 0), attached start at (0, +1), total length 8:
#      original task might implicitly choose direction from the attachment side and paint everything
#      with the segment color.
#      this combo instead chooses turn_side from obstacle color, so color A may produce
#      [(0,1),(1,0),(2,-1),(3,-1),(4,-1),(5,0),(6,1),(7,2)],
#      while color B mirrors the diagonal sense.
#      Phase lengths come from equal splitting of 7 steps: (2, 2, 3).
#      Cells 0..2 stay in the segment color, cells 3..4 take the obstacle color, cells 5..7 return to
#      the segment color.
#    - The difference is therefore multi-dimensional but coherent:
#      obstacle color simultaneously tells the solver which mirror wrap to use and which middle cells
#      should recolor, while the color switch location confirms the equal-split length rule.
#
# b) Complexity calibration
#    - Only two dedicated obstacle colors are used, giving a binary direction code that can be learned
#      from a few training pairs.
#    - Segment lengths are restricted to 7-9 so the three phases stay visually legible and the middle
#      recolored strip is never reduced to a hard-to-see degenerate point.
#    - Attachment side remains independent from obstacle color, preventing the task from collapsing back
#      to the original "look at which side the segment is attached on" heuristic.
#
# c) Hidden-assumption audit against the original solver
#    - The original solver assumed the input segment offset also implies the wrap direction. That is no
#      longer true; attach_side and turn_side are now separate variables.
#    - The original solver discovered the middle run length from a BFS skin walk. Here that would be an
#      unstable inference because the new rule says the three phases are length-coded from the segment's
#      total size. The solver therefore derives (p1, p2, p3) directly from L-1.
#    - Because the middle run is recolored, repainting can no longer be a single-color write. The solver
#      must emit both coordinates and per-cell colors.
#    - Since the solver now trusts obstacle color as semantics, the generator must keep those two colors
#      exclusive to obstacles.
#
# d) Placement legality
#    - Generation remains output-first: build the intended rerouted path first, place a support strip of
#      obstacle pixels on the inner side of the bend, then derive an unrelated irregular input segment of
#      the same length.
#    - Reserved expanded bounding boxes keep different triples separated so "nearest obstacle" and
#      attached-segment matching remain unambiguous.
#    - The obstacle is grown only inward from the support strip, avoiding overlap with the bar, input
#      segment, and intended output path.
#
# e) Visual observability
#    - A human can see three independent but coupled clues in examples:
#      1. obstacle color changes the mirror direction,
#      2. the middle contiguous strip changes to obstacle color,
#      3. that strip begins exactly after one-third of the route's steps and ends after the second third.
#    - Because training examples can contain both obstacle colors and both attachment sides, the color to
#      direction mapping is recoverable from visual comparison rather than hidden numeric ordering.
#
# f) Cleanliness
#    - The output is still a single-pixel stroke plus a compact nearby sprite. No extra markers, labels,
#      or overlays are introduced.
#    - Obstacle growth is constrained to the pocket beside the phase-2 strip, so the recolored middle run
#      stays readable instead of being buried inside a large blob.
#
# g) Inferability of color change
#    - Only the obstacle carries the special direction-coding colors, and the recolored path segment always
#      matches that same obstacle color. This gives a strong structural cue for the new color rule.
#    - Since the recolored cells are also exactly the equal-split middle phase, the solver can validate
#      the segmentation from where the color switches occur; this is the intended synergy of D2/1, D3/3,
#      and D17/0.

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, paint_coords


CW_OBSTACLE_COLOR = 2
CCW_OBSTACLE_COLOR = 8
OBSTACLE_TURN_BY_COLOR = {
    CW_OBSTACLE_COLOR: 1,
    CCW_OBSTACLE_COLOR: -1,
}

GRID_SIZE_MIN, GRID_SIZE_MAX = 11, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
SEG_LEN_MIN, SEG_LEN_MAX = 7, 9
OBS_SIZE_MIN, OBS_SIZE_MAX = 3, 9
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 800
PLACEMENT_PAD = 1

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


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


def _bar_frame(bar_comp: list[tuple[int, int]], h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    e0, e1 = _bar_endpoints(bar_comp)
    touch0 = _touches_border(e0, h, w)
    terminal = e1 if touch0 else e0
    rs = [r for r, _ in bar_comp]
    cs = [c for _, c in bar_comp]
    if len(set(cs)) == 1:
        fwd = (1, 0) if min(rs) == 0 else (-1, 0)
        side = (0, 1) if fwd == (1, 0) else (0, -1)
    else:
        fwd = (0, 1) if min(cs) == 0 else (0, -1)
        side = (-1, 0) if fwd == (0, 1) else (1, 0)
    return terminal, fwd, side


def _to_local(
    p: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]
) -> tuple[int, int]:
    dr = p[0] - terminal[0]
    dc = p[1] - terminal[1]
    lf = dr * fwd[0] + dc * fwd[1]
    ls = dr * side[0] + dc * side[1]
    return int(lf), int(ls)


def _from_local(
    loc: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]
) -> tuple[int, int]:
    lf, ls = loc
    return (
        terminal[0] + lf * fwd[0] + ls * side[0],
        terminal[1] + lf * fwd[1] + ls * side[1],
    )


def _equal_phase_lengths(seg_len: int) -> tuple[int, int, int]:
    total_steps = seg_len - 1
    base = total_steps // 3
    rem = total_steps % 3
    p1 = base
    p2 = base
    p3 = base + rem
    assert p1 >= 1 and p2 >= 1 and p3 >= 1
    return p1, p2, p3


def _build_output_local(seg_len: int, attach_side: int, turn_side: int) -> list[tuple[int, int]]:
    p1, p2, p3 = _equal_phase_lengths(seg_len)
    coords = [(0, attach_side)]
    for phase_len, step in zip([p1, p2, p3], [(1, turn_side), (1, 0), (1, -turn_side)]):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
    assert len(coords) == seg_len
    return coords


def _expected_support_cells(attach_side: int, turn_side: int, seg_len: int) -> set[tuple[int, int]]:
    p1, p2, _ = _equal_phase_lengths(seg_len)
    inner_col = attach_side + (p1 - 1) * turn_side
    return {(p1 + k, inner_col) for k in range(p2 + 1)}


def _output_colors(seg_col: int, obstacle_col: int, seg_len: int) -> list[int]:
    p1, p2, p3 = _equal_phase_lengths(seg_len)
    cols = [seg_col] * (p1 + 1) + [obstacle_col] * p2 + [seg_col] * p3
    assert len(cols) == seg_len
    return cols


def _plan_moves(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})

    def bar_candidates():
        for col in colors:
            if col in OBSTACLE_TURN_BY_COLOR:
                continue
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
                if v != bg and v != bar_col and v not in OBSTACLE_TURN_BY_COLOR:
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
                seg_len = len(path)
                key = (seg_len, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (seg_len, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        best_obstacle = None
        for obstacle_col in OBSTACLE_TURN_BY_COLOR:
            for comp in _components(g, obstacle_col, conn8=True):
                d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                key = (d, min(r for r, _ in comp), min(c for _, c in comp))
                if best_obstacle is None or key < best_obstacle[0]:
                    best_obstacle = (key, obstacle_col, set(comp))
        assert best_obstacle is not None, "must find a coded obstacle"
        _, obstacle_col, obstacle_set = best_obstacle

        _, fwd, side = _bar_frame(bar_comp, h, w)
        attach_local = _to_local(seg_start, terminal, fwd, side)
        assert attach_local[0] == 0 and abs(attach_local[1]) == 1, "segment must attach laterally to the terminal"
        attach_side = attach_local[1]
        turn_side = OBSTACLE_TURN_BY_COLOR[obstacle_col]

        support_local = _expected_support_cells(attach_side, turn_side, seg_len)
        obstacle_local = {_to_local(p, terminal, fwd, side) for p in obstacle_set}
        assert support_local <= obstacle_local, "obstacle must contain the inner support strip for the equal split"

        new_local = _build_output_local(seg_len, attach_side, turn_side)
        new_pts = [_from_local(p, terminal, fwd, side) for p in new_local]
        new_cols = _output_colors(int(seg_col), int(obstacle_col), seg_len)

        assert new_pts[0] == seg_start, "rerouted path must keep the attached start pixel"
        assert len(new_pts) == len(new_cols) == seg_len
        assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), "output segment must stay within grid"
        assert not any(p in obstacle_set for p in new_pts), "rerouted segment must not overlap the obstacle"

        p1, p2, p3 = _equal_phase_lengths(seg_len)
        step_types = []
        for i in range(seg_len - 1):
            dr = new_local[i + 1][0] - new_local[i][0]
            dc = new_local[i + 1][1] - new_local[i][1]
            if i < p1:
                assert (dr, dc) == (1, turn_side)
                step_types.append("diag-in")
            elif i < p1 + p2:
                assert (dr, dc) == (1, 0)
                step_types.append("straight")
            else:
                assert (dr, dc) == (1, -turn_side)
                step_types.append("diag-out")

        moves.append(
            {
                "seg_comp": seg_comp,
                "seg_col": int(seg_col),
                "new_pts": new_pts,
                "new_cols": new_cols,
                "seg_len": seg_len,
                "attach_side": attach_side,
                "turn_side": turn_side,
                "obstacle_col": int(obstacle_col),
                "step_types": step_types,
            }
        )
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route each attached stroke into a three-phase wrap whose mirror direction is encoded by obstacle color.

    Input structure:
    - Background cells, one or more border-anchored bars, one irregular attached segment per bar, and one nearby
      obstacle sprite per group.
    - The obstacle uses one of two dedicated colors; those colors are semantic and indicate which mirror-direction
      the rerouted wrap must use.

    Rule:
    - Keep each bar and obstacle fixed.
    - For each bar, detach the irregular segment from its original shape but preserve its total pixel count.
    - Rebuild the segment as three monotone phases in the bar's forward frame: diagonal approach, straight
      alongside run, and opposite diagonal exit.
    - The three phase lengths are evenly split from the total step count: phase1 = floor((L-1)/3),
      phase2 = floor((L-1)/3), phase3 gets the remainder.
    - The obstacle color decides whether the diagonal phases bend clockwise or counterclockwise.
    - Cells in phase 2 are painted with the obstacle color; phases 1 and 3 keep the segment color.

    Key insight:
    - The obstacle does double duty: its color selects the wrap direction, and that same color reappears only on
      the equal-length middle phase, so the color-switch points reveal the segmentation rule.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    moves = _plan_moves(g)

    for move in moves:
        for r, c in move["seg_comp"]:
            out[r, c] = bg

    for move in moves:
        for (r, c), col in zip(move["new_pts"], move["new_cols"]):
            if int(out[r, c]) == bg:
                out[r, c] = int(col)
    return out


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


def _build_input_path(seg_len: int, turns: int, attach_side: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, attach_side)]
    step_by_mode = {
        "diag": (-1, attach_side),
        "up": (-1, 0),
        "side": (0, attach_side),
    }
    modes = list(step_by_mode)
    last_mode = None
    turns_left = turns
    while remaining > 0:
        if last_mode is None:
            mode = random.choice(modes)
        elif turns_left <= 0:
            mode = last_mode
        else:
            alt_modes = [m for m in modes if m != last_mode]
            if remaining == turns_left:
                mode = random.choice(alt_modes)
            elif last_mode == "diag":
                mode = last_mode if random.random() < DIAG_CONTINUE_PROB else random.choice(alt_modes)
            else:
                mode = random.choice(alt_modes) if random.random() < DIAG_START_PROB else last_mode
            if mode != last_mode:
                turns_left -= 1
        max_run = remaining if turns_left <= 0 else remaining - turns_left
        run = random.randint(1, max(1, min(3, max_run)))
        step = step_by_mode[mode]
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
        last_mode = mode
    assert len(coords) == len(set(coords)), "input stroke must stay simple"
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


def _grow_obstacle_from_support(
    support: set[tuple[int, int]], turn_side: int, obs_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    obstacle = set(support)
    inner_col = min(c for _, c in support) if turn_side == 1 else max(c for _, c in support)
    min_r = min(r for r, _ in support)
    max_r = max(r for r, _ in support)

    def inward_ok(cell: tuple[int, int]) -> bool:
        r, c = cell
        if cell in forbidden or cell in obstacle:
            return False
        if not (min_r <= r <= max_r):
            return False
        if (c - inner_col) * turn_side > 0:
            return False
        return True

    while len(obstacle) < obs_size:
        cands = []
        for r, c in obstacle:
            for nr, nc in neighbors4(r, c):
                if inward_ok((nr, nc)):
                    cands.append((nr, nc))
        if not cands:
            break
        obstacle.add(random.choice(cands))

    if not support <= obstacle:
        return None
    return list(obstacle)


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _make_triple(n: int, turn_side: int, obstacle_color: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - SEG_LEN_MIN + 1)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    attach_side = random.choice((-1, 1))

    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)
    output_seg = _build_output_local(seg_len, attach_side, turn_side)
    seg_set = set(output_seg)
    support = _expected_support_cells(attach_side, turn_side, seg_len)
    if support & (seg_set | bar_set):
        return None

    min_obs = max(OBS_SIZE_MIN, len(support))
    obs_size = random.randint(min_obs, OBS_SIZE_MAX)
    obstacle = _grow_obstacle_from_support(support, turn_side, obs_size, seg_set | bar_set)
    if obstacle is None:
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, attach_side)
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + list(obstacle) + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": list(obstacle),
        "obstacle_color": obstacle_color,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "attach_side": attach_side,
        "turn_side": turn_side,
        "seg_len": seg_len,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


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
            "obstacle_color": triple["obstacle_color"],
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
            "attach_side": triple["attach_side"],
            "turn_side": triple["turn_side"],
            "seg_len": triple["seg_len"],
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 13:
        max_triples = 1
    elif n < 17:
        max_triples = 2
    else:
        max_triples = 3
    triple_count = random.randint(1, max_triples)

    non_obstacle_colors = [c for c in range(10) if c not in OBSTACLE_TURN_BY_COLOR]
    bg, bar_color, seg_color = random.sample(non_obstacle_colors, 3)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    obstacle_colors = [CW_OBSTACLE_COLOR, CCW_OBSTACLE_COLOR]
    chosen_edges = random.sample(list(EDGE_DATA), triple_count)
    color_schedule = [random.choice(obstacle_colors) for _ in range(triple_count)]
    if triple_count >= 2:
        color_schedule[0] = CW_OBSTACLE_COLOR
        color_schedule[1] = CCW_OBSTACLE_COLOR
        random.shuffle(color_schedule)

    for edge, obstacle_color in zip(chosen_edges, color_schedule):
        turn_side = OBSTACLE_TURN_BY_COLOR[obstacle_color]
        triple = _make_triple(n, turn_side, obstacle_color)
        if triple is None:
            return None
        placed = _place_triple(triple, edge, n, reserved)
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["obstacle"], placed["obstacle_color"])
        paint_coords(inp, placed["input_seg"], seg_color)

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["obstacle"], placed["obstacle_color"])
        for (r, c), col in zip(placed["output_seg"], _output_colors(seg_color, placed["obstacle_color"], placed["seg_len"])):
            out[r, c] = col

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
