# Combo variant of 88bcf3b4 (index 13)
# combo_spec: D2/1+D8/0
# Name: 颜色双重配对绕行
#
# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    - Original 88bcf3b4 uses local geometry twice: each bar chooses the nearest obstacle, and the side where the
#      irregular segment attaches to the terminal implicitly determines whether the output first leans left or right.
#    - This combo breaks both dependencies and replaces them with a two-stage color logic:
#      1) the bar color decides which obstacle color family it must pair with;
#      2) the matched obstacle color decides the wrap direction.
#    - Concrete example in local terminal coordinates:
#      a bar of color 1 maps to obstacle color 2, and obstacle color 2 means `turn_side=+1`.
#      Another bar of color 3 maps to obstacle color 8, and obstacle color 8 means `turn_side=-1`.
#      If both input segments start at `(0, +1)`, the first output can begin `(1, +1), (2, +2), ...`,
#      while the second begins `(1, -1), (2, -2), ...`, even though the input attachment side is identical.
#      Also, if the color-8 obstacle happens to be closer to the color-1 bar terminal, that bar must still ignore it
#      and wrap the farther color-2 obstacle.
#    - The visible difference from the original task is therefore twofold on the same sample:
#      nearest obstacle can be wrong, and identical attachment geometry can yield opposite mirror wraps.
# b) Complexity calibration:
#    - Use exactly two triples per generated sample. That is the smallest count that can simultaneously expose:
#      color-based pairing across obstacles and color-based direction across the two obstacle colors.
#    - Keep only two bar colors and two obstacle colors globally. This is enough structure to infer a fixed mapping
#      without turning the task into a large memorization table.
#    - Both triples share the same `attach_side` in generated samples, so humans can directly compare outputs and see
#      that attachment side alone no longer determines the wrap direction.
# c) Hidden assumptions audited:
#    - Original `main()` assumed obstacle choice can be made by nearest-Manhattan distance. That is invalid now, so the
#      combo solver must look up the target obstacle by bar color instead.
#    - Original `main()` reused the segment's lateral offset to infer the wrap mirror. That also breaks, so the solver
#      now separates `attach_side` from `turn_side`.
#    - The D2/1 single-variant solver relies on the matched obstacle exposing a support column that uniquely reveals
#      `(p1, p2, p3)`. The combo generator must preserve that invariant for the color-matched obstacle specifically.
#    - Because the matched obstacle is no longer selected by distance, generator placement must explicitly guarantee
#      that every bar has exactly one obstacle of its mapped color; otherwise color matching would be ambiguous.
# d) Placement legality:
#    - Generation stays output-first. Each triple is built locally from a valid output wrap, then an irregular input
#      segment of equal length is synthesized backwards from the terminal.
#    - Global placement searches only adjacent edge pairs and rejects any overlap across the full union of each triple's
#      input/output geometry, so bars, obstacles, and paths remain separated and in-bounds.
#    - The generator also rejects any sample where the obstacle growth corrupts the intended support-column signature.
# e) Visual observability:
#    - Each sample contains both obstacle colors and both bar colors. That means the solver must notice a stable color
#      pairing, not a one-off local coincidence.
#    - At least one triple is forced to have a nearer wrong-color obstacle than its own matched obstacle, so distance
#      visibly fails as a rule.
#    - Because both triples share the same input attachment side but have opposite obstacle colors, the mirror-direction
#      mapping is also directly observable from a single input/output pair.
# f) Cleanliness:
#    - Only five colors appear: background, one segment color, two bar colors, and two obstacle colors.
#    - The obstacle itself stays small and 4-connected, and the routed segment remains a one-pixel path, so the color
#      cues remain readable instead of dissolving into large mixed regions.
# g) Inference support:
#    - No decision depends on color ids as numbers. The rules use fixed visible color categories:
#      bar color selects the obstacle family, obstacle color selects the wrap direction.
#    - To avoid degeneracy, generation rejects samples where color-based pairing would accidentally agree with the old
#      nearest-obstacle rule for both triples.

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

BAR_TO_OBSTACLE_COLOR = {
    1: CW_OBSTACLE_COLOR,
    3: CCW_OBSTACLE_COLOR,
}

BAR_COLORS = tuple(BAR_TO_OBSTACLE_COLOR)
OBSTACLE_COLORS = tuple(OBSTACLE_TURN_BY_COLOR)

GRID_SIZE_MIN, GRID_SIZE_MAX = 15, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 1200

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}

ADJACENT_EDGE_PAIRS = [
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
]


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


def _stroke_neighbors(p: tuple[int, int], sset: set[tuple[int, int]]):
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


def _segment_main_path(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist: dict[tuple[int, int], int] = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in _stroke_neighbors(cur, sset):
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
    return {p: sum(1 for _ in _stroke_neighbors(p, sset)) for p in sset}


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


def _build_output_local(seg_len: int, attach_side: int, turn_side: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(0, attach_side)]
    for phase_len, step in zip([p1, p2, p3], [(1, turn_side), (1, 0), (1, -turn_side)]):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
    assert len(coords) == seg_len
    return coords


def _infer_wrap_lengths(
    obstacle_local: set[tuple[int, int]], seg_len: int, attach_side: int, turn_side: int
) -> tuple[int, int, int]:
    candidates = sorted(
        lf
        for lf, ls in obstacle_local
        if lf >= 1 and ls == attach_side + (lf - 1) * turn_side and lf <= seg_len - 1
    )
    assert candidates, "obstacle must expose a support-diagonal apex candidate"
    p1 = candidates[0]
    inner_col = attach_side + (p1 - 1) * turn_side
    remain = seg_len - 1 - p1
    p2 = 0
    while p2 < remain and (p1 + p2 + 1, inner_col) in obstacle_local:
        p2 += 1
    p3 = seg_len - 1 - p1 - p2
    assert p3 >= 0
    return p1, p2, p3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route each irregular rope segment around the obstacle selected by a color pairing, preserving its length.

    Input structure:
    - Background plus two dedicated bar colors. Each bar is a straight border-perpendicular line with one endpoint on
      the grid edge and an irregular segment attached laterally at its inner terminal.
    - All movable segments share one non-bar, non-obstacle color.
    - Obstacles use exactly two dedicated colors. A bar's color determines which obstacle color it must target, and the
      obstacle's own color determines whether the wrap follows the clockwise-like or counterclockwise-like mirror.

    Rule:
    - Detect every valid bar and its attached one-pixel-thick segment endpoint.
    - Ignore nearest-obstacle distance. Instead, map the bar color to its designated obstacle color and select that
      obstacle component.
    - Convert the matched obstacle color into a turn direction, infer the three monotone output phases from the
      obstacle's inner support column, and redraw a same-length wrapped segment while leaving bars and obstacles fixed.

    Key insight:
    - Color controls two separate reasoning steps. Bar color answers "which obstacle is mine?", and obstacle color
      answers "which mirrored way do I go around it?". Attachment side and raw proximity are only distractions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    def bar_candidates():
        for col in colors:
            if col not in BAR_TO_OBSTACLE_COLOR:
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
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (x[0], x[2])):
        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

        best_seg = None
        for cand_col, cand_start in neigh:
            if cand_col in BAR_COLORS or cand_col in OBSTACLE_COLORS:
                continue
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

        obstacle_col = BAR_TO_OBSTACLE_COLOR[bar_col]
        obstacle_components = _components(g, obstacle_col, conn8=True)
        assert len(obstacle_components) == 1, f"expected exactly one obstacle of color {obstacle_col}"
        obstacle_set = set(obstacle_components[0])

        _, fwd, side = _bar_frame(bar_comp, h, w)
        attach_local = _to_local(seg_start, terminal, fwd, side)
        assert attach_local[0] == 0 and abs(attach_local[1]) == 1, "segment must attach laterally to terminal"
        attach_side = attach_local[1]
        turn_side = OBSTACLE_TURN_BY_COLOR[obstacle_col]

        obstacle_local = {_to_local(p, terminal, fwd, side) for p in obstacle_set}
        p1, p2, p3 = _infer_wrap_lengths(obstacle_local, L, attach_side, turn_side)
        new_local = _build_output_local(L, attach_side, turn_side, p1, p2, p3)
        new_pts = [_from_local(p, terminal, fwd, side) for p in new_local]

        assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), "output segment must stay within grid"
        assert new_pts[0] == seg_start, "rerouted segment must keep the same attached start pixel"
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
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


def _grow_obstacle(
    starts: list[tuple[int, int]], max_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    obstacle = set(starts)
    while len(obstacle) < max_size:
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle and (nr, nc) not in forbidden
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _build_input_path(seg_len: int, turns: int, attach_side: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, attach_side)]
    diag_step = (-1, attach_side)
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


def _make_triple(n: int, attach_side: int, bar_color: int) -> dict | None:
    obstacle_color = BAR_TO_OBSTACLE_COLOR[bar_color]
    turn_side = OBSTACLE_TURN_BY_COLOR[obstacle_color]

    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)

    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    remaining = seg_len - 1
    p1 = random.randint(1, min(PHASE_MAX, remaining))
    remaining -= p1
    p2 = random.randint(0, min(PHASE_MAX, remaining))
    p3 = remaining - p2

    output_seg = _build_output_local(seg_len, attach_side, turn_side, p1, p2, p3)
    seg_set = set(output_seg)
    apex = output_seg[p1]
    inner_col = apex[1] - turn_side
    support = [(p1 + k, inner_col) for k in range(p2 + 1)]
    if any(p in seg_set or p in bar_set for p in support):
        return None

    obs_size = random.randint(max(OBS_SIZE_MIN, len(support)), OBS_SIZE_MAX)
    obstacle = _grow_obstacle(support, obs_size, seg_set | bar_set)
    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    if _infer_wrap_lengths(set(obstacle), seg_len, attach_side, turn_side) != (p1, p2, p3):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, attach_side)
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_color": bar_color,
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "obstacle_color": obstacle_color,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "attach_side": attach_side,
        "turn_side": turn_side,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


def _place_triple_at(triple: dict, edge: str, n: int, pos: int) -> dict | None:
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"]
    union_g = _transform(union, edge, n, triple["bar_len"], pos)
    if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
        return None
    return {
        "bar_color": triple["bar_color"],
        "bar": _transform(triple["bar"], edge, n, triple["bar_len"], pos),
        "obstacle": _transform(triple["obstacle"], edge, n, triple["bar_len"], pos),
        "obstacle_color": triple["obstacle_color"],
        "input_seg": _transform(triple["input_seg"], edge, n, triple["bar_len"], pos),
        "output_seg": _transform(triple["output_seg"], edge, n, triple["bar_len"], pos),
        "terminal": _transform([(0, 0)], edge, n, triple["bar_len"], pos)[0],
        "attach_side": triple["attach_side"],
        "turn_side": triple["turn_side"],
        "union": set(union_g),
    }


def _candidate_placements(triple: dict, edge: str, n: int) -> list[dict]:
    positions = list(range(n))
    random.shuffle(positions)
    out = []
    for pos in positions:
        placed = _place_triple_at(triple, edge, n, pos)
        if placed is not None:
            out.append(placed)
    return out


def _pairing_conflict_stats(placed_triples: list[dict]) -> tuple[int, bool]:
    count = 0
    has_tie = False
    for i, placed in enumerate(placed_triples):
        tr, tc = placed["terminal"]
        own = min(abs(r - tr) + abs(c - tc) for r, c in placed["obstacle"])
        other = min(
            abs(r - tr) + abs(c - tc)
            for j, other_placed in enumerate(placed_triples)
            if j != i
            for r, c in other_placed["obstacle"]
        )
        if other == own:
            has_tie = True
        if other < own:
            count += 1
    return count, has_tie


def _find_placement_pair(n: int, triples: list[dict], edges: tuple[str, str]) -> list[dict] | None:
    cand_lists = [_candidate_placements(triple, edge, n) for triple, edge in zip(triples, edges)]
    if any(not cands for cands in cand_lists):
        return None
    first, second = cand_lists
    random.shuffle(first)
    random.shuffle(second)
    for a in first:
        for b in second:
            if a["union"] & b["union"]:
                continue
            placed = [a, b]
            conflict_count, has_tie = _pairing_conflict_stats(placed)
            if has_tie or conflict_count < 1:
                continue
            return placed
    return None


def _build_grids(n: int, bg: int, seg_color: int, placed_triples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    for placed in placed_triples:
        paint_coords(inp, placed["bar"], placed["bar_color"])
        paint_coords(inp, placed["obstacle"], placed["obstacle_color"])
        paint_coords(inp, placed["input_seg"], seg_color)

        paint_coords(out, placed["bar"], placed["bar_color"])
        paint_coords(out, placed["obstacle"], placed["obstacle_color"])
        paint_coords(out, placed["output_seg"], seg_color)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    attach_side = random.choice((-1, 1))
    bg, seg_color = random.sample([c for c in range(10) if c not in BAR_COLORS and c not in OBSTACLE_COLORS], 2)

    edges = random.choice(ADJACENT_EDGE_PAIRS)
    bar_colors = list(BAR_COLORS)
    random.shuffle(bar_colors)

    triples = []
    for bar_color in bar_colors:
        triple = _make_triple(n, attach_side, bar_color)
        if triple is None:
            return None
        triples.append(triple)

    placed_triples = _find_placement_pair(n, triples, edges)
    if placed_triples is None:
        return None

    if len({placed["turn_side"] for placed in placed_triples}) != 2:
        return None
    if any(placed["attach_side"] != attach_side for placed in placed_triples):
        return None
    if sum(int(placed["attach_side"] != placed["turn_side"]) for placed in placed_triples) < 1:
        return None

    inp, out = _build_grids(n, bg, seg_color, placed_triples)

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
    raise RuntimeError(f"failed to generate a valid combo input after {MAX_RETRIES} attempts")
