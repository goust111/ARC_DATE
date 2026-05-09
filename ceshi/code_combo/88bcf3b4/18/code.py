# Combo variant of 88bcf3b4 (index 18)
# combo_spec: D1/0+D18/0
# Name: 螺旋圈数编码绕行

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, paint_coords


# Design analysis for the fused variant:
# a) Concrete behavior:
#    - Example: a top-edge bar of length 5 has its terminal at local (0,0), the segment root at (0,+1),
#      and a one-pixel obstacle at (4,0). The original task would form one monotone wrap near the obstacle.
#      Here, length 5 decodes to one full spiral turn: approach straight to (4,+1), then trace the fixed
#      square-spiral prefix around the obstacle until the route reaches the east side of radius 2. If extra
#      pixels remain, they extend outward as a short tail; they do not add more turning.
#    - Length 3 decodes to a half-turn and stops on the opposite side of radius 1.
#    - Length 7 decodes to one-and-a-half turns and reaches the opposite side of radius 2.
# b) Complexity calibration:
#    - Only bar lengths {3,5,7} are used. This gives exactly three visible turn-count categories with repeated
#      support across examples, instead of a large continuous parameter range.
#    - Obstacles are reduced to one-pixel anchors. That is a deliberate simplification: after adding encoded
#      spiral counts, multi-pixel irregular anchors would make the spiral center ambiguous and hurt inference.
# c) Hidden assumptions replaced:
#    - The original solver assumed a single monotone "approach / skin / exit" path. That no longer holds:
#      the new route revisits both sides of the obstacle and changes direction several times.
#    - The old "nearest skin" search and monotonicity asserts are therefore removed. The new solver assumes
#      each obstacle component is a one-pixel center marker, bar lengths are odd, and the segment attaches
#      perpendicularly to the terminal. The generator enforces all of these.
# d) Placement legality:
#    - The generator builds the output route first, derives a compact snake-like input stroke of equal length,
#      reserves the union of bar/obstacle/input/output with padding, and rejects placements that overlap or
#      leave the grid.
# e/f/g) Visual observability and cleanliness:
#    - The human-visible control variable is bar length, and the visible effect is spiral depth. Because the
#      obstacle is only one pixel and the tail is short, the output is easy to read: bigger bars produce visibly
#      deeper spirals rather than noisy tangles. When multiple triples are present, the generator tries to use
#      different bar lengths so the encoded rule can be compared within a single sample.


EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}

BAR_LENGTHS = [3, 5, 7]
HALF_TURNS_BY_BAR = {3: 1, 5: 2, 7: 3}
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 27
OBS_FWD_MIN, OBS_FWD_MAX = 3, 5
TAIL_MIN, TAIL_MAX = 0, 2
PLACEMENT_PAD = 2
MAX_RETRIES = 400
_GEN_CALL_COUNT = 0


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    conn = 8 if conn8 else 4
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=conn)]


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


def _edge_for_bar(comp: list[tuple[int, int]], h: int, w: int) -> tuple[str, tuple[int, int]]:
    e0, e1 = _bar_endpoints(comp)
    touch0 = _touches_border(e0, h, w)
    touch1 = _touches_border(e1, h, w)
    assert touch0 != touch1
    border_ep = e0 if touch0 else e1
    terminal = e1 if touch0 else e0
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        edge = "top" if border_ep[0] == 0 else "bottom"
    else:
        edge = "left" if border_ep[1] == 0 else "right"
    return edge, terminal


def _segment_neighbors(seg_pts: list[tuple[int, int]], p: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)
    r, c = p
    out = []
    for q in neighbors4(r, c):
        if q in sset:
            out.append(q)
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        q = (r + dr, c + dc)
        if q not in sset:
            continue
        mid1 = (r, c + dc)
        mid2 = (r + dr, c)
        if mid1 in sset or mid2 in sset:
            continue
        out.append(q)
    return out


def _segment_degrees(seg_pts: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {p: len(_segment_neighbors(seg_pts, p)) for p in seg_pts}


def _segment_main_path(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist: dict[tuple[int, int], int] = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in _segment_neighbors(seg_pts, cur):
            if nxt in prev or nxt not in sset:
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


def _local_to_global(
    local_pts: list[tuple[int, int]], edge: str, terminal: tuple[int, int]
) -> list[tuple[int, int]]:
    tr, tc = terminal
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lf * fr + ls * sr, tc + lf * fc + ls * sc) for lf, ls in local_pts]


def _global_delta_to_local(
    delta: tuple[int, int], edge: str
) -> tuple[int, int]:
    dr, dc = delta
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    lf = dr * fr + dc * fc
    ls = dr * sr + dc * sc
    return int(lf), int(ls)


def _base_output_local_path(
    obs_fwd: int, side_sign: int, half_turns: int
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    if half_turns == 1:
        norm = [(f, 1) for f in range(obs_fwd + 1)]
        norm.extend([(obs_fwd + 1, 1), (obs_fwd + 1, 0), (obs_fwd + 1, -1), (obs_fwd, -1)])
        exit_step = (0, -1)
    elif half_turns == 2:
        norm = [(0, 1), (0, 2)]
        norm.extend((f, 2) for f in range(1, obs_fwd))
        norm.extend(
            [
                (obs_fwd - 1, 1),
                (obs_fwd - 1, 0),
                (obs_fwd - 1, -1),
                (obs_fwd, -1),
                (obs_fwd + 1, -1),
                (obs_fwd + 1, 0),
                (obs_fwd + 1, 1),
                (obs_fwd + 1, 2),
                (obs_fwd, 2),
            ]
        )
        exit_step = (0, 1)
    else:
        norm = [(0, 1), (0, 2), (0, 3)]
        norm.extend((f, 3) for f in range(1, obs_fwd - 1))
        norm.extend(
            [
                (obs_fwd - 2, 2),
                (obs_fwd - 2, 1),
                (obs_fwd - 1, 1),
                (obs_fwd - 1, 0),
                (obs_fwd - 1, -1),
                (obs_fwd, -1),
                (obs_fwd + 1, -1),
                (obs_fwd + 1, 0),
                (obs_fwd + 1, 1),
                (obs_fwd + 1, 2),
                (obs_fwd + 2, 2),
                (obs_fwd + 2, 1),
                (obs_fwd + 2, 0),
                (obs_fwd + 2, -1),
                (obs_fwd + 2, -2),
                (obs_fwd + 1, -2),
                (obs_fwd, -2),
            ]
        )
        exit_step = (0, -1)

    route = [(lf, side_sign * ls) for lf, ls in norm]
    step = (exit_step[0], side_sign * exit_step[1])
    assert len(set(route)) == len(route), "base spiral template must stay simple"
    return route, step


def _build_output_local_path(
    obs_fwd: int, side_sign: int, half_turns: int, seg_len: int
) -> list[tuple[int, int]]:
    route, exit_step = _base_output_local_path(obs_fwd, side_sign, half_turns)
    assert seg_len >= len(route)
    while len(route) < seg_len:
        lf, ls = route[-1]
        route.append((lf + exit_step[0], ls + exit_step[1]))
    return route


def _build_input_local_path(seg_len: int, obs_fwd: int, side_sign: int) -> list[tuple[int, int]]:
    start = (0, side_sign)
    max_fwd = obs_fwd + 6
    max_side = max(5, (seg_len + max_fwd - 1) // max_fwd + 3)
    blocked = {(obs_fwd, 0)}

    def in_region(p: tuple[int, int]) -> bool:
        lf, ls = p
        return 0 <= lf <= max_fwd and 1 <= side_sign * ls <= max_side

    def is_clear(p: tuple[int, int]) -> bool:
        lf, ls = p
        return abs(lf - obs_fwd) + abs(ls) >= 2

    def candidate_steps(cur: tuple[int, int]) -> list[tuple[int, int]]:
        lf, ls = cur
        cands = []
        for dlf, dls in [
            (1, 0),
            (1, side_sign),
            (0, side_sign),
            (1, -side_sign),
            (0, -side_sign),
            (-1, 0),
            (-1, side_sign),
            (-1, -side_sign),
        ]:
            nxt = (lf + dlf, ls + dls)
            if not in_region(nxt) or not is_clear(nxt) or nxt in blocked:
                continue
            cands.append(nxt)
        random.shuffle(cands)
        cands.sort(key=lambda p: (p[0], side_sign * p[1]))
        return cands

    def valid_extension(path: list[tuple[int, int]], nxt: tuple[int, int]) -> bool:
        if nxt in path:
            return False
        trial = path + [nxt]
        deg = _segment_degrees(trial)
        if any(v > 2 for v in deg.values()):
            return False
        if len(_segment_neighbors(trial, nxt)) != 1:
            return False
        return True

    for _ in range(80):
        path = [start]
        stack: list[list[tuple[int, int]]] = [candidate_steps(start)]
        while path:
            if len(path) == seg_len:
                deg = _segment_degrees(path)
                endpoints = [p for p, d in deg.items() if d == 1]
                if len(endpoints) == 2 and start in endpoints:
                    return path
            if not stack[-1]:
                stack.pop()
                path.pop()
                continue
            nxt = stack[-1].pop()
            if not valid_extension(path, nxt):
                continue
            path.append(nxt)
            stack.append(candidate_steps(nxt))
    raise AssertionError("failed to build a simple input path")


def _nearest_obstacle_center(
    g: np.ndarray, colors: list[int], bar_col: int, seg_col: int, terminal: tuple[int, int]
) -> tuple[int, int]:
    best = None
    best_center = None
    for col in colors:
        if col in (bar_col, seg_col):
            continue
        for comp in _components(g, col, conn8=False):
            assert len(comp) == 1, "spiral-center obstacles must be one-pixel components"
            center = comp[0]
            d = abs(center[0] - terminal[0]) + abs(center[1] - terminal[1])
            key = (d, center[0], center[1])
            if best is None or key < best:
                best = key
                best_center = center
    assert best_center is not None
    return best_center


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rewind each irregular stroke into a bar-coded spiral around a nearby center marker, preserving length.

    Input structure:
    - Background plus three foreground roles: a straight border-anchored bar, a one-pixel obstacle center,
      and a one-cell-thick segment whose endpoint is attached perpendicularly to the bar terminal.
    - Bar lengths are restricted to three odd values; that length is part of the instruction, not decoration.

    Rule:
    - Detect the bar terminal, measure the attached segment length, and find the nearest obstacle center.
    - Decode spiral depth from bar length: 3 -> half-turn, 5 -> full turn, 7 -> one-and-a-half turns.
    - Repaint the segment so it approaches the obstacle on its attachment side, traces that many spiral
      stages around the obstacle, then uses any leftover pixels as a short straight tail pointing outward.

    Key insight:
    - The segment length still only tells you how many pixels to preserve. The new functional parameter is
      bar length: it specifies how much of the canonical spiral to draw around the obstacle center.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    def bar_candidates():
        for col in colors:
            for comp in _components(g, col, conn8=False):
                if len(comp) not in HALF_TURNS_BY_BAR or not _is_straight_line(comp):
                    continue
                edge, terminal = _edge_for_bar(comp, h, w)
                tr, tc = terminal
                attached = []
                for nr, nc in neighbors4(tr, tc):
                    if 0 <= nr < h and 0 <= nc < w:
                        v = int(g[nr, nc])
                        if v != bg and v != col:
                            attached.append((v, (nr, nc)))
                if attached:
                    yield col, comp, edge, terminal, attached

    moves: list[tuple[list[tuple[int, int]], int, list[tuple[int, int]]]] = []

    for bar_col, bar_comp, edge, terminal, attached in sorted(
        list(bar_candidates()), key=lambda x: (x[0], x[3][0], x[3][1])
    ):
        best_seg = None
        for cand_col, cand_start in attached:
            for comp in _components(g, cand_col, conn8=True):
                if cand_start not in comp:
                    continue
                deg = _segment_degrees(comp)
                if deg.get(cand_start, 0) != 1:
                    break
                path = _segment_main_path(comp, cand_start)
                key = (len(path), len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp, path)
                break
        assert best_seg is not None, "bar must have a valid attached segment"
        (_, _, _, _, _), seg_col, seg_start, seg_comp, seg_path = best_seg
        seg_len = len(seg_path)

        lf0, ls0 = _global_delta_to_local((seg_start[0] - terminal[0], seg_start[1] - terminal[1]), edge)
        assert lf0 == 0 and abs(ls0) == 1, "segment root must attach on a side cell"
        side_sign = 1 if ls0 > 0 else -1

        obstacle_center = _nearest_obstacle_center(g, colors, bar_col, seg_col, terminal)
        obs_fwd, obs_side = _global_delta_to_local(
            (obstacle_center[0] - terminal[0], obstacle_center[1] - terminal[1]), edge
        )
        assert obs_side == 0 and obs_fwd >= 3, "obstacle center must sit straight ahead of the terminal"

        half_turns = HALF_TURNS_BY_BAR[len(bar_comp)]
        new_local = _build_output_local_path(obs_fwd, side_sign, half_turns, seg_len)
        new_pts = _local_to_global(new_local, edge, terminal)

        assert len(new_pts) == seg_len
        assert len(set(new_pts)) == len(new_pts), "spiral route must stay simple"
        assert obstacle_center not in set(new_pts), "segment cannot overwrite obstacle center"
        assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), "route must stay in bounds"

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
    return _local_to_global(locals_, edge, _terminal_for(edge, n, bar_len, pos))


def _make_triple(bar_len: int) -> dict:
    half_turns = HALF_TURNS_BY_BAR[bar_len]
    obs_fwd = random.randint(OBS_FWD_MIN, OBS_FWD_MAX)
    side = random.choice((-1, 1))
    min_len = len(_base_output_local_path(obs_fwd, side, half_turns)[0])
    tail_len = random.randint(TAIL_MIN, TAIL_MAX)
    seg_len = min_len + tail_len

    bar = [(-k, 0) for k in range(bar_len)]
    obstacle = [(obs_fwd, 0)]
    output_seg = _build_output_local_path(obs_fwd, side, half_turns, seg_len)
    input_seg = _build_input_local_path(seg_len, obs_fwd, side)

    assert len(set(output_seg)) == len(output_seg)
    assert len(set(input_seg)) == len(input_seg)
    assert not (set(bar) & set(output_seg))
    assert not (set(bar) & set(input_seg))
    assert not (set(obstacle) & set(output_seg))
    assert not (set(obstacle) & set(input_seg))
    assert all(abs(r - obs_fwd) + abs(c) >= 2 for r, c in input_seg)

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "half_turns": half_turns,
        "obs_fwd": obs_fwd,
        "seg_len": seg_len,
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
        placed = {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "half_turns": triple["half_turns"],
            "bar_len": bar_len,
            "reserve": (r0, c0, r1, c1),
        }
        return placed
    return None


def _pick_bar_lengths(triple_count: int, preferred_len: int) -> list[int]:
    if triple_count == 1:
        return [preferred_len]

    others = [v for v in BAR_LENGTHS if v != preferred_len]
    random.shuffle(others)
    chosen = [preferred_len]
    for v in others:
        if len(chosen) == triple_count:
            break
        chosen.append(v)
    while len(chosen) < triple_count:
        chosen.append(random.choice(BAR_LENGTHS))
    random.shuffle(chosen)
    return chosen


def _gen_attempt(preferred_len: int) -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 20:
        triple_count = 1
    elif n < 24:
        triple_count = random.randint(1, 2)
    else:
        triple_count = random.randint(2, 3)

    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    edges = random.sample(list(EDGE_DATA), triple_count)
    chosen_lengths = _pick_bar_lengths(triple_count, preferred_len)

    for edge, bar_len in zip(edges, chosen_lengths):
        triple = _make_triple(bar_len)
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
    global _GEN_CALL_COUNT
    preferred_len = BAR_LENGTHS[_GEN_CALL_COUNT % len(BAR_LENGTHS)]
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(preferred_len)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
