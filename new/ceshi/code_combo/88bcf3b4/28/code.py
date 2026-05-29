# Combo variant of 88bcf3b4 (index 28)
# combo_spec: D4/3+D8/3+D16/0
# Name: U形方向匹配正面障碍
#
# Design analysis
# ---------------
# 1. Fusion rule:
#    - Keep the original straight border bar vocabulary, but replace the single dangling rope with a fixed
#      double-ended U-like 8-connected segment whose two endpoints both touch the same terminal laterally.
#    - One U endpoint has an outward-diagonal first step; that visible local direction is the D8/3 signal.
#      The obstacle on that same side is the one to route around, even though another obstacle component may
#      sit closer to the terminal.
#    - Only the direction-matched endpoint detaches (D4/3 keeps the other endpoint attached), and the
#      rerouted output becomes a frontal-obstacle detour that heads across the bar axis and exits on the
#      matched side (D16/0).
# 2. Observability / non-degeneracy:
#    - Every sample contains both a matched frontal obstacle and a distractor obstacle. The distractor is
#      intentionally closer to the terminal, so nearest-obstacle reasoning is wrong.
#    - The matched side is readable from shape alone: one endpoint immediately steps outward-diagonally while
#      the other starts with a plain forward step.
#    - The frontal obstacle is visibly ahead of the bar and stays fixed while the rerouted rope emerges on
#      the same side, so the fusion can be inferred from 2-4 examples.
# 3. New solver assumptions audited and generator guarantees:
#    - The segment is a simple 8-connected path of exactly 8 cells with both degree-1 endpoints in the
#      terminal's 4-neighborhood. The generator uses one of two mirrored canonical templates.
#    - Exactly one obstacle component on the matched side contains an axis cell directly in front of the bar.
#      The generator builds that component explicitly and keeps the distractor axis-free.
#    - The output route is a same-length canonical frontal detour with no overlaps against the bar or either
#      obstacle. The generator is output-first and validates `main(inp) == out` before accepting a sample.

from __future__ import annotations

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


def _segment_neighbors(p: tuple[int, int], sset: set[tuple[int, int]]) -> list[tuple[int, int]]:
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
    sset = set(seg_pts)
    return {p: len(_segment_neighbors(p, sset)) for p in sset}


def _segment_path_from_endpoint(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)
    path = [start]
    prev = None
    cur = start
    seen = {start}
    while True:
        nxts = [q for q in _segment_neighbors(cur, sset) if q != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise AssertionError("segment must be a simple path")
        nxt = nxts[0]
        if nxt in seen:
            raise AssertionError("segment path must not self-intersect")
        path.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt
    if len(path) != len(seg_pts):
        raise AssertionError("endpoint traversal must cover the whole segment")
    return path


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
    if touch0 == touch1:
        raise AssertionError("bar must touch exactly one border endpoint")
    border_ep = e0 if touch0 else e1
    terminal = e1 if touch0 else e0
    if len({c for _, c in comp}) == 1:
        return ("top" if border_ep[0] == 0 else "bottom"), terminal
    return ("left" if border_ep[1] == 0 else "right"), terminal


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


def _input_template(move_side: int) -> list[tuple[int, int]]:
    base = [(0, -1), (1, -2), (2, -2), (3, -1), (3, 0), (2, 1), (1, 1), (0, 1)]
    if move_side == -1:
        return base
    return [(r, -c) for r, c in base]


def _output_template(move_side: int) -> list[tuple[int, int]]:
    base = [(0, 1), (1, 1), (2, 0), (3, -1), (4, -2), (5, -2), (6, -2), (7, -2)]
    if move_side == -1:
        return base
    return [(r, -c) for r, c in base]


def _matched_obstacle(move_side: int) -> list[tuple[int, int]]:
    pts = {(5, 0), (5, move_side), (6, move_side)}
    if random.random() < 0.5:
        pts.add((6, 0))
    else:
        pts.add((7, move_side))
    return sorted(pts)


def _distractor_obstacle(move_side: int) -> list[tuple[int, int]]:
    side = -3 * move_side
    pts = {(1, side), (2, side), (3, side)}
    if random.random() < 0.5:
        pts.add((4, side))
    return sorted(pts)


def _bar_candidates(g: np.ndarray) -> list[tuple[int, list[tuple[int, int]], tuple[int, int]]]:
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = []
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
            out.append((col, comp, terminal))
    return out


def _collect_moves(g: np.ndarray) -> tuple[int, list[dict]]:
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    moves = []

    for bar_col, bar_comp, terminal in sorted(_bar_candidates(g), key=lambda x: (-len(x[1]), x[0], x[2])):
        edge, terminal = _edge_for_bar(bar_comp, h, w)
        fwd = EDGE_DATA[edge]["fwd"]
        side_vec = EDGE_DATA[edge]["side"]

        terminal_neigh = {}
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    terminal_neigh[(nr, nc)] = v
        if len(terminal_neigh) < 2:
            continue

        best_seg = None
        for cand_col in sorted(set(terminal_neigh.values())):
            for seg_comp in _components(g, cand_col, conn8=True):
                attached = [p for p in terminal_neigh if p in set(seg_comp)]
                if len(attached) < 2:
                    continue
                deg = _segment_degrees(seg_comp)
                endpoints = sorted([p for p, d in deg.items() if d == 1])
                if len(endpoints) != 2:
                    continue
                if sorted(attached) != endpoints:
                    continue
                path = _segment_path_from_endpoint(seg_comp, endpoints[0])
                if len(path) != 8:
                    continue
                local_eps = sorted(_to_local(p, terminal, fwd, side_vec) for p in endpoints)
                if local_eps != [(0, -1), (0, 1)]:
                    continue
                key = (len(path), cand_col)
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, seg_comp, endpoints)
        if best_seg is None:
            continue

        (_, _), seg_col, seg_comp, endpoints = best_seg
        endpoint_paths = {ep: _segment_path_from_endpoint(seg_comp, ep) for ep in endpoints}

        move_side = None
        keep_side = None
        for ep in endpoints:
            ep_local = _to_local(ep, terminal, fwd, side_vec)
            nxt_local = _to_local(endpoint_paths[ep][1], terminal, fwd, side_vec)
            step = (nxt_local[0] - ep_local[0], nxt_local[1] - ep_local[1])
            if step == (1, ep_local[1]):
                move_side = 1 if ep_local[1] > 0 else -1
            elif step == (1, 0):
                keep_side = 1 if ep_local[1] > 0 else -1
        if move_side is None or keep_side is None or move_side != -keep_side:
            raise AssertionError("failed to decode U endpoint directions")

        obs_candidates = []
        for obs_col in colors:
            if obs_col in (bar_col, seg_col):
                continue
            for obs_comp in _components(g, obs_col, conn8=True):
                obs_local = [_to_local(p, terminal, fwd, side_vec) for p in obs_comp]
                side_sum = sum(ls for _, ls in obs_local)
                side_hint = 0 if side_sum == 0 else (1 if side_sum > 0 else -1)
                has_axis = any(lf >= 1 and ls == 0 for lf, ls in obs_local)
                obs_candidates.append((side_hint, has_axis, set(obs_comp)))

        chosen_obstacle = None
        for side_hint, has_axis, obs_set in obs_candidates:
            if side_hint == move_side and has_axis:
                chosen_obstacle = obs_set
                break
        if chosen_obstacle is None:
            raise AssertionError("missing matched frontal obstacle")

        new_pts = [_from_local(p, terminal, fwd, side_vec) for p in _output_template(move_side)]
        if len(set(new_pts)) != 8:
            raise AssertionError("output path must stay simple")
        if not all(0 <= r < h and 0 <= c < w for r, c in new_pts):
            raise AssertionError("output path out of bounds")
        if any(p in set(bar_comp) for p in new_pts):
            raise AssertionError("output overlaps bar")
        if any(p in chosen_obstacle for p in new_pts):
            raise AssertionError("output overlaps chosen obstacle")

        moves.append({"seg_comp": seg_comp, "seg_col": seg_col, "new_pts": new_pts})
    return bg, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route one side of a double-ended U segment after decoding which endpoint direction matches the frontal obstacle.

    Input structure:
    - Background plus 1-2 triples. Each triple contains a straight bar touching one grid edge, an 8-connected
      U-like segment with both endpoints attached laterally to the same terminal, a frontal obstacle on one
      side of the bar axis, and a closer distractor obstacle on the opposite side.

    Rule:
    - Recover the U segment whose two degree-1 endpoints are the two terminal-adjacent cells.
    - Inspect the first step from each endpoint: the endpoint whose first move goes outward-diagonally marks
      the matched side, so the frontal obstacle on that side is the one that matters.
    - Remove the whole input U and repaint a same-length route that keeps the opposite endpoint attached,
      crosses the axis, and exits on the matched side around that frontal obstacle.

    Key insight:
    - The obstacle is not chosen by proximity. The decisive signal is the U endpoint's initial direction:
      one side visibly kicks outward toward its frontal obstacle, and that side is the one that detaches.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg, moves = _collect_moves(g)

    for move in moves:
        for r, c in move["seg_comp"]:
            out[r, c] = bg
    for move in moves:
        for r, c in move["new_pts"]:
            if int(out[r, c]) == bg:
                out[r, c] = move["seg_col"]
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 14, 22
BAR_LEN_MIN, BAR_LEN_MAX = 3, 6
PLACEMENT_PAD = 1
MAX_RETRIES = 400


def _terminal_for(edge: str, n: int, bar_len: int, pos: int) -> tuple[int, int]:
    if edge == "top":
        return (bar_len - 1, pos)
    if edge == "bottom":
        return (n - bar_len, pos)
    if edge == "left":
        return (pos, bar_len - 1)
    return (pos, n - bar_len)


def _transform(locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _make_triple(n: int) -> dict | None:
    bar_len = random.randint(BAR_LEN_MIN, min(BAR_LEN_MAX, n - 8))
    move_side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    input_seg = _input_template(move_side)
    output_seg = _output_template(move_side)
    obstacle = _matched_obstacle(move_side) + _distractor_obstacle(move_side)

    union = bar + input_seg + output_seg + obstacle
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    if max(rs) - min(rs) + 1 > n or max(cs) - min(cs) + 1 > n:
        return None

    return {
        "bar_len": bar_len,
        "bar": bar,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "obstacle": obstacle,
        "move_side": move_side,
    }


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    bar_len = triple["bar_len"]
    union = triple["bar"] + triple["input_seg"] + triple["output_seg"] + triple["obstacle"]
    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        return {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    triple_count = 1

    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed = []

    for edge in random.sample(list(EDGE_DATA), triple_count):
        triple = _make_triple(n)
        if triple is None:
            return None
        placement = _place_triple(triple, edge, n, reserved)
        if placement is None:
            return None

        paint_coords(inp, placement["bar"], bar_color)
        paint_coords(inp, placement["input_seg"], seg_color)
        paint_coords(inp, placement["obstacle"], obstacle_color)

        paint_coords(out, placement["bar"], bar_color)
        paint_coords(out, placement["output_seg"], seg_color)
        paint_coords(out, placement["obstacle"], obstacle_color)

        r0, c0, r1, c1 = placement["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed.append(placement)

    if not placed:
        return None

    # Fusion guard: the frontal obstacle used by the output is not the closest obstacle component.
    obs_comps = _components(inp, obstacle_color, conn8=True)
    if len(obs_comps) < 2:
        return None
    for p in placed:
        tr, tc = p["terminal"]
        comp_dists = sorted(
            min(abs(r - tr) + abs(c - tc) for r, c in comp)
            for comp in obs_comps
        )
        if len(comp_dists) < 2 or comp_dists[0] == comp_dists[1]:
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
    raise RuntimeError("failed to generate valid input for 88bcf3b4 combo D4/3+D8/3+D16/0")
