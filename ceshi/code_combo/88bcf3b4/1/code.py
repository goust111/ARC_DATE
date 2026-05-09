# Combo variant of 88bcf3b4 (index 1)
# combo_spec: D2/1+D3/3
# Name: 障碍物颜色双义绕行
#
# Design analysis
# ---------------
# fusion_spec:
#   summary:
#     - Keep the original bar + irregular input segment + nearby obstacle setup.
#     - Fuse two changes into one shared cue carried by obstacle color:
#         1) obstacle color decides which mirrored wrap direction is used
#         2) the obstacle-hugging middle phase is painted with that same obstacle color
#     - So obstacle color has dual meaning: it controls geometry and it appears again on the trajectory.
#   behavior_trace:
#     - Local frame uses terminal=(0,0), forward=away from the touched border, side=left/right of the bar axis.
#     - Example:
#         attach_side = +1, seg_len = 7, obstacle_color = 8, turn_side = -1, phase lengths = (2,2,2)
#         input segment still starts at (0,+1) and wanders backward away from the obstacle.
#         old task / root-side-tied geometry would wrap to the +1 side:
#             [(0,1),(1,2),(2,3),(3,3),(4,3),(5,2),(6,1)]
#         fused task instead uses obstacle color 8 -> turn_side -1:
#             coords = [(0,1),(1,0),(2,-1),(3,-1),(4,-1),(5,0),(6,1)]
#             colors = [seg, seg, seg, obs, obs, seg, seg]
#         difference:
#           same attached start pixel, same total length, but both the mirror direction and the middle paint
#           band are now read from obstacle color instead of from the attachment side alone.
#   complexity_calibration:
#     - Only two dedicated obstacle colors are used, giving a binary mapping that humans can infer from a few
#       examples instead of needing arbitrary per-color lookup.
#     - Generator forces visible three-phase outputs with p1>=1, p2>=2, p3>=1 so both the direction cue and
#       the recolored middle phase are observable.
#     - attach_side stays independent from obstacle color, and each generated grid rejects the degenerate case
#       where every triple satisfies attach_side == turn_side; this prevents silent collapse back to the
#       original side-driven geometry.
#   implicit_assumption_audit:
#     - Original solver implicitly tied `seg_start` side offset to the future wrap side. That assumption is
#       invalid after D2/1, so solver now separates:
#           attach_side = where the input segment touches the terminal
#           turn_side   = mirror wrap side decoded from obstacle color
#     - Repainting all rerouted cells with `seg_col` is invalid after D3/3, so route planning returns both
#       coordinates and per-cell colors.
#     - The new solver assumes obstacle colors are reserved and never reused by bars/segments. Generator
#       guarantees that by sampling bar/segment/bg from the remaining 8 colors.
#     - The solver infers p1/p2 from a support column embedded in the obstacle on the pocket's inner side.
#       Generator guarantees that support column exists and that extra obstacle growth stays inside the pocket,
#       so no earlier false support candidate appears.
#   placement_legality:
#     - Generation remains output-first.
#     - First build the intended wrapped output in local coordinates.
#     - Then place the obstacle from the apex-inner support column and only grow inward inside the pocket.
#     - Then build the detached input segment of the same length, ensuring it is not 4-adjacent to the
#       obstacle.
#     - A reserved padded bbox still separates multiple triples so nearest-obstacle matching remains unambiguous.
#   visual_observability:
#     - The same obstacle color is visible both on the obstacle sprite and on the middle strip of the rerouted
#       segment, making the color rule structural rather than arbitrary.
#     - Because attach_side can disagree with turn_side, viewers can see that obstacle color affects mirror
#       choice, not merely recoloring.
#     - When two or more triples fit, generator schedules both obstacle colors in the same grid, giving direct
#       side-by-side evidence of the binary mapping.
#   cleanliness:
#     - Obstacles grow only inside the wrap pocket and not around the full segment, so the colored middle strip
#       stays legible and does not visually merge into unrelated clutter.

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
SEG_COLOR_PLACEHOLDER = 99


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


def _build_output_local(seg_len: int, attach_side: int, turn_side: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(0, attach_side)]
    for phase_len, step in zip([p1, p2, p3], [(1, turn_side), (1, 0), (1, -turn_side)]):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
    assert len(coords) == seg_len
    return coords


def _build_output_colors(seg_len: int, p1: int, p2: int, p3: int, seg_col: int, obstacle_col: int) -> list[int]:
    cols = [int(seg_col)] * (p1 + 1)
    cols.extend([int(obstacle_col)] * p2)
    cols.extend([int(seg_col)] * p3)
    assert len(cols) == seg_len
    return cols


def _infer_wrap_lengths(
    obstacle_local: set[tuple[int, int]], seg_len: int, attach_side: int, turn_side: int
) -> tuple[int, int, int]:
    candidates = sorted(
        lf
        for lf, ls in obstacle_local
        if 1 <= lf <= seg_len - 1 and ls == attach_side + (lf - 1) * turn_side
    )
    assert candidates, "obstacle must expose an apex-inner support seed"
    p1 = candidates[0]
    inner_col = attach_side + (p1 - 1) * turn_side
    remain = seg_len - 1 - p1
    p2 = 0
    while p2 < remain and (p1 + p2 + 1, inner_col) in obstacle_local:
        p2 += 1
    p3 = seg_len - 1 - p1 - p2
    assert p2 >= 0 and p3 >= 0
    return p1, p2, p3


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
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

        best_seg = None
        for cand_col, cand_start in neigh:
            if cand_col in OBSTACLE_TURN_BY_COLOR:
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
                mr = min(r for r, _ in comp)
                mc = min(c for _, c in comp)
                key = (d, mr, mc)
                if best_obstacle is None or key < best_obstacle[0]:
                    best_obstacle = (key, obstacle_col, set(comp))
        assert best_obstacle is not None, "must find a dedicated obstacle component"
        _, obstacle_col, obstacle_set = best_obstacle

        _, fwd, side = _bar_frame(bar_comp, h, w)
        attach_local = _to_local(seg_start, terminal, fwd, side)
        assert attach_local[0] == 0 and abs(attach_local[1]) == 1, "segment must attach laterally to terminal"
        attach_side = attach_local[1]
        turn_side = OBSTACLE_TURN_BY_COLOR[obstacle_col]

        obstacle_local = {_to_local(p, terminal, fwd, side) for p in obstacle_set}
        p1, p2, p3 = _infer_wrap_lengths(obstacle_local, seg_len, attach_side, turn_side)
        new_local = _build_output_local(seg_len, attach_side, turn_side, p1, p2, p3)
        new_pts = [_from_local(p, terminal, fwd, side) for p in new_local]
        new_cols = _build_output_colors(seg_len, p1, p2, p3, int(seg_col), int(obstacle_col))

        assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), "output segment must stay within grid"
        assert new_pts[0] == seg_start, "rerouted segment must keep the same attached start pixel"
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
        moves.append(
            {
                "seg_comp": seg_comp,
                "seg_col": int(seg_col),
                "obstacle_col": int(obstacle_col),
                "attach_side": int(attach_side),
                "turn_side": int(turn_side),
                "p1": int(p1),
                "p2": int(p2),
                "p3": int(p3),
                "new_pts": new_pts,
                "new_cols": new_cols,
                "middle_color_cells": int(sum(col == int(obstacle_col) for col in new_cols)),
            }
        )
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route each irregular segment around the nearest obstacle while preserving its length, but let obstacle
    color determine both the wrap direction and the color of the obstacle-hugging middle phase.

    Input structure:
    - Background, a straight bar touching one border, an irregular 1-cell-thick segment attached to the bar's
      inner endpoint, and a nearby obstacle sprite.
    - Obstacles use one of two dedicated colors; bars and input segments use other colors.

    Rule:
    - Read the attached segment endpoint at the bar terminal to get the input attachment side.
    - Find the nearest obstacle; its color decides whether the rerouted path mirrors clockwise-like or
      counterclockwise-like around the obstacle.
    - Rebuild a monotone 3-phase wrapped path of the same length: diagonal approach, straight alongside, then
      diagonal exit. Phase 1 and phase 3 keep the original segment color, while phase 2 uses the obstacle color.

    Key insight:
    - Obstacle color carries two linked semantics at once: it selects which mirrored wrap geometry is used, and
      the same color reappears on the segment exactly where the path runs alongside the obstacle.
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


GRID_SIZE_MIN, GRID_SIZE_MAX = 17, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
PHASE2_MIN = 2
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 4000
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


def _build_input_path(seg_len: int, attach_side: int, turns: int) -> list[tuple[int, int]]:
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


def _grow_obstacle_in_pocket(
    support: list[tuple[int, int]], turn_side: int, obs_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    support_set = set(support)
    if support_set & forbidden:
        return None
    obstacle = set(support_set)
    inner_col = support[0][1]
    min_row = min(r for r, _ in support)
    max_row = max(r for r, _ in support)

    def inward_ok(cell: tuple[int, int]) -> bool:
        r, c = cell
        if cell in forbidden or cell in obstacle:
            return False
        if not (min_row <= r <= max_row):
            return False
        if turn_side == 1 and c > inner_col:
            return False
        if turn_side == -1 and c < inner_col:
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
        random.shuffle(cands)
        obstacle.add(cands[0])
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


def _make_triple(n: int, obstacle_color: int) -> dict | None:
    turn_side = OBSTACLE_TURN_BY_COLOR[obstacle_color]
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
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

    remaining = seg_len - 1
    max_p1 = min(PHASE_MAX, remaining - (PHASE2_MIN + 1))
    if max_p1 < 1:
        return None
    p1 = random.randint(1, max_p1)
    remaining -= p1
    max_p2 = min(PHASE_MAX, remaining - 1)
    if max_p2 < PHASE2_MIN:
        return None
    p2 = random.randint(PHASE2_MIN, max_p2)
    p3 = remaining - p2
    if p3 < 1:
        return None

    output_seg = _build_output_local(seg_len, attach_side, turn_side, p1, p2, p3)
    output_cols = _build_output_colors(seg_len, p1, p2, p3, SEG_COLOR_PLACEHOLDER, obstacle_color)
    seg_set = set(output_seg)

    apex = output_seg[p1]
    inner_col = apex[1] - turn_side
    support = [(p1 + k, inner_col) for k in range(p2 + 1)]
    obs_size = random.randint(max(OBS_SIZE_MIN, len(support)), OBS_SIZE_MAX)
    obstacle = _grow_obstacle_in_pocket(support, turn_side, obs_size, seg_set | bar_set)
    if obstacle is None:
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, attach_side, turns)
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "obstacle_color": int(obstacle_color),
        "input_seg": input_seg,
        "output_seg": output_seg,
        "output_cols": output_cols,
        "attach_side": int(attach_side),
        "turn_side": int(turn_side),
        "p1": int(p1),
        "p2": int(p2),
        "p3": int(p3),
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
            "output_cols": list(triple["output_cols"]),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
            "attach_side": triple["attach_side"],
            "turn_side": triple["turn_side"],
            "p1": triple["p1"],
            "p2": triple["p2"],
            "p3": triple["p3"],
        }
    return None


def _paint_output_path(out: np.ndarray, pts: list[tuple[int, int]], cols: list[int]) -> None:
    for (r, c), col in zip(pts, cols):
        out[r, c] = int(col)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    # Prefer exactly two simultaneous triples so one grid shows both obstacle colors side by side
    # without the extra placement pressure of a third unrelated instance.
    triple_count = 2

    non_obstacle_colors = [c for c in range(10) if c not in OBSTACLE_TURN_BY_COLOR]
    bg, bar_color, seg_color = random.sample(non_obstacle_colors, 3)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    chosen_edges = random.sample(list(EDGE_DATA), triple_count)
    color_schedule = [CW_OBSTACLE_COLOR, CCW_OBSTACLE_COLOR]
    random.shuffle(color_schedule)

    for edge, obstacle_color in zip(chosen_edges, color_schedule):
        triple = _make_triple(n, obstacle_color)
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
        resolved_cols = [seg_color if col == SEG_COLOR_PLACEHOLDER else int(col) for col in placed["output_cols"]]
        _paint_output_path(out, placed["output_seg"], resolved_cols)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append(placed)

    if not any(placed["attach_side"] != placed["turn_side"] for placed in placed_triples):
        return None

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
        moves = _plan_moves(inp)
        if len(moves) != len(placed_triples):
            return None
        if any(move["middle_color_cells"] < PHASE2_MIN for move in moves):
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
