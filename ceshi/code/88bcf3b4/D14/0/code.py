# Variant of 88bcf3b4
# Dimension: D14 — 颜色使用规则 (meta)
# Change: 每个 triple 使用 3 种不同颜色（bar 色、线段色、障碍物色），背景色为最频繁色；多个 triple 之间可共享颜色。→ bar 与线段同色
#
# Design analysis for the D14 variant
# - Concretization:
#   Each triple still contains a border-anchored bar, an irregular rope segment, and one obstacle sprite.
#   The visible change is that bar and segment now share one color, so they merge into a single same-color
#   connected component. The only reliable cue that separates them is shape: the bar is a straight prefix
#   that starts on the border, while the segment is the irregular suffix attached at the inner endpoint.
# - Behavior trace:
#   Old solver flow: detect a straight component of color A, then search for a different-colored neighbor B
#   at its terminal and reroute only the B pixels around the nearest obstacle color.
#   New solver flow: classify every non-background 8-connected component either as a "rope" component or an
#   obstacle component. A rope component must be a simple path with exactly one border endpoint; walking from
#   that endpoint inward exposes a maximal straight prefix (bar) and the remaining suffix (segment).
# - Complexity calibration:
#   The original grid sizes, bar lengths, segment lengths, and obstacle sizes are preserved. Multiple triples
#   are still generated, and colors may still be reused across triples, so the task does not collapse into a
#   trivial unique-color-per-role puzzle.
# - Hidden assumptions audited:
#   The original code assumed `bar_color != seg_color` in bar detection, terminal-neighbor lookup, obstacle
#   filtering, walkability, and repainting. Those assumptions are invalid here. The new solver therefore uses
#   geometry to split a same-color rope, and only the current segment suffix cells are considered walkable for
#   rerouting.
# - Placement legality:
#   Because bar and segment now merge, the generator must forbid same-color shapes that would create a 2x2
#   block, a degree-3 junction, or a second border endpoint. It does this by forcing the input rope to leave
#   the terminal diagonally first and by validating the combined bar+input-segment component with the same
#   geometric classifier used by the solver.
# - Visual observability:
#   Across samples, humans can see that a straight border bar and an irregular dangling suffix share a color,
#   while the obstacle keeps a contrasting color. This makes the D14 change visible without introducing extra
#   hidden conventions.
# - Visual neatness / color inference:
#   Obstacles stay monochrome, and each triple only needs two non-background colors, which keeps the grid
#   readable even when multiple triples appear together.

import random
import re
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _touches_border(p: tuple[int, int], h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _bfs_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _bfs_path_with_diag_filter(
    start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray
) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}

    def can_step(r: int, c: int, nr: int, nc: int) -> bool:
        dr = nr - r
        dc = nc - c
        if dr == 0 or dc == 0:
            return True
        mid1 = (r, c + dc)
        mid2 = (r + dr, c)
        if 0 <= mid1[0] < h and 0 <= mid1[1] < w and bool(walkable[mid1]):
            return False
        if 0 <= mid2[0] < h and 0 <= mid2[1] < w and bool(walkable[mid2]):
            return False
        return True

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            if not can_step(r, c, nr, nc):
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _greedy_manhattan_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape

    def md(p: tuple[int, int]) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    cur = start
    path = [cur]
    seen = {cur}
    while cur != goal:
        cur_md = md(cur)
        best = None
        for nr, nc in neighbors8(cur[0], cur[1]):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            nmd = md(nxt)
            if nmd >= cur_md:
                continue
            is_diag = 1 if (nr != cur[0] and nc != cur[1]) else 0
            key = (nmd, is_diag, nr, nc)
            if best is None or key < best:
                best = key
        if best is None:
            return None
        _, _, nr, nc = best
        cur = (nr, nc)
        path.append(cur)
        seen.add(cur)
    return path


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


def _allowed_border_fwds(p: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    r, c = p
    out = set()
    if r == 0:
        out.add((1, 0))
    if r == h - 1:
        out.add((-1, 0))
    if c == 0:
        out.add((0, 1))
    if c == w - 1:
        out.add((0, -1))
    return out


def _classify_bar_segment_component(
    comp: list[tuple[int, int]], h: int, w: int
) -> dict[str, object] | None:
    if len(comp) < BAR_LEN_MIN + SEG_LEN_MIN:
        return None

    deg = _segment_degrees(comp)
    if any(d < 1 or d > 2 for d in deg.values()):
        return None

    endpoints = sorted([p for p, d in deg.items() if d == 1])
    if len(endpoints) != 2:
        return None

    border_eps = [p for p in endpoints if _touches_border(p, h, w)]
    if len(border_eps) != 1:
        return None

    border_ep = border_eps[0]
    path = _segment_main_path(comp, border_ep)
    if len(path) != len(comp) or set(path) != set(comp):
        return None
    if len(path) < 2:
        return None

    fwd = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    if fwd not in _allowed_border_fwds(border_ep, h, w):
        return None
    if abs(fwd[0]) + abs(fwd[1]) != 1:
        return None

    terminal_idx = 1
    while terminal_idx < len(path):
        step = (path[terminal_idx][0] - path[terminal_idx - 1][0], path[terminal_idx][1] - path[terminal_idx - 1][1])
        if step != fwd:
            break
        terminal_idx += 1
    terminal_idx -= 1
    if terminal_idx + 1 >= len(path):
        return None

    bar_comp = path[: terminal_idx + 1]
    seg_comp = path[terminal_idx + 1 :]
    if len(bar_comp) < BAR_LEN_MIN or len(seg_comp) < SEG_LEN_MIN:
        return None

    attach_step = (seg_comp[0][0] - bar_comp[-1][0], seg_comp[0][1] - bar_comp[-1][1])
    if abs(attach_step[0]) + abs(attach_step[1]) != 1:
        return None
    if attach_step[0] * fwd[0] + attach_step[1] * fwd[1] != 0:
        return None

    return {
        "path": path,
        "bar_comp": bar_comp,
        "seg_comp": seg_comp,
        "seg_set": set(seg_comp),
        "terminal": bar_comp[-1],
        "seg_start": seg_comp[0],
        "fwd": fwd,
    }


def _classify_scene_components(grid: np.ndarray) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})

    bars: list[dict[str, object]] = []
    obstacles: list[dict[str, object]] = []
    for col in colors:
        for comp in _components(g, col, conn8=True):
            item = {"color": col, "pts": comp, "set": set(comp)}
            parsed = _classify_bar_segment_component(comp, h, w)
            if parsed is None:
                obstacles.append(item)
            else:
                parsed["color"] = col
                parsed["full_comp"] = comp
                parsed["full_set"] = set(comp)
                bars.append(parsed)
    return bg, bars, obstacles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swing the irregular suffix of each same-color rope around the nearest obstacle, preserving its length.
    The bar is the straight border-anchored prefix of the rope; the movable segment is the remaining suffix.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, bar_infos, obstacle_infos = _classify_scene_components(g)

    out = g.copy()
    moves = []
    for info in sorted(bar_infos, key=lambda x: (-len(x["bar_comp"]), x["color"], x["terminal"])):
        bar_comp = info["bar_comp"]
        seg_comp = info["seg_comp"]
        seg_set = info["seg_set"]
        seg_color = int(info["color"])
        terminal = info["terminal"]
        seg_start = info["seg_start"]
        fwd = info["fwd"]
        L = len(seg_comp)

        best = None
        obstacle_set = None
        for obstacle in obstacle_infos:
            comp = obstacle["pts"]
            d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
            mr = min(r for r, _ in comp)
            mc = min(c for _, c in comp)
            key = (d, mr, mc)
            if best is None or key < best:
                best = key
                obstacle_set = obstacle["set"]
        assert obstacle_set is not None, "must find at least one obstacle component"

        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]

        walkable = g == bg
        walkable[terminal] = True
        for p in seg_set:
            walkable[p] = True

        def current_seg_walkable(p: tuple[int, int]) -> bool:
            return int(g[p]) == bg or p in seg_set

        def pick_p_near() -> tuple[int, int]:
            good = []
            for pr, pc in obstacle_set:
                er, ec = pr + off_r, pc + off_c
                if not (0 <= er < h and 0 <= ec < w):
                    continue
                if (er, ec) in obstacle_set:
                    continue
                if not current_seg_walkable((er, ec)):
                    continue
                d = abs(pr - terminal[0]) + abs(pc - terminal[1])
                good.append((d, pr, pc))
            assert good, "no obstacle pixel has a valid translated endpoint outside obstacle"
            _, pr, pc = min(good)
            return (pr, pc)

        p_near = pick_p_near()
        walkable[p_near] = True
        path_t_to_obs = _greedy_manhattan_path(terminal, p_near, walkable) or _bfs_path(terminal, p_near, walkable)
        assert path_t_to_obs is not None, f"no path from terminal {terminal} to obstacle {p_near}"

        shifted = [(r + off_r, c + off_c) for r, c in path_t_to_obs]
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted segment path must stay within grid"
        assert shifted and shifted[0] == seg_start, "shifted path must start at seg_start"
        entry = shifted[-1]

        skin = set()
        for r, c in obstacle_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
        cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
        away = (cr - terminal[0], cc - terminal[1])

        def skin_ok(p: tuple[int, int]) -> bool:
            return current_seg_walkable(p)

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]

        def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
            r, c = p
            md = abs(r - terminal[0]) + abs(c - terminal[1])
            dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
            return (md, dot, -r, -c)

        exit_cell = max(skin_candidates, key=exit_key)

        skin_walkable = np.zeros((h, w), dtype=bool)
        for p in skin:
            if skin_ok(p):
                skin_walkable[p] = True
        skin_walkable[entry] = True
        skin_walkable[exit_cell] = True
        boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]
        assert len(boundary_path) >= 2, "boundary path must have at least 2 cells to derive extension direction"

        dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
        dc_ext = boundary_path[-1][1] - boundary_path[-2][1]

        route = list(shifted)
        if boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        cur = route[-1]
        while len(route) < L:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if not current_seg_walkable(nxt):
                break
            route.append(nxt)
            cur = nxt

        new_pts = route[:L]
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
        assert not any(p in set(bar_comp) for p in new_pts), "output segment overlaps the bar"

        step_classes = []
        for j in range(len(new_pts) - 1):
            dr = new_pts[j + 1][0] - new_pts[j][0]
            dc = new_pts[j + 1][1] - new_pts[j][1]
            fwd_comp = dr * fwd[0] + dc * fwd[1]
            perp_comp = dr * fwd[1] + dc * (-fwd[0])
            if abs(dr) + abs(dc) == 1:
                step_classes.append("F" if fwd_comp > 0 else "X")
            elif fwd_comp > 0 and perp_comp > 0:
                step_classes.append("FR")
            elif fwd_comp > 0 and perp_comp < 0:
                step_classes.append("FL")
            else:
                step_classes.append("X")
        s = "".join(step_classes)
        assert re.fullmatch(r"(FL)*(F)*(FR)*", s) or re.fullmatch(r"(FR)*(F)*(FL)*", s), (
            f"output segment not monotone wrap: {' '.join(step_classes)}"
        )

        first_step = step_classes[0] if step_classes else ""
        if first_step in ("FL", "FR"):
            p1_end = 0
            for j2, sc in enumerate(step_classes):
                if sc == first_step:
                    p1_end = j2 + 1
                else:
                    break
            apex = new_pts[p1_end]
            perp_dir = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
            inner = (-perp_dir[0], -perp_dir[1])
            apex_inner = (apex[0] + inner[0], apex[1] + inner[1])
            assert apex_inner in obstacle_set, f"obstacle not tight at V apex: apex={apex}, inner={apex_inner}"

        moves.append((seg_comp, seg_color, new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_color, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_color
    return out


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
MAX_RETRIES = 1200
PER_EDGE_RETRIES = 40
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


def _grow_obstacle(
    start: tuple[int, int], max_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    obstacle = {start}
    for _ in range(max_size - 1):
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


def _build_input_path(seg_len: int, turns: int, side: int) -> list[tuple[int, int]]:
    # Force an initial diagonal step so the same-color rope stays a simple path instead of
    # creating a 2x2 block or a degree-3 junction against the bar.
    coords = [(0, side), (-1, 2 * side)]
    remaining = seg_len - 2
    diag_step = (-1, side)
    orth_step = (-1, 0)
    last_was_diag = True
    turns_left = turns
    while remaining > 0:
        if turns_left <= 0:
            step = diag_step if last_was_diag else orth_step
            run = remaining
        else:
            if last_was_diag:
                is_diag = random.random() < DIAG_CONTINUE_PROB
            else:
                is_diag = random.random() < DIAG_START_PROB
            step = diag_step if is_diag else orth_step
            if is_diag != last_was_diag:
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


def _make_triple(n: int) -> dict[str, object] | None:
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

    remaining = seg_len - 1
    p1 = random.randint(1, min(remaining - 1, PHASE_MAX))
    remaining -= p1
    p2 = random.randint(0, min(remaining, PHASE_MAX))
    p3 = remaining - p2

    output_seg = _build_output_path(seg_len, side, p1, p2, p3)
    seg_set = set(output_seg)

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    obs_size = random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(apex_inner, obs_size, seg_set | bar_set)

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
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


def _place_triple(
    triple: dict[str, object], edge: str, n: int, reserved: np.ndarray
) -> dict[str, object] | None:
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
            "reserve": (r0, c0, r1, c1),
        }
        if _classify_bar_segment_component(placed["bar"] + placed["input_seg"], n, n) is None:
            continue
        return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    triple_count = random.choice([1, 2, 3])
    min_n = {1: GRID_SIZE_MIN, 2: 12, 3: 16}[triple_count]
    n = random.randint(min_n, GRID_SIZE_MAX)

    bg = random.randrange(10)
    rope_color, obstacle_color = random.sample([c for c in range(10) if c != bg], 2)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    for edge in random.sample(list(EDGE_DATA), triple_count):
        placed = None
        for _ in range(PER_EDGE_RETRIES):
            triple = _make_triple(n)
            if triple is None:
                continue
            placed = _place_triple(triple, edge, n, reserved)
            if placed is not None:
                break
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], rope_color)
        paint_coords(inp, placed["input_seg"], rope_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)

        paint_coords(out, placed["bar"], rope_color)
        paint_coords(out, placed["output_seg"], rope_color)
        paint_coords(out, placed["obstacle"], obstacle_color)

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
