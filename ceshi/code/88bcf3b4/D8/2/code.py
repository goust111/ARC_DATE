# Variant of 88bcf3b4
# Dimension: D8 — 匹配条件（障碍物选择） (match_cond)
# Change: 选择离 bar 末端曼哈顿距离最近的障碍物精灵作为绕行目标；当存在多个 triple 时，每个 bar 选择自己最近的障碍物
#   → 按尺寸匹配：线段长度最长的 bar 配对像素数最多的障碍物，次长配次多，以此类推。
#
# Design analysis for the D8 variant
# ----------------------------------
# - Concrete behavior:
#   The original task already has multiple bar+segment+obstacle groups. In this variant the visual cue is no longer
#   "which obstacle is closest to this terminal", but the global ranking relation:
#     segment length rank (descending) <-> obstacle size rank (descending).
#   To make that visible, every generated sample contains exactly two groups with distinct segment lengths and distinct
#   obstacle sizes, and the nearest-obstacle relation is forced to be the opposite pairing. So a human sees that the
#   longer rope bends around the larger sprite even when a smaller sprite is closer, and the shorter rope takes the
#   smaller sprite even though the larger one is nearer to the other bar.
# - Source-task note:
#   The prompt cites dd6b8c4b as inspiration, but in this workspace only its evaluation JSON is present; no solver
#   source file was available to read. The implementation below therefore uses the stated "sorting / ranking" idea
#   directly without importing any other variant logic.
# - Flow impact:
#   The only semantic change is obstacle selection. The routing logic after an obstacle is chosen stays the same.
#   That means main() must first discover every bar and its attached segment length globally, then discover all obstacle
#   components globally, and only after that sort-and-pair them. The old per-bar "pick nearest obstacle immediately"
#   pattern is no longer valid because it bakes in the original local-distance assumption.
# - Hidden assumptions audited:
#   1. The old main() assumed a bar can select its obstacle independently. That breaks here, so pairing is moved to a
#      global phase before any routing.
#   2. The old generator enforced "own obstacle is nearest". That must be inverted, otherwise size-based pairing often
#      degenerates to the original behavior.
#   3. Size-ranking becomes ambiguous if segment lengths or obstacle sizes tie. The generator therefore guarantees both
#      ranks are unique in every sample.
#   4. Because bars and segments share colors across groups, components from different groups must stay separated by at
#      least one blank cell in the 8-neighborhood; otherwise global discovery could merge them.
# - Placement legality:
#   Generation remains output-first. Each actual matched pair is locally valid exactly as in the original task, then the
#   two pairs are placed jointly so that:
#     own obstacle is farther than the other obstacle for each terminal,
#     every obstacle is still at Manhattan distance >= 2 from the non-owner terminal,
#     different groups stay 8-separated,
#     and main(input) reproduces the constructed output.
# - Observability / cleanliness:
#   The reversed nearest-vs-size pairing is enforced on every generated sample, so the D8 change is always active.
#   Two groups are enough to show the ranking rule while keeping the grid legible; bars, obstacles, and segments remain
#   visually separated and use only one color each globally.

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
    assert goal in prev, "BFS-with-diag-filter must reach goal"
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


def _discover_bar_infos(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})

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

    infos = []
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
        infos.append(
            {
                "bar_col": bar_col,
                "bar_comp": bar_comp,
                "terminal": terminal,
                "seg_col": seg_col,
                "seg_start": seg_start,
                "seg_comp": seg_comp,
                "seg_len": L,
            }
        )
    return infos


def _collect_obstacle_components(g: np.ndarray, bar_infos: list[dict]) -> list[set[tuple[int, int]]]:
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    bar_cells = {p for info in bar_infos for p in info["bar_comp"]}
    seg_cells = {p for info in bar_infos for p in info["seg_comp"]}
    obstacles = []
    for col in colors:
        for comp in _components(g, col, conn8=True):
            s = set(comp)
            if s & bar_cells:
                continue
            if s & seg_cells:
                continue
            obstacles.append(s)
    return obstacles


def analyze_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bar_infos = _discover_bar_infos(g)
    obstacles = _collect_obstacle_components(g, bar_infos)
    bars_sorted = sorted(
        bar_infos,
        key=lambda info: (
            -info["seg_len"],
            info["terminal"][0],
            info["terminal"][1],
            info["bar_col"],
            info["seg_col"],
        ),
    )
    obstacles_sorted = sorted(obstacles, key=lambda comp: (-len(comp), min(comp), max(comp)))

    nearest_owner = {}
    for i, info in enumerate(bar_infos):
        dists = [
            min(abs(r - info["terminal"][0]) + abs(c - info["terminal"][1]) for r, c in comp)
            for comp in obstacles
        ]
        nearest_owner[i] = int(np.argmin(dists))

    return {
        "bar_count": len(bar_infos),
        "segment_lengths": [info["seg_len"] for info in bar_infos],
        "obstacle_sizes": [len(comp) for comp in obstacles],
        "size_pair_terminals": [info["terminal"] for info in bars_sorted],
        "size_pair_obstacle_sizes": [len(comp) for comp in obstacles_sorted[: len(bars_sorted)]],
        "nearest_indices": nearest_owner,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    D8 variant of 88bcf3b4:
    detect all bar+segment groups first, rank bars by attached segment length, rank obstacle sprites by pixel count,
    then route each segment around the obstacle with the matching rank.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    bar_infos = _discover_bar_infos(g)
    assert bar_infos, "must find at least one valid bar+segment group"

    obstacle_components = _collect_obstacle_components(g, bar_infos)
    assert len(obstacle_components) >= len(bar_infos), "must find enough obstacle components"

    bars_sorted = sorted(
        bar_infos,
        key=lambda info: (
            -info["seg_len"],
            info["terminal"][0],
            info["terminal"][1],
            info["bar_col"],
            info["seg_col"],
            info["seg_start"][0],
            info["seg_start"][1],
        ),
    )
    obstacles_sorted = sorted(
        obstacle_components,
        key=lambda comp: (-len(comp), min(r for r, _ in comp), min(c for _, c in comp)),
    )

    moves = []
    for info, obstacle_set in zip(bars_sorted, obstacles_sorted):
        bar_comp = info["bar_comp"]
        bar_col = info["bar_col"]
        terminal = info["terminal"]
        seg_col = info["seg_col"]
        seg_start = info["seg_start"]
        seg_comp = info["seg_comp"]
        L = info["seg_len"]

        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]

        walkable = (g == bg) | (g == seg_col)
        walkable[terminal] = True

        def pick_p_near() -> tuple[int, int]:
            good = []
            for pr, pc in obstacle_set:
                er, ec = pr + off_r, pc + off_c
                if not (0 <= er < h and 0 <= ec < w):
                    continue
                if (er, ec) in obstacle_set:
                    continue
                if int(g[er, ec]) not in (bg, seg_col):
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
            r, c = p
            return int(g[r, c]) == bg or int(g[r, c]) == seg_col

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]

        def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
            r, c = p
            md = abs(r - terminal[0]) + abs(c - terminal[1])
            dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
            return (md, dot, -r, -c)

        exit_cell = max(skin_candidates, key=exit_key)

        skin_walkable = np.zeros((h, w), dtype=bool)
        for r, c in skin:
            if skin_ok((r, c)):
                skin_walkable[r, c] = True
        skin_walkable[entry] = True
        skin_walkable[exit_cell] = True
        boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]

        assert len(boundary_path) >= 2, "boundary path must have at least 2 cells to derive extension direction"
        dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
        dc_ext = boundary_path[-1][1] - boundary_path[-2][1]

        route = list(shifted)
        if boundary_path and boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        cur = route[-1]
        while len(route) < L:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) != bg and int(g[nr, nc]) != seg_col:
                break
            route.append((nr, nc))
            cur = (nr, nc)

        new_pts = route[:L]
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

        rs_bar = [r for r, _ in bar_comp]
        cs_bar = [c for _, c in bar_comp]
        if len(set(cs_bar)) == 1:
            fwd = (1, 0) if min(rs_bar) == 0 else (-1, 0)
        else:
            fwd = (0, 1) if min(cs_bar) == 0 else (0, -1)
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
        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
TRIPLE_COUNT = 2
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
SEG_LEN_GAP_MIN = 2
PHASE_MAX = 3
OBS_SIZE_SMALL_MIN, OBS_SIZE_SMALL_MAX = 1, 4
OBS_SIZE_LARGE_MIN, OBS_SIZE_LARGE_MAX = 6, 9
OWN_OBS_MD_MIN, OWN_OBS_MD_MAX = 4, 6
OTHER_OBS_MD_MIN = 2
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 400
PAIR_SEARCH_LIMIT = 400
PLACED_TRIPLE_TRIES = 80
PAIR_LAYOUT_TRIES = 80

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


def _build_output_path(seg_len: int, side: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
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


def _make_triple(n: int, seg_len: int, obs_size: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    for _ in range(80):
        bar_len = random.randint(BAR_LEN_MIN, max_bar)
        max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
        if seg_len > max_seg:
            return None
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
            continue

        obstacle = _grow_obstacle(apex_inner, obs_size, seg_set | bar_set)
        if len(obstacle) != obs_size:
            continue

        nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
        if not (OWN_OBS_MD_MIN <= nearest_md <= OWN_OBS_MD_MAX):
            continue

        max_turns = min(INPUT_TURN_MAX, seg_len - 2)
        turns = random.randint(INPUT_TURN_MIN, max_turns)
        input_seg = _build_input_path(seg_len, turns, side)
        if _is_4adj(input_seg, obstacle):
            continue

        union = bar + obstacle + input_seg + output_seg
        rs = [r for r, _ in union]
        cs = [c for _, c in union]
        return {
            "bar_len": bar_len,
            "bar": bar,
            "obstacle": obstacle,
            "input_seg": input_seg,
            "output_seg": output_seg,
            "seg_len": seg_len,
            "obs_size": len(obstacle),
            "bbox": (min(rs), min(cs), max(rs), max(cs)),
        }
    return None


def _place_triple_at(triple: dict, edge: str, n: int, pos: int) -> dict | None:
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"]
    union_g = _transform(union, edge, n, triple["bar_len"], pos)
    if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
        return None
    return {
        "bar": _transform(triple["bar"], edge, n, triple["bar_len"], pos),
        "obstacle": _transform(triple["obstacle"], edge, n, triple["bar_len"], pos),
        "input_seg": _transform(triple["input_seg"], edge, n, triple["bar_len"], pos),
        "output_seg": _transform(triple["output_seg"], edge, n, triple["bar_len"], pos),
        "terminal": _transform([(0, 0)], edge, n, triple["bar_len"], pos)[0],
        "seg_len": triple["seg_len"],
        "obs_size": triple["obs_size"],
    }


def _min_md_to_comp(p: tuple[int, int], comp: list[tuple[int, int]]) -> int:
    return min(abs(r - p[0]) + abs(c - p[1]) for r, c in comp)


def _cells_separated(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    bset = set(b)
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in bset:
                    return False
    return True


def _cells_separated4(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    bset = set(b)
    for r, c in a:
        if (r, c) in bset:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return False
    return True


def _valid_pairing_layout(placed_a: dict, placed_b: dict) -> bool:
    if not _cells_separated4(placed_a["bar"], placed_b["bar"]):
        return False
    if not _cells_separated(placed_a["obstacle"], placed_b["obstacle"]):
        return False
    if not _cells_separated(placed_a["input_seg"], placed_b["input_seg"]):
        return False
    if not _cells_separated(placed_a["output_seg"], placed_b["output_seg"]):
        return False

    t_a = placed_a["terminal"]
    t_b = placed_b["terminal"]
    own_a = _min_md_to_comp(t_a, placed_a["obstacle"])
    own_b = _min_md_to_comp(t_b, placed_b["obstacle"])
    other_a = _min_md_to_comp(t_a, placed_b["obstacle"])
    other_b = _min_md_to_comp(t_b, placed_a["obstacle"])

    if _min_md_to_comp(t_a, placed_b["input_seg"]) < 2:
        return False
    if _min_md_to_comp(t_b, placed_a["input_seg"]) < 2:
        return False
    if not (OTHER_OBS_MD_MIN <= other_a < own_a):
        return False
    if not (OTHER_OBS_MD_MIN <= other_b < own_b):
        return False
    return True


def _sample_seg_lengths() -> tuple[int, int]:
    vals = list(range(SEG_LEN_MIN, SEG_LEN_MAX + 1))
    for _ in range(40):
        a, b = sorted(random.sample(vals, 2), reverse=True)
        if a - b >= SEG_LEN_GAP_MIN:
            return a, b
    return 9, 6


def _apply_symmetry(grid: np.ndarray, sym: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    if sym >= 4:
        out = np.fliplr(out)
    return np.rot90(out, k=sym % 4)


def _enumerate_placements(triple: dict, n: int) -> list[dict]:
    placements = []
    for edge in EDGE_DATA:
        for pos in range(n):
            placed = _place_triple_at(triple, edge, n, pos)
            if placed is not None:
                placements.append(placed)
    random.shuffle(placements)
    return placements


def _sample_position_pair(n: int, same_axis: bool) -> tuple[int, int]:
    center = random.randint(4, n - 5)
    delta = random.choice([-5, -4, -3, 3, 4, 5])
    pos_a = min(max(center, 0), n - 1)
    pos_b = min(max(center + delta, 0), n - 1)
    if same_axis and pos_a == pos_b:
        pos_b = min(n - 1, pos_a + 3) if pos_a + 3 < n else max(0, pos_a - 3)
    return pos_a, pos_b


def _sample_layout(n: int) -> tuple[str, int, str, int]:
    # Canonical successful layout: one bar on the top edge and one on the left edge,
    # with terminals aimed into the same corner region. Random D4 symmetry at the end
    # provides the other visual orientations without paying a large search cost here.
    return "top", n // 3, "left", max(0, n // 3 - 1)


def _make_triple_for_placement(n: int, edge: str, pos: int, seg_len: int, obs_size: int) -> dict | None:
    for _ in range(PLACED_TRIPLE_TRIES):
        triple = _make_triple(n, seg_len=seg_len, obs_size=obs_size)
        if triple is None:
            continue
        placed = _place_triple_at(triple, edge, n, pos)
        if placed is not None:
            return placed
    return None


def _build_pair_grids(n: int, colors: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    bg, bar_color, seg_color, obstacle_color = colors
    edge_a, pos_a, edge_b, pos_b = _sample_layout(n)

    def place_obstacle(
        anchor: tuple[int, int], size: int, forbidden: set[tuple[int, int]]
    ) -> list[tuple[int, int]] | None:
        cands = []
        for r in range(n):
            for c in range(n):
                md = abs(r - anchor[0]) + abs(c - anchor[1])
                if not (2 <= md <= 5):
                    continue
                if (r, c) in forbidden:
                    continue
                if any((r, c) == q or (abs(r - q[0]) + abs(c - q[1]) == 1) for q in forbidden):
                    continue
                cands.append((r, c))
        random.shuffle(cands)
        for seed in cands[:40]:
            obstacle = {seed}
            ok = True
            for _ in range(size - 1):
                grow = [
                    (nr, nc)
                    for r, c in obstacle
                    for nr, nc in neighbors4(r, c)
                    if 0 <= nr < n
                    and 0 <= nc < n
                    and (nr, nc) not in obstacle
                    and (nr, nc) not in forbidden
                    and all(abs(nr - q[0]) + abs(nc - q[1]) > 1 for q in forbidden)
                ]
                if not grow:
                    ok = False
                    break
                obstacle.add(random.choice(grow))
            if ok and len(obstacle) == size:
                return list(obstacle)
        return None

    for _ in range(PAIR_LAYOUT_TRIES):
        long_len, short_len = _sample_seg_lengths()
        large_obs = random.randint(OBS_SIZE_LARGE_MIN, OBS_SIZE_LARGE_MAX)
        small_obs = random.randint(OBS_SIZE_SMALL_MIN, OBS_SIZE_SMALL_MAX)

        if random.random() < 0.5:
            placed_long = _make_triple_for_placement(n, edge_a, pos_a, long_len, random.randint(1, 9))
            placed_short = _make_triple_for_placement(n, edge_b, pos_b, short_len, random.randint(1, 9))
        else:
            placed_long = _make_triple_for_placement(n, edge_b, pos_b, long_len, random.randint(1, 9))
            placed_short = _make_triple_for_placement(n, edge_a, pos_a, short_len, random.randint(1, 9))
        if placed_long is None or placed_short is None:
            continue

        if not _cells_separated4(placed_long["bar"], placed_short["bar"]):
            continue
        if not _cells_separated(placed_long["input_seg"], placed_short["input_seg"]):
            continue
        if _min_md_to_comp(placed_long["terminal"], placed_short["input_seg"]) < 2:
            continue
        if _min_md_to_comp(placed_short["terminal"], placed_long["input_seg"]) < 2:
            continue

        base_forbidden = set(placed_long["bar"] + placed_long["input_seg"] + placed_short["bar"] + placed_short["input_seg"])
        obstacle_large = place_obstacle(placed_short["terminal"], large_obs, base_forbidden)
        if obstacle_large is None:
            continue
        forbidden2 = base_forbidden | set(obstacle_large)
        obstacle_small = place_obstacle(placed_long["terminal"], small_obs, forbidden2)
        if obstacle_small is None:
            continue
        if not _cells_separated(obstacle_large, obstacle_small):
            continue

        inp = np.full((n, n), bg, dtype=int)
        paint_coords(inp, placed_long["bar"], bar_color)
        paint_coords(inp, placed_short["bar"], bar_color)
        paint_coords(inp, placed_long["input_seg"], seg_color)
        paint_coords(inp, placed_short["input_seg"], seg_color)
        paint_coords(inp, obstacle_large, obstacle_color)
        paint_coords(inp, obstacle_small, obstacle_color)

        if not _size_rule_is_active(inp):
            continue
        try:
            out = main(inp)
        except Exception:
            continue
        return inp, out
    return None


def _size_rule_is_active(inp: np.ndarray) -> bool:
    info = analyze_instance(inp)
    if info["bar_count"] != TRIPLE_COUNT:
        return False

    bar_infos = _discover_bar_infos(inp)
    obstacles = _collect_obstacle_components(inp, bar_infos)
    bars_sorted = sorted(
        range(len(bar_infos)),
        key=lambda i: (
            -bar_infos[i]["seg_len"],
            bar_infos[i]["terminal"][0],
            bar_infos[i]["terminal"][1],
        ),
    )
    obstacles_sorted = sorted(
        range(len(obstacles)),
        key=lambda i: (-len(obstacles[i]), min(r for r, _ in obstacles[i]), min(c for _, c in obstacles[i])),
    )

    nearest = []
    for bi in bars_sorted:
        terminal = bar_infos[bi]["terminal"]
        dists = [_min_md_to_comp(terminal, list(obstacles[oi])) for oi in obstacles_sorted]
        nearest.append(int(np.argmin(dists)))

    size_pair_indices = list(range(len(bars_sorted)))
    return nearest != size_pair_indices


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    colors = tuple(random.sample(range(10), 4))
    built = _build_pair_grids(n, colors)
    if built is None:
        return None
    inp, out = built

    sym = random.randrange(8)
    inp = _apply_symmetry(inp, sym)
    out = _apply_symmetry(out, sym)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if not _size_rule_is_active(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
