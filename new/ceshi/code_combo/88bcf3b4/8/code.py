# Combo variant of 88bcf3b4 (index 8)
# combo_spec: D8/2+D11/0
# Name: 尺寸排序级联绕行
#
# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    - Keep the original bar + irregular segment + obstacle structure, but when multiple triples exist
#      the solver must first rank all attached segments by length and rank all static obstacle sprites by size.
#      The longest segment is assigned to the largest obstacle, the second-longest to the second-largest, etc.
#    - After that fixed size-based pairing is decided, the triples are not solved in parallel. They are processed
#      from the terminal closest to any border to the farthest one. Each solved rope is painted back immediately.
#      Later ropes still target their preassigned static obstacle, but they must route through the updated grid,
#      so earlier ropes behave like new blocking obstacles and can shorten or deflect later routes.
#    - In the canonical template used by the generator, the nearer terminal belongs to the shorter bar while the
#      longer rope is matched to the larger obstacle even though that obstacle is not its nearest one. The earlier
#      rope lands first near the shared corner region; the later rope then has to wrap around its own size-matched
#      obstacle while avoiding the already-painted rope, which creates a visible cascade.
# b) Complexity calibration:
#    - Fix the generator to exactly two triples. That is enough to make the three reasoning steps legible:
#      1) compare segment lengths, 2) compare obstacle sizes, 3) notice later routing depends on earlier output.
#    - Distinct segment lengths and distinct obstacle sizes are mandatory; otherwise size ranking is ambiguous.
#    - Use one verified geometric template and expand only by D4/reflection symmetries plus recoloring, which keeps
#      the rule visible without introducing extra latent degrees of freedom.
# c) Hidden assumptions audited:
#    - Original main() selected obstacles locally by distance; that assumption breaks, so obstacle selection is moved
#      to a global matching phase before any routing.
#    - Original multi-triple logic cleared and repainted all segments at the end; that assumes independence and breaks
#      under cascading, so the combo solver writes each solved rope back immediately.
#    - Triple discovery must still happen on the original input, not the evolving work grid, or earlier outputs could
#      be misread as part of later triples.
#    - Original code also asserted that a diagonal approach implies the obstacle sits exactly on the inner side of the
#      V apex. That is reliable when the target is the nearest local obstacle, but not after global size-based remapping,
#      so the combo solver drops that local apex-tightness assumption.
#    - Under the full combo, the old strict 3-phase shape regex (`FL* F* FR*` or mirror) can also fail: a rope may be
#      redirected toward a farther size-matched obstacle and then clipped by an earlier routed rope. The combo solver
#      therefore enforces the causal rule itself (matching + sequential blocking) instead of validating against the
#      original single-obstacle phase template.
#    - The phrase "previous output becomes a new obstacle" is implemented as a new blocker in walkability, not as a
#      new target candidate in the size ranking. Letting target candidates mutate mid-solve would make the ranking rule
#      unstable and visually underdetermined for humans.
#    - Some symmetric layouts can collapse the obstacle-skin boundary path to one cell. The original code asserted a
#      boundary length of at least two; this combo solver adds a local fallback direction so the same geometric rule
#      still resolves cleanly after symmetry transforms.
# d) Placement legality:
#    - The shipped template was searched offline from valid D8 samples and kept only if the combo solver, the
#      size-parallel reference, and the nearest-sequential reference all succeed.
#    - Rotations, mirror flips, and recoloring preserve non-overlap, in-bounds placement, component separation, and
#      the bar/segment/obstacle object identities.
# e) Visual observability:
#    - The size rule is visible because the longer rope bends around the larger sprite despite proximity suggesting the
#      opposite pairing.
#    - The cascade rule is visible because the later rope's final shape differs from the size-matched parallel result.
#      The closer terminal is also visually shallower, so the processing order is inferable from geometry alone.
# f) Cleanliness:
#    - Only four colors are used globally: background, one bar color, one segment color, one obstacle color.
#      Both triples stay separated in the input, and the conflict region appears only in the output routing zone.
# g) Inference support:
#    - No sorting depends on color ids or any hidden metadata. Humans can infer the combo from visible features only:
#      segment length, obstacle size, and edge-depth order.

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


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


def _edge_distance(p: tuple[int, int], h: int, w: int) -> int:
    r, c = p
    return min(r, c, h - 1 - r, w - 1 - c)


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


def _classify_steps(new_pts: list[tuple[int, int]], bar_comp: list[tuple[int, int]]) -> list[str]:
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
    return step_classes


def _discover_triples(g: np.ndarray) -> tuple[list[dict], list[set[tuple[int, int]]], int]:
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    triples = []

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
        triples.append(
            {
                "bar_col": bar_col,
                "bar_comp": list(bar_comp),
                "terminal": terminal,
                "seg_col": seg_col,
                "seg_start": seg_start,
                "seg_comp": list(seg_comp),
                "off": (seg_start[0] - terminal[0], seg_start[1] - terminal[1]),
                "seg_len": L,
                "edge_distance": _edge_distance(terminal, h, w),
            }
        )

    occ = g != bg
    for triple in triples:
        for p in triple["bar_comp"]:
            occ[p] = False
        for p in triple["seg_comp"]:
            occ[p] = False
    obstacle_components = [{(int(r), int(c)) for r, c in comp} for comp in components_mask(occ, connectivity=8)]
    return triples, obstacle_components, bg


def _size_pair_obstacles(triples: list[dict], obstacle_components: list[set[tuple[int, int]]]) -> dict[int, set[tuple[int, int]]]:
    assert len(obstacle_components) >= len(triples), "must find enough obstacle components"
    triples_sorted = sorted(
        range(len(triples)),
        key=lambda i: (
            -triples[i]["seg_len"],
            triples[i]["terminal"][0],
            triples[i]["terminal"][1],
            triples[i]["bar_col"],
            triples[i]["seg_col"],
            triples[i]["seg_start"][0],
            triples[i]["seg_start"][1],
        ),
    )
    obstacles_sorted = sorted(
        obstacle_components,
        key=lambda comp: (
            -len(comp),
            min(r for r, _ in comp),
            min(c for _, c in comp),
            max(r for r, _ in comp),
            max(c for _, c in comp),
        ),
    )
    return {triple_idx: obstacles_sorted[rank] for rank, triple_idx in enumerate(triples_sorted)}


def _nearest_pair_obstacles(triples: list[dict], obstacle_components: list[set[tuple[int, int]]]) -> dict[int, set[tuple[int, int]]]:
    out = {}
    for i, triple in enumerate(triples):
        best = None
        best_comp = None
        for comp in obstacle_components:
            d = min(abs(r - triple["terminal"][0]) + abs(c - triple["terminal"][1]) for r, c in comp)
            key = (d, min(r for r, _ in comp), min(c for _, c in comp), -len(comp))
            if best is None or key < best:
                best = key
                best_comp = comp
        assert best_comp is not None, "must find a nearest obstacle"
        out[i] = best_comp
    return out


def _extension_dir(
    boundary_path: list[tuple[int, int]],
    route: list[tuple[int, int]],
    terminal: tuple[int, int],
    seg_start: tuple[int, int],
) -> tuple[int, int]:
    if len(boundary_path) >= 2:
        return (
            boundary_path[-1][0] - boundary_path[-2][0],
            boundary_path[-1][1] - boundary_path[-2][1],
        )
    if len(route) >= 2:
        return route[-1][0] - route[-2][0], route[-1][1] - route[-2][1]
    return seg_start[0] - terminal[0], seg_start[1] - terminal[1]


def _route_triple(
    work: np.ndarray,
    triple: dict,
    obstacle_set: set[tuple[int, int]],
    bg: int,
) -> list[tuple[int, int]]:
    h, w = work.shape
    terminal = triple["terminal"]
    seg_start = triple["seg_start"]
    off_r, off_c = triple["off"]
    seg_len = triple["seg_len"]

    walkable = work == bg
    walkable[terminal] = True

    good = []
    for pr, pc in obstacle_set:
        er, ec = pr + off_r, pc + off_c
        if not (0 <= er < h and 0 <= ec < w):
            continue
        if (er, ec) in obstacle_set:
            continue
        if int(work[er, ec]) != bg:
            continue
        d = abs(pr - terminal[0]) + abs(pc - terminal[1])
        good.append((d, pr, pc))
    assert good, "no obstacle pixel has a valid translated endpoint outside obstacle"

    _, pr, pc = min(good)
    p_near = (pr, pc)
    walkable[p_near] = True
    path_t_to_obs = _greedy_manhattan_path(terminal, p_near, walkable) or _bfs_path(terminal, p_near, walkable)
    assert path_t_to_obs is not None, f"no path from terminal {terminal} to obstacle {p_near}"

    shifted = [(r + off_r, c + off_c) for r, c in path_t_to_obs]
    assert shifted and shifted[0] == seg_start, "shifted path must start at seg_start"
    assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted path must stay within grid"
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
        return int(work[r, c]) == bg

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

    route = list(shifted)
    if boundary_path and boundary_path[0] == entry:
        route.extend(boundary_path[1:])

    dr_ext, dc_ext = _extension_dir(boundary_path, route, terminal, seg_start)
    cur = route[-1]
    while len(route) < seg_len:
        nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(work[nr, nc]) != bg:
            break
        route.append((nr, nc))
        cur = (nr, nc)

    new_pts = route[:seg_len]
    assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

    return new_pts


def _solve_with_mode(grid: np.ndarray, sequential: bool, size_match: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    triples, obstacle_components, bg = _discover_triples(g)
    assert triples, "must find at least one valid triple"
    assert obstacle_components, "must find at least one static obstacle component"

    obstacle_map = _size_pair_obstacles(triples, obstacle_components) if size_match else _nearest_pair_obstacles(triples, obstacle_components)
    ordered = sorted(
        range(len(triples)),
        key=lambda i: (
            triples[i]["edge_distance"],
            triples[i]["terminal"][0],
            triples[i]["terminal"][1],
            triples[i]["bar_col"],
            triples[i]["seg_col"],
        ),
    )

    if sequential:
        cur = g.copy()
        for idx in ordered:
            triple = triples[idx]
            work = cur.copy()
            for r, c in triple["seg_comp"]:
                work[r, c] = bg
            new_pts = _route_triple(work, triple, obstacle_map[idx], bg)
            for r, c in triple["seg_comp"]:
                cur[r, c] = bg
            for r, c in new_pts:
                if int(cur[r, c]) == bg:
                    cur[r, c] = triple["seg_col"]
        return cur

    out = g.copy()
    moves = []
    for idx in ordered:
        triple = triples[idx]
        work = g.copy()
        for r, c in triple["seg_comp"]:
            work[r, c] = bg
        new_pts = _route_triple(work, triple, obstacle_map[idx], bg)
        moves.append((triple["seg_comp"], triple["seg_col"], new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg
    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route each irregular rope by combining global size matching with sequential cascading.

    Input structure:
    - The grid contains one or more triples: a straight bar that touches a border at one end, an irregular 1-cell-thick
      segment attached near the bar terminal, and a separate obstacle sprite elsewhere in the interior.

    Rule:
    - First discover every bar-attached segment and every static obstacle sprite in the original input.
    - Rank segments by length and obstacles by pixel count; pair longest-to-largest, second-longest-to-second-largest.
    - Then process the triples from the terminal nearest the border to the farthest. Each segment detaches, wraps around
      its size-matched obstacle, and is written back immediately, so earlier outputs block later routes.

    Key insight:
    - The transformation is not "go to the nearest obstacle" and not "solve all ropes independently".
      The correct output needs three linked inferences: compare segment lengths, compare obstacle sizes, then apply the
      reroutes in edge-depth order so previously rerouted ropes become new blockers.
    """
    return _solve_with_mode(grid, sequential=True, size_match=True)


def _main_parallel_size_reference(grid: np.ndarray) -> np.ndarray:
    return _solve_with_mode(grid, sequential=False, size_match=True)


def _main_nearest_sequential_reference(grid: np.ndarray) -> np.ndarray:
    return _solve_with_mode(grid, sequential=True, size_match=False)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    combo = main(grid)
    parallel = _main_parallel_size_reference(grid)
    nearest = _main_nearest_sequential_reference(grid)
    return {
        "cascade_diff": int(np.sum(combo != parallel)),
        "size_diff": int(np.sum(combo != nearest)),
    }


_BASE_TEMPLATE = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 1, 3, 1, 1, 5, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 1, 3, 1, 1, 5, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 1, 3, 1, 5, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 5, 5, 5, 5, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=int,
)


def _apply_symmetry(grid: np.ndarray, sym: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    if sym >= 4:
        out = np.fliplr(out)
    return np.rot90(out, k=sym % 4)


def _recolor_template(grid: np.ndarray) -> np.ndarray:
    old_bg, old_bar, old_seg, old_obs = 1, 3, 5, 6
    new_bg, new_bar, new_seg, new_obs = random.sample(range(10), 4)
    mapping = {old_bg: new_bg, old_bar: new_bar, old_seg: new_seg, old_obs: new_obs}
    return np.vectorize(lambda v: mapping[int(v)])(grid).astype(int)


def generate_input() -> np.ndarray:
    for _ in range(32):
        grid = _recolor_template(_apply_symmetry(_BASE_TEMPLATE, random.randrange(8)))
        metrics = _fusion_metrics(grid)
        if metrics["cascade_diff"] <= 0 or metrics["size_diff"] <= 0:
            continue
        return grid
    raise RuntimeError("failed to generate a valid combo sample")
