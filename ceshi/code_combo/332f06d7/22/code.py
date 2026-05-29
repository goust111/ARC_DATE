# Combo variant of 332f06d7 (index 22)
# combo_spec: D5/1+D10/2
# Name: 距离编码停止+迭代推进
#
# Design analysis
# ===============
# fused_rule:
#   - Preserve the original visual family: green background, a blue winding river, one
#     red destination square at the river end, and one same-sized mover square near an edge.
#   - The mover is no longer a solid black square. It is a full n x n square whose cells are
#     black(0) and grey(5). The number of grey cells is visible and countable.
#   - That grey count K is the per-round step length on the placement graph.
#   - The mover does not teleport. It repeatedly advances exactly K placement-steps along a
#     shortest path toward its goal. If the remaining distance is < K, that partial round is
#     not taken. Final travel distance = floor(goal_dist / K) * K.
#   - Goal selection remains compatible with the original family: if the red square placement
#     is reachable, it is the goal; otherwise use the farthest reachable placement. The
#     generator deliberately keeps the red square reachable so the visible stopping rule is
#     numeric rather than geometric.
#
# concrete_trace:
#   - Example: n=4 mover, grey_count K=3, shortest placement distance from mover to red = 8.
#   - Original 332f06d7: the mover would slide all 8 steps and cover red.
#   - Single D5/1: with budget 3 it would stop after only 3 steps.
#   - This combo: round 1 moves 3, round 2 moves 3, remaining 2 is too short for another
#     round, so the mover stops 6 steps from the start. The red target stays visible.
#   - If the distance were 9 instead, the same K=3 mover would reach the red target exactly
#     after three full rounds.
#
# complexity_calibration:
#   - n is restricted to 3 or 4. This keeps the mover large enough to show an internal count
#     clearly, while leaving enough room for a multi-turn river on a 12-20 grid.
#   - Grey counts are chosen from a small visible range (2..6, capped by n*n-1), so humans can
#     count them quickly across 2-4 examples.
#   - The generator enforces at least two full rounds. This avoids degeneration into either
#     "budget stop after K once" or "teleport to red directly".
#   - The rendered river is validated so the feasible n x n placements are exactly the intended
#     path cells. That removes route ambiguity and makes the iterative stepping observable.
#
# implicit_assumption_audit_for_main:
#   - Original solid-square detection for the mover is invalid because the mover is now black+grey.
#     Therefore the solver detects the red destination as a solid red square and detects the mover
#     on the union mask {black, grey}, requiring a fully filled square of the same size.
#   - Original stop logic assumed either "red reachable -> land on red" or "blocked -> farthest
#     reachable". That is invalid here because even with a fully wide corridor the mover may stop
#     early due to the encoded step size. The solver now reconstructs a shortest path and quantizes
#     the landing index by K.
#   - Path reconstruction assumes there is at least one shortest path to the chosen goal. BFS
#     guarantees that for any reachable goal. Ties are broken deterministically by lexicographic
#     predecessor order. The generator additionally makes the path effectively unique.
#   - The river color is still inferred as the most common non-background color excluding the mover
#     colors and red. The generator guarantees only one such river color.
#   - New assumption: the mover contains at least one grey and at least one black cell, so K>=1 and
#     the object is distinguishable from a solid grey or solid black block. The generator enforces
#     this and uses K>=2.
#
# placement_legality:
#   - The generator first chooses a self-avoiding placement path from a start near the edge but not
#     touching it to an edge-touching destination placement, then paints the river as the union of
#     those n x n windows.
#   - Start and destination do not overlap, remain in bounds, and lie on the river footprint.
#   - After painting, the generator verifies that the only feasible placements are exactly the path
#     cells; this prevents accidental extra landings created by dilation.
#
# observability_and_cleanliness:
#   - Because the corridor stays fully wide and the destination is reachable, a failure to reach red
#     cannot be explained by chokepoints. The only changing cue is the count of grey cells.
#   - The mover pattern translates intact; no colors are remapped, and the same grey count remains
#     visible before and after the move.
#   - Only five colors are used: green background, blue river, black and grey inside the mover, and
#     red destination. There are no hidden numeric ordering tricks.
#
# generator_non_degeneracy_target:
#   - Generated samples should usually satisfy:
#       goal_dist >= 2*K
#       landing_dist = floor(goal_dist / K) * K > K
#     so the combo differs from both:
#       original: landing_dist = goal_dist
#       D5/1-only budget stop: landing_dist = K

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label


_BG, _RIVER, _MOVER, _DEST, _GREY = 3, 1, 0, 2, 5
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _bbox_of_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_red_square(g: np.ndarray) -> dict:
    mask = g == _DEST
    labeled, num = scipy_label(mask.astype(int))
    squares: list[dict] = []
    for idx in range(1, int(num) + 1):
        coords = np.argwhere(labeled == int(idx))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_of_coords(coords)
        n = int(r1 - r0 + 1)
        if n != int(c1 - c0 + 1):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if bool((patch == _DEST).all()):
            squares.append({"r0": r0, "c0": c0, "n": n})
    assert squares
    squares.sort(key=lambda s: (-int(s["n"]), int(s["r0"]), int(s["c0"])))
    return squares[0]


def _find_mover_square(g: np.ndarray, n: int) -> dict:
    mover_mask = (g == _MOVER) | (g == _GREY)
    labeled, num = scipy_label(mover_mask.astype(int))
    cands: list[dict] = []
    for idx in range(1, int(num) + 1):
        coords = np.argwhere(labeled == int(idx))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_of_coords(coords)
        if int(r1 - r0 + 1) != int(n) or int(c1 - c0 + 1) != int(n):
            continue
        patch_mask = mover_mask[r0 : r1 + 1, c0 : c1 + 1]
        if not bool(patch_mask.all()):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        grey_count = int((patch == _GREY).sum())
        black_count = int((patch == _MOVER).sum())
        if grey_count <= 0 or black_count <= 0:
            continue
        cands.append(
            {
                "r0": r0,
                "c0": c0,
                "n": n,
                "grey_count": grey_count,
                "patch": patch.copy(),
            }
        )
    assert cands
    cands.sort(key=lambda s: (int(s["r0"]), int(s["c0"])))
    return cands[0]


def _feasible_windows(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _bfs(feasible: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h2, w2 = feasible.shape
    sr, sc = start
    dist = -np.ones((h2, w2), dtype=int)
    assert bool(feasible[int(sr), int(sc)])
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((rr, cc))
    return dist


def _select_goal(dist: np.ndarray, feasible: np.ndarray, dest: tuple[int, int], start: tuple[int, int]) -> tuple[int, int]:
    dr, dc = dest
    h2, w2 = feasible.shape
    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[int(dr), int(dc)]) and int(dist[int(dr), int(dc)]) >= 0:
        return int(dr), int(dc)

    best = -1
    tr, tc = map(int, start)
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return tr, tc


def _reconstruct_shortest_path(dist: np.ndarray, goal: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, goal)
    assert int(dist[r, c]) >= 0
    path = [(r, c)]
    while int(dist[r, c]) > 0:
        prevs = []
        for dr, dc in _DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < dist.shape[0] and 0 <= cc < dist.shape[1] and int(dist[rr, cc]) == int(dist[r, c]) - 1:
                prevs.append((rr, cc))
        assert prevs
        prevs.sort()
        r, c = prevs[0]
        path.append((r, c))
    path.reverse()
    return path


def _landing_after_iterative_steps(dist: np.ndarray, goal: tuple[int, int], step_len: int) -> tuple[int, int]:
    path = _reconstruct_shortest_path(dist, goal)
    goal_dist = len(path) - 1
    landing_idx = (int(goal_dist) // int(step_len)) * int(step_len)
    return path[int(landing_idx)]


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    dest = _find_red_square(g)
    mover = _find_mover_square(g, int(dest["n"]))

    n = int(dest["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    step_len = int(mover["grey_count"])
    mover_patch = np.array(mover["patch"], dtype=int)

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) in (int(bg), _MOVER, _GREY, _DEST):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == int(river)) | (g == _MOVER) | (g == _GREY) | (g == _DEST)
    feasible = _feasible_windows(trav, n)
    h2, w2 = feasible.shape
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2
    assert bool(feasible[int(sr), int(sc)])

    dist = _bfs(feasible, (sr, sc))
    goal = _select_goal(dist, feasible, (dr, dc), (sr, sc))
    tr, tc = _landing_after_iterative_steps(dist, goal, step_len)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = mover_patch

    info = {
        "n": n,
        "start": (sr, sc),
        "dest": (dr, dc),
        "goal": goal,
        "goal_dist": int(dist[int(goal[0]), int(goal[1])]),
        "step_len": step_len,
        "landing": (tr, tc),
        "landing_dist": (int(dist[int(goal[0]), int(goal[1])]) // step_len) * step_len,
        "dest_reachable": bool(goal == (dr, dc)),
        "river": int(river),
        "mover_patch": mover_patch,
        "bg": bg,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    out, _info = _solve_core(grid)
    return out


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (int(path[i][0] - path[i - 1][0]), int(path[i][1] - path[i - 1][1]))
        d2 = (int(path[i + 1][0] - path[i][0]), int(path[i + 1][1] - path[i][1]))
        if d1 != d2:
            turns += 1
    return turns


def _make_mover_patch(n: int, grey_count: int) -> np.ndarray:
    patch = np.full((n, n), _MOVER, dtype=int)
    flat = patch.reshape(-1)
    flat[:grey_count] = _GREY
    return patch


def _render_channel(grid: np.ndarray, path: list[tuple[int, int]], n: int) -> None:
    for r, c in path:
        grid[int(r) : int(r + n), int(c) : int(c + n)] = _RIVER


def _candidate_positions(h2: int, w2: int, *, edge: bool) -> list[tuple[int, int]]:
    out = []
    if edge:
        for c in range(1, w2 - 1):
            out.append((0, c))
            out.append((h2 - 1, c))
        for r in range(1, h2 - 1):
            out.append((r, 0))
            out.append((r, w2 - 1))
    else:
        if h2 >= 3:
            for c in range(1, w2 - 1):
                out.append((1, c))
                out.append((h2 - 2, c))
        if w2 >= 3:
            for r in range(2, h2 - 2):
                out.append((r, 1))
                out.append((r, w2 - 2))
    uniq = sorted(set(out))
    random.shuffle(uniq)
    return uniq


def _can_step(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    visited: set[tuple[int, int]],
    h2: int,
    w2: int,
    end: tuple[int, int],
    remaining_after: int,
) -> bool:
    nr, nc = nxt
    if not (0 <= nr < h2 and 0 <= nc < w2):
        return False
    if nxt in visited:
        return False
    for dr, dc in _DIR4:
        adj = (int(nr + dr), int(nc + dc))
        if adj in visited and adj != cur:
            return False
    dist = abs(int(nr - end[0])) + abs(int(nc - end[1]))
    if dist > remaining_after:
        return False
    if (remaining_after - dist) % 2 != 0:
        return False
    return True


def _find_path(h2: int, w2: int, start: tuple[int, int], end: tuple[int, int], total_steps: int) -> list[tuple[int, int]] | None:
    path = [start]
    visited = {start}

    def dfs(cur: tuple[int, int], remaining: int) -> bool:
        if remaining == 0:
            return cur == end

        neighbors = []
        for dr, dc in _DIR4:
            nxt = (int(cur[0] + dr), int(cur[1] + dc))
            if _can_step(nxt, cur, visited, h2, w2, end, remaining - 1):
                neighbors.append(nxt)
        random.shuffle(neighbors)
        neighbors.sort(key=lambda p: abs(int(p[0] - end[0])) + abs(int(p[1] - end[1])))

        for nxt in neighbors:
            visited.add(nxt)
            path.append(nxt)
            if dfs(nxt, remaining - 1):
                return True
            path.pop()
            visited.remove(nxt)
        return False

    start_dist = abs(int(start[0] - end[0])) + abs(int(start[1] - end[1]))
    if start_dist > total_steps or (total_steps - start_dist) % 2 != 0:
        return None
    if dfs(start, total_steps):
        return list(path)
    return None


def _analyze_path_render(path: list[tuple[int, int]], H: int, W: int, n: int) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    grid = np.full((H, W), _BG, dtype=int)
    _render_channel(grid, path, n)
    feasible = _feasible_windows(grid == _RIVER, n)
    sr, sc = path[0]
    dr, dc = path[-1]
    if not bool(feasible[int(sr), int(sc)]) or not bool(feasible[int(dr), int(dc)]):
        return False, None, None
    dist = _bfs(feasible, (sr, sc))
    dist_to_dest = _bfs(feasible, (dr, dc))
    total = len(path) - 1
    if int(dist[int(dr), int(dc)]) != total:
        return False, None, None
    if int((dist >= 0).sum()) != len(path):
        return False, None, None
    for step, (r, c) in enumerate(path):
        if int(dist[int(r), int(c)]) != step or int(dist_to_dest[int(r), int(c)]) != total - step:
            return False, None, None
        if int(((dist == step) & (dist_to_dest == total - step)).sum()) != 1:
            return False, None, None
    return True, dist, dist_to_dest


def _choose_step_and_distance(n: int, scenario: str) -> tuple[int, int, int] | None:
    max_step = min(6, n * n - 1)
    step_len = random.randint(2, max_step)
    max_rounds = min(3, 14 // step_len)
    if max_rounds < 2:
        return None
    rounds = random.randint(2, max_rounds)
    if scenario == "exact":
        total_steps = rounds * step_len
    else:
        if step_len <= 2:
            remainder = 1
        else:
            remainder = random.randint(1, step_len - 1)
        total_steps = rounds * step_len + remainder
    if total_steps > 14:
        return None
    return step_len, rounds, total_steps


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choices(["partial", "exact"], weights=[4, 2], k=1)[0]
    n = random.choice([3, 4])

    choice = _choose_step_and_distance(n, scenario)
    if choice is None:
        return None
    step_len, rounds, total_steps = choice

    min_grid = max(12, n + max(4, total_steps // 2) + 4)
    if min_grid > 20:
        return None
    H = W = random.randint(min_grid, 20)
    h2, w2 = H - n + 1, W - n + 1
    if h2 < 5 or w2 < 5:
        return None

    start_cands = _candidate_positions(h2, w2, edge=False)
    end_cands = _candidate_positions(h2, w2, edge=True)
    if not start_cands or not end_cands:
        return None

    path = None
    for _ in range(150):
        start = start_cands[random.randrange(len(start_cands))]
        end = end_cands[random.randrange(len(end_cands))]
        if start == end:
            continue
        if abs(int(start[0] - end[0])) + abs(int(start[1] - end[1])) > total_steps:
            continue
        maybe = _find_path(h2, w2, start, end, total_steps)
        if maybe is None:
            continue
        if _turn_count(maybe) < 3:
            continue
        path = maybe
        break
    if path is None:
        return None

    ok, dist, dist_to_dest = _analyze_path_render(path, H, W, n)
    if not ok or dist is None or dist_to_dest is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    _render_channel(grid, path, n)

    river_ratio = float((grid == _RIVER).sum()) / float(H * W)
    if not (0.18 <= river_ratio <= 0.62):
        return None

    mover_r, mover_c = path[0]
    dest_r, dest_c = path[-1]
    mover_patch = _make_mover_patch(n, step_len)
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = mover_patch
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    landing_idx = (total_steps // step_len) * step_len
    expected_r, expected_c = path[int(landing_idx)]
    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    output[expected_r : expected_r + n, expected_c : expected_c + n] = mover_patch

    if total_steps < 2 * step_len:
        return None
    if landing_idx <= step_len:
        return None
    if scenario == "partial" and total_steps % step_len == 0:
        return None
    if scenario == "exact" and total_steps % step_len != 0:
        return None
    if int(dist[int(dest_r), int(dest_c)]) != total_steps:
        return None
    if int(dist_to_dest[int(mover_r), int(mover_c)]) != total_steps:
        return None

    try:
        solved, info = _solve_core(grid)
    except Exception:
        return None
    if not bool(info["dest_reachable"]):
        return None
    if int(info["step_len"]) != step_len or int(info["goal_dist"]) != total_steps:
        return None
    if tuple(info["landing"]) != (int(expected_r), int(expected_c)):
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
