# Combo variant of 332f06d7 (index 7)
# combo_spec: D3/2+D5/1
# Name: 渐变色轨迹+距离编码停止
#
# Design analysis for the fused variant
# a) Concrete behavior and trace
#    - Input keeps the original river scene: green background, one blue winding river, one red
#      destination square touching an edge, and one mover square of the same size.
#    - D5/1 changes the mover body: it is now a full n x n square made of black(0) and grey(5),
#      where grey-count = the maximum number of placement-steps the mover may take.
#    - D3/2 changes the output residue: the mover no longer restores vacated cells to blue. Instead,
#      every vacated placement on the actual traversed prefix gets recolored by a fixed 3-band
#      gradient: near-start=4, middle=8, near-stop=6. The final mover keeps its original black/grey
#      pattern.
#    - Example in placement coordinates:
#      path = [(2,2),(2,3),(3,3),(4,3),(4,4)], total geometric distance=4, grey-count=3.
#      Original 332f06d7 would slide to the red square if the river stayed wide enough.
#      This fused variant moves only 3 steps, stops at (4,3), paints trail colors on
#      (2,2)->4, (2,3)->8, (3,3)->6, and leaves the mover patch unchanged at (4,3).
# b) Complexity calibration
#    - n is restricted to 3 or 4 so the grey-count signal is visually meaningful; n=1/2 would make
#      the budget too small and ambiguous.
#    - Trail colors are limited to exactly 3 fixed bands. This keeps the gradient inferable from
#      2-4 examples instead of introducing arbitrary per-step colors.
#    - Generator enforces actual movement >= 3 so all three trail colors appear in every sample.
# c) Implicit assumption audit against the original main()
#    - Original square detection assumed the mover was monochrome. That breaks once grey pixels are
#      embedded, so solver now detects the red square separately and detects the mover as a filled
#      square component on the {black, grey} mask.
#    - Original stop logic used chokepoints and "farthest reachable placement". That is invalid here:
#      the river is always wide enough, and stop position is budget-limited. Solver therefore computes
#      the shortest placement path toward the red square and clips it by grey-count.
#    - Original "restore start to river blue" would erase the fused trail. Rendering must now paint
#      the traversed prefix with gradient colors before placing the final mover patch.
#    - A new assumption is introduced: the geometric shortest path to the destination should be
#      unique so the trail order is unambiguous. The generator guarantees this by building a
#      self-avoiding placement path whose dilated river exposes exactly those placements.
# d) Placement legality
#    - Generator works in placement space first, then paints the river as the union of n x n windows
#      along that path. Start is near an edge but not touching; destination touches an edge.
#    - Path cells are non-overlapping, in-bounds, and forbidden from non-consecutive 4-adjacency,
#      preventing accidental branches or alternate shortest routes after dilation.
# e) Visual observability
#    - Across examples, humans can see two independent cues: grey-count inside the mover predicts how
#      far it travels, and the path it actually used is highlighted by the ordered trail 4->8->6.
#    - In budget-limited samples the red target remains geometrically reachable, so width can no
#      longer explain stopping; the only visible stopper is the grey-count.
# f) Visual cleanliness
#    - Only five input colors appear: green bg, blue river, red target, black mover cells, grey
#      budget cells. Output adds exactly three trail colors. Trail is confined to occupied mover
#      footprints, so the river itself remains readable.
# g) Color-change inferability
#    - Trail colors always play the same structural role (start/middle/end of actual traversal).
#      Generator rejects short moves that would hide one of the bands.
# h) New main() assumptions guaranteed by generate_input()
#    - Exactly one solid red square exists.
#    - Exactly one full black/grey square of the same size exists, with at least one black and one
#      grey cell.
#    - The red square is geometrically reachable through the river, and the shortest placement path
#      is unique.
#    - The actual move length is at least 3, so all trail colors appear.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label


_BG, _RIVER, _MOVER, _DEST, _GREY = 3, 1, 0, 2, 5
_TRAIL_COLORS = (4, 8, 6)
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox_of_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_red_square(g: np.ndarray) -> dict:
    mask = g == _DEST
    lab, num = scipy_label(mask.astype(int))
    cands: list[dict] = []
    for idx in range(1, int(num) + 1):
        coords = np.argwhere(lab == idx)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_of_coords(coords)
        n = int(r1 - r0 + 1)
        if n != int(c1 - c0 + 1):
            continue
        if not bool((g[r0 : r1 + 1, c0 : c1 + 1] == _DEST).all()):
            continue
        cands.append({"r0": r0, "c0": c0, "n": n})
    assert cands
    cands.sort(key=lambda s: (-int(s["n"]), int(s["r0"]), int(s["c0"])))
    return cands[0]


def _find_mover_square(g: np.ndarray, n: int) -> dict:
    mover_mask = (g == _MOVER) | (g == _GREY)
    lab, num = scipy_label(mover_mask.astype(int))
    cands: list[dict] = []
    for idx in range(1, int(num) + 1):
        coords = np.argwhere(lab == idx)
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
        cands.append({"r0": r0, "c0": c0, "n": n, "grey_count": grey_count, "patch": patch.copy()})
    assert cands
    cands.sort(key=lambda s: (int(s["r0"]), int(s["c0"])))
    return cands[0]


def _build_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = int(H - n + 1), int(W - n + 1)
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _bfs_with_parent(
    feasible: np.ndarray, start: tuple[int, int]
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int] | None]]:
    h2, w2 = feasible.shape
    sr, sc = start
    assert bool(feasible[int(sr), int(sc)])
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(int(sr), int(sc)): None}
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                parent[(rr, cc)] = (int(r), int(c))
                q.append((rr, cc))
    return dist, parent


def _choose_target(
    dist_from_start: np.ndarray,
    dist_to_dest: np.ndarray,
    dest_pos: tuple[int, int],
    budget: int,
) -> tuple[int, int]:
    dr, dc = dest_pos
    total = int(dist_from_start[int(dr), int(dc)])
    assert total >= 0
    steps = min(int(budget), total)
    cands = np.argwhere(
        (dist_from_start == steps)
        & (dist_to_dest >= 0)
        & (dist_from_start + dist_to_dest == total)
    )
    if cands.size:
        return min(
            ((int(r), int(c)) for r, c in cands),
            key=lambda p: (
                int(dist_to_dest[p[0], p[1]]),
                abs(int(p[0] - dr)) + abs(int(p[1] - dc)),
                int(p[0]),
                int(p[1]),
            ),
        )

    best = None
    for r, c in np.argwhere((dist_from_start >= 0) & (dist_from_start <= budget)):
        cand = (int(r), int(c))
        score = (
            int(dist_from_start[cand[0], cand[1]]),
            -int(dist_to_dest[cand[0], cand[1]] >= 0),
            int(cand[0]),
            int(cand[1]),
        )
        if best is None or score > best[0]:
            best = (score, cand)
    assert best is not None
    return best[1]


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None], target: tuple[int, int]
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _trail_color(step_idx: int, trail_steps: int) -> int:
    if trail_steps <= 1:
        return int(_TRAIL_COLORS[0])
    band = (step_idx * (len(_TRAIL_COLORS) - 1)) // (trail_steps - 1)
    return int(_TRAIL_COLORS[band])


def _solve_info(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(
        Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0]
    )

    dest = _find_red_square(g)
    mover = _find_mover_square(g, int(dest["n"]))
    n = int(dest["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) in (int(bg), _MOVER, _GREY, _DEST):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == int(river)) | (g == _MOVER) | (g == _GREY) | (g == _DEST)
    feasible = _build_feasible(trav, n)
    h2, w2 = feasible.shape
    assert 0 <= sr < h2 and 0 <= sc < w2 and 0 <= dr < h2 and 0 <= dc < w2
    assert bool(feasible[sr, sc]) and bool(feasible[dr, dc])

    dist_from_start, parent = _bfs_with_parent(feasible, (sr, sc))
    dist_to_dest, _ = _bfs_with_parent(feasible, (dr, dc))
    assert int(dist_from_start[dr, dc]) >= 0

    target = _choose_target(dist_from_start, dist_to_dest, (dr, dc), int(mover["grey_count"]))
    assert target in parent
    path = _reconstruct_path(parent, target)

    return {
        "grid": g,
        "shape": (H, W),
        "bg": bg,
        "river": int(river),
        "n": n,
        "start": (sr, sc),
        "dest": (dr, dc),
        "budget": int(mover["grey_count"]),
        "mover_patch": np.array(mover["patch"], dtype=int),
        "target": target,
        "path": path,
        "dist_from_start": dist_from_start,
        "dist_to_dest": dist_to_dest,
    }


def _render_output_from_info(info: dict) -> np.ndarray:
    out = np.array(info["grid"], dtype=int).copy()
    n = int(info["n"])
    path = list(info["path"])
    trail_steps = max(0, len(path) - 1)
    for step_idx, (r, c) in enumerate(path[:-1]):
        out[int(r) : int(r + n), int(c) : int(c + n)] = _trail_color(step_idx, trail_steps)
    tr, tc = info["target"]
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = np.array(info["mover_patch"], dtype=int)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - The red destination is always geometrically reachable through the blue river.
    - The mover is a black/grey n x n square; grey-count is the movement budget in placement steps.
    - The mover travels along the unique shortest placement path toward the red square and stops after
      min(distance-to-dest, grey-count) steps.
    - Every vacated placement on the actual traversed prefix is recolored with a 3-band gradient
      4 -> 8 -> 6; the final mover keeps its original black/grey pattern.
    """
    info = _solve_info(grid)
    return _render_output_from_info(info)


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


def _find_path(
    h2: int, w2: int, start: tuple[int, int], end: tuple[int, int], total_steps: int
) -> list[tuple[int, int]] | None:
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


def _bfs(feasible: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist, _parent = _bfs_with_parent(feasible, start)
    return dist


def _analyze_path_render(
    path: list[tuple[int, int]], H: int, W: int, n: int
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    grid = np.full((H, W), _BG, dtype=int)
    _render_channel(grid, path, n)
    trav = grid == _RIVER
    feasible = _build_feasible(trav, n)
    sr, sc = path[0]
    dr, dc = path[-1]
    if not bool(feasible[sr, sc]) or not bool(feasible[dr, dc]):
        return False, None, None
    dist = _bfs(feasible, (sr, sc))
    dist_to_dest = _bfs(feasible, (dr, dc))
    total = len(path) - 1
    if int(dist[dr, dc]) != total:
        return False, None, None
    if int((dist >= 0).sum()) != len(path):
        return False, None, None
    for step, (r, c) in enumerate(path):
        if int(dist[r, c]) != step or int(dist_to_dest[r, c]) != total - step:
            return False, None, None
        if int(((dist == step) & (dist_to_dest == total - step)).sum()) != 1:
            return False, None, None
    return True, dist, dist_to_dest


def _trail_stats(move_steps: int) -> dict:
    used = {_trail_color(step_idx, move_steps) for step_idx in range(move_steps)}
    return {"move_steps": int(move_steps), "used_colors": used}


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    max_budget = n * n - 1
    reached = random.random() < 0.5

    if reached:
        total_steps = random.randint(4, max_budget)
        grey_count = random.randint(total_steps, max_budget)
    else:
        total_steps = random.randint(6, min(14, max_budget + 4))
        grey_count = random.randint(3, min(max_budget, total_steps - 1))

    move_steps = min(grey_count, total_steps)
    if move_steps < len(_TRAIL_COLORS):
        return None

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
    for _ in range(160):
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
    mover_patch = _make_mover_patch(n, grey_count)
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = mover_patch
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    expected_path = path[: move_steps + 1]
    output = grid.copy()
    for step_idx, (r, c) in enumerate(expected_path[:-1]):
        output[r : r + n, c : c + n] = _trail_color(step_idx, move_steps)
    final_r, final_c = expected_path[-1]
    output[final_r : final_r + n, final_c : final_c + n] = mover_patch

    if reached != (grey_count >= total_steps):
        return None
    if int(dist[dest_r, dest_c]) != total_steps:
        return None
    if int(dist_to_dest[mover_r, mover_c]) != total_steps:
        return None

    trail_info = _trail_stats(move_steps)
    if len(trail_info["used_colors"]) < len(_TRAIL_COLORS):
        return None
    if not trail_info["used_colors"].issubset(set(map(int, np.unique(output)))):
        return None
    if set(map(int, np.unique(grid))) - {_BG, _RIVER, _MOVER, _DEST, _GREY}:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
