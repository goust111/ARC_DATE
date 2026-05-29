# Combo variant of 332f06d7 (index 3)
# combo_spec: D1/3+D12/0
# Name: 最短路径+分支选择

"""
Design analysis for the combo variant
=====================================

This fusion keeps the original objects of 332f06d7:
- green background (3)
- blue traversable river (1)
- black mover square (0)
- red destination square (2)

The change is structural and visible:
- the input river is no longer a single corridor
- the river contains a real branch choice
- there is a unique shortest route from mover to destination
- there is also a longer alternative route to the destination
- there is at least one side branch that is a dead end
- the output keeps only the chosen shortest route and moves the black square to the red square

a) Concreteization and behavior trace
-------------------------------------
- Original task: feasible top-left placements of the n x n mover form an almost-linear chain.
  The solver's BFS mostly answers "can I reach the red square?".
- Combo input: placements form a branched graph. From the start junction, one branch is a dead
  end, while two different routes eventually lead to the red target; one is strictly shorter.
- Combo output: the black square still ends on the red square, but the blue river is pruned to
  exactly the shortest placement path. The longer route and dead-end branch disappear to green.
- Small trace:
  1. Start at black square.
  2. BFS reaches a junction. One side branch stops; another continues directly; another loops
     around and reaches the destination later.
  3. Parent pointers reconstruct only the unique shortest route to the red square.
  4. Output paints only that route blue and overwrites the red square with black.
- Difference from original: in the original task the final black position is enough; here the
  chosen route itself becomes part of the output, so branch choice and shortest-path selection
  are both observable.

b) Complexity calibration
-------------------------
- Only four colors are used, same as the source task.
- The graph has one main route family, one longer detour, and one dead-end branch. This is
  enough to express the fusion without turning the grid into a maze.
- The generator enforces a unique shortest path. Otherwise the output would depend on invisible
  BFS tie-breaking rather than a visually inferable rule.
- I keep the task in reachable-only mode. Introducing stuck cases on top of route pruning would
  mix two distinct explanations in too few examples and make the fused rule less inferable.

c) Implicit-assumption audit of the original `main()`
----------------------------------------------------
- Solid-square detection still works only if the river is not split into isolated solid squares.
  The generator renders one connected irregular blue region and validates the final graph.
- The original code sorted same-sized squares by color. I keep the same fixed mover/destination
  colors, so this is only an implementation detail, not a new hidden semantic.
- The original "reachable -> move to destination, else farthest reachable" is insufficient for
  the combo because route choice must be visible. The solver must therefore keep BFS parents and
  reconstruct the path.
- The original code assumed a single corridor, so it never had to reason about multiple routes.
  The combo solver explicitly validates feasibility on the full placement graph, then uses BFS on
  that graph.

d) Placement legality
---------------------
- The generator works in placement space first. It defines a set of legal top-left positions,
  then renders the pixel river as the union of those n x n windows.
- Start and destination are chosen directly from placement coordinates, so they are guaranteed to
  be legal mover placements before symmetry.
- After rendering and symmetry, the generator recomputes the feasible-placement mask from pixels.
  Any accidental extra placements or tie-inducing shortcuts cause rejection.

e) Visual observability
-----------------------
- A human sees a branched blue river in the input and a single surviving blue corridor in the
  output.
- The dead-end branch explains that not every branch is relevant.
- The longer loop-to-goal branch explains that even among goal-reaching branches, the selected
  one is the shortest.
- No invisible ordering of objects or colors is introduced by the fusion rule itself.

f) Visual cleanliness
---------------------
- The layout is built from one rectilinear backbone, one loop-like detour, and one vertical side
  branch. This keeps the graph readable.
- Branches are separated by at least two placement rows/cols before rendering, so thick n x n
  rivers do not blur together.

g) Change inferability
----------------------
- Blue means traversable river in both input and output.
- Green means removed/unselected area.
- The structural cue is enough to infer the change: the output is the shortest route through the
  branched network, not merely the final moved black square.

9) New assumptions introduced here and how the generator enforces them
----------------------------------------------------------------------
- Assumption: destination is reachable. Enforced by BFS validation.
- Assumption: the shortest route is unique. Enforced by shortest-path counting.
- Assumption: there exists a longer alternative route to the same destination. Enforced by
  checking that at least one shortest-path edge can be removed while a route to the goal remains.
- Assumption: there exists a visible dead-end side branch. Enforced by requiring an off-path leaf.
- Assumption: route choice is visible on the chosen path itself. Enforced by requiring a branch
  node of degree >= 3 on the shortest path.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random
import sys

import numpy as np

for _p in (
    "/home/lixinyu/snap/new/arc2_solver_select",
    "/home/lixinyu/snap/new/solvers",
):
    if _p not in sys.path:
        sys.path.append(_p)

from arc_utils import bbox_h, bbox_w, find_components


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _detect_squares(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == int(col)).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})

    assert len(squares) >= 2
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2
    cand.sort(key=lambda sq: int(sq["col"]))
    return cand[0], cand[1]


def _detect_river(g: np.ndarray, bg: int, mover_col: int, dest_col: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in (int(mover_col), int(dest_col)):
            continue
        return int(col)
    raise AssertionError("No river color found")


def _compute_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _bfs_parents(
    feasible: np.ndarray, start: tuple[int, int]
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int] | None]]:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    dist[start[0], start[1]] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                parent[(rr, cc)] = (r, c)
                q.append((rr, cc))
    return dist, parent


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _path_pixels(path: list[tuple[int, int]], shape: tuple[int, int], n: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in path:
        mask[r : r + n, c : c + n] = True
    return mask


def _select_fallback(dist: np.ndarray, goal: tuple[int, int]) -> tuple[int, int]:
    best_key = None
    best = None
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if dist[r, c] < 0:
                continue
            key = (abs(r - goal[0]) + abs(c - goal[1]), -int(dist[r, c]), r, c)
            if best_key is None or key < best_key:
                best_key = key
                best = (r, c)
    assert best is not None
    return best


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _detect_squares(g, bg)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    river = _detect_river(g, bg, int(mover["col"]), int(dest["col"]))

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _compute_feasible(trav, n)
    assert feasible[sr, sc]

    dist, parent = _bfs_parents(feasible, (sr, sc))
    reached = 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and feasible[dr, dc] and dist[dr, dc] >= 0
    tr, tc = (dr, dc) if reached else _select_fallback(dist, (dr, dc))
    path = _reconstruct_path(parent, (tr, tc))

    out = np.full((H, W), int(bg), dtype=int)
    out[_path_pixels(path, (H, W), n)] = int(river)
    if reached:
        out[dr : dr + n, dc : dc + n] = int(mover["col"])
    else:
        out[dr : dr + n, dc : dc + n] = int(dest["col"])
        out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(np.array(grid, dtype=int))


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    out: list[tuple[int, int]] = []
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        for c in range(c1, c2 + step, step):
            out.append((r1, c))
    else:
        step = 1 if r2 >= r1 else -1
        for r in range(r1, r2 + step, step):
            out.append((r, c1))
    return out


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _line_cells(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _render_from_placements(placements: set[tuple[int, int]], G: int, n: int) -> np.ndarray:
    grid = np.full((G, G), _BG, dtype=int)
    for r, c in placements:
        grid[r : r + n, c : c + n] = _RIVER
    return grid


def _shortest_path_count(
    feasible: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> tuple[int, np.ndarray, dict[tuple[int, int], tuple[int, int] | None]]:
    dist, parent = _bfs_parents(feasible, start)
    if dist[goal] < 0:
        return 0, dist, parent
    counts = np.zeros_like(dist, dtype=np.int64)
    counts[start] = 1
    maxd = int(dist[goal])
    for d in range(maxd):
        for r, c in np.argwhere(dist == d):
            cur_count = int(counts[r, c])
            if cur_count == 0:
                continue
            for dr, dc in _DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < dist.shape[0] and 0 <= cc < dist.shape[1] and feasible[rr, cc] and dist[rr, cc] == d + 1:
                    counts[rr, cc] += cur_count
    return int(counts[goal]), dist, parent


def _reachable_with_banned_edge(
    feasible: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    banned: frozenset[tuple[int, int]],
) -> bool:
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        r, c = cur
        for dr, dc in _DIR4:
            nxt = (r + dr, c + dc)
            rr, cc = nxt
            if not (0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and feasible[rr, cc]):
                continue
            if frozenset((cur, nxt)) == banned:
                continue
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def _has_alternative_route(feasible: np.ndarray, path: list[tuple[int, int]]) -> bool:
    for a, b in zip(path, path[1:]):
        if _reachable_with_banned_edge(feasible, path[0], path[-1], frozenset((a, b))):
            return True
    return False


def _graph_metrics(feasible: np.ndarray, start: tuple[int, int]) -> dict[str, object]:
    dist, _parent = _bfs_parents(feasible, start)
    reachable = [(r, c) for r in range(feasible.shape[0]) for c in range(feasible.shape[1]) if dist[r, c] >= 0]
    degrees: dict[tuple[int, int], int] = {}
    for r, c in reachable:
        deg = 0
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and feasible[rr, cc] and dist[rr, cc] >= 0:
                deg += 1
        degrees[(r, c)] = deg
    leaves = [cell for cell, deg in degrees.items() if deg == 1]
    branches = [cell for cell, deg in degrees.items() if deg >= 3]
    return {"dist": dist, "degrees": degrees, "leaves": leaves, "branches": branches}


def _combo_metrics(
    feasible: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> dict[str, object]:
    num_shortest, dist, parent = _shortest_path_count(feasible, start, goal)
    if dist[goal] < 0:
        return {
            "reachable": False,
            "num_shortest": int(num_shortest),
            "shortest_len": -1,
            "has_alt_route": False,
            "branch_on_shortest": False,
            "off_path_leaf_count": 0,
        }
    path = _reconstruct_path(parent, goal)
    path_set = set(path)
    graph = _graph_metrics(feasible, start)
    degrees = graph["degrees"]
    leaves = graph["leaves"]
    branch_on_shortest = any(cell in path_set and int(degrees[cell]) >= 3 for cell in path_set)
    off_path_leaf_count = sum(1 for cell in leaves if cell not in path_set)
    return {
        "reachable": True,
        "num_shortest": int(num_shortest),
        "shortest_len": int(dist[goal]),
        "path": path,
        "has_alt_route": bool(_has_alternative_route(feasible, path)),
        "branch_on_shortest": bool(branch_on_shortest),
        "off_path_leaf_count": int(off_path_leaf_count),
        "branch_count": int(len(graph["branches"])),
        "leaf_count": int(len(leaves)),
    }


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    if random.random() < 0.5:
        g = np.flipud(g)
    if random.random() < 0.5:
        g = np.fliplr(g)
    if random.random() < 0.5:
        g = g.T
    return np.array(g, dtype=int)


def _build_canonical_placements(h2: int, w2: int) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]] | None:
    if h2 < 9 or w2 < 12:
        return None

    start_c = 1
    dest_c = w2 - 1
    if dest_c - start_c < 8:
        return None

    r0 = random.randint(1, h2 - 7)
    r1 = random.randint(r0 + 2, h2 - 4)
    r2 = random.randint(r1 + 2, h2 - 2)

    c1 = random.randint(3, max(3, w2 // 3))
    c2_lo = c1 + 3
    c2_hi = dest_c - 4
    if c2_lo > c2_hi:
        return None
    c2 = random.randint(c2_lo, c2_hi)

    start = (r0, start_c)
    dest = (r2, dest_c)
    backbone = _polyline([start, (r0, c1), (r1, c1), (r1, c2), (r2, c2), dest])
    if len(backbone) != len(set(backbone)):
        return None

    a_lo = c2 + 1
    a_hi = dest_c - 3
    if a_lo > a_hi:
        return None
    a_c = random.randint(a_lo, a_hi)
    b_lo = a_c + 2
    b_hi = dest_c - 1
    if b_lo > b_hi:
        return None
    b_c = random.randint(b_lo, b_hi)

    loop_rows = [r for r in range(1, h2 - 1) if r not in (r0, r1, r2) and abs(r - r2) >= 2]
    if not loop_rows:
        return None
    loop_row = random.choice(loop_rows)
    loop = _polyline([(r2, a_c), (loop_row, a_c), (loop_row, b_c), (r2, b_c)])

    branch_candidates = [c for c in range(c1 + 1, c2) if c not in (a_c, b_c)]
    if not branch_candidates:
        return None
    branch_c = random.choice(branch_candidates)
    branch_rows = [r for r in range(1, h2 - 1) if r not in (r0, r1, r2, loop_row) and abs(r - r1) >= 2]
    if not branch_rows:
        return None
    branch_row = random.choice(branch_rows)
    dead_end = _polyline([(r1, branch_c), (branch_row, branch_c)])
    if len(dead_end) < 3:
        return None

    placements = set(backbone) | set(loop) | set(dead_end)
    return start, dest, placements


def _gen_attempt() -> np.ndarray | None:
    size_ranges = {
        1: (12, 18),
        2: (14, 19),
        3: (17, 20),
        4: (20, 20),
    }
    n = random.choice([1, 2, 3, 4])
    G = random.randint(*size_ranges[n])
    h2 = w2 = G - n + 1

    built = _build_canonical_placements(h2, w2)
    if built is None:
        return None
    start, dest, placements = built

    grid = _render_from_placements(placements, G, n)
    sr, sc = start
    dr, dc = dest
    grid[sr : sr + n, sc : sc + n] = _MOVER
    grid[dr : dr + n, dc : dc + n] = _DEST
    grid = _apply_random_symmetry(grid)

    bg = int(Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0])
    if bg != _BG:
        return None

    mover, dest_sq = _detect_squares(grid, bg)
    if int(mover["col"]) != _MOVER or int(dest_sq["col"]) != _DEST:
        return None
    if int(mover["n"]) != n or int(dest_sq["n"]) != n:
        return None

    river = _detect_river(grid, bg, _MOVER, _DEST)
    trav = (grid == river) | (grid == _MOVER) | (grid == _DEST)
    feasible = _compute_feasible(trav, n)

    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest_sq["r0"]), int(dest_sq["c0"])
    if not feasible[sr, sc] or not feasible[dr, dc]:
        return None

    metrics = _combo_metrics(feasible, (sr, sc), (dr, dc))
    if not bool(metrics["reachable"]):
        return None
    if int(metrics["num_shortest"]) != 1:
        return None
    if not bool(metrics["has_alt_route"]):
        return None
    if not bool(metrics["branch_on_shortest"]):
        return None
    if int(metrics["off_path_leaf_count"]) < 1:
        return None
    if int(metrics["shortest_len"]) < 8:
        return None

    river_ratio = float(np.count_nonzero(grid == _RIVER)) / float(G * G)
    if not (0.16 <= river_ratio <= 0.60):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(grid == _RIVER)) <= int(np.count_nonzero(out == _RIVER)):
        return None
    if not bool((out == _MOVER).sum() == n * n):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo variant after 400 attempts")
