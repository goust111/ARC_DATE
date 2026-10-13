from __future__ import annotations

import heapq
import random
from collections import Counter, deque

import numpy as np


BG = 8
RED = 2
GREEN = 3
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    h, _w = grid.shape
    bars: list[dict[str, int]] = []
    for comp in _components(grid == RED):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert c0 == c1, "each obstacle must be a 1-cell-wide vertical bar"
        assert r1 == h - 1, "bars must be bottom-aligned"
        assert int(comp.shape[0]) == r1 - r0 + 1, "bar must be contiguous"
        bars.append({"row0": r0, "row1": r1, "col": c0, "height": r1 - r0 + 1})
    bars.sort(key=lambda bar: (bar["col"], bar["row0"]))
    assert bars, "expected at least one red bar"
    return bars


def _parse(grid: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must stay within 30x30"

    bg = _get_bg(g)
    assert bg == BG, "background must be azure (8)"

    starts = np.argwhere(g == GREEN)
    assert starts.shape[0] == 1, "expected exactly one green start pixel"
    sr, sc = map(int, starts[0])
    assert sc == 0, "start must lie on the left edge"

    bars = _extract_bars(g)
    return g, bg, (sr, sc), bars


def _blocked_mask(shape: tuple[int, int], bars: list[dict[str, int]]) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    for bar in bars:
        radius = int(bar["height"])
        rr0 = max(0, int(bar["row0"]) - radius)
        rr1 = min(h - 1, int(bar["row1"]) + radius)
        cc0 = max(0, int(bar["col"]) - radius)
        cc1 = min(w - 1, int(bar["col"]) + radius)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    for bar in bars:
        blocked[int(bar["row0"]) : int(bar["row1"]) + 1, int(bar["col"])] = True
    return blocked


def _canonical_solve(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], tuple[int, int], int, dict[tuple[int, int, int], tuple[int, int]]]:
    g, _bg, (sr, sc), bars = _parse(grid)
    h, w = g.shape
    blocked = _blocked_mask(g.shape, bars)
    blocked[sr, sc] = False
    assert w >= 3 and not blocked[sr, sc + 1], "the first step to the right must be legal"

    start = (sr, sc, 1)  # initial heading is right
    inf = (10**9, 10**9)
    best: dict[tuple[int, int, int], tuple[int, int]] = {start: (0, 0)}
    count: dict[tuple[int, int, int], int] = {start: 1}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[int, int, int, int, int]] = [(0, 0, sr, sc, 1)]

    while heap:
        steps, turns, r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best.get((r, c, dprev), inf):
            continue
        for nd, (dr, dc) in enumerate(DIRS):
            if steps == 0 and nd != 1:
                continue
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            old_cost = best.get(key)
            if old_cost is None or ncost < old_cost:
                best[key] = ncost
                count[key] = count[(r, c, dprev)]
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost[0], ncost[1], rr, cc, nd))
            elif ncost == old_cost:
                count[key] = min(2, count[key] + count[(r, c, dprev)])
                if (r, c, dprev) < prev.get(key, (10**9, 10**9, 10**9)):
                    prev[key] = (r, c, dprev)

    goal_states = [state for state in best if state[1] == w - 1]
    assert goal_states, "path must reach the right edge"

    best_cost = min(best[state] for state in goal_states)
    best_row = min(state[0] for state in goal_states if best[state] == best_cost)
    chosen_states = [state for state in goal_states if best[state] == best_cost and state[0] == best_row]
    chosen_state = min(chosen_states)
    chosen_count = min(2, sum(count[state] for state in chosen_states))

    path: list[tuple[int, int]] = []
    cur = chosen_state
    while True:
        r, c, _d = cur
        path.append((int(r), int(c)))
        if cur == start:
            break
        cur = prev[cur]
    path.reverse()
    return path, best_cost, chosen_count, best


def _shortest_step_turn_span(grid: np.ndarray) -> tuple[int, int, int]:
    g, _bg, (sr, sc), bars = _parse(grid)
    h, w = g.shape
    blocked = _blocked_mask(g.shape, bars)
    blocked[sr, sc] = False

    start = (sr, sc, 1)
    step_dist: dict[tuple[int, int, int], int] = {start: 0}
    order = [start]
    q = deque([start])

    while q:
        r, c, dprev = q.popleft()
        step = step_dist[(r, c, dprev)]
        for nd, (dr, dc) in enumerate(DIRS):
            if step == 0 and nd != 1:
                continue
            rr, cc = r + dr, c + dc
            key = (rr, cc, nd)
            if not (0 <= rr < h and 0 <= cc < w) or blocked[rr, cc] or key in step_dist:
                continue
            step_dist[key] = step + 1
            order.append(key)
            q.append(key)

    goals = [state for state, dist in step_dist.items() if state[1] == w - 1]
    assert goals, "shortest-step analysis requires a reachable goal"
    shortest_steps = min(step_dist[state] for state in goals)

    min_turns = {start: 0}
    max_turns = {start: 0}
    by_step: dict[int, list[tuple[int, int, int]]] = {}
    for state, step in step_dist.items():
        by_step.setdefault(step, []).append(state)
        if state != start:
            min_turns[state] = 10**9
            max_turns[state] = -10**9

    for step in range(shortest_steps):
        for r, c, dprev in by_step.get(step, []):
            for nd, (dr, dc) in enumerate(DIRS):
                if step == 0 and nd != 1:
                    continue
                rr, cc = r + dr, c + dc
                nxt = (rr, cc, nd)
                if step_dist.get(nxt) != step + 1:
                    continue
                extra_turn = 0 if nd == dprev else 1
                min_turns[nxt] = min(min_turns[nxt], min_turns[(r, c, dprev)] + extra_turn)
                max_turns[nxt] = max(max_turns[nxt], max_turns[(r, c, dprev)] + extra_turn)

    goal_min_turns = min(min_turns[state] for state in goals if step_dist[state] == shortest_steps)
    goal_max_turns = max(max_turns[state] for state in goals if step_dist[state] == shortest_steps)
    return shortest_steps, goal_min_turns, goal_max_turns


def _paint_solution(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in path:
        if out[r, c] != RED:
            out[r, c] = GREEN
    return out


def _path_uses_only_safe_cells(grid: np.ndarray, path: list[tuple[int, int]]) -> bool:
    _g, _bg, (sr, sc), bars = _parse(grid)
    blocked = _blocked_mask(grid.shape, bars)
    blocked[sr, sc] = False
    return all(not blocked[r, c] for r, c in path)


def _remove_bar(grid: np.ndarray, bar_index: int) -> np.ndarray:
    g, bg, _start, bars = _parse(grid)
    bar = bars[bar_index]
    out = g.copy()
    out[int(bar["row0"]) : int(bar["row1"]) + 1, int(bar["col"])] = bg
    return out


def _lower_tallest_bar(grid: np.ndarray) -> np.ndarray:
    g, bg, _start, bars = _parse(grid)
    tallest_idx = max(range(len(bars)), key=lambda i: (bars[i]["height"], bars[i]["col"]))
    tallest = bars[tallest_idx]
    assert tallest["height"] >= 3
    out = g.copy()
    out[int(tallest["row0"]), int(tallest["col"])] = bg
    return out


def _validate_instance(grid: np.ndarray) -> None:
    g, _bg, (sr, _sc), bars = _parse(grid)
    assert 2 <= len(bars) <= 4, "generated grids must contain 2-4 bars"
    heights = [bar["height"] for bar in bars]
    assert all(2 <= height <= 5 for height in heights), "bar heights must stay in [2,5]"
    assert len(set(heights)) >= 2, "need at least two distinct danger radii"
    assert list(heights) == sorted(heights), "generator keeps heights increasing left-to-right"

    path, best_cost, chosen_count, _best = _canonical_solve(g)
    assert chosen_count == 1, "the official tie-breaks must pick a unique path"
    assert best_cost[0] > g.shape[1] - 1, "path must do more than go straight across"
    assert best_cost[1] == 2, "generated samples should use a single chosen bend-up-then-right route"
    assert _path_uses_only_safe_cells(g, path), "solution path must respect every keep-out zone"
    assert path[0] == (sr, 0) and path[-1][1] == g.shape[1] - 1

    shortest_steps, min_turns, max_turns = _shortest_step_turn_span(g)
    assert shortest_steps == best_cost[0]
    assert min_turns == best_cost[1]
    assert max_turns > min_turns, "there must be a same-length but higher-turn detour candidate"

    out = main(g)
    assert np.array_equal(out, _paint_solution(g, path))

    without_first = _remove_bar(g, 0)
    path_without_first, _cost_without_first, _count_without_first, _ = _canonical_solve(without_first)
    assert path_without_first != path, "the first bar must matter to the chosen route"

    lower_tallest = _lower_tallest_bar(g)
    path_lower_tallest, _cost_lower_tallest, _count_lower_tallest, _ = _canonical_solve(lower_tallest)
    assert path_lower_tallest != path, "the tallest bar must determine the chosen clearance row"


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the shortest green route to the right edge while staying outside bar-height keep-out zones.

    Input structure:
    - The background is azure (8).
    - There are 2-4 red one-cell-wide vertical bars, each bottom-aligned and contiguous.
    - A single green pixel sits on the left edge and marks the path start.

    Rule:
    - Each red bar's keep-out radius equals that bar's height in pixels.
    - Draw a 4-connected green path from the start to the right edge so that every path
      cell has Chebyshev distance strictly greater than the corresponding bar height from
      every pixel of every bar.
    - Among all valid routes, choose the one with the fewest steps; break equal-length ties
      by fewer turns, then by the smaller goal row on the right edge. Keep the red bars unchanged.

    Key insight:
    - The bar height is not decorative: it encodes the forbidden radius, so solving the task
      requires reading each bar's height first and then planning the route around the combined zones.
    """
    path, _cost, _count, _best = _canonical_solve(grid)
    return _paint_solution(grid, path)


def _build_candidate() -> np.ndarray | None:
    count = random.randint(2, 4)
    heights = sorted(random.sample([2, 3, 4, 5], count))

    h_min = 2 * heights[-1] + 4
    if h_min > 20:
        return None
    h = random.randint(h_min, 20)

    start_cols = [2]
    for _ in range(1, count):
        start_cols.append(start_cols[-1] + random.randint(2, 4))
    bar_cols = [start + height for start, height in zip(start_cols, heights)]

    w = max(start + 2 * height for start, height in zip(start_cols, heights)) + random.randint(3, 6)
    if w > 30:
        return None

    first_safe_row = h - 2 * heights[0] - 1
    if first_safe_row + 1 > h - 2:
        return None
    start_row = min(h - 2, first_safe_row + random.randint(1, min(4, h - 1 - first_safe_row)))

    grid = np.full((h, w), BG, dtype=int)
    for col, height in zip(bar_cols, heights):
        grid[h - height :, col] = RED
    grid[start_row, 0] = GREEN
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            _validate_instance(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
