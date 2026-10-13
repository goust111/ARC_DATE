from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

PALETTE = (GREEN, YELLOW, BLUE, MAGENTA)
GEN_ATTEMPTS = 500
CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in CARDINAL_DIRS:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _grow_paths(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    starts = [(h - 1, int(c)) for c in np.flatnonzero(out[h - 1] == RED)]
    for start_r, start_c in sorted(starts, key=lambda rc: rc[1]):
        r, c = start_r, start_c
        while r > 0:
            above = int(out[r - 1, c])
            if above == BLACK:
                r -= 1
                out[r, c] = RED
                continue
            if above == GREY:
                nc = c + 1
                if nc < w and int(out[r, nc]) == BLACK:
                    c = nc
                    out[r, c] = RED
                    continue
            break
    return out


def _distance_map(grown: np.ndarray) -> np.ndarray:
    h, w = grown.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(grown == RED):
        rr, cc = int(r), int(c)
        dist[rr, cc] = 0
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grown.shape, nr, nc):
                continue
            if dist[nr, nc] != -1 or int(grown[nr, nc]) == GREY:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    return dist


def _paint_rings(grown: np.ndarray) -> np.ndarray:
    out = grown.copy()
    dist = _distance_map(grown)
    for r, c in np.argwhere(grown == BLACK):
        rr, cc = int(r), int(c)
        d = int(dist[rr, cc])
        assert d > 0, "all background cells should be reachable from the path sources"
        out[rr, cc] = PALETTE[(d - 1) % len(PALETTE)]
    return out


def _red_signature(grid: np.ndarray) -> frozenset[tuple[int, int]]:
    grown = _grow_paths(grid)
    return frozenset((int(r), int(c)) for r, c in np.argwhere(grown == RED))


def _grey_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    grey = {(int(r), int(c)) for r, c in np.argwhere(grid == GREY)}
    comps: list[list[tuple[int, int]]] = []
    while grey:
        seed = grey.pop()
        comp = [seed]
        q = deque([seed])
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in grey:
                    grey.remove((nr, nc))
                    q.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(comp)
    return comps


def _obstacle_geometry_ok(grid: np.ndarray) -> bool:
    comps = _grey_components(grid)

    comp_of: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        if len(comp) not in (1, 2):
            return False
        if len(comp) == 2:
            (r0, c0), (r1, c1) = comp
            if abs(r0 - r1) + abs(c0 - c1) != 1:
                return False
        for cell in comp:
            comp_of[cell] = idx

    for (r, c), idx in comp_of.items():
        for nr, nc in _neighbors8(r, c):
            other = comp_of.get((nr, nc))
            if other is not None and other != idx:
                return False
    return True


def _valid_obstacle_layout(grid: np.ndarray) -> bool:
    comps = _grey_components(grid)
    return 4 <= len(comps) <= 8 and _obstacle_geometry_ok(grid)


def _distance_warp_exists(grown: np.ndarray) -> bool:
    red_cells = [(int(r), int(c)) for r, c in np.argwhere(grown == RED)]
    dist = _distance_map(grown)
    for r, c in np.argwhere(grown == BLACK):
        rr, cc = int(r), int(c)
        bfs_dist = int(dist[rr, cc])
        manhattan = min(abs(rr - sr) + abs(cc - sc) for sr, sc in red_cells)
        if bfs_dist > manhattan:
            return True
    return False


def _pick_event_rows(height: int, count: int) -> list[int] | None:
    chosen: list[int] = []
    candidates = list(range(2, height - 1))
    while len(chosen) < count and candidates:
        row = random.choice(candidates)
        chosen.append(row)
        candidates = [cand for cand in candidates if abs(cand - row) >= 2]
    if len(chosen) != count:
        return None
    return sorted(chosen, reverse=True)


def _make_plan(height: int, width: int) -> list[dict[str, object]] | None:
    if width < 10 or height < 8:
        return None

    if random.random() < 0.45:
        terminate = random.random() < 0.4
        max_turns = min(4, width - 1 if terminate else width - 2)
        if max_turns < 1:
            return None
        turns = random.randint(1, max_turns)
        start_c = width - 1 - turns if terminate else random.randint(0, width - 2 - turns)
        rows = _pick_event_rows(height, turns + int(terminate))
        if rows is None:
            return None
        return [{
            "start_c": start_c,
            "turn_rows": rows[:turns],
            "terminate_row": rows[-1] if terminate else None,
        }]

    right_terminate = random.random() < 0.35
    right_turns = random.randint(1, min(3, width - 1))
    if right_terminate:
        right_start = width - 1 - right_turns
    else:
        hi = width - 2 - right_turns
        lo = max(width // 2, 5)
        if lo > hi:
            return None
        right_start = random.randint(lo, hi)

    left_turns = random.randint(0, 2)
    left_hi = min(width // 2 - 1, right_start - left_turns - 4)
    if left_hi < 0:
        return None
    left_start = random.randint(0, left_hi)

    left_rows = _pick_event_rows(height, left_turns)
    right_rows = _pick_event_rows(height, right_turns + int(right_terminate))
    if left_rows is None or right_rows is None:
        return None

    return [
        {
            "start_c": left_start,
            "turn_rows": left_rows,
            "terminate_row": None,
        },
        {
            "start_c": right_start,
            "turn_rows": right_rows[:right_turns],
            "terminate_row": right_rows[-1] if right_terminate else None,
        },
    ]


def _render_plans(height: int, width: int, plans: list[dict[str, object]]) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    for plan in plans:
        start_c = int(plan["start_c"])
        grid[height - 1, start_c] = RED

    for plan in plans:
        current_c = int(plan["start_c"])
        turn_rows = list(plan["turn_rows"])
        turn_set = set(turn_rows)
        terminate_row = plan["terminate_row"]
        events = sorted(turn_rows + ([int(terminate_row)] if terminate_row is not None else []), reverse=True)
        for row in events:
            block_r = row - 1
            if block_r < 0 or grid[block_r, current_c] != BLACK:
                return None
            grid[block_r, current_c] = GREY
            if row in turn_set:
                current_c += 1
                if current_c >= width:
                    return None
    return grid


def _candidate_block(height: int, width: int) -> list[tuple[int, int]]:
    size = 1 if random.random() < 0.55 else 2
    if size == 1:
        return [(random.randrange(0, height - 1), random.randrange(width))]

    if random.random() < 0.5:
        r = random.randrange(0, height - 1)
        c = random.randrange(width - 1)
        return [(r, c), (r, c + 1)]

    r = random.randrange(0, height - 2)
    c = random.randrange(width)
    return [(r, c), (r + 1, c)]


def _can_place_block(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if grid[r, c] != BLACK:
            return False
        for nr, nc in _neighbors8(r, c):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) in cell_set:
                continue
            if int(grid[nr, nc]) == GREY:
                return False
    return True


def _add_extra_blocks(grid: np.ndarray, target_blocks: int, baseline_red: frozenset[tuple[int, int]]) -> np.ndarray | None:
    out = grid.copy()
    blocks = len(_grey_components(out))
    for _ in range(400):
        if blocks >= target_blocks:
            return out
        cells = _candidate_block(*out.shape)
        if not _can_place_block(out, cells):
            continue
        trial = out.copy()
        for r, c in cells:
            trial[r, c] = GREY
        if _red_signature(trial) != baseline_red:
            continue
        if not _obstacle_geometry_ok(trial):
            continue
        out = trial
        blocks = len(_grey_components(out))
    return None


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(13, 18)
    width = random.randint(12, 18)
    plans = _make_plan(height, width)
    if plans is None:
        return None

    base = _render_plans(height, width, plans)
    if base is None:
        return None

    if not _obstacle_geometry_ok(base):
        return None

    mandatory_blocks = len(_grey_components(base))
    if mandatory_blocks > 8:
        return None

    baseline_red = _red_signature(base)
    target_blocks = random.randint(max(4, mandatory_blocks), 8)
    grid = _add_extra_blocks(base, target_blocks, baseline_red)
    if grid is None or not _valid_obstacle_layout(grid):
        return None

    grown = _grow_paths(grid)
    dist = _distance_map(grown)
    max_dist = int(dist.max())
    bottom_red = int(np.count_nonzero(grid[-1] == RED))
    grown_red = int(np.count_nonzero(grown == RED))
    if bottom_red not in (1, 2) or grown_red <= bottom_red + 4:
        return None
    if max_dist < 5 or not _distance_warp_exists(grown):
        return None
    if int(np.count_nonzero(grown[0] == RED)) == 0 and all(plan["terminate_row"] is None for plan in plans):
        return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow bottom red controls into bent paths, then color obstacle-aware concentric rings around the full path set.

    Input structure
    - The grid contains black background, one or two red control pixels on the bottom row, and several grey obstacle blocks elsewhere.
    - Each obstacle block is a 1x1 or 1x2 connected body, and obstacle blocks do not touch each other even diagonally.
    - All non-obstacle, non-control cells start as black background.

    Rule
    - Starting from each bottom-row red control, grow a red path upward on the working grid.
    - If the cell directly above the current path tip is black, extend upward into it; if it is grey, step one cell right on the current row and continue only if that right cell is black, otherwise stop that path.
    - After all growth ends, treat every red path cell as a multi-source seed set and run 4-neighbor BFS through all non-grey cells.
    - Recolor each black cell by its obstacle-aware distance to the red seed set using the cycle 1->green, 2->yellow, 3->blue, 4->magenta, then repeat every four steps; keep red paths and grey obstacles unchanged.

    Key insight
    - The final rings are not centered on a point: they inherit the entire bent red path as their source set, while grey obstacles both bend the path and force the distance field to detour around them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, RED, GREY}, "input should contain only black, red, and grey"
    red_bottom = np.flatnonzero(g[-1] == RED)
    assert len(red_bottom) in (1, 2), "expected one or two red control pixels on the bottom row"
    assert int(np.count_nonzero(g == RED)) == len(red_bottom), "input red cells should appear only on the bottom row"

    grown = _grow_paths(g)
    return _paint_rings(grown)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
