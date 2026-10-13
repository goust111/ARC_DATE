from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
TEAL = 8
OBSTACLE_COLORS = (1, 3, 4, 5, 6, 7, 9)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
MAX_GENERATION_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """Trace a diagonal bouncing ball from the marked missing corner and repaint the first hit rectangle red.

    Input structure:
    - The background is black.
    - Exactly one 2x2 launcher block is present: three teal fill cells and one red marker cell at a corner.
    - Several separate solid colored rectangular obstacles appear elsewhere and do not touch the launcher.

    Rule:
    - Use the diagonal from the fill cell opposite the marker toward the marker as the launch direction, then continue outward in that same diagonal direction.
    - Start from the first cell outside the marker, move one diagonal cell at a time, and paint every visited black cell red.
    - Whenever the next step would leave the grid, reflect the row component and/or column component that would cross the boundary, then continue.
    - When the next step would enter an obstacle, stop immediately and repaint that entire obstacle red. Leave the launcher and every unhit obstacle unchanged.

    Key insight:
    - The marker itself is not an arrow; the missing teal corner of the 2x2 block determines the diagonal, and the target is found only by following the full bounce path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    launcher = _find_launcher(g)
    obstacles = _find_obstacles(g, launcher["cells"])
    assert obstacles, "expected at least one obstacle"

    trace = _trace_ball(g.shape, launcher, obstacles)

    out = g.copy()
    for r, c in trace["painted"]:
        assert g[r, c] == BLACK, "trajectory may only pass through black cells before the hit"
        out[r, c] = RED
    for r, c in obstacles[trace["hit_obstacle"]]["cells"]:
        out[r, c] = RED
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_launcher(grid: np.ndarray) -> dict:
    H, W = grid.shape
    candidates = []
    for r0 in range(H - 1):
        for c0 in range(W - 1):
            block = grid[r0 : r0 + 2, c0 : c0 + 2]
            if int(np.count_nonzero(block == RED)) != 1 or int(np.count_nonzero(block == TEAL)) != 3:
                continue
            local_marker = np.argwhere(block == RED)
            mr, mc = map(int, local_marker[0])
            dr = -1 if mr == 0 else 1
            dc = -1 if mc == 0 else 1
            marker = (r0 + mr, c0 + mc)
            cells = {(r0 + rr, c0 + cc) for rr in range(2) for cc in range(2)}
            opposite = (marker[0] - dr, marker[1] - dc)
            if opposite not in cells:
                continue
            expected_teal = cells - {marker}
            if not all(grid[r, c] == TEAL for r, c in expected_teal):
                continue
            start = (marker[0] + dr, marker[1] + dc)
            if not _in_bounds(grid.shape, start[0], start[1]):
                continue
            candidates.append(
                {
                    "bbox": (r0, c0, r0 + 1, c0 + 1),
                    "marker": marker,
                    "direction": (dr, dc),
                    "cells": cells,
                    "start": start,
                }
            )
    assert len(candidates) == 1, "expected exactly one valid launcher block"
    return candidates[0]


def _component_cells(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = mask.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return seen


def _find_obstacles(grid: np.ndarray, launcher_cells: set[tuple[int, int]]) -> list[dict]:
    H, W = grid.shape
    seen: set[tuple[int, int]] = set()
    obstacles = []
    obstacle_map: dict[tuple[int, int], int] = {}
    for r in range(H):
        for c in range(W):
            if (r, c) in launcher_cells or grid[r, c] == BLACK or (r, c) in seen:
                continue
            color = int(grid[r, c])
            assert color not in (RED, TEAL), "obstacle colors must not reuse launcher colors"
            mask = grid == color
            cells = _component_cells(mask, (r, c))
            seen |= cells
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            expected = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
            assert cells == expected, "each obstacle must be a solid rectangle"
            index = len(obstacles)
            for cell in cells:
                assert cell not in obstacle_map
                obstacle_map[cell] = index
            obstacles.append({"color": color, "cells": cells, "bbox": (r0, c0, r1, c1)})
    for obs in obstacles:
        for r, c in obs["cells"]:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) in launcher_cells:
                        raise AssertionError("obstacles must not touch the launcher")
    for obs in obstacles:
        obs["map"] = obstacle_map
    return obstacles


def _trace_ball(
    shape: tuple[int, int], launcher: dict, obstacles: list[dict]
) -> dict:
    H, W = shape
    obstacle_map = obstacles[0]["map"] if obstacles else {}
    marker_r, marker_c = launcher["marker"]
    dr, dc = launcher["direction"]
    r, c = launcher["start"]
    assert _in_bounds(shape, r, c), "start cell must be inside the grid"
    assert (r, c) not in launcher["cells"]
    assert obstacle_map.get((r, c)) is None

    painted: list[tuple[int, int]] = []
    states = set()
    row_bounces = 0
    col_bounces = 0
    corner_bounces = 0

    while True:
        state = (r, c, dr, dc)
        assert state not in states, "ball repeated a state before hitting an obstacle"
        states.add(state)
        painted.append((r, c))

        flip_r = not (0 <= r + dr < H)
        flip_c = not (0 <= c + dc < W)
        if flip_r:
            dr = -dr
            row_bounces += 1
        if flip_c:
            dc = -dc
            col_bounces += 1
        if flip_r and flip_c:
            corner_bounces += 1

        nr, nc = r + dr, c + dc
        hit = obstacle_map.get((nr, nc))
        if hit is not None:
            return {
                "painted": painted,
                "path_len": len(painted),
                "hit_cell": (nr, nc),
                "hit_obstacle": hit,
                "row_bounces": row_bounces,
                "col_bounces": col_bounces,
                "corner_bounces": corner_bounces,
                "start": (launcher["start"][0], launcher["start"][1]),
                "marker": (marker_r, marker_c),
            }
        assert (nr, nc) not in launcher["cells"], "trajectory may not re-enter the launcher on valid inputs"
        r, c = nr, nc


def _empty_path_candidates(shape: tuple[int, int], launcher: dict) -> list[dict]:
    H, W = shape
    r, c = launcher["start"]
    dr, dc = launcher["direction"]
    path = [launcher["start"]]
    states = set()
    row_bounces = 0
    col_bounces = 0
    corner_bounces = 0
    candidates = []

    while True:
        state = (r, c, dr, dc)
        if state in states:
            break
        states.add(state)

        flip_r = not (0 <= r + dr < H)
        flip_c = not (0 <= c + dc < W)
        if flip_r:
            dr = -dr
            row_bounces += 1
        if flip_c:
            dc = -dc
            col_bounces += 1
        if flip_r and flip_c:
            corner_bounces += 1

        nr, nc = r + dr, c + dc
        if (nr, nc) in launcher["cells"]:
            break
        candidates.append(
            {
                "hit_cell": (nr, nc),
                "painted": tuple(path),
                "path_len": len(path),
                "row_bounces": row_bounces,
                "col_bounces": col_bounces,
                "corner_bounces": corner_bounces,
                "last_flip_r": flip_r,
                "last_flip_c": flip_c,
            }
        )
        path.append((nr, nc))
        r, c = nr, nc
    return candidates


def _scenario_match(kind: str, cand: dict) -> bool:
    if cand["path_len"] < 2:
        return False
    row = cand["row_bounces"]
    col = cand["col_bounces"]
    corner = cand["corner_bounces"]
    if kind == "direct":
        return row == 0 and col == 0
    if kind == "row":
        return row > 0 and col == 0
    if kind == "col":
        return row == 0 and col > 0
    if kind == "both":
        return row > 0 and col > 0 and corner == 0
    if kind == "corner":
        return corner > 0
    raise ValueError(kind)


def _choose_launcher(rng: random.Random, H: int, W: int) -> dict | None:
    dr, dc = rng.choice(DIAGONALS)
    mr = 0 if dr < 0 else 1
    mc = 0 if dc < 0 else 1
    r_lo = 1 if dr < 0 else 0
    r_hi = H - 2 if dr < 0 else H - 3
    c_lo = 1 if dc < 0 else 0
    c_hi = W - 2 if dc < 0 else W - 3
    if r_lo > r_hi or c_lo > c_hi:
        return None
    r0 = rng.randint(r_lo, r_hi)
    c0 = rng.randint(c_lo, c_hi)
    marker = (r0 + mr, c0 + mc)
    cells = {(r0 + rr, c0 + cc) for rr in range(2) for cc in range(2)}
    return {
        "bbox": (r0, c0, r0 + 1, c0 + 1),
        "marker": marker,
        "direction": (dr, dc),
        "cells": cells,
        "start": (marker[0] + dr, marker[1] + dc),
    }


def _dist_to_marker(rect: tuple[int, int, int, int], marker: tuple[int, int]) -> int:
    r0, c0, r1, c1 = rect
    mr, mc = marker
    best = 10**9
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            best = min(best, abs(r - mr) + abs(c - mc))
    return best


def _touches_any(
    rect: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    require_gap: int = 1,
) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        if not (r1 + require_gap < or0 or or1 + require_gap < r0 or c1 + require_gap < oc0 or oc1 + require_gap < c0):
            return True
    return False


def _target_rect_options(
    shape: tuple[int, int],
    launcher: dict,
    candidate: dict,
    placed_rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    H, W = shape
    hit_r, hit_c = candidate["hit_cell"]
    painted = set(candidate["painted"])
    options = []
    sizes = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
    for h, w in sizes:
        for top_offset in range(h):
            for left_offset in range(w):
                r0 = hit_r - top_offset
                c0 = hit_c - left_offset
                r1 = r0 + h - 1
                c1 = c0 + w - 1
                rect = (r0, c0, r1, c1)
                if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
                    continue
                if _touches_any(rect, placed_rects, require_gap=1):
                    continue
                cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                if cells & launcher["cells"]:
                    continue
                if cells & painted:
                    continue
                options.append(rect)
    return options


def _random_rect_options(
    shape: tuple[int, int],
    launcher: dict,
    forbidden: set[tuple[int, int]],
    placed_rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    H, W = shape
    sizes = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (1, 3), (3, 1), (3, 3)]
    options = []
    for h, w in sizes:
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if _touches_any(rect, placed_rects, require_gap=1):
                    continue
                cells = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
                if cells & launcher["cells"]:
                    continue
                if cells & forbidden:
                    continue
                options.append(rect)
    return options


def _render_grid(
    shape: tuple[int, int],
    launcher: dict,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    mr, mc = launcher["marker"]
    dr, dc = launcher["direction"]
    r0, c0, _, _ = launcher["bbox"]
    grid[r0 : r0 + 2, c0 : c0 + 2] = TEAL
    grid[mr, mc] = RED
    for color, (rr0, cc0, rr1, cc1) in zip(colors, rects):
        grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)
    return grid


def _oracle_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    launcher = _find_launcher(grid)
    obstacles = _find_obstacles(grid, launcher["cells"])
    trace = _trace_ball(grid.shape, launcher, obstacles)
    out = grid.copy()
    for r, c in trace["painted"]:
        out[r, c] = RED
    for r, c in obstacles[trace["hit_obstacle"]]["cells"]:
        out[r, c] = RED
    return out, trace


def _semantic_checks(grid: np.ndarray, scenario: str) -> bool:
    try:
        out, trace = _oracle_output(grid)
    except AssertionError:
        return False

    launcher = _find_launcher(grid)
    obstacles = _find_obstacles(grid, launcher["cells"])
    if len(obstacles) < 2:
        return False
    if not _scenario_match(scenario, trace):
        return False
    if trace["path_len"] < 2:
        return False

    hit_idx = trace["hit_obstacle"]
    hit_box = obstacles[hit_idx]["bbox"]
    hit_dist = _dist_to_marker(hit_box, launcher["marker"])
    distractor_dists = [_dist_to_marker(obs["bbox"], launcher["marker"]) for i, obs in enumerate(obstacles) if i != hit_idx]
    if not distractor_dists or min(distractor_dists) >= hit_dist:
        return False

    path_set = set(trace["painted"])
    for i, obs in enumerate(obstacles):
        cells = obs["cells"]
        if path_set & cells:
            return False
        expected_color = RED if i == hit_idx else obs["color"]
        if any(int(out[r, c]) != expected_color for r, c in cells):
            return False
    if any(int(out[r, c]) != RED for r, c in trace["painted"]):
        return False
    if any(int(out[r, c]) != int(grid[r, c]) for r, c in launcher["cells"]):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    scenario = rng.choices(
        ["direct", "row", "col", "both"],
        weights=[2, 3, 3, 4],
        k=1,
    )[0]
    H = rng.randint(12, 22)
    W = rng.randint(12, 22)
    launcher = _choose_launcher(rng, H, W)
    if launcher is None:
        return None

    candidates = [cand for cand in _empty_path_candidates((H, W), launcher) if _scenario_match(scenario, cand)]
    if not candidates:
        return None

    rng.shuffle(candidates)
    placed_rects = [launcher["bbox"]]
    for cand in candidates:
        target_options = _target_rect_options((H, W), launcher, cand, placed_rects)
        if not target_options:
            continue
        target_rect = rng.choice(target_options)
        rects = [target_rect]

        forbidden = set(cand["painted"])
        target_dist = _dist_to_marker(target_rect, launcher["marker"])
        close_options = [
            rect
            for rect in _random_rect_options((H, W), launcher, forbidden, placed_rects + rects)
            if _dist_to_marker(rect, launcher["marker"]) < target_dist
        ]
        if not close_options:
            continue
        rects.append(rng.choice(close_options))

        extra_count = rng.randint(0, 2)
        for _ in range(extra_count):
            options = _random_rect_options((H, W), launcher, forbidden, placed_rects + rects)
            if not options:
                break
            rects.append(rng.choice(options))

        colors = rng.sample(list(OBSTACLE_COLORS), len(rects))
        grid = _render_grid((H, W), launcher, rects, colors)

        try:
            expected, _ = _oracle_output(grid)
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        if not _semantic_checks(grid, scenario):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
